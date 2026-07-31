"""
_scene_analyzer.py

Analizador de secuencia para el modo automático/desatendido.

Objetivo (lo que promete el README y no existía): mirar la secuencia
COMPLETA de imágenes, no cada frame de forma aislada, para decidir si un
objeto dinámico (persona, vehículo, animal, ave) detectado por YOLO debe
enmascararse o no.

Regla explícita del proyecto: en modo automático NUNCA se enmascara una
categoría dinámica solo por pertenecer a esa clase. Solo se enmascara si el
objeto es una ANOMALÍA respecto al resto de la escena, es decir, si:

  (a) aparece de forma transitoria (en menos de la mitad de los frames en
      los que, por su posición, "debería" seguir siendo visible si formara
      parte de la escena), o
  (b) su movimiento entre apariciones es inconsistente con el movimiento de
      cámara estimado en esa misma zona del frame (se mueve "por su
      cuenta": un peatón cruzando, un coche pasando, un pájaro volando).

Si un objeto detectado es persistente Y su movimiento es coherente con el
del resto de la escena (por ejemplo, un coche aparcado, una persona que en
realidad es el sujeto fijo de la reconstrucción), NO se enmascara: se trata
como parte de la escena.

Este análisis es deliberadamente heurístico (no hay tracking multi-objeto
robusto tipo DeepSORT aquí): la intención es dar un resultado razonable y
transparente, no una solución de investigación. El report deja constancia
de qué se enmascaró y por qué.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from ._segmentation_backend import detect_dynamic_boxes

logger = logging.getLogger(__name__)

# Un objeto se considera "transitorio" (posible anomalía) si aparece en
# menos de esta fracción de los frames analizados en su vecindad temporal.
PRESENCE_RATIO_THRESHOLD = 0.5

# Fracción de transiciones con movimiento inconsistente respecto al fondo
# a partir de la cual se considera que el objeto se mueve "por su cuenta".
MOTION_INCONSISTENCY_THRESHOLD = 0.5

# IoU mínimo para emparejar una detección con una pista (track) existente
# entre frames consecutivos.
TRACK_IOU_THRESHOLD = 0.3

# Para acotar el coste computacional en secuencias largas, el flujo óptico
# de fondo se calcula sobre una versión reducida del frame.
FLOW_SCALE = 0.5


def _iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _box_center(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


class _Track:
    __slots__ = ("category", "entries")

    def __init__(self, category: str, frame_idx: int, box: Tuple[int, int, int, int]):
        self.category = category
        self.entries: List[Tuple[int, Tuple[int, int, int, int]]] = [(frame_idx, box)]

    @property
    def last_box(self) -> Tuple[int, int, int, int]:
        return self.entries[-1][1]

    @property
    def last_frame_idx(self) -> int:
        return self.entries[-1][0]


def _build_tracks(per_frame_detections: List[List[Dict[str, Any]]]) -> List[_Track]:
    """
    Tracking simple por IoU entre frames consecutivos, por categoría.
    No pretende ser robusto ante oclusiones largas; es suficiente para
    estimar persistencia y coherencia de movimiento.
    """
    active: List[_Track] = []
    finished: List[_Track] = []

    for frame_idx, detections in enumerate(per_frame_detections):
        matched_track_ids = set()
        matched_det_ids = set()

        for t_idx, track in enumerate(active):
            if track.last_frame_idx != frame_idx - 1:
                continue
            best_det_idx, best_iou = None, 0.0
            for d_idx, det in enumerate(detections):
                if d_idx in matched_det_ids or det["category"] != track.category:
                    continue
                iou = _iou(track.last_box, det["box"])
                if iou > best_iou:
                    best_iou, best_det_idx = iou, d_idx
            if best_det_idx is not None and best_iou >= TRACK_IOU_THRESHOLD:
                track.entries.append((frame_idx, detections[best_det_idx]["box"]))
                matched_track_ids.add(t_idx)
                matched_det_ids.add(best_det_idx)

        still_active = []
        for t_idx, track in enumerate(active):
            if t_idx in matched_track_ids or track.last_frame_idx == frame_idx:
                still_active.append(track)
            else:
                finished.append(track)
        active = still_active

        for d_idx, det in enumerate(detections):
            if d_idx not in matched_det_ids:
                active.append(_Track(det["category"], frame_idx, det["box"]))

    finished.extend(active)
    return finished


def _background_flow_near(
    prev_gray_small: np.ndarray,
    gray_small: np.ndarray,
    box: Tuple[int, int, int, int],
    full_shape: Tuple[int, int],
) -> Tuple[float, float]:
    """
    Estima el flujo óptico "de fondo" (inducido por la cámara) en el
    entorno de una caja, excluyendo la propia caja, sobre imágenes
    reducidas para acotar el coste.
    """
    h_full, w_full = full_shape
    h_s, w_s = gray_small.shape[:2]
    sx, sy = w_s / w_full, h_s / h_full

    x1, y1, x2, y2 = box
    x1s, y1s, x2s, y2s = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)

    margin = max(5, int(0.15 * max(x2s - x1s, y2s - y1s, 1)))
    rx1, ry1 = max(0, x1s - margin), max(0, y1s - margin)
    rx2, ry2 = min(w_s, x2s + margin), min(h_s, y2s + margin)
    if rx2 <= rx1 or ry2 <= ry1:
        return 0.0, 0.0

    try:
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray_small, gray_small, None, 0.5, 2, 15, 3, 5, 1.2, 0
        )
    except Exception:
        return 0.0, 0.0

    region = flow[ry1:ry2, rx1:rx2].copy()
    # Excluye la propia caja del cálculo de fondo, si hay margen suficiente.
    inner_y1, inner_y2 = max(0, y1s - ry1), max(0, y2s - ry1)
    inner_x1, inner_x2 = max(0, x1s - rx1), max(0, x2s - rx1)
    if 0 <= inner_y1 < inner_y2 <= region.shape[0] and 0 <= inner_x1 < inner_x2 <= region.shape[1]:
        region[inner_y1:inner_y2, inner_x1:inner_x2] = np.nan

    valid = region[~np.isnan(region).any(axis=2)] if region.size else np.empty((0, 2))
    if valid.size == 0:
        return 0.0, 0.0
    return float(np.median(valid[:, 0])), float(np.median(valid[:, 1]))


def analyze_dynamic_objects(frame_list: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Analiza la secuencia completa y devuelve, por nombre de frame, la lista
    de detecciones dinámicas que se consideran ANOMALÍAS (y por tanto deben
    enmascararse). Los objetos persistentes y coherentes con el movimiento
    de cámara no aparecen en el resultado (no se enmascaran).

    Devuelve: {frame_name: [{"category":..., "box": (...), "reason": ...}]}
    """
    total = len(frame_list)
    if total == 0:
        return {}

    logger.info("Analizando secuencia (%d frames) en busca de objetos dinámicos anómalos...", total)

    gray_frames: List[np.ndarray] = [None] * total  # type: ignore[list-item]
    per_frame_detections: List[List[Dict[str, Any]]] = []
    full_shape = None

    for idx, frame_path in enumerate(frame_list):
        img = cv2.imread(str(frame_path))
        if img is None:
            per_frame_detections.append([])
            continue
        if full_shape is None:
            full_shape = img.shape[:2]

        detections = detect_dynamic_boxes(img)
        per_frame_detections.append(detections)

        if detections:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_frames[idx] = cv2.resize(gray, None, fx=FLOW_SCALE, fy=FLOW_SCALE)

    if full_shape is None:
        return {}

    if not any(per_frame_detections):
        logger.info("No se detectaron objetos dinámicos en la secuencia.")
        return {}

    tracks = _build_tracks(per_frame_detections)
    logger.info("%d pista(s) de objetos dinámicos construidas.", len(tracks))

    result: Dict[str, List[Dict[str, Any]]] = {}

    for track in tracks:
        span = track.entries[-1][0] - track.entries[0][0] + 1
        presence_ratio = len(track.entries) / span if span > 0 else 1.0

        inconsistent = 0
        comparisons = 0
        for (idx_a, box_a), (idx_b, box_b) in zip(track.entries, track.entries[1:]):
            gap = idx_b - idx_a
            if gap <= 0 or gap > 5:
                # Hueco temporal grande: no hay flujo de fondo fiable entre
                # ambos frames, se ignora esa transición para la coherencia.
                continue
            if gray_frames[idx_a] is None or gray_frames[idx_b] is None:
                continue

            cx_a, cy_a = _box_center(box_a)
            cx_b, cy_b = _box_center(box_b)
            object_motion = np.array([cx_b - cx_a, cy_b - cy_a])

            bg_dx, bg_dy = _background_flow_near(
                gray_frames[idx_a], gray_frames[idx_b], box_a, full_shape
            )
            # El flujo de Farneback está en la escala reducida; se reescala
            # a píxeles de imagen completa para comparar magnitudes.
            background_motion = np.array([bg_dx, bg_dy]) * gap / FLOW_SCALE

            comparisons += 1
            norm_obj = np.linalg.norm(object_motion)
            norm_bg = np.linalg.norm(background_motion)

            if norm_obj < 1.0 and norm_bg < 1.0:
                # Ambos prácticamente estáticos: coherente.
                continue

            if norm_bg < 1.0:
                # El fondo no se mueve pero el objeto sí: se mueve por su
                # cuenta -> inconsistente.
                inconsistent += 1
                continue

            cos_sim = float(np.dot(object_motion, background_motion) / (norm_obj * norm_bg + 1e-6))
            magnitude_ratio = norm_obj / (norm_bg + 1e-6)
            if cos_sim < 0.2 or magnitude_ratio > 3.0 or magnitude_ratio < 0.2:
                inconsistent += 1

        inconsistency_ratio = (inconsistent / comparisons) if comparisons > 0 else 0.0

        is_transient = presence_ratio < PRESENCE_RATIO_THRESHOLD
        is_erratic = inconsistency_ratio > MOTION_INCONSISTENCY_THRESHOLD

        if not (is_transient or is_erratic):
            # Persistente y coherente con el movimiento de cámara: se trata
            # como parte de la escena, no se enmascara.
            continue

        reason = []
        if is_transient:
            reason.append(f"presencia transitoria ({presence_ratio:.0%} de sus frames esperados)")
        if is_erratic:
            reason.append(f"movimiento incoherente con la cámara ({inconsistency_ratio:.0%} de transiciones)")
        reason_text = " y ".join(reason)

        for frame_idx, box in track.entries:
            frame_name = Path(frame_list[frame_idx]).name
            result.setdefault(frame_name, []).append({
                "category": track.category,
                "box": box,
                "reason": reason_text,
            })

    n_anomalous_frames = len(result)
    n_anomalous_boxes = sum(len(v) for v in result.values())
    logger.info(
        "Objetos dinámicos anómalos: %d detección(es) a enmascarar en %d frame(s) de %d.",
        n_anomalous_boxes, n_anomalous_frames, total,
    )
    return result
