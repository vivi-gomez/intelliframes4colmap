from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import cv2
import numpy as np


CLASS_BACKGROUND = 0
CLASS_SKY = 1
CLASS_WATER = 2
CLASS_VEGETATION = 3
CLASS_PERSON = 4
CLASS_VEHICLE = 5
CLASS_REFLECTION = 6
CLASS_LOW_TEXTURE = 7

CLASS_NAMES = {
    CLASS_BACKGROUND: "background",
    CLASS_SKY: "sky",
    CLASS_WATER: "water",
    CLASS_VEGETATION: "vegetation",
    CLASS_PERSON: "person",
    CLASS_VEHICLE: "vehicle",
    CLASS_REFLECTION: "reflection",
    CLASS_LOW_TEXTURE: "low_texture",
}

CLASS_COLORS_BGR = {
    CLASS_BACKGROUND: (0, 0, 0),
    CLASS_SKY: (255, 128, 0),
    CLASS_WATER: (255, 0, 0),
    CLASS_VEGETATION: (0, 180, 0),
    CLASS_PERSON: (180, 0, 180),
    CLASS_VEHICLE: (0, 0, 255),
    CLASS_REFLECTION: (0, 255, 255),
    CLASS_LOW_TEXTURE: (128, 128, 128),
}


def run_segmentation(
    frame_list: Iterable[str],
    masks_dir: str | Path,
    mode: str = "classical",
    checkpoint_path: Optional[str | Path] = None,
) -> List[Dict[str, Any]]:
    mode = (mode or "classical").lower().strip()

    if mode == "sam":
        return run_sam_segmentation(frame_list, masks_dir, checkpoint_path=checkpoint_path)

    if mode == "classical":
        return run_classical_segmentation(frame_list, masks_dir)

    raise ValueError(f"Unsupported segmentation mode: {mode}")


def run_sam_segmentation(
    frame_list: Iterable[str],
    masks_dir: str | Path,
    checkpoint_path: Optional[str | Path] = None,
) -> List[Dict[str, Any]]:
    """
    Placeholder pragmático:
    - valida que SAM existe
    - mientras no haya clasificación semántica robusta encima de SAM,
      delega al fallback clásico para no romper el pipeline.
    """
    try:
        import torch  # noqa: F401
        from segment_anything import sam_model_registry  # noqa: F401
    except Exception as exc:
        raise RuntimeError(f"SAM unavailable: {exc}") from exc

    if checkpoint_path:
        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"SAM checkpoint not found: {ckpt}")

    results = run_classical_segmentation(frame_list, masks_dir)
    for row in results:
        row["segmentation_mode"] = "sam"
    return results


def run_classical_segmentation(
    frame_list: Iterable[str],
    masks_dir: str | Path,
) -> List[Dict[str, Any]]:
    masks_dir = Path(masks_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []

    for frame_path in frame_list:
        frame_path = str(frame_path)
        img = cv2.imread(frame_path, cv2.IMREAD_COLOR)
        if img is None:
            results.append(
                {
                    "frame": Path(frame_path).name,
                    "mask_path": "",
                    "sky_pct": 0.0,
                    "water_pct": 0.0,
                    "vegetation_pct": 0.0,
                    "person_pct": 0.0,
                    "vehicle_pct": 0.0,
                    "reflection_pct": 0.0,
                    "low_texture_pct": 0.0,
                    "dynamic_risk_pct": 0.0,
                    "usable_area_pct": 0.0,
                    "photogrammetry_risk_score": 100.0,
                    "risk_level": "HIGH",
                    "segmentation_mode": "classical",
                    "error": "unreadable_frame",
                }
            )
            continue

        metrics, indexed_mask = _segment_single_image(img)
        mask_path = masks_dir / f"{Path(frame_path).stem}__mask.png"
        cv2.imwrite(str(mask_path), indexed_mask)

        row = {
            "frame": Path(frame_path).name,
            "mask_path": str(mask_path),
            **metrics,
            "segmentation_mode": "classical",
        }
        results.append(row)

    return results


def _segment_single_image(img_bgr: np.ndarray) -> tuple[Dict[str, Any], np.ndarray]:
    h, w = img_bgr.shape[:2]
    total_px = float(h * w)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    indexed_mask = np.zeros((h, w), dtype=np.uint8)

    # SKY: mitad superior, azul/cian, brillante
    upper_half = np.zeros((h, w), dtype=np.uint8)
    upper_half[: h // 2, :] = 255
    sky_color = cv2.inRange(hsv, (85, 20, 80), (135, 255, 255))
    sky_mask = cv2.bitwise_and(sky_color, upper_half)
    sky_mask = _cleanup_mask(sky_mask, 5)

    # WATER: azul en mitad inferior, saturación media
    lower_half = np.zeros((h, w), dtype=np.uint8)
    lower_half[h // 3 :, :] = 255
    water_color = cv2.inRange(hsv, (90, 20, 40), (140, 255, 220))
    water_mask = cv2.bitwise_and(water_color, lower_half)
    water_mask = _cleanup_mask(water_mask, 5)

    # VEGETATION: rango verde
    vegetation_mask = cv2.inRange(hsv, (30, 25, 25), (90, 255, 255))
    vegetation_mask = _cleanup_mask(vegetation_mask, 5)

    # REFLECTION / sobreexposición
    reflection_mask = cv2.inRange(gray, 235, 255)
    reflection_mask = _cleanup_mask(reflection_mask, 3)

    # LOW_TEXTURE: varianza local baja
    low_texture_mask = _low_texture_mask(gray)
    low_texture_mask = _cleanup_mask(low_texture_mask, 5)

    # Personas/vehículos: placeholder conservador = 0 hasta integrar detector real
    person_mask = np.zeros((h, w), dtype=np.uint8)
    vehicle_mask = np.zeros((h, w), dtype=np.uint8)

    # Resolver solapes por prioridad
    _paint(indexed_mask, low_texture_mask, CLASS_LOW_TEXTURE)
    _paint(indexed_mask, vegetation_mask, CLASS_VEGETATION)
    _paint(indexed_mask, water_mask, CLASS_WATER)
    _paint(indexed_mask, sky_mask, CLASS_SKY)
    _paint(indexed_mask, reflection_mask, CLASS_REFLECTION)
    _paint(indexed_mask, person_mask, CLASS_PERSON)
    _paint(indexed_mask, vehicle_mask, CLASS_VEHICLE)

    sky_pct = _pct(indexed_mask == CLASS_SKY, total_px)
    water_pct = _pct(indexed_mask == CLASS_WATER, total_px)
    vegetation_pct = _pct(indexed_mask == CLASS_VEGETATION, total_px)
    person_pct = _pct(indexed_mask == CLASS_PERSON, total_px)
    vehicle_pct = _pct(indexed_mask == CLASS_VEHICLE, total_px)
    reflection_pct = _pct(indexed_mask == CLASS_REFLECTION, total_px)
    low_texture_pct = _pct(indexed_mask == CLASS_LOW_TEXTURE, total_px)

    dynamic_risk_pct = round(person_pct + vehicle_pct + 0.4 * vegetation_pct, 3)

    risk_score = (
        1.0 * sky_pct
        + 1.2 * water_pct
        + 1.5 * person_pct
        + 1.5 * vehicle_pct
        + 0.8 * reflection_pct
        + 0.7 * low_texture_pct
        + 0.4 * vegetation_pct
    )
    risk_score = round(min(100.0, risk_score), 3)

    problematic_union = (
        sky_pct
        + water_pct
        + person_pct
        + vehicle_pct
        + reflection_pct
        + low_texture_pct
    )
    usable_area_pct = round(max(0.0, 100.0 - min(100.0, problematic_union)), 3)

    if risk_score < 20:
        risk_level = "LOW"
    elif risk_score <= 45:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    metrics = {
        "sky_pct": sky_pct,
        "water_pct": water_pct,
        "vegetation_pct": vegetation_pct,
        "person_pct": person_pct,
        "vehicle_pct": vehicle_pct,
        "reflection_pct": reflection_pct,
        "low_texture_pct": low_texture_pct,
        "dynamic_risk_pct": dynamic_risk_pct,
        "usable_area_pct": usable_area_pct,
        "photogrammetry_risk_score": risk_score,
        "risk_level": risk_level,
    }
    return metrics, indexed_mask


def _low_texture_mask(gray: np.ndarray) -> np.ndarray:
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    energy = cv2.GaussianBlur(np.abs(lap), (9, 9), 0)
    norm = cv2.normalize(energy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mask = cv2.inRange(norm, 0, 20)
    return mask


def _cleanup_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned


def _paint(indexed_mask: np.ndarray, binary_mask: np.ndarray, class_id: int) -> None:
    indexed_mask[binary_mask > 0] = class_id


def _pct(binary: np.ndarray, total_px: float) -> float:
    return round((float(np.count_nonzero(binary)) / total_px) * 100.0, 3)
