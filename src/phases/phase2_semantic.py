"""
Fase 2 — Análisis semántico y máscaras

- Texture richness y exposición: reales, solo necesitan OpenCV/numpy.
- Máscaras (convención COLMAP: blanco = procesar, negro = ignorar):
    * Modo ATENDIDO (ctx.unattended == False): se pregunta una única vez,
      antes de procesar ningún frame, "What do you want to ignore from
      images?". Solo se excluyen las categorías que el usuario pida.
    * Modo AUTOMÁTICO (ctx.unattended == True): las categorías estáticas
      "universales" (cielo, agua, reflejos) se excluyen por defecto
      (desactivable con --no-auto-environment-mask). Las categorías
      dinámicas (personas, vehículos, animales, aves) SOLO se excluyen si
      _scene_analyzer.py las marca como anomalía respecto al resto de la
      secuencia (aparición transitoria o movimiento incoherente con la
      cámara) — nunca por pertenecer a esa clase sin más.
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from ..pipeline.context import PipelineContext
from ..pipeline.phase import Phase
from ..pipeline.tool_check import DependencyReport, check_python_package
from . import _segmentation_backend as seg

logger = logging.getLogger(__name__)


class SemanticPhase(Phase):
    """
    Fase 2: análisis semántico ligero (textura, exposición) + máscaras.
    """

    name = "semantic"
    optional = False

    def check_dependencies(self) -> DependencyReport:
        return DependencyReport(
            phase_name=self.name,
            checks=[
                check_python_package("cv2", "opencv-python-headless"),
                check_python_package("numpy", "numpy"),
            ],
        )

    def run(self, ctx: PipelineContext) -> None:
        try:
            self._run(ctx)
        except Exception:
            logger.error("Fallo en la fase semántica", exc_info=True)
            raise

    def _run(self, ctx: PipelineContext) -> None:
        frames = list(getattr(ctx, "frame_list", []) or [])
        if not frames:
            logger.warning("Fase semántica: no hay frames, se omite el análisis.")
            ctx.metrics.setdefault("semantic", {})
            ctx.metrics["semantic"]["status"] = "skipped_no_frames"
            ctx.semantic = {"frames": [], "summary": _empty_summary("none")}
            return

        metrics_dir = Path(ctx.metrics_dir)
        masks_dir = Path(ctx.masks_dir)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        texture_rows = self._compute_texture_metrics(frames)
        exposure_rows = self._compute_exposure_metrics(frames)
        self._write_csv(metrics_dir / "texture.csv", texture_rows)
        self._write_csv(metrics_dir / "exposure.csv", exposure_rows)

        static_categories, dynamic_boxes_per_frame, mask_mode = self._decide_masking_plan(ctx, frames)

        segmentation_rows = seg.run_segmentation(
            frame_list=frames,
            masks_dir=masks_dir,
            requested_static_categories=static_categories,
            dynamic_boxes_per_frame=dynamic_boxes_per_frame,
        )

        if segmentation_rows:
            self._write_csv(metrics_dir / "segmentation.csv", segmentation_rows)

        avg_texture = round(float(np.mean([r["texture_score"] for r in texture_rows])) if texture_rows else 0.0, 3)
        avg_exposure = round(float(np.mean([r["exposure_score"] for r in exposure_rows])) if exposure_rows else 0.0, 3)

        semantic_summary = self._build_semantic_summary(
            segmentation_rows, mask_mode, static_categories, dynamic_boxes_per_frame, avg_texture, avg_exposure
        )
        (metrics_dir / "segmentation_summary.json").write_text(
            json.dumps(semantic_summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        ctx.semantic = {"frames": segmentation_rows, "summary": semantic_summary}

        ctx.metrics.setdefault("semantic", {})
        ctx.metrics["semantic"].update({
            "status": "done",
            "texture_csv": str(metrics_dir / "texture.csv"),
            "exposure_csv": str(metrics_dir / "exposure.csv"),
            "avg_texture_score": avg_texture,
            "avg_exposure_score": avg_exposure,
            "mask_mode": mask_mode,
            "masked_static_categories": sorted(static_categories),
            "avg_usable_area_pct": semantic_summary["avg_usable_area_pct"],
        })

    # -- Plan de máscaras ---------------------------------------------------

    def _decide_masking_plan(self, ctx: PipelineContext, frames: List[str]):
        """
        Decide qué se enmascara y cómo, según el modo de ejecución.

        Devuelve (static_categories, dynamic_boxes_per_frame, mode_label).
        """
        if not ctx.unattended:
            requested = seg.prompt_ignore_categories()
            static_categories = requested & seg.STATIC_CATEGORIES
            dynamic_categories = requested & seg.DYNAMIC_CATEGORIES

            dynamic_boxes_per_frame: Dict[str, List[Dict[str, Any]]] = {}
            if dynamic_categories:
                dynamic_boxes_per_frame = self._detect_requested_dynamic_categories(
                    frames, dynamic_categories
                )
            return static_categories, dynamic_boxes_per_frame, "attended"

        # Modo automático.
        auto_environment = getattr(ctx, "auto_environment_mask", True)
        static_categories = set(seg.AUTO_ENVIRONMENT_CATEGORIES) if auto_environment else set()

        try:
            from ._scene_analyzer import analyze_dynamic_objects
            dynamic_boxes_per_frame = analyze_dynamic_objects(frames)
        except Exception:
            logger.warning(
                "No se pudo ejecutar el análisis de anomalías de secuencia; "
                "no se enmascarará ningún objeto dinámico.",
                exc_info=True,
            )
            dynamic_boxes_per_frame = {}

        return static_categories, dynamic_boxes_per_frame, "automatic"

    def _detect_requested_dynamic_categories(
        self, frames: List[str], dynamic_categories: set
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Modo atendido: el usuario pidió explícitamente ignorar personas,
        vehículos, animales y/o aves. Aquí SÍ se enmascara toda detección de
        esas categorías (es una petición explícita, no una inferencia).
        """
        result: Dict[str, List[Dict[str, Any]]] = {}
        for frame_path in frames:
            img = cv2.imread(str(frame_path))
            if img is None:
                continue
            detections = [
                d for d in seg.detect_dynamic_boxes(img)
                if d["category"] in dynamic_categories
            ]
            if detections:
                result[Path(frame_path).name] = detections
        return result

    # -- Métricas de textura / exposición -----------------------------------

    def _compute_texture_metrics(self, frames: List[str]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for frame_path in frames:
            img = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                rows.append({"frame": Path(frame_path).name, "texture_score": 0.0, "texture_level": "UNKNOWN", "error": "unreadable_frame"})
                continue
            score = float(cv2.Laplacian(img, cv2.CV_64F).var())
            level = "LOW" if score < 20 else ("MEDIUM" if score < 80 else "HIGH")
            rows.append({"frame": Path(frame_path).name, "texture_score": round(score, 3), "texture_level": level})
        return rows

    def _compute_exposure_metrics(self, frames: List[str]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for frame_path in frames:
            img = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                rows.append({"frame": Path(frame_path).name, "mean_brightness": 0.0, "std_brightness": 0.0, "exposure_score": 0.0, "exposure_level": "UNKNOWN", "error": "unreadable_frame"})
                continue
            mean_val = float(np.mean(img))
            std_val = float(np.std(img))
            exposure_score = max(0.0, 100.0 - abs(mean_val - 127.5) * 0.75)
            level = "POOR" if exposure_score < 40 else ("FAIR" if exposure_score < 70 else "GOOD")
            rows.append({
                "frame": Path(frame_path).name,
                "mean_brightness": round(mean_val, 3),
                "std_brightness": round(std_val, 3),
                "exposure_score": round(exposure_score, 3),
                "exposure_level": level,
            })
        return rows

    def _build_semantic_summary(
        self,
        segmentation_rows: List[Dict[str, Any]],
        mask_mode: str,
        static_categories: set,
        dynamic_boxes_per_frame: Dict[str, List[Dict[str, Any]]],
        avg_texture: float,
        avg_exposure: float,
    ) -> Dict[str, Any]:
        if not segmentation_rows:
            return _empty_summary(mask_mode, avg_texture, avg_exposure)

        avg_usable = round(float(np.mean([r.get("usable_area_pct", 0.0) for r in segmentation_rows])), 3)
        frames_with_dynamic_mask = len(dynamic_boxes_per_frame)
        anomaly_reasons = sorted({
            det.get("reason", "")
            for dets in dynamic_boxes_per_frame.values()
            for det in dets
            if det.get("reason")
        })

        return {
            "mode": mask_mode,
            "processed_frames": len(segmentation_rows),
            "avg_texture_score": avg_texture,
            "avg_exposure_score": avg_exposure,
            "avg_usable_area_pct": avg_usable,
            "masked_static_categories": sorted(static_categories),
            "frames_with_dynamic_objects_masked": frames_with_dynamic_mask,
            "dynamic_anomaly_reasons": anomaly_reasons,
        }

    def _write_csv(self, path: Path, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        try:
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        except OSError:
            logger.error("No se pudo escribir el CSV %s", path, exc_info=True)
            raise


def _empty_summary(mode: str, avg_texture: float = 0.0, avg_exposure: float = 0.0) -> Dict[str, Any]:
    return {
        "mode": mode,
        "processed_frames": 0,
        "avg_texture_score": avg_texture,
        "avg_exposure_score": avg_exposure,
        "avg_usable_area_pct": 0.0,
        "masked_static_categories": [],
        "frames_with_dynamic_objects_masked": 0,
        "dynamic_anomaly_reasons": [],
    }
