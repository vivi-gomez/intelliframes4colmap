"""
Fase 2 — Análisis semántico

- Texture richness y exposición: reales, solo necesitan OpenCV/numpy (ya
  requeridos en fase 1), así que siempre se calculan.
- Segmentación (SAM / YOLO): pesada (requiere torch + checkpoints de varios
  cientos de MB). Es opcional: si no está disponible, esta sub-parte se salta
  y se registra en el manifest, pero el resto de la fase sí se ejecuta.
"""
from __future__ import annotations

import csv
from pathlib import Path

from ..pipeline.context import PipelineContext
from ..pipeline.phase import Phase
from ..pipeline.tool_check import DependencyReport, ToolStatus, check_model_file, check_python_package

SAM_CHECKPOINT_ENV_DEFAULT = Path.home() / ".cache" / "intelliframes4colmap" / "sam_vit_h_4b8939.pth"


class SemanticPhase(Phase):
    name = "semantic"
    optional = True  # la segmentación puede faltar sin romper el pipeline

    def check_dependencies(self) -> DependencyReport:
        # Núcleo (obligatorio dentro de la fase): opencv/numpy ya deberían
        # estar si la fase 1 corrió; se listan igualmente por si se ejecuta sola.
        checks = [
            check_python_package("cv2", "opencv-python-headless"),
            check_python_package("numpy", "numpy"),
        ]
        return DependencyReport(phase_name=self.name, checks=checks)

    def run(self, ctx: PipelineContext) -> None:
        import cv2
        import numpy as np

        texture_rows, exposure_rows = [], []

        for frame_path in ctx.frame_list:
            img = cv2.imread(frame_path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            texture_rows.append({
                "frame": Path(frame_path).name,
                "texture_score": round(_texture_richness(cv2, gray), 3),
            })
            exposure_rows.append({
                "frame": Path(frame_path).name,
                **_exposure_stats(np, gray),
            })

        _write_csv(ctx.metrics_dir / "texture.csv", texture_rows, ["frame", "texture_score"])
        _write_csv(
            ctx.metrics_dir / "exposure.csv",
            exposure_rows,
            ["frame", "mean_luminance", "clipping_low_pct", "clipping_high_pct"],
        )

        avg_texture = sum(r["texture_score"] for r in texture_rows) / max(len(texture_rows), 1)
        avg_clip_high = sum(r["clipping_high_pct"] for r in exposure_rows) / max(len(exposure_rows), 1)
        avg_clip_low = sum(r["clipping_low_pct"] for r in exposure_rows) / max(len(exposure_rows), 1)

        ctx.metrics["semantic"] = {
            "avg_texture_score": round(avg_texture, 3),
            "texture_level": _texture_level(avg_texture),
            "avg_clipping_high_pct": round(avg_clip_high, 2),
            "avg_clipping_low_pct": round(avg_clip_low, 2),
            "exposure_variation": "HIGH" if (avg_clip_high + avg_clip_low) > 15 else "LOW",
        }

        self._run_segmentation_if_available(ctx)

    def _run_segmentation_if_available(self, ctx: PipelineContext) -> None:
        """Sub-parte opcional y pesada. No forma parte del contrato principal
        de dependencias de la fase: se comprueba y se salta en caliente para
        no bloquear texture/exposure si falta torch/SAM/YOLO."""
        torch_status = check_python_package("torch", "torch")
        sam_status = check_python_package("segment_anything", "segment-anything")
        checkpoint_status = check_model_file("sam_vit_h_checkpoint", SAM_CHECKPOINT_ENV_DEFAULT)

        available = torch_status.found and sam_status.found and checkpoint_status.found
        ctx.dependency_log.setdefault(self.name, {})["segmentation"] = {
            "torch": torch_status.found,
            "segment_anything": sam_status.found,
            "checkpoint": checkpoint_status.found,
        }

        if not available:
            missing = [
                s.name for s in (torch_status, sam_status, checkpoint_status) if not s.found
            ]
            print(
                f"[{self.name}] Segmentación semántica omitida (falta: {missing}). "
                f"El resto de la fase 2 se ejecutó igualmente. "
                f"Para activarla: pip install torch segment-anything, y descarga el "
                f"checkpoint en {SAM_CHECKPOINT_ENV_DEFAULT}"
            )
            ctx.metrics["semantic"]["segmentation"] = "skipped"
            return

        # Import perezoso: solo si todo está disponible.
        from ._segmentation_backend import run_sam_segmentation

        ctx.masks_dir.mkdir(parents=True, exist_ok=True)
        run_sam_segmentation(ctx.frame_list, ctx.masks_dir, SAM_CHECKPOINT_ENV_DEFAULT)
        ctx.metrics["semantic"]["segmentation"] = "done"


def _texture_richness(cv2, gray) -> float:
    """Densidad de keypoints ORB por megapíxel."""
    orb = cv2.ORB_create(nfeatures=5000)
    kp = orb.detect(gray, None)
    h, w = gray.shape
    megapixels = (h * w) / 1_000_000
    return len(kp) / megapixels if megapixels else 0.0


def _texture_level(avg_score: float) -> str:
    if avg_score >= 4000:
        return "HIGH"
    if avg_score >= 1500:
        return "MEDIUM"
    return "LOW"


def _exposure_stats(np, gray) -> dict:
    hist = np.histogram(gray, bins=256, range=(0, 255))[0]
    total = hist.sum()
    clipping_low = 100.0 * hist[:5].sum() / total if total else 0.0
    clipping_high = 100.0 * hist[-5:].sum() / total if total else 0.0
    return {
        "mean_luminance": round(float(gray.mean()), 2),
        "clipping_low_pct": round(float(clipping_low), 2),
        "clipping_high_pct": round(float(clipping_high), 2),
    }


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
