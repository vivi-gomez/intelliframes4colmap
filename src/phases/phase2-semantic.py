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
import json
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from ..pipeline.context import PipelineContext
from ..pipeline.phase import Phase
from ..pipeline.tool_check import DependencyReport, check_python_package


class SemanticPhase(Phase):
    """
    Fase 2: análisis semántico ligero y segmentación.
    - siempre calcula textura + exposición
    - intenta segmentación con SAM si está disponible
    - si no, usa fallback clásico
    - si falla todo, no rompe el pipeline
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

    def run(self, ctx) -> None:
        frames = list(getattr(ctx, "frame_list", []) or [])
        if not frames:
            ctx.metrics.setdefault("semantic", {})
            ctx.metrics["semantic"]["status"] = "skipped_no_frames"
            ctx.semantic = {
                "frames": [],
                "summary": {
                    "mode": "none",
                    "processed_frames": 0,
                    "avg_texture_score": 0.0,
                    "avg_exposure_score": 0.0,
                    "avg_usable_area_pct": 0.0,
                    "avg_risk_score": 0.0,
                    "high_risk_frames": 0,
                    "sky_dominant_frames": 0,
                    "dynamic_content_frames": 0,
                    "reflection_problem_frames": 0,
                },
            }
            return

        metrics_dir = Path(ctx.metrics_dir)
        masks_dir = Path(ctx.masks_dir)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        texture_rows = self._compute_texture_metrics(frames)
        exposure_rows = self._compute_exposure_metrics(frames)

        self._write_csv(metrics_dir / "texture.csv", texture_rows)
        self._write_csv(metrics_dir / "exposure.csv", exposure_rows)

        segmentation_rows, segmentation_mode = self._run_segmentation(ctx, frames, masks_dir)

        if segmentation_rows:
            self._write_csv(metrics_dir / "segmentation.csv", segmentation_rows)
            self._write_segmentation_summary(
                metrics_dir / "segmentation_summary.json",
                segmentation_rows,
                segmentation_mode,
            )

        avg_texture = round(
            float(np.mean([row["texture_score"] for row in texture_rows])) if texture_rows else 0.0,
            3,
        )
        avg_exposure = round(
            float(np.mean([row["exposure_score"] for row in exposure_rows])) if exposure_rows else 0.0,
            3,
        )

        semantic_summary = self._build_semantic_summary(
            segmentation_rows=segmentation_rows,
            segmentation_mode=segmentation_mode,
            avg_texture=avg_texture,
            avg_exposure=avg_exposure,
        )

        ctx.semantic = {
            "frames": segmentation_rows,
            "summary": semantic_summary,
        }

        ctx.metrics.setdefault("semantic", {})
        ctx.metrics["semantic"].update(
            {
                "status": "done",
                "texture_csv": str(metrics_dir / "texture.csv"),
                "exposure_csv": str(metrics_dir / "exposure.csv"),
                "avg_texture_score": avg_texture,
                "avg_exposure_score": avg_exposure,
                "segmentation": "done" if segmentation_rows else "skipped",
                "segmentation_mode": segmentation_mode,
                "avg_usable_area_pct": semantic_summary["avg_usable_area_pct"],
                "avg_risk_score": semantic_summary["avg_risk_score"],
            }
        )

    def _compute_texture_metrics(self, frames: List[str]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []

        for frame_path in frames:
            img = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                rows.append(
                    {
                        "frame": Path(frame_path).name,
                        "texture_score": 0.0,
                        "texture_level": "UNKNOWN",
                        "error": "unreadable_frame",
                    }
                )
                continue

            lap = cv2.Laplacian(img, cv2.CV_64F)
            score = float(lap.var())

            if score < 20:
                level = "LOW"
            elif score < 80:
                level = "MEDIUM"
            else:
                level = "HIGH"

            rows.append(
                {
                    "frame": Path(frame_path).name,
                    "texture_score": round(score, 3),
                    "texture_level": level,
                }
            )

        return rows

    def _compute_exposure_metrics(self, frames: List[str]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []

        for frame_path in frames:
            img = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                rows.append(
                    {
                        "frame": Path(frame_path).name,
                        "mean_brightness": 0.0,
                        "std_brightness": 0.0,
                        "exposure_score": 0.0,
                        "exposure_level": "UNKNOWN",
                        "error": "unreadable_frame",
                    }
                )
                continue

            mean_val = float(np.mean(img))
            std_val = float(np.std(img))

            # ideal simple around midtones
            exposure_score = max(0.0, 100.0 - abs(mean_val - 127.5) * 0.75)

            if exposure_score < 40:
                level = "POOR"
            elif exposure_score < 70:
                level = "FAIR"
            else:
                level = "GOOD"

            rows.append(
                {
                    "frame": Path(frame_path).name,
                    "mean_brightness": round(mean_val, 3),
                    "std_brightness": round(std_val, 3),
                    "exposure_score": round(exposure_score, 3),
                    "exposure_level": level,
                }
            )

        return rows

    def _run_segmentation(self, ctx, frames: List[str], masks_dir: Path):
        dep_log = ctx.dependency_log.setdefault(self.name, {})
        checkpoint = self._find_sam_checkpoint(ctx)

        try:
            import torch  # noqa: F401
            from segment_anything import sam_model_registry  # noqa: F401
            from ._segmentation_backend import run_segmentation

            if checkpoint:
                rows = run_segmentation(
                    frame_list=frames,
                    masks_dir=masks_dir,
                    mode="sam",
                    checkpoint_path=checkpoint,
                )
                dep_log["segmentation"] = "sam"
                return rows, "sam"
        except Exception as exc:
            dep_log["sam_error"] = str(exc)

        try:
            from ._segmentation_backend import run_segmentation

            rows = run_segmentation(
                frame_list=frames,
                masks_dir=masks_dir,
                mode="classical",
                checkpoint_path=None,
            )
            dep_log["segmentation"] = "classical"
            return rows, "classical"
        except Exception as exc:
            dep_log["segmentation"] = "skipped"
            dep_log["segmentation_error"] = str(exc)
            return [], "none"

    def _find_sam_checkpoint(self, ctx) -> str | None:
        candidates = [
            getattr(ctx, "sam_checkpoint", None),
            "sam_vit_h_4b8939.pth",
            "models/sam_vit_h_4b8939.pth",
            "checkpoints/sam_vit_h_4b8939.pth",
        ]
        for candidate in candidates:
            if not candidate:
                continue
            p = Path(candidate)
            if p.exists():
                return str(p)
        return None

    def _build_semantic_summary(
        self,
        segmentation_rows: List[Dict[str, Any]],
        segmentation_mode: str,
        avg_texture: float,
        avg_exposure: float,
    ) -> Dict[str, Any]:
        if not segmentation_rows:
            return {
                "mode": segmentation_mode,
                "processed_frames": 0,
                "avg_texture_score": avg_texture,
                "avg_exposure_score": avg_exposure,
                "avg_usable_area_pct": 0.0,
                "avg_risk_score": 0.0,
                "high_risk_frames": 0,
                "sky_dominant_frames": 0,
                "dynamic_content_frames": 0,
                "reflection_problem_frames": 0,
            }

        avg_usable = round(
            float(np.mean([row.get("usable_area_pct", 0.0) for row in segmentation_rows])), 3
        )
        avg_risk = round(
            float(np.mean([row.get("photogrammetry_risk_score", 0.0) for row in segmentation_rows])),
            3,
        )
        high_risk = sum(1 for row in segmentation_rows if row.get("risk_level") == "HIGH")
        sky_dominant = sum(1 for row in segmentation_rows if row.get("sky_pct", 0.0) >= 35.0)
        dynamic_content = sum(
            1 for row in segmentation_rows if row.get("dynamic_risk_pct", 0.0) >= 10.0
        )
        reflection_problem = sum(
            1 for row in segmentation_rows if row.get("reflection_pct", 0.0) >= 10.0
        )

        return {
            "mode": segmentation_mode,
            "processed_frames": len(segmentation_rows),
            "avg_texture_score": avg_texture,
            "avg_exposure_score": avg_exposure,
            "avg_usable_area_pct": avg_usable,
            "avg_risk_score": avg_risk,
            "high_risk_frames": high_risk,
            "sky_dominant_frames": sky_dominant,
            "dynamic_content_frames": dynamic_content,
            "reflection_problem_frames": reflection_problem,
        }

    def _write_segmentation_summary(
        self,
        path: Path,
        rows: List[Dict[str, Any]],
        mode: str,
    ) -> None:
        summary = self._build_semantic_summary(
            segmentation_rows=rows,
            segmentation_mode=mode,
            avg_texture=0.0,
            avg_exposure=0.0,
        )
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    def _write_csv(self, path: Path, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return

        fieldnames = list(rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
