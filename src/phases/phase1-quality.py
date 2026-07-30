"""
Fase 1 — Calidad de imagen

Implementación real (no simulada) de:
- Sharpness Score: Laplaciano + imagen integral + ventana deslizante + std local
- Motion blur: derivado del sharpness score bajo
- Optical flow: Farneback, para distinguir movimiento de cámara vs de objetos
- Overlap dinámico: nº de features (ORB) en común entre frames consecutivos
"""
from __future__ import annotations

import logging

import csv
from pathlib import Path

from ..pipeline.context import PipelineContext
from ..pipeline.phase import Phase
from ..pipeline.tool_check import DependencyReport, check_python_package

SHARPNESS_WINDOW = 31          # sharpnessWindowSize del README
SHARPNESS_THRESHOLD = 15.0     # por debajo => descartado por desenfoque


class QualityPhase(Phase):
    name = "quality"
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
        import cv2
        import numpy as np
        
        
        logging.info("Iniciando ejecución del pipeline")
        for phase in self.phases:
            phase_name = phase.__class__.__name__
            logging.info(f"--- Ejecutando fase: {phase_name} ---")
            try:
                phase.execute(self.context)
                logging.info(f"Fase {phase_name} completada exitosamente")
            except Exception as e:
                logging.error(f"Error en fase {phase_name}: {str(e)}", exc_info=True)
                # Dependiendo de la estrategia, podrías detener o continuar
                raise  # o break, o manejar según política
        logging.info("Pipeline finalizado")
        
        
        if not ctx.frame_list:
            raise RuntimeError("No hay frames cargados; la fase 'ingest' debe ejecutarse antes.")

        sharpness_rows = []
        motion_rows = []
        orb = cv2.ORB_create(nfeatures=1000)
        prev_gray = None
        prev_kp_des = None

        for idx, frame_path in enumerate(ctx.frame_list):
            img = cv2.imread(frame_path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            sharpness = _sharpness_score(gray, SHARPNESS_WINDOW)
            sharpness_rows.append({"frame": Path(frame_path).name, "sharpness": round(sharpness, 3)})

            overlap_pct = None
            flow_magnitude = None
            if prev_gray is not None:
                flow_magnitude = _mean_optical_flow(prev_gray, gray)
                overlap_pct = _feature_overlap(orb, prev_kp_des, gray)

            motion_rows.append({
                "frame": Path(frame_path).name,
                "optical_flow_magnitude": round(flow_magnitude, 4) if flow_magnitude is not None else "",
                "feature_overlap_pct": round(overlap_pct, 2) if overlap_pct is not None else "",
            })

            prev_gray = gray
            prev_kp_des = orb.detectAndCompute(gray, None)

            # Clasificación básica: nítido/borroso -> selected/rejected
            dest_dir = ctx.frames_selected_dir if sharpness >= SHARPNESS_THRESHOLD else ctx.frames_rejected_dir
            _copy_frame(frame_path, dest_dir)

            if idx % 5 == 0:
                _write_thumbnail(cv2, frame_path, ctx.thumbnails_dir)

        _write_csv(ctx.metrics_dir / "sharpness.csv", sharpness_rows, ["frame", "sharpness"])
        _write_csv(
            ctx.metrics_dir / "motion.csv",
            motion_rows,
            ["frame", "optical_flow_magnitude", "feature_overlap_pct"],
        )

        avg_sharpness = sum(r["sharpness"] for r in sharpness_rows) / max(len(sharpness_rows), 1)
        overlaps = [r["feature_overlap_pct"] for r in motion_rows if r["feature_overlap_pct"] != ""]
        avg_overlap = sum(overlaps) / len(overlaps) if overlaps else None

        ctx.metrics["quality"] = {
            "frames_analyzed": len(sharpness_rows),
            "frames_selected": len(list(ctx.frames_selected_dir.glob("*"))),
            "frames_rejected": len(list(ctx.frames_rejected_dir.glob("*"))),
            "avg_sharpness": round(avg_sharpness, 3),
            "avg_feature_overlap_pct": round(avg_overlap, 2) if avg_overlap is not None else None,
            "sharpness_threshold": SHARPNESS_THRESHOLD,
        }


def _sharpness_score(gray_img, window_size: int) -> float:
    """Laplaciano -> imagen integral -> ventana deslizante -> std local -> máximo."""
    import cv2
    import numpy as np

    lap = cv2.Laplacian(gray_img, cv2.CV_64F)
    lap_sq = lap ** 2

    integral = cv2.integral(lap_sq)
    h, w = lap_sq.shape
    win = min(window_size, h, w)
    if win < 2:
        return float(np.std(lap))

    best_std = 0.0
    step = max(win // 2, 1)
    for y in range(0, h - win, step):
        for x in range(0, w - win, step):
            region_sum = (
                integral[y + win, x + win]
                - integral[y, x + win]
                - integral[y + win, x]
                + integral[y, x]
            )
            local_var = region_sum / (win * win)
            local_std = local_var ** 0.5
            if local_std > best_std:
                best_std = local_std
    return float(best_std)


def _mean_optical_flow(prev_gray, gray) -> float:
    import cv2

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    magnitude = (flow[..., 0] ** 2 + flow[..., 1] ** 2) ** 0.5
    return float(magnitude.mean())


def _feature_overlap(orb, prev_kp_des, gray) -> float | None:
    import cv2

    if prev_kp_des is None or prev_kp_des[1] is None:
        return None
    kp2, des2 = orb.detectAndCompute(gray, None)
    des1 = prev_kp_des[1]
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    denom = min(len(des1), len(des2))
    return 100.0 * len(matches) / denom if denom else 0.0


def _copy_frame(src: str, dest_dir: Path) -> None:
    import shutil

    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest_dir / Path(src).name)


def _write_thumbnail(cv2, frame_path: str, thumb_dir: Path) -> None:
    thumb_dir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(frame_path)
    if img is None:
        return
    h, w = img.shape[:2]
    scale = 320 / w
    resized = cv2.resize(img, (320, int(h * scale)))
    cv2.imwrite(str(thumb_dir / Path(frame_path).name), resized)


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
