"""
Fase 1 — Calidad de imagen

Implementación real (no simulada) de:
- Sharpness Score: Laplaciano + imagen integral + ventana deslizante + std local
- Motion blur: derivado del sharpness score bajo
- Optical flow: Farneback, para distinguir movimiento de cámara vs de objetos
- Overlap dinámico: nº de features (ORB) en común entre frames consecutivos
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

from ..pipeline.context import PipelineContext
from ..pipeline.phase import Phase
from ..pipeline.tool_check import DependencyReport, check_python_package

logger = logging.getLogger(__name__)

SHARPNESS_WINDOW = 31          # sharpnessWindowSize del README
SHARPNESS_THRESHOLD = 15.0     # por debajo => descartado por desenfoque
# Rango orientativo de sharpness usado solo para normalizar a 0-100 y poder
# combinar esta métrica con las de otras fases en el motor de decisión.
SHARPNESS_NORMALIZATION_CAP = 60.0


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
        try:
            self._run(ctx)
        except Exception:
            logger.error("Fallo en la fase de calidad", exc_info=True)
            raise

    def _run(self, ctx: PipelineContext) -> None:
        import cv2
        import numpy as np

        if not ctx.frame_list:
            raise RuntimeError("No hay frames cargados; la fase 'ingest' debe ejecutarse antes.")

        sharpness_rows = []
        motion_rows = []
        quality_frame_rows = []
        unreadable = 0
        orb = cv2.ORB_create(nfeatures=1000)
        prev_gray = None
        prev_kp_des = None

        for idx, frame_path in enumerate(ctx.frame_list):
            img = cv2.imread(frame_path)
            if img is None:
                unreadable += 1
                logger.warning("Frame ilegible, se omite: %s", frame_path)
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            try:
                sharpness = _sharpness_score(gray, SHARPNESS_WINDOW)
            except Exception:
                logger.warning("No se pudo calcular sharpness para %s", frame_path, exc_info=True)
                sharpness = 0.0

            sharpness_rows.append({"frame": Path(frame_path).name, "sharpness": round(sharpness, 3)})

            overlap_pct = None
            flow_magnitude = None
            if prev_gray is not None:
                try:
                    flow_magnitude = _mean_optical_flow(prev_gray, gray)
                    overlap_pct = _feature_overlap(orb, prev_kp_des, gray)
                except Exception:
                    logger.warning(
                        "No se pudo calcular motion/overlap para %s", frame_path, exc_info=True
                    )

            motion_rows.append({
                "frame": Path(frame_path).name,
                "optical_flow_magnitude": round(flow_magnitude, 4) if flow_magnitude is not None else "",
                "feature_overlap_pct": round(overlap_pct, 2) if overlap_pct is not None else "",
            })

            # Normalizamos sharpness y overlap a 0-100 para que la fase de
            # decisión pueda combinarlos con otras señales sin conocer sus
            # escalas originales.
            quality_score = round(min(100.0, (sharpness / SHARPNESS_NORMALIZATION_CAP) * 100.0), 3)
            overlap_score = round(overlap_pct, 3) if overlap_pct is not None else None
            quality_frame_rows.append({
                "frame": Path(frame_path).name,
                "sharpness_score": quality_score,
                "overlap_score": overlap_score,
                "quality_score": quality_score,
            })

            prev_gray = gray
            prev_kp_des = orb.detectAndCompute(gray, None)

            # Clasificación básica: nítido/borroso -> selected/rejected
            dest_dir = ctx.frames_selected_dir if sharpness >= SHARPNESS_THRESHOLD else ctx.frames_rejected_dir
            try:
                _copy_frame(frame_path, dest_dir)
            except OSError:
                logger.warning("No se pudo copiar el frame %s a %s", frame_path, dest_dir, exc_info=True)

            if idx % 5 == 0:
                try:
                    _write_thumbnail(cv2, frame_path, ctx.thumbnails_dir)
                except Exception:
                    logger.warning("No se pudo generar thumbnail para %s", frame_path, exc_info=True)

        if unreadable:
            logger.warning("%d frame(s) ilegibles durante la fase de calidad", unreadable)

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
            "frames_unreadable": unreadable,
            "avg_sharpness": round(avg_sharpness, 3),
            "avg_feature_overlap_pct": round(avg_overlap, 2) if avg_overlap is not None else None,
            "sharpness_threshold": SHARPNESS_THRESHOLD,
            # Datos por-frame que consume phase4_decision.py para puntuar
            # cada frame; antes de esta corrección no se exponían y la
            # fase de decisión siempre caía en el valor neutro por defecto.
            "frames": quality_frame_rows,
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
    try:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    except OSError:
        logger.error("No se pudo escribir el CSV %s", path, exc_info=True)
        raise
