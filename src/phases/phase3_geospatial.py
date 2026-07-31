"""
Fase 3 — Análisis geoespacial y de telemetría

Objetivos:
- Leer metadatos geoespaciales desde los frames seleccionados (EXIF).
- Si el usuario aporta un archivo de telemetría externo (GPX/CSV/LOG,
  ver --telemetry), sincronizarlo a cada frame mediante Spline Cúbica
  (sección 2.5 del README) y convertir a coordenadas cartesianas locales.
- Detectar si existe información GNSS/IMU utilizable.
- Estimar cobertura espacial básica del conjunto.
- Generar métricas exportables para fases posteriores.

Salidas principales:
- metrics/geospatial.csv
- metrics/geospatial_summary.json
- ctx.geospatial
- ctx.metrics["geospatial"]

Notas:
- No aborta el pipeline si no encuentra GPS/IMU ni telemetría externa.
- Prioriza EXIF estándar por frame; si un frame no tiene EXIF GPS pero sí
  hay telemetría externa sincronizada, se usa esta última.
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..pipeline.context import PipelineContext
from ..pipeline.phase import Phase
from ..pipeline.tool_check import DependencyReport, check_python_package
from ._telemetry_sync import TelemetryError, geodetic_to_local_enu, load_telemetry, sync_to_frame_times

logger = logging.getLogger(__name__)


class GeospatialPhase(Phase):
    """
    Fase 3: análisis geoespacial y de telemetría ligera.
    """

    name = "geospatial"
    optional = True  # el pipeline puede seguir aunque no haya GPS/IMU/telemetría

    def check_dependencies(self) -> DependencyReport:
        """
        Esta fase intenta usar Pillow para leer EXIF de imágenes. Si no
        está disponible, se reporta como dependencia faltante y, al ser
        una fase opcional, el pipeline la saltará en vez de abortar.
        """
        return DependencyReport(
            phase_name=self.name,
            checks=[check_python_package("PIL", "Pillow")],
        )

    def run(self, ctx: PipelineContext) -> None:
        try:
            self._run(ctx)
        except Exception:
            logger.error("Fallo en la fase geoespacial", exc_info=True)
            raise

    def _run(self, ctx: PipelineContext) -> None:
        """
        Ejecuta el análisis geoespacial sobre los frames disponibles.
        """
        frames = list(getattr(ctx, "frame_list", []) or [])
        metrics_dir = Path(ctx.metrics_dir)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        ctx.metrics.setdefault("geospatial", {})

        if not frames:
            ctx.geospatial = self._empty_geospatial_summary(mode="no_frames")
            ctx.metrics["geospatial"].update(
                {
                    "status": "skipped_no_frames",
                    "gps_available": False,
                    "imu_available": False,
                    "coverage": "none",
                    "processed_frames": 0,
                }
            )
            return

        rows: List[Dict[str, Any]] = []
        for frame_path in frames:
            row = self._extract_frame_geodata(frame_path)
            rows.append(row)

        telemetry_info = self._sync_external_telemetry(ctx, frames, rows)

        csv_path = metrics_dir / "geospatial.csv"
        self._write_csv(csv_path, rows)

        summary = self._build_summary(rows)
        summary["telemetry"] = telemetry_info

        summary_path = metrics_dir / "geospatial_summary.json"
        try:
            summary_path.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            logger.error("No se pudo escribir %s", summary_path, exc_info=True)
            raise

        ctx.geospatial = {
            "frames": rows,
            "summary": summary,
            "coverage": summary["coverage"],
            "gps_available": summary["gps_available"],
            "imu_available": summary["imu_available"],
            "bbox": summary["bbox"],
        }

        ctx.metrics["geospatial"].update(
            {
                "status": "done",
                "csv": str(csv_path),
                "summary_json": str(summary_path),
                "gps_available": summary["gps_available"],
                "imu_available": summary["imu_available"],
                "coverage": summary["coverage"],
                "processed_frames": summary["processed_frames"],
                "frames_with_gps": summary["frames_with_gps"],
                "frames_with_imu": summary["frames_with_imu"],
                "telemetry_source": telemetry_info["source"],
            }
        )

    # ------------------------------------------------------------------
    # Sincronización GNSS/IMU externa (README 2.5)
    # ------------------------------------------------------------------

    def _sync_external_telemetry(
        self,
        ctx: PipelineContext,
        frames: List[str],
        rows: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Si ctx.telemetry_path apunta a un archivo GPX/CSV/LOG, lo carga,
        lo sincroniza a cada frame mediante spline cúbica y rellena en
        `rows` los frames que no tenían GPS/IMU vía EXIF.

        Cualquier fallo (archivo inválido, formato no reconocido, tiempos
        no calculables) se registra en el log y en ctx.dependency_log, y
        la fase continúa solo con los datos EXIF ya extraídos: nunca
        rompe el pipeline por un problema de telemetría externa.
        """
        dep_log = ctx.dependency_log.setdefault(self.name, {})
        telemetry_path = getattr(ctx, "telemetry_path", None)

        info: Dict[str, Any] = {
            "source": "none",
            "samples": 0,
            "frames_filled_from_telemetry": 0,
        }

        if not telemetry_path:
            return info

        try:
            samples = load_telemetry(telemetry_path)
            frame_times = self._estimate_frame_times(ctx, frames, rows)
            synced = sync_to_frame_times(samples, frame_times)

            origin = next((s for s in synced if s.get("lat") is not None), None)
            filled = 0
            for row, sync_row in zip(rows, synced):
                if sync_row.get("lat") is None or sync_row.get("lon") is None:
                    continue
                if not row.get("has_gps"):
                    row["has_gps"] = True
                    row["latitude"] = round(sync_row["lat"], 8)
                    row["longitude"] = round(sync_row["lon"], 8)
                    if sync_row.get("alt") is not None:
                        row["altitude_m"] = round(sync_row["alt"], 3)
                    row["telemetry_quality"] = (
                        "PARTIAL" if sync_row.get("extrapolated") else "GOOD"
                    )
                    filled += 1
                if not row.get("has_imu"):
                    if sync_row.get("yaw") is not None:
                        row["heading_deg"] = round(sync_row["yaw"], 3)
                        row["has_imu"] = True
                    if sync_row.get("pitch") is not None:
                        row["pitch_deg"] = round(sync_row["pitch"], 3)
                        row["has_imu"] = True
                    if sync_row.get("roll") is not None:
                        row["roll_deg"] = round(sync_row["roll"], 3)
                        row["has_imu"] = True

                if origin is not None:
                    local = geodetic_to_local_enu(
                        lat=sync_row["lat"], lon=sync_row["lon"], alt=sync_row.get("alt"),
                        origin_lat=origin["lat"], origin_lon=origin["lon"], origin_alt=origin.get("alt"),
                    )
                    row.update(local)

            info.update(
                {
                    "source": Path(telemetry_path).suffix.lower().lstrip("."),
                    "samples": len(samples),
                    "frames_filled_from_telemetry": filled,
                }
            )
            dep_log["telemetry"] = "synced"
            logger.info(
                "Telemetría externa sincronizada: %d muestras, %d frames completados.",
                len(samples), filled,
            )
        except TelemetryError as exc:
            logger.warning("Telemetría externa no usable (%s); se continúa solo con EXIF.", exc)
            dep_log["telemetry_error"] = str(exc)
        except Exception as exc:
            logger.error("Fallo inesperado sincronizando telemetría externa.", exc_info=True)
            dep_log["telemetry_error"] = str(exc)

        return info

    def _estimate_frame_times(
        self,
        ctx: PipelineContext,
        frames: List[str],
        rows: List[Dict[str, Any]],
    ) -> List[float]:
        """
        Estima el tiempo (en segundos, relativo al primer frame) de cada
        frame para poder interpolar la telemetría sobre él.

        - Si el video fue extraído por ffmpeg y conocemos el FPS de
          muestreo (ctx.metadata["fps"]), se usa índice/fps: es exacto
          porque ffmpeg extrae frames a intervalos regulares.
        - Si no (carpeta de imágenes sueltas), se usan los timestamps
          EXIF ya extraídos por frame cuando existen.
        - Si no hay ninguna de las dos cosas, se asume 1 frame/segundo y
          se registra que la sincronización será aproximada.
        """
        fps = (ctx.metadata or {}).get("fps")
        if fps and fps > 0:
            return [i / float(fps) for i in range(len(frames))]

        timestamps = [row.get("timestamp") for row in rows]
        if any(timestamps):
            parsed = []
            for ts in timestamps:
                parsed.append(self._parse_exif_seconds(ts))
            valid = [p for p in parsed if p is not None]
            if valid:
                t0 = min(valid)
                return [p - t0 if p is not None else float(i) for i, p in enumerate(parsed)]

        logger.warning(
            "No hay FPS ni timestamps EXIF disponibles; se asume 1 frame/segundo "
            "para sincronizar telemetría (aproximado)."
        )
        return [float(i) for i in range(len(frames))]

    @staticmethod
    def _parse_exif_seconds(timestamp: Optional[str]) -> Optional[float]:
        if not timestamp:
            return None
        from datetime import datetime

        for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(timestamp[:19], fmt).timestamp()
            except ValueError:
                continue
        return None

    # ------------------------------------------------------------------
    # Extracción EXIF por frame
    # ------------------------------------------------------------------

    def _extract_frame_geodata(self, frame_path: str) -> Dict[str, Any]:
        """
        Extrae metadatos geoespaciales de un frame individual.

        Si el frame no tiene metadatos útiles, devuelve una fila válida
        con indicadores vacíos y estado legible.
        """
        from PIL import Image
        from PIL.ExifTags import GPSTAGS, TAGS

        frame_name = Path(frame_path).name

        base_row: Dict[str, Any] = {
            "frame": frame_name,
            "has_exif": False,
            "has_gps": False,
            "has_imu": False,
            "latitude": "",
            "longitude": "",
            "altitude_m": "",
            "gps_dop": "",
            "heading_deg": "",
            "pitch_deg": "",
            "roll_deg": "",
            "timestamp": "",
            "bbox_cell": "",
            "telemetry_quality": "NONE",
            "error": "",
        }

        try:
            img = Image.open(frame_path)
            exif_raw = img.getexif()

            if not exif_raw:
                base_row["error"] = "no_exif"
                return base_row

            base_row["has_exif"] = True

            exif_data = {}
            for tag_id, value in exif_raw.items():
                tag_name = TAGS.get(tag_id, tag_id)
                exif_data[tag_name] = value

            gps_info_raw = exif_data.get("GPSInfo")
            gps_info = {}
            if gps_info_raw:
                for key, value in gps_info_raw.items():
                    gps_tag_name = GPSTAGS.get(key, key)
                    gps_info[gps_tag_name] = value

            lat, lon = self._extract_lat_lon(gps_info)
            alt = self._extract_altitude(gps_info)
            dop = self._extract_gps_dop(gps_info)
            heading = self._extract_heading(gps_info, exif_data)
            pitch = self._extract_pitch(exif_data)
            roll = self._extract_roll(exif_data)
            timestamp = self._extract_timestamp(exif_data, gps_info)

            has_gps = lat is not None and lon is not None
            has_imu = any(v is not None for v in [heading, pitch, roll])

            base_row["has_gps"] = has_gps
            base_row["has_imu"] = has_imu
            base_row["latitude"] = "" if lat is None else round(lat, 8)
            base_row["longitude"] = "" if lon is None else round(lon, 8)
            base_row["altitude_m"] = "" if alt is None else round(alt, 3)
            base_row["gps_dop"] = "" if dop is None else round(dop, 3)
            base_row["heading_deg"] = "" if heading is None else round(heading, 3)
            base_row["pitch_deg"] = "" if pitch is None else round(pitch, 3)
            base_row["roll_deg"] = "" if roll is None else round(roll, 3)
            base_row["timestamp"] = timestamp or ""

            if has_gps:
                base_row["bbox_cell"] = self._make_bbox_cell(lat, lon)

            base_row["telemetry_quality"] = self._classify_telemetry_quality(
                has_gps=has_gps,
                altitude=alt,
                dop=dop,
                has_imu=has_imu,
            )

            return base_row

        except Exception as exc:
            logger.debug("No se pudo leer EXIF de %s: %s", frame_path, exc)
            base_row["error"] = f"read_error: {exc}"
            return base_row

    def _build_summary(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Construye el resumen geoespacial global.
        """
        processed_frames = len(rows)
        gps_rows = [r for r in rows if r.get("has_gps")]
        imu_rows = [r for r in rows if r.get("has_imu")]

        bbox = self._compute_bbox(gps_rows)
        approx_path_distance_m = self._compute_path_distance(gps_rows)
        coverage = self._classify_coverage(gps_rows, bbox, approx_path_distance_m)

        telemetry_counts = {
            "good": sum(1 for r in rows if r.get("telemetry_quality") == "GOOD"),
            "partial": sum(1 for r in rows if r.get("telemetry_quality") == "PARTIAL"),
            "poor": sum(1 for r in rows if r.get("telemetry_quality") == "POOR"),
            "none": sum(1 for r in rows if r.get("telemetry_quality") == "NONE"),
        }

        return {
            "mode": "exif",
            "processed_frames": processed_frames,
            "frames_with_gps": len(gps_rows),
            "frames_with_imu": len(imu_rows),
            "gps_available": len(gps_rows) > 0,
            "imu_available": len(imu_rows) > 0,
            "coverage": coverage,
            "bbox": bbox,
            "approx_path_distance_m": round(approx_path_distance_m, 3),
            "telemetry_quality_counts": telemetry_counts,
        }

    def _empty_geospatial_summary(self, mode: str = "none") -> Dict[str, Any]:
        """
        Devuelve un resumen vacío para casos en los que no hay datos.
        """
        empty_bbox = {
            "min_lat": None, "max_lat": None,
            "min_lon": None, "max_lon": None,
            "min_alt_m": None, "max_alt_m": None,
        }
        return {
            "frames": [],
            "summary": {
                "mode": mode,
                "processed_frames": 0,
                "frames_with_gps": 0,
                "frames_with_imu": 0,
                "gps_available": False,
                "imu_available": False,
                "coverage": "none",
                "bbox": empty_bbox,
                "approx_path_distance_m": 0.0,
                "telemetry_quality_counts": {"good": 0, "partial": 0, "poor": 0, "none": 0},
                "telemetry": {"source": "none", "samples": 0, "frames_filled_from_telemetry": 0},
            },
            "coverage": "none",
            "gps_available": False,
            "imu_available": False,
            "bbox": empty_bbox,
        }

    def _extract_lat_lon(self, gps_info: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
        try:
            lat = gps_info.get("GPSLatitude")
            lat_ref = gps_info.get("GPSLatitudeRef")
            lon = gps_info.get("GPSLongitude")
            lon_ref = gps_info.get("GPSLongitudeRef")

            if not lat or not lat_ref or not lon or not lon_ref:
                return None, None

            lat_dd = self._dms_to_decimal(lat, lat_ref)
            lon_dd = self._dms_to_decimal(lon, lon_ref)
            return lat_dd, lon_dd
        except Exception:
            return None, None

    def _extract_altitude(self, gps_info: Dict[str, Any]) -> Optional[float]:
        try:
            alt = gps_info.get("GPSAltitude")
            alt_ref = gps_info.get("GPSAltitudeRef", 0)

            if alt is None:
                return None

            alt_value = self._rational_to_float(alt)
            if alt_value is None:
                return None

            if alt_ref == 1:
                alt_value = -alt_value

            return alt_value
        except Exception:
            return None

    def _extract_gps_dop(self, gps_info: Dict[str, Any]) -> Optional[float]:
        try:
            dop = gps_info.get("GPSDOP")
            return self._rational_to_float(dop)
        except Exception:
            return None

    def _extract_heading(
        self,
        gps_info: Dict[str, Any],
        exif_data: Dict[str, Any],
    ) -> Optional[float]:
        try:
            heading = gps_info.get("GPSImgDirection")
            if heading is not None:
                return self._rational_to_float(heading)

            for key in ("ImageDirection", "CameraHeading", "Heading"):
                if key in exif_data:
                    return self._rational_to_float(exif_data[key])

            return None
        except Exception:
            return None

    def _extract_pitch(self, exif_data: Dict[str, Any]) -> Optional[float]:
        for key in ("CameraPitch", "Pitch", "GimbalPitchDegree"):
            if key in exif_data:
                return self._rational_to_float(exif_data[key])
        return None

    def _extract_roll(self, exif_data: Dict[str, Any]) -> Optional[float]:
        for key in ("CameraRoll", "Roll", "GimbalRollDegree"):
            if key in exif_data:
                return self._rational_to_float(exif_data[key])
        return None

    def _extract_timestamp(
        self,
        exif_data: Dict[str, Any],
        gps_info: Dict[str, Any],
    ) -> Optional[str]:
        for key in ("DateTimeOriginal", "DateTime", "DateTimeDigitized"):
            if key in exif_data and exif_data[key]:
                return str(exif_data[key])

        gps_date = gps_info.get("GPSDateStamp")
        gps_time = gps_info.get("GPSTimeStamp")

        if gps_date and gps_time:
            try:
                h = self._rational_to_float(gps_time[0])
                m = self._rational_to_float(gps_time[1])
                s = self._rational_to_float(gps_time[2])
                return f"{gps_date} {int(h):02d}:{int(m):02d}:{int(s):02d}Z"
            except Exception:
                return str(gps_date)

        return None

    def _classify_telemetry_quality(
        self,
        has_gps: bool,
        altitude: Optional[float],
        dop: Optional[float],
        has_imu: bool,
    ) -> str:
        if not has_gps and not has_imu:
            return "NONE"

        if has_gps:
            if (altitude is not None or has_imu) and (dop is None or dop <= 5.0):
                return "GOOD"
            return "PARTIAL"

        if has_imu:
            return "POOR"

        return "NONE"

    def _compute_bbox(self, gps_rows: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
        if not gps_rows:
            return {
                "min_lat": None, "max_lat": None,
                "min_lon": None, "max_lon": None,
                "min_alt_m": None, "max_alt_m": None,
            }

        lats = [float(r["latitude"]) for r in gps_rows if r.get("latitude") != ""]
        lons = [float(r["longitude"]) for r in gps_rows if r.get("longitude") != ""]
        alts = [float(r["altitude_m"]) for r in gps_rows if r.get("altitude_m") != ""]

        return {
            "min_lat": round(min(lats), 8) if lats else None,
            "max_lat": round(max(lats), 8) if lats else None,
            "min_lon": round(min(lons), 8) if lons else None,
            "max_lon": round(max(lons), 8) if lons else None,
            "min_alt_m": round(min(alts), 3) if alts else None,
            "max_alt_m": round(max(alts), 3) if alts else None,
        }

    def _compute_path_distance(self, gps_rows: List[Dict[str, Any]]) -> float:
        if len(gps_rows) < 2:
            return 0.0

        total = 0.0
        previous = None

        for row in gps_rows:
            current = (float(row["latitude"]), float(row["longitude"]))
            if previous is not None:
                total += self._haversine_m(previous[0], previous[1], current[0], current[1])
            previous = current

        return total

    def _classify_coverage(
        self,
        gps_rows: List[Dict[str, Any]],
        bbox: Dict[str, Optional[float]],
        approx_path_distance_m: float,
    ) -> str:
        if not gps_rows:
            return "none"

        if len(gps_rows) == 1:
            return "single_point"

        lat_span = 0.0
        lon_span = 0.0

        if bbox["min_lat"] is not None and bbox["max_lat"] is not None:
            lat_span = abs(bbox["max_lat"] - bbox["min_lat"])
        if bbox["min_lon"] is not None and bbox["max_lon"] is not None:
            lon_span = abs(bbox["max_lon"] - bbox["min_lon"])

        if approx_path_distance_m < 5:
            return "very_local"

        if approx_path_distance_m < 50:
            return "local"

        if lat_span > 0.001 or lon_span > 0.001:
            return "wide"

        return "moderate"

    def _make_bbox_cell(self, lat: float, lon: float, decimals: int = 4) -> str:
        return f"{round(lat, decimals)}|{round(lon, decimals)}"

    def _dms_to_decimal(self, dms: Any, ref: str) -> Optional[float]:
        try:
            degrees = self._rational_to_float(dms[0])
            minutes = self._rational_to_float(dms[1])
            seconds = self._rational_to_float(dms[2])

            if degrees is None or minutes is None or seconds is None:
                return None

            decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)

            if str(ref).upper() in ("S", "W"):
                decimal *= -1.0

            return decimal
        except Exception:
            return None

    def _rational_to_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None

        try:
            if isinstance(value, (int, float)):
                return float(value)

            if isinstance(value, tuple) and len(value) == 2:
                num, den = value
                den = float(den)
                if den == 0:
                    return None
                return float(num) / den

            if hasattr(value, "numerator") and hasattr(value, "denominator"):
                den = float(value.denominator)
                if den == 0:
                    return None
                return float(value.numerator) / den

            return float(value)
        except Exception:
            return None

    def _haversine_m(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
    ) -> float:
        from math import asin, cos, radians, sin, sqrt

        r = 6371000.0

        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)

        a = (
            sin(dlat / 2) ** 2
            + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
        )
        c = 2 * asin(sqrt(a))
        return r * c

    def _write_csv(self, path: Path, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return

        fieldnames = sorted({key for row in rows for key in row.keys()})
        try:
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        except OSError:
            logger.error("No se pudo escribir el CSV %s", path, exc_info=True)
            raise
