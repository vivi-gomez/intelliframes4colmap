    """
    Fase 3: análisis geoespacial y de telemetría ligera.

    Objetivos:
    - Leer metadatos geoespaciales desde los frames seleccionados.
    - Detectar si existe información GNSS utilizable.
    - Detectar si existe información tipo IMU/orientación si está disponible.
    - Estimar cobertura espacial básica del conjunto.
    - Generar métricas exportables para fases posteriores.

    Salidas principales:
    - metrics/geospatial.csv
    - metrics/geospatial_summary.json
    - ctx.geospatial
    - ctx.metrics["geospatial"]

    Notas:
    - No aborta el pipeline si no encuentra GPS/IMU.
    - Prioriza EXIF estándar.
    - Puede ampliarse más adelante con sidecars, logs de vuelo o sensores externos.
    """

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.pipeline.phase import Phase


class GeospatialPhase(Phase):


    def __init__(self) -> None:
        super().__init__("phase3-geospatial")

    def check_dependencies(self) -> bool:
        """
        Comprueba dependencias mínimas para esta fase.

        Esta fase intenta usar Pillow para leer EXIF de imágenes.
        Si no está disponible, la fase puede marcarse como no ejecutable.
        """
        try:
            from PIL import Image  # noqa: F401
            from PIL.ExifTags import GPSTAGS, TAGS  # noqa: F401
        except Exception as exc:
            self._last_dependency_error = f"Missing geospatial deps: {exc}"
            return False

        return True

    def run(self, ctx) -> None:
        """
        Ejecuta el análisis geoespacial sobre los frames disponibles.

        Comportamiento:
        - Recorre ctx.frame_list.
        - Intenta extraer EXIF GPS y orientación básica.
        - Calcula cobertura, caja envolvente y calidad básica de telemetría.
        - Escribe resultados estructurados en contexto y en disco.
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

        csv_path = metrics_dir / "geospatial.csv"
        self._write_csv(csv_path, rows)

        summary = self._build_summary(rows)

        summary_path = metrics_dir / "geospatial_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

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
            }
        )

    def _extract_frame_geodata(self, frame_path: str) -> Dict[str, Any]:
        """
        Extrae metadatos geoespaciales de un frame individual.

        Campos buscados:
        - latitud / longitud
        - altitud
        - rumbo/orientación si existe
        - timestamp EXIF si existe

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
            base_row["error"] = f"read_error: {exc}"
            return base_row

    def _build_summary(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Construye el resumen geoespacial global.

        Incluye:
        - disponibilidad de GPS e IMU
        - número de frames con telemetría
        - caja envolvente espacial
        - cobertura estimada
        - distancia secuencial aproximada
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
                "bbox": {
                    "min_lat": None,
                    "max_lat": None,
                    "min_lon": None,
                    "max_lon": None,
                    "min_alt_m": None,
                    "max_alt_m": None,
                },
                "approx_path_distance_m": 0.0,
                "telemetry_quality_counts": {
                    "good": 0,
                    "partial": 0,
                    "poor": 0,
                    "none": 0,
                },
            },
            "coverage": "none",
            "gps_available": False,
            "imu_available": False,
            "bbox": {
                "min_lat": None,
                "max_lat": None,
                "min_lon": None,
                "max_lon": None,
                "min_alt_m": None,
                "max_alt_m": None,
            },
        }

    def _extract_lat_lon(self, gps_info: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
        """
        Convierte coordenadas EXIF GPS a grados decimales.
        """
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
        """
        Extrae la altitud si está disponible.
        """
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
        """
        Extrae el DOP GPS si existe.
        """
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
        """
        Extrae rumbo/orientación si está disponible.

        Se prioriza GPSImgDirection y luego campos EXIF equivalentes
        si más adelante aparecen en ciertos dispositivos.
        """
        try:
            heading = gps_info.get("GPSImgDirection")
            if heading is not None:
                return self._rational_to_float(heading)

            # Posibles extensiones futuras de fabricantes.
            for key in ("ImageDirection", "CameraHeading", "Heading"):
                if key in exif_data:
                    return self._rational_to_float(exif_data[key])

            return None
        except Exception:
            return None

    def _extract_pitch(self, exif_data: Dict[str, Any]) -> Optional[float]:
        """
        Extrae pitch si existe en EXIF extendido de algún fabricante.
        """
        for key in ("CameraPitch", "Pitch", "GimbalPitchDegree"):
            if key in exif_data:
                return self._rational_to_float(exif_data[key])
        return None

    def _extract_roll(self, exif_data: Dict[str, Any]) -> Optional[float]:
        """
        Extrae roll si existe en EXIF extendido de algún fabricante.
        """
        for key in ("CameraRoll", "Roll", "GimbalRollDegree"):
            if key in exif_data:
                return self._rational_to_float(exif_data[key])
        return None

    def _extract_timestamp(
        self,
        exif_data: Dict[str, Any],
        gps_info: Dict[str, Any],
    ) -> Optional[str]:
        """
        Extrae timestamp desde EXIF o GPS time/date si existe.
        """
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
        """
        Clasifica la calidad de telemetría de forma simple.

        GOOD:
        - GPS disponible
        - y además altitud o IMU, con DOP razonable o desconocido

        PARTIAL:
        - GPS disponible pero incompleto
        - o IMU sin GPS

        POOR:
        - datos presentes pero débiles

        NONE:
        - sin telemetría
        """
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
        """
        Calcula la caja envolvente espacial del conjunto con GPS.
        """
        if not gps_rows:
            return {
                "min_lat": None,
                "max_lat": None,
                "min_lon": None,
                "max_lon": None,
                "min_alt_m": None,
                "max_alt_m": None,
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
        """
        Estima la distancia recorrida sumando distancias entre frames
        consecutivos que tengan GPS.
        """
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
        """
        Clasifica la cobertura espacial de forma heurística.
        """
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
        """
        Agrupa coordenadas en una celda textual simple para análisis básico.
        """
        return f"{round(lat, decimals)}|{round(lon, decimals)}"

    def _dms_to_decimal(self, dms: Any, ref: str) -> Optional[float]:
        """
        Convierte coordenadas EXIF en formato grados/minutos/segundos a decimal.
        """
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
        """
        Convierte valores EXIF racionales a float.

        Soporta:
        - enteros y floats
        - tuplas tipo (num, den)
        - objetos con atributos numerator/denominator
        """
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
        """
        Distancia Haversine en metros entre dos coordenadas.
        """
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
        """
        Escribe un CSV simple con las filas generadas.
        """
        if not rows:
            return

        fieldnames = list(rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
