"""
Fase 4 — Motor de decisión

Prepara una configuración recomendada de COLMAP a partir de métricas de
calidad, semántica y geoespaciales.

Objetivos:
- Consolidar señales de fases anteriores.
- Estimar si el conjunto es apto para una reconstrucción estable.
- Proponer una configuración razonable para COLMAP.
- Generar artefactos simples y trazables para la siguiente fase.

Entradas esperadas:
- ctx.frame_list
- ctx.metrics["quality"]              (si existe)
- ctx.metrics["semantic"]             (si existe)
- ctx.semantic                        (si existe)
- ctx.geospatial                      (si existe)

Salidas principales:
- colmap/colmap_config.json
- colmap/frames_for_colmap.txt
- colmap/decision_summary.json
- ctx.decision
- ctx.metrics["decision"]

Notas:
- No modifica físicamente los frames.
- No ejecuta COLMAP; solo recomienda configuración y selección.
- Debe degradar con elegancia si faltan métricas previas.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..pipeline.context import PipelineContext
from ..pipeline.phase import Phase
from ..pipeline.tool_check import DependencyReport

logger = logging.getLogger(__name__)


class DecisionPhase(Phase):
    """
    Fase 4: motor de decisión para preparar una configuración recomendada
    de COLMAP a partir de métricas de calidad, semántica y geoespaciales.
    """

    name = "decision"
    optional = False

    def check_dependencies(self) -> DependencyReport:
        """
        Esta fase no depende de librerías externas pesadas.
        """
        return DependencyReport(phase_name=self.name, checks=[])

    def run(self, ctx: PipelineContext) -> None:
        try:
            self._run(ctx)
        except Exception:
            logger.error("Fallo en la fase de decisión", exc_info=True)
            raise

    def _run(self, ctx: PipelineContext) -> None:
        """
        Ejecuta el motor de decisión.

        Flujo:
        - recopila datos de contexto,
        - calcula una puntuación por frame,
        - selecciona frames recomendados,
        - genera una configuración sugerida para COLMAP,
        - persiste el resultado en disco y en contexto.
        """
        frames = list(getattr(ctx, "frame_list", []) or [])
        colmap_dir = Path(ctx.colmap_dir)
        colmap_dir.mkdir(parents=True, exist_ok=True)

        ctx.metrics.setdefault("decision", {})

        if not frames:
            decision = self._build_empty_decision()
            self._persist_decision(ctx, decision, colmap_dir)
            ctx.metrics["decision"].update(
                {
                    "status": "skipped_no_frames",
                    "recommended_frames": 0,
                    "dataset_readiness": "LOW",
                }
            )
            return

        quality_rows = self._collect_quality_rows(ctx, frames)
        semantic_rows = self._collect_semantic_rows(ctx)
        geospatial_summary = self._collect_geospatial_summary(ctx)

        frame_decisions = self._score_frames(
            frames=frames,
            quality_rows=quality_rows,
            semantic_rows=semantic_rows,
        )

        selected_frames = [row for row in frame_decisions if row["selected"]]
        dataset_summary = self._build_dataset_summary(
            total_frames=len(frames),
            selected_frames=selected_frames,
            all_frame_decisions=frame_decisions,
            geospatial_summary=geospatial_summary,
        )

        colmap_config = self._build_colmap_config(
            dataset_summary=dataset_summary,
            geospatial_summary=geospatial_summary,
        )

        decision = {
            "frames": frame_decisions,
            "selected_frame_paths": [row["frame_path"] for row in selected_frames],
            "summary": dataset_summary,
            "colmap_config": colmap_config,
        }

        self._persist_decision(ctx, decision, colmap_dir)

        ctx.metrics["decision"].update(
            {
                "status": "done",
                "recommended_frames": len(selected_frames),
                "dataset_readiness": dataset_summary["dataset_readiness"],
                "decision_summary_json": str(colmap_dir / "decision_summary.json"),
                "colmap_config_json": str(colmap_dir / "colmap_config.json"),
                "frames_for_colmap_txt": str(colmap_dir / "frames_for_colmap.txt"),
            }
        )

    def _collect_quality_rows(self, ctx, frames: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Recoge métricas de calidad por frame si existen.

        Se toleran estructuras parciales o ausentes.
        """
        quality_map: Dict[str, Dict[str, Any]] = {}

        quality_metrics = getattr(ctx, "metrics", {}).get("quality", {})
        if isinstance(quality_metrics, dict):
            rows = quality_metrics.get("frames", [])
            if isinstance(rows, list):
                for row in rows:
                    frame_name = row.get("frame")
                    if frame_name:
                        quality_map[frame_name] = row
            else:
                logger.warning(
                    "ctx.metrics['quality']['frames'] no es una lista; "
                    "se usará puntuación neutra para todos los frames."
                )
        else:
            logger.info("No hay métricas de calidad previas; se usará puntuación neutra.")

        return quality_map

    def _collect_semantic_rows(self, ctx) -> Dict[str, Dict[str, Any]]:
        """
        Recoge resultados semánticos por frame si existen.
        """
        semantic_map: Dict[str, Dict[str, Any]] = {}

        semantic = getattr(ctx, "semantic", {}) or {}
        rows = semantic.get("frames", [])
        if isinstance(rows, list):
            for row in rows:
                frame_name = row.get("frame")
                if frame_name:
                    semantic_map[frame_name] = row

        return semantic_map

    def _collect_geospatial_summary(self, ctx) -> Dict[str, Any]:
        """
        Recupera el resumen geoespacial si existe.
        """
        geospatial = getattr(ctx, "geospatial", {}) or {}
        return geospatial.get("summary", {}) if isinstance(geospatial, dict) else {}

    def _score_frames(
        self,
        frames: List[str],
        quality_rows: Dict[str, Dict[str, Any]],
        semantic_rows: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Puntúa cada frame combinando calidad y semántica.

        Filosofía:
        - penalizar frames con alto riesgo fotogramétrico,
        - favorecer textura suficiente y exposición razonable,
        - mantener reglas simples y transparentes.
        """
        decisions: List[Dict[str, Any]] = []

        for frame_path in frames:
            frame_name = Path(frame_path).name
            q = quality_rows.get(frame_name, {})
            s = semantic_rows.get(frame_name, {})

            quality_score = self._estimate_quality_score(q)
            semantic_penalty = self._estimate_semantic_penalty(s)

            final_score = max(0.0, min(100.0, quality_score - semantic_penalty))

            selected, reject_reasons = self._selection_rule(
                quality_row=q,
                semantic_row=s,
                final_score=final_score,
            )

            decisions.append(
                {
                    "frame": frame_name,
                    "frame_path": str(frame_path),
                    "quality_score": round(quality_score, 3),
                    "semantic_penalty": round(semantic_penalty, 3),
                    "final_score": round(final_score, 3),
                    "selected": selected,
                    "reject_reasons": reject_reasons,
                    "risk_level": s.get("risk_level", "UNKNOWN"),
                    "usable_area_pct": s.get("usable_area_pct", ""),
                    "photogrammetry_risk_score": s.get("photogrammetry_risk_score", ""),
                }
            )

        return decisions

    def _estimate_quality_score(self, quality_row: Dict[str, Any]) -> float:
        """
        Estima una puntuación base de calidad.

        Regla:
        - si existen métricas por frame, las combina;
        - si no existen, usa un valor neutro.
        """
        if not quality_row:
            return 60.0

        components: List[float] = []

        for key in (
            "sharpness_score",
            "overlap_score",
            "motion_score",
            "quality_score",
        ):
            value = quality_row.get(key)
            if isinstance(value, (int, float)):
                components.append(float(value))

        if not components:
            return 60.0

        return sum(components) / len(components)

    def _estimate_semantic_penalty(self, semantic_row: Dict[str, Any]) -> float:
        """
        Estima una penalización a partir de señales semánticas.
        """
        if not semantic_row:
            return 0.0

        if isinstance(semantic_row.get("photogrammetry_risk_score"), (int, float)):
            return float(semantic_row["photogrammetry_risk_score"]) * 0.7

        penalty = 0.0
        for key, weight in (
            ("sky_pct", 0.25),
            ("water_pct", 0.35),
            ("reflection_pct", 0.25),
            ("low_texture_pct", 0.20),
            ("vegetation_pct", 0.10),
            ("person_pct", 0.50),
            ("vehicle_pct", 0.50),
        ):
            value = semantic_row.get(key)
            if isinstance(value, (int, float)):
                penalty += float(value) * weight

        return penalty

    def _selection_rule(
        self,
        quality_row: Dict[str, Any],
        semantic_row: Dict[str, Any],
        final_score: float,
    ) -> Tuple[bool, List[str]]:
        """
        Decide si un frame debe recomendarse para COLMAP.

        Reglas iniciales:
        - rechaza puntuaciones finales bajas,
        - rechaza riesgo semántico muy alto,
        - rechaza área utilizable demasiado baja.
        """
        reasons: List[str] = []

        if final_score < 35.0:
            reasons.append("low_final_score")

        risk_level = semantic_row.get("risk_level")
        if risk_level == "HIGH":
            reasons.append("high_semantic_risk")

        usable_area = semantic_row.get("usable_area_pct")
        if isinstance(usable_area, (int, float)) and usable_area < 35.0:
            reasons.append("low_usable_area")

        if not quality_row and not semantic_row:
            # Si no hay datos, no bloquear por completo.
            # Mantenemos el frame como provisionalmente válido.
            return True, []

        return len(reasons) == 0, reasons

    def _build_dataset_summary(
        self,
        total_frames: int,
        selected_frames: List[Dict[str, Any]],
        all_frame_decisions: List[Dict[str, Any]],
        geospatial_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Construye un resumen global del conjunto.
        """
        selected_count = len(selected_frames)
        rejected_count = max(0, total_frames - selected_count)
        selection_ratio = (selected_count / total_frames) * 100.0 if total_frames else 0.0

        avg_final_score = (
            sum(row["final_score"] for row in all_frame_decisions) / len(all_frame_decisions)
            if all_frame_decisions
            else 0.0
        )

        avg_selected_score = (
            sum(row["final_score"] for row in selected_frames) / len(selected_frames)
            if selected_frames
            else 0.0
        )

        coverage = geospatial_summary.get("coverage", "none")
        gps_available = bool(geospatial_summary.get("gps_available", False))

        dataset_readiness = self._classify_dataset_readiness(
            selected_ratio=selection_ratio,
            avg_final_score=avg_final_score,
            gps_available=gps_available,
        )

        return {
            "total_frames": total_frames,
            "selected_frames": selected_count,
            "rejected_frames": rejected_count,
            "selection_ratio_pct": round(selection_ratio, 3),
            "avg_final_score": round(avg_final_score, 3),
            "avg_selected_score": round(avg_selected_score, 3),
            "dataset_readiness": dataset_readiness,
            "gps_available": gps_available,
            "coverage": coverage,
        }

    def _classify_dataset_readiness(
        self,
        selected_ratio: float,
        avg_final_score: float,
        gps_available: bool,
    ) -> str:
        """
        Clasifica la preparación del conjunto para COLMAP.
        """
        if avg_final_score >= 65.0 and selected_ratio >= 60.0:
            return "HIGH" if gps_available else "MEDIUM_HIGH"

        if avg_final_score >= 45.0 and selected_ratio >= 40.0:
            return "MEDIUM"

        return "LOW"

    def _build_colmap_config(
        self,
        dataset_summary: Dict[str, Any],
        geospatial_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Genera una configuración sugerida para COLMAP.

        Es una recomendación inicial, no una verdad absoluta.
        """
        readiness = dataset_summary["dataset_readiness"]
        coverage = geospatial_summary.get("coverage", "none")
        gps_available = bool(geospatial_summary.get("gps_available", False))

        config = {
            "feature_extraction": {
                "SiftExtraction.max_num_features": 8192,
                "SiftExtraction.estimate_affine_shape": False,
                "SiftExtraction.domain_size_pooling": False,
            },
            "matching": {
                "strategy": "exhaustive",
                "SiftMatching.guided_matching": True,
            },
            "mapping": {
                "Mapper.ba_global_max_num_iterations": 50,
                "Mapper.min_num_matches": 15,
                "Mapper.init_min_num_inliers": 80,
            },
            "priors": {
                "use_gps_priors": gps_available,
                "coverage": coverage,
            },
        }

        # Ajustes heurísticos por nivel del dataset.
        if readiness in ("HIGH", "MEDIUM_HIGH"):
            config["feature_extraction"]["SiftExtraction.max_num_features"] = 12000
            config["mapping"]["Mapper.min_num_matches"] = 20

        elif readiness == "LOW":
            config["feature_extraction"]["SiftExtraction.max_num_features"] = 6000
            config["mapping"]["Mapper.min_num_matches"] = 12
            config["mapping"]["Mapper.init_min_num_inliers"] = 60

        # Selección de estrategia de matching.
        if coverage in ("wide", "moderate"):
            config["matching"]["strategy"] = "sequential"
        elif coverage in ("local", "very_local", "single_point", "none"):
            config["matching"]["strategy"] = "exhaustive"

        return config

    def _build_empty_decision(self) -> Dict[str, Any]:
        """
        Devuelve una estructura vacía consistente.
        """
        return {
            "frames": [],
            "selected_frame_paths": [],
            "summary": {
                "total_frames": 0,
                "selected_frames": 0,
                "rejected_frames": 0,
                "selection_ratio_pct": 0.0,
                "avg_final_score": 0.0,
                "avg_selected_score": 0.0,
                "dataset_readiness": "LOW",
                "gps_available": False,
                "coverage": "none",
            },
            "colmap_config": {
                "feature_extraction": {},
                "matching": {},
                "mapping": {},
                "priors": {},
            },
        }

    def _persist_decision(self, ctx, decision: Dict[str, Any], colmap_dir: Path) -> None:
        """
        Persiste artefactos de decisión en disco y contexto.
        """
        try:
            decision_summary_path = colmap_dir / "decision_summary.json"
            decision_summary_path.write_text(
                json.dumps(decision["summary"], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            colmap_config_path = colmap_dir / "colmap_config.json"
            colmap_config_path.write_text(
                json.dumps(decision["colmap_config"], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            frames_txt_path = colmap_dir / "frames_for_colmap.txt"
            frames_txt_path.write_text(
                "\n".join(decision["selected_frame_paths"]),
                encoding="utf-8",
            )
        except OSError:
            logger.error("No se pudieron escribir los artefactos de decisión en %s", colmap_dir, exc_info=True)
            raise

        ctx.decision = {
            "frames": decision["frames"],
            "summary": decision["summary"],
            "colmap_config": decision["colmap_config"],
            "selected_frame_paths": decision["selected_frame_paths"],
        }
