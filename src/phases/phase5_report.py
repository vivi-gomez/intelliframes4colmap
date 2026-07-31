"""
Fase 5 — Generación de artefactos finales de análisis

Objetivos:
- Consolidar los resultados de todas las fases previas.
- Generar un analysis.json como salida estructurada canónica.
- Generar un report.html legible para inspección humana.
- Registrar las rutas de salida en el contexto y en métricas.

Entradas esperadas:
- ctx.metrics
- ctx.semantic
- ctx.geospatial
- ctx.decision
- ctx.dependency_log

Salidas principales:
- analysis.json
- report.html
- ctx.metrics["report"]
- ctx.report

Notas:
- No debe romper el pipeline si faltan datos de alguna fase.
- Debe funcionar incluso con resultados parciales.
"""
from __future__ import annotations

import logging
from pathlib import Path

from ..pipeline.context import PipelineContext
from ..pipeline.phase import Phase
from ..pipeline.tool_check import DependencyReport
from ._report_builder import (
    build_analysis_payload,
    build_html_report,
    save_analysis_json,
    save_html_report,
)

logger = logging.getLogger(__name__)


class ReportPhase(Phase):

    name = "report"
    optional = False

    def check_dependencies(self) -> DependencyReport:
        """
        Esta fase no requiere dependencias externas pesadas.
        """
        return DependencyReport(phase_name=self.name, checks=[])

    def run(self, ctx: PipelineContext) -> None:
        try:
            self._run(ctx)
        except Exception:
            logger.error("Fallo en la fase de reporte", exc_info=True)
            raise

    def _run(self, ctx: PipelineContext) -> None:
        """
        Genera los artefactos finales del pipeline.

        Flujo:
        - construye el payload consolidado,
        - lo guarda como JSON,
        - genera un informe HTML simple,
        - registra las salidas en el contexto.
        """
        # PipelineContext expone `output_dir`, no `workspace_dir`: usar el
        # nombre correcto evita que el reporte se escriba fuera de la
        # carpeta de salida del proyecto.
        output_dir = Path(getattr(ctx, "output_dir", "."))
        output_dir.mkdir(parents=True, exist_ok=True)

        analysis_path = self._resolve_analysis_path(ctx, output_dir)
        report_path = self._resolve_report_path(ctx, output_dir)

        ctx.metrics.setdefault("report", {})

        try:
            analysis = build_analysis_payload(ctx)
            save_analysis_json(analysis_path, analysis)

            html = build_html_report(ctx, analysis)
            save_html_report(report_path, html)
        except Exception:
            logger.error("No se pudieron generar los artefactos finales del reporte", exc_info=True)
            raise

        ctx.report = {
            "analysis_json": str(analysis_path),
            "report_html": str(report_path),
        }

        ctx.metrics["report"].update(
            {
                "status": "done",
                "analysis_json": str(analysis_path),
                "report_html": str(report_path),
            }
        )
        logger.info("Reporte generado: %s / %s", analysis_path, report_path)

    def _resolve_analysis_path(self, ctx, output_dir: Path) -> Path:
        """
        Resuelve la ruta de salida para analysis.json.

        Si el contexto ya define una ruta específica, la respeta.
        """
        explicit = getattr(ctx, "analysis_json_path", None)
        if explicit:
            return Path(explicit)

        return output_dir / "analysis.json"

    def _resolve_report_path(self, ctx, output_dir: Path) -> Path:
        """
        Resuelve la ruta de salida para report.html.

        Si el contexto ya define una ruta específica, la respeta.
        """
        explicit = getattr(ctx, "report_html_path", None)
        if explicit:
            return Path(explicit)

        return output_dir / "report.html"
