"""
    Fase 5: generación de artefactos finales de análisis.

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

from pathlib import Path

from src.pipeline.phase import Phase
from src.phases._report_builder import (
    build_analysis_payload,
    build_html_report,
    save_analysis_json,
    save_html_report,
)


class ReportPhase(Phase):


    def __init__(self) -> None:
        super().__init__("phase5-report")

    def check_dependencies(self) -> bool:
        """
        Esta fase no requiere dependencias externas pesadas.
        """
        return True

    def run(self, ctx) -> None:
        """
        Genera los artefactos finales del pipeline.

        Flujo:
        - construye el payload consolidado,
        - lo guarda como JSON,
        - genera un informe HTML simple,
        - registra las salidas en el contexto.
        """
        workspace_dir = Path(getattr(ctx, "workspace_dir", "."))
        metrics_dir = Path(getattr(ctx, "metrics_dir", workspace_dir / "metrics"))
        metrics_dir.mkdir(parents=True, exist_ok=True)

        analysis_path = self._resolve_analysis_path(ctx, workspace_dir)
        report_path = self._resolve_report_path(ctx, workspace_dir)

        ctx.metrics.setdefault("report", {})

        analysis = build_analysis_payload(ctx)
        save_analysis_json(analysis_path, analysis)

        html = build_html_report(ctx, analysis)
        save_html_report(report_path, html)

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

    def _resolve_analysis_path(self, ctx, workspace_dir: Path) -> Path:
        """
        Resuelve la ruta de salida para analysis.json.

        Si el contexto ya define una ruta específica, la respeta.
        """
        explicit = getattr(ctx, "analysis_json_path", None)
        if explicit:
            return Path(explicit)

        return workspace_dir / "analysis.json"

    def _resolve_report_path(self, ctx, workspace_dir: Path) -> Path:
        """
        Resuelve la ruta de salida para report.html.

        Si el contexto ya define una ruta específica, la respeta.
        """
        explicit = getattr(ctx, "report_html_path", None)
        if explicit:
            return Path(explicit)

        return workspace_dir / "report.html"
