from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.pipeline.context import PipelineContext
from src.pipeline.logging_setup import setup_logging
from src.pipeline.runner import PipelineRunner
from src.phases.phase0_ingest import IngestPhase
from src.phases.phase1_quality import QualityPhase
from src.phases.phase2_semantic import SemanticPhase
from src.phases.phase3_geospatial import GeospatialPhase
from src.phases.phase4_decision import DecisionPhase
from src.phases.phase5_report import ReportPhase


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="intelliframes4colmap - análisis inteligente de fotogramas previo a COLMAP",
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="input",
        help="Ruta de entrada: video o carpeta de imágenes (por defecto: ./input)",
    )
    parser.add_argument(
        "-o", "--output",
        default="output",
        help="Carpeta de salida (por defecto: ./output)",
    )
    parser.add_argument(
        "--telemetry",
        default=None,
        help="Ruta opcional a un archivo de telemetría GNSS/IMU (.gpx, .csv, .log)",
    )
    parser.add_argument(
        "--unattended",
        action="store_true",
        help="No preguntar nada por consola; aplica --on-missing-dep automáticamente",
    )
    parser.add_argument(
        "--on-missing-dep",
        choices=["install", "skip", "fail"],
        default="install",
        help="Qué hacer cuando falta una dependencia de una fase (por defecto: install)",
    )
    parser.add_argument(
        "--only-phases",
        nargs="*",
        default=None,
        help="Ejecutar solo estas fases (por nombre), ej: --only-phases ingest quality",
    )
    parser.add_argument(
        "--skip-phases",
        nargs="*",
        default=None,
        help="Saltar estas fases (por nombre)",
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Carpeta donde se escriben los logs de ejecución (por defecto: ./logs)",
    )
    parser.add_argument(
        "--no-auto-environment-mask",
        action="store_true",
        help=(
            "En modo automático, no excluir por defecto cielo/agua/reflejos "
            "de las máscaras (por defecto sí se excluyen)."
        ),
    )
    return parser.parse_args(argv)


def build_runner() -> PipelineRunner:
    return PipelineRunner([
        IngestPhase(),
        QualityPhase(),
        SemanticPhase(),
        GeospatialPhase(),
        DecisionPhase(),
        ReportPhase(),
    ])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    log_file = setup_logging(input_name=args.input, log_dir=args.log_dir)

    logging.info("Iniciando pipeline para input: %s", args.input)

    ctx = PipelineContext(
        input_path=Path(args.input),
        output_dir=Path(args.output),
        unattended=args.unattended,
        on_missing_dep=args.on_missing_dep,
        auto_environment_mask=not args.no_auto_environment_mask,
    )
    if args.telemetry:
        ctx.telemetry_path = Path(args.telemetry)

    runner = build_runner()

    try:
        runner.run(
            ctx,
            only_phases=set(args.only_phases) if args.only_phases else None,
            skip_phases=set(args.skip_phases) if args.skip_phases else None,
        )
    except Exception:
        # Cualquier error no controlado por una fase concreta llega aquí.
        # Se deja constancia completa (traceback) en el log en disco.
        logging.critical("Error fatal no capturado. Pipeline abortado.", exc_info=True)
        return 1
    finally:
        logging.info("Pipeline finalizado. Log guardado en: %s", log_file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
