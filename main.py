import logging
import os
from datetime import datetime

def setup_logging(input_file: str = None):
    # Crear carpeta logs si no existe
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Nombre del archivo de log con timestamp y (opcional) nombre del input
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(input_file))[0] if input_file else "no_input"
    log_filename = f"{log_dir}/{timestamp}_{base_name}.log"
    
    # Configurar logging básico: nivel INFO, formato con fecha-hora
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()  # También muestra en consola
        ]
    )
    # Log de inicio
    logging.info(f"Iniciando pipeline para input: {input_file}")
    return log_filename



from __future__ import annotations

from pathlib import Path

from src.pipeline.context import PipelineContext
from src.pipeline.runner import PipelineRunner
from src.phases.phase0-ingest import IngestPhase
from src.phases.phase1-quality import QualityPhase
from src.phases.phase2-semantic import SemanticPhase
from src.phases.phase3-geospatial import GeospatialPhase
from src.phases.phase4-decision import DecisionPhase
from src.phases.phase5-report import ReportPhase


def build_runner() -> PipelineRunner:
    return PipelineRunner([
        IngestPhase(),
        QualityPhase(),
        SemanticPhase(),
        GeospatialPhase(),
        DecisionPhase(),
        ReportPhase(),
    ])


def main() -> None:
    input_path = Path("input")
    output_dir = Path("output")

    ctx = PipelineContext(
        input_path=input_path,
        output_dir=output_dir,
        unattended=False,
        on_missing_dep="install",
    )

    runner = build_runner()
    runner.run(ctx)


if __name__ == "__main__":
    main()
    args = parse_args()  # asumiendo que existe
    log_file = setup_logging(args.input)
    try:
        pipeline = PipelineRunner(...)
        pipeline.run()
    except Exception as e:
        logging.critical("Error fatal no capturado", exc_info=True)
        sys.exit(1)
    finally:
        logging.info(f"Pipeline finalizado. Log guardado en {log_file}")
        
python        
