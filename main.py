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
