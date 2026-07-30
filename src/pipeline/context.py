"""
context.py

Objeto compartido que las fases van leyendo/escribiendo a medida que se
ejecutan. Si una fase se salta, simplemente no rellena su parte y las fases
siguientes deben saber trabajar con datos ausentes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class PipelineContext:
    input_path: Path
    output_dir: Path
    unattended: bool = False
    on_missing_dep: str = "install"      # "install" | "skip" | "fail"

    # Rutas de trabajo (se crean bajo demanda)
    frames_dir: Path = None
    frames_selected_dir: Path = None
    frames_rejected_dir: Path = None
    thumbnails_dir: Path = None
    masks_dir: Path = None
    metrics_dir: Path = None
    colmap_dir: Path = None

    # Datos que las fases van rellenando
    metadata: dict[str, Any] = field(default_factory=dict)
    frame_list: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)          # sharpness, motion, texture, exposure...
    semantic: dict[str, Any] = field(default_factory=dict)         # segmentación
    geospatial: dict[str, Any] = field(default_factory=dict)       # GNSS/IMU
    decision: dict[str, Any] = field(default_factory=dict)         # config final agnóstica

    # Auditoría
    executed_phases: list[str] = field(default_factory=list)
    skipped_phases: list[str] = field(default_factory=list)
    dependency_log: dict[str, Any] = field(default_factory=dict)

    def setup_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir = self.output_dir / "frames"
        self.frames_selected_dir = self.output_dir / "frames_selected"
        self.frames_rejected_dir = self.output_dir / "frames_rejected"
        self.thumbnails_dir = self.output_dir / "thumbnails"
        self.masks_dir = self.output_dir / "masks"
        self.metrics_dir = self.output_dir / "metrics"
        self.colmap_dir = self.output_dir / "colmap"
        for d in (
            self.frames_selected_dir,
            self.frames_rejected_dir,
            self.thumbnails_dir,
            self.metrics_dir,
            self.colmap_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def write_manifest(self) -> Path:
        manifest = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "input_path": str(self.input_path),
            "output_dir": str(self.output_dir),
            "executed_phases": self.executed_phases,
            "skipped_phases": self.skipped_phases,
            "dependency_log": self.dependency_log,
        }
        manifest_path = self.output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return manifest_path
