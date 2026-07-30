"""
tool_check.py

Comprobación de dependencias ANTES de instalar nada. Cada fase declara qué
necesita y este módulo responde "ya está" o "falta", sin efectos secundarios.
"""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ToolStatus:
    name: str
    kind: str          # "binary" | "python_package" | "model_file"
    found: bool
    version: str | None = None
    path: str | None = None
    install_hint: str = ""


def check_binary(name: str, version_flag: str = "-version") -> ToolStatus:
    """Comprueba si un ejecutable existe en el PATH (ffmpeg, exiftool, colmap...)."""
    found_path = shutil.which(name)
    version = None
    if found_path:
        try:
            result = subprocess.run(
                [name, version_flag], capture_output=True, text=True, timeout=5
            )
            first_line = (result.stdout or result.stderr).splitlines()
            version = first_line[0] if first_line else None
        except Exception:
            version = None
    return ToolStatus(
        name=name,
        kind="binary",
        found=found_path is not None,
        version=version,
        path=found_path,
    )


def check_python_package(module_name: str, pip_name: str | None = None) -> ToolStatus:
    """Comprueba si un paquete Python ya está instalado, sin importarlo (rápido)."""
    spec = importlib.util.find_spec(module_name)
    version = None
    if spec is not None:
        try:
            import importlib.metadata as md
            version = md.version(pip_name or module_name)
        except Exception:
            version = None
    return ToolStatus(
        name=pip_name or module_name,
        kind="python_package",
        found=spec is not None,
        version=version,
        install_hint=f"pip install {pip_name or module_name}",
    )


def check_model_file(label: str, path: Path) -> ToolStatus:
    """Comprueba si un checkpoint de modelo (SAM, YOLO...) ya está descargado en disco."""
    exists = path.exists() and path.is_file()
    return ToolStatus(
        name=label,
        kind="model_file",
        found=exists,
        path=str(path) if exists else None,
    )


@dataclass
class DependencyReport:
    phase_name: str
    checks: list[ToolStatus] = field(default_factory=list)

    @property
    def missing(self) -> list[ToolStatus]:
        return [c for c in self.checks if not c.found]

    @property
    def satisfied(self) -> bool:
        return len(self.missing) == 0

    def summary(self) -> str:
        lines = [f"Dependencias de la fase '{self.phase_name}':"]
        for c in self.checks:
            mark = "OK" if c.found else "FALTA"
            extra = f" ({c.version})" if c.version else ""
            lines.append(f"  [{mark}] {c.name}{extra}")
        return "\n".join(lines)
