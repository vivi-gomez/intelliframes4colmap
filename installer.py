"""
installer.py

Instalación granular: SOLO se llama para una herramienta concreta que ya se
comprobó que falta (ver tool_check.py). Nunca instala "todo de golpe".
"""
from __future__ import annotations

import platform
import subprocess
import sys

from .tool_check import ToolStatus

# Nombre del paquete del sistema por herramienta y gestor de paquetes.
_SYSTEM_PACKAGE_MAP = {
    "ffmpeg": {"apt": "ffmpeg", "dnf": "ffmpeg", "brew": "ffmpeg"},
    "ffprobe": {"apt": "ffmpeg", "dnf": "ffmpeg", "brew": "ffmpeg"},  # viene con ffmpeg
    "exiftool": {"apt": "libimage-exiftool-perl", "dnf": "perl-Image-ExifTool", "brew": "exiftool"},
    "colmap": {"apt": "colmap", "dnf": "colmap", "brew": "colmap"},
}


def _run(cmd: list[str]) -> bool:
    print(f">> Ejecutando: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"   Error: {result.stderr.strip()[:500]}")
        return False
    return True


def install_binary(tool_name: str) -> bool:
    """Instala un único binario del sistema, detectando el gestor de paquetes disponible."""
    system = platform.system()
    pkg_map = _SYSTEM_PACKAGE_MAP.get(tool_name)
    if pkg_map is None:
        print(f"No sé cómo instalar '{tool_name}' automáticamente. Instálalo manualmente.")
        return False

    if system == "Linux":
        if shutil_which("apt-get"):
            _run(["sudo", "apt-get", "update"])
            return _run(["sudo", "apt-get", "install", "-y", pkg_map["apt"]])
        if shutil_which("dnf"):
            return _run(["sudo", "dnf", "install", "-y", pkg_map["dnf"]])
        print("No se detectó apt ni dnf. Instala manualmente:", tool_name)
        return False

    if system == "Darwin":
        if not shutil_which("brew"):
            print("Homebrew no está instalado. Instálalo desde https://brew.sh antes de continuar.")
            return False
        return _run(["brew", "install", pkg_map["brew"]])

    if system == "Windows":
        print(f"En Windows, instala '{tool_name}' manualmente:")
        hints = {
            "ffmpeg": "https://ffmpeg.org/download.html",
            "exiftool": "https://exiftool.org/",
            "colmap": "https://colmap.github.io/install.html",
        }
        print("  ", hints.get(tool_name, "(sin enlace conocido)"))
        return False

    return False


def install_python_package(pip_name: str) -> bool:
    """Instala un único paquete Python con pip."""
    return _run([sys.executable, "-m", "pip", "install", pip_name])


def shutil_which(name: str) -> str | None:
    import shutil
    return shutil.which(name)


def resolve_missing(missing: list[ToolStatus], policy: str) -> dict[str, bool]:
    """
    Aplica la política de resolución a una lista de dependencias que faltan.
    policy: "install" | "skip" | "fail"
    Devuelve {nombre_herramienta: instalado_con_exito}
    """
    results: dict[str, bool] = {}
    for tool in missing:
        if policy == "fail":
            raise RuntimeError(f"Falta la dependencia obligatoria '{tool.name}' y la política es 'fail'.")
        if policy == "skip":
            results[tool.name] = False
            continue
        # policy == "install"
        if tool.kind == "binary":
            results[tool.name] = install_binary(tool.name)
        elif tool.kind == "python_package":
            results[tool.name] = install_python_package(tool.name)
        else:
            print(f"'{tool.name}' es un fichero de modelo y debe descargarse aparte (ver documentación).")
            results[tool.name] = False
    return results
