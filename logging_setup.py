"""
logging_setup.py

Configuración centralizada de logging para todo el pipeline. Todas las
fases usan `logging.getLogger(__name__)`, así que basta con configurar
el logger raíz una sola vez, aquí, para que los mensajes de cualquier
fase terminen en el mismo archivo bajo logs/.

No se llama nunca desde dentro de una fase individual: eso duplicaría
handlers y volvería a escribir cada línea N veces.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(input_name: str | None = None, log_dir: str | Path = "logs") -> Path:
    """
    Crea (si no existe) la carpeta de logs y configura el logger raíz con
    dos salidas: archivo (logs/<timestamp>_<input>.log) y consola.

    Devuelve la ruta del archivo de log creado, para poder informarla al
    usuario al finalizar la ejecución.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = Path(input_name).stem if input_name else "no_input"
    log_path = log_dir / f"{timestamp}_{base_name}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Evita duplicar handlers si setup_logging se llama más de una vez
    # (por ejemplo, en tests) dentro del mismo proceso.
    root_logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logging.info("Logging inicializado. Archivo de log: %s", log_path)
    return log_path
