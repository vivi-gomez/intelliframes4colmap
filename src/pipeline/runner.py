"""
runner.py

Orquesta la ejecución de las fases en orden, respetando qué fases pidió el
usuario ejecutar/saltar, y preguntando de forma interactiva cuando aplique.

El control de errores del pipeline vive aquí, en un único lugar: cada fase
se ejecuta dentro de un try/except que registra en el log (archivo bajo
logs/, vía logging_setup.setup_logging) qué pasó, y decide si el pipeline
debe abortar o continuar según si la fase es opcional.
"""
from __future__ import annotations

import logging

from .context import PipelineContext
from .phase import Phase
from .tool_check import ToolStatus

logger = logging.getLogger(__name__)


def _ask_policy(phase_name: str, missing: list[ToolStatus]) -> str:
    names = ", ".join(t.name for t in missing)
    print(f"\nA la fase '{phase_name}' le faltan: {names}")
    while True:
        choice = input("¿Instalar (i) / saltar fase (s) / cancelar todo (c)? [i/s/c]: ").strip().lower()
        if choice in ("i", "install"):
            return "install"
        if choice in ("s", "skip"):
            return "skip"
        if choice in ("c", "cancel"):
            return "fail"
        print("Respuesta no válida.")


def _ask_run_phase(phase_name: str) -> bool:
    choice = input(f"\n¿Ejecutar la fase '{phase_name}'? [S/n]: ").strip().lower()
    return choice in ("", "s", "si", "sí", "y", "yes")


class PipelineRunner:
    def __init__(self, phases: list[Phase]):
        self.phases = phases

    def run(
        self,
        ctx: PipelineContext,
        only_phases: set[str] | None = None,
        skip_phases: set[str] | None = None,
    ) -> None:
        ctx.setup_dirs()
        logger.info("Iniciando ejecución del pipeline (%d fases)", len(self.phases))

        for phase in self.phases:
            if only_phases and phase.name not in only_phases:
                ctx.skipped_phases.append(phase.name)
                continue
            if skip_phases and phase.name in skip_phases:
                print(f"\n== Fase '{phase.name}' saltada por --skip-phases ==")
                logger.info("Fase '%s' saltada por --skip-phases", phase.name)
                ctx.skipped_phases.append(phase.name)
                continue

            print(f"\n== Fase: {phase.name} ==")

            if not ctx.unattended and only_phases is None:
                # En modo interactivo (y si el usuario no fijó explícitamente
                # qué fases quiere), se le da la opción de saltarla.
                if not _ask_run_phase(phase.name):
                    logger.info("Fase '%s' saltada por decisión del usuario", phase.name)
                    ctx.skipped_phases.append(phase.name)
                    continue

            ask_fn = None if ctx.unattended else _ask_policy

            logger.info("--- Ejecutando fase: %s ---", phase.name)
            try:
                phase.resolve_and_run(ctx, ask_fn=ask_fn)
                logger.info("Fase '%s' completada exitosamente", phase.name)
            except RuntimeError as e:
                logger.error("Error en fase '%s': %s", phase.name, e, exc_info=True)
                print(f"ERROR en fase '{phase.name}': {e}")
                if not phase.optional:
                    logger.critical(
                        "Fase '%s' no es opcional. Abortando pipeline.", phase.name
                    )
                    print("Esta fase no es opcional. Abortando pipeline.")
                    ctx.write_manifest()
                    raise
            except Exception as e:
                # Cualquier excepción inesperada (no RuntimeError) también
                # queda registrada antes de decidir si abortar.
                logger.error(
                    "Error inesperado en fase '%s': %s", phase.name, e, exc_info=True
                )
                if not phase.optional:
                    ctx.write_manifest()
                    raise
                print(f"[{phase.name}] Error inesperado, fase opcional: se continúa. ({e})")
                ctx.skipped_phases.append(phase.name)

        manifest_path = ctx.write_manifest()
        logger.info("Pipeline finalizado. Manifest: %s", manifest_path)
        print(f"\nPipeline finalizado. Manifest: {manifest_path}")
