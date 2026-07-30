"""
runner.py

Orquesta la ejecución de las fases en orden, respetando qué fases pidió el
usuario ejecutar/saltar, y preguntando de forma interactiva cuando aplique.
"""
from __future__ import annotations

from .context import PipelineContext
from .phase import Phase
from .tool_check import ToolStatus


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

def run(self):
    logging.info("Iniciando ejecución del pipeline")
    for phase in self.phases:
        phase_name = phase.__class__.__name__
        logging.info(f"--- Ejecutando fase: {phase_name} ---")
        try:
            phase.execute(self.context)
            logging.info(f"Fase {phase_name} completada exitosamente")
        except Exception as e:
            logging.error(f"Error en fase {phase_name}: {str(e)}", exc_info=True)
            # Dependiendo de la estrategia, podrías detener o continuar
            raise  # o break, o manejar según política
    logging.info("Pipeline finalizado")
    
    
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

        for phase in self.phases:
            if only_phases and phase.name not in only_phases:
                ctx.skipped_phases.append(phase.name)
                continue
            if skip_phases and phase.name in skip_phases:
                print(f"\n== Fase '{phase.name}' saltada por --skip-phases ==")
                ctx.skipped_phases.append(phase.name)
                continue

            print(f"\n== Fase: {phase.name} ==")

            if not ctx.unattended and only_phases is None:
                # En modo interactivo (y si el usuario no fijó explícitamente
                # qué fases quiere), se le da la opción de saltarla.
                if not _ask_run_phase(phase.name):
                    ctx.skipped_phases.append(phase.name)
                    continue

            ask_fn = None if ctx.unattended else _ask_policy
            try:
                phase.resolve_and_run(ctx, ask_fn=ask_fn)
            except RuntimeError as e:
                print(f"ERROR en fase '{phase.name}': {e}")
                if not phase.optional:
                    print("Esta fase no es opcional. Abortando pipeline.")
                    ctx.write_manifest()
                    raise

        manifest_path = ctx.write_manifest()
        print(f"\nPipeline finalizado. Manifest: {manifest_path}")
