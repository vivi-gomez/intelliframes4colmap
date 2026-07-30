"""
phase.py

Contrato común para todas las fases del pipeline.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from .context import PipelineContext
from .installer import resolve_missing
from .tool_check import DependencyReport, ToolStatus


class Phase(ABC):
    name: str = "unnamed_phase"
    optional: bool = False  # si es True, el pipeline puede seguir aunque falle o se salte

    @abstractmethod
    def check_dependencies(self) -> DependencyReport:
        """Devuelve el estado de las herramientas que esta fase necesita."""
        raise NotImplementedError

    @abstractmethod
    def run(self, ctx: PipelineContext) -> None:
        """Ejecuta la fase, leyendo/escribiendo en el contexto compartido."""
        raise NotImplementedError

    def resolve_and_run(self, ctx: PipelineContext, ask_fn=None) -> bool:
        """
        Comprueba dependencias, las resuelve según la política, y ejecuta la fase.
        ask_fn: función opcional (usada en modo interactivo) que pregunta al
                usuario qué hacer con cada dependencia faltante.
        Devuelve True si la fase se ejecutó, False si se saltó.
        """
        report = self.check_dependencies()
        print(report.summary())

        if report.satisfied:
            self.run(ctx)
            ctx.executed_phases.append(self.name)
            return True

        policy = ctx.on_missing_dep
        if not ctx.unattended and ask_fn is not None:
            policy = ask_fn(self.name, report.missing)

        try:
            results = resolve_missing(report.missing, policy)
        except RuntimeError as e:
            print(f"[{self.name}] {e}")
            raise

        ctx.dependency_log[self.name] = {t.name: ok for t, ok in zip(report.missing, results.values())}

        still_missing = [t for t in report.missing if not results.get(t.name, False)]
        if still_missing:
            if self.optional:
                print(f"[{self.name}] Se salta la fase (opcional): faltan {[t.name for t in still_missing]}")
                ctx.skipped_phases.append(self.name)
                return False
            else:
                raise RuntimeError(
                    f"[{self.name}] Faltan dependencias obligatorias: {[t.name for t in still_missing]}"
                )

        self.run(ctx)
        ctx.executed_phases.append(self.name)
        return True
