"""
Fase 0 — Ingesta

Extrae metadata real con ffprobe y, si la entrada es un video, extrae los
frames con ffmpeg. Si la entrada ya es una carpeta de imágenes, las usa
directamente. Nada aquí está simulado.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ..pipeline.context import PipelineContext
from ..pipeline.phase import Phase
from ..pipeline.tool_check import DependencyReport, check_binary

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".mts", ".m4v"}


class IngestPhase(Phase):
    name = "ingest"
    optional = False

    def check_dependencies(self) -> DependencyReport:
        return DependencyReport(
            phase_name=self.name,
            checks=[check_binary("ffmpeg"), check_binary("ffprobe")],
        )

    def run(self, ctx: PipelineContext) -> None:
        input_path = ctx.input_path

        if input_path.is_dir():
            self._ingest_image_folder(ctx, input_path)
        elif input_path.suffix.lower() in VIDEO_EXTS:
            self._ingest_video(ctx, input_path)
        elif input_path.suffix.lower() in IMAGE_EXTS:
            self._ingest_image_folder(ctx, input_path.parent)
        else:
            raise RuntimeError(f"Tipo de entrada no reconocido: {input_path}")

    def _ingest_video(self, ctx: PipelineContext, video_path: Path) -> None:
        ctx.metadata = _probe_video(video_path)
        ctx.frames_dir.mkdir(parents=True, exist_ok=True)

        # Extrae todos los frames como PNG numerados. Para videos muy largos,
        # el usuario puede limitar el FPS de muestreo con --sample-fps.
        pattern = str(ctx.frames_dir / "frame_%06d.png")
        sample_fps = ctx.metadata.get("sample_fps")
        cmd = ["ffmpeg", "-y", "-i", str(video_path)]
        if sample_fps:
            cmd += ["-vf", f"fps={sample_fps}"]
        cmd += [pattern]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg falló extrayendo frames:\n{result.stderr[-1000:]}")

        ctx.frame_list = sorted(str(p) for p in ctx.frames_dir.glob("frame_*.png"))
        ctx.metadata["frames_extracted"] = len(ctx.frame_list)

    def _ingest_image_folder(self, ctx: PipelineContext, folder: Path) -> None:
        frames = sorted(
            p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS
        )
        if not frames:
            raise RuntimeError(f"No se encontraron imágenes en {folder}")
        ctx.frame_list = [str(p) for p in frames]
        ctx.frames_dir = folder
        ctx.metadata = {
            "source_type": "image_folder",
            "frames_total": len(frames),
        }


def _probe_video(video_path: Path) -> dict:
    """Metadata real con ffprobe (resolución, fps, duración, nº frames)."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
        "-of", "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe falló:\n{result.stderr[-1000:]}")

    data = json.loads(result.stdout)
    stream = data["streams"][0]

    fps_str = stream.get("r_frame_rate", "0/1")
    num, den = (int(x) for x in fps_str.split("/"))
    fps = round(num / den, 2) if den else 0

    return {
        "source_type": "video",
        "resolution": f"{stream.get('width')}x{stream.get('height')}",
        "fps": fps,
        "duration_seconds": float(stream.get("duration", 0) or 0),
        "frames_total": int(stream.get("nb_frames", 0) or 0),
    }
