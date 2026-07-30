"""
    Genera un informe HTML autocontenido y legible.

    El objetivo es ser simple, portable y fácil de inspeccionar sin
    dependencias frontend adicionales.
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any, Dict, List


def build_analysis_payload(ctx) -> Dict[str, Any]:
    """
    Construye un payload consolidado con los resultados del pipeline.

    Este JSON está pensado como salida canónica para inspección,
    exportación o consumo por herramientas externas.
    """
    return {
        "input": {
            "input_path": getattr(ctx, "input_path", None),
            "workspace_dir": getattr(ctx, "workspace_dir", None),
            "total_frames": len(list(getattr(ctx, "frame_list", []) or [])),
        },
        "quality": getattr(ctx, "metrics", {}).get("quality", {}),
        "semantic": {
            "metrics": getattr(ctx, "metrics", {}).get("semantic", {}),
            "data": getattr(ctx, "semantic", {}),
        },
        "geospatial": {
            "metrics": getattr(ctx, "metrics", {}).get("geospatial", {}),
            "data": getattr(ctx, "geospatial", {}),
        },
        "decision": {
            "metrics": getattr(ctx, "metrics", {}).get("decision", {}),
            "data": getattr(ctx, "decision", {}),
        },
        "dependency_log": getattr(ctx, "dependency_log", {}),
    }


def build_html_report(ctx, analysis: Dict[str, Any]) -> str:

    decision_summary = analysis.get("decision", {}).get("data", {}).get("summary", {}) or {}
    semantic_summary = analysis.get("semantic", {}).get("data", {}).get("summary", {}) or {}
    geospatial_summary = analysis.get("geospatial", {}).get("data", {}).get("summary", {}) or {}
    quality_metrics = analysis.get("quality", {}) or {}

    selected_frames = decision_summary.get("selected_frames", 0)
    total_frames = decision_summary.get("total_frames", analysis.get("input", {}).get("total_frames", 0))
    readiness = decision_summary.get("dataset_readiness", "UNKNOWN")
    avg_risk = semantic_summary.get("avg_risk_score", 0.0)
    avg_usable = semantic_summary.get("avg_usable_area_pct", 0.0)
    gps_available = geospatial_summary.get("gps_available", False)
    coverage = geospatial_summary.get("coverage", "none")

    top_rejections = _summarize_rejections(
        analysis.get("decision", {}).get("data", {}).get("frames", []) or []
    )

    dependency_log_pretty = json.dumps(
        analysis.get("dependency_log", {}),
        indent=2,
        ensure_ascii=False,
    )

    analysis_pretty = json.dumps(analysis, indent=2, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>IntelliFrames4COLMAP - Report</title>
  <style>
    body {{
      font-family: Arial, Helvetica, sans-serif;
      margin: 32px;
      color: #222;
      background: #fafafa;
    }}
    h1, h2, h3 {{
      color: #111;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin: 20px 0 28px;
    }}
    .card {{
      background: white;
      border: 1px solid #ddd;
      border-radius: 10px;
      padding: 16px;
      box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }}
    .metric {{
      font-size: 1.6rem;
      font-weight: bold;
      margin-top: 8px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      background: white;
      margin: 16px 0 28px;
    }}
    th, td {{
      border: 1px solid #ddd;
      text-align: left;
      padding: 10px;
      vertical-align: top;
    }}
    th {{
      background: #f0f0f0;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      background: #111;
      color: #f5f5f5;
      padding: 16px;
      border-radius: 8px;
      overflow: auto;
    }}
    .muted {{
      color: #666;
    }}
  </style>
</head>
<body>
  <h1>IntelliFrames4COLMAP - Report</h1>
  <p class="muted">
    Resumen consolidado del análisis de frames, señales semánticas,
    telemetría geoespacial y recomendación para COLMAP.
  </p>

  <div class="grid">
    <div class="card">
      <h3>Readiness</h3>
      <div class="metric">{escape(str(readiness))}</div>
    </div>
    <div class="card">
      <h3>Frames recomendados</h3>
      <div class="metric">{selected_frames} / {total_frames}</div>
    </div>
    <div class="card">
      <h3>Riesgo medio</h3>
      <div class="metric">{escape(str(avg_risk))}</div>
    </div>
    <div class="card">
      <h3>Área utilizable media</h3>
      <div class="metric">{escape(str(avg_usable))}%</div>
    </div>
    <div class="card">
      <h3>GPS disponible</h3>
      <div class="metric">{escape(str(gps_available))}</div>
    </div>
    <div class="card">
      <h3>Cobertura</h3>
      <div class="metric">{escape(str(coverage))}</div>
    </div>
  </div>

  <h2>Resumen por fase</h2>
  <table>
    <thead>
      <tr>
        <th>Fase</th>
        <th>Estado</th>
        <th>Datos clave</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Quality</td>
        <td>{escape(str(quality_metrics.get("status", "unknown")))}</td>
        <td>{escape(json.dumps(quality_metrics, ensure_ascii=False))}</td>
      </tr>
      <tr>
        <td>Semantic</td>
        <td>{escape(str(analysis.get("semantic", {}).get("metrics", {}).get("status", "unknown")))}</td>
        <td>{escape(json.dumps(semantic_summary, ensure_ascii=False))}</td>
      </tr>
      <tr>
        <td>Geospatial</td>
        <td>{escape(str(analysis.get("geospatial", {}).get("metrics", {}).get("status", "unknown")))}</td>
        <td>{escape(json.dumps(geospatial_summary, ensure_ascii=False))}</td>
      </tr>
      <tr>
        <td>Decision</td>
        <td>{escape(str(analysis.get("decision", {}).get("metrics", {}).get("status", "unknown")))}</td>
        <td>{escape(json.dumps(decision_summary, ensure_ascii=False))}</td>
      </tr>
    </tbody>
  </table>

  <h2>Motivos de rechazo más frecuentes</h2>
  <table>
    <thead>
      <tr>
        <th>Motivo</th>
        <th>Conteo</th>
      </tr>
    </thead>
    <tbody>
      {_render_rejection_rows(top_rejections)}
    </tbody>
  </table>

  <h2>Dependency log</h2>
  <pre>{escape(dependency_log_pretty)}</pre>

  <h2>Analysis JSON</h2>
  <pre>{escape(analysis_pretty)}</pre>
</body>
</html>
"""


def save_analysis_json(path: str | Path, analysis: Dict[str, Any]) -> None:
    """
    Guarda el análisis consolidado como JSON.
    """
    path = Path(path)
    path.write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_html_report(path: str | Path, html: str) -> None:
    """
    Guarda el informe HTML en disco.
    """
    path = Path(path)
    path.write_text(html, encoding="utf-8")


def _summarize_rejections(frame_rows: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Cuenta los motivos de rechazo devueltos por la fase de decisión.
    """
    counts: Dict[str, int] = {}

    for row in frame_rows:
        for reason in row.get("reject_reasons", []) or []:
            counts[reason] = counts.get(reason, 0) + 1

    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))


def _render_rejection_rows(rejections: Dict[str, int]) -> str:
    """
    Renderiza filas HTML para la tabla de rechazos.
    """
    if not rejections:
        return "<tr><td colspan='2'>No hay rechazos registrados.</td></tr>"

    rows = []
    for reason, count in rejections.items():
        rows.append(
            f"<tr><td>{escape(str(reason))}</td><td>{escape(str(count))}</td></tr>"
        )
    return "\n".join(rows)
