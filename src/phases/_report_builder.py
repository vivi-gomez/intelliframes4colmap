"""
_report_builder.py

Genera analysis.json (salida canónica para máquinas) y report.html (para
humanos).

report.html se rediseñó por completo: la versión anterior era, en esencia,
el JSON completo volcado dentro de un <pre>, con tablas de datos crudos y
sin ningún apoyo visual. Para un vistazo rápido era engorroso.

Este report:
- usa gráficos SVG simples e inline (sin dependencias externas: sigue
  siendo un único fichero HTML autocontenido y portable),
- explica en lenguaje natural qué significa cada número y por qué se
  descarta un frame,
- dice explícitamente qué se enmascaró y por qué (o si no se enmascaró
  nada), en vez de forzar al usuario a leer sky_pct/water_pct/etc.,
- deja el JSON técnico disponible pero colapsado (<details>), no como
  contenido principal.
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any, Dict, List

_REASON_LABELS_ES = {
    "low_sharpness": "Imagen borrosa o con motion blur",
    "heavy_dynamic_occlusion": "Un objeto en movimiento (persona/vehículo/animal) tapaba gran parte del encuadre",
    "low_final_score": "Puntuación de calidad global baja",
}

_CATEGORY_LABELS_ES = {
    "sky": "cielo",
    "water": "agua",
    "reflection": "reflejos / sobreexposición",
    "ground": "suelo",
    "vegetation": "vegetación",
    "uniform_wall": "superficies lisas / fondo desenfocado",
    "person": "personas",
    "vehicle": "vehículos",
    "animal": "animales",
    "bird": "aves",
}

_READINESS_LABELS_ES = {
    "HIGH": ("Alta", "El dataset tiene buena pinta: nitidez y cobertura suficientes para una reconstrucción estable."),
    "MEDIUM_HIGH": ("Media-alta", "Buen punto de partida; con GPS/telemetría la reconstrucción sería aún más robusta."),
    "MEDIUM": ("Media", "Debería reconstruir, pero conviene revisar los frames rechazados y las zonas enmascaradas."),
    "LOW": ("Baja", "Alto riesgo de reconstrucción inestable o fallida: revisa el material de origen antes de lanzar COLMAP."),
    "UNKNOWN": ("Desconocida", "No hay datos suficientes para estimar la preparación del dataset."),
}


def build_analysis_payload(ctx) -> Dict[str, Any]:
    """
    Construye un payload consolidado con los resultados del pipeline.
    Salida canónica para inspección, exportación o consumo por herramientas
    externas (no pensada para lectura humana directa: para eso está el
    report.html).
    """
    return {
        "input": {
            "input_path": str(getattr(ctx, "input_path", "") or ""),
            "output_dir": str(getattr(ctx, "output_dir", "") or ""),
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
    decision_data = analysis.get("decision", {}).get("data", {}) or {}
    decision_summary = decision_data.get("summary", {}) or {}
    semantic_summary = analysis.get("semantic", {}).get("data", {}).get("summary", {}) or {}
    geospatial_summary = analysis.get("geospatial", {}).get("data", {}).get("summary", {}) or {}
    quality_metrics = analysis.get("quality", {}) or {}
    frame_rows = decision_data.get("frames", []) or []

    total_frames = decision_summary.get("total_frames", analysis.get("input", {}).get("total_frames", 0))
    selected_frames = decision_summary.get("selected_frames", 0)
    rejected_frames = decision_summary.get("rejected_frames", max(0, total_frames - selected_frames))
    readiness = decision_summary.get("dataset_readiness", "UNKNOWN")
    readiness_label, readiness_explanation = _READINESS_LABELS_ES.get(readiness, _READINESS_LABELS_ES["UNKNOWN"])
    avg_usable = semantic_summary.get("avg_usable_area_pct", 0.0)
    gps_available = geospatial_summary.get("gps_available", False)
    coverage = geospatial_summary.get("coverage", "none")
    avg_texture = semantic_summary.get("avg_texture_score", 0.0)
    avg_exposure = semantic_summary.get("avg_exposure_score", 0.0)

    rejection_counts = _summarize_rejections(frame_rows)
    masking_section = _build_masking_section(semantic_summary)

    donut_svg = _donut_chart(selected_frames, rejected_frames)
    rejection_chart_svg = _bar_chart(
        [(_REASON_LABELS_ES.get(reason, reason), count) for reason, count in rejection_counts.items()],
        color="#c0392b",
    )
    quality_bar_svg = _bar_chart(
        [("Nitidez media", _clamp01_100(quality_metrics.get("avg_sharpness", 0.0), scale=60.0)),
         ("Textura media", _clamp01_100(avg_texture, scale=80.0)),
         ("Exposición media", avg_exposure),
         ("Área útil media", avg_usable)],
        color="#2e7d32",
        max_value=100.0,
        show_raw_values=[quality_metrics.get("avg_sharpness", 0.0), avg_texture, avg_exposure, avg_usable],
    )

    dependency_log_pretty = json.dumps(analysis.get("dependency_log", {}), indent=2, ensure_ascii=False)
    analysis_pretty = json.dumps(analysis, indent=2, ensure_ascii=False)

    selection_ratio = (selected_frames / total_frames * 100.0) if total_frames else 0.0

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>IntelliFrames4COLMAP - Informe</title>
  <style>
    body {{
      font-family: -apple-system, Segoe UI, Roboto, Arial, Helvetica, sans-serif;
      margin: 0; padding: 32px; color: #1a1a1a; background: #f4f5f7;
      max-width: 980px; margin-left: auto; margin-right: auto;
    }}
    h1 {{ margin-bottom: 4px; }}
    h2 {{ margin-top: 40px; border-bottom: 2px solid #e0e0e0; padding-bottom: 6px; }}
    .subtitle {{ color: #666; margin-top: 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin: 20px 0 28px; }}
    .card {{ background: white; border: 1px solid #e2e2e2; border-radius: 12px; padding: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
    .card h3 {{ margin: 0 0 6px; font-size: 0.85rem; color: #666; text-transform: uppercase; letter-spacing: 0.04em; }}
    .metric {{ font-size: 1.9rem; font-weight: 700; }}
    .metric-sub {{ color: #666; font-size: 0.9rem; margin-top: 4px; }}
    .readiness-HIGH, .readiness-MEDIUM_HIGH {{ color: #1b7d3b; }}
    .readiness-MEDIUM {{ color: #b8860b; }}
    .readiness-LOW {{ color: #c0392b; }}
    .readiness-UNKNOWN {{ color: #666; }}
    .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; align-items: start; }}
    .chart-card {{ background: white; border: 1px solid #e2e2e2; border-radius: 12px; padding: 20px; }}
    .chart-card h3 {{ margin-top: 0; }}
    .explain {{ background: #eef4ff; border-left: 4px solid #3b6fd6; padding: 14px 18px; border-radius: 6px; margin: 16px 0; line-height: 1.5; }}
    .explain p {{ margin: 6px 0; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 12px 0 24px; border-radius: 8px; overflow: hidden; }}
    th, td {{ border-bottom: 1px solid #eee; text-align: left; padding: 10px 12px; vertical-align: top; }}
    th {{ background: #fafafa; font-size: 0.85rem; text-transform: uppercase; color: #666; }}
    .glossary dt {{ font-weight: 600; margin-top: 10px; }}
    .glossary dd {{ margin: 2px 0 0; color: #333; }}
    details summary {{ cursor: pointer; font-weight: 600; padding: 10px 0; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #111; color: #f5f5f5; padding: 16px; border-radius: 8px; overflow: auto; font-size: 0.82rem; }}
    .muted {{ color: #666; }}
    .badge {{ display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.8rem; font-weight: 600; }}
    .badge-yes {{ background: #e2f6e8; color: #1b7d3b; }}
    .badge-no {{ background: #f6e2e2; color: #8a2a2a; }}
  </style>
</head>
<body>
  <h1>IntelliFrames4COLMAP</h1>
  <p class="subtitle">Análisis de: <strong>{escape(str(analysis.get("input", {}).get("input_path", "")))}</strong></p>

  <div class="grid">
    <div class="card">
      <h3>Preparación del dataset</h3>
      <div class="metric readiness-{escape(str(readiness))}">{escape(readiness_label)}</div>
      <div class="metric-sub">{escape(readiness_explanation)}</div>
    </div>
    <div class="card">
      <h3>Frames recomendados</h3>
      <div class="metric">{selected_frames} / {total_frames}</div>
      <div class="metric-sub">{selection_ratio:.0f}% de la secuencia se recomienda para COLMAP</div>
    </div>
    <div class="card">
      <h3>Área útil media tras máscaras</h3>
      <div class="metric">{_fmt_num(avg_usable)}%</div>
      <div class="metric-sub">Porcentaje medio de cada imagen que SÍ se procesa (blanco en la máscara)</div>
    </div>
    <div class="card">
      <h3>GPS / telemetría</h3>
      <div class="metric"><span class="badge {'badge-yes' if gps_available else 'badge-no'}">{'Disponible' if gps_available else 'No disponible'}</span></div>
      <div class="metric-sub">Cobertura estimada: {escape(str(coverage))}</div>
    </div>
  </div>

  <h2>De un vistazo: selección de frames</h2>
  <div class="charts">
    <div class="chart-card">
      <h3>Aceptados vs. descartados</h3>
      {donut_svg}
    </div>
    <div class="chart-card">
      <h3>Motivos de descarte</h3>
      {rejection_chart_svg if rejection_counts else '<p class="muted">No se ha descartado ningún frame.</p>'}
    </div>
  </div>

  <div class="explain">
    <p><strong>¿Por qué se descarta un frame?</strong> Un frame se descarta solo por dos motivos: (1) salió borroso
    o con motion blur y no aporta información fiable para el matching, o (2) un objeto en movimiento ajeno a la
    escena (una persona, un vehículo, un animal) tapaba una parte tan grande del encuadre que, incluso enmascarándolo,
    apenas quedaba contenido útil. Elementos como el cielo, el agua o un fondo desenfocado (típico con lentes macro)
    <strong>no descartan el frame</strong>: se gestionan recortándolos con una máscara, no eliminando la imagen entera.</p>
  </div>

  <h2>Calidad de la grabación</h2>
  <div class="chart-card">
    {quality_bar_svg}
  </div>
  <div class="explain">
    <p><strong>Nitidez media:</strong> mide si las imágenes están enfocadas o presentan motion blur. Cuanto más alta, mejor.</p>
    <p><strong>Textura media:</strong> cuánto "detalle visual" hay para que COLMAP encuentre puntos característicos.
    Paredes lisas o cielos despejados dan textura baja; ladrillo, piedra o vegetación dan textura alta.</p>
    <p><strong>Exposición media:</strong> qué tan equilibrada está la luz (ni quemada ni demasiado oscura).</p>
    <p><strong>Área útil media:</strong> el porcentaje real de cada imagen que queda disponible después de aplicar
    las máscaras (ver más abajo).</p>
  </div>

  {masking_section}

  <h2>Resumen técnico por fase</h2>
  <table>
    <thead><tr><th>Fase</th><th>Estado</th></tr></thead>
    <tbody>
      <tr><td>Calidad</td><td>{escape(str(quality_metrics.get("status", "unknown")))}</td></tr>
      <tr><td>Semántica / máscaras</td><td>{escape(str(analysis.get("semantic", {}).get("metrics", {}).get("status", "unknown")))}</td></tr>
      <tr><td>Geoespacial</td><td>{escape(str(analysis.get("geospatial", {}).get("metrics", {}).get("status", "unknown")))}</td></tr>
      <tr><td>Decisión</td><td>{escape(str(analysis.get("decision", {}).get("metrics", {}).get("status", "unknown")))}</td></tr>
    </tbody>
  </table>

  <h2>Glosario rápido</h2>
  <dl class="glossary">
    <dt>Máscara</dt><dd>Imagen en blanco y negro del mismo tamaño que el frame: blanco = COLMAP procesa ese píxel, negro = lo ignora.</dd>
    <dt>Área útil</dt><dd>Porcentaje de la imagen que queda en blanco (procesable) tras aplicar la máscara.</dd>
    <dt>Nitidez / sharpness</dt><dd>Medida de si la imagen está enfocada. Valores bajos indican desenfoque o motion blur.</dd>
    <dt>Solapamiento (overlap)</dt><dd>Cuánto contenido comparten dos frames consecutivos; hace falta suficiente para que COLMAP pueda emparejar puntos entre ellos.</dd>
    <dt>Objeto anómalo</dt><dd>En modo automático, un objeto dinámico (persona/vehículo/animal/ave) se enmascara solo si aparece de forma pasajera o se mueve de forma incoherente con el resto de la escena; si es persistente y coherente con el movimiento de cámara, se trata como parte de la escena y no se enmascara.</dd>
  </dl>

  <h2>Datos técnicos completos</h2>
  <details>
    <summary>Registro de dependencias</summary>
    <pre>{escape(dependency_log_pretty)}</pre>
  </details>
  <details>
    <summary>analysis.json completo</summary>
    <pre>{escape(analysis_pretty)}</pre>
  </details>
</body>
</html>
"""


def save_analysis_json(path: str | Path, analysis: Dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")


def save_html_report(path: str | Path, html: str) -> None:
    Path(path).write_text(html, encoding="utf-8")


def _summarize_rejections(frame_rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in frame_rows:
        for reason in row.get("reject_reasons", []) or []:
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))


def _build_masking_section(semantic_summary: Dict[str, Any]) -> str:
    mode = semantic_summary.get("mode", "none")
    static_categories = semantic_summary.get("masked_static_categories", []) or []
    frames_with_dynamic = semantic_summary.get("frames_with_dynamic_objects_masked", 0)
    anomaly_reasons = semantic_summary.get("dynamic_anomaly_reasons", []) or []

    mode_label = {
        "attended": "atendido (elegido por el usuario)",
        "automatic": "automático (decidido por el análisis de la secuencia)",
        "none": "sin análisis de máscaras",
    }.get(mode, mode)

    if static_categories:
        static_labels = ", ".join(_CATEGORY_LABELS_ES.get(c, c) for c in static_categories)
        static_text = f"Se excluyó de las máscaras: <strong>{escape(static_labels)}</strong>."
    else:
        static_text = "No se excluyó ninguna categoría estática (cielo, agua, etc.)."

    if frames_with_dynamic:
        reasons_text = "; ".join(escape(r) for r in anomaly_reasons) if anomaly_reasons else ""
        dynamic_text = (
            f"Se detectaron y enmascararon objetos dinámicos anómalos en "
            f"<strong>{frames_with_dynamic}</strong> frame(s)"
            + (f" — motivo: {reasons_text}." if reasons_text else ".")
        )
    else:
        dynamic_text = (
            "No se enmascaró ningún objeto dinámico (persona/vehículo/animal/ave): "
            "no se detectó ninguna anomalía respecto al resto de la escena, "
            "y en modo automático nunca se enmascara solo por pertenecer a esa clase."
        )

    return f"""
  <h2>Máscaras</h2>
  <div class="explain">
    <p>Modo: <strong>{escape(mode_label)}</strong></p>
    <p>{static_text}</p>
    <p>{dynamic_text}</p>
  </div>
"""


def _fmt_num(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.1f}"
    return str(value)


def _clamp01_100(value: Any, scale: float) -> float:
    """Normaliza un valor arbitrario a 0-100 usando `scale` como el valor
    que se considera "óptimo" (100%), para poder compararlo en el mismo
    gráfico de barras que porcentajes ya nativos 0-100."""
    if not isinstance(value, (int, float)) or scale <= 0:
        return 0.0
    return max(0.0, min(100.0, (float(value) / scale) * 100.0))


# ---------------------------------------------------------------------------
# Gráficos SVG minimalistas, sin dependencias externas.
# ---------------------------------------------------------------------------

def _donut_chart(selected: int, rejected: int) -> str:
    total = selected + rejected
    if total == 0:
        return '<p class="muted">Sin frames analizados.</p>'

    selected_frac = selected / total
    circumference = 2 * 3.14159265 * 70
    selected_len = circumference * selected_frac
    rest_len = circumference - selected_len

    return f"""
    <svg viewBox="0 0 220 220" width="220" height="220" role="img" aria-label="Frames seleccionados vs descartados">
      <circle cx="110" cy="110" r="70" fill="none" stroke="#e74c3c" stroke-width="28"
              stroke-dasharray="{circumference:.2f}" stroke-dashoffset="0" transform="rotate(-90 110 110)"/>
      <circle cx="110" cy="110" r="70" fill="none" stroke="#27ae60" stroke-width="28"
              stroke-dasharray="{selected_len:.2f} {rest_len:.2f}" stroke-dashoffset="0" transform="rotate(-90 110 110)"/>
      <text x="110" y="104" text-anchor="middle" font-size="26" font-weight="700" fill="#1a1a1a">{selected_frac*100:.0f}%</text>
      <text x="110" y="126" text-anchor="middle" font-size="12" fill="#666">recomendados</text>
    </svg>
    <div style="display:flex; gap:18px; justify-content:center; margin-top:8px; font-size:0.9rem;">
      <span><span style="display:inline-block;width:10px;height:10px;background:#27ae60;border-radius:2px;margin-right:6px;"></span>Seleccionados ({selected})</span>
      <span><span style="display:inline-block;width:10px;height:10px;background:#e74c3c;border-radius:2px;margin-right:6px;"></span>Descartados ({rejected})</span>
    </div>
    """


def _bar_chart(
    items: List[tuple],
    color: str,
    max_value: float | None = None,
    show_raw_values: List[Any] | None = None,
) -> str:
    if not items:
        return '<p class="muted">Sin datos.</p>'

    values = [v for _, v in items]
    top = max_value if max_value is not None else max(values) if values else 1.0
    top = top or 1.0

    row_height = 34
    svg_height = row_height * len(items) + 10
    bars = []
    for i, (label, value) in enumerate(items):
        y = i * row_height + 6
        bar_w = max(2.0, (float(value) / top) * 560.0) if top else 2.0
        raw = show_raw_values[i] if show_raw_values and i < len(show_raw_values) else value
        value_label = _fmt_num(raw)
        bars.append(f"""
          <text x="0" y="{y + 14}" font-size="13" fill="#333">{escape(str(label))}</text>
          <rect x="180" y="{y}" width="{bar_w:.1f}" height="18" rx="4" fill="{color}"/>
          <text x="{180 + bar_w + 8:.1f}" y="{y + 14}" font-size="12" fill="#333">{escape(value_label)}</text>
        """)

    return f"""
    <svg viewBox="0 0 780 {svg_height}" width="100%" height="{svg_height}" role="img" aria-label="Gráfico de barras">
      {''.join(bars)}
    </svg>
    """
