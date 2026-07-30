"""
src/utils/reporter.py - Generador de reportes y datos estructurados.
"""

import json
from typing import Dict


def generate_analysis_json(data: Dict, output_path: str):
    """Guarda los datos de análisis en analysis.json."""
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Datos de análisis guardados en {output_path}")


def generate_html_report(analysis_data: Dict, output_path: str):
    """Genera un reporte visual simple en HTML."""
    
    scene = analysis_data.get("scene_analysis", {})
    metadata = analysis_data.get("metadata", {})
    
    html_content = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>intelliframes4colmap Report</title>
    <style>
        body {{ font-family: sans-serif; padding: 20px; background-color: #f4f4f9; }}
        .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; }}
        .section {{ margin-bottom: 20px; }}
        .metric {{ display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px solid #eee; }}
        .label {{ font-weight: bold; color: #555; }}
        .value {{ color: #007bff; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>intelliframes4colmap | VIDEO ANALYSIS REPORT</h1>
        
        <div class="section">
            <h2>[METADATA]</h2>
            <div class="metric"><span class="label">Resolution:</span> <span class="value">{metadata.get('resolution', 'N/A')}</span></div>
            <div class="metric"><span class="label">FPS:</span> <span class="value">{metadata.get('fps', 'N/A')}</span></div>
            <div class="metric"><span class="label">Duration:</span> <span class="value">{metadata.get('duration', 'N/A')}</span></div>
        </div>

        <div class="section">
            <h2>[SCENE ANALYSIS]</h2>
            <div class="metric"><span class="label">Camera movement:</span> <span class="value">{scene.get('camera_movement', 'N/A')}</span></div>
            <div class="metric"><span class="label">Texture density:</span> <span class="value">{scene.get('texture_density', 'N/A')}</span></div>
            <div class="metric"><span class="label">Motion Blur:</span> <span class="value">{scene.get('motion_blur', 'N/A')}</span></div>
        </div>

        <div class="section">
            <h2>[COLMAP RECOMMENDATIONS]</h2>
            <div class="metric"><span class="label">Matching:</span> <span class="value">{analysis_data.get('colmap_config', {}).get('matching_strategy', 'N/A')}</span></div>
            <div class="metric"><span class="label">Overlap:</span> <span class="value">{analysis_data.get('colmap_config', {}).get('sequential_overlap', 'N/A')}</span></div>
            <div class="metric"><span class="label">SIFT Features:</span> <span class="value">{analysis_data.get('colmap_config', {}).get('sift_max_features', 'N/A')}</span></div>
        </div>
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    print(f"Reporte HTML guardado en {output_path}")
