#!/usr/bin/env python3
"""
main.py - Punto de entrada para intelliframes4colmap.
CLI para orquestar el análisis de video/imágenes y generar configuración COLMAP.
"""

import argparse
import os
import sys
import json
from datetime import datetime

# Añadir la ruta del proyecto al path para imports relativos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.analysis import calculate_sharpness_score, detect_motion_blur, analyze_optical_flow
from src.core.semantic import calculate_texture_richness, analyze_exposure
from src.engine.decision import DecisionEngine
from src.utils.reporter import generate_analysis_json, generate_html_report


def parse_args():
    parser = argparse.ArgumentParser(description="intelliframes4colmap - Optimización inteligente para COLMAP")
    parser.add_argument("input", type=str, help="Ruta al video o carpeta de imágenes de entrada")
    parser.add_argument("-o", "--output-dir", type=str, default="./output", help="Directorio de salida (default: ./output)")
    return parser.parse_args()


def analyze_video(input_path: str):
    """
    Simula el análisis completo del video según el README.
    En una implementación real, aquí se iteraría frame a frame usando FFmpeg/OpenCV.
    """
    print(f"Analizando entrada: {input_path}")
    
    # Datos simulados basados en la lectura del README
    analysis_data = {
        "metadata": {
            "resolution": "3840x2160",
            "fps": 59.94,
            "duration": "02:18",
            "frames_total": 8276
        },
        "scene_analysis": {
            "camera_movement": "HIGH",
            "rotation": "MEDIUM/HIGH",
            "translation": "HIGH",
            "motion_blur": "MEDIUM",
            "texture_density": "HIGH",
            "repeated_texture": False,
            "dynamic_objects": "LOW"
        },
        "frame_selection": {
            "recommended": 1247,
            "rejected": 7029
        }
    }
    
    # Simular análisis de textura y exposición (Nivel 2)
    analysis_data["texture_richness"] = calculate_texture_richness(None) # Placeholder
    analysis_data["exposure_analysis"] = analyze_exposure(None) # Placeholder
    
    return analysis_data


def main():
    args = parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: La ruta '{args.input}' no existe.")
        sys.exit(1)

    # Crear directorio de salida
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Análisis (Nivel 1 y 2)
    analysis_data = analyze_video(args.input)
    
    # 2. Motor de Decisión (Nivel 3)
    engine = DecisionEngine()
    engine.update_from_analysis(analysis_data)
    colmap_config_path = os.path.join(args.output_dir, "colmap_config.json")
    final_config = engine.generate_config(colmap_config_path)
    
    # Actualizar datos con config final para reportes
    analysis_data["colmap_config"] = final_config
    
    # 3. Generar Reportes
    generate_analysis_json(analysis_data, os.path.join(args.output_dir, "analysis.json"))
    generate_html_report(analysis_data, os.path.join(args.output_dir, "report.html"))
    
    print("\nProceso completado. Archivos generados en:", args.output_dir)


if __name__ == "__main__":
    main()
