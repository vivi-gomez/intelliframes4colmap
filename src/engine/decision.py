"""
src/engine/decision.py - Nivel 3: Motor de decisión
Genera colmap_config.json basado en los análisis previos.
"""

import json
from typing import Dict, List


class DecisionEngine:
    def __init__(self):
        self.config = {
            "matching_strategy": "SEQUENTIAL",
            "sequential_overlap": 10,
            "loop_detection": False,
            "vocabulary_tree": False,
            "sift_max_features": 16000,
            "camera_model": "SIMPLE_RADIAL",
            "guided_matching": True,
            "tone_mapping": False
        }

    def update_from_analysis(self, analysis_data: Dict):
        """Actualiza la configuración COLMAP basada en los resultados del análisis."""
        
        # 1. Matching Strategy & Overlap (basado en movimiento)
        movement = analysis_data.get("scene_analysis", {}).get("camera_movement", "LOW")
        if movement == "HIGH":
            self.config["matching_strategy"] = "SEQUENTIAL"
            self.config["sequential_overlap"] = 25 # Mayor overlap para movimiento rápido
        else:
            self.config["sequential_overlap"] = 14
            
        # Loop detection si hay bucles o alta redundancia
        if analysis_data.get("scene_analysis", {}).get("repeated_texture", False):
            self.config["loop_detection"] = True
            self.config["vocabulary_tree"] = True

        # 2. SIFT Max Features (basado en textura)
        texture = analysis_data.get("texture_richness", "MEDIUM")
        if texture == "HIGH":
            self.config["sift_max_features"] = 16000
        elif texture == "LOW":
            self.config["sift_max_features"] = 8000
        else:
            self.config["sift_max_features"] = 12000

        # 3. Tone Mapping (basado en exposición)
        if analysis_data.get("exposure_analysis", {}).get("tone_mapping_recommended", False):
            self.config["tone_mapping"] = True

    def generate_config(self, output_path: str):
        """Guarda la configuración final en un archivo JSON."""
        with open(output_path, 'w') as f:
            json.dump(self.config, f, indent=4)
        print(f"Configuración COLMAP guardada en {output_path}")
        return self.config
