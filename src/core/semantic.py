"""
src/core/semantic.py - Nivel 2: Análisis semántico
Implementa Segmentación, Textura Richness y Control de Exposición.
"""

import cv2
import numpy as np
from typing import Dict


def calculate_texture_richness(frame: np.ndarray) -> str:
    """
    Calcula la riqueza de textura (densidad de características).
    Retorna 'HIGH', 'MEDIUM' o 'LOW'.
    """
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    # Detectar esquinas (SURF/SIFT-like features count approximation)
    # Usamos goodFeaturesToTrack como proxy para densidad de textura
    corners = cv2.goodFeaturesToTrack(
        gray, 
        maxCorners=10000, 
        qualityLevel=0.01, 
        minDistance=3, 
        blockSize=3
    )

    if corners is not None:
        count = len(corners)
        # Heurística basada en resolución típica (ej. 4K ~ 8MP)
        resolution = frame.shape[0] * frame.shape[1]
        features_per_mp = count / (resolution / 1_000_000)

        if features_per_mp > 5:
            return "HIGH"
        elif features_per_mp > 2:
            return "MEDIUM"
        else:
            return "LOW"
    return "LOW"


def analyze_exposure(frame: np.ndarray) -> Dict[str, any]:
    """
    Analiza la exposición (luminancia, histogramas).
    Retorna recomendación de Tone Mapping.
    """
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    mean_brightness = np.mean(gray)
    
    # Detectar clipping (zonas completamente blancas o negras)
    white_pixels = np.sum(gray == 255)
    black_pixels = np.sum(gray == 0)
    total_pixels = gray.shape[0] * gray.shape[1]
    
    clip_ratio = (white_pixels + black_pixels) / total_pixels
    
    return {
        "mean_brightness": float(mean_brightness),
        "highlights_clipping": white_pixels > (total_pixels * 0.05),
        "shadows_clipping": black_pixels > (total_pixels * 0.05),
        "tone_mapping_recommended": clip_ratio > 0.1 # Si hay mucho clipping, sugerir Tone Mapping
    }


def detect_dynamic_objects(frame: np.ndarray) -> np.ndarray:
    """
    Placeholder para detección de objetos dinámicos (YOLO/SAM).
    En una implementación completa, aquí se cargarían los modelos.
    Retorna una máscara binaria.
    """
    # Simulación: retornar un frame negro (sin objetos detectados)
    return np.zeros(frame.shape[:2], dtype=np.uint8)
