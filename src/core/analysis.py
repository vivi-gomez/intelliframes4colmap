"""
src/core/analysis.py - Nivel 1: Calidad de imagen
Implementa Sharpness Score y Optical Flow según el README.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict


def calculate_sharpness_score(frame: np.ndarray) -> float:
    """
    Calcula el índice de nitidez (Sharpness Score).
    Método: Laplaciano + desviación estándar local.
    """
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    # Calculo del Laplaciano
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # Desviación estándar como métrica de nitidez
    score = np.std(laplacian)
    return float(score)


def detect_motion_blur(frame: np.ndarray, threshold: float = 10.0) -> bool:
    """Detecta motion blur basado en el Sharpness Score."""
    score = calculate_sharpness_score(frame)
    return score < threshold


def analyze_optical_flow(prev_frame: np.ndarray, curr_frame: np.ndarray) -> Dict[str, any]:
    """
    Analiza el flujo óptico entre dos fotogramas.
    Distingue movimiento de cámara (coherente) vs objetos dinámicos.
    """
    if len(prev_frame.shape) == 3:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    else:
        prev_gray = prev_frame
        
    if len(curr_frame.shape) == 3:
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    else:
        curr_gray = curr_frame

    # Detectar puntos clave para Lucas-Kanade
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
    
    if prev_pts is None:
        return {"camera_movement": "LOW", "dynamic_objects": False}

    # Calcular flujo óptico
    next_pts, status, win = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
    
    if next_pts is None:
        return {"camera_movement": "UNKNOWN", "dynamic_objects": False}

    # Filtrar puntos buenos
    good_prev = prev_pts[status == 1]
    good_next = next_pts[status == 1]

    # Calcular desplazamiento medio
    if len(good_prev) > 0 and len(good_next) > 0:
        displacement = np.abs(good_next - good_prev).mean()
        
        # Heurística simple para determinar tipo de movimiento
        # En una implementación real, se usaría clustering (RANSAC) para separar plano de fondo de objetos
        if displacement > 5.0:
            return {"camera_movement": "HIGH", "dynamic_objects": True}
        else:
            return {"camera_movement": "LOW", "dynamic_objects": False}
    
    return {"camera_movement": "MEDIUM", "dynamic_objects": False}
