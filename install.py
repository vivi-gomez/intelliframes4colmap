#!/usr/bin/env python3
"""
Script de instalación para intelliframes4colmap.
Maneja dependencias del sistema y de Python.
"""

import os
import sys
import subprocess
import platform

def run_command(cmd, check=True):
    """Ejecuta un comando shell."""
    print(f">> Ejecutando: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error ejecutando '{cmd}':\n{result.stderr}")
        return False
    return True

def install_system_deps():
    """Instala dependencias del sistema operativo."""
    system = platform.system()
    
    if system == "Linux":
        # Detectar distribucion (apt vs yum/dnf)
        if os.path.exists("/etc/apt"):
            print("Detectado sistema basado en Debian/Ubuntu.")
            run_command("sudo apt-get update")
            packages = ["ffmpeg", "exiftool", "libopencv-dev", "python3-opencv"]
            cmd = f"sudo apt-get install -y {' '.join(packages)}"
        else:
            print("Detectado sistema basado en RPM (Fedora/RedHat).")
            run_command("sudo dnf update -y")
            packages = ["ffmpeg", "exiftool", "opencv-devel"]
            cmd = f"sudo dnf install -y {' '.join(packages)}"
            
        if not run_command(cmd):
            print("Advertencia: Algunas dependencias del sistema podrían no haberse instalado correctamente.")
            
    elif system == "Darwin": # macOS
        print("Detectado macOS. Asegúrate de tener Homebrew instalado.")
        run_command("brew install ffmpeg exiftool opencv")
        
    elif system == "Windows":
        print("Para Windows, instala manualmente:")
        print("- FFmpeg: https://ffmpeg.org/download.html")
        print("- ExifTool: https://exiftool.org/")
        print("- OpenCV: pip install opencv-python (incluye binarios)")

def install_python_deps():
    """Instala dependencias de Python."""
    packages = [
        "torch",
        "torchvision", 
        "opencv-python-headless",
        "pandas",
        "scipy",
        "pyproj",
        "segment-anything", # SAM
        "ultralytics",       # YOLOv8
        "matplotlib",
        "numpy",
        "tqdm"
    ]
    
    print("\nInstalando dependencias de Python...")
    cmd = f"{sys.executable} -m pip install {' '.join(packages)} --upgrade"
    if not run_command(cmd):
        print("Error al instalar dependencias de Python.")

def main():
    print("="*50)
    print("Instalador de intelliframes4colmap")
    print("="*50)
    
    install_system_deps()
    install_python_deps()
    
    print("\n¡Instalación completada!")
    print("Ejecuta 'python main.py --help' para ver las opciones.")

if __name__ == "__main__":
    main()
