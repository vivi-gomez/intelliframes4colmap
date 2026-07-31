#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELPERS_DIR="$SCRIPT_DIR/src/helpers_to_colmap"

source "$HELPERS_DIR/lib_colmap.sh"

FORCE_ALL=0
WITH_MASKS=0
DENSE=0
GPU_INDEX=0
USE_CPU=0
OUTPUT_ARG=""

usage() {
cat <<EOF
Uso:
  ./run_colmap.sh [RUTA_OUTPUT] [opciones]

Opciones:
  --force-all
      Usar todas las imágenes, ignorando frames_selected si aplica.

  --with-masks
      Usar máscaras de output/masks si existen.

  --dense
      Ejecutar también reconstrucción densa.

  --gpu-index N
      Índice de GPU a usar. Por defecto: 0

  --cpu
      Forzar CPU y deshabilitar GPU.

  -h, --help
      Mostrar esta ayuda.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force-all)
            FORCE_ALL=1
            shift
            ;;
        --with-masks)
            WITH_MASKS=1
            shift
            ;;
        --dense)
            DENSE=1
            shift
            ;;
        --cpu)
            USE_CPU=1
            shift
            ;;
        --gpu-index)
            [[ $# -ge 2 ]] || Die "Falta valor para --gpu-index"
            GPU_INDEX="$2"
            shift 2
            ;;
        --gpu-index=*)
            GPU_INDEX="${1#*=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            Die "Opción desconocida: $1"
            ;;
        *)
            [[ -z "$OUTPUT_ARG" ]] || Die "Solo se admite una ruta output"
            OUTPUT_ARG="$1"
            shift
            ;;
    esac
done

[[ "$GPU_INDEX" =~ ^[0-9]+$ ]] || Die "--gpu-index debe ser un entero >= 0"

CheckCOLMAP

COLMAP_VERSION="$(DetectVersion)"
CASPAR_AVAILABLE="$(DetectCaspar)"
OUTPUT_DIR="$(DetectOutput "$SCRIPT_DIR" "$OUTPUT_ARG")"
CONFIG="$(CheckConfig "$OUTPUT_DIR")"

export FORCE_ALL WITH_MASKS DENSE GPU_INDEX USE_CPU
export COLMAP_VERSION CASPAR_AVAILABLE OUTPUT_DIR CONFIG
export SCRIPT_DIR HELPERS_DIR

InitJSON
LoadConfig "$CONFIG"

cd "$OUTPUT_DIR"
PrintBanner

Info "Configuración encontrada: $CONFIG"
Info "COLMAP version detectada: $COLMAP_VERSION"
Info "CASPAR disponible: $CASPAR_AVAILABLE"

IMAGE_DIR="$(FindImages "$OUTPUT_DIR" "$FORCE_ALL")"
MASK_DIR="$(FindMasks "$OUTPUT_DIR" "$WITH_MASKS" || true)"
export IMAGE_DIR MASK_DIR

PrepareWorkspace
PrintConfigSummary

if VersionGE "$COLMAP_VERSION" "4.0.0"; then
    source "$HELPERS_DIR/run_colmap_v4.sh"
else
    source "$HELPERS_DIR/run_colmap_v3.sh"
fi
