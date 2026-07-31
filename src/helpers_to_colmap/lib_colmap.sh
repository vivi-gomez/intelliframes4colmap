#!/usr/bin/env bash
set -euo pipefail

Die()  { echo "[ERROR] $*" >&2; exit 1; }
Info() { echo "[INFO]  $*" >&2; }
Warn() { echo "[AVISO] $*" >&2; }

VersionGE() {
    # devuelve 0 si $1 >= $2
    printf '%s\n%s\n' "$2" "$1" | sort -V -C
}

BoolToInt() {
    local v="${1:-false}"
    v="${v,,}"
    [[ "$v" == "true" || "$v" == "1" || "$v" == "yes" ]] && echo 1 || echo 0
}

CheckCOLMAP() {
    command -v colmap >/dev/null 2>&1 || Die "'colmap' no está en el PATH. Instálalo primero."
}

DetectVersion() {
    local v=""
    if colmap version >/dev/null 2>&1; then
        v="$(colmap version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -n1 || true)"
    fi
    if [[ -z "$v" ]]; then
        v="$(colmap help 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -n1 || true)"
    fi
    [[ -n "$v" ]] || v="4.0.0"
    echo "$v"
}

DetectCaspar() {
    local help_txt
    help_txt="$(colmap mapper -h 2>&1 || true)"
    if grep -qE 'Mapper\.ba_local_backend|Mapper\.ba_global_backend|CASPAR' <<<"$help_txt"; then
        echo "yes"
    else
        echo "no"
    fi
}

DetectOutput() {
    local base_dir="${1:-$(pwd)}"
    local explicit="${2:-}"
    local out=""

    if [[ -n "$explicit" ]]; then
        [[ -d "$explicit" ]] || Die "El directorio no existe: $explicit"
        out="$(cd "$explicit" && pwd)"
    elif [[ -d "$base_dir/output" ]]; then
        out="$(cd "$base_dir/output" && pwd)"
        Info "Usando output detectado automáticamente: $out"
    else
        Die "No se encontró carpeta output/. Especifica la ruta manualmente."
    fi

    echo "$out"
}

CheckConfig() {
    local out="$1"
    local cfg=""
    cfg="$(find "$out" -maxdepth 2 -name "colmap_config.json" -type f 2>/dev/null | head -n1 || true)"
    [[ -n "$cfg" ]] || Die "No se encuentra colmap_config.json en $out ni subcarpetas."
    echo "$cfg"
}

InitJSON() {
    if command -v jq >/dev/null 2>&1; then
        HAS_JQ=1
    else
        HAS_JQ=0
        Info "'jq' no instalado. Usando fallback Python para leer JSON."
    fi
    export HAS_JQ
}

ReadJSON() {
    local key="$1"
    local default="${2:-}"
    local cfg="${3:?cfg requerido}"

    if [[ "${HAS_JQ:-0}" -eq 1 ]]; then
        jq -r "$key // empty" "$cfg" 2>/dev/null || echo "$default"
    else
        python3 - "$key" "$default" "$cfg" <<'PY' 2>/dev/null
import json, sys
key, default, path = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    v = data
    for k in key.split("."):
        v = v[k]
    print(v if v is not None else default)
except Exception:
    print(default)
PY
    fi
}

FindImages() {
    local out="$1"
    local force_all="${2:-0}"
    local dir=""
    local count=0

    cd "$out"

    if [[ "$force_all" -eq 0 && -d "frames_selected" ]]; then
        count=$(find frames_selected -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.tif" -o -iname "*.tiff" \) | wc -l)
        if [[ "$count" -gt 0 ]]; then
            Info "Usando frames seleccionados: $count imágenes"
            echo "frames_selected"
            return 0
        fi
    fi

    for d in frames_selected frames_rejected input images; do
        if [[ -d "$d" ]]; then
            count=$(find "$d" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.tif" -o -iname "*.tiff" \) | wc -l)
            if [[ "$count" -gt 0 ]]; then
                local mode=""
                [[ "$force_all" -eq 1 ]] && mode=" (forzado)"
                Info "Usando carpeta '$d' con $count imágenes$mode"
                echo "$d"
                return 0
            fi
        fi
    done

    Die "No se encontró ninguna carpeta con imágenes en $out"
}

FindMasks() {
    local out="$1"
    local with_masks="${2:-0}"

    cd "$out"
    [[ "$with_masks" -eq 1 ]] || return 0
    [[ -d "masks" ]] || return 0

    local count
    count=$(find masks -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.tif" -o -iname "*.tiff" -o -iname "*.pbm" -o -iname "*.pgm" -o -iname "*.ppm" \) | wc -l)

    if [[ "$count" -gt 0 ]]; then
        Info "Usando máscaras: $count archivos en masks/"
        echo "masks"
    else
        Warn "--with-masks activado pero masks/ está vacía."
    fi
}

LoadConfig() {
    local cfg="$1"

    declare -gA CFG
    CFG[MAX_FEATURES]="$(ReadJSON 'feature_extraction.SiftExtraction.max_num_features' '8000' "$cfg")"
    CFG[AFFINE]="$(BoolToInt "$(ReadJSON 'feature_extraction.SiftExtraction.estimate_affine_shape' 'false' "$cfg")")"
    CFG[POOLING]="$(BoolToInt "$(ReadJSON 'feature_extraction.SiftExtraction.domain_size_pooling' 'false' "$cfg")")"
    CFG[MATCH_STRATEGY]="$(ReadJSON 'matching.strategy' 'exhaustive' "$cfg")"
    CFG[GUIDED_MATCHING]="$(BoolToInt "$(ReadJSON 'matching.SiftMatching.guided_matching' 'true' "$cfg")")"
    CFG[BA_ITERATIONS]="$(ReadJSON 'mapping.Mapper.ba_global_max_num_iterations' '50' "$cfg")"
    CFG[MIN_MATCHES]="$(ReadJSON 'mapping.Mapper.min_num_matches' '15' "$cfg")"
    CFG[MIN_INLIERS]="$(ReadJSON 'mapping.Mapper.init_min_num_inliers' '50' "$cfg")"
    CFG[USE_GPS]="$(BoolToInt "$(ReadJSON 'priors.use_gps_priors' 'false' "$cfg")")"
    CFG[OVERLAP]="$(ReadJSON 'matching.overlap' '10' "$cfg")"
}

PrepareWorkspace() {
    mkdir -p colmap_result/database colmap_result/sparse/0 colmap_result/dense/0
    export DB_PATH="colmap_result/database/database.db"
    export SPARSE_DIR="colmap_result/sparse"
    export DENSE_DIR="colmap_result/dense"
}

FindSparseModel() {
    local sparse_dir="$1"

    if [[ -d "$sparse_dir/0" && -n "$(ls -A "$sparse_dir/0" 2>/dev/null)" ]]; then
        echo "$sparse_dir/0"
        return 0
    fi

    local d
    for d in "$sparse_dir"/*; do
        [[ -d "$d" ]] || continue
        [[ -n "$(ls -A "$d" 2>/dev/null)" ]] || continue
        echo "$d"
        return 0
    done

    return 1
}

PrintBanner() {
    echo "========================================"
    echo "  IntelliFrames4COLMAP -> COLMAP"
    echo "  Output: $(pwd)"
    echo "========================================"
}

PrintConfigSummary() {
    local gpu_msg="GPU $GPU_INDEX"
    [[ "${USE_CPU:-0}" -eq 1 ]] && gpu_msg="CPU (forzado)"

    echo
    echo "----- Configuración leída de $CONFIG -----"
    echo "  COLMAP version    : $COLMAP_VERSION"
    echo "  Max features      : ${CFG[MAX_FEATURES]}"
    echo "  Affine shape      : ${CFG[AFFINE]}"
    echo "  Domain pooling    : ${CFG[POOLING]}"
    echo "  Matching strategy : ${CFG[MATCH_STRATEGY]}"
    echo "  Guided matching   : ${CFG[GUIDED_MATCHING]}"
    echo "  BA iterations     : ${CFG[BA_ITERATIONS]}"
    echo "  Min matches       : ${CFG[MIN_MATCHES]}"
    echo "  Min inliers       : ${CFG[MIN_INLIERS]}"
    echo "  Use GPS priors    : ${CFG[USE_GPS]}"
    echo "  Modo GPU/CPU      : $gpu_msg"
    echo "  CASPAR disponible : $CASPAR_AVAILABLE"
    [[ -n "${MASK_DIR:-}" ]] && echo "  Máscaras          : ${MASK_DIR}/"
    echo "-------------------------------------------"
    echo
}

PrintSummary() {
    local model_dir="$1"

    echo
    echo "========================================"
    echo "  RESUMEN"
    echo "========================================"
    echo "  Directorio base  : $(pwd)"
    echo "  Imágenes usadas  : ${IMAGE_DIR}/"
    echo "  Resultado COLMAP : colmap_result/"
    echo "  Modelo esparcido : ${model_dir}"
    [[ -n "${MASK_DIR:-}" ]] && echo "  Máscaras usadas  : ${MASK_DIR}/"
    [[ "${DENSE:-0}" -eq 1 ]] && echo "  Densa            : ${DENSE_DIR}/"
    echo
    echo "  Comandos útiles:"
    echo "    Ver modelo GUI:"
    echo "      colmap gui --database_path $DB_PATH --import_path $model_dir"
    echo
    echo "    Convertir a PLY/TXT:"
    echo "      colmap model_converter --input_path $model_dir --output_path $model_dir/model.ply --output_type PLY"
    echo
    echo "    Exportar cámaras para Gaussian Splatting:"
    echo "      colmap model_converter --input_path $model_dir --output_path $model_dir --output_type TXT"
    echo "========================================"
}
