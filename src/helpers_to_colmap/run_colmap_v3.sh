#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib_colmap.sh"

Info "Backend seleccionado: COLMAP 3.x"

if [[ "${USE_CPU}" -eq 1 ]]; then
    FE_USE_GPU=0
    MATCH_USE_GPU=0
else
    FE_USE_GPU=1
    MATCH_USE_GPU=1
fi

echo "[1/5] Extrayendo características SIFT..."
FE_ARGS=(
    --database_path "$DB_PATH"
    --image_path "$IMAGE_DIR"
    --SiftExtraction.max_num_features "${CFG["MAX_FEATURES"]}"
    --SiftExtraction.estimate_affine_shape "${CFG[AFFINE]}"
    --SiftExtraction.domain_size_pooling "${CFG[POOLING]}"
    --SiftExtraction.use_gpu "$FE_USE_GPU"
)

if [[ "$USE_CPU" -eq 0 ]]; then
    FE_ARGS+=(--SiftExtraction.gpu_index "$GPU_INDEX")
fi

if [[ -n "${MASK_DIR:-}" ]]; then
    FE_ARGS+=(--ImageReader.mask_path "$MASK_DIR")
fi

if [[ "${CFG[USE_GPS]}" -eq 1 ]]; then
    FE_ARGS+=(--ImageReader.use_camera_exif 1)
fi

colmap feature_extractor "${FE_ARGS[@]}"

echo
echo "[2/5] Emparejando features (estrategia: ${CFG[MATCH_STRATEGY]})..."

MATCH_ARGS=(
    --database_path "$DB_PATH"
    --SiftMatching.guided_matching "${CFG[GUIDED_MATCHING]}"
    --SiftMatching.use_gpu "$MATCH_USE_GPU"
)

if [[ "$USE_CPU" -eq 0 ]]; then
    MATCH_ARGS+=(--SiftMatching.gpu_index "$GPU_INDEX")
fi

case "${CFG[MATCH_STRATEGY]}" in
    exhaustive)
        colmap exhaustive_matcher "${MATCH_ARGS[@]}"
        ;;
    sequential)
        colmap sequential_matcher \
            "${MATCH_ARGS[@]}" \
            --SequentialMatching.overlap "${CFG[OVERLAP]}"
        ;;
    spatial)
        colmap spatial_matcher "${MATCH_ARGS[@]}"
        ;;
    vocab_tree)
        Warn "vocab_tree requiere un archivo de vocabulario."
        read -rp "Ruta al vocab tree (vocab-tree.bin): " VT_PATH
        [[ -n "$VT_PATH" ]] || Die "No se indicó ruta para vocab-tree.bin"
        colmap vocab_tree_matcher \
            "${MATCH_ARGS[@]}" \
            --VocabTreeMatching.vocab_tree_path "$VT_PATH"
        ;;
    *)
        Die "Estrategia de matching desconocida: ${CFG[MATCH_STRATEGY]}"
        ;;
esac

echo
echo "[3/5] Reconstrucción incremental (mapper)..."
colmap mapper \
    --database_path "$DB_PATH" \
    --image_path "$IMAGE_DIR" \
    --output_path "$SPARSE_DIR" \
    --Mapper.ba_global_max_num_iterations "${CFG[BA_ITERATIONS]}" \
    --Mapper.min_num_matches "${CFG[MIN_MATCHES]}" \
    --Mapper.init_min_num_inliers "${CFG[MIN_INLIERS]}"

echo
MODEL_DIR="$(FindSparseModel "$SPARSE_DIR" || true)"
[[ -n "$MODEL_DIR" ]] || Die "El mapper no generó un modelo válido."

echo "[4/5] Modelo esparcido generado en: $MODEL_DIR"

if [[ "$DENSE" -eq 1 ]]; then
    echo
    echo "[5/5] Reconstrucción densa..."

    echo "  -> Undistorsionando imágenes..."
    colmap image_undistorter \
        --image_path "$IMAGE_DIR" \
        --input_path "$MODEL_DIR" \
        --output_path "$DENSE_DIR" \
        --output_type COLMAP

    echo "  -> Patch match stereo..."
    PM_ARGS=(
        --workspace_path "$DENSE_DIR"
        --workspace_format COLMAP
        --PatchMatchStereo.geom_consistency true
    )
    if [[ "$USE_CPU" -eq 0 ]]; then
        PM_ARGS+=(--PatchMatchStereo.gpu_index "$GPU_INDEX")
    fi
    colmap patch_match_stereo "${PM_ARGS[@]}"

    echo "  -> Fusión estéreo..."
    colmap stereo_fusion \
        --workspace_path "$DENSE_DIR" \
        --workspace_format COLMAP \
        --input_type geometric \
        --output_path "$DENSE_DIR/fused.ply"

    echo "  -> Malla Poisson..."
    colmap poisson_mesher \
        --input_path "$DENSE_DIR/fused.ply" \
        --output_path "$DENSE_DIR/meshed-poisson.ply"
else
    echo "[5/5] Reconstrucción densa omitida"
fi

PrintSummary "$MODEL_DIR"
