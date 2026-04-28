#!/usr/bin/env bash
# COLMAP 3D reconstruction pipeline
# Usage: bash run_colmap.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_PATH="${DATASET_PATH:-$SCRIPT_DIR/data}"
IMAGE_PATH="$DATASET_PATH/images"
COLMAP_PATH="$DATASET_PATH/colmap"
SPARSE_ONLY="${SPARSE_ONLY:-0}"

if ! command -v colmap >/dev/null 2>&1; then
    echo "COLMAP was not found in PATH. Install COLMAP or add it to PATH before running this script." >&2
    exit 1
fi

mkdir -p "$COLMAP_PATH/sparse"
mkdir -p "$COLMAP_PATH/dense"

echo "=== Step 1: Feature Extraction ==="
colmap feature_extractor \
    --database_path "$COLMAP_PATH/database.db" \
    --image_path "$IMAGE_PATH" \
    --ImageReader.camera_model PINHOLE \
    --ImageReader.single_camera 1 \
    --FeatureExtraction.use_gpu 1

echo "=== Step 2: Feature Matching ==="
colmap exhaustive_matcher \
    --database_path "$COLMAP_PATH/database.db" \
    --FeatureMatching.use_gpu 1

echo "=== Step 3: Sparse Reconstruction (Bundle Adjustment) ==="
colmap mapper \
    --database_path "$COLMAP_PATH/database.db" \
    --image_path "$IMAGE_PATH" \
    --output_path "$COLMAP_PATH/sparse"

if [ ! -d "$COLMAP_PATH/sparse/0" ]; then
    echo "Sparse model was not created at $COLMAP_PATH/sparse/0. Check COLMAP logs." >&2
    exit 1
fi

if [ "$SPARSE_ONLY" = "1" ]; then
    echo "=== Sparse-only run complete ==="
    echo "Sparse: $COLMAP_PATH/sparse/0/"
    exit 0
fi

echo "=== Step 4: Image Undistortion ==="
colmap image_undistorter \
    --image_path "$IMAGE_PATH" \
    --input_path "$COLMAP_PATH/sparse/0" \
    --output_path "$COLMAP_PATH/dense"

echo "=== Step 5: Dense Reconstruction (Patch Match Stereo) ==="
colmap patch_match_stereo \
    --workspace_path "$COLMAP_PATH/dense" \
    --PatchMatchStereo.geom_consistency true

echo "=== Step 6: Stereo Fusion ==="
colmap stereo_fusion \
    --workspace_path "$COLMAP_PATH/dense" \
    --output_path "$COLMAP_PATH/dense/fused.ply"


echo "=== Done! ==="
echo "Results:"
echo "  Sparse: $COLMAP_PATH/sparse/0/"
echo "  Dense:  $COLMAP_PATH/dense/fused.ply"
