param(
    [string]$DatasetPath = "",
    [string]$ColmapExe = "colmap",
    [switch]$SparseOnly
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($DatasetPath)) {
    $DatasetPath = Join-Path $PSScriptRoot "data"
}

$ImagePath = Join-Path $DatasetPath "images"
$ColmapPath = Join-Path $DatasetPath "colmap"
$SparsePath = Join-Path $ColmapPath "sparse"
$DensePath = Join-Path $ColmapPath "dense"
$DatabasePath = Join-Path $ColmapPath "database.db"

if (-not (Get-Command $ColmapExe -ErrorAction SilentlyContinue)) {
    throw "COLMAP executable '$ColmapExe' was not found. Add COLMAP to PATH or pass -ColmapExe C:\path\to\colmap.bat."
}

if (-not (Test-Path $ImagePath)) {
    throw "Image directory not found: $ImagePath"
}

New-Item -ItemType Directory -Force -Path $SparsePath | Out-Null
New-Item -ItemType Directory -Force -Path $DensePath | Out-Null

Write-Host "=== Step 1: Feature Extraction ==="
& $ColmapExe feature_extractor `
    --database_path $DatabasePath `
    --image_path $ImagePath `
    --ImageReader.camera_model PINHOLE `
    --ImageReader.single_camera 1 `
    --FeatureExtraction.use_gpu 1

Write-Host "=== Step 2: Feature Matching ==="
& $ColmapExe exhaustive_matcher `
    --database_path $DatabasePath `
    --FeatureMatching.use_gpu 1

Write-Host "=== Step 3: Sparse Reconstruction (Bundle Adjustment) ==="
& $ColmapExe mapper `
    --database_path $DatabasePath `
    --image_path $ImagePath `
    --output_path $SparsePath

$SparseModelPath = Join-Path $SparsePath "0"
if (-not (Test-Path $SparseModelPath)) {
    throw "Sparse model was not created at $SparseModelPath. Check COLMAP matching/mapper logs."
}

if ($SparseOnly) {
    Write-Host "=== Sparse-only run complete ==="
    Write-Host "Sparse model: $SparseModelPath"
    exit 0
}

Write-Host "=== Step 4: Image Undistortion ==="
& $ColmapExe image_undistorter `
    --image_path $ImagePath `
    --input_path $SparseModelPath `
    --output_path $DensePath

Write-Host "=== Step 5: Dense Reconstruction (Patch Match Stereo) ==="
& $ColmapExe patch_match_stereo `
    --workspace_path $DensePath `
    --PatchMatchStereo.geom_consistency true

Write-Host "=== Step 6: Stereo Fusion ==="
& $ColmapExe stereo_fusion `
    --workspace_path $DensePath `
    --output_path (Join-Path $DensePath "fused.ply")

Write-Host "=== Done ==="
Write-Host "Sparse: $SparseModelPath"
Write-Host "Dense:  $(Join-Path $DensePath 'fused.ply')"
