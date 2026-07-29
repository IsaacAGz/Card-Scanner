# One-time setup for local MTG Card Scanner development.
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$AppDir = Join-Path $RepoRoot "app"

Write-Host "MTG Card Scanner setup"
Write-Host "Repository: $RepoRoot"

Set-Location $RepoRoot

Write-Host "`nInstalling Python dependencies..."
python -m pip install -r requirements.txt

$onnxRuntime = Join-Path $AppDir "onnx_dinov2\model.onnx"
$onnxRoot = Join-Path $RepoRoot "onnx_dinov2\model.onnx"
if (-not (Test-Path $onnxRuntime)) {
    if (Test-Path $onnxRoot) {
        Write-Host "Copying onnx_dinov2/ into app/..."
        Copy-Item -Recurse -Force (Join-Path $RepoRoot "onnx_dinov2") (Join-Path $AppDir "onnx_dinov2")
    } else {
        Write-Host "Building ONNX model (this may take a few minutes)..."
        python (Join-Path $AppDir "build_onnx.py")
        Copy-Item -Recurse -Force (Join-Path $RepoRoot "onnx_dinov2") (Join-Path $AppDir "onnx_dinov2")
    }
}

foreach ($file in @("mtg_cards.db", "mtg_cards.index")) {
    $target = Join-Path $AppDir $file
    $source = Join-Path $RepoRoot $file
    if (-not (Test-Path $target) -and (Test-Path $source)) {
        Write-Host "Copying $file to app/..."
        Copy-Item -Force $source $target
    }
}

$weights = Join-Path $AppDir "mtg_yolo_best.pt"
if (-not (Test-Path $weights)) {
    $trained = Join-Path $RepoRoot "model_training\runs\detect\mtg_card\weights\best.pt"
    if (Test-Path $trained) {
        Write-Host "Copying trained YOLO weights to app/mtg_yolo_best.pt..."
        Copy-Item -Force $trained $weights
    }
}

if (-not (Test-Path (Join-Path $RepoRoot ".env")) -and (Test-Path (Join-Path $RepoRoot ".env.example"))) {
    Write-Host "Creating .env from .env.example..."
    Copy-Item (Join-Path $RepoRoot ".env.example") (Join-Path $RepoRoot ".env")
}

Write-Host "`nRunning artifact check..."
python (Join-Path $RepoRoot "scripts\check_artifacts.py")
exit $LASTEXITCODE
