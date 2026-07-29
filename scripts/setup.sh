#!/usr/bin/env bash
# One-time setup for local MTG Card Scanner development.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="$REPO_ROOT/app"

echo "MTG Card Scanner setup"
echo "Repository: $REPO_ROOT"

cd "$REPO_ROOT"

echo
echo "Installing Python dependencies..."
python3 -m pip install -r requirements.txt

if [[ ! -f "$APP_DIR/onnx_dinov2/model.onnx" ]]; then
  if [[ -f "$REPO_ROOT/onnx_dinov2/model.onnx" ]]; then
    echo "Copying onnx_dinov2/ into app/..."
    cp -R "$REPO_ROOT/onnx_dinov2" "$APP_DIR/onnx_dinov2"
  else
    echo "Building ONNX model (this may take a few minutes)..."
    python3 "$APP_DIR/build_onnx.py"
    cp -R "$REPO_ROOT/onnx_dinov2" "$APP_DIR/onnx_dinov2"
  fi
fi

for file in mtg_cards.db mtg_cards.index; do
  if [[ ! -f "$APP_DIR/$file" && -f "$REPO_ROOT/$file" ]]; then
    echo "Copying $file to app/..."
    cp "$REPO_ROOT/$file" "$APP_DIR/$file"
  fi
done

if [[ ! -f "$APP_DIR/mtg_yolo_best.pt" && -f "$REPO_ROOT/model_training/runs/detect/mtg_card/weights/best.pt" ]]; then
  echo "Copying trained YOLO weights to app/mtg_yolo_best.pt..."
  cp "$REPO_ROOT/model_training/runs/detect/mtg_card/weights/best.pt" "$APP_DIR/mtg_yolo_best.pt"
fi

if [[ ! -f "$REPO_ROOT/.env" && -f "$REPO_ROOT/.env.example" ]]; then
  echo "Creating .env from .env.example..."
  cp "$REPO_ROOT/.env.example" "$REPO_ROOT/.env"
fi

echo
echo "Running artifact check..."
python3 "$REPO_ROOT/scripts/check_artifacts.py"
