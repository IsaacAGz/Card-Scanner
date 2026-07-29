"""Validate runtime artifacts required by the MTG Card Scanner API."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / "app"

ARTIFACTS = [
    {
        "name": "YOLO weights",
        "runtime": APP_DIR / "mtg_yolo_best.pt",
        "fallback": APP_DIR / "mtg_yolo_best.pt",
        "help": "Train with model_training/train.py, then copy runs/detect/mtg_card/weights/best.pt to app/mtg_yolo_best.pt",
    },
    {
        "name": "SQLite card database",
        "runtime": APP_DIR / "mtg_cards.db",
        "fallback": REPO_ROOT / "mtg_cards.db",
        "help": "Copy mtg_cards.db to app/, or run scripts/setup.ps1 / scripts/setup.sh",
    },
    {
        "name": "FAISS index",
        "runtime": APP_DIR / "mtg_cards.index",
        "fallback": REPO_ROOT / "mtg_cards.index",
        "help": "Build with model_training/create_index.py (local, gitignored) and copy to app/",
    },
    {
        "name": "DINOv2 ONNX model",
        "runtime": APP_DIR / "onnx_dinov2" / "model.onnx",
        "fallback": REPO_ROOT / "onnx_dinov2" / "model.onnx",
        "help": "Run: python app/build_onnx.py, then copy onnx_dinov2/ into app/ or run setup script",
    },
]


def resolve_path(runtime: Path, fallback: Path) -> tuple[Path, str]:
    if runtime.exists():
        return runtime, "ok"
    if fallback.exists() and fallback != runtime:
        return runtime, f"missing at {runtime.relative_to(REPO_ROOT)} (found at {fallback.relative_to(REPO_ROOT)})"
    return runtime, "missing"


def main() -> int:
    print(f"Repository root: {REPO_ROOT}")
    print(f"Expected API working directory: {APP_DIR}")
    print()

    missing = []
    for artifact in ARTIFACTS:
        runtime = artifact["runtime"]
        fallback = artifact["fallback"]
        path, status = resolve_path(runtime, fallback)

        if status == "ok":
            print(f"[OK] {artifact['name']}: {path.relative_to(REPO_ROOT)}")
            continue

        missing.append(artifact)
        print(f"[MISSING] {artifact['name']}: {status}")
        print(f"         -> {artifact['help']}")

    print()
    if missing:
        print("Artifact check failed.")
        print("Run scripts/setup.ps1 (Windows) or scripts/setup.sh (macOS/Linux), then re-run this check.")
        return 1

    print("All required artifacts are present under app/.")
    print("Start the API with:")
    print("  cd app")
    print("  uvicorn main:app --host 0.0.0.0 --port 8000")
    return 0


if __name__ == "__main__":
    sys.exit(main())
