"""Extract card crops from image files or a ZIP archive (Phase 2/3 CLI harness)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / "app"

sys.path.insert(0, str(APP_DIR))
os.chdir(APP_DIR)

from dotenv import load_dotenv  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from image_crops import (  # noqa: E402
    load_image_files,
    process_images_crops,
    process_images_crops_from_uploads,
)
from inference_runtime import load_identification_runtime  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract card crops from image files or a ZIP archive.",
    )
    parser.add_argument(
        "image_paths",
        nargs="*",
        help="One or more image files (PNG, JPG, JPEG, WEBP).",
    )
    parser.add_argument(
        "--zip",
        dest="zip_path",
        help="ZIP archive containing image files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where image_crops.zip will be written.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.75,
        help="YOLO detection confidence threshold (default: 0.75).",
    )
    parser.add_argument(
        "--identify",
        action="store_true",
        help="Identify crops via FAISS and add names to manifest.json.",
    )
    parser.add_argument(
        "--dist-threshold",
        type=float,
        default=300.0,
        help="Identification distance threshold (default: 300).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=50,
        help="Maximum number of images to extract from a ZIP (default: 50).",
    )
    parser.add_argument(
        "--weights",
        default=os.getenv("YOLO_WEIGHTS", "mtg_yolo_best.pt"),
        help="Path to YOLO weights (default: YOLO_WEIGHTS env or mtg_yolo_best.pt).",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()

    if bool(args.image_paths) == bool(args.zip_path):
        print("Provide either image file paths or --zip, but not both.", file=sys.stderr)
        return 1

    weights_path = Path(args.weights)
    if not weights_path.is_file():
        print(f"YOLO weights not found: {weights_path}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    identify_crops = None
    if args.identify:
        print("Loading identification runtime (ONNX + FAISS)...")
        runtime = load_identification_runtime(dist_threshold=args.dist_threshold)
        identify_crops = runtime.identify_crops

    print(f"Loading YOLO weights from {weights_path}...")
    yolo = YOLO(str(weights_path))

    if args.zip_path:
        zip_path = Path(args.zip_path)
        if not zip_path.is_file():
            print(f"ZIP file not found: {zip_path}", file=sys.stderr)
            return 1

        print(f"Processing ZIP archive {zip_path}...")
        result = process_images_crops_from_uploads(
            zip_file=zip_path.read_bytes(),
            yolo=yolo,
            conf=args.conf,
            identify=args.identify,
            identify_crops=identify_crops,
            dist_threshold=args.dist_threshold,
            max_images=args.max_images,
        )
    else:
        image_paths = [Path(path) for path in args.image_paths]
        for image_path in image_paths:
            if not image_path.is_file():
                print(f"Image file not found: {image_path}", file=sys.stderr)
                return 1

        print(f"Processing {len(image_paths)} image(s)...")
        images = load_image_files(image_paths)
        result = process_images_crops(
            images,
            yolo=yolo,
            conf=args.conf,
            identify=args.identify,
            identify_crops=identify_crops,
            dist_threshold=args.dist_threshold,
        )

    zip_path = output_dir / "image_crops.zip"
    zip_path.write_bytes(result.zip_bytes)

    summary = {
        "crop_count": result.crop_count,
        "images_processed": result.images_processed,
        "errors": result.errors,
        "zip_path": str(zip_path),
        "manifest": result.manifest,
    }
    print(json.dumps(summary, indent=2))
    print(f"\nZIP written to: {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
