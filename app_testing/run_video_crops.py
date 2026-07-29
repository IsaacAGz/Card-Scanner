"""Extract deduplicated card crops from a video file."""

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
from inference_runtime import load_embedding_runtime, load_identification_runtime  # noqa: E402
from video_crops import process_video_crops  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract deduplicated card crops from a video and write a ZIP archive.",
    )
    parser.add_argument("video_path", help="Path to an input video file.")
    parser.add_argument(
        "--output-dir",
        help="Directory for crop JPEGs and video_crops.zip (default: temp dir).",
    )
    parser.add_argument(
        "--sample-interval-sec",
        type=float,
        default=5.0,
        help="Seconds between sampled frames (default: 5).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum number of sample points to process (0 = no cap).",
    )
    parser.add_argument(
        "--track-expiry-samples",
        type=int,
        default=3,
        help="Drop tracks after this many missed sample points (default: 3).",
    )
    parser.add_argument(
        "--embedding-dedup-threshold",
        type=float,
        default=100.0,
        help="Skip crops visually similar to saved crops below this L2 distance (default: 100).",
    )
    parser.add_argument(
        "--no-embedding-dedup",
        action="store_true",
        help="Disable visual embedding deduplication (track dedup only).",
    )
    parser.add_argument(
        "--identify",
        action="store_true",
        help="Identify saved crops via FAISS and add names to manifest.json.",
    )
    parser.add_argument(
        "--dist-threshold",
        type=float,
        default=300.0,
        help="Identification distance threshold (default: 300).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.75,
        help="YOLO detection confidence threshold (default: 0.75).",
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

    video_path = Path(args.video_path)
    if not video_path.is_file():
        print(f"Video file not found: {video_path}", file=sys.stderr)
        return 1

    weights_path = Path(args.weights)
    if not weights_path.is_file():
        print(f"YOLO weights not found: {weights_path}", file=sys.stderr)
        return 1

    get_embedding = None
    identify_crops = None

    if args.identify:
        print("Loading identification runtime (ONNX + FAISS)...")
        runtime = load_identification_runtime(dist_threshold=args.dist_threshold)
        get_embedding = runtime.get_embedding
        identify_crops = runtime.identify_crops
    elif not args.no_embedding_dedup:
        print("Loading embedding runtime for visual deduplication...")
        runtime = load_embedding_runtime()
        get_embedding = runtime.get_embedding

    print(f"Loading YOLO weights from {weights_path}...")
    yolo = YOLO(str(weights_path))

    print(f"Processing {video_path}...")
    result = process_video_crops(
        str(video_path),
        yolo=yolo,
        conf=args.conf,
        sample_interval_sec=args.sample_interval_sec,
        max_samples=args.max_samples,
        track_expiry_samples=args.track_expiry_samples,
        output_dir=args.output_dir,
        get_embedding=get_embedding,
        embedding_dedup_threshold=args.embedding_dedup_threshold,
        identify=args.identify,
        identify_crops=identify_crops,
        dist_threshold=args.dist_threshold,
    )

    print(json.dumps(result, indent=2))
    if result["zip_path"]:
        print(f"\nZIP written to: {result['zip_path']}")
    else:
        print("\nNo crops detected; ZIP was not created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
