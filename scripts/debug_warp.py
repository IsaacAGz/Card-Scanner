"""Save perspective-warp debug previews for a test image."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / "app"

sys.path.insert(0, str(APP_DIR))
os.chdir(APP_DIR)

from card_warp import (  # noqa: E402
    find_card_quad,
    get_padded_crop,
    order_corners,
    rectify_crop,
    warp_card_bgr,
)


def draw_box(image: np.ndarray, box: list[int], color: tuple[int, int, int], label: str) -> np.ndarray:
    output = image.copy()
    x1, y1, x2, y2 = box
    cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        output,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )
    return output


def draw_quad(crop_bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
    output = crop_bgr.copy()
    points = order_corners(quad).astype(int).reshape(-1, 1, 2)
    cv2.polylines(output, [points], isClosed=True, color=(0, 255, 0), thickness=2)
    for idx, (x, y) in enumerate(points.reshape(-1, 2)):
        cv2.circle(output, (int(x), int(y)), 4, (0, 0, 255), -1)
        cv2.putText(
            output,
            str(idx),
            (int(x) + 4, int(y) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return output


def save_warp_previews(
    image_path: Path,
    output_dir: Path,
    conf: float,
    yolo_weights: str,
) -> int:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"Could not read image: {image_path}")

    if not Path(yolo_weights).exists():
        raise FileNotFoundError(f"YOLO weights not found: {yolo_weights}")

    yolo = YOLO(yolo_weights)
    results = yolo(frame, conf=conf, save=False)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotated = frame.copy()
    saved_count = 0

    if not results or results[0].boxes is None:
        cv2.imwrite(str(output_dir / "00_no_detections.jpg"), annotated)
        print("No cards detected.")
        return 0

    height, width, _ = frame.shape
    for idx, prediction in enumerate(results[0].boxes, start=1):
        xyxy = prediction.xyxy[0].tolist()
        box = [
            max(0, int(xyxy[0])),
            max(0, int(xyxy[1])),
            min(width, int(xyxy[2])),
            min(height, int(xyxy[3])),
        ]

        rgb_crop, was_warped = rectify_crop(frame, box)
        padded_crop, _ = get_padded_crop(frame, box)
        quad = find_card_quad(padded_crop)

        prefix = f"{idx:02d}"
        status = "warped" if was_warped else "fallback"

        cv2.imwrite(str(output_dir / f"{prefix}_padded_crop.jpg"), padded_crop)
        cv2.imwrite(str(output_dir / f"{prefix}_{status}_embedding_input.jpg"), cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2BGR))

        if quad is not None:
            cv2.imwrite(str(output_dir / f"{prefix}_quad_overlay.jpg"), draw_quad(padded_crop, quad))
            cv2.imwrite(str(output_dir / f"{prefix}_warped.jpg"), warp_card_bgr(padded_crop, quad))

        annotated = draw_box(annotated, box, (0, 255, 0), f"#{idx} {status}")
        saved_count += 1
        print(f"[{idx}] box={box} status={status}")

    cv2.imwrite(str(output_dir / "00_annotated.jpg"), annotated)
    print(f"Saved {saved_count} detection preview set(s) to {output_dir}")
    return saved_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save YOLO + perspective-warp debug previews.")
    parser.add_argument("image_path", help="Path to a test image.")
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "debug_warp"),
        help="Directory for preview images.",
    )
    parser.add_argument("--conf", type=float, default=0.75, help="YOLO confidence threshold.")
    parser.add_argument(
        "--yolo-weights",
        default=os.getenv("YOLO_WEIGHTS", "mtg_yolo_best.pt"),
        help="Path to YOLO weights (relative to app/ by default).",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()

    try:
        save_warp_previews(
            image_path=Path(args.image_path),
            output_dir=Path(args.output_dir),
            conf=args.conf,
            yolo_weights=args.yolo_weights,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
