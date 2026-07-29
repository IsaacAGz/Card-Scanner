import argparse
import json
import sys

import requests

DEFAULT_URL = "http://127.0.0.1:8000/scan/video"


def scan_video(video_path: str, base_url: str, frame_stride: int, max_frames: int) -> None:
    with open(video_path, "rb") as video_file:
        response = requests.post(
            base_url,
            files={"file": (video_path, video_file, "video/mp4")},
            params={"frame_stride": frame_stride, "max_frames": max_frames},
            timeout=600,
        )

    response.raise_for_status()
    print(json.dumps(response.json(), indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a video to the MTG card scanner API.")
    parser.add_argument("video_path", help="Path to a video file.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Video scan endpoint URL.")
    parser.add_argument("--frame-stride", type=int, default=5, help="Process every Nth frame.")
    parser.add_argument("--max-frames", type=int, default=300, help="Maximum sampled frames.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        scan_video(args.video_path, args.url, args.frame_stride, args.max_frames)
    except requests.RequestException as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        sys.exit(1)
