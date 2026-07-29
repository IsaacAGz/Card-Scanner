"""Upload a video, poll crop job status, and download the resulting ZIP."""

from __future__ import annotations

import argparse
import json
import sys
import time

import requests

DEFAULT_START_URL = "http://127.0.0.1:8000/scan/video/crops"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start and monitor a video crop extraction job.")
    parser.add_argument("video_path", help="Path to a video file.")
    parser.add_argument("--url", default=DEFAULT_START_URL, help="Crop job start endpoint URL.")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Seconds between status polls.")
    parser.add_argument("--timeout", type=float, default=3600.0, help="Max seconds to wait for completion.")
    parser.add_argument("--sample-interval-sec", type=float, default=5.0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.75)
    parser.add_argument("--identify", action="store_true")
    parser.add_argument("--no-embedding-dedup", action="store_true")
    parser.add_argument("--output", default="video_crops.zip", help="Path to save the downloaded ZIP.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    params = {
        "sample_interval_sec": args.sample_interval_sec,
        "max_samples": args.max_samples,
        "conf": args.conf,
        "identify": str(args.identify).lower(),
        "no_embedding_dedup": str(args.no_embedding_dedup).lower(),
    }

    with open(args.video_path, "rb") as video_file:
        response = requests.post(
            args.url,
            files={"file": (args.video_path, video_file, "video/mp4")},
            params=params,
            timeout=120,
        )

    response.raise_for_status()
    start_payload = response.json()
    job_id = start_payload["job_id"]
    base_url = args.url.rsplit("/scan/video/crops", 1)[0]
    status_url = f"{base_url}/scan/video/crops/{job_id}"
    download_url = f"{base_url}/scan/video/crops/{job_id}/download"

    print(json.dumps(start_payload, indent=2))

    deadline = time.time() + args.timeout
    while time.time() < deadline:
        status_response = requests.get(status_url, timeout=30)
        status_response.raise_for_status()
        status_payload = status_response.json()
        print(json.dumps(status_payload, indent=2))

        if status_payload["status"] == "completed":
            if status_payload.get("crop_count", 0) == 0:
                print("Job completed with no crops detected.")
                return 0

            download_response = requests.get(download_url, timeout=120)
            download_response.raise_for_status()
            with open(args.output, "wb") as output_file:
                output_file.write(download_response.content)
            print(f"ZIP saved to: {args.output}")
            return 0

        if status_payload["status"] == "failed":
            print(status_payload.get("error") or "Crop job failed.", file=sys.stderr)
            return 1

        time.sleep(args.poll_interval)

    print("Timed out waiting for crop job to complete.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
