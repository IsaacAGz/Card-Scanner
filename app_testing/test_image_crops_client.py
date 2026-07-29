"""Upload images or a ZIP archive and download the resulting crop ZIP."""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import requests

DEFAULT_URL = "http://127.0.0.1:8000/scan/images/crops-zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Request image crop extraction and save the ZIP response.")
    parser.add_argument(
        "--images",
        nargs="+",
        metavar="PATH",
        help="One or more image files to upload.",
    )
    parser.add_argument(
        "--zip",
        dest="zip_path",
        help="ZIP archive containing image files.",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="Image crop endpoint URL.",
    )
    parser.add_argument("--conf", type=float, default=0.75)
    parser.add_argument("--identify", action="store_true")
    parser.add_argument("--dist-threshold", type=float, default=300.0)
    parser.add_argument(
        "-o",
        "--output",
        default="image_crops.zip",
        help="Path to save the downloaded ZIP.",
    )
    parser.add_argument(
        "--print-manifest",
        action="store_true",
        help="Print manifest.json from the downloaded ZIP.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if bool(args.images) == bool(args.zip_path):
        print("Provide either --images or --zip, but not both.", file=sys.stderr)
        return 1

    params = {
        "conf": args.conf,
        "identify": str(args.identify).lower(),
        "dist_threshold": args.dist_threshold,
    }

    if args.zip_path:
        zip_path = Path(args.zip_path)
        if not zip_path.is_file():
            print(f"ZIP file not found: {zip_path}", file=sys.stderr)
            return 1

        with zip_path.open("rb") as zip_file:
            response = requests.post(
                args.url,
                files={"file": (zip_path.name, zip_file, "application/zip")},
                params=params,
                timeout=300,
            )
    else:
        image_paths = [Path(path) for path in args.images]
        for image_path in image_paths:
            if not image_path.is_file():
                print(f"Image file not found: {image_path}", file=sys.stderr)
                return 1

        multipart_files = []
        for image_path in image_paths:
            multipart_files.append(
                ("files", (image_path.name, image_path.open("rb"), "application/octet-stream"))
            )

        try:
            response = requests.post(
                args.url,
                files=multipart_files,
                params=params,
                timeout=300,
            )
        finally:
            for _, (_, file_handle, _) in multipart_files:
                file_handle.close()

    if not response.ok:
        detail = response.text
        try:
            detail = response.json().get("detail", detail)
        except ValueError:
            pass
        print(f"Request failed ({response.status_code}): {detail}", file=sys.stderr)
        return 1

    output_path = Path(args.output)
    output_path.write_bytes(response.content)
    print(f"ZIP saved to: {output_path}")

    if args.print_manifest:
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            manifest = json.loads(archive.read("manifest.json"))
        print(json.dumps(manifest, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
