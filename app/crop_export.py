"""Build ZIP archives of detected card crops from camera footage."""

from __future__ import annotations

import io
import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import PurePosixPath

import cv2
import numpy as np

CROPS_DIR = "crops"


def sanitize_filename(value: str) -> str:
    cleaned = re.sub(r"[^\w\- ]+", "", value).strip().replace(" ", "_")
    return cleaned or "card"


def make_crop_filename(index: int, timestamp_sec: float) -> str:
    """Return a ZIP entry path like crops/crop_0001_t042s.jpg."""
    return f"{CROPS_DIR}/crop_{index:04d}_t{int(round(timestamp_sec)):03d}s.jpg"


def source_name_stem(source_name: str) -> str:
    """Return a sanitized filename stem from an image path or archive entry."""
    stem = PurePosixPath(source_name.replace("\\", "/")).stem
    return sanitize_filename(stem) or "image"


def make_image_crop_filename(source_name: str, crop_index: int) -> str:
    """Return a ZIP entry path like crops/photo1_crop_001.jpg."""
    stem = source_name_stem(source_name)
    return f"{CROPS_DIR}/{stem}_crop_{crop_index:03d}.jpg"


@dataclass
class CropRecord:
    image_rgb: np.ndarray
    timestamp_sec: float
    track_id: int
    box: list[int]
    filename: str | None = None
    name: str | None = None
    set: str | None = None
    dist: float | None = None
    identified: bool | None = None
    extra: dict = field(default_factory=dict)

    def resolved_filename(self, index: int) -> str:
        return self.filename or make_crop_filename(index, self.timestamp_sec)

    def resolved_image_filename(self, source_name: str, crop_index: int) -> str:
        return self.filename or make_image_crop_filename(source_name, crop_index)


def encode_crop_jpeg(rgb: np.ndarray, quality: int = 90) -> bytes:
    """Encode an RGB crop as JPEG bytes."""
    if rgb.size == 0:
        raise ValueError("Cannot encode an empty crop image.")

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("Failed to encode crop as JPEG.")
    return encoded.tobytes()


def crop_record_to_manifest_entry(record: CropRecord, *, index: int, filename: str) -> dict:
    entry: dict = {
        "index": index,
        "filename": filename,
        "timestamp_sec": round(record.timestamp_sec, 3),
        "track_id": record.track_id,
        "box": record.box,
    }

    if record.name is not None:
        entry["name"] = record.name
    if record.set is not None:
        entry["set"] = record.set
    if record.dist is not None:
        entry["dist"] = round(record.dist, 3)
    if record.identified is not None:
        entry["identified"] = record.identified

    source_image = record.extra.get("source_image")
    if source_image is not None:
        entry["source_image"] = source_image

    for key, value in record.extra.items():
        if key == "source_image":
            continue
        entry[key] = value

    return entry


def build_crops_zip(
    crops: list[CropRecord],
    *,
    jpeg_quality: int = 90,
    errors: list[dict] | None = None,
) -> bytes:
    """Pack crop images and metadata into a ZIP archive."""
    if not crops:
        raise ValueError("At least one crop is required to build a ZIP archive.")

    buffer = io.BytesIO()
    manifest: list[dict] = []

    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, record in enumerate(crops, start=1):
            filename = record.resolved_filename(index)
            image_bytes = encode_crop_jpeg(record.image_rgb, quality=jpeg_quality)
            archive.writestr(filename, image_bytes)
            manifest.append(crop_record_to_manifest_entry(record, index=index, filename=filename))

        manifest_payload: dict = {"crop_count": len(manifest), "crops": manifest}
        if errors:
            manifest_payload["errors"] = errors

        archive.writestr(
            "manifest.json",
            json.dumps(manifest_payload, indent=2),
        )

    buffer.seek(0)
    return buffer.getvalue()
