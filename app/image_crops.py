"""Batch image crop extraction for multi-photo uploads."""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable

import cv2
import numpy as np

from crop_export import (
    CropRecord,
    build_crops_zip,
    crop_record_to_manifest_entry,
    make_image_crop_filename,
)
from video_scan import detect_card_boxes

IDENTIFY_BATCH_SIZE = 32
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_MAX_IMAGES = 50
DEFAULT_MAX_ZIP_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024


@dataclass
class ImageCropsResult:
    zip_bytes: bytes
    manifest: list[dict]
    errors: list[dict]
    crop_count: int
    images_processed: int


def load_image_file(path: str | Path) -> tuple[str, np.ndarray]:
    """Load an image from disk and return (source_name, bgr frame)."""
    image_path = Path(path)
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"Could not read image file: {image_path}")

    return image_path.name, frame


def load_image_files(paths: list[str | Path]) -> list[tuple[str, np.ndarray]]:
    images: list[tuple[str, np.ndarray]] = []
    for path in paths:
        images.append(load_image_file(path))
    return images


def _normalize_zip_entry_name(name: str) -> str:
    normalized = name.replace("\\", "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _is_safe_zip_entry(name: str) -> bool:
    normalized = _normalize_zip_entry_name(name)
    if not normalized:
        return False

    if normalized.startswith("/") or (len(normalized) > 1 and normalized[1] == ":"):
        return False

    for part in normalized.split("/"):
        if part in {"", ".."}:
            return False

    return True


def _should_skip_zip_entry(name: str) -> bool:
    normalized = _normalize_zip_entry_name(name)
    if not normalized or normalized.endswith("/"):
        return True

    parts = PurePosixPath(normalized).parts
    if parts and parts[0] == "__MACOSX":
        return True

    basename = PurePosixPath(normalized).name
    return basename.startswith(".")


def _is_image_zip_entry(name: str) -> bool:
    return PurePosixPath(name).suffix.lower() in IMAGE_EXTENSIONS


def extract_images_from_zip(
    zip_bytes: bytes,
    *,
    max_images: int = DEFAULT_MAX_IMAGES,
    max_uncompressed_bytes: int = DEFAULT_MAX_ZIP_UNCOMPRESSED_BYTES,
) -> list[tuple[str, bytes]]:
    """Extract supported image entries from a ZIP archive with basic safety checks."""
    if max_images < 1:
        raise ValueError("max_images must be >= 1.")
    if max_uncompressed_bytes < 1:
        raise ValueError("max_uncompressed_bytes must be >= 1.")

    extracted: list[tuple[str, bytes]] = []
    total_uncompressed = 0

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        for info in archive.infolist():
            entry_name = info.filename
            if _should_skip_zip_entry(entry_name):
                continue

            if not _is_safe_zip_entry(entry_name):
                raise ValueError(f"Unsafe path in ZIP archive: {entry_name}")

            if not _is_image_zip_entry(entry_name):
                continue

            declared_size = info.file_size
            if declared_size < 0:
                raise ValueError(f"Invalid entry size in ZIP archive: {entry_name}")

            if total_uncompressed + declared_size > max_uncompressed_bytes:
                raise ValueError("ZIP archive exceeds maximum uncompressed size.")

            if len(extracted) >= max_images:
                raise ValueError(f"ZIP archive contains more than {max_images} images.")

            source_name = _normalize_zip_entry_name(entry_name)
            raw = archive.read(entry_name)
            total_uncompressed += len(raw)
            if total_uncompressed > max_uncompressed_bytes:
                raise ValueError("ZIP archive exceeds maximum uncompressed size.")

            extracted.append((source_name, raw))

    if not extracted:
        raise ValueError("ZIP archive does not contain any supported image files.")

    return extracted


def decode_image_bytes(source_name: str, raw: bytes) -> np.ndarray | None:
    """Decode raw image bytes to a BGR frame, or return None when decoding fails."""
    if not raw:
        return None

    buffer = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def _decode_uploaded_images(
    uploaded_files: list[tuple[str, bytes]],
) -> tuple[list[tuple[str, np.ndarray]], list[dict]]:
    images: list[tuple[str, np.ndarray]] = []
    errors: list[dict] = []

    for source_name, raw in uploaded_files:
        frame = decode_image_bytes(source_name, raw)
        if frame is None:
            errors.append(
                {
                    "source_image": source_name,
                    "error": "undecodable_image",
                }
            )
            continue
        images.append((source_name, frame))

    return images, errors


def process_images_crops_from_uploads(
    *,
    image_files: list[tuple[str, bytes]] | None = None,
    zip_file: bytes | None = None,
    yolo,
    conf: float = 0.75,
    identify: bool = False,
    identify_crops: Callable[[list, list, float], list[dict]] | None = None,
    dist_threshold: float = 300,
    jpeg_quality: int = 90,
    max_images: int = DEFAULT_MAX_IMAGES,
    max_uncompressed_bytes: int = DEFAULT_MAX_ZIP_UNCOMPRESSED_BYTES,
) -> ImageCropsResult:
    """Process either raw image uploads or a ZIP archive of images into a crop ZIP."""
    has_images = bool(image_files)
    has_zip = zip_file is not None and len(zip_file) > 0

    if has_images == has_zip:
        raise ValueError("Provide either image_files or zip_file, but not both.")

    decode_errors: list[dict] = []

    if has_zip:
        uploaded_files = extract_images_from_zip(
            zip_file,
            max_images=max_images,
            max_uncompressed_bytes=max_uncompressed_bytes,
        )
    else:
        uploaded_files = image_files or []

    images, decode_errors = _decode_uploaded_images(uploaded_files)
    if not images:
        raise ValueError("No decodable images were found in the upload.")

    return process_images_crops(
        images,
        yolo=yolo,
        conf=conf,
        identify=identify,
        identify_crops=identify_crops,
        dist_threshold=dist_threshold,
        jpeg_quality=jpeg_quality,
        preflight_errors=decode_errors,
    )


def _identify_crop_records(
    records: list[CropRecord],
    *,
    identify_crops: Callable[[list, list, float], list[dict]],
    dist_threshold: float,
    batch_size: int = IDENTIFY_BATCH_SIZE,
) -> None:
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        crop_list = [record.image_rgb for record in batch]
        boxes = [record.box for record in batch]
        detections = identify_crops(crop_list, boxes, dist_threshold)

        for record, detection in zip(batch, detections):
            record.dist = float(detection["dist"])
            record.identified = bool(detection["identified"])
            if detection["identified"]:
                record.name = detection["name"]
                record.set = detection["set"]


def process_images_crops(
    images: list[tuple[str, np.ndarray]],
    *,
    yolo,
    conf: float = 0.75,
    identify: bool = False,
    identify_crops: Callable[[list, list, float], list[dict]] | None = None,
    dist_threshold: float = 300,
    jpeg_quality: int = 90,
    preflight_errors: list[dict] | None = None,
) -> ImageCropsResult:
    """Detect cards across multiple images and pack warped crops into a ZIP archive."""
    if not images:
        raise ValueError("At least one image is required.")
    if identify and identify_crops is None:
        raise ValueError("identify_crops must be provided when identify=True.")

    crop_records: list[CropRecord] = []
    errors: list[dict] = list(preflight_errors or [])

    for source_name, frame in images:
        boxes, crops, _ = detect_card_boxes(frame, yolo, conf=conf, save_yolo=False)
        if not crops:
            errors.append(
                {
                    "source_image": source_name,
                    "error": "no_cards_detected",
                }
            )
            continue

        for crop_index_within_image, (box, crop) in enumerate(zip(boxes, crops), start=1):
            filename = make_image_crop_filename(source_name, crop_index_within_image)
            record = CropRecord(
                image_rgb=crop,
                timestamp_sec=0.0,
                track_id=0,
                box=box,
                filename=filename,
                extra={"source_image": source_name},
            )
            crop_records.append(record)

    if not crop_records:
        raise ValueError("No card crops were detected in the provided images.")

    if identify and identify_crops is not None:
        _identify_crop_records(
            crop_records,
            identify_crops=identify_crops,
            dist_threshold=dist_threshold,
        )

    manifest = [
        crop_record_to_manifest_entry(record, index=index, filename=record.filename or "")
        for index, record in enumerate(crop_records, start=1)
    ]

    zip_bytes = build_crops_zip(crop_records, jpeg_quality=jpeg_quality, errors=errors or None)

    return ImageCropsResult(
        zip_bytes=zip_bytes,
        manifest=manifest,
        errors=errors,
        crop_count=len(crop_records),
        images_processed=len(images),
    )
