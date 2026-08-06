"""Video crop extraction with track-based and embedding deduplication."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from crop_export import (
    CROPS_DIR,
    CropRecord,
    build_crops_zip,
    crop_record_to_manifest_entry,
    encode_crop_jpeg,
    make_crop_filename,
)
from video_scan import TrackManager, detect_card_boxes

IDENTIFY_BATCH_SIZE = 32


def _resolve_output_dir(output_dir: str | None) -> Path:
    if output_dir:
        path = Path(output_dir)
    else:
        path = Path(tempfile.mkdtemp(prefix="mtg_video_crops_"))
    path.mkdir(parents=True, exist_ok=True)
    (path / CROPS_DIR).mkdir(parents=True, exist_ok=True)
    return path


def _read_frame_at_sec(cap: cv2.VideoCapture, timestamp_sec: float) -> tuple[bool, object]:
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000.0)
    return cap.read()


def _save_crop_jpeg(
    crop_rgb,
    *,
    output_dir: Path,
    filename: str,
    jpeg_quality: int,
) -> None:
    crop_path = output_dir / filename
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    crop_path.write_bytes(encode_crop_jpeg(crop_rgb, quality=jpeg_quality))


def _min_l2_distance(embedding: np.ndarray, seen_embeddings: list[np.ndarray]) -> float | None:
    if not seen_embeddings:
        return None

    stacked = np.stack(seen_embeddings).astype("float32")
    diff = stacked - embedding.astype("float32")
    distances = np.linalg.norm(diff, axis=1)
    return float(distances.min())


def _is_duplicate_embedding(
    embedding: np.ndarray,
    seen_embeddings: list[np.ndarray],
    threshold: float,
) -> bool:
    min_distance = _min_l2_distance(embedding, seen_embeddings)
    return min_distance is not None and min_distance < threshold


def _load_crop_records(manifest: list[dict], output_dir: Path) -> list[CropRecord]:
    records: list[CropRecord] = []
    for entry in manifest:
        crop_path = output_dir / entry["filename"]
        bgr = cv2.imread(str(crop_path))
        if bgr is None:
            raise ValueError(f"Failed to read saved crop image: {crop_path}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        records.append(
            CropRecord(
                image_rgb=rgb,
                timestamp_sec=entry["timestamp_sec"],
                track_id=entry["track_id"],
                box=entry["box"],
                filename=entry["filename"],
                name=entry.get("name"),
                set=entry.get("set"),
                dist=entry.get("dist"),
                identified=entry.get("identified"),
            )
        )
    return records


def _identify_manifest_entries(
    manifest: list[dict],
    *,
    output_dir: Path,
    identify_crops: Callable[[list, list, float], list[dict]],
    dist_threshold: float,
    batch_size: int = IDENTIFY_BATCH_SIZE,
) -> None:
    for start in range(0, len(manifest), batch_size):
        batch = manifest[start : start + batch_size]
        crop_list: list[np.ndarray] = []
        boxes: list[list[int]] = []

        for entry in batch:
            crop_path = output_dir / entry["filename"]
            bgr = cv2.imread(str(crop_path))
            if bgr is None:
                raise ValueError(f"Failed to read saved crop image: {crop_path}")
            crop_list.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            boxes.append(entry["box"])

        detections = identify_crops(crop_list, boxes, dist_threshold)
        for entry, detection in zip(batch, detections):
            entry["dist"] = round(float(detection["dist"]), 3)
            entry["identified"] = bool(detection["identified"])
            if detection["identified"]:
                entry["name"] = detection["name"]
                entry["set"] = detection["set"]


def process_video_crops(
    path: str,
    *,
    yolo,
    conf: float = 0.75,
    sample_interval_sec: float = 5.0,
    max_samples: int = 0,
    track_expiry_samples: int = 3,
    output_dir: str | None = None,
    jpeg_quality: int = 90,
    get_embedding: Callable[[list], np.ndarray] | None = None,
    embedding_dedup_threshold: float = 100,
    identify: bool = False,
    identify_crops: Callable[[list, list, float], list[dict]] | None = None,
    dist_threshold: float = 300,
    on_progress: Callable[[dict], None] | None = None,
) -> dict:
    """Sample a video on a time interval, detect cards, and export deduplicated crops."""
    if sample_interval_sec <= 0:
        raise ValueError("sample_interval_sec must be > 0.")
    if max_samples < 0:
        raise ValueError("max_samples must be >= 0.")
    if track_expiry_samples < 1:
        raise ValueError("track_expiry_samples must be >= 1.")
    if embedding_dedup_threshold <= 0:
        raise ValueError("embedding_dedup_threshold must be > 0.")
    if identify and identify_crops is None:
        raise ValueError("identify_crops must be provided when identify=True.")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = round(total_frames / fps, 3) if total_frames > 0 else 0.0

    output_path = _resolve_output_dir(output_dir)
    track_manager = TrackManager()
    manifest: list[dict] = []
    seen_embeddings: list[np.ndarray] = []
    processed_samples = 0
    crop_count = 0
    skipped_embedding_duplicates = 0
    timestamp_sec = 0.0

    while True:
        if max_samples > 0 and processed_samples >= max_samples:
            break
        if duration_sec > 0 and timestamp_sec > duration_sec:
            break

        ok, frame = _read_frame_at_sec(cap, timestamp_sec)
        if not ok or frame is None:
            break

        track_manager.expire_stale(processed_samples, expiry_frames=track_expiry_samples)
        boxes, crops, _ = detect_card_boxes(frame, yolo, conf=conf, save_yolo=False)

        used_tracks: set[int] = set()
        for box, crop in zip(boxes, crops):
            track = track_manager.match_box(box)
            if track is not None and track.track_id in used_tracks:
                track = None

            if track is None:
                track = track_manager.create_track(box, processed_samples, timestamp_sec)
            else:
                used_tracks.add(track.track_id)

            track.last_box = box
            track.last_seen_frame = processed_samples

            if track.crop_saved:
                continue

            if get_embedding is not None:
                embedding = get_embedding([crop])[0]
                if _is_duplicate_embedding(embedding, seen_embeddings, embedding_dedup_threshold):
                    skipped_embedding_duplicates += 1
                    track.crop_saved = True
                    continue
                seen_embeddings.append(embedding)

            crop_count += 1
            filename = make_crop_filename(crop_count, timestamp_sec)
            _save_crop_jpeg(
                crop,
                output_dir=output_path,
                filename=filename,
                jpeg_quality=jpeg_quality,
            )

            record = CropRecord(
                image_rgb=crop,
                timestamp_sec=timestamp_sec,
                track_id=track.track_id,
                box=box,
                filename=filename,
            )
            manifest.append(crop_record_to_manifest_entry(record, index=crop_count, filename=filename))

            track.crop_saved = True
            track.saved_crop_index = crop_count

        processed_samples += 1
        timestamp_sec = round(timestamp_sec + sample_interval_sec, 3)

        if on_progress is not None:
            on_progress(
                {
                    "processed_samples": processed_samples,
                    "crops_saved": crop_count,
                    "duration_sec": duration_sec,
                    "skipped_embedding_duplicates": skipped_embedding_duplicates,
                }
            )

    cap.release()

    if identify and manifest and identify_crops is not None:
        _identify_manifest_entries(
            manifest,
            output_dir=output_path,
            identify_crops=identify_crops,
            dist_threshold=dist_threshold,
        )

    zip_path: Path | None = None
    if manifest:
        crop_records = _load_crop_records(manifest, output_path)
        zip_bytes = build_crops_zip(crop_records, jpeg_quality=jpeg_quality)
        zip_path = output_path / "video_crops.zip"
        zip_path.write_bytes(zip_bytes)

    return {
        "video": {
            "duration_sec": duration_sec,
            "fps": round(fps, 3),
            "processed_samples": processed_samples,
            "sample_interval_sec": sample_interval_sec,
        },
        "crop_count": crop_count,
        "skipped_embedding_duplicates": skipped_embedding_duplicates,
        "identified": identify,
        "manifest": manifest,
        "output_dir": str(output_path),
        "zip_path": str(zip_path) if zip_path else None,
    }
