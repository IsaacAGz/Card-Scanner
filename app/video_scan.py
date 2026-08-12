"""Video scanning with track-based deduplication."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from card_warp import rectify_crop

DIST_THRESHOLD = 300
TRACK_IOU_THRESHOLD = 0.5
TRACK_EXPIRY_FRAMES = 15


def box_iou(box_a: list[int], box_b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


@dataclass
class Track:
    track_id: int
    last_box: list[int]
    last_seen_frame: int
    card: dict | None = None
    first_seen_sec: float = 0.0
    crop_saved: bool = False
    saved_crop_index: int | None = None


@dataclass
class TrackManager:
    tracks: list[Track] = field(default_factory=list)
    next_track_id: int = 0
    unique_cards: dict[tuple[str, str], dict] = field(default_factory=dict)

    def expire_stale(self, current_frame: int, expiry_frames: int | None = None) -> None:
        limit = expiry_frames if expiry_frames is not None else TRACK_EXPIRY_FRAMES
        self.tracks = [
            track
            for track in self.tracks
            if current_frame - track.last_seen_frame <= limit
        ]

    def match_box(self, box: list[int]) -> Track | None:
        best_track = None
        best_iou = TRACK_IOU_THRESHOLD
        for track in self.tracks:
            iou = box_iou(box, track.last_box)
            if iou > best_iou:
                best_iou = iou
                best_track = track
        return best_track

    def create_track(self, box: list[int], frame_idx: int, timestamp_sec: float) -> Track:
        track = Track(
            track_id=self.next_track_id,
            last_box=box,
            last_seen_frame=frame_idx,
            first_seen_sec=timestamp_sec,
        )
        self.next_track_id += 1
        self.tracks.append(track)
        return track

    def record_identified_card(self, track: Track, card: dict, timestamp_sec: float) -> None:
        track.card = card
        key = (card["name"], card["set"])
        if key not in self.unique_cards:
            self.unique_cards[key] = {
                "name": card["name"],
                "set": card["set"],
                "best_dist": card["dist"],
                "first_seen_sec": track.first_seen_sec,
                "last_seen_sec": timestamp_sec,
            }
            return

        existing = self.unique_cards[key]
        existing["best_dist"] = min(existing["best_dist"], card["dist"])
        existing["first_seen_sec"] = min(existing["first_seen_sec"], track.first_seen_sec)
        existing["last_seen_sec"] = max(existing["last_seen_sec"], timestamp_sec)


def detect_card_boxes(frame, yolo, conf: float = 0.75, save_yolo: bool = False) -> tuple[list[list[int]], list[np.ndarray], list[bool]]:
    """
        Does:
            Sends the frame to yolo model to detect cards, then builds 
            boxes using yolo coordinate predictions, and rgb_crops and was_warped from rectify_crops().

            was_warped is used to build warped_flags.
        
        Args:
            frame: 3D Numpy Array (BGR)
            yolo: yolo model loaded with mtg_yolo_best.pt
            conf: float
            save_yolo: bool

        Uses:
            rectify_crop()


        Returns:
            boxes: list[list[int]]
            crops: list[np.ndarrar]
            warped_flags: list[bool]

    """
    results = yolo(frame, save=save_yolo, conf=conf)
    boxes: list[list[int]] = []
    crops: list[np.ndarray] = []
    warped_flags: list[bool] = []

    height, width, _ = frame.shape
    if not results or results[0].boxes is None:
        return boxes, crops, warped_flags

    for prediction in results[0].boxes:
        xyxy = prediction.xyxy[0].tolist()
        xmin = max(0, int(xyxy[0]))
        ymin = max(0, int(xyxy[1]))
        xmax = min(width, int(xyxy[2]))
        ymax = min(height, int(xyxy[3]))

        box = [xmin, ymin, xmax, ymax]
        rgb_crop, was_warped = rectify_crop(frame, box)
        if rgb_crop.size == 0:
            continue

        boxes.append(box)
        crops.append(rgb_crop)
        warped_flags.append(was_warped)

    return boxes, crops, warped_flags


def process_video(
    path: str,
    *,
    yolo,
    identify_crops: Callable[[list, list, float], list[dict]],
    frame_stride: int = 5,
    max_frames: int = 300,
    conf: float = 0.75,
    dist_threshold: float = DIST_THRESHOLD,
) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    track_manager = TrackManager()
    frame_idx = 0
    processed_count = 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue

        timestamp_sec = round(frame_idx / fps, 3)
        boxes, crops, _ = detect_card_boxes(frame, yolo, conf=conf, save_yolo=False)
        track_manager.expire_stale(processed_count)

        pending_boxes: list[list[int]] = []
        pending_crops: list[np.ndarray] = []
        pending_tracks: list[Track] = []

        used_tracks: set[int] = set()
        for box, crop in zip(boxes, crops):
            track = track_manager.match_box(box)
            if track is not None and track.track_id in used_tracks:
                track = None

            if track is None:
                track = track_manager.create_track(box, processed_count, timestamp_sec)
            else:
                used_tracks.add(track.track_id)

            track.last_box = box
            track.last_seen_frame = processed_count

            if track.card is not None:
                track_manager.record_identified_card(track, track.card, timestamp_sec)
                continue

            pending_boxes.append(box)
            pending_crops.append(crop)
            pending_tracks.append(track)

        if pending_crops:
            identified = identify_crops(pending_crops, pending_boxes, dist_threshold)
            for track, card in zip(pending_tracks, identified):
                if not card["identified"]:
                    continue
                track_manager.record_identified_card(track, card, timestamp_sec)

        processed_count += 1
        frame_idx += 1

        if processed_count >= max_frames:
            break

    cap.release()

    cards = sorted(
        track_manager.unique_cards.values(),
        key=lambda card: card["first_seen_sec"],
    )

    duration_sec = round(total_frames / fps, 3) if total_frames > 0 else round(frame_idx / fps, 3)

    return {
        "video": {
            "duration_sec": duration_sec,
            "fps": round(fps, 3),
            "processed_frames": processed_count,
        },
        "count": len(cards),
        "cards": cards,
    }
