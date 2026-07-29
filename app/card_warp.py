"""Perspective correction for detected MTG card crops."""

from __future__ import annotations

import cv2
import numpy as np

# MTG card aspect ratio is approximately 2.5" x 3.5" (5:7).
CARD_W = 488
CARD_H = 680
DEFAULT_PADDING = 0.08
MIN_WARP_DIMENSION = 80


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Return corners in top-left, top-right, bottom-right, bottom-left order."""
    pts = np.array(pts, dtype=np.float32).reshape(4, 2)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)

    top_left = pts[np.argmin(sums)]
    bottom_right = pts[np.argmax(sums)]
    top_right = pts[np.argmin(diffs)]
    bottom_left = pts[np.argmax(diffs)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def pad_box(box: list[int], frame_shape: tuple[int, int, int], padding: float = DEFAULT_PADDING) -> list[int]:
    """Expand a bounding box by a fraction of its width and height."""
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = box

    pad_x = int((x2 - x1) * padding)
    pad_y = int((y2 - y1) * padding)

    return [
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    ]


def get_padded_crop(frame_bgr: np.ndarray, box: list[int], padding: float = DEFAULT_PADDING) -> tuple[np.ndarray, list[int]]:
    """Return a padded crop and the padded box coordinates."""
    padded_box = pad_box(box, frame_bgr.shape, padding=padding)
    x1, y1, x2, y2 = padded_box
    crop = frame_bgr[y1:y2, x1:x2]
    return crop, padded_box


def find_card_quad(crop_bgr: np.ndarray) -> np.ndarray | None:
    """Find the largest convex quadrilateral contour in a YOLO crop."""
    if crop_bgr.size == 0:
        return None

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )
    edges = cv2.Canny(thresh, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    crop_h, crop_w = gray.shape
    min_area = 0.25 * crop_w * crop_h
    best_quad: np.ndarray | None = None
    best_area = 0.0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        if area > best_area:
            best_quad = approx.reshape(4, 2)
            best_area = area

    return best_quad


def warp_card_bgr(
    crop_bgr: np.ndarray,
    quad: np.ndarray,
    width: int = CARD_W,
    height: int = CARD_H,
) -> np.ndarray:
    """Perspective-warp a card crop to a canonical rectangle (BGR)."""
    source = order_corners(quad)
    destination = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(crop_bgr, matrix, (width, height))


def warp_card_rgb(
    crop_bgr: np.ndarray,
    quad: np.ndarray,
    width: int = CARD_W,
    height: int = CARD_H,
) -> np.ndarray:
    """Perspective-warp a card crop and return an RGB image for embedding."""
    warped_bgr = warp_card_bgr(crop_bgr, quad, width=width, height=height)
    return cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB)


def rectify_crop(
    frame_bgr: np.ndarray,
    box: list[int],
    padding: float = DEFAULT_PADDING,
) -> tuple[np.ndarray, bool]:
    """
    Try to perspective-correct a card inside a YOLO box.

    Returns an RGB crop suitable for embedding and whether a warp was applied.
    Falls back to the padded axis-aligned crop when corner detection fails.
    """
    crop_bgr, _ = get_padded_crop(frame_bgr, box, padding=padding)
    if crop_bgr.size == 0:
        return crop_bgr, False

    quad = find_card_quad(crop_bgr)
    if quad is None:
        return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB), False

    warped_rgb = warp_card_rgb(crop_bgr, quad)
    if min(warped_rgb.shape[:2]) < MIN_WARP_DIMENSION:
        return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB), False

    return warped_rgb, True
