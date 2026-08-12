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

def quad_aspect_score(quad: np.ndarray) -> float:
    """
        Does:
            Score how close a quad is to the MTG aspect (~5:7). 0 = reject, ~1 = ideal.
        
        Args:
            quad: np.ndarray

        Uses:
            order_corners()

        Returns:
            float: score of the quad to having the correct aspect ratio
    
    """
    ordered = order_corners(quad.astype(np.float32))
    top = np.linalg.norm(ordered[1] - ordered[0])
    bottom = np.linalg.norm(ordered[2] - ordered[3])
    left = np.linalg.norm(ordered[3] - ordered[0])
    right = np.linalg.norm(ordered[2] - ordered[1])

    width = (top + bottom) / 2.0
    height = (left + right) / 2.0

    if width < 1.0 or height < 1.0:
        return 0.0

    ratio = height / width

    target = CARD_H / CARD_W

    if ratio < 1.05 or ratio > 1.393:
        return 0.0

    return float(max(0.0, 1.0 - abs(ratio-target) / target))

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

def _edge_maps(gray: np.ndarray) -> list[np.ndarray]:
    """Build several edge images; glare often kills only one stragety"""
    maps: list[np.array] = []
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # adaptive threshold + Canny
    ath = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    maps.append(cv2.Canny(ath, 50, 150))

    # CLAHE helps with uneven lighting
    clahe = cv2.createCLAHE(clipLimit=2.0, tilerGridSize=(8, 8))
    eq = clahe.apply(gray)
    maps.append(cv2.Canny(cv2.GaussianBlur(eq, (5, 5), 0), 40, 120))

    # Plain Canny on blur
    maps.append(cv2.Canny(blur, 60, 180))

    return maps

def find_card_quad(crop_bgr: np.ndarray) -> np.ndarray | None:
    """
        Does:
            Find the largest card-shaped convex quad in a YOLO crop.

        Args:
            crop_bgr: 3D numpy array (BGR)

        Uses:
            _edge_map()
            _collect_quads_from_edges()
            quad_aspect_score()

        Returns:
            bext_quad: np.ndarry
    """
    if crop_bgr.size == 0:
        return None

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    crop_h, crop_w = gray.shape
    crop_area = float(crop_w * crop_h)
    min_area = 0.20 * crop_area

    best_quad: np.ndarray | None = None
    best_score = -1.0

    for edges in _edge_maps(gray):
        for quad, area in _collect_quads_from_edges(edges, min_area=min_area):
            aspecr = quad_aspect_score(quad)
            if aspect <= 0.0:
                continue
            
            area_norm = min(1.0, area  crop_size)
            score = 0.5 * area_norm + 0.5 * aspect

            if score >= best_score:
                best_quad = quad

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
        Does:
            gets the padded crops and finds quadrants, 

        Args:
            frame_bgr: 3D numpy arry (BGR)
            box: list[int]
            padding: float

        Uses:
            get_padded_crop()
            find_card_quad()

        Returns:
            RGB crop suitable for embedding and whether a warp was applied.
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

def _collect_quads_from_edges(
    edges: np.ndarray,
    min_area: float,
    approx_eps: float = 0.02,
) -> list[tuple[np.ndarray, float]]:
    """
        Does:
            Finds quadrilateral shapes from a binary edge image

        Args:
            edges: np.ndarray
            min_area: float
            approx_epss: float

        Uses:
            a lot of cv2

        Return: 
            found: list[tuple[np.ndarray, float]]
    """
    # Forms full closed boundaries
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Locates the outer bondaries of all connected white spaces int he edge map
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    found: list[tuple[np.ndarray, float]] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, approx_eps * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        
        quad = approx.reshape(4, 2).astype(np.float32)
        found.append((quad, area))

    return found
