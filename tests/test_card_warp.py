import numpy as np

from card_warp import CARD_H, CARD_W, order_corners, pad_box, quad_aspect_score


def test_order_corners_sorts_clockwise_from_top_left():
    scrambled = np.array([[10, 14], [0, 0], [0, 14], [10, 0]], dtype=np.float32)
    ordered = order_corners(scrambled)
    np.testing.assert_allclose(ordered[0], [0, 0])
    np.testing.assert_allclose(ordered[1], [10, 0])
    np.testing.assert_allclose(ordered[2], [10, 14])
    np.testing.assert_allclose(ordered[3], [0, 14])


def test_quad_aspect_score_accepts_card_ratio():
    height = 100 * (CARD_H / CARD_W)
    quad = np.array([[0, 0], [100, 0], [100, height], [0, height]], dtype=np.float32)
    score = quad_aspect_score(quad)
    assert score > 0.95


def test_quad_aspect_score_rejects_square():
    quad = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    assert quad_aspect_score(quad) == 0.0


def test_pad_box_expands_and_clamps():
    padded = pad_box([10, 20, 110, 220], (300, 400, 3), padding=0.1)
    assert padded == [0, 0, 120, 240]
