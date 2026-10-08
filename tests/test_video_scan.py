from video_scan import TrackManager, box_iou


def test_box_iou_identical_is_one():
    box = [0, 0, 100, 140]
    assert box_iou(box, box) == 1.0


def test_box_iou_disjoint_is_zero():
    assert box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_track_manager_matches_overlapping_box_and_expires():
    manager = TrackManager()
    track = manager.create_track([0, 0, 100, 140], frame_idx=0, timestamp_sec=0.0)
    matched = manager.match_box([5, 5, 105, 145])
    assert matched is track

    manager.expire_stale(current_frame=20, expiry_frames=15)
    assert manager.tracks == []


def test_track_manager_keeps_best_distance_for_same_card():
    manager = TrackManager()
    track = manager.create_track([0, 0, 10, 10], frame_idx=1, timestamp_sec=1.0)
    manager.record_identified_card(track, {"name": "Sol Ring", "set": "cmr", "dist": 800}, 1.2)
    manager.record_identified_card(track, {"name": "Sol Ring", "set": "cmr", "dist": 500}, 2.0)

    saved = manager.unique_cards[("Sol Ring", "cmr")]
    assert saved["best_dist"] == 500
    assert saved["first_seen_sec"] == 1.0
    assert saved["last_seen_sec"] == 2.0
