import numpy as np
import pytest

from tennis_tracker.pose import (
    FramePoses,
    PlayerPose,
    PlayerTracker,
    _detect_pose_in_crop,
    _filter_by_court_polygon,
    _filter_to_most_active_tracks,
    _pad_and_clip_box,
    _yolo_person_boxes,
    draw_pose_overlay,
    ensure_model,
)


def _fake_detection(center_x, center_y, size=20.0):
    landmarks = np.tile([center_x, center_y], (33, 1)).astype(np.float32)
    landmarks[:, 0] += np.linspace(-size, size, 33)
    visibility = np.ones(33, dtype=np.float32)
    bbox = (int(center_x - size), int(center_y - size), int(center_x + size), int(center_y + size))
    return landmarks, visibility, bbox


def test_assigns_distinct_ids_to_two_players():
    tracker = PlayerTracker(max_players=2, max_match_distance=50.0)
    left = _fake_detection(100, 200)
    right = _fake_detection(500, 200)

    players = tracker.update([left, right])

    assert {p.player_id for p in players} == {0, 1}


def test_keeps_ids_stable_as_players_move():
    tracker = PlayerTracker(max_players=2, max_match_distance=50.0)
    tracker.update([_fake_detection(100, 200), _fake_detection(500, 200)])

    # Left player drifts right, right player drifts left, but each moves less
    # than max_match_distance between frames -> IDs should not swap.
    players = tracker.update([_fake_detection(120, 200), _fake_detection(480, 200)])

    by_id = {p.player_id: p for p in players}
    assert by_id[0].landmarks_px[:, 0].mean() == pytest.approx(120, abs=15)
    assert by_id[1].landmarks_px[:, 0].mean() == pytest.approx(480, abs=15)


def test_caps_tracked_players_at_max_players():
    tracker = PlayerTracker(max_players=2, max_match_distance=50.0)

    players = tracker.update(
        [_fake_detection(100, 200), _fake_detection(500, 200), _fake_detection(900, 200)]
    )

    assert len(players) <= 2


def test_new_detection_far_from_existing_tracks_gets_new_id():
    tracker = PlayerTracker(max_players=2, max_match_distance=50.0)
    tracker.update([_fake_detection(100, 200)])

    players = tracker.update([_fake_detection(100, 200), _fake_detection(500, 200)])

    assert {p.player_id for p in players} == {0, 1}


def test_survives_a_brief_miss_by_extrapolating_velocity():
    # Player moving 40px/frame in x. A frozen last-known-position match
    # (100 -> 140, then missed a frame, reappears at 220) would be 80px
    # away from the last real detection — past max_match_distance=50 — so
    # without velocity prediction this would incorrectly get a new ID.
    tracker = PlayerTracker(max_players=2, max_match_distance=50.0)
    tracker.update([_fake_detection(100, 200)])
    tracker.update([_fake_detection(140, 200)])

    tracker.update([])  # missed frame: motion blur / brief occlusion mid-swing

    players = tracker.update([_fake_detection(220, 200)])

    assert {p.player_id for p in players} == {0}


def test_track_expires_after_too_many_consecutive_misses():
    tracker = PlayerTracker(max_players=2, max_match_distance=50.0, max_missed_frames=3)
    tracker.update([_fake_detection(100, 200)])

    for _ in range(4):  # more misses than max_missed_frames -> track 0 expires
        tracker.update([])

    # A new detection at the same old position must not reuse the expired
    # track's ID — that would silently misattribute it to whoever left.
    players = tracker.update([_fake_detection(100, 200)])

    assert {p.player_id for p in players} == {1}


def test_ensure_model_rejects_unknown_variant():
    with pytest.raises(ValueError, match="Unknown pose model variant"):
        ensure_model(variant="ultra")


def test_ensure_model_reuses_cached_file_without_redownloading(tmp_path, monkeypatch):
    fake_path = tmp_path / "pose_landmarker_lite.task"
    fake_path.write_bytes(b"fake model bytes")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("should not attempt to download when the file already exists")

    monkeypatch.setattr("tennis_tracker.pose.urllib.request.urlretrieve", fail_if_called)

    result = ensure_model(variant="lite", model_path=fake_path)

    assert result == fake_path
    assert result.read_bytes() == b"fake model bytes"


def _make_frame_poses(frame_index, tracks: dict):
    """``tracks`` maps player_id -> (x, y) centroid for this one frame."""
    players = []
    for player_id, (x, y) in tracks.items():
        landmarks, visibility, bbox = _fake_detection(x, y)
        players.append(PlayerPose(player_id, landmarks, visibility, bbox))
    return FramePoses(frame_index=frame_index, timestamp=frame_index / 30, players=players)


def test_filter_to_most_active_tracks_drops_stationary_bystander():
    # Tracks 0 and 1 move substantially (real players); track 2 never
    # moves (e.g. a ball kid or umpire standing still) and must be dropped.
    frame_poses_list = [
        _make_frame_poses(i, {0: (100 + i * 20, 200), 1: (900 - i * 20, 200), 2: (500, 500)})
        for i in range(20)
    ]

    filtered = _filter_to_most_active_tracks(frame_poses_list, max_players=2)

    all_ids = {p.player_id for fp in filtered for p in fp.players}
    assert all_ids == {0, 1}
    assert len(filtered) == 20
    assert all(len(fp.players) == 2 for fp in filtered)


def test_filter_to_most_active_tracks_remaps_ids_sequentially():
    # The two movers have non-trivial original track IDs (5, 7); a
    # stationary bystander has ID 2 in between them and must still be
    # excluded, with the survivors remapped to 0/1.
    frame_poses_list = [
        _make_frame_poses(i, {5: (100 + i * 30, 200), 7: (900 - i * 30, 200), 2: (400, 400)})
        for i in range(10)
    ]

    filtered = _filter_to_most_active_tracks(frame_poses_list, max_players=2)

    all_ids = {p.player_id for fp in filtered for p in fp.players}
    assert all_ids == {0, 1}


def test_filter_to_most_active_tracks_ignores_briefly_seen_track():
    # A track seen for only one frame contributes zero movement and should
    # lose out to tracks with a real, sustained movement history.
    frame_poses_list = [
        _make_frame_poses(i, {0: (100 + i * 20, 200), 1: (900 - i * 20, 200)})
        for i in range(20)
    ]
    frame_poses_list[0].players.append(PlayerPose(99, *_fake_detection(500, 500)))

    filtered = _filter_to_most_active_tracks(frame_poses_list, max_players=2)

    all_ids = {p.player_id for fp in filtered for p in fp.players}
    assert 99 not in all_ids
    assert all_ids == {0, 1}


# A simple square "court" in pixel space, for testing court-polygon filtering
# without needing a real camera calibration.
_COURT_POLYGON = np.array([[0, 0], [1000, 0], [1000, 1000], [0, 1000]], dtype=np.float32)


def test_filter_by_court_polygon_drops_a_track_mostly_outside_it():
    # Track 0 stays on-court the whole time; track 1 (a ball kid pacing by
    # the fence, moving enough to fool the movement filter alone) stays
    # entirely outside the court polygon.
    frame_poses_list = [
        _make_frame_poses(i, {0: (500, 500 + i), 1: (1500 + i * 5, 500)})
        for i in range(10)
    ]

    filtered = _filter_by_court_polygon(frame_poses_list, _COURT_POLYGON)

    all_ids = {p.player_id for fp in filtered for p in fp.players}
    assert all_ids == {0}


def test_filter_by_court_polygon_keeps_a_track_mostly_inside_it():
    # Track spends 8/10 frames on-court and 2/10 briefly outside (e.g.
    # chasing a wide ball past the sideline) -- above the default 50%
    # threshold, so it must be kept, not penalized for a brief excursion.
    tracks = {0: (500, 500)}
    frame_poses_list = []
    for i in range(10):
        x = 1500 if i < 2 else 500  # outside court for the first 2 frames only
        frame_poses_list.append(_make_frame_poses(i, {0: (x, 500)}))

    filtered = _filter_by_court_polygon(frame_poses_list, _COURT_POLYGON)

    all_ids = {p.player_id for fp in filtered for p in fp.players}
    assert all_ids == {0}


def test_filter_by_court_polygon_respects_custom_threshold():
    # Same 2/10-outside track as above, but with a stricter 90% threshold
    # it should now be dropped.
    frame_poses_list = []
    for i in range(10):
        x = 1500 if i < 2 else 500
        frame_poses_list.append(_make_frame_poses(i, {0: (x, 500)}))

    filtered = _filter_by_court_polygon(frame_poses_list, _COURT_POLYGON, min_fraction_inside=0.9)

    all_ids = {p.player_id for fp in filtered for p in fp.players}
    assert all_ids == set()


def test_draw_pose_overlay_skips_low_visibility_landmarks():
    # Index 16 of _fake_detection's landmarks sits at the exact centroid
    # (linspace(-size, size, 33) crosses zero at its middle element),
    # comfortably inside the bbox interior so it can't collide with the
    # bbox rectangle or label text drawn regardless of visibility.
    landmarks, visibility, bbox = _fake_detection(100, 100)
    center_x, center_y = int(landmarks[16, 0]), int(landmarks[16, 1])

    low_visibility = visibility * 0.1  # below the default 0.5 threshold
    frame_poses = FramePoses(0, 0.0, [PlayerPose(0, landmarks, low_visibility, bbox)])
    blank = np.zeros((200, 200, 3), dtype=np.uint8)

    annotated = draw_pose_overlay(blank, frame_poses)

    assert tuple(annotated[center_y, center_x]) == (0, 0, 0)


def test_draw_pose_overlay_draws_high_visibility_landmarks():
    landmarks, visibility, bbox = _fake_detection(100, 100)
    center_x, center_y = int(landmarks[16, 0]), int(landmarks[16, 1])

    frame_poses = FramePoses(0, 0.0, [PlayerPose(0, landmarks, visibility, bbox)])
    blank = np.zeros((200, 200, 3), dtype=np.uint8)

    annotated = draw_pose_overlay(blank, frame_poses)

    assert tuple(annotated[center_y, center_x]) != (0, 0, 0)


class _FakeYoloBox:
    def __init__(self, xyxy, conf):
        self.xyxy = [xyxy]
        self.conf = [conf]


class _FakeYoloResults:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeYoloModel:
    def __init__(self, boxes):
        self._boxes = boxes

    def predict(self, frame, classes, conf, device, verbose):
        return [_FakeYoloResults(self._boxes)]


def test_yolo_person_boxes_sorts_by_confidence_and_caps_count():
    boxes = [
        _FakeYoloBox((0.0, 0.0, 10.0, 10.0), 0.3),
        _FakeYoloBox((20.0, 20.0, 30.0, 30.0), 0.9),
        _FakeYoloBox((40.0, 40.0, 50.0, 50.0), 0.6),
    ]
    model = _FakeYoloModel(boxes)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    result = _yolo_person_boxes(model, frame, confidence=0.1, max_people=2, device="cpu")

    assert result == [(20.0, 20.0, 30.0, 30.0), (40.0, 40.0, 50.0, 50.0)]


def test_pad_and_clip_box_adds_proportional_padding():
    # A 20x20 box padded 25% each side should grow by 5px on every edge.
    result = _pad_and_clip_box((10, 10, 30, 30), padding_frac=0.25, frame_width=100, frame_height=100)

    assert result == (5, 5, 35, 35)


def test_pad_and_clip_box_clips_at_frame_edges():
    result = _pad_and_clip_box((0, 0, 20, 20), padding_frac=0.5, frame_width=100, frame_height=100)

    assert result == (0, 0, 30, 30)


class _FakeLandmark:
    def __init__(self, x, y, visibility=1.0):
        self.x = x
        self.y = y
        self.visibility = visibility


class _FakeLandmarker:
    def __init__(self, pose_landmarks=None):
        self._pose_landmarks = pose_landmarks or []

    def detect(self, mp_image):
        class _Result:
            pass

        result = _Result()
        result.pose_landmarks = self._pose_landmarks
        return result


def test_detect_pose_in_crop_maps_landmarks_back_to_full_frame_coords():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    crop_box = (50, 60, 150, 160)  # 100x100 crop
    landmarker = _FakeLandmarker(pose_landmarks=[[_FakeLandmark(0.5, 0.5)]])

    result = _detect_pose_in_crop(landmarker, frame, crop_box)

    assert result is not None
    pts, visibility, bbox = result
    assert tuple(pts[0]) == pytest.approx((100.0, 110.0))


def test_detect_pose_in_crop_returns_none_when_no_pose_found():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    landmarker = _FakeLandmarker(pose_landmarks=[])

    result = _detect_pose_in_crop(landmarker, frame, (50, 60, 150, 160))

    assert result is None


def test_detect_pose_in_crop_returns_none_for_degenerate_box():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    landmarker = _FakeLandmarker(pose_landmarks=[[_FakeLandmark(0.5, 0.5)]])

    result = _detect_pose_in_crop(landmarker, frame, (50, 60, 50, 160))  # zero width

    assert result is None
