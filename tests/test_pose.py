import numpy as np
import pytest

from tennis_tracker.pose import PlayerTracker


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
