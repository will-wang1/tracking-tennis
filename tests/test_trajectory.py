from tennis_tracker.ball import BallDetection
from tennis_tracker.trajectory import detect_hits, smooth_trajectory

FPS = 30.0
DT = 1.0 / FPS


def _make_detections(hit_frames, n=70):
    positions = []
    x, y = 50.0, 200.0
    vx, vy = 240.0, -90.0  # px/sec
    for i in range(n):
        if i in hit_frames:
            vx, vy = -vx * 0.8, vy - 450
        x += vx * DT
        y += vy * DT
        positions.append((x, y))
    return [BallDetection(frame_index=i, timestamp=i * DT, position=positions[i]) for i in range(n)]


def test_detects_hits_near_known_direction_changes():
    detections = _make_detections(hit_frames=[20, 45])
    tracked = smooth_trajectory(detections, fps=FPS)

    hits = detect_hits(tracked)

    assert [h.frame_index for h in hits] == [20, 45]


def test_no_false_positives_on_constant_velocity():
    detections = _make_detections(hit_frames=[])
    tracked = smooth_trajectory(detections, fps=FPS)

    hits = detect_hits(tracked)

    assert hits == []


def test_smooth_trajectory_bridges_missing_detections():
    detections = _make_detections(hit_frames=[])
    for i in (10, 11, 12):
        detections[i].position = None

    tracked = smooth_trajectory(detections, fps=FPS)

    # Every input frame still produces a tracked point (predict-only when no measurement).
    assert len(tracked) == len(detections)
    assert all(p.residual is None for p in tracked if p.frame_index in (10, 11, 12))
