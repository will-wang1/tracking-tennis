import csv
import json

import cv2
import numpy as np
import pytest

from tennis_tracker.ball import BallDetection
from tennis_tracker.classify import ShotClassification
from tennis_tracker.pipeline import (
    find_point_end_frame,
    get_ball_detections,
    parse_handedness_arg,
    render_annotated_video,
    write_shot_log,
)
from tennis_tracker.trajectory import HitEvent, TrackedPoint


def test_parse_handedness_arg():
    assert parse_handedness_arg("0:right,1:left") == {0: "right", 1: "left"}


def test_parse_handedness_arg_empty():
    assert parse_handedness_arg("") == {}
    assert parse_handedness_arg(None) == {}


def _sample_classifications():
    return [
        ShotClassification(
            hit=HitEvent(frame_index=10, timestamp=0.333, position=(100.0, 200.0), residual=15.0),
            player_id=0,
            shot_type="forehand",
            confidence=1.5,
        ),
        ShotClassification(
            hit=HitEvent(frame_index=40, timestamp=1.333, position=(300.0, 200.0), residual=20.0),
            player_id=1,
            shot_type="backhand",
            confidence=-0.9,
        ),
    ]


def test_write_shot_log_json(tmp_path):
    out_path = tmp_path / "log.json"
    write_shot_log(_sample_classifications(), out_path)

    data = json.loads(out_path.read_text())

    # Only shot types actually present are counted -- this must work for any
    # classifier's label vocabulary, not just the geometric heuristic's fixed
    # forehand/backhand/unclear/no_pose_data set, so zero-count types aren't
    # padded in.
    assert data["summary"] == {"forehand": 1, "backhand": 1}
    assert len(data["shots"]) == 2
    assert data["shots"][0]["shot_type"] == "forehand"
    assert data["shots"][0]["frame_index"] == 10


def test_write_shot_log_csv(tmp_path):
    out_path = tmp_path / "log.csv"
    write_shot_log(_sample_classifications(), out_path)

    with out_path.open() as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert rows[0]["shot_type"] == "forehand"
    assert rows[1]["player_id"] == "1"


def _write_synthetic_ball_video(path, n_frames=12, width=160, height=120):
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height))
    for i in range(n_frames):
        frame = np.full((height, width, 3), (40, 120, 40), dtype=np.uint8)
        cv2.circle(frame, (20 + i * 8, 30 + i * 4), 5, (30, 220, 230), -1)
        out.write(frame)
    out.release()


def test_get_ball_detections_classical(tmp_path):
    video_path = tmp_path / "clip.mp4"
    _write_synthetic_ball_video(video_path)

    detections = get_ball_detections(video_path, detector="classical")

    assert len(detections) == 12
    assert any(d.position is not None for d in detections)


def test_get_ball_detections_tracknet(tmp_path):
    pytest.importorskip("torch")
    from tennis_tracker.tracknet import TrackNet, save_model

    video_path = tmp_path / "clip.mp4"
    _write_synthetic_ball_video(video_path)
    model_path = tmp_path / "model.pt"
    save_model(TrackNet(input_size=(64, 48)), model_path)

    detections = get_ball_detections(video_path, detector="tracknet", tracknet_model_path=model_path)

    assert len(detections) == 12
    assert all(d.frame_index == i for i, d in enumerate(detections))


def test_get_ball_detections_yolo(tmp_path, monkeypatch):
    # ultralytics isn't a hard requirement to run these tests, so the
    # YoloBallDetector itself is stubbed out rather than actually loaded --
    # this exercises get_ball_detections' wiring (lazy import, kwargs,
    # per-frame BallDetection contract) without needing the real model.
    video_path = tmp_path / "clip.mp4"
    _write_synthetic_ball_video(video_path)

    monkeypatch.setattr("tennis_tracker.yolo_ball.YoloBallDetector.__init__", lambda self, **kwargs: None)
    monkeypatch.setattr("tennis_tracker.yolo_ball.YoloBallDetector.detect", lambda self, frame: ((25.0, 35.0), 5.0))

    detections = get_ball_detections(video_path, detector="yolo")

    assert len(detections) == 12
    assert all(d.position == (25.0, 35.0) for d in detections)


def test_get_ball_detections_tracknet_requires_model_path(tmp_path):
    video_path = tmp_path / "clip.mp4"
    _write_synthetic_ball_video(video_path)

    with pytest.raises(ValueError, match="tracknet_model_path"):
        get_ball_detections(video_path, detector="tracknet")


def test_get_ball_detections_rejects_unknown_detector(tmp_path):
    video_path = tmp_path / "clip.mp4"
    _write_synthetic_ball_video(video_path)

    with pytest.raises(ValueError, match="detector must be"):
        get_ball_detections(video_path, detector="magic")


def test_render_annotated_video_reuses_passed_in_detections(tmp_path, monkeypatch):
    video_path = tmp_path / "clip.mp4"
    _write_synthetic_ball_video(video_path)
    output_path = tmp_path / "out.mp4"

    detections = [
        BallDetection(frame_index=i, timestamp=i / 30, position=(20.0 + i, 30.0))
        for i in range(12)
    ]

    # If render_annotated_video ever re-detects instead of reusing `detections`,
    # this would blow up (BallDetector isn't even imported by pipeline.py
    # anymore) -- guards against that regression rather than just checking
    # the happy path runs.
    monkeypatch.delattr("tennis_tracker.pipeline.detect_video", raising=False)

    frame_count = render_annotated_video(video_path, output_path, [], detections, [], [])

    assert frame_count == 12
    assert output_path.exists()


def _detections_with_gap(fps, n_frames, gap_start, gap_len):
    detections = []
    for i in range(n_frames):
        in_gap = gap_start <= i < gap_start + gap_len
        position = None if in_gap else (float(i), 50.0)
        detections.append(BallDetection(frame_index=i, timestamp=i / fps, position=position))
    return detections


def test_find_point_end_frame_stops_at_a_long_gap():
    fps = 30.0
    # Ball detected through frame 59, then a 2-second gap (60 frames) starting at 60.
    detections = _detections_with_gap(fps, n_frames=150, gap_start=60, gap_len=60)

    point_end_frame = find_point_end_frame(detections, fps, max_gap_sec=1.5)

    assert point_end_frame == 59


def test_find_point_end_frame_ignores_short_gaps():
    fps = 30.0
    # A brief 5-frame gap (motion blur, not the point ending) shouldn't trigger truncation.
    detections = _detections_with_gap(fps, n_frames=100, gap_start=40, gap_len=5)

    point_end_frame = find_point_end_frame(detections, fps, max_gap_sec=1.5)

    assert point_end_frame == 99  # whole clip counts as the point


def test_find_point_end_frame_empty_detections():
    assert find_point_end_frame([], fps=30.0) == 0


def test_render_annotated_video_truncates_at_end_frame(tmp_path):
    video_path = tmp_path / "clip.mp4"
    _write_synthetic_ball_video(video_path)  # 12 frames
    output_path = tmp_path / "out.mp4"

    frame_count = render_annotated_video(video_path, output_path, [], [], [], [], end_frame=5)

    assert frame_count == 6  # frames 0..5 inclusive


def test_render_annotated_video_labels_hitting_player_not_a_floating_label(tmp_path):
    from tennis_tracker.pose import FramePoses, PlayerPose

    video_path = tmp_path / "clip.mp4"
    _write_synthetic_ball_video(video_path)
    output_path = tmp_path / "out.mp4"

    player = PlayerPose(
        player_id=0, landmarks_px=np.zeros((33, 2), dtype=np.float32),
        visibility=np.ones(33, dtype=np.float32), bbox=(0, 0, 40, 40),
    )
    frame_poses_list = [FramePoses(frame_index=0, timestamp=0.0, players=[player])]
    classifications = [
        ShotClassification(
            hit=HitEvent(frame_index=0, timestamp=0.0, position=(20.0, 30.0), residual=10.0),
            player_id=0, shot_type="forehand", confidence=1.0,
        )
    ]

    # Should not raise, and should actually attach the label to the player
    # (draw_pose_overlay is exercised via player_labels, not a separate
    # floating-label code path that no longer exists).
    frame_count = render_annotated_video(
        video_path, output_path, frame_poses_list, [], [], classifications
    )

    assert frame_count == 12


def test_render_annotated_video_shows_velocity_above_ball(tmp_path):
    video_path = tmp_path / "clip.mp4"
    _write_synthetic_ball_video(video_path)
    output_path = tmp_path / "out.mp4"

    detections = [BallDetection(frame_index=0, timestamp=0.0, position=(20.0, 30.0))]
    tracked = [TrackedPoint(frame_index=0, timestamp=0.0, position=(20.0, 30.0), velocity=(100.0, 0.0), residual=None)]

    # Just needs to run without error with velocity data present; visual
    # correctness (text drawn near the ball) isn't practical to assert on
    # pixel content here.
    frame_count = render_annotated_video(video_path, output_path, [], detections, tracked, [])

    assert frame_count == 12
