import csv
import json

import cv2
import numpy as np
import pytest

from tennis_tracker.ball import BallDetection
from tennis_tracker.classify import ShotClassification
from tennis_tracker.pipeline import get_ball_detections, parse_handedness_arg, render_annotated_video, write_shot_log
from tennis_tracker.trajectory import HitEvent


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

    assert data["summary"] == {"forehand": 1, "backhand": 1, "unclear": 0, "no_pose_data": 0}
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
    save_model(TrackNet(num_frames=3), model_path)

    detections = get_ball_detections(video_path, detector="tracknet", tracknet_model_path=model_path)

    assert len(detections) == 12
    assert all(d.frame_index == i for i, d in enumerate(detections))


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

    frame_count = render_annotated_video(video_path, output_path, [], detections, [])

    assert frame_count == 12
    assert output_path.exists()
