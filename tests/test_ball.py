import cv2
import numpy as np

from tennis_tracker.ball import detect_video


def _write_synthetic_ball_video(path, num_frames=40, width=320, height=180):
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height))
    positions = []
    for i in range(num_frames):
        frame = np.full((height, width, 3), (40, 120, 40), dtype=np.uint8)
        cv2.line(frame, (20, 20), (width - 20, 20), (255, 255, 255), 2)
        cv2.line(frame, (20, height - 20), (width - 20, height - 20), (255, 255, 255), 2)
        cv2.line(frame, (width // 2, 20), (width // 2, height - 20), (255, 255, 255), 1)

        x = 30 + i * 6
        y = int(90 + 60 * np.sin(i * 0.25))
        positions.append((x, y))
        cv2.circle(frame, (x, y), 6, (30, 220, 230), -1)
        out.write(frame)
    out.release()
    return positions


def test_detects_ball_close_to_ground_truth(tmp_path):
    video_path = tmp_path / "ball.mp4"
    ground_truth = _write_synthetic_ball_video(video_path)

    detections = list(detect_video(video_path))

    assert len(detections) == len(ground_truth)
    found = [d for d in detections if d.position is not None]
    assert len(found) / len(detections) > 0.9

    errors = [
        ((d.position[0] - gt[0]) ** 2 + (d.position[1] - gt[1]) ** 2) ** 0.5
        for d, gt in zip(detections, ground_truth)
        if d.position is not None
    ]
    assert max(errors) < 5.0


def test_no_detection_on_blank_video(tmp_path):
    video_path = tmp_path / "blank.mp4"
    out = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (100, 100))
    for _ in range(10):
        out.write(np.full((100, 100, 3), (40, 120, 40), dtype=np.uint8))
    out.release()

    detections = list(detect_video(video_path))

    assert all(d.position is None for d in detections)
