import json

import cv2
import numpy as np
import pytest

from tennis_tracker.calibration import COURT_LANDMARKS, calibrate_camera
from tennis_tracker.physics import TennisBallParams, simulate_trajectory
from tennis_tracker.physics_pipeline import PhysicsShotResult, _build_hit_window, write_physics_shot_log
from tennis_tracker.ball import BallDetection
from tennis_tracker.trajectory import HitEvent


def _synthetic_calibration(width, height):
    fx, fy = 1400.0, 1400.0
    cx, cy = width / 2, height / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    cam_pos = np.array([0.0, -22.0, 5.0])
    look_at = np.array([0.0, 2.0, 0.8])
    up = np.array([0.0, 0.0, 1.0])
    forward = look_at - cam_pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, forward)
    R = np.array([right, -cam_up, forward])
    t = -R @ cam_pos
    rvec, _ = cv2.Rodrigues(R)

    names = list(COURT_LANDMARKS.keys())
    object_points = np.array([COURT_LANDMARKS[n] for n in names], dtype=np.float64)
    image_points, _ = cv2.projectPoints(object_points, rvec, t, K, None)
    correspondences = {n: tuple(p) for n, p in zip(names, image_points.reshape(-1, 2))}
    return calibrate_camera(correspondences, (width, height))


def test_build_hit_window_excludes_gap_and_out_of_range_frames():
    detections = [
        BallDetection(frame_index=i, timestamp=i / 30, position=(float(i), float(i)))
        for i in range(0, 40)
    ]
    hit = HitEvent(frame_index=20, timestamp=20 / 30, position=(20.0, 20.0), residual=10.0)

    pre_obs, post_obs = _build_hit_window(detections, hit, window_size=10, gap=2)

    pre_frame_indices = [round(t * 30) for t, _, _ in pre_obs]
    post_frame_indices = [round(t * 30) for t, _, _ in post_obs]
    assert max(pre_frame_indices) == 17  # 20 - gap(2) - 1
    assert min(pre_frame_indices) == 10  # 20 - window_size(10)
    assert min(post_frame_indices) == 23  # 20 + gap(2) + 1
    assert max(post_frame_indices) == 30  # 20 + window_size(10)


def test_build_hit_window_skips_missing_detections():
    detections = [
        BallDetection(frame_index=i, timestamp=i / 30, position=None if i == 15 else (float(i), float(i)))
        for i in range(0, 40)
    ]
    hit = HitEvent(frame_index=20, timestamp=20 / 30, position=(20.0, 20.0), residual=10.0)

    pre_obs, _ = _build_hit_window(detections, hit, window_size=10, gap=2)

    assert 15 not in [round(t * 30) for t, _, _ in pre_obs]


def test_write_physics_shot_log_handles_failed_fits(tmp_path):
    results = [
        PhysicsShotResult(
            hit_frame_index=10, hit_timestamp=0.33, player_id=0, shot_type="forehand",
            classification_confidence=1.2, contact_event=None, fit_note="insufficient_observations",
        )
    ]
    out_path = tmp_path / "log.json"

    write_physics_shot_log(results, out_path)
    data = json.loads(out_path.read_text())

    assert data["shots"][0]["contact_time"] is None
    assert data["shots"][0]["fit_note"] == "insufficient_observations"


def test_end_to_end_recovers_contact_time_and_speed_order_of_magnitude(tmp_path):
    """Render a video whose on-screen ball motion is physically consistent with a
    known 3D trajectory + calibration, over a plain background (no pose needed
    for this check — classification is exercised separately), then verify the
    physics pipeline's fitted contact time and speeds land in the right ballpark.
    """
    width, height = 640, 480
    calib = _synthetic_calibration(width, height)
    params = TennisBallParams()

    true_contact_time = 1.0
    incoming_launch = np.array([-2.0, -9.0, 1.0, 6.0, 12.0, 4.0])
    pos_at_contact = simulate_trajectory(incoming_launch, np.array([true_contact_time]), params)[0]
    outgoing_velocity = np.array([-8.0, -15.0, 6.0])
    outgoing_launch = np.concatenate([pos_at_contact, outgoing_velocity])

    fps = 30.0
    dt = 1.0 / fps
    n_frames = 60
    gap_sec = 2 / fps

    video_path = tmp_path / "phase_g.mp4"
    out = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for i in range(n_frames):
        t_abs = i * dt
        frame = np.full((height, width, 3), (40, 120, 40), dtype=np.uint8)
        if t_abs < true_contact_time - gap_sec:
            pos3d = simulate_trajectory(incoming_launch, np.array([t_abs]), params)[0]
            draw = True
        elif t_abs > true_contact_time + gap_sec:
            pos3d = simulate_trajectory(outgoing_launch, np.array([t_abs - true_contact_time]), params)[0]
            draw = True
        else:
            draw = False
        if draw:
            px = calib.project(pos3d.reshape(1, 3))[0]
            if 0 <= px[0] < width and 0 <= px[1] < height:
                cv2.circle(frame, (int(px[0]), int(px[1])), 6, (30, 220, 230), -1)
        out.write(frame)
    out.release()

    from tennis_tracker.ball import detect_video
    from tennis_tracker.trajectory import detect_hits, smooth_trajectory

    detections = list(detect_video(video_path))
    tracked = smooth_trajectory(detections, fps=fps)
    hits = detect_hits(tracked)
    assert len(hits) >= 1

    hit = min(hits, key=lambda h: abs(h.timestamp - true_contact_time))
    pre_obs, post_obs = _build_hit_window(detections, hit, window_size=15, gap=2)
    assert len(pre_obs) >= 6 and len(post_obs) >= 6

    from tennis_tracker.contact_fit import fit_contact_event

    event = fit_contact_event(pre_obs, post_obs, calib, params)

    assert abs(event.contact_time - true_contact_time) < 0.2
    true_incoming_speed_kmh = np.linalg.norm(
        (simulate_trajectory(incoming_launch, np.array([true_contact_time + 1e-3]), params)[0]
         - simulate_trajectory(incoming_launch, np.array([true_contact_time - 1e-3]), params)[0]) / 2e-3
    ) * 3.6
    true_outgoing_speed_kmh = np.linalg.norm(outgoing_velocity) * 3.6

    # Loose bounds: the classical ball detector's pixel noise and the short
    # fit window mean this won't be exact — we're checking the pipeline
    # recovers the right physical ballpark, not pixel-perfect precision.
    assert event.incoming_speed * 3.6 == pytest.approx(true_incoming_speed_kmh, rel=0.3)
    assert event.outgoing_speed * 3.6 == pytest.approx(true_outgoing_speed_kmh, rel=0.5)
