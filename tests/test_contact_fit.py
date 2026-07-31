import cv2
import numpy as np
import pytest

from tennis_tracker.calibration import COURT_LANDMARKS, calibrate_camera
from tennis_tracker.contact_fit import fit_contact_event
from tennis_tracker.physics import TennisBallParams, simulate_trajectory


def _synthetic_calibration():
    fx, fy = 1000.0, 1000.0
    width, height = 1920, 1080
    cx, cy = width / 2, height / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    cam_pos = np.array([0.0, -30.0, 6.5])
    look_at = np.array([0.0, 0.0, 1.0])
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


def _build_synthetic_contact(noise_std, rng, calib, params):
    true_contact_time = 1.0
    incoming_launch = np.array([-4.0, -11.0, 1.2, 10.0, 15.0, 3.0])
    pos_at_contact = simulate_trajectory(incoming_launch, np.array([true_contact_time]), params)[0]
    outgoing_velocity_at_contact = np.array([-14.0, -30.0, 8.0])

    pre_times = np.linspace(0.0, true_contact_time - 0.05, 10)
    pre_positions = simulate_trajectory(incoming_launch, pre_times, params)
    pre_pixels = calib.project(pre_positions)
    if noise_std > 0:
        pre_pixels = pre_pixels + rng.normal(0, noise_std, pre_pixels.shape)
    pre_observations = [
        (float(pre_times[i]), float(pre_pixels[i, 0]), float(pre_pixels[i, 1])) for i in range(len(pre_times))
    ]

    outgoing_launch_state = np.concatenate([pos_at_contact, outgoing_velocity_at_contact])
    post_times_rel = np.linspace(0.05, 0.5, 10)
    post_positions = simulate_trajectory(outgoing_launch_state, post_times_rel, params)
    post_times_abs = true_contact_time + post_times_rel
    post_pixels = calib.project(post_positions)
    if noise_std > 0:
        post_pixels = post_pixels + rng.normal(0, noise_std, post_pixels.shape)
    post_observations = [
        (float(post_times_abs[i]), float(post_pixels[i, 0]), float(post_pixels[i, 1]))
        for i in range(len(post_times_rel))
    ]

    true_v_incoming = simulate_trajectory(
        incoming_launch, np.array([true_contact_time - 1e-3, true_contact_time + 1e-3]), params
    )
    true_incoming_speed = float(np.linalg.norm((true_v_incoming[1] - true_v_incoming[0]) / 2e-3))
    true_outgoing_speed = float(np.linalg.norm(outgoing_velocity_at_contact))

    return (
        pre_observations,
        post_observations,
        true_contact_time,
        pos_at_contact,
        true_incoming_speed,
        true_outgoing_speed,
    )


def test_contact_event_recovers_ground_truth_without_noise():
    calib = _synthetic_calibration()
    params = TennisBallParams()
    pre_obs, post_obs, true_t, true_pos, true_v_in, true_v_out = _build_synthetic_contact(
        0.0, None, calib, params
    )

    event = fit_contact_event(pre_obs, post_obs, calib, params)

    assert event.contact_time == pytest.approx(true_t, abs=1e-3)
    assert event.position_gap_m < 1e-3
    assert event.incoming_speed == pytest.approx(true_v_in, abs=1e-3)
    assert event.outgoing_speed == pytest.approx(true_v_out, abs=1e-3)


def test_contact_event_within_uncertainty_under_noise():
    rng = np.random.default_rng(3)
    calib = _synthetic_calibration()
    params = TennisBallParams()
    pre_obs, post_obs, true_t, true_pos, true_v_in, true_v_out = _build_synthetic_contact(
        1.0, rng, calib, params
    )

    event = fit_contact_event(pre_obs, post_obs, calib, params)

    assert abs(event.contact_time - true_t) < 0.1
    assert abs(event.incoming_speed - true_v_in) < 4 * event.incoming_speed_std
    assert abs(event.outgoing_speed - true_v_out) < 4 * event.outgoing_speed_std
