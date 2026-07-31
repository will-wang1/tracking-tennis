import cv2
import numpy as np
import pytest

from tennis_tracker.calibration import COURT_LANDMARKS, calibrate_camera
from tennis_tracker.physics import TennisBallParams, simulate_trajectory
from tennis_tracker.trajectory_fit import fit_trajectory


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


def _synthetic_observations(launch_state, times, calibration, noise_std=0.0, rng=None):
    positions = simulate_trajectory(launch_state, times, TennisBallParams())
    pixels = calibration.project(positions)
    if noise_std > 0:
        pixels = pixels + rng.normal(0, noise_std, pixels.shape)
    return [(float(t), float(px), float(py)) for t, (px, py) in zip(times, pixels)]


def test_fit_recovers_launch_state_exactly_without_noise():
    calibration = _synthetic_calibration()
    true_launch = np.array([-4.0, -11.0, 1.0, 18.0, 22.0, 6.0])
    times = np.linspace(0.0, 0.6, 12)
    observations = _synthetic_observations(true_launch, times, calibration)

    fit = fit_trajectory(observations, calibration)

    assert fit.success
    assert fit.residual_rms_px < 1e-6
    np.testing.assert_allclose(fit.launch_state, true_launch, atol=1e-6)


def test_fit_recovers_launch_state_within_uncertainty_under_noise():
    rng = np.random.default_rng(7)
    calibration = _synthetic_calibration()
    true_launch = np.array([-4.0, -11.0, 1.0, 18.0, 22.0, 6.0])
    times = np.linspace(0.0, 0.6, 12)
    observations = _synthetic_observations(true_launch, times, calibration, noise_std=1.0, rng=rng)

    fit = fit_trajectory(observations, calibration)

    assert fit.success
    assert fit.residual_rms_px < 3.0
    errors_in_sigma = np.abs(fit.launch_state - true_launch) / fit.param_std
    # With honestly-calibrated uncertainty, a single noisy fit landing within
    # 4 sigma on every one of 6 correlated parameters is an extremely safe bound.
    assert np.all(errors_in_sigma < 4.0)


def test_position_at_matches_ground_truth():
    calibration = _synthetic_calibration()
    true_launch = np.array([-4.0, -11.0, 1.0, 18.0, 22.0, 6.0])
    times = np.linspace(0.0, 0.6, 12)
    observations = _synthetic_observations(true_launch, times, calibration)
    fit = fit_trajectory(observations, calibration)

    true_positions = simulate_trajectory(true_launch, times, TennisBallParams())
    predicted = fit.position_at(times[5])

    np.testing.assert_allclose(predicted, true_positions[5], atol=1e-3)


def test_raises_with_too_few_observations():
    calibration = _synthetic_calibration()
    with pytest.raises(ValueError, match="at least"):
        fit_trajectory([(0.0, 500.0, 500.0), (0.1, 510.0, 505.0)], calibration)
