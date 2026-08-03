import cv2
import numpy as np
import pytest

from tennis_tracker.calibration import (
    COURT_LANDMARKS,
    _HALF_DOUBLES_WIDTH,
    _HALF_LENGTH,
    calibrate_camera,
    court_boundary_polygon,
    load_correspondences_json,
    save_correspondences_json,
)


def _synthetic_correspondences(fx, fy, width, height, cam_pos, look_at, noise_std=0.0, rng=None):
    cx, cy = width / 2, height / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    up = np.array([0.0, 0.0, 1.0])
    forward = look_at - cam_pos
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    cam_up = np.cross(right, forward)
    R = np.array([right, -cam_up, forward])
    t = -R @ cam_pos
    rvec, _ = cv2.Rodrigues(R)

    names = list(COURT_LANDMARKS.keys())
    object_points = np.array([COURT_LANDMARKS[n] for n in names], dtype=np.float64)
    image_points, _ = cv2.projectPoints(object_points, rvec, t, K, None)
    image_points = image_points.reshape(-1, 2)

    if noise_std > 0:
        image_points = image_points + rng.normal(0, noise_std, image_points.shape)

    return {n: tuple(p) for n, p in zip(names, image_points)}, object_points


def test_calibration_recovers_focal_length_and_camera_position_exactly():
    cam_pos = np.array([0.0, -30.0, 7.0])
    correspondences, _ = _synthetic_correspondences(
        1100.0, 1100.0, 1280, 720, cam_pos, look_at=np.array([0.0, 0.0, 0.0])
    )

    calib = calibrate_camera(correspondences, (1280, 720))

    assert calib.fx == pytest.approx(1100.0, rel=1e-3)
    assert calib.reprojection_error_px < 0.01
    np.testing.assert_allclose(calib.camera_position_world(), cam_pos, atol=0.01)


def test_calibration_robust_to_pixel_noise():
    rng = np.random.default_rng(42)
    cam_pos = np.array([1.5, -28.0, 6.0])
    correspondences, _ = _synthetic_correspondences(
        950.0, 950.0, 1920, 1080, cam_pos, look_at=np.array([0.0, 2.0, 0.5]),
        noise_std=1.5, rng=rng,
    )

    calib = calibrate_camera(correspondences, (1920, 1080))

    assert calib.fx == pytest.approx(950.0, rel=0.02)
    np.testing.assert_allclose(calib.camera_position_world(), cam_pos, atol=0.5)


def test_project_round_trips_court_points():
    cam_pos = np.array([0.0, -30.0, 7.0])
    correspondences, object_points = _synthetic_correspondences(
        1100.0, 1100.0, 1280, 720, cam_pos, look_at=np.array([0.0, 0.0, 0.0])
    )
    original_pixels = np.array(list(correspondences.values()))

    calib = calibrate_camera(correspondences, (1280, 720))
    reprojected = calib.project(object_points)

    np.testing.assert_allclose(reprojected, original_pixels, atol=0.1)


def test_raises_with_too_few_points():
    with pytest.raises(ValueError, match="at least"):
        calibrate_camera(
            {"net_center_ground": (640, 360), "baseline_near_center_mark": (640, 600)},
            (1280, 720),
        )


def test_raises_with_unknown_landmark_name():
    correspondences = {f"bad_name_{i}": (i * 10.0, i * 10.0) for i in range(8)}
    with pytest.raises(ValueError, match="Unknown landmark"):
        calibrate_camera(correspondences, (1280, 720))


def test_correspondences_json_round_trip(tmp_path):
    correspondences = {"net_center_ground": (640.0, 360.0), "baseline_near_center_mark": (640.5, 599.25)}
    path = tmp_path / "correspondences.json"

    save_correspondences_json(correspondences, path)
    loaded = load_correspondences_json(path)

    assert loaded == correspondences


def test_court_boundary_polygon_expands_with_margin():
    cam_pos = np.array([0.0, -30.0, 7.0])
    correspondences, _ = _synthetic_correspondences(
        1100.0, 1100.0, 1280, 720, cam_pos, look_at=np.array([0.0, 0.0, 0.0])
    )
    calib = calibrate_camera(correspondences, (1280, 720))

    no_margin = court_boundary_polygon(calib, margin_m=0.0)
    with_margin = court_boundary_polygon(calib, margin_m=6.0)

    assert no_margin.shape == (4, 2)
    assert with_margin.shape == (4, 2)

    # The zero-margin polygon's corners must exactly match the projected
    # doubles-line corners; the actual net_center_ground landmark's known
    # world position lets us sanity-check the projection is at least in the
    # right place by re-deriving one corner directly.
    expected_corner = calib.project(np.array([[-_HALF_DOUBLES_WIDTH, -_HALF_LENGTH, 0.0]]))[0]
    np.testing.assert_allclose(no_margin[0], expected_corner, atol=0.5)

    # With a margin, every corner should sit farther from the court center
    # (in world space, which projecting preserves the ordering of for a
    # camera looking roughly straight down the court) than the no-margin one.
    center_px = calib.project(np.array([[0.0, 0.0, 0.0]]))[0]
    no_margin_dists = np.linalg.norm(no_margin - center_px, axis=1)
    with_margin_dists = np.linalg.norm(with_margin - center_px, axis=1)
    assert np.all(with_margin_dists > no_margin_dists)
