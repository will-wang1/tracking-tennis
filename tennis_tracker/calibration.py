"""Phase A: camera calibration from known tennis court landmarks.

A tennis court's line positions are fixed, known real-world geometry (ITF
dimensions). If we know where a handful of those landmarks land in image
pixels, we can solve for the camera's full projection: where it sits, how
it's oriented, and its focal length. That's what turns 2D pixel detections
into something a 3D physics model (Phase B/C) can reason about.

Court coordinate system (meters): origin at the center of the court at
ground level under the net; X across the net (sideline direction), Y along
the length of the court (baseline direction), Z vertical (up).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import minimize_scalar

# ITF standard tennis court dimensions, in meters.
COURT_LENGTH = 23.77
DOUBLES_WIDTH = 10.97
SINGLES_WIDTH = 8.23
SERVICE_LINE_FROM_NET = 6.40
NET_HEIGHT_CENTER = 0.914
NET_HEIGHT_POST = 1.07
NET_POST_OFFSET_BEYOND_DOUBLES = 0.914

_HALF_LENGTH = COURT_LENGTH / 2
_HALF_DOUBLES_WIDTH = DOUBLES_WIDTH / 2
_HALF_SINGLES_WIDTH = SINGLES_WIDTH / 2
_NET_POST_X = _HALF_DOUBLES_WIDTH + NET_POST_OFFSET_BEYOND_DOUBLES

# Named court landmarks -> (X, Y, Z) in the court coordinate system.
# These are the points you'd click on in a frame to calibrate against.
COURT_LANDMARKS: dict[str, tuple[float, float, float]] = {
    "baseline_near_left_doubles": (-_HALF_DOUBLES_WIDTH, -_HALF_LENGTH, 0.0),
    "baseline_near_right_doubles": (_HALF_DOUBLES_WIDTH, -_HALF_LENGTH, 0.0),
    "baseline_far_left_doubles": (-_HALF_DOUBLES_WIDTH, _HALF_LENGTH, 0.0),
    "baseline_far_right_doubles": (_HALF_DOUBLES_WIDTH, _HALF_LENGTH, 0.0),
    "baseline_near_left_singles": (-_HALF_SINGLES_WIDTH, -_HALF_LENGTH, 0.0),
    "baseline_near_right_singles": (_HALF_SINGLES_WIDTH, -_HALF_LENGTH, 0.0),
    "baseline_far_left_singles": (-_HALF_SINGLES_WIDTH, _HALF_LENGTH, 0.0),
    "baseline_far_right_singles": (_HALF_SINGLES_WIDTH, _HALF_LENGTH, 0.0),
    "baseline_near_center_mark": (0.0, -_HALF_LENGTH, 0.0),
    "baseline_far_center_mark": (0.0, _HALF_LENGTH, 0.0),
    "service_near_left": (-_HALF_SINGLES_WIDTH, -SERVICE_LINE_FROM_NET, 0.0),
    "service_near_right": (_HALF_SINGLES_WIDTH, -SERVICE_LINE_FROM_NET, 0.0),
    "service_far_left": (-_HALF_SINGLES_WIDTH, SERVICE_LINE_FROM_NET, 0.0),
    "service_far_right": (_HALF_SINGLES_WIDTH, SERVICE_LINE_FROM_NET, 0.0),
    "service_near_center_t": (0.0, -SERVICE_LINE_FROM_NET, 0.0),
    "service_far_center_t": (0.0, SERVICE_LINE_FROM_NET, 0.0),
    "net_center_ground": (0.0, 0.0, 0.0),
    "net_center_top": (0.0, 0.0, NET_HEIGHT_CENTER),
    "net_post_left_top": (-_NET_POST_X, 0.0, NET_HEIGHT_POST),
    "net_post_right_top": (_NET_POST_X, 0.0, NET_HEIGHT_POST),
}

MIN_CALIBRATION_POINTS = 6


@dataclass
class CameraCalibration:
    fx: float
    fy: float
    cx: float
    cy: float
    rotation_matrix: np.ndarray  # 3x3, world-to-camera
    translation_vector: np.ndarray  # 3, world-to-camera
    reprojection_error_px: float

    @property
    def camera_matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float64)

    @property
    def rvec(self) -> np.ndarray:
        rvec, _ = cv2.Rodrigues(self.rotation_matrix)
        return rvec

    def camera_position_world(self) -> np.ndarray:
        """World-space (X, Y, Z) position of the camera."""
        return -self.rotation_matrix.T @ self.translation_vector

    def project(self, points_3d: np.ndarray) -> np.ndarray:
        """Project Nx3 world points to Nx2 pixel coordinates."""
        points_3d = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
        pixels, _ = cv2.projectPoints(
            points_3d, self.rvec, self.translation_vector, self.camera_matrix, None
        )
        return pixels.reshape(-1, 2)


def _reprojection_error(object_points: np.ndarray, image_points: np.ndarray, camera_matrix, rvec, tvec) -> float:
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, None)
    projected = projected.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum((projected - image_points) ** 2, axis=1))))


def calibrate_camera(
    point_correspondences: dict[str, tuple[float, float]],
    image_size: tuple[int, int],
    focal_search_bounds: tuple[float, float] | None = None,
) -> CameraCalibration:
    """Solve for camera position, orientation, and focal length from court landmarks.

    ``point_correspondences`` maps landmark names (from ``COURT_LANDMARKS``) to
    their pixel location in the frame. At least 6 well-spread points are
    needed for a stable solve. Distortion is assumed negligible (fine for
    typical broadcast/consumer lenses at moderate zoom).
    """
    if len(point_correspondences) < MIN_CALIBRATION_POINTS:
        raise ValueError(
            f"Need at least {MIN_CALIBRATION_POINTS} point correspondences, got {len(point_correspondences)}"
        )

    unknown = set(point_correspondences) - set(COURT_LANDMARKS)
    if unknown:
        raise ValueError(f"Unknown landmark name(s): {sorted(unknown)}")

    object_points = np.array(
        [COURT_LANDMARKS[name] for name in point_correspondences], dtype=np.float64
    )
    image_points = np.array(
        [point_correspondences[name] for name in point_correspondences], dtype=np.float64
    )

    width, height = image_size
    cx, cy = width / 2.0, height / 2.0
    low, high = focal_search_bounds or (0.3 * width, 5.0 * width)

    def solve_at_focal(f: float):
        camera_matrix = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(
            object_points, image_points, camera_matrix, None, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            return None
        error = _reprojection_error(object_points, image_points, camera_matrix, rvec, tvec)
        return error, camera_matrix, rvec, tvec

    result = minimize_scalar(
        lambda f: solve_at_focal(f)[0], bounds=(low, high), method="bounded",
        options={"xatol": 1e-2},
    )
    best_error, best_camera_matrix, best_rvec, best_tvec = solve_at_focal(result.x)

    rotation_matrix, _ = cv2.Rodrigues(best_rvec)
    fx = float(best_camera_matrix[0, 0])
    fy = float(best_camera_matrix[1, 1])

    return CameraCalibration(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        rotation_matrix=rotation_matrix,
        translation_vector=best_tvec.reshape(3),
        reprojection_error_px=best_error,
    )
