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

import json
from dataclasses import dataclass
from pathlib import Path

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

# Real play routinely extends well beyond the doubles lines -- players stand
# several meters behind the baseline to serve or return, and run wide past
# the sideline for an angled shot -- so the raw court rectangle would wrongly
# exclude real players. This margin defines a bigger "play area" rectangle
# instead, generous enough to keep genuine play in bounds while still
# excluding people clearly off to the side (ball kids, umpire chair,
# spectators).
DEFAULT_COURT_MARGIN_M = 6.0


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


def court_boundary_polygon(calibration: CameraCalibration, margin_m: float = DEFAULT_COURT_MARGIN_M) -> np.ndarray:
    """Project a court-plus-play-area rectangle (doubles lines expanded by ``margin_m``) into pixel coordinates.

    Returns the 4 corners as an (4, 2) pixel-coordinate polygon, in order
    (near-left, near-right, far-right, far-left) -- suitable for
    cv2.pointPolygonTest or as tennis_tracker.pose.track_poses'
    ``court_polygon_px`` argument.
    """
    half_length = _HALF_LENGTH + margin_m
    half_width = _HALF_DOUBLES_WIDTH + margin_m
    corners_world = np.array(
        [
            [-half_width, -half_length, 0.0],
            [half_width, -half_length, 0.0],
            [half_width, half_length, 0.0],
            [-half_width, half_length, 0.0],
        ]
    )
    return calibration.project(corners_world)


def save_correspondences_json(correspondences: dict[str, tuple[float, float]], path: str | Path) -> None:
    """Persist landmark-name -> pixel-position correspondences, e.g. from an interactive click session."""
    with Path(path).open("w") as f:
        json.dump({name: list(pixel) for name, pixel in correspondences.items()}, f, indent=2)


def load_correspondences_json(path: str | Path) -> dict[str, tuple[float, float]]:
    with Path(path).open() as f:
        data = json.load(f)
    return {name: tuple(pixel) for name, pixel in data.items()}


def collect_correspondences_interactive(
    reference_frame: np.ndarray, landmark_names: list[str] | None = None
) -> dict[str, tuple[float, float]]:
    """Click each named court landmark, one at a time, on a reference frame.

    Requires a real display (cv2.imshow) — not runnable headlessly. Prompts
    for each landmark in turn; left-click sets its pixel position, 's' skips
    a landmark that isn't visible in this frame (occluded/out of shot), and
    at least MIN_CALIBRATION_POINTS must be set for calibrate_camera to work.
    """
    import cv2 as _cv2  # local import: this function is display-only and untestable headlessly

    names = landmark_names or list(COURT_LANDMARKS.keys())
    correspondences: dict[str, tuple[float, float]] = {}
    clicked = {"pos": None}

    def on_mouse(event, x, y, flags, param):
        if event == _cv2.EVENT_LBUTTONDOWN:
            clicked["pos"] = (float(x), float(y))

    window_name = "Court Calibration  (click landmark, s=skip, q=finish early)"
    _cv2.namedWindow(window_name)
    _cv2.setMouseCallback(window_name, on_mouse)

    try:
        for name in names:
            clicked["pos"] = None
            while clicked["pos"] is None:
                frame = reference_frame.copy()
                _cv2.putText(
                    frame, f"Click: {name}  ({len(correspondences)} placed so far)",
                    (10, 25), _cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )
                _cv2.imshow(window_name, frame)
                key = _cv2.waitKey(20) & 0xFF
                if key == ord("s"):
                    break
                if key == ord("q"):
                    return correspondences
            if clicked["pos"] is not None:
                correspondences[name] = clicked["pos"]
    finally:
        _cv2.destroyAllWindows()

    return correspondences


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Camera calibration from tennis court landmarks.")
    sub = parser.add_subparsers(dest="command", required=True)

    collect_p = sub.add_parser("collect", help="Interactively click landmarks on a video frame (needs a display).")
    collect_p.add_argument("video", help="Path to the source video file.")
    collect_p.add_argument("output_json", help="Path to save the clicked correspondences to.")
    collect_p.add_argument("--frame", type=int, default=0, help="Which frame to use as the reference frame.")

    solve_p = sub.add_parser("solve", help="Solve calibration from a saved correspondences JSON and print it.")
    solve_p.add_argument("correspondences_json", help="Path to a correspondences JSON from 'collect'.")
    solve_p.add_argument("video", help="Source video, used only to read the frame size.")

    args = parser.parse_args(argv)

    if args.command == "collect":
        cap = cv2.VideoCapture(str(args.video))
        for _ in range(args.frame + 1):
            ok, frame = cap.read()
            if not ok:
                break
        cap.release()
        if not ok:
            raise RuntimeError(f"Could not read frame {args.frame} from {args.video}")
        correspondences = collect_correspondences_interactive(frame)
        save_correspondences_json(correspondences, args.output_json)
        print(f"Saved {len(correspondences)} correspondences to {args.output_json}")
        return 0

    if args.command == "solve":
        cap = cv2.VideoCapture(str(args.video))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        correspondences = load_correspondences_json(args.correspondences_json)
        calib = calibrate_camera(correspondences, (width, height))
        print(f"fx={calib.fx:.2f} fy={calib.fy:.2f} cx={calib.cx:.2f} cy={calib.cy:.2f}")
        print(f"camera position (world, m): {calib.camera_position_world()}")
        print(f"reprojection error: {calib.reprojection_error_px:.3f} px")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
