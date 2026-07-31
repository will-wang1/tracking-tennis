"""Phase C: recover a physics-constrained 3D trajectory from 2D pixel detections.

A single camera can't directly give 3D position from one frame — but a ball's
flight under known gravity + drag traces a very particular curve, and how
that curve's depth-dependent curvature projects into the image is enough to
disambiguate depth, *given* a calibrated camera (Phase A) and the physics
model (Phase B). This fits the 6 launch parameters (3D position + 3D velocity
at a reference time) via Levenberg-Marquardt nonlinear least squares, so the
"prediction" being fit isn't a generic curve — it's an actual physical flight
path, projected into the image and compared against every detection's pixel
coordinates and timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from tennis_tracker.calibration import CameraCalibration
from tennis_tracker.physics import TennisBallParams, simulate_trajectory


@dataclass
class TrajectoryFitResult:
    launch_state: np.ndarray  # [x,y,z,vx,vy,vz] at anchor_time
    anchor_time: float  # absolute timestamp that launch_state's t=0 corresponds to
    covariance: np.ndarray  # 6x6 parameter covariance from the fit
    residual_rms_px: float
    success: bool

    @property
    def param_std(self) -> np.ndarray:
        return np.sqrt(np.diag(self.covariance))

    def position_at(self, absolute_time: float, params: TennisBallParams | None = None) -> np.ndarray:
        return simulate_trajectory(self.launch_state, np.array([absolute_time - self.anchor_time]), params)[0]

    def velocity_at(self, absolute_time: float, params: TennisBallParams | None = None) -> np.ndarray:
        # Central difference on the physics model — cheap and accurate enough
        # since the model is smooth (no discontinuities away from impacts).
        dt = 1e-3
        p1 = self.position_at(absolute_time - dt, params)
        p2 = self.position_at(absolute_time + dt, params)
        return (p2 - p1) / (2 * dt)

    def speed_at(self, absolute_time: float, params: TennisBallParams | None = None) -> tuple[float, float]:
        """Speed (m/s) at ``absolute_time`` plus its 1-sigma uncertainty, propagated from the fit covariance."""
        velocity = self.velocity_at(absolute_time, params)
        speed = float(np.linalg.norm(velocity))

        # Numerical gradient of speed w.r.t. the 6 launch parameters, then
        # standard first-order error propagation: var(speed) = grad @ cov @ grad.
        def speed_given_launch_state(launch_state: np.ndarray) -> float:
            dt = 1e-3
            rel_times = np.array([absolute_time - self.anchor_time - dt, absolute_time - self.anchor_time + dt])
            p1, p2 = simulate_trajectory(launch_state, rel_times, params)
            velocity = (p2 - p1) / (2 * dt)
            return float(np.linalg.norm(velocity))

        gradient = np.zeros(6)
        step = 1e-4
        for i in range(6):
            plus = self.launch_state.copy()
            plus[i] += step
            minus = self.launch_state.copy()
            minus[i] -= step
            gradient[i] = (speed_given_launch_state(plus) - speed_given_launch_state(minus)) / (2 * step)

        variance = float(gradient @ self.covariance @ gradient)
        speed_std = float(np.sqrt(max(variance, 0.0)))
        return speed, speed_std


def _back_project_to_height(
    calibration: CameraCalibration, pixel: tuple[float, float], height: float
) -> np.ndarray:
    """Estimate a 3D point by intersecting the camera ray through ``pixel`` with the z=height plane."""
    camera_pos = calibration.camera_position_world()
    K_inv = np.linalg.inv(calibration.camera_matrix)
    pixel_h = np.array([pixel[0], pixel[1], 1.0])
    direction_camera = K_inv @ pixel_h
    direction_world = calibration.rotation_matrix.T @ direction_camera

    if abs(direction_world[2]) < 1e-9:
        direction_world[2] = 1e-9  # avoid division by zero for a near-horizontal ray
    t = (height - camera_pos[2]) / direction_world[2]
    return camera_pos + t * direction_world


def _initial_guess(
    times: np.ndarray, pixels: np.ndarray, calibration: CameraCalibration, assumed_height: float
) -> np.ndarray:
    first_pos = _back_project_to_height(calibration, pixels[0], assumed_height)
    last_pos = _back_project_to_height(calibration, pixels[-1], assumed_height)
    dt = times[-1] - times[0]
    velocity = (last_pos - first_pos) / dt if dt != 0 else np.zeros(3)
    return np.concatenate([first_pos, velocity])


def fit_trajectory(
    observations: list[tuple[float, float, float]],
    calibration: CameraCalibration,
    params: TennisBallParams | None = None,
    assumed_initial_height: float = 1.0,
) -> TrajectoryFitResult:
    """Fit the 6 launch parameters to ``observations`` = [(timestamp, pixel_x, pixel_y), ...].

    The anchor time (t=0 for the fitted launch state) is the first
    observation's timestamp; all other times are fit relative to it.
    """
    if len(observations) < 6:
        raise ValueError(f"Need at least 6 observations to fit 6 parameters, got {len(observations)}")

    observations = sorted(observations, key=lambda o: o[0])
    anchor_time = observations[0][0]
    times = np.array([t - anchor_time for t, _, _ in observations])
    observed_pixels = np.array([[px, py] for _, px, py in observations])

    x0 = _initial_guess(times, observed_pixels, calibration, assumed_initial_height)

    def residuals(launch_state):
        positions_3d = simulate_trajectory(launch_state, times, params)
        predicted_pixels = calibration.project(positions_3d)
        return (predicted_pixels - observed_pixels).ravel()

    result = least_squares(residuals, x0, method="lm", max_nfev=5000)

    n_residuals = len(result.fun)
    n_params = len(x0)
    dof = n_residuals - n_params
    residual_variance = float(np.sum(result.fun**2) / dof) if dof > 0 else float("nan")

    jacobian = result.jac
    try:
        covariance = residual_variance * np.linalg.inv(jacobian.T @ jacobian)
    except np.linalg.LinAlgError:
        covariance = residual_variance * np.linalg.pinv(jacobian.T @ jacobian)

    residual_rms_px = float(np.sqrt(np.mean(result.fun**2)))

    return TrajectoryFitResult(
        launch_state=result.x,
        anchor_time=anchor_time,
        covariance=covariance,
        residual_rms_px=residual_rms_px,
        success=result.success,
    )
