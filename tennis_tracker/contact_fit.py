"""Phase D: fit racket-contact time, position, and speed from two trajectories.

Rather than reading contact time off a frame-difference discontinuity (which
is what tennis_tracker.trajectory does for the classical pipeline), this
fits the incoming shot and the outgoing shot as two independent physics
trajectories (Phase C), then finds the time at which those two curves come
closest together. That time — and the speed each trajectory implies at that
instant — are fitted outputs of the optimization, not read off raw pixel
deltas, and come with real uncertainty from each fit's covariance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from tennis_tracker.calibration import CameraCalibration
from tennis_tracker.physics import TennisBallParams
from tennis_tracker.trajectory_fit import TrajectoryFitResult, fit_trajectory


@dataclass
class ContactEvent:
    contact_time: float
    contact_position: np.ndarray  # (3,) — midpoint of the two fits' positions at contact_time
    position_gap_m: float  # distance between the two fits' positions at contact_time; a fit-quality signal
    incoming_speed: float
    incoming_speed_std: float
    outgoing_speed: float
    outgoing_speed_std: float
    incoming_fit: TrajectoryFitResult
    outgoing_fit: TrajectoryFitResult


def fit_contact_event(
    pre_contact_observations: list[tuple[float, float, float]],
    post_contact_observations: list[tuple[float, float, float]],
    calibration: CameraCalibration,
    params: TennisBallParams | None = None,
    search_window: tuple[float, float] | None = None,
) -> ContactEvent:
    """Fit incoming/outgoing trajectories and locate where they meet.

    ``pre_contact_observations`` and ``post_contact_observations`` are each
    [(timestamp, pixel_x, pixel_y), ...] — the ball's detections before and
    after the suspected hit (there's typically a short gap right at contact
    from motion blur/racket occlusion, which is fine; each side is fit
    independently and extrapolated to close that gap).
    """
    incoming_fit = fit_trajectory(pre_contact_observations, calibration, params)
    outgoing_fit = fit_trajectory(post_contact_observations, calibration, params)

    if search_window is None:
        search_window = (pre_contact_observations[-1][0], post_contact_observations[0][0])
    lo, hi = search_window
    if lo > hi:
        lo, hi = hi, lo

    def gap(t: float) -> float:
        p1 = incoming_fit.position_at(t, params)
        p2 = outgoing_fit.position_at(t, params)
        return float(np.linalg.norm(p1 - p2))

    result = minimize_scalar(gap, bounds=(lo, hi), method="bounded", options={"xatol": 1e-5})
    contact_time = float(result.x)

    p1 = incoming_fit.position_at(contact_time, params)
    p2 = outgoing_fit.position_at(contact_time, params)
    contact_position = (p1 + p2) / 2

    incoming_speed, incoming_speed_std = incoming_fit.speed_at(contact_time, params)
    outgoing_speed, outgoing_speed_std = outgoing_fit.speed_at(contact_time, params)

    return ContactEvent(
        contact_time=contact_time,
        contact_position=contact_position,
        position_gap_m=float(np.linalg.norm(p1 - p2)),
        incoming_speed=incoming_speed,
        incoming_speed_std=incoming_speed_std,
        outgoing_speed=outgoing_speed,
        outgoing_speed_std=outgoing_speed_std,
        incoming_fit=incoming_fit,
        outgoing_fit=outgoing_fit,
    )
