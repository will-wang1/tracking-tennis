"""Phase B: 3D tennis ball flight under gravity + quadratic aerodynamic drag.

Unlike a shuttlecock, a tennis ball's mass-to-drag ratio is high enough that
gravity dominates for most of a normal shot's flight, but drag still matters
over the full trajectory and for its true terminal velocity (~22 m/s,
compared to a shuttlecock's ~6.7 m/s — the shuttle is far more drag-dominated
by comparison). There's no closed-form solution with quadratic drag, so the
equations of motion are integrated numerically.

Coordinates match tennis_tracker.calibration: X across the net, Y along the
court length, Z vertical (up). Gravity is enabled by default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

# Standard tennis ball physical constants.
DEFAULT_MASS_KG = 0.058
DEFAULT_RADIUS_M = 0.033
DEFAULT_DRAG_COEFFICIENT = 0.55
DEFAULT_AIR_DENSITY = 1.204  # kg/m^3 at ~20C, sea level
GRAVITY = 9.81


@dataclass
class TennisBallParams:
    mass: float = DEFAULT_MASS_KG
    radius: float = DEFAULT_RADIUS_M
    drag_coefficient: float = DEFAULT_DRAG_COEFFICIENT
    air_density: float = DEFAULT_AIR_DENSITY
    gravity: float = GRAVITY

    @property
    def cross_section_area(self) -> float:
        return float(np.pi * self.radius**2)

    @property
    def drag_k(self) -> float:
        """Coefficient such that drag deceleration = drag_k * |v| * v."""
        return 0.5 * self.air_density * self.drag_coefficient * self.cross_section_area / self.mass

    @property
    def terminal_velocity(self) -> float:
        """Speed at which drag exactly balances gravity in free fall (m/s)."""
        return float(np.sqrt(self.gravity / self.drag_k))


def _acceleration(velocity: np.ndarray, params: TennisBallParams) -> np.ndarray:
    speed = float(np.linalg.norm(velocity))
    drag = -params.drag_k * speed * velocity
    gravity_accel = np.array([0.0, 0.0, -params.gravity])
    return gravity_accel + drag


def _deriv(t: float, state: np.ndarray, params: TennisBallParams) -> np.ndarray:
    velocity = state[3:6]
    acceleration = _acceleration(velocity, params)
    return np.concatenate([velocity, acceleration])


def simulate_trajectory(
    launch_state: np.ndarray, times: np.ndarray, params: TennisBallParams | None = None
) -> np.ndarray:
    """Integrate the ball's flight from ``launch_state`` (at t=0) and sample it at ``times``.

    ``launch_state`` is [x, y, z, vx, vy, vz] at t=0 — the "six launch
    parameters" that, together with the physics constants above, fully
    determine the 3D path. ``times`` may include values before t=0 (the
    filter integrates backward as needed) and need not be sorted; results
    are returned in the same order as ``times``.
    """
    params = params or TennisBallParams()
    launch_state = np.asarray(launch_state, dtype=float)
    times = np.asarray(times, dtype=float)

    positions = np.empty((len(times), 3))

    # solve_ivp's initial condition applies at t_span[0], so times before the
    # t=0 launch reference must be integrated as a separate backward pass
    # (t_span[1] < t_span[0] is a valid, decreasing integration direction) —
    # a single call spanning both directions would anchor launch_state at
    # the wrong end of the span.
    zero_mask = times == 0.0
    positions[zero_mask] = launch_state[:3]

    forward_mask = times > 0.0
    if np.any(forward_mask):
        forward_times = times[forward_mask]
        order = np.argsort(forward_times)
        solution = solve_ivp(
            _deriv, (0.0, float(forward_times.max())), launch_state,
            t_eval=forward_times[order], args=(params,), method="RK45", rtol=1e-9, atol=1e-9,
        )
        if not solution.success:
            raise RuntimeError(f"Trajectory integration failed: {solution.message}")
        forward_positions = np.empty((len(forward_times), 3))
        forward_positions[order] = solution.y[:3].T
        positions[forward_mask] = forward_positions

    backward_mask = times < 0.0
    if np.any(backward_mask):
        backward_times = times[backward_mask]
        order = np.argsort(backward_times)[::-1]  # descending, toward t_span[1]
        solution = solve_ivp(
            _deriv, (0.0, float(backward_times.min())), launch_state,
            t_eval=backward_times[order], args=(params,), method="RK45", rtol=1e-9, atol=1e-9,
        )
        if not solution.success:
            raise RuntimeError(f"Trajectory integration failed: {solution.message}")
        backward_positions = np.empty((len(backward_times), 3))
        backward_positions[order] = solution.y[:3].T
        positions[backward_mask] = backward_positions

    return positions
