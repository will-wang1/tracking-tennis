import numpy as np
import pytest
from scipy.integrate import solve_ivp

from tennis_tracker.physics import TennisBallParams, _deriv, simulate_trajectory


def test_terminal_velocity_is_in_expected_tennis_ball_range():
    params = TennisBallParams()
    # Real tennis balls have a terminal velocity around 20-23 m/s (much higher
    # than a badminton shuttle's ~6.7 m/s, since a tennis ball is far less
    # drag-dominated relative to its mass).
    assert 20.0 < params.terminal_velocity < 23.0


def test_free_fall_velocity_converges_to_terminal_velocity():
    params = TennisBallParams()
    launch = np.array([0.0, 0.0, 500.0, 0.0, 0.0, 0.0])
    solution = solve_ivp(_deriv, (0, 20), launch, t_eval=[20.0], args=(params,), rtol=1e-9, atol=1e-9)

    vz_final = solution.y[5, -1]
    assert abs(vz_final) == pytest.approx(params.terminal_velocity, rel=1e-3)


def test_negligible_drag_matches_textbook_projectile_motion():
    tiny_drag = TennisBallParams(drag_coefficient=1e-9)
    launch = np.array([0.0, 0.0, 1.0, 10.0, 0.0, 15.0])
    times = np.array([0.5, 1.0, 1.5])

    positions = simulate_trajectory(launch, times, params=tiny_drag)

    for t, pos in zip(times, positions):
        expected_x = 10.0 * t
        expected_z = 1.0 + 15.0 * t - 0.5 * 9.81 * t**2
        assert pos[0] == pytest.approx(expected_x, abs=1e-3)
        assert pos[2] == pytest.approx(expected_z, abs=1e-3)


def test_drag_reduces_horizontal_distance_versus_no_drag():
    launch = np.array([0.0, 0.0, 1.0, 30.0, 0.0, 5.0])
    times = np.array([1.0])

    with_drag = simulate_trajectory(launch, times, params=TennisBallParams())
    without_drag = simulate_trajectory(launch, times, params=TennisBallParams(drag_coefficient=1e-9))

    assert with_drag[0, 0] < without_drag[0, 0]


def test_simulate_trajectory_handles_unsorted_and_negative_times():
    launch = np.array([0.0, 0.0, 1.0, 10.0, 0.0, 15.0])
    times = np.array([1.0, -0.5, 0.5, 0.0])

    positions = simulate_trajectory(launch, times)

    # t=0 should return exactly the launch position regardless of input order.
    zero_index = list(times).index(0.0)
    np.testing.assert_allclose(positions[zero_index], launch[:3], atol=1e-6)
