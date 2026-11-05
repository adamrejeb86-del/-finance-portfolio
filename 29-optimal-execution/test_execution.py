import numpy as np
import pytest

from execution import Market, ac_trajectory, expected_cost_variance, simulate_shortfall, vwap_trajectory

M = Market()


def test_zero_risk_aversion_gives_twap():
    x = ac_trajectory(M, 0.0)
    assert np.allclose(np.diff(x), -M.X / M.N)
    assert np.allclose(ac_trajectory(M, 1e-14), x, rtol=1e-5)


def test_trajectories_start_full_and_end_empty():
    for lam in (1e-7, 1e-6, 1e-5):
        x = ac_trajectory(M, lam)
        assert x[0] == pytest.approx(M.X) and x[-1] == pytest.approx(0, abs=1e-6) and np.all(np.diff(x) < 0)


def test_risk_aversion_front_loads_and_trades_cost_for_risk():
    slow, fast = ac_trajectory(M, 1e-7), ac_trajectory(M, 1e-5)
    assert fast[5] < slow[5]                            # sold more by the same time
    (E1, V1), (E2, V2) = expected_cost_variance(M, slow), expected_cost_variance(M, fast)
    assert E2 > E1 and V2 < V1


def test_closed_form_matches_simulation():
    x = ac_trajectory(M, 1e-6)
    E, V = expected_cost_variance(M, x)
    sim = simulate_shortfall(M, x, n_paths=40_000, seed=1)
    assert sim.mean() == pytest.approx(E, abs=4 * sim.std() / np.sqrt(len(sim)))
    assert sim.std() == pytest.approx(np.sqrt(V), rel=0.02)


def test_vwap_sells_everything():
    x = vwap_trajectory(M)
    assert x[0] == pytest.approx(M.X) and x[-1] == pytest.approx(0, abs=1e-6)
