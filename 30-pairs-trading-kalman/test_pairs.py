import numpy as np
import pandas as pd
import pytest

from pairs import (ADF_CRITICAL_5PCT, adf_stat, backtest_spread, engle_granger, half_life, kalman_hedge, rolling_z,
                   simulate_pair, zscore_positions)


def test_adf_separates_stationary_from_random_walk():
    rng = np.random.default_rng(0)
    e = rng.standard_normal(3000)
    ar = np.zeros(3000)
    for t in range(1, 3000):
        ar[t] = 0.9 * ar[t - 1] + e[t]
    assert adf_stat(ar) < ADF_CRITICAL_5PCT < adf_stat(np.cumsum(e))


def test_engle_granger_detects_cointegration_only_when_present():
    p, _ = simulate_pair(seed=1, beta_drift=0.0)
    q, _ = simulate_pair(seed=1, beta_drift=0.0, cointegrated=False)
    assert engle_granger(np.log(p["A"]), np.log(p["B"]))["cointegrated_5pct"]
    assert not engle_granger(np.log(q["A"]), np.log(q["B"]))["cointegrated_5pct"]


def test_half_life_recovered():
    p, _ = simulate_pair(n=6000, seed=2, half_life=15, beta_drift=0.0)
    eg = engle_granger(np.log(p["A"]), np.log(p["B"]))
    assert half_life(eg["residuals"]) == pytest.approx(15, rel=0.2)


def test_kalman_tracks_a_drifting_beta():
    p, beta = simulate_pair(seed=3, beta_drift=0.002)
    k = kalman_hedge(np.log(p["A"]), np.log(p["B"]))
    late = slice(500, None)
    static_err = np.abs(engle_granger(np.log(p["A"]), np.log(p["B"]))["beta"] - beta.values[late]).mean()
    kalman_err = np.abs(k["beta"][late] - beta.values[late]).mean()
    assert kalman_err < 0.5 * static_err


def test_positions_use_hysteresis():
    z = np.array([0, 2.5, 1.5, 0.8, 0.3, -2.5, -1.0, -0.2])
    assert list(zscore_positions(z)) == [0, -1, -1, -1, 0, 1, 1, 0]


def test_no_look_ahead_and_costs():
    p, _ = simulate_pair(seed=4)
    k = kalman_hedge(np.log(p["A"]), np.log(p["B"]))
    z = rolling_z(k["spread"])
    r0, _ = backtest_spread(p, k["beta"], z, cost_bps=0)
    r5, _ = backtest_spread(p, k["beta"], z, cost_bps=5)
    assert r5.sum() < r0.sum()
    shocked = p.copy(); shocked.iloc[2000:] *= 1.3
    ks = kalman_hedge(np.log(shocked["A"]), np.log(shocked["B"]))
    _, pos_a = backtest_spread(p, k["beta"], z)
    _, pos_b = backtest_spread(shocked, ks["beta"], rolling_z(ks["spread"]))
    assert np.array_equal(pos_a[:2000], pos_b[:2000])
