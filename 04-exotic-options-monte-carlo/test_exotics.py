import numpy as np
import pytest

from exotics import (asian_call_mc, barrier_call_mc, bs_call, brownian_bridge, down_and_out_call,
                     down_and_out_call_discrete, geometric_asian_call, gbm_paths, lookback_call_mc,
                     lookback_floating_call)

S, K, T, r, q, sig = 100.0, 100.0, 1.0, 0.04, 0.01, 0.25


def test_brownian_bridge_has_the_right_covariance():
    z = np.random.default_rng(0).standard_normal((200_000, 8))
    W = brownian_bridge(z, 1.0)
    t = np.linspace(1 / 8, 1, 8)
    assert np.cov(W.T) == pytest.approx(np.minimum.outer(t, t), abs=0.01)   # Cov(W_s, W_t) = min(s, t)


def test_paths_are_martingales_after_discounting():
    paths = gbm_paths(S, r, q, sig, T, 12, 400_000, seed=1, antithetic=True)
    assert np.exp(-(r - q) * T) * paths[:, -1].mean() == pytest.approx(S, rel=2e-3)


def test_geometric_asian_closed_form_matches_simulation():
    paths = gbm_paths(S, r, q, sig, T, 12, 400_000, seed=2)
    geo = np.exp(-r * T) * np.maximum(np.exp(np.log(paths[:, 1:]).mean(axis=1)) - K, 0)
    assert geo.mean() == pytest.approx(geometric_asian_call(S, K, T, r, q, sig, 12),
                                       abs=4 * geo.std() / np.sqrt(len(geo)))


def test_variance_reduction_methods_agree_and_reduce_error():
    plain = asian_call_mc(S, K, T, r, q, sig, 12, 65_536, method="plain")
    control = asian_call_mc(S, K, T, r, q, sig, 12, 65_536, method="control")
    sobol = asian_call_mc(S, K, T, r, q, sig, 12, 65_536, method="sobol")
    assert control[1] < plain[1] / 10 and sobol[1] < plain[1] / 10
    assert control[0] == pytest.approx(sobol[0], abs=4 * np.hypot(control[1], sobol[1]))


def test_in_out_parity_is_exact():
    res = barrier_call_mc(S, K, 90, T, r, q, sig, 52, 100_000)
    assert res["in"][0] + res["out"][0] == pytest.approx(res["vanilla"][0], abs=1e-10)


def test_bgk_correction_matches_discrete_barrier():
    mc, se = barrier_call_mc(S, K, 90, T, r, q, sig, 52, 400_000)["out"]
    assert down_and_out_call_discrete(S, K, 90, T, r, q, sig, 52) == pytest.approx(mc, abs=4 * se + 0.02)
    assert down_and_out_call(S, K, 90, T, r, q, sig) < mc        # continuous barrier knocks out more often


def test_barrier_far_away_gives_vanilla():
    assert down_and_out_call(S, K, 1e-6, T, r, q, sig) == pytest.approx(bs_call(S, K, T, r, q, sig), abs=1e-10)


def test_lookback_with_continuity_correction():
    mc, se = lookback_call_mc(S, T, r, q, sig, 1000, 200_000)
    assert mc == pytest.approx(lookback_floating_call(S, T, r, q, sig), abs=4 * se + 0.03)
