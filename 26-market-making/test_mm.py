from dataclasses import replace

import numpy as np
import pytest

from mm import Params, optimal_spread, reservation_price, simulate

P = Params()


def test_reservation_price_skews_against_inventory():
    assert reservation_price(100, 5, 0.0, P) < 100 < reservation_price(100, -5, 0.0, P)
    assert reservation_price(100, 5, P.T, P) == pytest.approx(100)       # no skew at the horizon


def test_spread_tends_to_2_over_k_without_risk_aversion():
    p = replace(P, gamma=1e-8)
    assert optimal_spread(0.0, p) == pytest.approx(2 / P.k, rel=1e-4)


def test_fill_rate_matches_intensity():
    """With a fixed spread, the number of fills per side follows A exp(-k delta) T."""
    p = replace(P, gamma=1e-8)
    res = simulate(p, "symmetric", n_paths=2000, seed=1)
    expected = 2 * P.A * np.exp(-P.k * optimal_spread(0, p) / 2) * P.T
    assert res["trades"].mean() == pytest.approx(expected, rel=0.03)


def test_inventory_strategy_controls_risk():
    inv = simulate(P, "inventory", n_paths=2000, seed=2)
    sym = simulate(P, "symmetric", n_paths=2000, seed=2)
    assert inv["inventory"].std() < 0.5 * sym["inventory"].std()
    assert inv["pnl"].std() < sym["pnl"].std()


def test_inventory_limit_is_respected():
    res = simulate(replace(P, max_inventory=3), "symmetric", n_paths=500, seed=3)
    assert np.abs(res["inventory"]).max() <= 3
