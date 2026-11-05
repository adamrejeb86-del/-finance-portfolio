import numpy as np
import pytest

from voltrading import (realised_variance, short_straddle_hedged, short_variance_strategy, simulate_market, skew_smile,
                        variance_swap_pnl, variance_swap_strike)


def test_replication_recovers_flat_volatility():
    k = variance_swap_strike(100, 0.5, 0.02, lambda x: np.full_like(x, 0.25))
    assert np.sqrt(k) == pytest.approx(0.25, abs=2e-4)


def test_skew_makes_variance_more_expensive_than_atm():
    k = variance_swap_strike(100, 0.5, 0.02, skew_smile(0.20, skew=-0.3, curvature=0.5))
    assert np.sqrt(k) > 0.20


def test_realised_variance_of_constant_returns():
    prices = 100 * np.exp(np.cumsum(np.full(253, 0.01)))
    assert realised_variance(np.concatenate([[100], prices])) == pytest.approx(252 * 1e-4)


def test_variance_swap_pnl_in_vega_notional():
    # realised vol 25 vs strike 20, vega notional 100k: variance notional 2,500 per vol point squared
    assert variance_swap_pnl(0.25**2, 0.20**2, 100_000) == pytest.approx(100_000 / 40 * (625 - 400))


def test_short_variance_earns_the_premium_with_negative_skew():
    prices, _, implied = simulate_market(10, seed=1)
    pnl = np.array([x["pnl"] for x in short_variance_strategy(prices, implied)])
    assert pnl.mean() > 0 and ((pnl - pnl.mean()) ** 3).mean() < 0


def test_hedged_straddle_breaks_even_without_premium():
    prices, _, implied = simulate_market(20, seed=2, premium=0.0)
    with_premium = simulate_market(20, seed=2, premium=0.04)
    flat = np.mean([x["pnl"] for x in short_straddle_hedged(prices, implied)])
    rich = np.mean([x["pnl"] for x in short_straddle_hedged(with_premium[0], with_premium[2])])
    assert rich > flat
