import numpy as np
import pytest

from hedging import bs_call, gamma_attribution, hedge_short_call, simulate_paths

S0, K, T, r, q, vol = 100.0, 100.0, 0.5, 0.03, 0.0, 0.20


def pnl(n_steps, sigma_real=vol, sigma_hedge=None, tx=0.0, n=40_000, seed=1):
    paths = simulate_paths(S0, 0.08, q, sigma_real, T, n_steps, n, seed)
    return hedge_short_call(paths, K, T, r, q, vol, sigma_hedge, tx), paths


def test_fair_vol_gives_zero_mean_pnl():
    x, _ = pnl(252)
    assert abs(x.mean()) < 4 * x.std() / np.sqrt(len(x)) + 0.01


def test_hedging_error_shrinks_like_one_over_sqrt_n():
    s1, s4 = pnl(50)[0].std(), pnl(200)[0].std()
    assert s1 / s4 == pytest.approx(2.0, rel=0.15)        # 4x more rebalancing -> half the error


def test_hedging_at_realised_vol_locks_in_the_vol_spread():
    """Hedge with the true volatility: the P&L converges to BS(implied) - BS(realised)."""
    x, _ = pnl(1000, sigma_real=0.25, sigma_hedge=0.25, n=20_000)
    locked = bs_call(S0, K, T, r, q, vol) - bs_call(S0, K, T, r, q, 0.25)
    assert x.mean() == pytest.approx(locked, abs=0.03)
    hedged_at_implied, _ = pnl(1000, sigma_real=0.25, n=20_000)
    assert x.std() < 0.5 * hedged_at_implied.std()     # only the discrete-rebalancing noise is left


def test_selling_vol_too_cheap_loses_money():
    assert pnl(252, sigma_real=0.30)[0].mean() < -1.0


def test_gamma_attribution_explains_the_pnl():
    x, paths = pnl(252, sigma_real=0.25, n=10_000)
    approx = gamma_attribution(paths, K, T, r, q, vol)
    assert np.corrcoef(x, approx)[0, 1] > 0.95


def test_transaction_costs_reduce_pnl():
    assert pnl(252, tx=0.002)[0].mean() < pnl(252)[0].mean()
