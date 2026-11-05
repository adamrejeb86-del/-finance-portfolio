import numpy as np
import pandas as pd
import pytest

from backtest import deflated_sharpe, long_only_signal, run_backtest, simulate_markets, stats, tsmom_signal

R = simulate_markets(252 * 8, seed=1)


def test_no_look_ahead():
    """Changing tomorrow's returns must not change today's position."""
    sig = tsmom_signal(R)
    base = run_backtest(R, sig, target_vol_portfolio=None, rebalance=None)["positions"]
    t = 1500
    shocked = R.copy(); shocked.iloc[t:] *= -3
    pos = run_backtest(shocked, tsmom_signal(shocked), target_vol_portfolio=None, rebalance=None)["positions"]
    assert np.allclose(pos.iloc[: t + 1], base.iloc[: t + 1])


def test_costs_reduce_returns_by_turnover_times_cost():
    sig = tsmom_signal(R)
    free = run_backtest(R, sig, cost_bps=0.0)
    costly = run_backtest(R, sig, cost_bps=5.0)
    assert (free["returns"] - costly["returns"]).sum() == pytest.approx(free["turnover"].sum() * 5e-4)


def test_portfolio_vol_target_is_roughly_met():
    res = run_backtest(R, long_only_signal(R), target_vol_portfolio=0.10)
    assert stats(res["returns"])["Volatility"] == pytest.approx(0.10, abs=0.025)


def test_random_walks_give_no_trend_edge_on_average():
    sharpes = [stats(run_backtest(m, tsmom_signal(m))["returns"])["Sharpe"]
               for m in (simulate_markets(252 * 10, trend_strength=0.0, seed=s) for s in range(6))]
    assert abs(np.mean(sharpes)) < 0.3


def test_deflated_sharpe_penalises_many_trials():
    trials = np.random.default_rng(0).normal(0, 0.02, 50)
    p_one, _ = deflated_sharpe(0.05, trials[:2], 2500)
    p_many, sr0 = deflated_sharpe(0.05, trials, 2500)
    assert p_many < p_one and sr0 > 0
