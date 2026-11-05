"""
Dispersion trading: implied correlation, the correlation risk premium, and a short-correlation
strategy with variance swaps.

The variance of an index is set by the variances of its members and by how they move together:
    sigma_I^2 = sum_i w_i^2 sigma_i^2 + sum_{i != j} w_i w_j rho_ij sigma_i sigma_j.
Given the index implied vol and the members' implied vols, the **implied correlation** is the
single rho that makes the equation hold. Index options are in demand as portfolio insurance, so
implied correlation is usually above the correlation that is then realised. A dispersion trade
sells index variance and buys single-stock variance to collect that premium. It is short
correlation: it earns small amounts most months and loses heavily when correlations jump in a
sell-off.

Market data are simulated with calm and stressed regimes; implied levels are set from the
regime prevailing when each monthly trade is struck.

Author: Adam Rejeb
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- correlation algebra
def index_vol_from_corr(vols, weights, rho):
    """Index volatility for a single average pairwise correlation."""
    v, w = np.asarray(vols, float), np.asarray(weights, float)
    wv = w * v
    var = (wv**2).sum() + rho * (wv.sum() ** 2 - (wv**2).sum())
    return np.sqrt(var)


def implied_correlation(index_vol, vols, weights):
    """Average pairwise correlation implied by index and single-stock volatilities."""
    v, w = np.asarray(vols, float), np.asarray(weights, float)
    wv = w * v
    own = (wv**2).sum()
    return (index_vol**2 - own) / (wv.sum() ** 2 - own)


# ---------------------------------------------------------------- simulated market
@dataclass(frozen=True)
class Regimes:
    corr: tuple = (0.30, 0.75)          # calm, stress
    vol_mult: tuple = (1.0, 1.9)
    stay: tuple = (0.995, 0.97)         # daily probability of staying in the regime
    base_vols: tuple = (0.20, 0.22, 0.24, 0.25, 0.26, 0.28, 0.30, 0.32, 0.35, 0.38)


def simulate_market(n_days=2520, seed=0, reg: Regimes = Regimes()):
    """Daily returns of 10 stocks with regime-switching correlation and volatility.
    Returns (returns DataFrame, regime Series with 0 = calm, 1 = stress)."""
    rng = np.random.default_rng(seed)
    base = np.array(reg.base_vols)
    k = len(base)
    state = np.zeros(n_days, int)
    for t in range(1, n_days):
        s = state[t - 1]
        state[t] = s if rng.random() < reg.stay[s] else 1 - s
    chol = [np.linalg.cholesky(np.full((k, k), c) + (1 - c) * np.eye(k)) for c in reg.corr]
    z = rng.standard_normal((n_days, k))
    rets = np.empty((n_days, k))
    for t in range(n_days):
        s = state[t]
        rets[t] = (chol[s] @ z[t]) * base * reg.vol_mult[s] / np.sqrt(252)
    idx = pd.bdate_range("2016-01-04", periods=n_days)
    cols = [f"Stock {i + 1}" for i in range(k)]
    return pd.DataFrame(rets, idx, cols), pd.Series(state, idx, name="stress")


def realised_vol(returns):
    return np.sqrt(252 * (np.asarray(returns) ** 2).mean(axis=0))


# ---------------------------------------------------------------- variance swaps
def var_swap_pnl(vega_notional, strike_vol, realised):
    """P&L of a long variance swap with the usual vega-notional convention (vols in %):
    variance notional = vega / (2 x strike), P&L = variance notional x (realised^2 - strike^2).
    Close to vega x (realised - strike) for small moves, convex for large ones."""
    k, r = 100 * np.asarray(strike_vol), 100 * np.asarray(realised)
    return vega_notional / (2 * k) * (r**2 - k**2)


def dispersion_weights(index_vega, index_vol, vols, weights, scheme="vega"):
    """Vega notionals of the long single-stock legs against a short index leg of `index_vega`.
    - "vega": the single-stock vegas sum to the index vega. Vega-flat on paper, but net long
      volatility: the members' vols are higher than the index vol, so a general rise in vols
      makes the long legs gain more than the short leg loses.
    - "correlation": scaled so that a proportional move of every vol, with correlation unchanged,
      has no first-order P&L (sum V_i sigma_i = V_I sigma_I). What is left is a bet on correlation."""
    w = np.asarray(weights, float) / np.sum(weights)
    if scheme == "vega":
        return index_vega * w
    v = np.asarray(vols, float)
    return index_vega * index_vol / (w * v).sum() * w


# ---------------------------------------------------------------- monthly strategy
def expected_variances(state_now, reg: Regimes, weights, horizon=21):
    """Expected average variance over the next `horizon` days for each stock and for the index,
    given the current regime (exact Markov-chain expectation). This is the fair strike before any
    risk premium."""
    P = np.array([[reg.stay[0], 1 - reg.stay[0]], [1 - reg.stay[1], reg.stay[1]]])
    p = np.eye(2)[state_now]
    occ = np.zeros(2)
    for _ in range(horizon):
        p = p @ P
        occ += p / horizon
    base = np.array(reg.base_vols)
    single = base**2 * (occ[0] * reg.vol_mult[0] ** 2 + occ[1] * reg.vol_mult[1] ** 2)
    index = sum(occ[s] * index_vol_from_corr(base * reg.vol_mult[s], weights, reg.corr[s]) ** 2 for s in (0, 1))
    return single, index


def run_dispersion(returns: pd.DataFrame, regime: pd.Series, reg: Regimes = Regimes(), index_vega=10_000.0,
                   single_premium=0.03, index_premium=0.12, scheme="vega", horizon=21):
    """Roll one-month variance swaps: short the index, long the members, struck every month.
    Implied vols = fair vols x (1 + premium); the index carries a larger premium than the stocks
    (demand for index protection), which is the source of the correlation risk premium."""
    k = returns.shape[1]
    w = np.full(k, 1 / k)
    rows = []
    for start in range(1, len(returns) - horizon + 1, horizon):
        s_now = int(regime.iloc[start - 1])
        e_single, e_index = expected_variances(s_now, reg, w, horizon)
        iv = np.sqrt(e_single) * (1 + single_premium)
        iv_index = np.sqrt(e_index) * (1 + index_premium)
        window = returns.iloc[start:start + horizon]
        rv = realised_vol(window.values)
        rv_index = realised_vol(window.values @ w)
        vegas = dispersion_weights(index_vega, iv_index, iv, w, scheme)
        pnl_single = var_swap_pnl(vegas, iv, rv).sum()
        pnl_index = -var_swap_pnl(index_vega, iv_index, rv_index)
        rows.append({"date": window.index[0], "stress at start": s_now, "stress share": regime.iloc[start:start + horizon].mean(),
                     "implied index vol": iv_index, "realised index vol": rv_index,
                     "implied corr": implied_correlation(iv_index, iv, w), "realised corr": implied_correlation(rv_index, rv, w),
                     "P&L single legs": pnl_single, "P&L index leg": pnl_index, "P&L": pnl_single + pnl_index})
    return pd.DataFrame(rows).set_index("date")


def strategy_stats(pnl: pd.Series, periods_per_year=12) -> dict:
    cum = pnl.cumsum()
    return {"Annual P&L (EUR)": pnl.mean() * periods_per_year, "Annual vol (EUR)": pnl.std() * np.sqrt(periods_per_year),
            "Sharpe": pnl.mean() / pnl.std() * np.sqrt(periods_per_year), "Positive months": (pnl > 0).mean(),
            "Worst month (EUR)": pnl.min(), "Skewness": pnl.skew(), "Max drawdown (EUR)": (cum - cum.cummax()).min()}
