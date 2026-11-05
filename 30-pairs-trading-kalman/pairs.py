"""
Pairs trading: cointegration testing, dynamic hedge ratios with a Kalman filter, and a
mean-reversion strategy on the spread.

Two stocks exposed to the same fundamentals (two banks, two oil majors) can drift apart in the
short run but are pulled back together. If log A - beta log B - alpha is stationary, the pair is
**cointegrated** and the spread can be traded: sell it when it is unusually wide, buy it when it
is unusually narrow.

  1. Engle-Granger test: OLS hedge ratio, then an augmented Dickey-Fuller test on the residuals.
  2. Half-life of mean reversion from an AR(1) fit of the spread.
  3. Kalman filter: the hedge ratio and intercept follow random walks and are updated every
     day, so the strategy adapts when the relationship drifts.
  4. Z-score strategy with entry/exit bands, one-day execution lag and transaction costs.

Author: Adam Rejeb
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EG_CRITICAL_5PCT = -3.34      # Engle-Granger 5% critical value, two variables with constant (MacKinnon)
ADF_CRITICAL_5PCT = -2.86     # ADF with constant, no cointegration estimation


# ---------------------------------------------------------------- data
def simulate_pair(n=2520, seed=0, half_life=15.0, beta0=1.2, beta_drift=0.0015, cointegrated=True):
    """Log prices of two stocks. B follows a random walk; A = alpha + beta_t B + spread with an
    Ornstein-Uhlenbeck spread (or a random walk if cointegrated=False) and a slowly drifting beta."""
    rng = np.random.default_rng(seed)
    logB = np.log(50) + np.cumsum(rng.normal(0.0002, 0.015, n))
    beta = beta0 + np.cumsum(rng.normal(0, beta_drift, n))
    phi = np.exp(-np.log(2) / half_life)
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = (phi if cointegrated else 1.0) * spread[t - 1] + rng.normal(0, 0.008)
    logA = 0.4 + beta * logB + spread
    idx = pd.bdate_range("2016-01-04", periods=n)
    return pd.DataFrame({"A": np.exp(logA), "B": np.exp(logB)}, index=idx), pd.Series(beta, idx)


# ---------------------------------------------------------------- statistics
def ols(y, X):
    X = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    return coef, resid


def adf_stat(x, lags=1):
    """t-statistic of rho in  dx_t = c + rho x_{t-1} + sum phi_i dx_{t-i} + e_t."""
    x = np.asarray(x, float)
    dx = np.diff(x)
    y = dx[lags:]
    cols = [x[lags:-1]] + [dx[lags - i:-i] for i in range(1, lags + 1)]
    X = np.column_stack([np.ones(len(y))] + cols)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    s2 = resid @ resid / (len(y) - X.shape[1])
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    return float(coef[1] / se)


def engle_granger(logA, logB, lags=1):
    (alpha, beta), resid = ols(np.asarray(logA), np.asarray(logB))
    stat = adf_stat(resid, lags)
    return {"alpha": alpha, "beta": beta, "adf_stat": stat, "cointegrated_5pct": stat < EG_CRITICAL_5PCT, "residuals": resid}


def half_life(spread):
    s = np.asarray(spread, float)
    (c, phi), _ = ols(s[1:], s[:-1])
    return float(-np.log(2) / np.log(phi)) if 0 < phi < 1 else np.inf


# ---------------------------------------------------------------- Kalman filter
def kalman_hedge(logA, logB, delta_beta=1e-5, delta_alpha=1e-9, obs_var=7e-4):
    """State theta_t = [beta_t, alpha_t] following random walks. The hedge ratio is allowed to
    move (delta_beta) while the intercept is almost fixed (delta_alpha): if the intercept could
    move freely, the filter would absorb the very spread we want to trade.
    Observation: logA_t = beta_t logB_t + alpha_t + e_t.
    Returns the filtered states and the one-step-ahead spread logA_t - (beta_{t-1} logB_t + alpha_{t-1})."""
    y, x = np.asarray(logA, float), np.asarray(logB, float)
    n = len(y)
    theta = np.array([0.0, 0.0])
    P = np.diag([1.0, 1.0])
    Q = np.diag([delta_beta, delta_alpha]) / (1 - delta_beta)
    betas, alphas, spread = (np.empty(n) for _ in range(3))
    for t in range(n):
        H = np.array([x[t], 1.0])
        P = P + Q                                   # predict: the state may have drifted
        e = y[t] - H @ theta                        # spread measured with yesterday's hedge ratio
        S = H @ P @ H + obs_var
        K = P @ H / S                               # update
        theta = theta + K * e
        P = P - np.outer(K, H) @ P
        betas[t], alphas[t], spread[t] = theta[0], theta[1], e
    return {"beta": betas, "alpha": alphas, "spread": spread}


def rolling_z(x, window=60):
    """Z-score of x_t against the mean and std of the previous `window` observations (no look-ahead)."""
    s = pd.Series(np.asarray(x, float))
    mu = s.rolling(window).mean().shift(1)
    sd = s.rolling(window).std().shift(1)
    return ((s - mu) / sd).fillna(0.0).values


# ---------------------------------------------------------------- strategy
def zscore_positions(z, entry=2.0, exit=0.5):
    """+1 = long the spread (buy A, sell beta x B), -1 = short the spread. Hysteresis bands."""
    pos = np.zeros(len(z))
    for t in range(1, len(z)):
        p = pos[t - 1]
        if p == 0:
            p = -1.0 if z[t] > entry else (1.0 if z[t] < -entry else 0.0)
        elif (p == 1 and z[t] > -exit) or (p == -1 and z[t] < exit):
            p = 0.0
        pos[t] = p
    return pos


def backtest_spread(prices: pd.DataFrame, hedge_ratio, z, entry=2.0, exit=0.5, cost_bps=5.0, gross=1.0):
    """Daily returns of a spread position sized to `gross` (sum of absolute leg weights).
    Signals at today's close are traded tomorrow."""
    rA, rB = prices["A"].pct_change().fillna(0).values, prices["B"].pct_change().fillna(0).values
    pos = zscore_positions(np.asarray(z), entry, exit)
    h = np.asarray(hedge_ratio)
    wA = pos * gross / (1 + np.abs(h))
    wB = -pos * h * gross / (1 + np.abs(h))
    wA, wB = np.roll(wA, 1), np.roll(wB, 1)
    wA[0] = wB[0] = 0.0
    turnover = np.abs(np.diff(wA, prepend=0)) + np.abs(np.diff(wB, prepend=0))
    ret = wA * rA + wB * rB - turnover * cost_bps * 1e-4
    return pd.Series(ret, prices.index), pos


def stats(r: pd.Series) -> dict:
    wealth = (1 + r).cumprod()
    return {"Annual return": r.mean() * 252, "Volatility": r.std() * np.sqrt(252), "Sharpe": r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan,
            "Max drawdown": (wealth / wealth.cummax() - 1).min()}
