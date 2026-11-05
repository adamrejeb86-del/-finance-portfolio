"""
A vectorised backtesting engine for systematic multi-asset strategies, built to avoid the
classic mistakes: look-ahead bias, ignored costs and overfitting.

Engine
  - Signals computed with data up to the close of day t are traded at t+1 (one-day lag).
  - Positions sized by volatility targeting (EWMA volatility) so that every market
    contributes a similar risk; optional portfolio-level volatility target.
  - Proportional transaction costs charged on turnover.

Strategies
  - Time-series momentum (trend following): sign of the past 12-month return
  - Moving-average crossover (50 / 200 days)
  - Cross-sectional momentum: long the strongest third, short the weakest third
  - Benchmark: long-only, equal risk

Evaluation
  - CAGR, volatility, Sharpe, Sortino, max drawdown, Calmar, skewness, turnover
  - Parameter sensitivity, and the Deflated Sharpe Ratio (Bailey and Lopez de Prado, 2014),
    which corrects the best backtest for the number of variants tried.

Author: Adam Rejeb
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis

MARKETS = ["S&P 500", "Euro Stoxx 50", "Nikkei 225", "US 10Y", "Bund", "JGB", "EUR/USD", "USD/JPY", "GBP/USD",
           "Gold", "Crude oil", "Copper"]
VOLS = np.array([0.17, 0.20, 0.21, 0.06, 0.055, 0.03, 0.08, 0.09, 0.085, 0.15, 0.32, 0.24])


# ---------------------------------------------------------------- data
def simulate_markets(n_days=252 * 20, trend_strength=1.0, seed=0) -> pd.DataFrame:
    """Daily returns with slowly varying hidden drifts (trends), so that trend following
    has a modest, realistic edge. trend_strength=0 gives pure random walks."""
    rng = np.random.default_rng(seed)
    n = len(MARKETS)
    corr = np.full((n, n), 0.15); np.fill_diagonal(corr, 1.0)
    corr[:3, :3] = 0.7; corr[3:6, 3:6] = 0.6
    for i in range(3):
        corr[i, i] = 1; corr[3 + i, 3 + i] = 1
    L = np.linalg.cholesky(corr)
    daily_vol = VOLS / np.sqrt(252)
    drift = np.zeros(n)
    phi = 1 - 1 / 250                                    # trends persist for about a year
    drift_sd = 0.9 * VOLS / 252 * trend_strength         # hidden trend worth a Sharpe of about 0.9 per market
    out = np.empty((n_days, n))
    for t in range(n_days):
        drift = phi * drift + np.sqrt(1 - phi**2) * drift_sd * rng.standard_normal(n)
        z = rng.standard_t(5, n) * np.sqrt(3 / 5)
        out[t] = drift + daily_vol * (L @ z)
    return pd.DataFrame(out, index=pd.bdate_range("2006-01-02", periods=n_days), columns=MARKETS)


# ---------------------------------------------------------------- signals
def tsmom_signal(returns: pd.DataFrame, lookback=252) -> pd.DataFrame:
    return np.sign(returns.rolling(lookback).sum())


def ma_crossover_signal(returns: pd.DataFrame, fast=50, slow=200) -> pd.DataFrame:
    price = (1 + returns).cumprod()
    return np.sign(price.rolling(fast).mean() - price.rolling(slow).mean())


def xs_momentum_signal(returns: pd.DataFrame, lookback=252, quantile=1 / 3) -> pd.DataFrame:
    """Long the best third, short the worst third, on risk-adjusted past returns."""
    score = returns.rolling(lookback).sum() / (returns.rolling(lookback).std() * np.sqrt(lookback))
    ranks = score.rank(axis=1, pct=True)
    return (ranks > 1 - quantile).astype(float) - (ranks <= quantile).astype(float)


def long_only_signal(returns: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(1.0, index=returns.index, columns=returns.columns)


# ---------------------------------------------------------------- engine
def ewma_vol(returns: pd.DataFrame, halflife=60) -> pd.DataFrame:
    return np.sqrt((returns**2).ewm(halflife=halflife, min_periods=60).mean() * 252)


def run_backtest(returns: pd.DataFrame, signal: pd.DataFrame, target_vol_asset=0.10, target_vol_portfolio=0.10,
                 cost_bps=2.0, rebalance="W-FRI") -> dict:
    """Returns daily strategy returns, positions and turnover.

    1. Position_i = signal_i x (target vol / EWMA vol_i) / n_markets, decided at the close.
    2. Positions are held constant between rebalancing dates (weekly by default).
    3. Positions are applied to the NEXT day's returns (no look-ahead).
    4. Optional scaling of the whole book to a portfolio volatility target (using past data only).
    5. Costs = cost_bps x turnover (sum of absolute position changes).
    """
    vol = ewma_vol(returns)
    raw = signal * (target_vol_asset / vol) / returns.shape[1]
    if rebalance:
        mask = pd.Series(returns.index, index=returns.index).groupby(pd.Grouper(freq=rebalance)).transform("last") == returns.index
        raw = raw.where(mask).ffill()
    raw = raw.fillna(0.0)
    if target_vol_portfolio:
        gross = (raw.shift(1) * returns).sum(axis=1)
        realised = gross.ewm(halflife=60, min_periods=60).std() * np.sqrt(252)
        scale = (target_vol_portfolio / realised).clip(upper=5).shift(1).fillna(0.0)
        raw = raw.mul(scale, axis=0)
    positions = raw.shift(1).fillna(0.0)                            # trade at t+1
    turnover = positions.diff().abs().sum(axis=1).fillna(0.0)
    pnl = (positions * returns).sum(axis=1) - turnover * cost_bps * 1e-4
    return {"returns": pnl, "positions": positions, "turnover": turnover}


# ---------------------------------------------------------------- evaluation
def stats(r: pd.Series, turnover: pd.Series | None = None) -> dict:
    r = r[r.index >= r.ne(0).idxmax()]                              # drop the warm-up period
    wealth = (1 + r).cumprod()
    dd = wealth / wealth.cummax() - 1
    years = len(r) / 252
    cagr = wealth.iloc[-1] ** (1 / years) - 1
    vol = r.std() * np.sqrt(252)
    downside = np.sqrt((np.minimum(r, 0) ** 2).mean()) * np.sqrt(252)
    out = {"CAGR": cagr, "Volatility": vol, "Sharpe": r.mean() * 252 / vol, "Sortino": r.mean() * 252 / downside,
           "Max drawdown": dd.min(), "Calmar": cagr / abs(dd.min()), "Skewness": skew(r), "Hit rate": (r > 0).mean()}
    if turnover is not None:
        out["Turnover (x/yr)"] = turnover.loc[r.index].sum() / years
    return out


def deflated_sharpe(sr_best, sr_trials, n_obs, skewness=0.0, kurt=3.0):
    """Probability that the best Sharpe ratio (per period, not annualised) is genuine after
    trying `len(sr_trials)` variants. Expected maximum of N noise Sharpes via extreme value theory."""
    N = len(sr_trials)
    var_sr = np.var(sr_trials, ddof=1)
    gamma = 0.5772156649
    sr0 = np.sqrt(var_sr) * ((1 - gamma) * norm.ppf(1 - 1 / N) + gamma * norm.ppf(1 - 1 / (N * np.e)))
    num = (sr_best - sr0) * np.sqrt(n_obs - 1)
    den = np.sqrt(1 - skewness * sr_best + (kurt - 1) / 4 * sr_best**2)
    return float(norm.cdf(num / den)), float(sr0)
