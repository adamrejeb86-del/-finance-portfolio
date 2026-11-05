"""
Rates relative value with principal component analysis: curve factors, DV01-neutral and
PCA-neutral packages, rich/cheap signals and a butterfly mean-reversion backtest.

About 95% of daily moves in a yield curve are explained by three factors: level, slope and
curvature. Rates traders use this in two ways:
  1. Hedging: a package such as a 2s10s steepener or a 5s10s30s butterfly should only carry the
     risk it is meant to carry. Equal DV01 removes the parallel risk, but a real "level" move is
     not exactly parallel, so desks also weight packages with the PCA loadings.
  2. Relative value: once the first three factors are removed, what is left in each tenor is a
     small residual that tends to mean-revert. A tenor whose yield sits above the 3-factor fit is
     "cheap"; below, it is "rich".

Conventions: yields in percent, changes in basis points, risk positions in EUR per bp (DV01).
A positive DV01 position is long the bond / receives fixed, so it gains when yields fall.

Author: Adam Rejeb
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TENORS = np.array([2, 3, 5, 7, 10, 15, 20, 30], dtype=float)


# ---------------------------------------------------------------- curve history
def nelson_siegel_loadings(tau, lam=3.0):
    """Slope and curvature loadings of the Nelson-Siegel model."""
    x = np.asarray(tau) / lam
    slope = (1 - np.exp(-x)) / x
    return slope, slope - np.exp(-x)


def simulate_curves(n=2520, seed=0, tenors=TENORS, idio_bp=3.0, idio_half_life=20.0, level_slope_corr=0.3):
    """Daily yield curves (percent) driven by three Nelson-Siegel factors plus a small
    mean-reverting residual in each tenor (the relative value opportunity).

    The level and slope shocks are positively correlated: when rates fall, the short end falls
    more (a "bull steepening"), as when a central bank cuts. That makes the real first factor
    heavier at the short end than a parallel shift, which is why DV01 weights are not enough.
    Returns (curves in percent, true residuals in bp)."""
    rng = np.random.default_rng(seed)
    f_s, f_c = nelson_siegel_loadings(tenors)
    level, slope, curv = np.empty(n), np.empty(n), np.empty(n)
    level[0], slope[0], curv[0] = 3.0, -1.6, -0.8
    for t in range(1, n):
        e_level, e_other = rng.standard_normal(2)
        e_slope = level_slope_corr * e_level + np.sqrt(1 - level_slope_corr**2) * e_other
        level[t] = level[t - 1] + 0.002 * (3.0 - level[t - 1]) + 0.05 * e_level          # slow mean reversion
        slope[t] = slope[t - 1] + 0.01 * (-1.6 - slope[t - 1]) + 0.12 * e_slope
        curv[t] = curv[t - 1] + 0.01 * (-0.8 - curv[t - 1]) + rng.normal(0, 0.09)
    phi = np.exp(-np.log(2) / idio_half_life)
    idio = np.zeros((n, len(tenors)))
    sd = idio_bp * np.sqrt(1 - phi**2)
    for t in range(1, n):
        idio[t] = phi * idio[t - 1] + rng.normal(0, sd, len(tenors))
    curves = level[:, None] + slope[:, None] * f_s + curv[:, None] * f_c + idio / 100
    idx = pd.bdate_range("2016-01-04", periods=n)
    cols = [f"{int(m)}y" for m in tenors]
    return pd.DataFrame(curves, idx, cols), pd.DataFrame(idio, idx, cols)


# ---------------------------------------------------------------- PCA
def pca(changes: pd.DataFrame):
    """Eigen-decomposition of the covariance of daily changes (bp). Loadings are signed so that
    PC1 raises all yields, PC2 steepens the curve and PC3 raises the belly."""
    X = changes.values - changes.values.mean(axis=0)
    cov = X.T @ X / (len(X) - 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    if vecs[:, 0].sum() < 0:
        vecs[:, 0] *= -1
    if vecs[-1, 1] - vecs[0, 1] < 0:
        vecs[:, 1] *= -1
    mid = len(vecs) // 2
    if vecs[mid, 2] - 0.5 * (vecs[0, 2] + vecs[-1, 2]) < 0:
        vecs[:, 2] *= -1
    loadings = pd.DataFrame(vecs, index=changes.columns, columns=[f"PC{i + 1}" for i in range(len(vals))])
    return {"eigenvalues": vals, "explained": vals / vals.sum(), "loadings": loadings, "cov": cov}


def fit_residuals(curves: pd.DataFrame, loadings: pd.DataFrame, k=3):
    """Cross-sectional fit of each day's curve on a constant plus the first k PCA loadings.
    Residual (bp) > 0 means the tenor yields more than the fit: cheap. < 0: rich."""
    L = np.column_stack([np.ones(len(loadings)), loadings.values[:, :k]])
    coef, *_ = np.linalg.lstsq(L, curves.values.T, rcond=None)
    fitted = (L @ coef).T
    return pd.DataFrame((curves.values - fitted) * 100, curves.index, curves.columns)


# ---------------------------------------------------------------- risk
def dv01_per_million(tenor, yield_pct):
    """DV01 (EUR per bp) of a EUR 1m par swap with an annual fixed leg, using the tenor's own
    yield to discount (a standard first approximation of the annuity)."""
    y = yield_pct / 100
    t = np.arange(1, int(tenor) + 1)
    return 1e6 * np.sum((1 + y) ** -t) * 1e-4


def factor_exposure(dv01_position: pd.Series, pca_result: dict, k=3):
    """P&L (EUR) of a DV01 position for a one-standard-deviation daily move of each factor.
    A move of PCk changes yields by loadings[:, k] x sqrt(eigenvalue_k) bp; P&L = -sum(DV01_i x dy_i)."""
    L = pca_result["loadings"].loc[dv01_position.index].iloc[:, :k] * np.sqrt(pca_result["eigenvalues"][:k])
    return -(L.T @ dv01_position)


def curve_package(legs, kind, loadings=None, size=10_000.0):
    """Risk weights (EUR per bp) of a curve package. `legs` are column names.

    - steepener, 2 legs (short end, long end): long the short end, short the long end.
        "dv01": equal and opposite DV01.  "pca": long-end leg scaled so that PC1 exposure is zero.
    - butterfly, 3 legs (wing, belly, wing): short the belly for `size` per bp (gains if it cheapens).
        "dv01": 50/50 wings (parallel-neutral).  "pca": wings solved so PC1 and PC2 exposures are zero.
    """
    if len(legs) == 2:
        if kind == "dv01":
            w = np.array([size, -size])
        else:
            l1 = loadings.loc[legs, "PC1"].values
            w = np.array([size, -size * l1[0] / l1[1]])
    else:
        if kind == "dv01":
            w = np.array([0.5 * size, -size, 0.5 * size])
        else:
            L = loadings.loc[legs, ["PC1", "PC2"]].values
            wings = np.linalg.solve(L[[0, 2]].T, size * L[1])   # wings offset the belly's PC1 and PC2
            w = np.array([wings[0], -size, wings[1]])
    return pd.Series(w, index=legs)


def package_pnl(dv01_position: pd.Series, changes_bp: pd.DataFrame):
    """Daily P&L (EUR) of a constant DV01 position."""
    return -(changes_bp[dv01_position.index] @ dv01_position)


# ---------------------------------------------------------------- relative value strategy
def package_level(curves: pd.DataFrame, dv01_position: pd.Series):
    """Level of the package in bp, scaled so that (change in level) x (DV01 of its short leg)
    equals the package P&L. For a butterfly that is short the belly, a higher level means the
    belly is cheap relative to the wings."""
    scale = -dv01_position.min()
    return -(curves[dv01_position.index] @ dv01_position) * 100 / scale


def mean_reversion_backtest(curves: pd.DataFrame, dv01_position: pd.Series, window=60, entry=1.5, exit=0.25,
                            cost_bp=0.25):
    """Trade the package against its own recent history.
    - z-score of the package level against the previous `window` days (no look-ahead);
    - z > entry (level unusually high, belly cheap): hold -1 unit, i.e. receive the belly and pay
      the wings, which gains if the level falls back; z < -entry: hold +1 unit;
    - exit when the z-score is back inside +/- `exit`;
    - signal at today's close, traded tomorrow; cost of `cost_bp` bp on the DV01 of every leg traded.
    Returns daily P&L (EUR), z-scores and positions."""
    lvl = package_level(curves, dv01_position)
    mu, sd = lvl.rolling(window).mean().shift(1), lvl.rolling(window).std().shift(1)
    z = ((lvl - mu) / sd).fillna(0.0).values
    pos = np.zeros(len(z))
    for t in range(1, len(z)):
        p = pos[t - 1]
        if p == 0:
            p = -1.0 if z[t] > entry else (1.0 if z[t] < -entry else 0.0)
        elif (p == -1 and z[t] < exit) or (p == 1 and z[t] > -exit):
            p = 0.0
        pos[t] = p
    held = np.roll(pos, 1)
    held[0] = 0.0
    unit_pnl = package_pnl(dv01_position, curves.diff().fillna(0.0) * 100).values
    turnover = np.abs(np.diff(held, prepend=0.0)) * dv01_position.abs().sum() * cost_bp
    return pd.Series(held * unit_pnl - turnover, curves.index), pd.Series(z, curves.index), pos


def pnl_stats(pnl: pd.Series) -> dict:
    sd = pnl.std()
    cum = pnl.cumsum()
    return {"Annual P&L (EUR)": pnl.mean() * 252, "Annual vol (EUR)": sd * np.sqrt(252),
            "Sharpe": pnl.mean() / sd * np.sqrt(252) if sd > 0 else np.nan, "Max drawdown (EUR)": (cum - cum.cummax()).min()}
