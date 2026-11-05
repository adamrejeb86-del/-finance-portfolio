"""
Factor models and performance attribution for fund analysis.

1. Returns-based style analysis: regress a fund's excess returns on factor returns
   (market, size, value, momentum, quality) to separate alpha from factor exposure, with
   Newey-West (HAC) standard errors because monthly returns are autocorrelated and
   heteroskedastic.
2. Rolling exposures to detect style drift.
3. Brinson-Fachler holdings-based attribution: allocation, selection and interaction
   effects by sector, linked over many periods with Carino's method so that the effects
   add up exactly to the compounded active return.
4. Standard performance and risk statistics.

Author: Adam Rejeb
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FACTORS = ["MKT", "SMB", "HML", "MOM", "QMJ"]


# ---------------------------------------------------------------- synthetic data
def simulate_factors(n_months=180, seed=0) -> pd.DataFrame:
    """Monthly factor returns with realistic premia, volatilities and correlations."""
    rng = np.random.default_rng(seed)
    mean = np.array([0.0065, 0.0015, 0.0020, 0.0045, 0.0030])
    vol = np.array([0.045, 0.025, 0.028, 0.040, 0.020])
    corr = np.array([[1, 0.2, 0.0, -0.2, -0.5], [0.2, 1, 0.1, -0.1, -0.4], [0.0, 0.1, 1, -0.4, -0.1],
                     [-0.2, -0.1, -0.4, 1, 0.3], [-0.5, -0.4, -0.1, 0.3, 1]])
    L = np.linalg.cholesky(corr)
    z = rng.standard_t(6, (n_months, 5)) * np.sqrt(4 / 6)
    data = mean + (z @ L.T) * vol
    return pd.DataFrame(data, index=pd.period_range("2011-01", periods=n_months, freq="M"), columns=FACTORS)


def simulate_fund(factors: pd.DataFrame, betas, alpha_annual=0.0, idio_vol_annual=0.03, seed=1, drift=None):
    """Fund excess returns = alpha + betas . factors + noise. `drift` = (month, new_betas)."""
    rng = np.random.default_rng(seed)
    B = np.tile(np.asarray(betas, float), (len(factors), 1))
    if drift is not None:
        month, new_betas = drift
        B[month:] = new_betas
    noise = rng.standard_normal(len(factors)) * idio_vol_annual / np.sqrt(12)
    noise[1:] += 0.25 * noise[:-1]                                     # mild autocorrelation (stale prices)
    return pd.Series(alpha_annual / 12 + (B * factors.values).sum(axis=1) + noise, index=factors.index)


# ---------------------------------------------------------------- regression with HAC errors
def ols_newey_west(y, X, lags=3, add_const=True):
    """OLS coefficients with Newey-West standard errors (Bartlett kernel)."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    if add_const:
        X = np.column_stack([np.ones(len(X)), X])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        G = Xu[l:].T @ Xu[:-l]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    ols_se = np.sqrt(np.diag(XtX_inv) * (u @ u) / (len(y) - X.shape[1]))
    r2 = 1 - (u @ u) / ((y - y.mean()) @ (y - y.mean()))
    return {"coef": beta, "se_hac": se, "t_hac": beta / se, "se_ols": ols_se, "r2": r2, "resid": u}


def factor_regression(fund: pd.Series, factors: pd.DataFrame, lags=3) -> pd.DataFrame:
    res = ols_newey_west(fund.values, factors.values, lags)
    names = ["alpha (monthly)"] + list(factors.columns)
    out = pd.DataFrame({"coef": res["coef"], "t-stat (HAC)": res["t_hac"], "se (HAC)": res["se_hac"], "se (OLS)": res["se_ols"]}, index=names)
    out.attrs["r2"] = res["r2"]
    return out


def rolling_betas(fund: pd.Series, factors: pd.DataFrame, window=36) -> pd.DataFrame:
    rows = {}
    for end in range(window, len(fund) + 1):
        res = ols_newey_west(fund.values[end - window:end], factors.values[end - window:end], lags=0)
        rows[fund.index[end - 1]] = res["coef"][1:]
    return pd.DataFrame(rows, index=factors.columns).T


def return_decomposition(fund: pd.Series, factors: pd.DataFrame, coef: np.ndarray) -> pd.Series:
    """Average annual return explained by each factor (beta x factor mean x 12), alpha and residual."""
    contrib = {f: coef[i + 1] * factors[f].mean() * 12 for i, f in enumerate(factors.columns)}
    contrib["Alpha"] = coef[0] * 12
    total = fund.mean() * 12
    contrib["Residual"] = total - sum(contrib.values())
    return pd.Series(contrib)


# ---------------------------------------------------------------- Brinson-Fachler
def brinson_fachler(wp, wb, rp, rb):
    """Single-period effects by sector. wp/wb: weights, rp/rb: sector returns.
    Allocation  = (wp - wb)(rb - Rb)    : overweighting sectors that beat the benchmark
    Selection   = wb (rp - rb)          : picking better stocks within each sector
    Interaction = (wp - wb)(rp - rb)
    Their sum equals the active return Rp - Rb exactly."""
    wp, wb, rp, rb = map(np.asarray, (wp, wb, rp, rb))
    Rb = wb @ rb
    return pd.DataFrame({"allocation": (wp - wb) * (rb - Rb), "selection": wb * (rp - rb), "interaction": (wp - wb) * (rp - rb)})


def carino_link(period_effects: list[pd.DataFrame], Rp: np.ndarray, Rb: np.ndarray) -> pd.DataFrame:
    """Carino (1999) smoothing: scale each period's arithmetic effects so that they add up to
    the compounded active return (1+Rp_total) - (1+Rb_total) over all periods."""
    Rp, Rb = np.asarray(Rp), np.asarray(Rb)
    RP, RB = np.prod(1 + Rp) - 1, np.prod(1 + Rb) - 1
    k_total = (np.log(1 + RP) - np.log(1 + RB)) / (RP - RB) if RP != RB else 1 / (1 + RP)
    k = np.where(Rp != Rb, (np.log(1 + Rp) - np.log(1 + Rb)) / (Rp - Rb), 1 / (1 + Rp))
    linked = sum(e * (kt / k_total) for e, kt in zip(period_effects, k))
    return linked


# ---------------------------------------------------------------- statistics
def performance_stats(r: pd.Series, bench: pd.Series | None = None, rf_monthly=0.0) -> dict:
    ex = r - rf_monthly
    wealth = (1 + r).cumprod()
    out = {"Annual return": (1 + r).prod() ** (12 / len(r)) - 1, "Annual vol": r.std() * np.sqrt(12),
           "Sharpe": ex.mean() / r.std() * np.sqrt(12),
           "Sortino": ex.mean() / downside * np.sqrt(12) if (downside := np.sqrt((np.minimum(ex, 0) ** 2).mean())) > 0 else np.inf,
           "Max drawdown": (wealth / wealth.cummax() - 1).min()}
    if bench is not None:
        active = r - bench
        out["Tracking error"] = active.std() * np.sqrt(12)
        out["Information ratio"] = active.mean() / active.std() * np.sqrt(12)
    return out
