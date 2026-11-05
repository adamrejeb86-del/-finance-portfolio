"""
Multi-asset allocation: mean-variance, risk parity and Black-Litterman.

- Mean-variance (Markowitz): efficient frontier, minimum variance and maximum Sharpe
  portfolios, long-only.
- Risk budgeting: risk contributions RC_i = w_i (Sigma w)_i / sigma_p, which add up to the
  portfolio volatility; equal risk contribution (ERC, "risk parity") solved with the convex
  formulation of Spinu (2013).
- Black-Litterman (1992): start from the returns implied by market weights (reverse
  optimisation, Pi = delta Sigma w_mkt), blend them with the manager's views, and optimise
  on the posterior. This avoids the extreme portfolios Markowitz produces with raw forecasts.

Author: Adam Rejeb
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ASSETS = ["Global equities", "EM equities", "Govt bonds", "IG credit", "High yield", "Listed real estate", "Commodities", "Gold"]
VOLS = np.array([0.15, 0.21, 0.05, 0.065, 0.10, 0.18, 0.20, 0.16])
CORR = np.array([
    [1.00, 0.75, -0.10, 0.30, 0.65, 0.75, 0.35, 0.05],
    [0.75, 1.00, -0.05, 0.30, 0.60, 0.60, 0.45, 0.15],
    [-0.10, -0.05, 1.00, 0.65, 0.05, 0.10, -0.15, 0.25],
    [0.30, 0.30, 0.65, 1.00, 0.55, 0.40, 0.05, 0.20],
    [0.65, 0.60, 0.05, 0.55, 1.00, 0.60, 0.35, 0.05],
    [0.75, 0.60, 0.10, 0.40, 0.60, 1.00, 0.25, 0.10],
    [0.35, 0.45, -0.15, 0.05, 0.35, 0.25, 1.00, 0.30],
    [0.05, 0.15, 0.25, 0.20, 0.05, 0.10, 0.30, 1.00]])
MARKET_WEIGHTS = np.array([0.38, 0.07, 0.25, 0.14, 0.04, 0.04, 0.04, 0.04])     # stylised global market portfolio


def covariance(vols=VOLS, corr=CORR):
    return np.outer(vols, vols) * corr


def portfolio_stats(w, mu, cov, rf=0.02):
    ret, vol = float(w @ mu), float(np.sqrt(w @ cov @ w))
    return {"return": ret, "vol": vol, "sharpe": (ret - rf) / vol}


def risk_contributions(w, cov):
    """Euler decomposition: sum of contributions = portfolio volatility."""
    marginal = cov @ w / np.sqrt(w @ cov @ w)
    return w * marginal


# ---------------------------------------------------------------- mean-variance
def _long_only(n, objective, extra=()):
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}, *extra]
    res = minimize(objective, np.full(n, 1 / n), method="SLSQP", bounds=[(0, 1)] * n, constraints=cons,
                   options={"ftol": 1e-12, "maxiter": 1000})
    return res.x


def min_variance(cov):
    return _long_only(len(cov), lambda w: w @ cov @ w)


def max_sharpe(mu, cov, rf=0.02):
    return _long_only(len(cov), lambda w: -(w @ mu - rf) / np.sqrt(w @ cov @ w))


def efficient_frontier(mu, cov, n_points=30):
    lo, hi = float(min_variance(cov) @ mu), float(mu.max())
    out = []
    for target in np.linspace(lo, hi * 0.999, n_points):
        w = _long_only(len(cov), lambda w: w @ cov @ w, extra=({"type": "eq", "fun": lambda w, t=target: w @ mu - t},))
        out.append((float(np.sqrt(w @ cov @ w)), target, w))
    return out


# ---------------------------------------------------------------- risk parity
def risk_budget_portfolio(cov, budgets=None):
    """Weights whose risk contributions are proportional to `budgets` (equal by default).
    Spinu's convex problem: minimise 1/2 y'Sigma y - sum b_i ln y_i, then w = y / sum(y)."""
    n = len(cov)
    b = np.full(n, 1 / n) if budgets is None else np.asarray(budgets) / np.sum(budgets)
    obj = lambda y: 0.5 * y @ cov @ y - b @ np.log(y)
    grad = lambda y: cov @ y - b / y
    res = minimize(obj, np.full(n, 1.0), jac=grad, method="L-BFGS-B", bounds=[(1e-10, None)] * n,
                   options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 10_000})
    return res.x / res.x.sum()


def inverse_volatility(cov):
    iv = 1 / np.sqrt(np.diag(cov))
    return iv / iv.sum()


# ---------------------------------------------------------------- Black-Litterman
def implied_returns(cov, w_mkt, risk_aversion=2.5, rf=0.02):
    """Expected returns that make the market portfolio optimal: Pi = delta Sigma w_mkt (+ rf)."""
    return rf + risk_aversion * cov @ w_mkt


def black_litterman(cov, pi, P, Q, tau=0.05, confidences=None, omega=None):
    """Posterior expected returns and covariance.

    P: (k x n) view matrix, Q: (k,) view returns. Omega defaults to the He-Litterman choice
    diag(tau P Sigma P'), optionally scaled by 1/confidence (0 < confidence <= 1).
        mu_BL = [(tau Sigma)^-1 + P' Omega^-1 P]^-1 [(tau Sigma)^-1 Pi + P' Omega^-1 Q]
    """
    P, Q = np.atleast_2d(P), np.asarray(Q, dtype=float)
    if omega is None:
        omega = np.diag(np.diag(tau * P @ cov @ P.T))
        if confidences is not None:
            omega = omega / np.asarray(confidences)
    ts_inv = np.linalg.inv(tau * cov)
    om_inv = np.linalg.inv(omega)
    post_prec = ts_inv + P.T @ om_inv @ P
    mu = np.linalg.solve(post_prec, ts_inv @ pi + P.T @ om_inv @ Q)
    post_cov = cov + np.linalg.inv(post_prec)
    return mu, post_cov


def mean_variance_weights(mu, cov, risk_aversion=2.5, rf=0.02, long_only=True):
    """argmax w'mu - delta/2 w'Sigma w with weights summing to 1."""
    n = len(mu)
    obj = lambda w: -(w @ (mu - rf) - 0.5 * risk_aversion * w @ cov @ w)
    bounds = [(0, 1)] * n if long_only else [(-2, 2)] * n
    res = minimize(obj, np.full(n, 1 / n), method="SLSQP", bounds=bounds,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], options={"ftol": 1e-12})
    return res.x


def unconstrained_weights(mu, cov, risk_aversion=2.5, rf=0.02):
    """Closed-form optimum w = (delta Sigma)^-1 (mu - rf), the convention of the original
    Black-Litterman paper: with equilibrium returns it gives back the market weights, and with
    views it changes only the assets the views involve (the rest stays at market weight)."""
    return np.linalg.solve(risk_aversion * cov, mu - rf)
