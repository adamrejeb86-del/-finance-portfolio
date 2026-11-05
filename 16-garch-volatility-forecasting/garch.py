"""
Volatility forecasting: GARCH(1,1) and GJR-GARCH by maximum likelihood, against simple
benchmarks (rolling historical volatility, EWMA / RiskMetrics).

    r_t = mu + eps_t,   eps_t = sigma_t z_t,   z_t ~ N(0,1) or standardised Student-t
    GARCH:  sigma_t^2 = omega + alpha eps_{t-1}^2 + beta sigma_{t-1}^2
    GJR:    sigma_t^2 = omega + (alpha + gamma 1{eps_{t-1} < 0}) eps_{t-1}^2 + beta sigma_{t-1}^2

GJR adds the leverage effect: bad news raises volatility more than good news of the same size.
Forecasts are compared out of sample with the QLIKE loss, which is robust to the noise in
realised-variance proxies (Patton, 2011).

Author: Adam Rejeb
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln


@dataclass
class GarchFit:
    mu: float
    omega: float
    alpha: float
    gamma: float
    beta: float
    nu: float | None
    loglik: float
    n_obs: int

    @property
    def persistence(self):
        return self.alpha + 0.5 * self.gamma + self.beta

    @property
    def long_run_vol(self):
        return np.sqrt(self.omega / (1 - self.persistence))

    @property
    def half_life(self):
        """Days for a volatility shock to decay by half."""
        return np.log(0.5) / np.log(self.persistence)

    @property
    def aic(self):
        k = 5 if self.gamma else 4
        return 2 * (k + (self.nu is not None)) - 2 * self.loglik


def variance_path(eps, omega, alpha, gamma, beta, init=None):
    """Conditional variances sigma_t^2 for t = 0..n (the last one is the next-day forecast)."""
    n = len(eps)
    s2 = np.empty(n + 1)
    s2[0] = np.var(eps) if init is None else init
    for t in range(n):
        s2[t + 1] = omega + (alpha + gamma * (eps[t] < 0)) * eps[t] ** 2 + beta * s2[t]
    return s2


def _neg_loglik(params, r, asym, student):
    mu, omega, alpha, beta = params[:4]
    gamma = params[4] if asym else 0.0
    nu = params[-1] if student else None
    eps = r - mu
    s2 = variance_path(eps, omega, alpha, gamma, beta)[:-1]
    if np.any(s2 <= 0):
        return 1e10
    if student:
        # standardised Student-t (unit variance)
        c = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
        ll = c - 0.5 * np.log(s2) - (nu + 1) / 2 * np.log(1 + eps**2 / (s2 * (nu - 2)))
    else:
        ll = -0.5 * (np.log(2 * np.pi) + np.log(s2) + eps**2 / s2)
    return -ll.sum()


def fit(r, asym=False, student=False) -> GarchFit:
    """Maximum likelihood with stationarity enforced (alpha + gamma/2 + beta < 1)."""
    scale = 100.0                                   # optimise on returns in % for good numerical scaling
    r = np.asarray(r, dtype=float) * scale
    var = r.var()
    x0 = [r.mean(), var * 0.05, 0.06, 0.90] + ([0.05] if asym else []) + ([8.0] if student else [])
    bounds = [(-1, 1), (1e-12, 10 * var), (1e-6, 0.5), (0.0, 0.999)] + ([(0.0, 0.5)] if asym else []) + ([(2.1, 100)] if student else [])
    cons = [{"type": "ineq", "fun": lambda p: 0.9999 - p[2] - p[3] - (0.5 * p[4] if asym else 0.0)}]
    res = minimize(_neg_loglik, x0, args=(r, asym, student), method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-10})
    p = res.x
    # log-likelihood of the unscaled data differs by n x log(scale) (change of variable)
    return GarchFit(mu=p[0] / scale, omega=p[1] / scale**2, alpha=p[2], gamma=p[4] if asym else 0.0, beta=p[3],
                    nu=p[-1] if student else None, loglik=-res.fun + len(r) * np.log(scale), n_obs=len(r))


def forecast_variance(f: GarchFit, last_var_next, horizon):
    """E[sigma^2_{t+h}] for h = 1..horizon, mean-reverting to the long-run variance."""
    lr = f.omega / (1 - f.persistence)
    h = np.arange(horizon)
    return lr + f.persistence**h * (last_var_next - lr)


def news_impact(f: GarchFit, shocks, sigma2_prev=None):
    """Next-day variance as a function of today's shock, other things equal."""
    s2 = f.long_run_vol**2 if sigma2_prev is None else sigma2_prev
    return f.omega + (f.alpha + f.gamma * (shocks < 0)) * shocks**2 + f.beta * s2


def simulate(n, omega, alpha, gamma, beta, nu=None, mu=0.0, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.standard_t(nu, n) * np.sqrt((nu - 2) / nu) if nu else rng.standard_normal(n)
    eps = np.empty(n)
    s2 = omega / (1 - alpha - 0.5 * gamma - beta)
    for t in range(n):
        eps[t] = np.sqrt(s2) * z[t]
        s2 = omega + (alpha + gamma * (eps[t] < 0)) * eps[t] ** 2 + beta * s2
    return mu + eps


def ewma_variance(r, lam=0.94):
    s2 = np.empty(len(r) + 1)
    s2[0] = np.var(r[:250])
    for t in range(len(r)):
        s2[t + 1] = lam * s2[t] + (1 - lam) * r[t] ** 2
    return s2


def qlike(realised, forecast):
    """QLIKE loss: realised/forecast - log(realised/forecast) - 1. Lower is better."""
    ratio = realised / forecast
    return float(np.mean(ratio - np.log(ratio) - 1))
