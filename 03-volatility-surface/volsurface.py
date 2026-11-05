"""
Volatility surface toolkit: Heston stochastic-volatility pricing, SVI smile fitting,
butterfly-arbitrage checks and Heston calibration.

- Heston (1993) call prices by Fourier inversion (Gil-Pelaez), using the "little Heston
  trap" form of the characteristic function (Albrecher et al., 2007), which stays
  numerically stable for long maturities.
- Raw SVI parameterisation of total implied variance (Gatheral, 2004):
      w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sig^2)),   k = ln(K / F)
- Durrleman's condition g(k) >= 0 to detect butterfly arbitrage in a fitted smile.

Author: Adam Rejeb
"""
from __future__ import annotations

from dataclasses import dataclass, astuple

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.stats import norm


# ---------------------------------------------------------------- Black-Scholes helpers
def bs_call(S, K, T, r, sigma, q=0.0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d1 - sigma * np.sqrt(T))


def implied_vol_call(price, S, K, T, r, q=0.0):
    """Implied volatility of a call (Brent's method, robust on the whole smile)."""
    lower = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    if price <= lower + 1e-12:
        return np.nan
    return brentq(lambda s: bs_call(S, K, T, r, s, q) - price, 1e-4, 4.0, xtol=1e-10)


# ---------------------------------------------------------------- Heston model
@dataclass(frozen=True)
class HestonParams:
    v0: float = 0.04      # initial variance (vol 20%)
    kappa: float = 1.5    # speed of mean reversion of the variance
    theta: float = 0.05   # long-run variance
    sigma: float = 0.6    # volatility of variance ("vol of vol")
    rho: float = -0.7     # correlation between spot and variance shocks (drives the skew)

    @property
    def feller(self) -> float:
        """2*kappa*theta - sigma^2 > 0 means the variance never touches zero."""
        return 2 * self.kappa * self.theta - self.sigma**2


def heston_char_func(u, T, S0, r, q, p: HestonParams):
    """Characteristic function of ln(S_T), little-trap formulation."""
    v0, kappa, theta, sigma, rho = astuple(p)
    iu = 1j * u
    d = np.sqrt((rho * sigma * iu - kappa) ** 2 + sigma**2 * (iu + u**2))
    g = (kappa - rho * sigma * iu - d) / (kappa - rho * sigma * iu + d)
    exp_dt = np.exp(-d * T)
    C = (kappa * theta / sigma**2) * ((kappa - rho * sigma * iu - d) * T
                                       - 2 * np.log((1 - g * exp_dt) / (1 - g)))
    D = ((kappa - rho * sigma * iu - d) / sigma**2) * (1 - exp_dt) / (1 - g * exp_dt)
    return np.exp(iu * (np.log(S0) + (r - q) * T) + C + D * v0)


_U = np.linspace(1e-8, 250.0, 5001)   # integration grid (the integrand decays fast)


def heston_call(S0, K, T, r, q, p: HestonParams):
    """European call prices for an array of strikes K (Gil-Pelaez inversion):
        C = S0 e^{-qT} P1 - K e^{-rT} P2,  P_j = 1/2 + 1/pi * int Re(...) du
    """
    K = np.atleast_1d(np.asarray(K, dtype=float))
    u = _U[None, :]
    lnK = np.log(K)[:, None]
    phi = heston_char_func(u, T, S0, r, q, p)
    phi_shift = heston_char_func(u - 1j, T, S0, r, q, p)
    forward = S0 * np.exp((r - q) * T)
    integrand2 = np.real(np.exp(-1j * u * lnK) * phi / (1j * u))
    integrand1 = np.real(np.exp(-1j * u * lnK) * phi_shift / (1j * u * forward))
    P1 = 0.5 + np.trapezoid(integrand1, _U, axis=1) / np.pi
    P2 = 0.5 + np.trapezoid(integrand2, _U, axis=1) / np.pi
    return S0 * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2


def heston_monte_carlo(S0, K, T, r, q, p: HestonParams, n_paths=200_000, n_steps=200, seed=0):
    """Full-truncation Euler scheme, used only to validate the Fourier pricer."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    lnS = np.full(n_paths, np.log(S0))
    v = np.full(n_paths, p.v0)
    for _ in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = p.rho * z1 + np.sqrt(1 - p.rho**2) * rng.standard_normal(n_paths)
        v_pos = np.maximum(v, 0.0)
        lnS += (r - q - 0.5 * v_pos) * dt + np.sqrt(v_pos * dt) * z1
        v += p.kappa * (p.theta - v_pos) * dt + p.sigma * np.sqrt(v_pos * dt) * z2
    payoff = np.maximum(np.exp(lnS)[:, None] - np.atleast_1d(K)[None, :], 0.0) * np.exp(-r * T)
    return payoff.mean(axis=0), payoff.std(axis=0) / np.sqrt(n_paths)


def calibrate_heston(S0, r, q, maturities, strikes, market_vols, guess=HestonParams(0.03, 1.0, 0.03, 0.4, -0.3)):
    """Least-squares calibration on implied volatilities.

    Price errors are divided by Black-Scholes vega, a standard first-order approximation of
    the implied-vol error that avoids inverting thousands of prices at each iteration.
    """
    vegas = []
    for T, vols in zip(maturities, market_vols):
        d1 = (np.log(S0 / strikes) + (r - q + 0.5 * vols**2) * T) / (vols * np.sqrt(T))
        vegas.append(S0 * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T))
    market_prices = [bs_call(S0, strikes, T, r, vols, q) for T, vols in zip(maturities, market_vols)]

    def residuals(x):
        p = HestonParams(*x)
        return np.concatenate([(heston_call(S0, strikes, T, r, q, p) - mp) / vg
                               for T, mp, vg in zip(maturities, market_prices, vegas)])

    lb, ub = [1e-4, 0.05, 1e-4, 0.05, -0.99], [1.0, 10.0, 1.0, 2.0, 0.99]
    sol = least_squares(residuals, astuple(guess), bounds=(lb, ub), xtol=1e-10, ftol=1e-10)
    return HestonParams(*sol.x), float(np.sqrt(np.mean(sol.fun**2)))


# ---------------------------------------------------------------- SVI smile
def svi_total_variance(k, a, b, rho, m, sig):
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sig**2))


def fit_svi(k, total_var, penalty=10.0):
    """Fit raw SVI to total implied variances w = sigma_imp^2 * T on log-moneyness k.

    SVI has five parameters for a smile that is often observed on a narrow strike range,
    so a single local optimisation can land in a poor minimum. We therefore:
      1. run least squares from a grid of starting points and keep the best fit;
      2. add a penalty whenever Durrleman's g(k) turns negative on a wide grid, so the
         selected smile is free of butterfly arbitrage, including outside the quoted range.
    """
    k_check = np.linspace(k.min() - 0.5, k.max() + 0.5, 121)

    def residuals(x):
        fit = svi_total_variance(k, *x) - total_var
        arb = np.minimum(durrleman_g(k_check, x), 0.0)
        return np.concatenate([fit, penalty * total_var.mean() * arb])

    lb, ub = [-1.0, 1e-6, -0.999, -1.0, 1e-4], [1.0, 5.0, 0.999, 1.0, 2.0]
    best = None
    for rho0 in (-0.7, -0.3, 0.0):
        for m0 in (-0.1, 0.0, 0.1):
            for s0 in (0.05, 0.2):
                guess = [max(total_var.min() * 0.5, 1e-4), 0.1, rho0, m0, s0]
                sol = least_squares(residuals, guess, bounds=(lb, ub), xtol=1e-12, ftol=1e-12)
                if best is None or sol.cost < best.cost:
                    best = sol
    return best.x


def durrleman_g(k, params):
    """Butterfly-arbitrage function g(k); the smile is free of butterfly arbitrage iff g >= 0."""
    a, b, rho, m, sig = params
    w = svi_total_variance(k, *params)
    root = np.sqrt((k - m) ** 2 + sig**2)
    w1 = b * (rho + (k - m) / root)
    w2 = b * sig**2 / root**3
    return (1 - k * w1 / (2 * w)) ** 2 - (w1**2 / 4) * (1 / w + 0.25) + w2 / 2
