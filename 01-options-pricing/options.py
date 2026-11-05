"""
Vanilla option pricing toolkit.

- Black-Scholes-Merton closed-form prices (calls and puts, continuous dividend yield)
- Analytical Greeks: delta, gamma, vega, theta, rho
- Implied volatility solver (Newton-Raphson with a Brent fallback)
- Cox-Ross-Rubinstein binomial tree for European and American options
- Early-exercise boundary of an American put

Author: Adam Rejeb
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma, q=0.0):
    S, K, T, sigma = map(np.asarray, (S, K, T, sigma))
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def bs_price(S, K, T, r, sigma, q=0.0, option="call"):
    """Black-Scholes-Merton price of a European option."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    if option == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    if option == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    raise ValueError("option must be 'call' or 'put'")


def bs_greeks(S, K, T, r, sigma, q=0.0, option="call"):
    """Analytical Greeks. Vega and rho are per 1.00 move (divide by 100 for 1 point),
    theta is per year (divide by 365 for one calendar day)."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_q, disc_r = np.exp(-q * T), np.exp(-r * T)
    pdf_d1 = norm.pdf(d1)
    gamma = disc_q * pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * disc_q * pdf_d1 * np.sqrt(T)
    common_theta = -S * disc_q * pdf_d1 * sigma / (2 * np.sqrt(T))
    if option == "call":
        delta = disc_q * norm.cdf(d1)
        theta = common_theta - r * K * disc_r * norm.cdf(d2) + q * S * disc_q * norm.cdf(d1)
        rho = K * T * disc_r * norm.cdf(d2)
    elif option == "put":
        delta = -disc_q * norm.cdf(-d1)
        theta = common_theta + r * K * disc_r * norm.cdf(-d2) - q * S * disc_q * norm.cdf(-d1)
        rho = -K * T * disc_r * norm.cdf(-d2)
    else:
        raise ValueError("option must be 'call' or 'put'")
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def implied_vol(price, S, K, T, r, q=0.0, option="call", tol=1e-10, max_iter=50):
    """Implied volatility from a market price.

    Newton-Raphson on vega converges in a few iterations near the money; when vega is
    tiny (deep in or out of the money) the solver falls back to Brent's method.
    """
    intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0) if option == "call" \
        else max(K * np.exp(-r * T) - S * np.exp(-q * T), 0)
    upper = S * np.exp(-q * T) if option == "call" else K * np.exp(-r * T)
    if not intrinsic - 1e-12 <= price <= upper + 1e-12:
        raise ValueError("price outside no-arbitrage bounds")

    sigma = 0.2
    for _ in range(max_iter):
        diff = bs_price(S, K, T, r, sigma, q, option) - price
        if abs(diff) < tol:
            return float(sigma)
        vega = bs_greeks(S, K, T, r, sigma, q, option)["vega"]
        if vega < 1e-8:
            break
        sigma -= diff / vega
        if not 1e-4 < sigma < 5:
            break
    return float(brentq(lambda s: bs_price(S, K, T, r, s, q, option) - price, 1e-6, 5.0, xtol=tol))


def binomial_price(S, K, T, r, sigma, q=0.0, option="put", american=True, steps=500,
                   return_boundary=False):
    """Cox-Ross-Rubinstein binomial tree.

    With return_boundary=True (American put), also returns the early-exercise boundary:
    for each time step, the highest stock price at which immediate exercise is optimal.
    """
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not 0 < p < 1:
        raise ValueError("risk-neutral probability outside (0, 1): increase steps")
    disc = np.exp(-r * dt)

    j = np.arange(steps + 1)
    prices = S * u ** (steps - 2 * j)
    payoff = np.maximum(prices - K, 0) if option == "call" else np.maximum(K - prices, 0)
    values = payoff
    boundary = np.full(steps + 1, np.nan)
    boundary[steps] = K

    for n in range(steps - 1, -1, -1):
        prices = S * u ** (n - 2 * np.arange(n + 1))
        values = disc * (p * values[:-1] + (1 - p) * values[1:])
        if american:
            exercise = np.maximum(prices - K, 0) if option == "call" else np.maximum(K - prices, 0)
            early = exercise > values + 1e-12
            values = np.where(early, exercise, values)
            if return_boundary and early.any():
                boundary[n] = prices[early].max()
    if return_boundary:
        return float(values[0]), np.linspace(0, T, steps + 1), boundary
    return float(values[0])
