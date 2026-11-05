"""
Delta-hedging simulator: what really happens to an option seller's P&L.

A trader sells a European call at the implied volatility, then hedges it by holding
delta shares, rebalanced on a discrete grid, with the cash account earning the risk-free
rate. The simulator answers three desk questions:
  1. How large is the hedging error when rebalancing is discrete?
  2. What happens if realised volatility differs from the implied volatility sold?
  3. How do transaction costs change the optimal rebalancing frequency?

It also computes the classic gamma/theta attribution of the P&L,
      dP&L ~ 1/2 Gamma S^2 (sigma_implied^2 dt - (dS/S)^2),
which shows that a delta-hedged option is a bet on realised against implied volatility.

Author: Adam Rejeb
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def bs_call(S, K, tau, r, q, sigma):
    tau = np.maximum(tau, 1e-12)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    return S * np.exp(-q * tau) * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d1 - sigma * np.sqrt(tau))


def bs_delta(S, K, tau, r, q, sigma):
    tau = np.maximum(tau, 1e-12)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    return np.exp(-q * tau) * norm.cdf(d1)


def bs_gamma(S, K, tau, r, q, sigma):
    tau = np.maximum(tau, 1e-12)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    return np.exp(-q * tau) * norm.pdf(d1) / (S * sigma * np.sqrt(tau))


def simulate_paths(S0, mu, q, sigma_real, T, n_steps, n_paths, seed=0):
    """Real-world GBM paths (drift mu), shape (n_paths, n_steps + 1)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    log_ret = (mu - q - 0.5 * sigma_real**2) * dt + sigma_real * np.sqrt(dt) * z
    return S0 * np.exp(np.hstack([np.zeros((n_paths, 1)), np.cumsum(log_ret, axis=1)]))


def hedge_short_call(paths, K, T, r, q, sigma_implied, sigma_hedge=None, tx_cost=0.0):
    """P&L (discounted to today) of selling one call at sigma_implied and delta hedging.

    paths:       stock prices on the rebalancing grid, shape (n_paths, n_steps + 1)
    sigma_hedge: volatility used to compute the hedge ratio (defaults to sigma_implied)
    tx_cost:     proportional cost per unit of stock traded (e.g. 0.001 = 10 bp)
    """
    sigma_hedge = sigma_implied if sigma_hedge is None else sigma_hedge
    n_paths, n_cols = paths.shape
    n_steps = n_cols - 1
    dt = T / n_steps
    S = paths[:, 0]
    premium = bs_call(S, K, T, r, q, sigma_implied)
    delta = bs_delta(S, K, T, r, q, sigma_hedge)
    cash = premium - delta * S - tx_cost * np.abs(delta) * S

    for i in range(1, n_steps + 1):
        S_new = paths[:, i]
        cash = cash * np.exp(r * dt) + delta * S * (np.exp(q * dt) - 1)   # interest + dividends on the shares held
        tau = T - i * dt
        new_delta = bs_delta(S_new, K, tau, r, q, sigma_hedge) if i < n_steps else (S_new > K).astype(float)
        cash -= (new_delta - delta) * S_new + tx_cost * np.abs(new_delta - delta) * S_new
        delta, S = new_delta, S_new

    payoff = np.maximum(S - K, 0.0)
    return np.exp(-r * T) * (cash + delta * S - payoff)


def gamma_attribution(paths, K, T, r, q, sigma_implied):
    """Approximate P&L of the hedged short call from gamma and realised moves:
    sum over steps of 1/2 Gamma S^2 (sigma_implied^2 dt - (dS/S)^2), discounted."""
    n_steps = paths.shape[1] - 1
    dt = T / n_steps
    total = np.zeros(paths.shape[0])
    for i in range(n_steps):
        S = paths[:, i]
        ret = paths[:, i + 1] / S - 1
        gamma = bs_gamma(S, K, T - i * dt, r, q, sigma_implied)
        total += np.exp(-r * i * dt) * 0.5 * gamma * S**2 * (sigma_implied**2 * dt - ret**2)
    return total
