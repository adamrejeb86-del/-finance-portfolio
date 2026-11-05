"""
Exotic options by Monte Carlo, with the variance-reduction techniques used in production.

Products
  - Arithmetic and geometric Asian calls (average of discrete fixings)
  - Down-and-out / down-and-in calls (discretely monitored barrier)
  - Floating-strike lookback call (payoff S_T - min S_t)

Variance reduction
  - Antithetic variates
  - Control variate: the geometric Asian, which has a closed form, for the arithmetic Asian
  - Quasi-Monte Carlo: scrambled Sobol points with a Brownian-bridge path construction

Closed forms used as benchmarks
  - Geometric Asian with discrete fixings (Kemna-Vorst type)
  - Continuous down-and-out call (Reiner-Rubinstein) with the Broadie-Glasserman-Kou
    continuity correction for discrete monitoring
  - Continuous floating-strike lookback call (Goldman-Sosin-Gatto)

Author: Adam Rejeb
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm, qmc

BGK_BETA = 0.5825971579390106   # -zeta(1/2)/sqrt(2*pi), Broadie-Glasserman-Kou constant


# ---------------------------------------------------------------- path generation
def gbm_paths(S0, r, q, sigma, T, n_steps, n_paths, seed=0, antithetic=False, sobol=False):
    """Simulate GBM paths on a uniform grid, shape (n_paths, n_steps + 1).

    With sobol=True the Brownian motion is built with a Brownian bridge: the first Sobol
    coordinate (the best-distributed one) sets the terminal value, the next ones fill in
    the midpoints. This concentrates the low-discrepancy quality where it matters most.
    """
    dt = T / n_steps
    if sobol:
        m = int(np.ceil(np.log2(n_paths)))
        u = qmc.Sobol(d=n_steps, scramble=True, seed=seed).random_base2(m)[:n_paths]
        z = norm.ppf(np.clip(u, 1e-12, 1 - 1e-12))
        W = brownian_bridge(z, T)
    else:
        rng = np.random.default_rng(seed)
        half = n_paths // 2 if antithetic else n_paths
        z = rng.standard_normal((half, n_steps))
        if antithetic:
            z = np.concatenate([z, -z])
        W = np.cumsum(np.sqrt(dt) * z, axis=1)
    t = np.linspace(dt, T, n_steps)
    log_s = np.log(S0) + (r - q - 0.5 * sigma**2) * t + sigma * W
    return np.hstack([np.full((len(log_s), 1), S0), np.exp(log_s)])


def brownian_bridge(z, T):
    """Map standard normals z (n_paths, n_steps) to Brownian paths on a uniform grid,
    filling points in the order: terminal value, then midpoints of each interval."""
    n_paths, n = z.shape
    times = np.linspace(T / n, T, n)
    W = np.zeros((n_paths, n + 1))           # column 0 is W(0) = 0
    W[:, n] = np.sqrt(T) * z[:, 0]
    known = [0, n]
    col = 1
    while col < n:
        new_known = []
        for left, right in zip(known[:-1], known[1:]):
            if right - left < 2:
                continue
            mid = (left + right) // 2
            tl = 0.0 if left == 0 else times[left - 1]
            tm, tr = times[mid - 1], times[right - 1]
            mean = W[:, left] + (tm - tl) / (tr - tl) * (W[:, right] - W[:, left])
            std = np.sqrt((tm - tl) * (tr - tm) / (tr - tl))
            W[:, mid] = mean + std * z[:, col]
            col += 1
            new_known.append(mid)
        known = sorted(known + new_known)
    return W[:, 1:]


def _stats(discounted, pairs=False):
    x = 0.5 * (discounted[: len(discounted) // 2] + discounted[len(discounted) // 2:]) if pairs else discounted
    return float(discounted.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))


# ---------------------------------------------------------------- closed forms
def bs_call(S, K, T, r, q, sigma):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d1 - sigma * np.sqrt(T))


def geometric_asian_call(S, K, T, r, q, sigma, n_fixings):
    """Exact price of a call on the geometric average of n equally spaced fixings
    (fixings at T/n, 2T/n, ..., T). ln(G) is normal with known mean and variance."""
    n = n_fixings
    mu = np.log(S) + (r - q - 0.5 * sigma**2) * T * (n + 1) / (2 * n)
    var = sigma**2 * T * (n + 1) * (2 * n + 1) / (6 * n**2)
    sd = np.sqrt(var)
    d1 = (mu - np.log(K) + var) / sd
    return np.exp(-r * T) * (np.exp(mu + 0.5 * var) * norm.cdf(d1) - K * norm.cdf(d1 - sd))


def down_and_out_call(S, K, H, T, r, q, sigma):
    """Continuously monitored down-and-out call, barrier H <= K, no rebate."""
    lam = (r - q + 0.5 * sigma**2) / sigma**2
    y = np.log(H**2 / (S * K)) / (sigma * np.sqrt(T)) + lam * sigma * np.sqrt(T)
    down_in = (S * np.exp(-q * T) * (H / S) ** (2 * lam) * norm.cdf(y)
               - K * np.exp(-r * T) * (H / S) ** (2 * lam - 2) * norm.cdf(y - sigma * np.sqrt(T)))
    return bs_call(S, K, T, r, q, sigma) - down_in


def down_and_out_call_discrete(S, K, H, T, r, q, sigma, n_monitoring):
    """Broadie-Glasserman-Kou: a discrete barrier behaves like a continuous one shifted
    away from the spot by exp(-beta * sigma * sqrt(dt))."""
    return down_and_out_call(S, K, H * np.exp(-BGK_BETA * sigma * np.sqrt(T / n_monitoring)), T, r, q, sigma)


def lookback_floating_call(S, T, r, q, sigma):
    """Continuously monitored floating-strike lookback call, payoff S_T - min(S_t), at
    inception (running minimum = S). Goldman-Sosin-Gatto formula as given by Hull;
    requires r != q."""
    b = r - q
    a1 = (b + 0.5 * sigma**2) * T / (sigma * np.sqrt(T))
    a2 = a1 - sigma * np.sqrt(T)
    a3 = (-b + 0.5 * sigma**2) * T / (sigma * np.sqrt(T))
    k = sigma**2 / (2 * b)
    return (S * np.exp(-q * T) * norm.cdf(a1) - S * np.exp(-q * T) * k * norm.cdf(-a1)
            - S * np.exp(-r * T) * (norm.cdf(a2) - k * norm.cdf(-a3)))


# ---------------------------------------------------------------- Monte Carlo pricers
def asian_call_mc(S0, K, T, r, q, sigma, n_fixings, n_paths, seed=0, method="plain"):
    """Arithmetic Asian call. method in {'plain', 'antithetic', 'control', 'sobol'}."""
    disc = np.exp(-r * T)
    if method == "sobol":
        # A single Sobol set has no meaningful standard error: use 16 independent
        # scramblings and measure the dispersion of their means (randomised QMC).
        means = []
        for rep in range(16):
            p = gbm_paths(S0, r, q, sigma, T, n_fixings, n_paths // 16, seed=seed * 100 + rep, sobol=True)
            means.append((disc * np.maximum(p[:, 1:].mean(axis=1) - K, 0.0)).mean())
        return float(np.mean(means)), float(np.std(means, ddof=1) / np.sqrt(16))
    paths = gbm_paths(S0, r, q, sigma, T, n_fixings, n_paths, seed, antithetic=(method == "antithetic"))
    fixings = paths[:, 1:]
    arith = disc * np.maximum(fixings.mean(axis=1) - K, 0.0)
    if method in ("plain", "antithetic"):
        return _stats(arith, pairs=(method == "antithetic"))
    # Control variate: Y = X - c (G - E[G]) with the optimal c = Cov(X, G) / Var(G)
    geo = disc * np.maximum(np.exp(np.log(fixings).mean(axis=1)) - K, 0.0)
    exact = geometric_asian_call(S0, K, T, r, q, sigma, n_fixings)
    c = np.cov(arith, geo)[0, 1] / geo.var(ddof=1)
    return _stats(arith - c * (geo - exact))


def barrier_call_mc(S0, K, H, T, r, q, sigma, n_monitoring, n_paths, seed=0):
    """Down-and-out and down-and-in calls on the same paths. In + out = vanilla exactly."""
    paths = gbm_paths(S0, r, q, sigma, T, n_monitoring, n_paths, seed, antithetic=True)
    knocked = paths[:, 1:].min(axis=1) <= H
    payoff = np.exp(-r * T) * np.maximum(paths[:, -1] - K, 0.0)
    return {"out": _stats(np.where(knocked, 0.0, payoff), pairs=True),
            "in": _stats(np.where(knocked, payoff, 0.0), pairs=True),
            "vanilla": _stats(payoff, pairs=True)}


def lookback_call_mc(S0, T, r, q, sigma, n_steps, n_paths, seed=0, correct=True):
    """Floating lookback call. With correct=True the discrete minimum is shifted down by
    exp(-beta sigma sqrt(dt)) to approximate the continuously monitored minimum."""
    shift = np.exp(-BGK_BETA * sigma * np.sqrt(T / n_steps)) if correct else 1.0
    chunk = max(2, min(n_paths, 20_000_000 // n_steps) // 2 * 2)     # bound memory use
    pair_means = []
    for i, start in enumerate(range(0, n_paths, chunk)):
        paths = gbm_paths(S0, r, q, sigma, T, n_steps, min(chunk, n_paths - start), seed * 1000 + i, antithetic=True)
        pv = np.exp(-r * T) * (paths[:, -1] - paths.min(axis=1) * shift)
        pair_means.append(0.5 * (pv[: len(pv) // 2] + pv[len(pv) // 2:]))
    x = np.concatenate(pair_means)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))
