"""
Monte Carlo pricer for a Phoenix autocallable note on a worst-of basket.

Product (per observation date t_i, on the worst performance W(t_i) = min_k S_k(t_i)/S_k(0)):
  - Coupon:     paid if W(t_i) >= coupon barrier; with the memory feature, unpaid coupons
                from earlier dates are paid as well.
  - Autocall:   from the first call date, if W(t_i) >= autocall barrier the note redeems
                early at 100% and stops.
  - Maturity:   if never called, 100% is repaid when W(T) >= protection barrier, otherwise
                the investor receives W(T) x notional (capital loss, like a short put).

Model: correlated geometric Brownian motions under the risk-neutral measure, with
continuous dividend yields and a flat risk-free rate. Variance is reduced with antithetic
variates, and Greeks are computed by bump-and-reprice with common random numbers.

Author: Adam Rejeb
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Phoenix:
    maturity: float = 3.0              # years
    obs_per_year: int = 4              # quarterly observations
    autocall_barrier: float = 1.00     # in % of initial level
    coupon_barrier: float = 0.70
    protection_barrier: float = 0.60   # European barrier, observed at maturity only
    coupon: float = 0.025              # per period (here 2.5% per quarter = 10% p.a.)
    memory: bool = True
    first_call_obs: int = 4            # no autocall during the first year
    notional: float = 100.0

    @property
    def times(self) -> np.ndarray:
        n = int(round(self.maturity * self.obs_per_year))
        return np.arange(1, n + 1) / self.obs_per_year


@dataclass(frozen=True)
class Market:
    vols: tuple = (0.25, 0.30)
    divs: tuple = (0.02, 0.01)
    corr: float = 0.60
    r: float = 0.03


def simulate_performances(product: Phoenix, market: Market, n_paths: int = 200_000,
                          seed: int = 42, antithetic: bool = True) -> np.ndarray:
    """Performance S_k(t_i)/S_k(0) of each asset at each observation date,
    shape (n_paths, n_obs, n_assets).

    Only the observation dates matter for this product (no continuous barrier), so the
    GBM is simulated exactly from one date to the next: no discretisation error.
    """
    times = product.times
    dt = np.diff(np.concatenate([[0.0], times]))
    vols, divs = np.asarray(market.vols), np.asarray(market.divs)
    n_assets = len(vols)
    corr = np.full((n_assets, n_assets), market.corr)
    np.fill_diagonal(corr, 1.0)
    chol = np.linalg.cholesky(corr)

    rng = np.random.default_rng(seed)
    half = n_paths // 2 if antithetic else n_paths
    z = rng.standard_normal((half, len(times), n_assets))
    if antithetic:
        z = np.concatenate([z, -z])
    z = z @ chol.T                                   # correlate the shocks
    drift = (market.r - divs - 0.5 * vols**2)[None, None, :] * dt[None, :, None]
    shocks = vols[None, None, :] * np.sqrt(dt)[None, :, None] * z
    return np.exp(np.cumsum(drift + shocks, axis=1))


def simulate_worst_of(product: Phoenix, market: Market, n_paths: int = 200_000,
                      seed: int = 42, antithetic: bool = True) -> np.ndarray:
    """Worst-of performance at each observation date, shape (n_paths, n_obs)."""
    return simulate_performances(product, market, n_paths, seed, antithetic).min(axis=2)


def cash_flows_pv(product: Phoenix, market: Market, worst: np.ndarray) -> dict:
    """Discounted payoff of every path plus statistics on the note's life."""
    times = product.times
    n_paths, n_obs = worst.shape
    alive = np.ones(n_paths, dtype=bool)
    missed = np.zeros(n_paths)                       # coupons not yet paid (memory)
    pv = np.zeros(n_paths)
    life = np.full(n_paths, times[-1])
    call_prob = np.zeros(n_obs)
    coupon_prob = np.zeros(n_obs)

    for i, t in enumerate(times):
        df = np.exp(-market.r * t)
        w = worst[:, i]
        pays = alive & (w >= product.coupon_barrier)
        n_coupons = (1 + missed) if product.memory else 1.0
        pv += np.where(pays, product.coupon * product.notional * n_coupons * df, 0.0)
        coupon_prob[i] = pays.mean()
        missed = np.where(pays, 0.0, missed + alive)

        if i == n_obs - 1:                           # maturity
            redemption = np.where(w >= product.protection_barrier, 1.0, w) * product.notional
            pv += np.where(alive, redemption * df, 0.0)
        elif i + 1 >= product.first_call_obs:
            called = alive & (w >= product.autocall_barrier)
            pv += np.where(called, product.notional * df, 0.0)
            call_prob[i] = called.mean()
            life[called] = t
            alive &= ~called

    loss = alive & (worst[:, -1] < product.protection_barrier)
    return {"pv": pv, "call_prob": call_prob, "coupon_prob": coupon_prob,
            "expected_life": life.mean(), "prob_maturity": alive.mean(), "prob_loss": loss.mean(),
            "avg_loss_given_loss": float(1 - worst[loss, -1].mean()) if loss.any() else 0.0}


def price(product: Phoenix, market: Market, n_paths: int = 200_000, seed: int = 42,
          antithetic: bool = True) -> dict:
    """Price in % of notional, Monte Carlo standard error and life statistics."""
    worst = simulate_worst_of(product, market, n_paths, seed, antithetic)
    res = cash_flows_pv(product, market, worst)
    pv = res.pop("pv")
    # antithetic pairs are not independent: average each pair before the standard error
    samples = 0.5 * (pv[: len(pv) // 2] + pv[len(pv) // 2:]) if antithetic else pv
    res["price"] = pv.mean()
    res["stderr"] = samples.std(ddof=1) / np.sqrt(len(samples))
    return res


def fair_coupon(product: Phoenix, market: Market, target: float = 98.0, **kw) -> float:
    """Coupon per period that makes the note worth `target` (100 minus the issuer margin).

    With common random numbers the price is exactly linear in the coupon, so two
    valuations give the answer without any root search.
    """
    p0 = price(replace(product, coupon=0.0), market, **kw)["price"]
    p1 = price(replace(product, coupon=0.01), market, **kw)["price"]
    return 0.01 * (target - p0) / (p1 - p0)


def sensitivities(product: Phoenix, market: Market, n_paths: int = 200_000, seed: int = 42) -> dict:
    """Bump-and-reprice sensitivities in % of notional, all with the same random numbers
    (common random numbers remove most of the Monte Carlo noise from the differences)."""
    base = price(product, market, n_paths, seed)["price"]
    out = {}
    perf = simulate_performances(product, market, n_paths, seed)
    for k in range(perf.shape[2]):
        bumped = perf.copy()
        bumped[:, :, k] *= 1.01                      # spot +1%, strike levels fixed at inception
        out[f"Delta asset {k + 1} (spot +1%)"] = cash_flows_pv(product, market, bumped.min(axis=2))["pv"].mean() - base

    def reprice(**changes):
        return price(product, replace(market, **changes), n_paths, seed)["price"] - base

    out["Vega (+1 vol pt, both assets)"] = reprice(vols=tuple(v + 0.01 for v in market.vols))
    out["Correlation (+10 pts)"] = reprice(corr=min(market.corr + 0.10, 0.999))
    out["Rho (+1% rate)"] = reprice(r=market.r + 0.01)
    return out
