"""
Credit risk from two angles: the market (CDS) and the balance sheet (Merton).

CDS (reduced-form, hazard rate)
  - Default time with piecewise-constant hazard rates; survival S(t) = exp(-int lambda).
  - Premium leg: quarterly spread payments while the name survives, plus accrued premium
    on default. Protection leg: (1 - R) paid at default.
  - Bootstrapping of hazard rates from the CDS curve, standard-coupon upfront, risky PV01, CS01.

Merton (structural)
  - Equity is a call on the firm's assets with strike = debt. From the observed equity value
    and volatility, solve for the unobservable asset value and asset volatility, then derive
    the distance to default, the risk-neutral default probability and the credit spread.

Author: Adam Rejeb
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, fsolve
from scipy.stats import norm


# ---------------------------------------------------------------- hazard-rate curve
@dataclass
class HazardCurve:
    knots: np.ndarray        # end of each segment (years)
    hazards: np.ndarray      # constant hazard rate on each segment

    def cumulative(self, t):
        t = np.asarray(t, dtype=float)
        starts = np.concatenate([[0.0], self.knots[:-1]])
        lam = np.append(self.hazards, self.hazards[-1])
        ends = np.append(self.knots, np.inf)
        starts = np.append(starts, self.knots[-1])
        overlap = np.clip(t[..., None], starts, ends) - starts
        return (overlap * lam).sum(axis=-1)

    def survival(self, t):
        return np.exp(-self.cumulative(t))


def _legs(curve: HazardCurve, T, r, recovery, freq=4, grid_per_year=48):
    """Risky PV01 (premium leg per unit spread, incl. accrual on default) and protection PV."""
    pay = np.arange(1, int(round(T * freq)) + 1) / freq
    dfs = np.exp(-r * pay)
    surv = curve.survival(pay)
    rpv01 = (dfs * surv / freq).sum()

    grid = np.linspace(0, T, int(round(T * grid_per_year)) + 1)
    s = curve.survival(grid)
    mid = 0.5 * (grid[1:] + grid[:-1])
    default_prob = s[:-1] - s[1:]                               # probability of default in each sub-interval
    df_mid = np.exp(-r * mid)
    protection = (1 - recovery) * (df_mid * default_prob).sum()
    accrued = (df_mid * default_prob * ((mid * freq) % 1) / freq).sum()   # premium accrued since last payment
    return rpv01 + accrued, protection


def par_spread(curve: HazardCurve, T, r=0.03, recovery=0.4):
    rpv01, prot = _legs(curve, T, r, recovery)
    return prot / rpv01


def bootstrap(maturities, spreads, r=0.03, recovery=0.4) -> HazardCurve:
    """Hazard rates segment by segment so that every quoted CDS has zero value at its par spread."""
    knots, hazards = [], []
    for T, s in zip(maturities, spreads):
        def error(lam):
            c = HazardCurve(np.array(knots + [T], float), np.array(hazards + [lam]))
            return par_spread(c, T, r, recovery) - s
        hazards.append(brentq(error, 1e-8, 5.0, xtol=1e-14))
        knots.append(T)
    return HazardCurve(np.array(knots, float), np.array(hazards))


def upfront(curve: HazardCurve, T, coupon=0.01, r=0.03, recovery=0.4):
    """Upfront (per unit notional) paid by the protection buyer for a standard-coupon CDS."""
    rpv01, prot = _legs(curve, T, r, recovery)
    return prot - coupon * rpv01


def cs01(maturities, spreads, T, coupon=0.01, r=0.03, recovery=0.4):
    """Change in the protection buyer's value for +1 bp on all CDS quotes."""
    base = upfront(bootstrap(maturities, spreads, r, recovery), T, coupon, r, recovery)
    up = upfront(bootstrap(maturities, np.asarray(spreads) + 1e-4, r, recovery), T, coupon, r, recovery)
    return up - base


# ---------------------------------------------------------------- Merton model
def merton_equity(V, sigma_V, D, T, r):
    d1 = (np.log(V / D) + (r + 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))
    d2 = d1 - sigma_V * np.sqrt(T)
    return V * norm.cdf(d1) - D * np.exp(-r * T) * norm.cdf(d2), d1, d2


def merton_solve(E, sigma_E, D, T=1.0, r=0.03):
    """Asset value and asset volatility consistent with the observed equity value and volatility:
         E = V N(d1) - D e^{-rT} N(d2)       and       sigma_E E = N(d1) sigma_V V
    """
    def equations(x):
        V, sV = x
        eq, d1, _ = merton_equity(V, sV, D, T, r)
        return [eq - E, norm.cdf(d1) * sV * V - sigma_E * E]

    V, sV = fsolve(equations, [E + D * np.exp(-r * T), sigma_E * E / (E + D)], xtol=1e-13)
    _, d1, d2 = merton_equity(V, sV, D, T, r)
    debt_value = V - E
    spread = -np.log(debt_value / (D * np.exp(-r * T))) / T
    return {"asset_value": V, "asset_vol": sV, "distance_to_default": d2, "pd_risk_neutral": norm.cdf(-d2),
            "debt_value": debt_value, "credit_spread": spread, "leverage": D / V}


def merton_spread_curve(V, sigma_V, D, maturities, r=0.03):
    out = []
    for T in maturities:
        eq, _, _ = merton_equity(V, sigma_V, D, T, r)
        out.append(-np.log((V - eq) / (D * np.exp(-r * T))) / T)
    return np.array(out)
