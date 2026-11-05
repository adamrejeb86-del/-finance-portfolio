"""
Yield curve construction, interest rate swaps and bond analytics.

- Bootstrapping of a discount curve from money-market deposits and par swap rates
  (single-curve framework, annual fixed leg), with log-linear interpolation of discount
  factors (equivalent to piecewise-flat forward rates).
- Interest rate swap valuation, parallel DV01 and key-rate (bucketed) DV01s obtained by
  bumping each market quote and rebuilding the curve, the way risk systems do it.
- Nelson-Siegel-Svensson fit of the zero curve (the model used by central banks).
- Fixed-rate bonds: price from the curve, yield to maturity, Macaulay and modified
  duration, convexity.

Simplifications, stated on purpose: year fractions are in years (no day-count or
calendar conventions) and one curve is used for both discounting and forwarding.

Author: Adam Rejeb
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq, least_squares


@dataclass
class Curve:
    times: np.ndarray                    # pillar times (years), starting with 0
    log_dfs: np.ndarray                  # log discount factors at the pillars

    def df(self, t):
        t = np.asarray(t, dtype=float)
        last_rate = -self.log_dfs[-1] / self.times[-1]
        inside = np.interp(t, self.times, self.log_dfs)
        return np.exp(np.where(t <= self.times[-1], inside, -last_rate * t))

    def zero(self, t):
        t = np.maximum(np.asarray(t, dtype=float), 1e-8)
        return -np.log(self.df(t)) / t

    def forward(self, t1, t2):
        return np.log(self.df(t1) / self.df(t2)) / (np.asarray(t2) - np.asarray(t1))


@dataclass
class MarketQuotes:
    deposits: dict = field(default_factory=lambda: {1 / 12: 0.0200, 3 / 12: 0.0205, 6 / 12: 0.0210})
    swaps: dict = field(default_factory=lambda: {1: 0.0215, 2: 0.0222, 3: 0.0230, 4: 0.0237, 5: 0.0244,
                                                  7: 0.0255, 10: 0.0266, 15: 0.0276, 20: 0.0279, 30: 0.0275})

    def bumped(self, bp=1.0, only=None):
        """Copy with quotes shifted by `bp` basis points (all, or only the pillar `only`)."""
        b = bp * 1e-4
        dep = {t: r + (b if only in (None, ("dep", t)) else 0) for t, r in self.deposits.items()}
        sw = {t: r + (b if only in (None, ("swap", t)) else 0) for t, r in self.swaps.items()}
        return MarketQuotes(dep, sw)


def bootstrap(quotes: MarketQuotes) -> Curve:
    """Build the discount curve pillar by pillar so that every instrument reprices exactly."""
    times, log_dfs = [0.0], [0.0]
    for t, rate in sorted(quotes.deposits.items()):          # simple-interest deposits
        times.append(t)
        log_dfs.append(-np.log(1 + rate * t))

    for T, rate in sorted(quotes.swaps.items()):              # par swaps: annual fixed vs float
        def par_error(log_df_T):
            c = Curve(np.array(times + [T]), np.array(log_dfs + [log_df_T]))
            pay = np.arange(1, T + 1)
            return rate * c.df(pay).sum() - (1 - c.df(T))     # PV fixed leg - PV float leg (per unit)
        log_df_T = brentq(par_error, -5.0, 0.5, xtol=1e-14)
        times.append(float(T))
        log_dfs.append(log_df_T)
    return Curve(np.array(times), np.array(log_dfs))


# ---------------------------------------------------------------- swaps
def annuity(curve: Curve, T: float) -> float:
    return float(curve.df(np.arange(1, T + 1)).sum())


def par_swap_rate(curve: Curve, T: float) -> float:
    return float((1 - curve.df(T)) / annuity(curve, T))


def swap_npv(curve: Curve, notional: float, fixed_rate: float, T: float, pay_fixed: bool = True) -> float:
    """NPV of a spot-starting swap. Payer = pay fixed, receive float."""
    float_leg = 1 - curve.df(T)
    fixed_leg = fixed_rate * annuity(curve, T)
    sign = 1 if pay_fixed else -1
    return float(sign * notional * (float_leg - fixed_leg))


def dv01(quotes: MarketQuotes, pricer, bp: float = 1.0) -> float:
    """Change in value for a +1 bp parallel shift of all market quotes (curve rebuilt)."""
    return (pricer(bootstrap(quotes.bumped(bp))) - pricer(bootstrap(quotes.bumped(-bp)))) / 2


def key_rate_dv01s(quotes: MarketQuotes, pricer, bp: float = 1.0) -> dict:
    """Bucketed sensitivities: bump one quote at a time. They add up to (about) the parallel DV01."""
    out = {}
    for kind, book in (("dep", quotes.deposits), ("swap", quotes.swaps)):
        for t in book:
            up = pricer(bootstrap(quotes.bumped(bp, only=(kind, t))))
            down = pricer(bootstrap(quotes.bumped(-bp, only=(kind, t))))
            label = f"{round(t * 12)}M" if kind == "dep" else f"{t}Y"
            out[label] = (up - down) / 2
    return out


# ---------------------------------------------------------------- Nelson-Siegel-Svensson
def nss_zero(t, b0, b1, b2, b3, tau1, tau2):
    t = np.maximum(np.asarray(t, dtype=float), 1e-8)
    x1, x2 = t / tau1, t / tau2
    f1 = (1 - np.exp(-x1)) / x1
    return b0 + b1 * f1 + b2 * (f1 - np.exp(-x1)) + b3 * ((1 - np.exp(-x2)) / x2 - np.exp(-x2))


def fit_nss(t, zeros):
    """Fit NSS to observed zero rates. b0 is the long-run level, b1 the slope, b2/b3 the humps."""
    guess = [zeros[-1], zeros[0] - zeros[-1], 0.0, 0.0, 1.5, 10.0]
    lb, ub = [-0.1, -0.2, -0.3, -0.3, 0.05, 5.0], [0.2, 0.2, 0.3, 0.3, 5.0, 50.0]   # tau1 < tau2: distinct humps
    sol = least_squares(lambda p: nss_zero(t, *p) - zeros, guess, bounds=(lb, ub), xtol=1e-14, ftol=1e-14)
    return sol.x


# ---------------------------------------------------------------- bonds
def bond_cashflows(coupon: float, maturity: int, face: float = 100.0):
    times = np.arange(1, maturity + 1, dtype=float)
    flows = np.full(maturity, coupon * face)
    flows[-1] += face
    return times, flows


def bond_price_from_curve(curve: Curve, coupon: float, maturity: int, face: float = 100.0) -> float:
    times, flows = bond_cashflows(coupon, maturity, face)
    return float((flows * curve.df(times)).sum())


def bond_price_from_yield(y: float, coupon: float, maturity: int, face: float = 100.0) -> float:
    times, flows = bond_cashflows(coupon, maturity, face)
    return float((flows / (1 + y) ** times).sum())


def yield_to_maturity(price: float, coupon: float, maturity: int, face: float = 100.0) -> float:
    return brentq(lambda y: bond_price_from_yield(y, coupon, maturity, face) - price, -0.05, 0.5, xtol=1e-14)


def bond_risk(y: float, coupon: float, maturity: int, face: float = 100.0) -> dict:
    """Yield-based duration and convexity (annual compounding)."""
    times, flows = bond_cashflows(coupon, maturity, face)
    pv = flows / (1 + y) ** times
    price = pv.sum()
    macaulay = (times * pv).sum() / price
    modified = macaulay / (1 + y)
    convexity = (times * (times + 1) * pv).sum() / (price * (1 + y) ** 2)
    return {"price": price, "macaulay_duration": macaulay, "modified_duration": modified,
            "convexity": convexity, "dv01": modified * price * 1e-4}
