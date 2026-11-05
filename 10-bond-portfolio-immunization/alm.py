"""
Asset-liability management for a pension-type liability: immunisation strategies and
interest rate scenario analysis.

An insurer or pension fund owes a stream of future payments. Their present value moves
with interest rates, so the assets backing them must move the same way. Four strategies
are built from a universe of government bonds and compared:

  1. Duration matching (barbell of a short and a long bond): match PV and duration.
  2. Redington immunisation: match PV and duration, and hold more convexity than the
     liabilities (three bonds, convexity target set slightly above the liabilities').
  3. Key-rate duration matching: match the sensitivity to each part of the curve
     (non-negative least squares).
  4. Cash-flow matching: the cheapest bond portfolio whose cash flows cover every payment
     (linear programming), with surpluses reinvested at a conservative rate.

Surplus = PV(assets) - PV(liabilities) is then stressed with parallel, slope and
curvature scenarios and with random curve moves from a level/slope/curvature factor model.

Author: Adam Rejeb
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog, nnls


# ---------------------------------------------------------------- curve and scenarios
@dataclass(frozen=True)
class ZeroCurve:
    level: float = 0.030
    slope: float = -0.010
    curvature: float = 0.005
    tau: float = 3.0

    def zero(self, t, shock=None):
        t = np.maximum(np.asarray(t, dtype=float), 1e-8)
        x = t / self.tau
        f = (1 - np.exp(-x)) / x
        z = self.level + self.slope * f + self.curvature * (f - np.exp(-x))
        return z if shock is None else z + shock(t)

    def df(self, t, shock=None):
        return np.exp(-self.zero(t, shock) * np.asarray(t, dtype=float))


def parallel(bp):
    return lambda t: np.full_like(t, bp * 1e-4)


def twist(bp, pivot=10.0):
    """Steepener for bp > 0: short rates down, long rates up, rotating around the pivot."""
    return lambda t: bp * 1e-4 * np.clip((t - pivot) / pivot, -1, 2)


def butterfly(bp, belly=7.0):
    """Belly up (bp > 0) relative to both wings."""
    return lambda t: bp * 1e-4 * np.exp(-((t - belly) / 4.0) ** 2)


def key_rate_shock(pillar, pillars, bp=1.0):
    """Triangular bump of 1 bp at one pillar, fading linearly to the neighbouring pillars."""
    def shock(t):
        idx = list(pillars).index(pillar)
        left = pillars[idx - 1] if idx > 0 else 0.0
        right = pillars[idx + 1] if idx < len(pillars) - 1 else 1e9
        up = np.clip((t - left) / (pillar - left), 0, 1) if idx > 0 else np.ones_like(t)
        down = np.clip((right - t) / (right - pillar), 0, 1) if idx < len(pillars) - 1 else np.ones_like(t)
        return bp * 1e-4 * np.minimum(up, down)
    return shock


# ---------------------------------------------------------------- cash flows
def liability_cashflows(years=40, first=10.0, growth=0.03, peak=15, decay=0.12):
    """Pension-type payments (EUR m): growing until `peak`, then running off."""
    t = np.arange(1, years + 1, dtype=float)
    cf = np.where(t <= peak, first * (1 + growth) ** (t - 1), first * (1 + growth) ** (peak - 1) * np.exp(-decay * (t - peak)))
    return t, cf


def bond_universe(maturities=(2, 5, 10, 15, 20, 30, 40), coupons=(0.025, 0.027, 0.03, 0.032, 0.033, 0.034, 0.035)):
    """Cash-flow matrix: rows = bonds (face 100), columns = years 1..max."""
    horizon = max(maturities)
    flows = np.zeros((len(maturities), horizon))
    for i, (m, c) in enumerate(zip(maturities, coupons)):
        flows[i, :m] = 100 * c
        flows[i, m - 1] += 100
    return np.arange(1, horizon + 1, dtype=float), flows, list(maturities)


# ---------------------------------------------------------------- analytics
def pv(times, flows, curve: ZeroCurve, shock=None):
    return float(np.dot(flows, curve.df(times, shock)))


def risk_measures(times, flows, curve: ZeroCurve):
    """Fisher-Weil duration and convexity (continuous zero rates)."""
    d = flows * curve.df(times)
    price = d.sum()
    return {"pv": price, "duration": (times * d).sum() / price, "convexity": (times**2 * d).sum() / price}


def portfolio_flows(weights, bond_flows):
    return weights @ bond_flows


# ---------------------------------------------------------------- strategies
def match_pv_duration(liab_t, liab_cf, bond_t, bond_cf, curve, pair):
    """Two bonds (indices in `pair`) matching PV and duration of the liabilities."""
    L = risk_measures(liab_t, liab_cf, curve)
    p = np.array([pv(bond_t, bond_cf[i], curve) for i in pair])
    dur = np.array([risk_measures(bond_t, bond_cf[i], curve)["duration"] for i in pair])
    A = np.vstack([p, p * dur])
    units = np.linalg.solve(A, [L["pv"], L["pv"] * L["duration"]])
    w = np.zeros(len(bond_cf)); w[list(pair)] = units
    return w


def redington(liab_t, liab_cf, bond_t, bond_cf, curve, trio, convexity_buffer=1.10):
    """Three bonds matching PV and duration, with convexity = buffer x liability convexity."""
    L = risk_measures(liab_t, liab_cf, curve)
    p = np.array([pv(bond_t, bond_cf[i], curve) for i in trio])
    rm = [risk_measures(bond_t, bond_cf[i], curve) for i in trio]
    dur, conv = np.array([x["duration"] for x in rm]), np.array([x["convexity"] for x in rm])
    A = np.vstack([p, p * dur, p * conv])
    units = np.linalg.solve(A, [L["pv"], L["pv"] * L["duration"], L["pv"] * L["convexity"] * convexity_buffer])
    w = np.zeros(len(bond_cf)); w[list(trio)] = units
    return w


def key_rate_durations(times, flows, curve, pillars):
    """Sensitivity of PV to a 1 bp triangular bump at each pillar (positive = loses value)."""
    base = pv(times, flows, curve)
    return np.array([(base - pv(times, flows, curve, key_rate_shock(k, pillars))) for k in pillars])


def match_key_rates(liab_t, liab_cf, bond_t, bond_cf, curve, pillars):
    """Non-negative bond holdings matching the liabilities' PV and key-rate DV01s."""
    L_pv = pv(liab_t, liab_cf, curve)
    L_krd = key_rate_durations(liab_t, liab_cf, curve, pillars)
    B_pv = np.array([pv(bond_t, f, curve) for f in bond_cf])
    B_krd = np.array([key_rate_durations(bond_t, f, curve, pillars) for f in bond_cf])
    scale = L_pv / L_krd.sum()                                  # put PV and DV01 rows on a comparable scale
    A = np.vstack([B_pv[None, :], B_krd.T * scale])
    b = np.concatenate([[L_pv], L_krd * scale])
    w, _ = nnls(A, b)
    return w


def par_bond_universe(curve: ZeroCurve, maturities=range(1, 41)):
    """One par-coupon bond per maturity (face 100), coupons set to the curve's par yields."""
    maturities = list(maturities)
    horizon = max(maturities)
    t = np.arange(1, horizon + 1, dtype=float)
    flows = np.zeros((len(maturities), horizon))
    for i, m in enumerate(maturities):
        dfs = curve.df(t[:m])
        c = (1 - dfs[-1]) / dfs.sum()
        flows[i, :m] = 100 * c
        flows[i, m - 1] += 100
    return t, flows, maturities


def cash_flow_matching(liab_cf, bond_cf, curve, bond_t, reinvest=0.02):
    """Cheapest non-negative portfolio whose cash covers every payment.

    Surplus cash is carried forward and reinvested at a conservative rate `reinvest`
    (below the curve). The cash balance at year t is sum_{s<=t} (A_s - L_s)(1+reinvest)^(t-s)
    and must stay non-negative: a set of linear constraints, solved by linear programming.
    """
    horizon = len(liab_cf)
    prices = np.array([pv(bond_t, f, curve) for f in bond_cf])
    growth = (1 + reinvest) ** (np.arange(horizon)[:, None] - np.arange(horizon)[None, :])
    growth = np.tril(growth)                                   # only past cash flows compound into year t
    asset_balance = growth @ bond_cf[:, :horizon].T             # (years x bonds)
    liab_balance = growth @ liab_cf
    res = linprog(c=prices, A_ub=-asset_balance, b_ub=-liab_balance, bounds=[(0, None)] * len(prices), method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    return res.x


# ---------------------------------------------------------------- scenarios
def surplus_change(weights, bond_t, bond_cf, liab_t, liab_cf, curve, shock):
    assets = portfolio_flows(weights, bond_cf)
    base = pv(bond_t, assets, curve) - pv(liab_t, liab_cf, curve)
    stressed = pv(bond_t, assets, curve, shock) - pv(liab_t, liab_cf, curve, shock)
    return stressed - base


def random_curve_shocks(n, seed=0, level_bp=80, slope_bp=40, curv_bp=20):
    """Annual curve moves from a 3-factor model (level, slope, curvature), in the spirit of
    a PCA of historical euro curve changes."""
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((n, 3)) * np.array([level_bp, slope_bp, curv_bp]) * 1e-4
    shocks = []
    for lv, sl, cv in f:
        shocks.append(lambda t, lv=lv, sl=sl, cv=cv: lv + sl * np.clip((t - 10) / 10, -1, 2) + cv * np.exp(-((t - 7) / 4) ** 2))
    return shocks
