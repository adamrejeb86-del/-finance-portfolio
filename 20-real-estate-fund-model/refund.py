"""
Real estate fund model: cash-flow projection, NAV per share, bank covenants and liquidation
strategies for a fund in difficulty.

The fund holds a diversified portfolio of buildings financed partly with bank debt.
  - Each year: rents (indexed, net of vacancy and non-recoverable costs) - capex - interest
    = distributable cash flow; buildings are valued as NOI / capitalisation rate.
  - Bank covenants are tested every year:
        LTV = debt / portfolio value            must stay below a maximum (e.g. 40%)
        ICR = net rental income / interest      must stay above a minimum (e.g. 2.0x)
  - When the fund must be wound down, the order and speed of disposals change how much
    investors get back: a fire sale crystallises discounts, a slow sale costs time and keeps
    the fund exposed to the market, and selling the best assets first can breach covenants.

All figures are illustrative; no real fund or portfolio is represented.

Author: Adam Rejeb
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

FIRE_SALE_DISCOUNT = {"Office": 0.22, "Retail": 0.25, "Logistics": 0.08, "Residential": 0.10, "Hotel": 0.18}


@dataclass(frozen=True)
class Asset:
    name: str
    sector: str
    value: float          # EUR m, latest appraisal
    rent: float           # EUR m per year, contractual rent at full occupancy
    occupancy: float
    cap_rate: float       # valuation yield on NOI
    capex: float = 0.0    # EUR m per year, maintenance and works

    @property
    def noi(self):
        return self.rent * self.occupancy * 0.92            # 8% non-recoverable costs


@dataclass
class Fund:
    assets: list
    debt: float = 160.0               # EUR m
    interest_rate: float = 0.042
    cash: float = 12.0
    shares: float = 0.22              # million shares
    max_ltv: float = 0.40
    min_icr: float = 2.0
    fees: float = 0.012               # management fee on gross asset value

    def portfolio_value(self):
        return sum(a.value for a in self.assets)

    def nav(self):
        return self.portfolio_value() + self.cash - self.debt

    def nav_per_share(self):
        return self.nav() / self.shares

    def ltv(self):
        return self.debt / self.portfolio_value()

    def icr(self):
        return sum(a.noi for a in self.assets) / (self.debt * self.interest_rate)


def sample_fund() -> Fund:
    """A diversified fund of 14 buildings, after a market correction (values down, offices half-empty)."""
    rows = [("Paris La Défense tower", "Office", 64, 4.4, 0.62, 0.060, 1.2), ("Lyon Part-Dieu offices", "Office", 38, 2.6, 0.78, 0.058, 0.6),
            ("Lille business park", "Office", 22, 1.8, 0.70, 0.070, 0.4), ("Marseille Euromed offices", "Office", 30, 2.2, 0.85, 0.062, 0.4),
            ("Retail park Bordeaux", "Retail", 26, 2.1, 0.88, 0.068, 0.3), ("High-street Nice", "Retail", 18, 1.0, 0.95, 0.050, 0.1),
            ("Shopping centre Rennes", "Retail", 34, 3.1, 0.80, 0.072, 0.8), ("Logistics hub Lyon", "Logistics", 42, 2.4, 0.98, 0.050, 0.2),
            ("Last-mile Paris Nord", "Logistics", 28, 1.5, 1.00, 0.047, 0.1), ("Logistics Le Havre", "Logistics", 24, 1.5, 0.95, 0.055, 0.2),
            ("Residential Paris 15e", "Residential", 36, 1.4, 0.97, 0.036, 0.2), ("Student housing Toulouse", "Residential", 20, 1.1, 0.96, 0.050, 0.2),
            ("Hotel Paris Opéra", "Hotel", 40, 2.4, 0.90, 0.052, 0.5), ("Hotel Nantes", "Hotel", 16, 1.2, 0.80, 0.065, 0.3)]
    assets = []
    for name, sector, _, rent, occ, cap, capex in rows:
        a = Asset(name, sector, 0.0, rent, occ, cap, capex)
        assets.append(replace(a, value=a.noi / cap))     # appraisal consistent with income and yield
    return Fund(assets)


# ---------------------------------------------------------------- projection
def project(fund: Fund, years=10, rent_growth=0.02, cap_shift=None, occupancy_change=None, payout=0.85) -> pd.DataFrame:
    """Year-by-year projection. cap_shift / occupancy_change: dicts {year: change} applied to all assets."""
    cap_shift = cap_shift or {}
    occupancy_change = occupancy_change or {}
    f = Fund([replace(a) for a in fund.assets], fund.debt, fund.interest_rate, fund.cash, fund.shares, fund.max_ltv, fund.min_icr, fund.fees)
    rows = []
    for y in range(1, years + 1):
        new_assets = []
        for a in f.assets:
            occ = float(np.clip(a.occupancy + occupancy_change.get(y, 0.0), 0.3, 1.0))
            cap = a.cap_rate + cap_shift.get(y, 0.0)
            rent = a.rent * (1 + rent_growth)
            tmp = replace(a, rent=rent, occupancy=occ, cap_rate=cap)
            new_assets.append(replace(tmp, value=tmp.noi / cap))
        f.assets = new_assets
        noi = sum(a.noi for a in f.assets)
        interest = f.debt * f.interest_rate
        capex = sum(a.capex for a in f.assets)
        fees = f.fees * f.portfolio_value()
        cash_flow = noi - interest - capex - fees
        distribution = max(cash_flow, 0.0) * payout
        f.cash += cash_flow - distribution
        rows.append({"year": y, "NOI": noi, "interest": interest, "capex": capex, "fees": fees, "cash flow": cash_flow,
                     "distribution": distribution, "portfolio value": f.portfolio_value(), "debt": f.debt, "cash": f.cash,
                     "NAV": f.nav(), "NAV per share": f.nav_per_share(), "LTV": f.ltv(), "ICR": noi / interest,
                     "LTV breach": f.ltv() > f.max_ltv, "ICR breach": noi / interest < f.min_icr})
    return pd.DataFrame(rows).set_index("year")


# ---------------------------------------------------------------- liquidation strategies
def liquidation(fund: Fund, order: list[str], schedule: list[int], discounts: dict | None = None, market_path=None,
                transaction_cost=0.02, discount_rate=0.07, rent_growth=0.015):
    """Sell assets in `order`, the k-th asset in year schedule[k] (0 = today, as a fire sale).

    discounts: sale discount by sector for sales in year 0 (fire sale); later sales happen at
    appraised value, which follows `market_path` (cap-rate change by year, cumulative).
    Proceeds first repay debt (the bank is senior); the rest, plus net rents of assets still
    held, is distributed. Returns the cash returned per share, its present value, and the
    covenant path during the wind-down.
    """
    discounts = FIRE_SALE_DISCOUNT if discounts is None else discounts
    market_path = market_path or {}
    by_name = {a.name: a for a in fund.assets}
    held = dict(by_name)
    debt, cash = fund.debt, fund.cash
    flows, log = [], []
    horizon = max(schedule)
    cum_shift = 0.0
    for y in range(horizon + 1):
        cum_shift += market_path.get(y, 0.0)
        # income of the year from assets still held (none in year 0, valuation date)
        if y > 0:
            noi = sum(a.noi * (1 + rent_growth) ** y for a in held.values())
            fees = fund.fees * sum(a.value for a in held.values())
            cash += noi - debt * fund.interest_rate - sum(a.capex for a in held.values()) - fees
        sold_value = 0.0
        for name, when in zip(order, schedule):
            if when != y or name not in held:
                continue
            a = held.pop(name)
            if y == 0:
                price = a.value * (1 - discounts.get(a.sector, 0.0))
            else:
                price = a.noi * (1 + rent_growth) ** y / (a.cap_rate + cum_shift)
            sold_value += price * (1 - transaction_cost)
        repay = min(debt, sold_value)
        debt -= repay
        cash += sold_value - repay
        value_left = sum(a.noi * (1 + rent_growth) ** y / (a.cap_rate + cum_shift) for a in held.values())
        ltv = debt / value_left if value_left > 0 else 0.0
        payout = cash if not held else max(cash - 2.0, 0.0)        # keep a 2m liquidity buffer until the end
        cash -= payout
        flows.append(payout)
        log.append({"year": y, "sold (EUR m)": sold_value, "debt": debt, "remaining value": value_left, "LTV": ltv,
                    "LTV breach": bool(held) and ltv > fund.max_ltv, "paid to investors": payout})
    flows = np.array(flows) / fund.shares
    pv = float(sum(f / (1 + discount_rate) ** t for t, f in enumerate(flows)))
    return {"cash per share": flows, "total per share": float(flows.sum()), "PV per share": pv, "log": pd.DataFrame(log).set_index("year")}


def irr(cashflows):
    """Internal rate of return by bisection (cashflows[0] usually negative)."""
    cf = np.asarray(cashflows, dtype=float)
    f = lambda r: sum(c / (1 + r) ** t for t, c in enumerate(cf))
    lo, hi = -0.99, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)
