"""
A quant sales toolkit for structured notes: from a client brief to a priced, explained and
documented trade.

What a structured products salesperson does with a client:
  1. Understand the brief (horizon, view, coupon target, how much capital risk is acceptable).
  2. Propose products: a Phoenix autocall, a reverse convertible, a capital-protected note.
  3. Price them as the issuing bank would: the note is sold at 100, its cash flows are worth
     less than 100 to the bank (funding benefit + sales margin), and the coupon is the variable
     that makes the two match.
  4. Answer "what if" questions: what barrier for a 7% coupon? What if the stocks fall 30%?
  5. Send an indicative term sheet.

Every payoff here is linear in its coupon (or participation), so on a fixed set of Monte Carlo
paths the fair coupon is solved exactly in one step: price = A + coupon x B.

Underlyings are fictional and their parameters illustrative.

Author: Adam Rejeb
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- market
UNDERLYINGS = {                     # name: (volatility, dividend yield)
    "Euro large-cap index": (0.17, 0.030),
    "European bank": (0.30, 0.060),
    "Luxury goods": (0.27, 0.020),
    "Energy major": (0.25, 0.050),
    "Chip maker": (0.38, 0.005),
}
SECTOR_CORR = 0.55                  # pairwise correlation between single stocks
INDEX_CORR = 0.75                   # correlation between a stock and the index


@dataclass(frozen=True)
class Market:
    names: tuple = ("Euro large-cap index",)
    rate: float = 0.025             # risk-free rate (continuous)
    funding_spread: float = 0.008   # issuer's credit spread: note cash flows are discounted at r + s
    vol_shift: float = 0.0          # parallel shift of all volatilities, for scenarios

    @property
    def vols(self):
        return np.array([UNDERLYINGS[n][0] for n in self.names]) + self.vol_shift

    @property
    def divs(self):
        return np.array([UNDERLYINGS[n][1] for n in self.names])

    @property
    def corr(self):
        k = len(self.names)
        C = np.full((k, k), SECTOR_CORR)
        for i, n in enumerate(self.names):
            if "index" in n:
                C[i, :] = C[:, i] = INDEX_CORR
        np.fill_diagonal(C, 1.0)
        return C


def simulate_performance(m: Market, times, n_paths=20_000, seed=0):
    """Performances S_k(t)/S_k(0) at the observation dates under the risk-neutral measure
    (exact lognormal steps between dates, antithetic variates). Shape (paths, dates, assets)."""
    rng = np.random.default_rng(seed)
    times = np.asarray(times, float)
    dt = np.diff(np.concatenate([[0.0], times]))
    L = np.linalg.cholesky(m.corr)
    half = n_paths // 2
    z = rng.standard_normal((half, len(times), len(m.names))) @ L.T
    z = np.concatenate([z, -z])
    drift = (m.rate - m.divs - 0.5 * m.vols**2) * dt[:, None]
    log_perf = np.cumsum(drift + m.vols * np.sqrt(dt)[:, None] * z, axis=1)
    return np.exp(log_perf)


# ---------------------------------------------------------------- products
@dataclass(frozen=True)
class Phoenix:
    """Worst-of Phoenix autocall with memory coupon."""
    maturity: float = 5.0
    obs_per_year: int = 4
    autocall_level: float = 1.00
    coupon_barrier: float = 0.60
    protection_barrier: float = 0.60
    first_call: int = 4             # first observation at which the note can be called
    memory: bool = True
    kind: str = field(default="Phoenix autocall", init=False)

    @property
    def times(self):
        return np.arange(1, int(round(self.maturity * self.obs_per_year)) + 1) / self.obs_per_year


@dataclass(frozen=True)
class ReverseConvertible:
    """Fixed coupon every period; at maturity 100% back unless the worst-of is below the barrier."""
    maturity: float = 2.0
    obs_per_year: int = 4
    protection_barrier: float = 0.70
    kind: str = field(default="Barrier reverse convertible", init=False)

    @property
    def times(self):
        return np.arange(1, int(round(self.maturity * self.obs_per_year)) + 1) / self.obs_per_year


@dataclass(frozen=True)
class CapitalProtected:
    """100% capital protection at maturity plus a participation in the rise of the
    equally weighted basket (the "coupon" solved for is the participation rate)."""
    maturity: float = 6.0
    cap: float = np.inf             # maximum basket performance paid (e.g. 0.5 = +50%)
    kind: str = field(default="Capital-protected note", init=False)

    @property
    def times(self):
        return np.array([self.maturity])


def cashflows(product, perf):
    """Cash flows per EUR 100 of notional, split into a fixed part and a part proportional to the
    coupon. Returns (fixed, per_coupon, redemption_time_index, loss) with arrays of shape
    (paths, dates) for the first two."""
    n, d = perf.shape[0], perf.shape[1]
    fixed, unit = np.zeros((n, d)), np.zeros((n, d))
    if isinstance(product, CapitalProtected):
        basket = perf[:, -1, :].mean(axis=1)
        fixed[:, -1] = 100.0
        unit[:, -1] = 100.0 * np.minimum(np.maximum(basket - 1, 0.0), product.cap)
        return fixed, unit, np.full(n, d - 1), np.zeros(n)
    worst = perf.min(axis=2)
    alive = np.ones(n, bool)
    accrued = np.zeros(n)
    end = np.full(n, d - 1)
    for k in range(d):
        if isinstance(product, Phoenix):
            accrued[alive] += 1.0
            pay = alive & (worst[:, k] >= product.coupon_barrier)
            unit[pay, k] = 100.0 * (accrued[pay] if product.memory else 1.0)
            accrued[pay] = 0.0
            if k < d - 1 and k + 1 >= product.first_call:
                called = alive & (worst[:, k] >= product.autocall_level)
                fixed[called, k] = 100.0
                end[called] = k
                alive &= ~called
        else:
            unit[:, k] = 100.0
    protected = worst[:, -1] >= product.protection_barrier
    redemption = np.where(protected, 100.0, 100.0 * worst[:, -1])
    fixed[alive, -1] += redemption[alive]
    loss = np.where(alive & ~protected, 100.0 - redemption, 0.0)
    return fixed, unit, end, loss


def solve_coupon(product, m: Market, sales_margin=0.015, n_paths=20_000, seed=0):
    """Coupon per period (or participation) such that the note's value to the issuer equals
    100 - margin. Discounting at r + funding spread: the issuer pays its own credit spread,
    which is what funds part of the coupon."""
    perf = simulate_performance(m, product.times, n_paths, seed)
    fixed, unit, end, loss = cashflows(product, perf)
    df = np.exp(-(m.rate + m.funding_spread) * product.times)
    A, B = (fixed @ df).mean(), (unit @ df).mean()
    target = 100.0 * (1 - sales_margin)
    c = (target - A) / B
    life = product.times[end]
    out = {"coupon_per_period": c, "value_check": A + c * B, "target": target, "expected_life": life.mean(),
           "prob_capital_loss": (loss > 0).mean(), "avg_loss_if_loss": loss[loss > 0].mean() if (loss > 0).any() else 0.0,
           "expected_loss": loss.mean()}
    if isinstance(product, Phoenix):
        out["coupon_pa"] = c * product.obs_per_year
        calls = pd.Series(end).value_counts().sort_index()
        out["redemption_profile"] = pd.Series(calls.values / len(end), index=product.times[calls.index])
        out["prob_autocall"] = (fixed[:, :-1].sum(axis=1) > 0).mean()
    elif isinstance(product, ReverseConvertible):
        out["coupon_pa"] = c * product.obs_per_year
    else:
        out["participation"] = c
    return out


def solve_barrier(product, m: Market, target_coupon_pa, lo=0.30, hi=1.0, tol=1e-3, **kw):
    """Barrier (coupon barrier = protection barrier) giving a target coupon p.a.
    The coupon rises with the barrier, so a bisection is enough."""
    def coupon(b):
        return solve_coupon(replace(product, coupon_barrier=b, protection_barrier=b) if isinstance(product, Phoenix)
                            else replace(product, protection_barrier=b), m, **kw)["coupon_pa"]
    if not coupon(lo) <= target_coupon_pa <= coupon(hi):
        raise ValueError("target coupon not reachable in the barrier range")
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if coupon(mid) < target_coupon_pa else (lo, mid)
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------- client scenarios
def scenario_path(product, final_level, shape="linear"):
    """Worst-of performance at each observation date for a stylised scenario: a straight line
    from 100% to `final_level`, or a dip to `final_level` halfway then a recovery to 100%."""
    t = product.times / product.times[-1]
    if shape == "linear":
        w = 1 + (final_level - 1) * t
    else:
        w = np.where(t <= 0.5, 1 + (final_level - 1) * t / 0.5, final_level + (1 - final_level) * (t - 0.5) / 0.5)
    return w


def scenario_outcome(product, coupon_per_period, worst_path):
    """What the client receives on one scenario: redemption date, coupons, capital returned."""
    perf = worst_path[None, :, None]
    fixed, unit, end, loss = cashflows(product, perf)
    k = int(end[0])
    coupons = unit[0, : k + 1].sum() * coupon_per_period
    capital = fixed[0, : k + 1].sum()
    return {"Redeemed after (years)": product.times[k], "Coupons received (%)": coupons,
            "Capital returned (%)": capital, "Total received (%)": coupons + capital}


# ---------------------------------------------------------------- term sheet
def term_sheet_pdf(path, product, m: Market, quote: dict, scenarios: pd.DataFrame, notional="EUR 1,000,000"):
    """One-page indicative term sheet (illustration only)."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    navy = colors.HexColor("#1f3b5c")
    ss = getSampleStyleSheet()
    h = ParagraphStyle("h", parent=ss["Title"], fontSize=15, textColor=navy, spaceAfter=2)
    sub = ParagraphStyle("s", parent=ss["Normal"], fontSize=8.5, textColor=colors.HexColor("#7a1f2b"))
    body = ParagraphStyle("b", parent=ss["Normal"], fontSize=8.5, leading=11)
    sec = ParagraphStyle("sec", parent=ss["Heading4"], textColor=navy, spaceBefore=6, spaceAfter=3)
    is_phx = isinstance(product, Phoenix)
    coupon_txt = (f"{quote['coupon_pa']:.2%} p.a., paid {'quarterly' if product.obs_per_year == 4 else f'{product.obs_per_year} times a year'}"
                  if "coupon_pa" in quote else f"Participation of {quote['participation']:.0%} in the basket's rise")
    rows = [["Product", product.kind], ["Underlying(s)", ", ".join(m.names) + (" (worst-of)" if len(m.names) > 1 and not isinstance(product, CapitalProtected) else "")],
            ["Issuer", "Illustrative issuer, rated A"], ["Notional", notional], ["Issue price", "100%"],
            ["Maturity", f"{product.maturity:g} years"], ["Coupon", coupon_txt]]
    if is_phx:
        rows += [["Coupon barrier", f"{product.coupon_barrier:.0%} of initial level{' (memory effect)' if product.memory else ''}"],
                 ["Autocall", f"From year {product.first_call / product.obs_per_year:g}, early redemption at 100% on any observation date where the worst-of is at or above {product.autocall_level:.0%}"]]
    if not isinstance(product, CapitalProtected):
        rows += [["Capital at maturity", f"100% if the worst-of is at or above {product.protection_barrier:.0%}; otherwise 100% x final worst-of performance"]]
    else:
        rows += [["Capital at maturity", "100% protected (subject to issuer credit risk)"]]
    t = Table([[Paragraph(f"<b>{a}</b>", body), Paragraph(b, body)] for a, b in rows], colWidths=[42 * mm, 128 * mm])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c8c8c8")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef2f6"))]))
    sc = scenarios.rename_axis("Scenario").reset_index()
    st = Table([[Paragraph(f"<b>{c}</b>", body) for c in sc.columns]] +
               [[Paragraph(str(v) if not isinstance(v, float) else f"{v:.1f}", body) for v in r] for r in sc.values],
               colWidths=[60 * mm] + [27.5 * mm] * (sc.shape[1] - 1))
    st.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c8c8c8")), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f6"))]))
    stats = (f"Model indications (Monte Carlo, risk-neutral): expected life {quote['expected_life']:.1f} years; "
             f"probability of a capital loss {quote['prob_capital_loss']:.0%}; average loss when there is one {quote['avg_loss_if_loss']:.0f}%.")
    story = [Paragraph("Indicative Terms", h), Paragraph("For illustration only. Not an offer. Fictional underlyings and issuer.", sub),
             Spacer(1, 6), t, Paragraph("Client scenarios (per 100 invested)", sec), st, Spacer(1, 4), Paragraph(stats, body),
             Paragraph("Main risks", sec),
             Paragraph("Capital loss if the barrier is breached at maturity, potentially total. Issuer credit risk: payments depend on "
                       "the issuer's ability to pay. Early redemption risk: an autocall ends the coupons. Secondary market prices can be "
                       "well below 100 before maturity. Coupons are not guaranteed.", body)]
    SimpleDocTemplate(path, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=14 * mm,
                      title="Indicative term sheet", author="Adam Rejeb").build(story)
    return path
