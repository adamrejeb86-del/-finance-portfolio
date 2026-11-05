"""
Company valuation: discounted cash flows (DCF), trading comparables and a football field.

DCF (free cash flow to the firm)
  FCFF = EBIT (1 - t) + D&A - capex - change in working capital
  Enterprise value = sum of FCFF discounted at the WACC (mid-year convention)
                     + terminal value (Gordon growth, cross-checked with an exit multiple)
  Equity value = EV - net debt - pension deficit - minorities;  per share = / diluted shares

WACC: cost of equity from CAPM with the beta re-levered to the target capital structure
(Hamada), cost of debt after tax.

Also: WACC x growth sensitivity, Monte Carlo on the key drivers, reverse DCF (the growth the
share price implies), and valuation from peer multiples.

Author: Adam Rejeb
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.optimize import brentq


@dataclass(frozen=True)
class Company:
    name: str = "Helvia Industrial Tech (fictional)"
    revenue: float = 2_400.0          # EUR m, last fiscal year
    growth: tuple = (0.09, 0.08, 0.07, 0.06, 0.05)    # explicit forecast, then fades to terminal growth
    ebit_margin: tuple = (0.145, 0.150, 0.155, 0.160, 0.160)
    da_pct: float = 0.04
    capex_pct: float = 0.05
    nwc_pct: float = 0.15             # of incremental revenue
    tax_rate: float = 0.25
    net_debt: float = 900.0
    pension_deficit: float = 120.0
    minorities: float = 40.0
    shares: float = 120.0             # million, diluted
    share_price: float = 31.0


@dataclass(frozen=True)
class WaccInputs:
    risk_free: float = 0.030
    equity_risk_premium: float = 0.055
    unlevered_beta: float = 0.95
    target_debt_to_equity: float = 0.35
    pre_tax_cost_of_debt: float = 0.050
    tax_rate: float = 0.25

    @property
    def levered_beta(self):
        """Hamada: beta_L = beta_U x (1 + (1 - t) D/E)."""
        return self.unlevered_beta * (1 + (1 - self.tax_rate) * self.target_debt_to_equity)

    @property
    def cost_of_equity(self):
        return self.risk_free + self.levered_beta * self.equity_risk_premium

    @property
    def wacc(self):
        d = self.target_debt_to_equity / (1 + self.target_debt_to_equity)
        return (1 - d) * self.cost_of_equity + d * self.pre_tax_cost_of_debt * (1 - self.tax_rate)


def forecast(c: Company, terminal_growth=0.02, fade_years=5) -> pd.DataFrame:
    """Explicit years, then a linear fade of growth towards the terminal rate."""
    g = list(c.growth) + list(np.linspace(c.growth[-1], terminal_growth, fade_years + 1)[1:])
    margins = list(c.ebit_margin) + [c.ebit_margin[-1]] * fade_years
    rows, rev = [], c.revenue
    for t, (gt, mt) in enumerate(zip(g, margins), start=1):
        new_rev = rev * (1 + gt)
        ebit = new_rev * mt
        da, capex = c.da_pct * new_rev, c.capex_pct * new_rev
        d_nwc = c.nwc_pct * (new_rev - rev)
        fcff = ebit * (1 - c.tax_rate) + da - capex - d_nwc
        rows.append({"year": t, "revenue": new_rev, "growth": gt, "EBIT": ebit, "EBIT margin": mt, "NOPAT": ebit * (1 - c.tax_rate),
                     "D&A": da, "capex": capex, "change in NWC": d_nwc, "FCFF": fcff})
        rev = new_rev
    return pd.DataFrame(rows).set_index("year")


def gordon_terminal_value(last_fcff, wacc, g):
    if wacc <= g:
        raise ValueError("WACC must exceed terminal growth")
    return last_fcff * (1 + g) / (wacc - g)


def dcf_value(c: Company, wacc: float, terminal_growth=0.02, mid_year=True, exit_multiple=None) -> dict:
    f = forecast(c, terminal_growth)
    t = f.index.values.astype(float)
    disc_t = t - 0.5 if mid_year else t
    pv_fcff = (f["FCFF"] / (1 + wacc) ** disc_t).sum()
    if exit_multiple is None:
        tv = gordon_terminal_value(f["FCFF"].iloc[-1], wacc, terminal_growth)
    else:
        ebitda_last = f["EBIT"].iloc[-1] + f["D&A"].iloc[-1]
        tv = exit_multiple * ebitda_last
    pv_tv = tv / (1 + wacc) ** t[-1]
    ev = pv_fcff + pv_tv
    equity = ev - c.net_debt - c.pension_deficit - c.minorities
    ebitda_last = f["EBIT"].iloc[-1] + f["D&A"].iloc[-1]
    return {"forecast": f, "pv_fcff": pv_fcff, "terminal_value": tv, "pv_terminal": pv_tv, "ev": ev, "equity": equity,
            "per_share": equity / c.shares, "tv_share_of_ev": pv_tv / ev, "implied_exit_multiple": tv / ebitda_last}


def sensitivity(c: Company, waccs, growths) -> pd.DataFrame:
    return pd.DataFrame([[dcf_value(c, w, g)["per_share"] for g in growths] for w in waccs], index=waccs, columns=growths)


def reverse_dcf(c: Company, wacc: float, terminal_growth=0.02) -> float:
    """Uniform scaling of the explicit growth path that makes the DCF equal the share price."""
    def gap(k):
        return dcf_value(replace(c, growth=tuple(np.array(c.growth) * k)), wacc, terminal_growth)["per_share"] - c.share_price
    k = brentq(gap, -3.0, 5.0, xtol=1e-12)
    return float(np.mean(np.array(c.growth) * k))


def monte_carlo(c: Company, base_wacc: float, n=5000, seed=0) -> np.ndarray:
    """Value per share with uncertain growth, margins, WACC and terminal growth."""
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for i in range(n):
        g_shift = rng.normal(0, 0.02)
        m_shift = rng.normal(0, 0.012)
        w = base_wacc + rng.normal(0, 0.006)
        tg = float(np.clip(rng.normal(0.02, 0.004), 0.005, 0.03))
        ci = replace(c, growth=tuple(np.array(c.growth) + g_shift), ebit_margin=tuple(np.array(c.ebit_margin) + m_shift))
        out[i] = dcf_value(ci, max(w, tg + 0.02), tg)["per_share"]
    return out


def peers() -> pd.DataFrame:
    """Fictional listed peers with their trading multiples."""
    return pd.DataFrame({
        "EV/EBITDA (NTM)": [9.8, 11.2, 8.4, 12.5, 10.1, 9.2],
        "EV/EBIT (NTM)": [13.5, 15.0, 11.8, 17.2, 13.9, 12.6],
        "P/E (NTM)": [16.0, 19.5, 13.2, 22.0, 17.1, 15.4],
        "EBIT margin": [0.14, 0.17, 0.11, 0.19, 0.15, 0.13],
        "Revenue growth": [0.05, 0.08, 0.03, 0.10, 0.06, 0.04],
    }, index=["Peer A", "Peer B", "Peer C", "Peer D", "Peer E", "Peer F"])


def multiples_valuation(c: Company, multiples: pd.Series, ebitda_ntm: float, ebit_ntm: float, net_income_ntm: float) -> dict:
    """Value per share implied by each multiple (EV multiples go through the equity bridge)."""
    bridge = c.net_debt + c.pension_deficit + c.minorities
    return {"EV/EBITDA": (multiples["EV/EBITDA (NTM)"] * ebitda_ntm - bridge) / c.shares,
            "EV/EBIT": (multiples["EV/EBIT (NTM)"] * ebit_ntm - bridge) / c.shares,
            "P/E": multiples["P/E (NTM)"] * net_income_ntm / c.shares}
