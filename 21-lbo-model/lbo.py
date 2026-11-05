"""
Leveraged buyout model: sources and uses, operating projections, debt schedule with cash
sweep, credit statistics, returns (IRR, MoIC), value-creation bridge and sensitivity tables,
plus an Excel export laid out like a banker's model.

Conventions (stated, as in any model review):
  - Interest is charged on the opening debt balance of each year, which avoids the circular
    reference of average-balance interest.
  - Free cash flow above the minimum cash balance sweeps the term loan B (at a set rate);
    the senior notes are repaid only at exit.
  - Exit at the end of year N at an EV/EBITDA multiple; no interim dividends.

Author: Adam Rejeb
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Assumptions:
    revenue: float = 500.0             # EUR m, last twelve months
    ebitda_margin: float = 0.20
    margin_target: float = 0.23        # reached linearly by the exit year
    revenue_growth: float = 0.06
    da_pct: float = 0.03               # depreciation and amortisation, % of revenue
    capex_pct: float = 0.04
    nwc_pct: float = 0.10              # net working capital, % of revenue
    tax_rate: float = 0.25
    entry_multiple: float = 10.0
    exit_multiple: float = 10.0
    tlb_multiple: float = 4.0          # term loan B, x EBITDA
    tlb_rate: float = 0.070
    tlb_amort: float = 0.01            # mandatory amortisation, % of original amount
    notes_multiple: float = 1.5        # senior notes, x EBITDA
    notes_rate: float = 0.085
    sweep: float = 0.75                # share of excess cash used to prepay the term loan
    fees_pct: float = 0.02             # transaction fees, % of EV
    min_cash: float = 10.0
    years: int = 5

    @property
    def entry_ebitda(self):
        return self.revenue * self.ebitda_margin


def sources_and_uses(a: Assumptions) -> pd.DataFrame:
    ev = a.entry_multiple * a.entry_ebitda
    fees = a.fees_pct * ev
    tlb, notes = a.tlb_multiple * a.entry_ebitda, a.notes_multiple * a.entry_ebitda
    uses = {"Purchase of the company (EV)": ev, "Transaction fees": fees, "Cash to balance sheet": a.min_cash}
    total = sum(uses.values())
    sources = {"Term loan B": tlb, "Senior notes": notes, "Sponsor equity": total - tlb - notes}
    return pd.DataFrame({"Sources": pd.Series(sources), "Uses": pd.Series(uses)})


def build_model(a: Assumptions) -> dict:
    su = sources_and_uses(a)
    tlb0, notes = su.loc["Term loan B", "Sources"], su.loc["Senior notes", "Sources"]
    equity0 = su.loc["Sponsor equity", "Sources"]
    years = np.arange(0, a.years + 1)
    rev = a.revenue * (1 + a.revenue_growth) ** years
    margin = a.ebitda_margin + (a.margin_target - a.ebitda_margin) * years / a.years
    ebitda = rev * margin
    rows, tlb, cash = [], tlb0, a.min_cash
    for y in years[1:]:
        da, capex = a.da_pct * rev[y], a.capex_pct * rev[y]
        d_nwc = a.nwc_pct * (rev[y] - rev[y - 1])
        interest = tlb * a.tlb_rate + notes * a.notes_rate
        ebit = ebitda[y] - da
        taxes = max(ebit - interest, 0) * a.tax_rate
        net_income = ebit - interest - taxes
        fcf = net_income + da - capex - d_nwc                   # levered free cash flow
        mandatory = min(a.tlb_amort * tlb0, tlb)
        available = cash + fcf - mandatory - a.min_cash
        sweep = min(max(available, 0) * a.sweep, tlb - mandatory)
        tlb -= mandatory + sweep
        cash += fcf - mandatory - sweep
        rows.append({"Year": int(y), "Revenue": rev[y], "EBITDA": ebitda[y], "EBITDA margin": margin[y], "D&A": da, "EBIT": ebit,
                     "Interest": interest, "Taxes": taxes, "Net income": net_income, "Capex": capex, "Change in NWC": d_nwc,
                     "Free cash flow": fcf, "Mandatory amortisation": mandatory, "Cash sweep": sweep, "Term loan B": tlb,
                     "Senior notes": notes, "Cash": cash, "Net debt": tlb + notes - cash,
                     "Net debt / EBITDA": (tlb + notes - cash) / ebitda[y], "EBITDA / interest": ebitda[y] / interest})
    proj = pd.DataFrame(rows).set_index("Year")
    exit_ev = a.exit_multiple * ebitda[-1]
    exit_equity = exit_ev - proj["Net debt"].iloc[-1]
    moic = exit_equity / equity0
    irr = moic ** (1 / a.years) - 1                             # single in-flow and out-flow
    return {"sources_uses": su, "projection": proj, "entry_equity": equity0, "exit_ev": exit_ev,
            "exit_equity": exit_equity, "moic": moic, "irr": irr, "entry_ebitda": ebitda[0], "exit_ebitda": ebitda[-1]}


def value_bridge(a: Assumptions, m: dict) -> pd.Series:
    """Splits the equity gain into its sources. The parts add up exactly to exit - entry equity."""
    proj = m["projection"]
    ebitda_growth = a.entry_multiple * (m["exit_ebitda"] - m["entry_ebitda"])
    multiple = (a.exit_multiple - a.entry_multiple) * m["exit_ebitda"]
    debt0 = a.entry_ebitda * (a.tlb_multiple + a.notes_multiple)
    deleveraging = (debt0 - a.min_cash) - proj["Net debt"].iloc[-1]
    fees = -a.fees_pct * a.entry_multiple * a.entry_ebitda
    return pd.Series({"EBITDA growth": ebitda_growth, "Multiple expansion": multiple, "Debt paydown": deleveraging,
                      "Transaction fees": fees, "Total equity gain": ebitda_growth + multiple + deleveraging + fees})


def sensitivity(a: Assumptions, row_field, row_values, col_field, col_values, metric="irr") -> pd.DataFrame:
    data = [[build_model(replace(a, **{row_field: r, col_field: c}))[metric] for c in col_values] for r in row_values]
    return pd.DataFrame(data, index=row_values, columns=col_values)


def export_excel(a: Assumptions, m: dict, path: str):
    """Banker-style workbook: assumptions, sources and uses, projections, returns."""
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        pd.Series(vars(a)).to_frame("Value").to_excel(xw, sheet_name="Assumptions")
        m["sources_uses"].to_excel(xw, sheet_name="Sources & uses")
        m["projection"].T.to_excel(xw, sheet_name="Projections")
        pd.Series({"Entry equity": m["entry_equity"], "Exit EV": m["exit_ev"], "Exit equity": m["exit_equity"],
                   "MoIC": m["moic"], "IRR": m["irr"]}).to_frame("Value").to_excel(xw, sheet_name="Returns")
        header = PatternFill("solid", fgColor="1F3B5C")
        for ws in xw.book.worksheets:
            for cell in ws[1]:
                cell.fill, cell.font = header, Font(color="FFFFFF", bold=True)
            ws.column_dimensions["A"].width = 30
            for col in range(2, ws.max_column + 1):
                ws.column_dimensions[get_column_letter(col)].width = 14
                for row in ws.iter_rows(min_row=2, min_col=col, max_col=col):
                    for c in row:
                        if isinstance(c.value, float):
                            c.number_format = "0.0%" if abs(c.value) < 1 and ws.title in ("Assumptions",) else "#,##0.0"
                            c.alignment = Alignment(horizontal="right")
            ws.freeze_panes = "B2"
