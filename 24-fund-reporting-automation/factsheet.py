"""
Automated monthly fund reporting: from raw NAV and holdings data to a client-ready PDF
factsheet and an Excel data pack, for any number of funds in one run.

  1. Analytics: performance over standard periods (1M, 3M, YTD, 1Y, 3Y annualised, since
     inception), calendar-month table, risk statistics against the benchmark.
  2. Charts: cumulative performance, drawdowns, sector allocation.
  3. A short commentary generated from the numbers (rule-based, so every sentence is traceable
     to a figure and nothing is invented).
  4. A two-page PDF built with reportlab, and an Excel pack with the underlying data.

Funds, holdings and figures are fictitious.

Author: Adam Rejeb
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import pandas as pd

SECTORS = ["Technology", "Healthcare", "Financials", "Industrials", "Consumer", "Energy", "Utilities", "Materials"]


@dataclass
class FundInfo:
    name: str
    strategy: str
    benchmark: str
    share_class: str
    inception: str
    aum_m: float
    fees: str
    manager: str


# ---------------------------------------------------------------- synthetic data
def simulate_fund(seed=0, years=5, alpha=0.02, beta=0.95, te=0.04, n_holdings=40):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end="2026-10-30", periods=252 * years)
    bench = rng.standard_t(5, len(dates)) * 0.16 / np.sqrt(252) * np.sqrt(3 / 5) + 0.07 / 252
    active = rng.normal(alpha / 252, te / np.sqrt(252), len(dates))
    fund = beta * bench + active
    nav = pd.Series(100 * np.cumprod(1 + fund), dates)
    bnav = pd.Series(100 * np.cumprod(1 + bench), dates)
    names = [f"Company {chr(65 + i // 10)}{i % 10:d}" for i in range(n_holdings)]
    w = rng.dirichlet(np.ones(n_holdings) * 1.2)
    one_month = nav.iloc[-1] / nav[: nav.index[-1] - pd.DateOffset(months=1)].iloc[-1] - 1
    stock_r = rng.normal(0, 0.06, n_holdings)
    stock_r = stock_r - w @ stock_r + one_month                       # holdings add up to the fund's month
    holdings = pd.DataFrame({"holding": names, "sector": rng.choice(SECTORS, n_holdings), "weight": w,
                             "month_return": stock_r}).sort_values("weight", ascending=False)
    holdings["contribution"] = holdings["weight"] * holdings["month_return"]
    return nav, bnav, holdings.reset_index(drop=True)


# ---------------------------------------------------------------- analytics
def period_returns(nav: pd.Series, asof=None) -> pd.Series:
    asof = nav.index[-1] if asof is None else asof
    nav = nav[:asof]
    end = nav.iloc[-1]

    def back(offset):
        start = nav[: asof - offset]
        return end / start.iloc[-1] - 1 if len(start) else np.nan

    ytd_start = nav[: pd.Timestamp(asof.year - 1, 12, 31)]
    three_y = back(pd.DateOffset(years=3))
    return pd.Series({"1M": back(pd.DateOffset(months=1)), "3M": back(pd.DateOffset(months=3)),
                      "YTD": end / ytd_start.iloc[-1] - 1, "1Y": back(pd.DateOffset(years=1)),
                      "3Y p.a.": (1 + three_y) ** (1 / 3) - 1,
                      "Inception p.a.": (end / nav.iloc[0]) ** (252 / (len(nav) - 1)) - 1})


def monthly_table(nav: pd.Series) -> pd.DataFrame:
    m = nav.resample("ME").last()
    first = pd.Series([nav.iloc[0]], index=[m.index[0] - pd.offsets.MonthEnd(1)])
    r = pd.concat([first, m]).pct_change().dropna()
    table = pd.DataFrame({"year": r.index.year, "month": r.index.strftime("%b"), "r": r.values})
    out = table.pivot(index="year", columns="month", values="r")
    order = [pd.Timestamp(2000, i, 1).strftime("%b") for i in range(1, 13)]
    out = out.reindex(columns=[c for c in order if c in out.columns])
    out["Year"] = (1 + out.fillna(0)).prod(axis=1) - 1
    return out


def risk_stats(nav: pd.Series, bnav: pd.Series, window_years=3, rf=0.02) -> pd.Series:
    r, b = nav.pct_change().dropna(), bnav.pct_change().dropna()
    r, b = r[-252 * window_years:], b[-252 * window_years:]
    active = r - b
    beta = np.cov(r, b)[0, 1] / b.var()
    wealth = (1 + r).cumprod()
    return pd.Series({"Volatility": r.std() * np.sqrt(252), "Benchmark volatility": b.std() * np.sqrt(252),
                      "Sharpe ratio": (r.mean() * 252 - rf) / (r.std() * np.sqrt(252)), "Beta": beta,
                      "Tracking error": active.std() * np.sqrt(252),
                      "Information ratio": active.mean() * 252 / (active.std() * np.sqrt(252)) if active.std() > 0 else np.nan,
                      "Max drawdown": (wealth / wealth.cummax() - 1).min()})


def commentary(info: FundInfo, perf: pd.Series, bperf: pd.Series, holdings: pd.DataFrame) -> str:
    """Rule-based commentary: every statement is computed from the data."""
    diff = perf["1M"] - bperf["1M"]
    verb = "outperformed" if diff > 0 else "underperformed"
    top = holdings.nlargest(2, "contribution")
    worst = holdings.nsmallest(1, "contribution").iloc[0]
    sector = holdings.groupby("sector")["contribution"].sum()
    text = (f"Over the month, {info.name} returned {perf['1M']:+.2%} against {bperf['1M']:+.2%} for the {info.benchmark}, "
            f"and {verb} by {abs(diff) * 1e4:.0f} bp. The main contributors were {top.iloc[0]['holding']} "
            f"({top.iloc[0]['contribution'] * 1e4:+.0f} bp) and {top.iloc[1]['holding']} ({top.iloc[1]['contribution'] * 1e4:+.0f} bp), "
            f"while {worst['holding']} cost {worst['contribution'] * 1e4:+.0f} bp. By sector, {sector.idxmax()} added the most "
            f"and {sector.idxmin()} detracted the most. Year to date, the fund is {perf['YTD']:+.2%} against "
            f"{bperf['YTD']:+.2%} for its benchmark.")
    return text


# ---------------------------------------------------------------- charts
def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    buf.seek(0)
    return buf


def charts(nav, bnav, holdings):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    navy, grey, bord = "#1f3b5c", "#8a8a8a", "#7a1f2b"
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.plot(nav.index, nav / nav.iloc[0] * 100, color=navy, lw=1.4, label="Fund")
    ax.plot(bnav.index, bnav / bnav.iloc[0] * 100, color=grey, lw=1.2, label="Benchmark")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=8); ax.set_title("Performance since inception (base 100)", fontsize=9)
    perf_png = _png(fig); plt.close(fig)

    sec = holdings.groupby("sector")["weight"].sum().sort_values()
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.barh(sec.index, sec.values * 100, color=navy)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Sector allocation (%)", fontsize=9); ax.tick_params(labelsize=7)
    sector_png = _png(fig); plt.close(fig)

    wealth = nav / nav.cummax() - 1
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.fill_between(wealth.index, wealth * 100, 0, color=bord, alpha=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Drawdown (%)", fontsize=9); ax.tick_params(labelsize=7)
    dd_png = _png(fig); plt.close(fig)
    return perf_png, sector_png, dd_png


# ---------------------------------------------------------------- PDF and Excel
def build_factsheet(info: FundInfo, nav, bnav, holdings, path_pdf: str, path_xlsx: str | None = None):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    perf, bperf = period_returns(nav), period_returns(bnav)
    risk = risk_stats(nav, bnav)
    mt = monthly_table(nav)
    perf_png, sector_png, dd_png = charts(nav, bnav, holdings)
    navy = colors.HexColor("#1F3B5C")
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=16, textColor=navy, alignment=0, spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=ss["Heading3"], textColor=navy, spaceBefore=6, spaceAfter=3)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=8.5, leading=11)
    small = ParagraphStyle("s", parent=body, fontSize=6.5, textColor=colors.grey)

    def table(data, widths, header=True):
        t = Table(data, colWidths=widths)
        style = [("FONTSIZE", (0, 0), (-1, -1), 7.5), ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                 ("LINEBELOW", (0, 0), (-1, 0), 0.6, navy), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F5F8")])]
        if header:
            style += [("TEXTCOLOR", (0, 0), (-1, 0), navy), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
        t.setStyle(TableStyle(style))
        return t

    asof = nav.index[-1].strftime("%d %B %Y")
    story = [Paragraph(info.name, h1), Paragraph(f"{info.strategy} · Monthly factsheet as of {asof}", body), Spacer(1, 4 * mm)]
    facts = [["Key facts", ""], ["Share class", info.share_class], ["Benchmark", info.benchmark], ["Inception", info.inception],
             ["Assets under management", f"EUR {info.aum_m:,.0f}m"], ["Fees", info.fees], ["Portfolio manager", info.manager],
             ["NAV per share", f"{nav.iloc[-1]:,.2f}"]]
    top = Table([[table(facts, [38 * mm, 42 * mm]), Image(perf_png, width=100 * mm, height=36 * mm)]], colWidths=[85 * mm, 100 * mm])
    top.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story += [top]
    perf_rows = [["", *perf.index], ["Fund", *[f"{x:+.2%}" for x in perf]], ["Benchmark", *[f"{x:+.2%}" for x in bperf]],
                 ["Relative", *[f"{(a - b) * 1e4:+.0f} bp" for a, b in zip(perf, bperf)]]]
    story += [Paragraph("Performance", h2), table(perf_rows, [22 * mm] + [26 * mm] * 6)]
    story += [Paragraph("Manager commentary", h2), Paragraph(commentary(info, perf, bperf, holdings), body)]
    risk_rows = [["Risk (3 years)", "Value"]] + [[k, f"{v:.2%}" if k in ("Volatility", "Benchmark volatility", "Tracking error", "Max drawdown") else f"{v:.2f}"]
                                                 for k, v in risk.items()]
    top10 = holdings.head(10)
    top_rows = [["Top 10 holdings", "Sector", "Weight"]] + [[r.holding, r.sector, f"{r.weight:.1%}"] for r in top10.itertuples()]
    side = Table([[table(risk_rows, [38 * mm, 22 * mm]), table(top_rows, [38 * mm, 28 * mm, 18 * mm])]],
                 colWidths=[66 * mm, 90 * mm], hAlign="LEFT")
    side.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story += [Spacer(1, 3 * mm), side]
    story += [PageBreak(), Paragraph("Monthly returns", h2)]
    mrows = [["", *mt.columns]] + [[str(y), *["" if pd.isna(v) else f"{v:+.1%}" for v in row]] for y, row in mt.iterrows()]
    story += [table(mrows, [14 * mm] + [13 * mm] * (len(mt.columns))), Spacer(1, 4 * mm)]
    story += [Table([[Image(sector_png, width=85 * mm, height=65 * mm), Image(dd_png, width=85 * mm, height=65 * mm)]])]
    story += [Spacer(1, 6 * mm), Paragraph("This document is a fictitious example produced for a portfolio project. Past performance "
                                           "is not a reliable indicator of future performance.", small)]
    SimpleDocTemplate(path_pdf, pagesize=A4, leftMargin=12 * mm, rightMargin=12 * mm, topMargin=12 * mm, bottomMargin=12 * mm).build(story)

    if path_xlsx:
        with pd.ExcelWriter(path_xlsx, engine="openpyxl") as xw:
            pd.DataFrame({"Fund": perf, "Benchmark": bperf}).to_excel(xw, sheet_name="Performance")
            risk.to_frame("Value").to_excel(xw, sheet_name="Risk")
            mt.to_excel(xw, sheet_name="Monthly returns")
            holdings.to_excel(xw, sheet_name="Holdings", index=False)
            pd.DataFrame({"Fund NAV": nav, "Benchmark": bnav}).to_excel(xw, sheet_name="NAV history")
    return {"performance": perf, "benchmark": bperf, "risk": risk}
