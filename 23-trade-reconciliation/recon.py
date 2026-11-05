"""
Automated OTC trade reconciliation, in the spirit of EMIR reporting controls.

Under EMIR, both sides of a derivative report it to a trade repository, and the reports must
match on the Unique Trade Identifier (UTI) and on key economic fields. Operations teams
reconcile their bookings against counterparty data every day and chase every break.

Pipeline
  1. Match on UTI (exact key).
  2. For trades without a usable UTI, economic matching: same counterparty, product, trade
     date and currency, notional within a tolerance; ties resolved by the closest notional.
  3. Field-by-field comparison with tolerances (MTM relative tolerance, rates in basis
     points, dates exact), each difference classified by type and severity.
  4. A formatted Excel report for the operations team: summary, breaks, one-sided trades.

The trades, counterparties and identifiers are synthetic.

Author: Adam Rejeb
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

COUNTERPARTIES = ["JPMorgan", "Goldman Sachs", "BofA Securities", "Morgan Stanley", "BNP Paribas", "Societe Generale",
                  "Barclays", "Citigroup", "Deutsche Bank", "HSBC"]
PRODUCTS = ["IRS", "Cross-currency swap", "FX forward", "Equity option", "CDS"]
FIELDS = ["counterparty", "product", "trade_date", "maturity", "currency", "notional", "rate", "direction", "mtm"]
SEVERITY = {"missing at counterparty": "High", "missing internally": "High", "notional": "High", "direction": "High",
            "currency": "High", "rate": "Medium", "maturity": "Medium", "trade_date": "Medium", "mtm": "Low",
            "counterparty": "High", "product": "High"}


@dataclass(frozen=True)
class Tolerances:
    notional_rel: float = 1e-4        # 0.01%
    rate_bp: float = 0.5
    mtm_rel: float = 0.02             # 2% of notional-scaled MTM ...
    mtm_abs: float = 5_000.0          # ... or EUR 5k, whichever is larger


# ---------------------------------------------------------------- synthetic data
def generate_trades(n=5_000, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    trade_date = pd.Timestamp("2026-01-02") + pd.to_timedelta(rng.integers(0, 300, n), unit="D")
    product = rng.choice(PRODUCTS, n, p=[0.40, 0.10, 0.25, 0.15, 0.10])
    tenor_years = np.where(product == "FX forward", rng.choice([0.25, 0.5, 1.0], n), rng.choice([1, 2, 3, 5, 7, 10], n))
    notional = np.round(rng.lognormal(np.log(25e6), 0.9, n), -5)
    rate = np.round(np.where(product == "FX forward", rng.normal(1.09, 0.02, n), rng.normal(0.028, 0.006, n)), 6)
    df = pd.DataFrame({
        "uti": [f"UTI{seed:02d}{i:08d}" for i in range(n)],
        "counterparty": rng.choice(COUNTERPARTIES, n),
        "product": product,
        "trade_date": trade_date.normalize(),
        "maturity": (trade_date + pd.to_timedelta((tenor_years * 365.25).astype(int), unit="D")).normalize(),
        "currency": rng.choice(["EUR", "USD", "GBP"], n, p=[0.55, 0.35, 0.10]),
        "notional": notional,
        "rate": rate,
        "direction": rng.choice(["Pay", "Receive"], n),
        "mtm": np.round(notional * rng.normal(0, 0.015, n), 2),
    })
    return df


def counterparty_view(internal: pd.DataFrame, break_rate=0.04, seed=1):
    """Counterparty's copy of the trades with realistic breaks injected. Returns (external, truth)."""
    rng = np.random.default_rng(seed)
    ext = internal.copy()
    ext["mtm"] = ext["mtm"] + rng.normal(0, 500, len(ext))          # small valuation noise, within tolerance
    n = len(ext)
    idx = rng.permutation(n)
    k = int(break_rate * n / 8)
    groups = {t: idx[i * k:(i + 1) * k] for i, t in enumerate(["missing at counterparty", "notional", "rate", "direction",
                                                                "maturity", "currency", "mtm", "uti missing"])}
    truth = []
    ext.loc[ext.index[groups["notional"]], "notional"] *= 1.10
    ext.loc[ext.index[groups["rate"]], "rate"] += np.where(ext.loc[ext.index[groups["rate"]], "product"] == "FX forward", 0.01, 0.0010)
    ext.loc[ext.index[groups["direction"]], "direction"] = ext.loc[ext.index[groups["direction"]], "direction"].map({"Pay": "Receive", "Receive": "Pay"})
    ext.loc[ext.index[groups["maturity"]], "maturity"] += pd.Timedelta(days=1)
    ext.loc[ext.index[groups["currency"]], "currency"] = ext.loc[ext.index[groups["currency"]], "currency"].map({"EUR": "USD", "USD": "EUR", "GBP": "EUR"})
    ext.loc[ext.index[groups["mtm"]], "mtm"] += np.sign(rng.normal(size=k)) * np.maximum(0.05 * ext.loc[ext.index[groups["mtm"]], "notional"].abs() * 0.015, 50_000)
    ext.loc[ext.index[groups["uti missing"]], "uti"] = None                     # must be matched on economics
    for t, ids in groups.items():
        for i in ids:
            truth.append({"uti": internal.iloc[i]["uti"], "break": t})
    ext = ext.drop(index=ext.index[groups["missing at counterparty"]])
    extra = generate_trades(max(k, 1), seed=seed + 100)                       # trades only the counterparty has
    ext = pd.concat([ext, extra], ignore_index=True)
    truth += [{"uti": u, "break": "missing internally"} for u in extra["uti"]]
    return ext.sample(frac=1, random_state=seed).reset_index(drop=True), pd.DataFrame(truth)


# ---------------------------------------------------------------- reconciliation
def reconcile(internal: pd.DataFrame, external: pd.DataFrame, tol: Tolerances = Tolerances()) -> dict:
    t0 = time.perf_counter()
    a = internal.add_suffix("_int")
    b = external.add_suffix("_ext")
    b["row_ext"] = np.arange(len(b))                                   # unique id for counterparty rows

    # 1. UTI match
    m1 = a.merge(b.dropna(subset=["uti_ext"]), left_on="uti_int", right_on="uti_ext", how="inner")
    m1["match_method"] = "UTI"
    left = a[~a["uti_int"].isin(m1["uti_int"])]
    right = b[~b["row_ext"].isin(m1["row_ext"])]

    # 2. Economic match on what is left: same key fields, closest notional, one-to-one
    keys = ["counterparty", "product", "trade_date", "currency"]
    cand = left.merge(right, left_on=[k + "_int" for k in keys], right_on=[k + "_ext" for k in keys], how="inner")
    cand["notional_gap"] = (cand["notional_ext"] / cand["notional_int"] - 1).abs()
    cand = cand[cand["notional_gap"] <= 0.25].sort_values("notional_gap")
    used_int, used_ext, picked = set(), set(), []
    for row in cand.itertuples(index=False):
        if row.uti_int in used_int or row.row_ext in used_ext:
            continue
        used_int.add(row.uti_int); used_ext.add(row.row_ext); picked.append(row)
    m2 = pd.DataFrame(picked, columns=cand.columns).drop(columns="notional_gap")
    m2["match_method"] = "Economic"
    matched = pd.concat([m1, m2], ignore_index=True)
    only_int = a[~a["uti_int"].isin(matched["uti_int"])]
    only_ext = b[~b["row_ext"].isin(matched["row_ext"])]

    # 3. Field comparison
    breaks = []
    for f in ["counterparty", "product", "currency", "direction", "trade_date", "maturity"]:
        diff = matched[matched[f + "_int"] != matched[f + "_ext"]]
        breaks += [_brk(r, f, r[f + "_int"], r[f + "_ext"]) for _, r in diff.iterrows()]
    rel = (matched["notional_ext"] / matched["notional_int"] - 1).abs()
    breaks += [_brk(r, "notional", r["notional_int"], r["notional_ext"]) for _, r in matched[rel > tol.notional_rel].iterrows()]
    rate_gap = (matched["rate_ext"] - matched["rate_int"]).abs()
    rate_tol = np.where(matched["product_int"] == "FX forward", 1e-4 * matched["rate_int"].abs(), tol.rate_bp * 1e-4)
    breaks += [_brk(r, "rate", r["rate_int"], r["rate_ext"]) for _, r in matched[rate_gap > rate_tol].iterrows()]
    mtm_tol = np.maximum(tol.mtm_abs, tol.mtm_rel * matched["mtm_int"].abs())
    breaks += [_brk(r, "mtm", r["mtm_int"], r["mtm_ext"]) for _, r in matched[(matched["mtm_ext"] - matched["mtm_int"]).abs() > mtm_tol].iterrows()]
    breaks += [{"uti": r["uti_int"], "counterparty": r["counterparty_int"], "product": r["product_int"], "field": "missing at counterparty",
                "internal": r["notional_int"], "counterparty_value": None, "severity": "High", "match_method": "-"} for _, r in only_int.iterrows()]
    breaks += [{"uti": r["uti_ext"], "counterparty": r["counterparty_ext"], "product": r["product_ext"], "field": "missing internally",
                "internal": None, "counterparty_value": r["notional_ext"], "severity": "High", "match_method": "-"} for _, r in only_ext.iterrows()]
    breaks = pd.DataFrame(breaks)
    elapsed = time.perf_counter() - t0
    return {"matched": matched, "breaks": breaks, "only_internal": only_int, "only_external": only_ext, "seconds": elapsed}


def _brk(r, field, vi, ve):
    return {"uti": r["uti_int"], "counterparty": r["counterparty_int"], "product": r["product_int"], "field": field,
            "internal": vi, "counterparty_value": ve, "severity": SEVERITY[field], "match_method": r["match_method"]}


# ---------------------------------------------------------------- reporting
def summary(res: dict, n_internal: int) -> pd.DataFrame:
    b = res["breaks"]
    return pd.DataFrame({
        "Internal trades": [n_internal], "Matched on UTI": [(res["matched"]["match_method"] == "UTI").sum()],
        "Matched on economics": [(res["matched"]["match_method"] == "Economic").sum()],
        "Trades with a break": [b["uti"].nunique()], "High-severity breaks": [(b["severity"] == "High").sum()],
        "Run time (s)": [round(res["seconds"], 2)]}).T.rename(columns={0: "Value"})


def write_excel(res: dict, n_internal: int, path: str):
    from openpyxl.formatting.rule import CellIsRule
    from openpyxl.styles import Font, PatternFill
    b = res["breaks"].copy()
    by_type = b.pivot_table(index="field", columns="severity", values="uti", aggfunc="count", fill_value=0)
    by_cpty = b.pivot_table(index="counterparty", columns="field", values="uti", aggfunc="count", fill_value=0)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        summary(res, n_internal).to_excel(xw, sheet_name="Summary")
        by_type.to_excel(xw, sheet_name="Summary", startrow=10)
        by_cpty.to_excel(xw, sheet_name="By counterparty")
        b.sort_values(["severity", "counterparty"]).to_excel(xw, sheet_name="Breaks", index=False)
        head = PatternFill("solid", fgColor="1F3B5C")
        for ws in xw.book.worksheets:
            for cell in ws[1]:
                cell.fill, cell.font = head, Font(color="FFFFFF", bold=True)
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = max(12, min(40, max(len(str(c.value or "")) for c in col) + 2))
            ws.freeze_panes = "A2"
        ws = xw.book["Breaks"]
        ws.auto_filter.ref = ws.dimensions
        sev_col = [c.column_letter for c in ws[1] if c.value == "severity"][0]
        rng = f"{sev_col}2:{sev_col}{ws.max_row}"
        ws.conditional_formatting.add(rng, CellIsRule(operator="equal", formula=['"High"'], fill=PatternFill("solid", fgColor="F4C7C3")))
        ws.conditional_formatting.add(rng, CellIsRule(operator="equal", formula=['"Medium"'], fill=PatternFill("solid", fgColor="FCE8B2")))
