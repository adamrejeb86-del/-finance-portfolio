import pandas as pd
import pytest

from recon import counterparty_view, generate_trades, reconcile, write_excel

INT = generate_trades(4000, seed=3)
EXT, TRUTH = counterparty_view(INT, break_rate=0.05, seed=4)
RES = reconcile(INT, EXT)


def test_identical_books_have_no_breaks():
    res = reconcile(INT, INT.copy())
    assert res["breaks"].empty and len(res["matched"]) == len(INT)


def test_every_injected_break_is_found_and_classified():
    found = set(zip(RES["breaks"]["uti"], RES["breaks"]["field"]))
    for _, t in TRUTH[TRUTH["break"] != "uti missing"].iterrows():
        assert (t["uti"], t["break"]) in found


def test_no_false_positives():
    injected = set(zip(TRUTH["uti"], TRUTH["break"]))
    assert all((u, f) in injected for u, f in zip(RES["breaks"]["uti"], RES["breaks"]["field"]))


def test_trades_without_uti_are_matched_on_economics():
    no_uti = set(TRUTH.loc[TRUTH["break"] == "uti missing", "uti"])
    econ = RES["matched"][RES["matched"]["match_method"] == "Economic"]
    assert set(econ["uti_int"]) == no_uti


def test_small_valuation_noise_is_tolerated():
    ext = INT.copy()
    ext["mtm"] += 1_000.0
    assert reconcile(INT, ext)["breaks"].empty


def test_excel_report(tmp_path):
    path = tmp_path / "recon.xlsx"
    write_excel(RES, len(INT), str(path))
    sheets = pd.read_excel(path, sheet_name=None)
    assert {"Summary", "By counterparty", "Breaks"} <= set(sheets) and len(sheets["Breaks"]) == len(RES["breaks"])
