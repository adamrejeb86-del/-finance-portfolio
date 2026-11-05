from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from lbo import Assumptions, build_model, export_excel, sensitivity, sources_and_uses, value_bridge

A = Assumptions()
M = build_model(A)


def test_sources_equal_uses():
    su = sources_and_uses(A)
    assert su["Sources"].sum() == pytest.approx(su["Uses"].sum())


def test_debt_never_negative_and_cash_floor_respected():
    p = M["projection"]
    assert (p["Term loan B"] >= -1e-9).all() and (p["Cash"] >= A.min_cash - 1e-9).all()


def test_value_bridge_adds_up():
    b = value_bridge(A, M)
    assert b.drop("Total equity gain").sum() == pytest.approx(M["exit_equity"] - M["entry_equity"])
    assert b["Total equity gain"] == pytest.approx(M["exit_equity"] - M["entry_equity"])


def test_irr_consistent_with_moic():
    assert (1 + M["irr"]) ** A.years == pytest.approx(M["moic"])


def test_returns_move_in_the_right_direction():
    assert build_model(replace(A, exit_multiple=11))["irr"] > M["irr"]
    assert build_model(replace(A, entry_multiple=11))["irr"] < M["irr"]
    assert build_model(replace(A, revenue_growth=0.0))["irr"] < M["irr"]


def test_more_leverage_raises_irr_when_debt_is_cheap_enough():
    assert build_model(replace(A, tlb_multiple=5.0))["irr"] > build_model(replace(A, tlb_multiple=3.0))["irr"]


def test_excel_export(tmp_path):
    path = tmp_path / "lbo.xlsx"
    export_excel(A, M, str(path))
    sheets = pd.read_excel(path, sheet_name=None)
    assert set(sheets) == {"Assumptions", "Sources & uses", "Projections", "Returns"}
