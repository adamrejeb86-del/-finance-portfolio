import numpy as np
import pandas as pd
import pytest

from factsheet import FundInfo, build_factsheet, commentary, monthly_table, period_returns, risk_stats, simulate_fund

NAV, BNAV, HOLD = simulate_fund(seed=1)
INFO = FundInfo("Test Equity Fund", "European equities", "test index", "I EUR", "2021-11-01", 500, "0.80%", "A. Manager")


def test_constant_growth_series():
    dates = pd.bdate_range("2020-01-01", periods=252 * 4)
    nav = pd.Series(100 * 1.0004 ** np.arange(len(dates)), dates)
    p = period_returns(nav)
    assert p["Inception p.a."] == pytest.approx(1.0004**252 - 1, rel=1e-6)
    assert risk_stats(nav, nav)["Max drawdown"] == 0.0


def test_monthly_table_compounds_to_the_annual_figure():
    mt = monthly_table(NAV)
    year = 2025
    nav_y = NAV[str(year)]
    prev = NAV[: f"{year - 1}-12-31"].iloc[-1]
    assert mt.loc[year, "Year"] == pytest.approx(nav_y.iloc[-1] / prev - 1, abs=1e-12)


def test_risk_stats_are_sensible():
    r = risk_stats(NAV, BNAV)
    assert 0.5 < r["Beta"] < 1.5 and r["Tracking error"] > 0 and r["Max drawdown"] < 0


def test_commentary_quotes_the_numbers():
    text = commentary(INFO, period_returns(NAV), period_returns(BNAV), HOLD)
    assert f"{period_returns(NAV)['1M']:+.2%}" in text and HOLD.nlargest(1, "contribution").iloc[0]["holding"] in text


def test_pdf_and_excel_are_produced(tmp_path):
    pdf, xlsx = tmp_path / "f.pdf", tmp_path / "f.xlsx"
    build_factsheet(INFO, NAV, BNAV, HOLD, str(pdf), str(xlsx))
    assert pdf.read_bytes()[:4] == b"%PDF" and pdf.stat().st_size > 20_000
    assert set(pd.read_excel(xlsx, sheet_name=None)) == {"Performance", "Risk", "Monthly returns", "Holdings", "NAV history"}
