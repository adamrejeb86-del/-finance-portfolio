import numpy as np
import pytest

from dcf import Company, WaccInputs, dcf_value, forecast, gordon_terminal_value, reverse_dcf, sensitivity

C, W = Company(), WaccInputs()


def test_wacc_components():
    assert W.levered_beta == pytest.approx(0.95 * (1 + 0.75 * 0.35))
    d = 0.35 / 1.35
    assert W.wacc == pytest.approx((1 - d) * W.cost_of_equity + d * 0.05 * 0.75)


def test_gordon_formula_and_guard():
    assert gordon_terminal_value(100, 0.08, 0.02) == pytest.approx(102 / 0.06)
    with pytest.raises(ValueError):
        gordon_terminal_value(100, 0.02, 0.03)


def test_enterprise_value_is_sum_of_parts_and_equity_bridge():
    v = dcf_value(C, W.wacc)
    assert v["ev"] == pytest.approx(v["pv_fcff"] + v["pv_terminal"])
    assert v["equity"] == pytest.approx(v["ev"] - C.net_debt - C.pension_deficit - C.minorities)


def test_mid_year_convention_adds_value():
    assert dcf_value(C, W.wacc, mid_year=True)["ev"] > dcf_value(C, W.wacc, mid_year=False)["ev"]


def test_fcff_definition():
    f = forecast(C).iloc[0]
    assert f["FCFF"] == pytest.approx(f["NOPAT"] + f["D&A"] - f["capex"] - f["change in NWC"])


def test_sensitivity_monotonic():
    s = sensitivity(C, [0.07, 0.08, 0.09], [0.01, 0.02, 0.03])
    assert np.all(np.diff(s.values, axis=0) < 0) and np.all(np.diff(s.values, axis=1) > 0)


def test_reverse_dcf_reprices_the_share():
    from dataclasses import replace
    g = reverse_dcf(C, W.wacc)
    k = g / np.mean(C.growth)
    implied = dcf_value(replace(C, growth=tuple(np.array(C.growth) * k)), W.wacc)["per_share"]
    assert implied == pytest.approx(C.share_price, abs=1e-6)
