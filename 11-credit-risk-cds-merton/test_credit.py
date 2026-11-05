import numpy as np
import pytest

from credit import (HazardCurve, bootstrap, cs01, merton_equity, merton_solve, merton_spread_curve, par_spread,
                    upfront)

MATS = [1, 2, 3, 5, 7, 10]
SPREADS = np.array([60, 75, 90, 120, 135, 150]) * 1e-4


def test_bootstrap_reprices_every_cds():
    c = bootstrap(MATS, SPREADS)
    for T, s in zip(MATS, SPREADS):
        assert par_spread(c, T) == pytest.approx(s, abs=1e-12)


def test_credit_triangle():
    """With a flat hazard rate, spread ~ lambda x (1 - R)."""
    c = HazardCurve(np.array([10.0]), np.array([0.02]))
    assert par_spread(c, 5) == pytest.approx(0.02 * 0.6, rel=0.01)


def test_survival_is_decreasing_and_consistent():
    c = bootstrap(MATS, SPREADS)
    t = np.linspace(0, 10, 101)
    s = c.survival(t)
    assert s[0] == 1.0 and np.all(np.diff(s) < 0)


def test_upfront_is_zero_when_coupon_equals_par_spread():
    c = bootstrap(MATS, SPREADS)
    assert upfront(c, 5, coupon=par_spread(c, 5)) == pytest.approx(0.0, abs=1e-12)


def test_protection_buyer_gains_when_spreads_widen():
    assert cs01(MATS, SPREADS, 5) > 0


def test_merton_solution_is_consistent():
    res = merton_solve(E=40.0, sigma_E=0.45, D=80.0)
    eq, d1, _ = merton_equity(res["asset_value"], res["asset_vol"], 80.0, 1.0, 0.03)
    assert eq == pytest.approx(40.0, abs=1e-8)
    assert res["credit_spread"] > 0 and 0 < res["pd_risk_neutral"] < 1


def test_merton_spreads_rise_with_leverage():
    low = merton_spread_curve(100, 0.25, 50, [5])[0]
    high = merton_spread_curve(100, 0.25, 80, [5])[0]
    assert high > low > 0
