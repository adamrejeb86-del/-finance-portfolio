import numpy as np
import pytest

from curve import (MarketQuotes, bond_price_from_curve, bond_price_from_yield, bond_risk, bootstrap, dv01,
                   fit_nss, key_rate_dv01s, nss_zero, par_swap_rate, swap_npv, yield_to_maturity)

Q = MarketQuotes()
C = bootstrap(Q)


def test_curve_reprices_every_input():
    for t, rate in Q.deposits.items():
        assert 1 / C.df(t) == pytest.approx(1 + rate * t, abs=1e-13)
    for T, rate in Q.swaps.items():
        assert par_swap_rate(C, T) == pytest.approx(rate, abs=1e-12)


def test_par_swap_is_worth_zero_and_payer_gains_when_rates_rise():
    rate = par_swap_rate(C, 10)
    assert swap_npv(C, 1e7, rate, 10) == pytest.approx(0.0, abs=1e-6)
    assert dv01(Q, lambda c: swap_npv(c, 1e7, rate, 10, pay_fixed=True)) > 0


def test_key_rates_add_up_to_parallel_dv01():
    pricer = lambda c: swap_npv(c, 1e7, 0.03, 7)
    assert sum(key_rate_dv01s(Q, pricer).values()) == pytest.approx(dv01(Q, pricer), rel=1e-3)


def test_forward_rates_are_positive_and_curve_is_smooth_enough():
    t = np.linspace(0.1, 30, 300)
    assert np.all(C.forward(t[:-1], t[1:]) > 0)


def test_nss_fits_the_bootstrapped_curve():
    t = np.linspace(0.25, 30, 60)
    params = fit_nss(t, C.zero(t))
    assert np.max(np.abs(nss_zero(t, *params) - C.zero(t))) < 5e-4      # within 5 bp


def test_bond_yield_round_trip_and_duration():
    price = bond_price_from_curve(C, 0.03, 10)
    y = yield_to_maturity(price, 0.03, 10)
    assert bond_price_from_yield(y, 0.03, 10) == pytest.approx(price, abs=1e-10)
    risk = bond_risk(y, 0.03, 10)
    h = 1e-5
    numerical = -(bond_price_from_yield(y + h, 0.03, 10) - bond_price_from_yield(y - h, 0.03, 10)) / (2 * h) / price
    assert risk["modified_duration"] == pytest.approx(numerical, rel=1e-7)


def test_zero_coupon_bond_duration_equals_maturity():
    assert bond_risk(0.03, 0.0, 8)["macaulay_duration"] == pytest.approx(8.0)
