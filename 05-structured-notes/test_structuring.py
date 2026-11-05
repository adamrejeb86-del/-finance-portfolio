import numpy as np
import pytest

from structuring import (MarketData, call, capital_protected_note, digital_put, put, put_down_and_in,
                         put_european_barrier, reverse_convertible)

M = MarketData()


def _mc_terminal(m, T, n=1_000_000, seed=0):
    z = np.random.default_rng(seed).standard_normal(n)
    return m.spot * np.exp((m.rate - m.div - 0.5 * m.vol**2) * T + m.vol * np.sqrt(T) * z)


def test_put_call_parity():
    T = 2.0
    assert call(M, 95, T) - put(M, 95, T) == pytest.approx(M.spot * np.exp(-M.div * T) - 95 * np.exp(-M.rate * T))


def test_european_barrier_put_matches_simulation():
    T, H = 1.0, 70.0
    ST = _mc_terminal(M, T)
    mc = np.exp(-M.rate * T) * np.where(ST < H, 100 - ST, 0).mean()
    assert put_european_barrier(M, 100, H, T) == pytest.approx(mc, abs=0.02)


def test_digital_put_matches_simulation():
    ST = _mc_terminal(M, 1.0)
    assert digital_put(M, 90, 1.0) == pytest.approx(np.exp(-M.rate) * (ST < 90).mean(), abs=2e-3)


def test_down_and_in_put_limits():
    assert put_down_and_in(M, 100, 100 - 1e-9, 1.0) == pytest.approx(put(M, 100, 1.0), abs=1e-6)   # barrier at strike
    assert put_down_and_in(M, 100, 1e-3, 1.0) == pytest.approx(0.0, abs=1e-10)                    # barrier unreachable


def test_american_barrier_costs_more_than_european():
    # A barrier watched every day is hit more often than one checked only at maturity
    assert put_down_and_in(M, 100, 70, 1.0) > put_european_barrier(M, 100, 70, 1.0)


def test_capital_protected_note_budget_identity():
    res = capital_protected_note(M, T=5, protection=1.0, fees=0.02)
    assert res["zero_coupon"] + res["participation"] * res["option_cost_per_100pct"] == pytest.approx(98.0)


def test_cap_increases_participation():
    assert capital_protected_note(M, cap=0.4)["participation"] > capital_protected_note(M)["participation"]


def test_reverse_convertible_budget_identity():
    res = reverse_convertible(M, T=1, barrier=0.7)
    assert 100 * res["coupon_pa"] * res["annuity"] + res["zero_coupon"] - res["put_sold"] == pytest.approx(99.0)


def test_lower_barrier_means_lower_coupon():
    coupons = [reverse_convertible(M, barrier=b)["coupon_pa"] for b in (0.9, 0.8, 0.7, 0.6)]
    assert all(a > b for a, b in zip(coupons, coupons[1:]))
