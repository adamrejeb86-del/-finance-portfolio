from dataclasses import replace

import numpy as np
import pytest

from autocall import Market, Phoenix, fair_coupon, price

MKT = Market()
N = 60_000


def test_zero_coupon_limit():
    """No coupon, never called, capital always protected: the note is a zero-coupon bond."""
    note = Phoenix(coupon=0.0, autocall_barrier=1e9, protection_barrier=0.0)
    res = price(note, MKT, N)
    assert res["price"] == pytest.approx(100 * np.exp(-MKT.r * note.maturity), abs=1e-10)
    assert res["stderr"] == pytest.approx(0.0, abs=1e-12)


def test_fixed_coupon_bond_limit():
    """Coupon barrier at zero and no autocall: a fixed-coupon bond, priced exactly."""
    note = Phoenix(coupon_barrier=0.0, autocall_barrier=1e9, protection_barrier=0.0)
    dfs = np.exp(-MKT.r * note.times)
    expected = 100 * note.coupon * dfs.sum() + 100 * dfs[-1]
    assert price(note, MKT, N)["price"] == pytest.approx(expected, abs=1e-10)


def test_worst_of_is_long_correlation():
    low = price(Phoenix(), replace(MKT, corr=0.2), N)["price"]
    high = price(Phoenix(), replace(MKT, corr=0.9), N)["price"]
    assert high > low


def test_investor_is_short_volatility():
    low = price(Phoenix(), replace(MKT, vols=(0.15, 0.20)), N)["price"]
    high = price(Phoenix(), replace(MKT, vols=(0.35, 0.40)), N)["price"]
    assert low > high


def test_memory_feature_adds_value():
    assert price(Phoenix(memory=True), MKT, N)["price"] > price(Phoenix(memory=False), MKT, N)["price"]


def test_fair_coupon_reprices_to_target():
    c = fair_coupon(Phoenix(), MKT, target=98.0, n_paths=N)
    assert price(Phoenix(coupon=c), MKT, N)["price"] == pytest.approx(98.0, abs=1e-8)


def test_probabilities_are_consistent():
    res = price(Phoenix(), MKT, N)
    assert res["call_prob"].sum() + res["prob_maturity"] == pytest.approx(1.0, abs=1e-12)
    assert 0 <= res["prob_loss"] <= res["prob_maturity"]


def test_antithetic_reduces_error():
    plain = price(Phoenix(), MKT, N, antithetic=False)["stderr"]
    anti = price(Phoenix(), MKT, N, antithetic=True)["stderr"]
    assert anti < plain
