from dataclasses import replace

import pytest

from convertible import Convertible, Market, bond_floor, implied_credit_spread, price

CB, M = Convertible(), Market()


def test_no_conversion_right_gives_straight_bond():
    cb = replace(CB, conversion_ratio=1e-9)
    assert price(cb, M)["price"] == pytest.approx(bond_floor(cb, M), abs=1e-6)


def test_value_above_bond_floor_and_parity():
    for spot in (30, 80, 150, 300):
        m = replace(M, spot=spot)
        v = price(CB, m)["price"]
        assert v >= bond_floor(CB, m) - 1e-9 and v >= CB.conversion_ratio * spot - 1e-9


def test_deep_in_the_money_trades_at_parity():
    m = replace(M, spot=400)
    assert price(CB, m)["price"] == pytest.approx(400, rel=0.01)
    assert price(CB, m)["delta"] == pytest.approx(1.0, abs=0.02)


def test_issuer_call_reduces_value():
    callable_cb = replace(CB, call_price=110, call_from=2.0)
    assert price(callable_cb, M)["price"] < price(CB, M)["price"]


def test_holder_put_increases_value():
    puttable = replace(CB, put_price=100, put_time=3.0)
    assert price(puttable, M)["price"] > price(CB, M)["price"]


def test_wider_credit_spread_lowers_value():
    assert price(CB, replace(M, credit_spread=0.06))["price"] < price(CB, M)["price"]


def test_delta_between_zero_and_conversion_ratio():
    d = price(CB, M)["delta"]
    assert 0 < d < CB.conversion_ratio


def test_implied_credit_spread_round_trip():
    target = price(CB, replace(M, credit_spread=0.035))["price"]
    assert implied_credit_spread(CB, M, target) == pytest.approx(0.035, abs=1e-6)
