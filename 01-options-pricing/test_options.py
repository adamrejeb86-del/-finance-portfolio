import numpy as np
import pytest

from options import binomial_price, bs_greeks, bs_price, implied_vol

S, K, T, r, sigma, q = 100.0, 105.0, 0.75, 0.03, 0.25, 0.01


def test_put_call_parity():
    call = bs_price(S, K, T, r, sigma, q, "call")
    put = bs_price(S, K, T, r, sigma, q, "put")
    assert call - put == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T), abs=1e-12)


def test_known_value():
    # Textbook value (Hull): S=42, K=40, T=0.5, r=10%, sigma=20% -> call 4.76
    assert bs_price(42, 40, 0.5, 0.10, 0.20) == pytest.approx(4.7594, abs=1e-4)


@pytest.mark.parametrize("option", ["call", "put"])
def test_greeks_match_finite_differences(option):
    g = bs_greeks(S, K, T, r, sigma, q, option)
    h = 1e-4
    price = lambda **kw: bs_price(kw.get("S", S), K, kw.get("T", T), kw.get("r", r),
                                  kw.get("sigma", sigma), q, option)
    assert g["delta"] == pytest.approx((price(S=S + h) - price(S=S - h)) / (2 * h), rel=1e-6)
    assert g["gamma"] == pytest.approx((price(S=S + h) - 2 * price() + price(S=S - h)) / h**2, rel=1e-3)
    assert g["vega"] == pytest.approx((price(sigma=sigma + h) - price(sigma=sigma - h)) / (2 * h), rel=1e-6)
    assert g["rho"] == pytest.approx((price(r=r + h) - price(r=r - h)) / (2 * h), rel=1e-6)
    assert g["theta"] == pytest.approx(-(price(T=T + h) - price(T=T - h)) / (2 * h), rel=1e-5)


@pytest.mark.parametrize("strike", [60, 90, 100, 110, 160])
@pytest.mark.parametrize("option", ["call", "put"])
def test_implied_vol_round_trip(strike, option):
    price = bs_price(S, strike, T, r, 0.37, q, option)
    assert implied_vol(price, S, strike, T, r, q, option) == pytest.approx(0.37, abs=1e-7)


def test_binomial_converges_to_black_scholes():
    tree = binomial_price(S, K, T, r, sigma, q, "call", american=False, steps=2000)
    assert tree == pytest.approx(bs_price(S, K, T, r, sigma, q, "call"), abs=5e-3)


def test_american_put_worth_more_than_european():
    american = binomial_price(S, K, T, 0.08, sigma, 0.0, "put", american=True)
    european = bs_price(S, K, T, 0.08, sigma, 0.0, "put")
    assert american > european


def test_american_call_without_dividends_is_european():
    american = binomial_price(S, K, T, r, sigma, 0.0, "call", american=True, steps=1000)
    european = binomial_price(S, K, T, r, sigma, 0.0, "call", american=False, steps=1000)
    assert american == pytest.approx(european, abs=1e-10)
