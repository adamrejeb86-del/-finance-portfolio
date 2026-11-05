"""
Structuring toolkit: build retail structured notes from a zero-coupon bond and options,
and solve for the headline term (participation or coupon) the issuer can offer.

Every note is priced as the sum of its building blocks:
  - a zero-coupon bond discounted at the risk-free rate plus the issuer's funding spread
    (a bank borrows through the note, so its credit spread is part of the budget);
  - vanilla, barrier or digital options priced with Black-Scholes formulas.

Notes covered
  - Capital-protected note: protection level + participation in the upside (optionally capped)
  - Reverse convertible: high fixed coupon, financed by a put sold by the investor;
    the put can be plain, with a barrier observed at maturity only (European) or
    continuously (American).

Author: Adam Rejeb
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class MarketData:
    spot: float = 100.0
    rate: float = 0.03            # risk-free rate (continuous)
    funding_spread: float = 0.01  # issuer credit spread
    vol: float = 0.20
    div: float = 0.02

    def zc_issuer(self, T):
        """Price of 1 paid by the issuer at T (risky discount factor)."""
        return np.exp(-(self.rate + self.funding_spread) * T)


# ---------------------------------------------------------------- option building blocks
def _d1(S, K, T, r, q, v):
    return (np.log(S / K) + (r - q + 0.5 * v**2) * T) / (v * np.sqrt(T))


def call(m: MarketData, K, T, vol=None):
    v = m.vol if vol is None else vol
    d1 = _d1(m.spot, K, T, m.rate, m.div, v)
    return m.spot * np.exp(-m.div * T) * norm.cdf(d1) - K * np.exp(-m.rate * T) * norm.cdf(d1 - v * np.sqrt(T))


def put(m: MarketData, K, T, vol=None):
    v = m.vol if vol is None else vol
    d1 = _d1(m.spot, K, T, m.rate, m.div, v)
    return K * np.exp(-m.rate * T) * norm.cdf(-d1 + v * np.sqrt(T)) - m.spot * np.exp(-m.div * T) * norm.cdf(-d1)


def digital_put(m: MarketData, K, T):
    """Pays 1 if S_T < K."""
    d2 = _d1(m.spot, K, T, m.rate, m.div, m.vol) - m.vol * np.sqrt(T)
    return np.exp(-m.rate * T) * norm.cdf(-d2)


def put_european_barrier(m: MarketData, K, H, T):
    """Put struck at K that only pays if S_T < H (barrier observed at maturity only), H <= K.
    Payoff (K - S_T) 1{S_T < H} = (H - S_T)^+ + (K - H) 1{S_T < H}."""
    return put(m, H, T) + (K - H) * digital_put(m, H, T)


def put_down_and_in(m: MarketData, K, H, T):
    """Continuously monitored down-and-in put, H < K (Hull, Options, Futures and Other Derivatives)."""
    S, r, q, v = m.spot, m.rate, m.div, m.vol
    sq = v * np.sqrt(T)
    lam = (r - q + 0.5 * v**2) / v**2
    x1 = np.log(S / H) / sq + lam * sq
    y1 = np.log(H / S) / sq + lam * sq
    y = np.log(H**2 / (S * K)) / sq + lam * sq
    return (-S * np.exp(-q * T) * norm.cdf(-x1) + K * np.exp(-r * T) * norm.cdf(-x1 + sq)
            + S * np.exp(-q * T) * (H / S) ** (2 * lam) * (norm.cdf(y) - norm.cdf(y1))
            - K * np.exp(-r * T) * (H / S) ** (2 * lam - 2) * (norm.cdf(y - sq) - norm.cdf(y1 - sq)))


# ---------------------------------------------------------------- capital-protected note
def capital_protected_note(m: MarketData, T=5.0, protection=1.0, fees=0.02, cap=None):
    """Solve for the participation rate.

    Investor pays 100. At maturity receives 100 x [protection + participation x perf^+],
    where perf = S_T/S_0 - 1, capped at `cap` if given (the issuer then buys a call spread).
    Budget: 100 x (1 - fees) = 100 x protection x ZC + participation x option cost.
    """
    zc = m.zc_issuer(T)
    option = call(m, m.spot, T) - (call(m, m.spot * (1 + cap), T) if cap is not None else 0.0)
    option_per_100 = option / m.spot * 100
    budget = 100 * (1 - fees) - 100 * protection * zc
    return {"zero_coupon": 100 * protection * zc, "option_budget": budget,
            "option_cost_per_100pct": option_per_100, "participation": budget / option_per_100}


# ---------------------------------------------------------------- reverse convertible
def reverse_convertible(m: MarketData, T=1.0, barrier=None, barrier_type="european", fees=0.01,
                        coupon_freq=1):
    """Solve for the annual coupon of a reverse convertible.

    The investor receives fixed coupons and 100 at maturity, unless the stock ends below
    its initial level (plain RC) or below the barrier (barrier RC), in which case they
    receive 100 x S_T/S_0. That loss is a put sold to the issuer.
    Budget: 100 x (1 - fees) = coupons + 100 x ZC - put value.
    """
    K = m.spot
    if barrier is None:
        short_put = put(m, K, T)
    elif barrier_type == "european":
        short_put = put_european_barrier(m, K, barrier * m.spot, T)
    elif barrier_type == "american":
        short_put = put_down_and_in(m, K, barrier * m.spot, T)
    else:
        raise ValueError("barrier_type must be 'european' or 'american'")
    put_per_100 = short_put / m.spot * 100
    pay_times = np.arange(1, int(round(T * coupon_freq)) + 1) / coupon_freq
    annuity = m.zc_issuer(pay_times).sum() / coupon_freq      # value of 1% p.a. paid on the schedule
    zc = 100 * m.zc_issuer(T)
    coupon = (100 * (1 - fees) - zc + put_per_100) / (100 * annuity)
    return {"zero_coupon": zc, "put_sold": put_per_100, "annuity": annuity, "coupon_pa": coupon}


def payoff_capital_protected(perf, protection, participation, cap=None):
    upside = np.maximum(perf, 0.0) if cap is None else np.clip(perf, 0.0, cap)
    return 100 * (protection + participation * upside)


def payoff_reverse_convertible(final_level, coupon_total, barrier=None):
    """Final_level = S_T / S_0. Barrier observed at maturity (or path min for American)."""
    threshold = 1.0 if barrier is None else barrier
    capital = np.where(final_level >= threshold, 1.0, final_level)
    return 100 * (capital + coupon_total)
