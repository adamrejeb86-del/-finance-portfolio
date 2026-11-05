"""
Convertible bond pricer: Tsiveriotis-Fernandes binomial tree.

A convertible bond (CB) pays coupons like a bond and lets the holder exchange it for a
fixed number of shares (the conversion ratio). Its value mixes two kinds of cash flows
with different risks:
  - what will be received in shares behaves like equity: discounted at the risk-free rate;
  - what will be received in cash (coupons, redemption, put) is exposed to the issuer's
    default: discounted at the risk-free rate plus the credit spread.

Tsiveriotis and Fernandes (1998) track both parts on a binomial tree:
  V = total CB value,  B = cash-only part (COCB),  V - B = equity part.

Features: annual coupons, holder conversion at any time (American), issuer call
(hard call above a date, the holder may convert instead) and a holder put.

Author: Adam Rejeb
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Convertible:
    face: float = 100.0
    coupon_rate: float = 0.02          # annual coupon, in % of face
    maturity: float = 5.0
    conversion_ratio: float = 1.0      # shares received per bond (conversion price = face / ratio)
    call_price: float | None = None    # issuer call price (None = not callable)
    call_from: float = 0.0             # call protection: callable only after this time
    put_price: float | None = None     # holder put price
    put_time: float | None = None      # single put date


@dataclass(frozen=True)
class Market:
    spot: float = 80.0
    vol: float = 0.30
    rate: float = 0.03
    credit_spread: float = 0.02
    div: float = 0.01


def price(cb: Convertible, m: Market, steps_per_year: int = 200, full_output: bool = False):
    """Value of the convertible, its delta and gamma, from the first tree nodes."""
    n = int(round(cb.maturity * steps_per_year))
    dt = cb.maturity / n
    u = np.exp(m.vol * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((m.rate - m.div) * dt) - d) / (u - d)
    disc_r, disc_rc = np.exp(-m.rate * dt), np.exp(-(m.rate + m.credit_spread) * dt)
    coupon_steps = {int(round(k * steps_per_year)) for k in range(1, int(round(cb.maturity)) + 1)}
    coupon = cb.coupon_rate * cb.face
    put_step = int(round(cb.put_time * steps_per_year)) if cb.put_time is not None else None

    S = m.spot * u ** (n - 2 * np.arange(n + 1))
    conv = cb.conversion_ratio * S
    redemption = cb.face + coupon
    V = np.maximum(conv, redemption)
    B = np.where(conv >= redemption, 0.0, redemption)
    layers = {}

    for i in range(n - 1, -1, -1):
        S = m.spot * u ** (i - 2 * np.arange(i + 1))
        B_cont = disc_rc * (p * B[:-1] + (1 - p) * B[1:])
        E_cont = disc_r * (p * (V[:-1] - B[:-1]) + (1 - p) * (V[1:] - B[1:]))
        V, B = B_cont + E_cont, B_cont
        if i in coupon_steps and i != 0:
            V, B = V + coupon, B + coupon
        conv = cb.conversion_ratio * S
        t = i * dt

        if put_step is not None and i == put_step and cb.put_price is not None:
            put_better = cb.put_price > V
            V, B = np.where(put_better, cb.put_price, V), np.where(put_better, cb.put_price, B)
        if cb.call_price is not None and t >= cb.call_from:
            called = V > cb.call_price                           # issuer calls when the CB is worth more than the call price
            forced_conv = called & (conv >= cb.call_price)       # holder then converts if shares are worth more
            V = np.where(called, np.maximum(cb.call_price, conv), V)
            B = np.where(called & ~forced_conv, cb.call_price, np.where(forced_conv, 0.0, B))
        convert = conv > V                                       # voluntary conversion
        V, B = np.where(convert, conv, V), np.where(convert, 0.0, B)
        if i <= 2:
            layers[i] = (S.copy(), V.copy())

    value = float(V[0])
    (S1, V1), (S2, V2) = layers[1], layers[2]
    delta = (V1[0] - V1[1]) / (S1[0] - S1[1])
    gamma = ((V2[0] - V2[1]) / (S2[0] - S2[1]) - (V2[1] - V2[2]) / (S2[1] - S2[2])) / (0.5 * (S2[0] - S2[2]))
    out = {"price": value, "delta": delta, "gamma": gamma}
    if full_output:
        out.update(analytics(cb, m, value, delta))
    return out


def bond_floor(cb: Convertible, m: Market) -> float:
    """Value of the same bond without the conversion right (straight debt)."""
    times = np.arange(1, int(round(cb.maturity)) + 1, dtype=float)
    dfs = np.exp(-(m.rate + m.credit_spread) * times)
    return float(cb.coupon_rate * cb.face * dfs.sum() + cb.face * dfs[-1])


def analytics(cb: Convertible, m: Market, value: float, delta: float) -> dict:
    parity = cb.conversion_ratio * m.spot
    floor = bond_floor(cb, m)
    return {"parity": parity, "bond_floor": floor,
            "conversion_premium": value / parity - 1,      # how much more than the shares you pay
            "premium_over_floor": value / floor - 1,       # how much you pay for the option
            "equity_delta": delta * cb.conversion_ratio * m.spot / value}   # % change for a 1% stock move, per unit of value


def implied_credit_spread(cb: Convertible, m: Market, market_price: float, lo=0.0, hi=0.20) -> float:
    """Credit spread that makes the model price equal to a market price (bisection)."""
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if price(cb, replace(m, credit_spread=mid))["price"] > market_price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
