"""
FX options the way the interbank market quotes them.

- Garman-Kohlhagen pricing (Black-Scholes with a foreign interest rate) and forwards from
  covered interest parity.
- Delta conventions: spot vs forward delta, and premium-adjusted delta (used when the premium
  is paid in the foreign currency, e.g. USD/JPY).
- Market quotes are not strikes but deltas: an at-the-money delta-neutral straddle (ATM DNS),
  a 25-delta risk reversal (RR, the skew) and a 25-delta butterfly (BF, the smile curvature).
- From those three quotes, the vanna-volga method (Castagna and Mercurio, 2007) builds the
  whole smile: it adds to the Black-Scholes price the cost of hedging vega, vanna and volga
  with the three liquid instruments.

Author: Adam Rejeb
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def forward(S, T, rd, rf):
    return S * np.exp((rd - rf) * T)


def gk_price(S, K, T, rd, rf, vol, call=True):
    F = forward(S, T, rd, rf)
    d1 = (np.log(F / K) + 0.5 * vol**2 * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    if call:
        return np.exp(-rd * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return np.exp(-rd * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def delta(S, K, T, rd, rf, vol, call=True, kind="spot", premium_adjusted=False):
    F = forward(S, T, rd, rf)
    d1 = (np.log(F / K) + 0.5 * vol**2 * T) / (vol * np.sqrt(T))
    w = 1 if call else -1
    fwd_delta = w * norm.cdf(w * d1)
    d = fwd_delta * (np.exp(-rf * T) if kind == "spot" else 1.0)
    if premium_adjusted:
        prem = gk_price(S, K, T, rd, rf, vol, call)
        d = d - prem / S if kind == "spot" else d - prem / (S * np.exp(-rf * T))
    return d


def strike_from_delta(target, S, T, rd, rf, vol, call=True, kind="spot", premium_adjusted=False):
    if not premium_adjusted:
        F = forward(S, T, rd, rf)
        w = 1 if call else -1
        fwd = target / (np.exp(-rf * T) if kind == "spot" else 1.0)
        return F * np.exp(-w * norm.ppf(w * fwd) * vol * np.sqrt(T) + 0.5 * vol**2 * T)
    f = lambda K: delta(S, K, T, rd, rf, vol, call, kind, True) - target
    F = forward(S, T, rd, rf)
    lo, hi = F * np.exp(-8 * vol * np.sqrt(T)), F * np.exp(8 * vol * np.sqrt(T))
    if call:
        # premium-adjusted call delta is not monotonic at low strikes: search right of its maximum
        grid = np.linspace(lo, hi, 2000)
        lo = grid[np.argmax([delta(S, k, T, rd, rf, vol, True, kind, True) for k in grid])]
    return brentq(f, lo, hi, xtol=1e-12)


def atm_dns_strike(S, T, rd, rf, vol, premium_adjusted=False):
    """Strike where a call and a put have opposite deltas (delta-neutral straddle)."""
    F = forward(S, T, rd, rf)
    return F * np.exp((-0.5 if premium_adjusted else 0.5) * vol**2 * T)


def pillar_vols(atm, rr25, bf25):
    """Smile-strangle approximation: vol of the 25-delta call and put from the quotes."""
    return {"25P": atm + bf25 - 0.5 * rr25, "ATM": atm, "25C": atm + bf25 + 0.5 * rr25}


def vanna_volga_vol(K, pillars_K, pillars_vol):
    """First-order vanna-volga implied vol (Castagna-Mercurio closed form). Exact at the pillars."""
    K1, K2, K3 = pillars_K
    s1, s2, s3 = pillars_vol
    L = np.log
    y1 = L(K2 / K) * L(K3 / K) / (L(K2 / K1) * L(K3 / K1))
    y2 = L(K / K1) * L(K3 / K) / (L(K2 / K1) * L(K3 / K2))
    y3 = L(K / K1) * L(K / K2) / (L(K3 / K1) * L(K3 / K2))
    return y1 * s1 + y2 * s2 + y3 * s3


def _vega_vanna_volga(S, K, T, rd, rf, vol):
    F = forward(S, T, rd, rf)
    d1 = (np.log(F / K) + 0.5 * vol**2 * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    vega = S * np.exp(-rf * T) * norm.pdf(d1) * np.sqrt(T)
    return vega, -vega * d2 / (S * vol * np.sqrt(T)), vega * d1 * d2 / vol


def vanna_volga_price(S, K, T, rd, rf, pillars_K, pillars_vol, call=True):
    """Price = BS(ATM vol) + sum_i x_i (market price_i - BS price_i at ATM vol), with weights
    x_i chosen so that the three pillar options hedge the option's vega, vanna and volga."""
    atm = pillars_vol[1]
    A = np.array([_vega_vanna_volga(S, Ki, T, rd, rf, atm) for Ki in pillars_K]).T
    b = np.array(_vega_vanna_volga(S, K, T, rd, rf, atm))
    x = np.linalg.solve(A, b)
    cost = [gk_price(S, Ki, T, rd, rf, si, True) - gk_price(S, Ki, T, rd, rf, atm, True) for Ki, si in zip(pillars_K, pillars_vol)]
    return gk_price(S, K, T, rd, rf, atm, call) + float(x @ np.array(cost))


def implied_vol(price, S, K, T, rd, rf, call=True):
    return brentq(lambda v: gk_price(S, K, T, rd, rf, v, call) - price, 1e-4, 3.0, xtol=1e-12)
