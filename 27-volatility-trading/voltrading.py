"""
Volatility trading: variance swaps and the volatility risk premium.

1. Variance swap fair strike by static replication (Demeterfi, Derman, Kamal, Zou, 1999):
       K_var = (2 e^{rT} / T) [ sum_{K<F} P(K) dK/K^2 + sum_{K>F} C(K) dK/K^2 ]
   a strip of out-of-the-money options weighted by 1/K^2. With a skew, K_var is above the
   at-the-money variance: the swap pays for the whole smile, not just the ATM point.
2. Realised variance and the P&L of a variance swap in vega and variance notional.
3. The volatility risk premium: implied volatility is usually above the volatility that is
   later realised. A strategy that sells one-month variance every month earns that premium
   most of the time and loses heavily in volatility spikes, which the simulation shows.

Author: Adam Rejeb
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def bs_price(S, K, T, r, vol, call=True):
    d1 = (np.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    if call:
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def skew_smile(atm_vol, skew=-0.10, curvature=0.30):
    """Implied vol as a function of log-moneyness k = ln(K/F): atm + skew k + curvature k^2."""
    return lambda k: np.maximum(atm_vol + skew * k + curvature * k**2, 0.03)


def variance_swap_strike(S, T, r, smile, n_strikes=400, width=4.0):
    """Fair variance (annualised) by replication on a fine strike grid around the forward."""
    F = S * np.exp(r * T)
    sd = smile(0.0) * np.sqrt(T)
    strikes = F * np.exp(np.linspace(-width * sd, width * sd, n_strikes))
    dK = np.gradient(strikes)
    vols = smile(np.log(strikes / F))
    otm = np.where(strikes < F, bs_price(S, strikes, T, r, vols, call=False), bs_price(S, strikes, T, r, vols, call=True))
    return float(2 * np.exp(r * T) / T * np.sum(otm * dK / strikes**2))


def realised_variance(prices, periods_per_year=252):
    """Annualised realised variance with the zero-mean convention used in term sheets."""
    r = np.diff(np.log(np.asarray(prices, dtype=float)))
    return float(periods_per_year * np.mean(r**2))


def variance_swap_pnl(realised_var, strike_var, vega_notional):
    """P&L of a long variance swap, market convention with vols in points (20% -> 20):
    variance notional = vega notional / (2 x strike vol), P&L = variance notional x (sigma_R^2 - K^2)."""
    var_notional = vega_notional / (2 * np.sqrt(strike_var) * 100)
    return var_notional * (realised_var - strike_var) * 1e4


# ---------------------------------------------------------------- volatility risk premium
def simulate_market(n_years=15, seed=0, premium=0.035):
    """Daily index returns with stochastic volatility and occasional volatility spikes, plus
    the one-month implied volatility quoted each day (expected vol + a risk premium)."""
    rng = np.random.default_rng(seed)
    n = 252 * n_years
    v, theta, kappa, xi = 0.03, 0.03, 4.0, 0.5           # variance process (vol about 17%)
    rets, vars_, implied = np.empty(n), np.empty(n), np.empty(n)
    for t in range(n):
        if rng.random() < 1 / 500:                        # roughly one stress event every two years
            v = v + rng.uniform(0.05, 0.15)
        vars_[t] = v
        # expected average variance over the next month, then add the premium in vol points
        exp_var = theta + (v - theta) * (1 - np.exp(-kappa / 12)) / (kappa / 12)
        implied[t] = np.sqrt(exp_var) + premium
        z = rng.standard_normal()
        rets[t] = -0.5 * v / 252 + np.sqrt(v / 252) * z
        v = max(v + kappa * (theta - v) / 252 + xi * np.sqrt(v / 252) * (-0.7 * z + np.sqrt(1 - 0.49) * rng.standard_normal()), 0.004)
    prices = 100 * np.exp(np.cumsum(rets))
    return prices, np.sqrt(vars_), implied


def short_variance_strategy(prices, implied, vega_notional=100_000, days=21):
    """Sell a one-month variance swap at the implied vol every month; P&L per month."""
    out = []
    for start in range(0, len(prices) - days - 1, days):
        k_vol = implied[start]
        rv = realised_variance(prices[start:start + days + 1])
        out.append({"start": start, "strike_vol": k_vol, "realised_vol": np.sqrt(rv),
                    "pnl": -variance_swap_pnl(rv, k_vol**2, vega_notional)})
    return out


def short_straddle_hedged(prices, implied, r=0.0, days=21, notional_contracts=1000):
    """Sell a one-month ATM straddle at the implied vol each month and delta-hedge it daily."""
    out = []
    for start in range(0, len(prices) - days - 1, days):
        S0, vol, T = prices[start], implied[start], days / 252
        premium = bs_price(S0, S0, T, r, vol, True) + bs_price(S0, S0, T, r, vol, False)
        cash, hedge = premium, 0.0
        for d in range(days):
            S, tau = prices[start + d], T - d / 252
            d1 = (np.log(S / S0) + (r + 0.5 * vol**2) * tau) / (vol * np.sqrt(tau))
            shares = 2 * norm.cdf(d1) - 1                    # offsets the short straddle's delta of -(2N(d1) - 1)
            cash -= (shares - hedge) * S
            hedge = shares
        ST = prices[start + days]
        pnl = cash + hedge * ST - abs(ST - S0)
        out.append({"start": start, "pnl": pnl * notional_contracts, "premium": premium * notional_contracts})
    return out
