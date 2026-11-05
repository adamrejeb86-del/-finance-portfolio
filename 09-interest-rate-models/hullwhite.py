"""
Hull-White one-factor short-rate model: fit to today's curve, caps, swaptions, calibration.

    dr_t = (theta(t) - a r_t) dt + sigma dW_t

- theta(t) is chosen so that the model reproduces today's discount curve exactly.
- Zero-coupon bonds are affine in r_t:  P(t,T) = A(t,T) exp(-B(t,T) r_t).
- Options on zero-coupon bonds have Black-like closed forms, which give caplets (a caplet
  is a put on a zero-coupon bond) and, through Jamshidian's decomposition, European swaptions.
- Market caps are quoted in normal (Bachelier) volatilities, the standard since rates went
  negative; the model is calibrated to them by least squares.
- Exact Monte Carlo simulation of the short rate validates the closed forms.

Author: Adam Rejeb
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.stats import norm


# ---------------------------------------------------------------- today's curve
@dataclass(frozen=True)
class InitialCurve:
    """Smooth Nelson-Siegel zero curve used as 'today's market curve'."""
    level: float = 0.028
    slope: float = -0.008
    hump: float = 0.004
    tau: float = 2.5

    def zero(self, t):
        t = np.maximum(np.asarray(t, dtype=float), 1e-8)
        x = t / self.tau
        f = (1 - np.exp(-x)) / x
        return self.level + self.slope * f + self.hump * (f - np.exp(-x))

    def df(self, t):
        return np.exp(-self.zero(t) * np.asarray(t, dtype=float))

    def inst_forward(self, t, h=1e-5):
        t = np.asarray(t, dtype=float)
        return -(np.log(self.df(t + h)) - np.log(self.df(np.maximum(t - h, 0.0)))) / (t + h - np.maximum(t - h, 0.0))


# ---------------------------------------------------------------- Hull-White closed forms
@dataclass(frozen=True)
class HullWhite:
    a: float = 0.05        # mean-reversion speed
    sigma: float = 0.009   # normal volatility of the short rate (90 bp per year)
    curve: InitialCurve = InitialCurve()

    def B(self, t, T):
        return (1 - np.exp(-self.a * (T - t))) / self.a

    def zcb(self, t, T, r_t):
        """P(t, T) given the short rate r_t at time t."""
        c, B = self.curve, self.B(t, T)
        lnA = (np.log(c.df(T) / c.df(t)) + B * c.inst_forward(t)
               - self.sigma**2 / (4 * self.a) * (1 - np.exp(-2 * self.a * t)) * B**2)
        return np.exp(lnA - B * r_t)

    def _sigma_p(self, T, S):
        return self.sigma * self.B(T, S) * np.sqrt((1 - np.exp(-2 * self.a * T)) / (2 * self.a))

    def zcb_put(self, T, S, K):
        """Today's price of a put expiring at T on the zero-coupon bond maturing at S > T."""
        P_T, P_S = self.curve.df(T), self.curve.df(S)
        sp = self._sigma_p(T, S)
        h = np.log(P_S / (P_T * K)) / sp + sp / 2
        return K * P_T * norm.cdf(-h + sp) - P_S * norm.cdf(-h)

    def zcb_call(self, T, S, K):
        P_T, P_S = self.curve.df(T), self.curve.df(S)
        sp = self._sigma_p(T, S)
        h = np.log(P_S / (P_T * K)) / sp + sp / 2
        return P_S * norm.cdf(h) - K * P_T * norm.cdf(h - sp)

    def caplet(self, T, delta, K):
        """Caplet on the simple rate from T to T + delta, paid at T + delta (unit notional)."""
        return (1 + K * delta) * self.zcb_put(T, T + delta, 1 / (1 + K * delta))

    def cap(self, maturity, K, delta=0.5):
        """Cap = sum of caplets; the first period (already fixed) is excluded."""
        starts = np.arange(delta, maturity - 1e-9, delta)
        return float(sum(self.caplet(T, delta, K) for T in starts))

    def payer_swaption(self, expiry, tenor, K, delta=1.0):
        """European payer swaption by Jamshidian's decomposition.

        A payer swaption is a put on a coupon bond. Find r* such that the coupon bond is worth
        exactly 1 at expiry; the swaption is then a portfolio of zero-coupon puts struck at
        P(expiry, t_i; r*).
        """
        pay = expiry + delta * np.arange(1, int(round(tenor / delta)) + 1)
        c = np.full(len(pay), K * delta)
        c[-1] += 1
        r_star = brentq(lambda r: (c * self.zcb(expiry, pay, r)).sum() - 1, -1.0, 1.0, xtol=1e-14)
        strikes = self.zcb(expiry, pay, r_star)
        return float(sum(ci * self.zcb_put(expiry, ti, Ki) for ci, ti, Ki in zip(c, pay, strikes)))

    # ------------------------------------------------------------ simulation
    def simulate(self, T, n_steps, n_paths, seed=0):
        """Exact simulation: r_t = x_t + alpha(t), with x an Ornstein-Uhlenbeck process.
        Returns times, short rates (n_paths, n_steps + 1) and discount factors exp(-int r)."""
        dt = T / n_steps
        t = np.linspace(0, T, n_steps + 1)
        alpha = self.curve.inst_forward(t) + self.sigma**2 / (2 * self.a**2) * (1 - np.exp(-self.a * t)) ** 2
        rng = np.random.default_rng(seed)
        x = np.zeros((n_paths, n_steps + 1))
        decay, sd = np.exp(-self.a * dt), self.sigma * np.sqrt((1 - np.exp(-2 * self.a * dt)) / (2 * self.a))
        for i in range(n_steps):
            x[:, i + 1] = x[:, i] * decay + sd * rng.standard_normal(n_paths)
        r = x + alpha
        integral = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(0.5 * (r[:, 1:] + r[:, :-1]) * dt, axis=1)], axis=1)
        return t, r, np.exp(-integral)


# ---------------------------------------------------------------- market quotes in normal vol
def bachelier_caplet(F, K, T, delta, df_pay, normal_vol):
    d = (F - K) / (normal_vol * np.sqrt(T))
    return delta * df_pay * ((F - K) * norm.cdf(d) + normal_vol * np.sqrt(T) * norm.pdf(d))


def market_cap_price(curve: InitialCurve, maturity, K, normal_vol, delta=0.5):
    starts = np.arange(delta, maturity - 1e-9, delta)
    total = 0.0
    for T in starts:
        F = (curve.df(T) / curve.df(T + delta) - 1) / delta
        total += bachelier_caplet(F, K, T, delta, curve.df(T + delta), normal_vol)
    return float(total)


def atm_cap_strike(curve: InitialCurve, maturity, delta=0.5):
    """ATM strike = swap rate of the caplet schedule."""
    starts = np.arange(delta, maturity - 1e-9, delta)
    ann = sum(delta * curve.df(T + delta) for T in starts)
    return float((curve.df(starts[0]) - curve.df(starts[-1] + delta)) / ann)


def implied_cap_normal_vol(curve: InitialCurve, maturity, K, price, delta=0.5):
    return brentq(lambda v: market_cap_price(curve, maturity, K, v, delta) - price, 1e-5, 0.05, xtol=1e-12)


def bachelier_swaption(F, K, T, annuity, normal_vol):
    d = (F - K) / (normal_vol * np.sqrt(T))
    return annuity * ((F - K) * norm.cdf(d) + normal_vol * np.sqrt(T) * norm.pdf(d))


def implied_swaption_normal_vol(curve: InitialCurve, expiry, tenor, price, delta=1.0):
    pay = expiry + delta * np.arange(1, int(round(tenor / delta)) + 1)
    ann = float(delta * curve.df(pay).sum())
    F = float((curve.df(expiry) - curve.df(pay[-1])) / ann)
    return brentq(lambda v: bachelier_swaption(F, F, expiry, ann, v) - price, 1e-6, 0.05, xtol=1e-12), F


def calibrate(curve: InitialCurve, maturities, normal_vols, guess=(0.1, 0.01)):
    """Fit (a, sigma) to ATM cap prices quoted in normal volatility."""
    strikes = [atm_cap_strike(curve, T) for T in maturities]
    targets = np.array([market_cap_price(curve, T, K, v) for T, K, v in zip(maturities, strikes, normal_vols)])

    def residuals(x):
        hw = HullWhite(x[0], x[1], curve)
        return np.array([hw.cap(T, K) for T, K in zip(maturities, strikes)]) / targets - 1

    sol = least_squares(residuals, guess, bounds=([1e-3, 1e-4], [1.0, 0.05]), xtol=1e-12, ftol=1e-12)
    return HullWhite(sol.x[0], sol.x[1], curve), float(np.sqrt(np.mean(sol.fun**2)))
