"""
Counterparty credit risk for interest rate swaps: exposure simulation, netting, collateral,
CVA / DVA and wrong-way risk.

Steps
  1. Simulate future interest rates with a Hull-White model fitted to today's curve
     (exact Ornstein-Uhlenbeck simulation of the factor).
  2. Revalue each swap on every path and date with the model's closed-form bond prices.
  3. Exposure profiles: expected exposure EE(t), expected negative exposure, PFE 97.5%.
  4. Netting set effects and a collateral agreement (threshold, minimum transfer amount,
     10-day margin period of risk).
  5. CVA = (1 - R) sum_i E[ D(t_i) V(t_i)^+ ] x P(default in (t_{i-1}, t_i]), and DVA symmetric.
  6. Wrong-way risk: let the counterparty's default intensity rise with the rate factor.

Author: Adam Rejeb
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Curve:
    level: float = 0.028
    slope: float = -0.008
    tau: float = 2.5

    def zero(self, t):
        t = np.maximum(np.asarray(t, dtype=float), 1e-8)
        x = t / self.tau
        return self.level + self.slope * (1 - np.exp(-x)) / x

    def df(self, t):
        return np.exp(-self.zero(t) * np.asarray(t, dtype=float))

    def fwd(self, t, h=1e-5):
        t = np.asarray(t, dtype=float)
        return -(np.log(self.df(t + h)) - np.log(self.df(np.maximum(t - h, 0)))) / (t + h - np.maximum(t - h, 0))


@dataclass(frozen=True)
class HullWhite:
    a: float = 0.05
    sigma: float = 0.009
    curve: Curve = Curve()

    def B(self, t, T):
        return (1 - np.exp(-self.a * (T - t))) / self.a

    def alpha(self, t):
        """Deterministic shift so that r_t = x_t + alpha(t) fits today's curve."""
        return self.curve.fwd(t) + self.sigma**2 / (2 * self.a**2) * (1 - np.exp(-self.a * t))**2

    def zcb_from_x(self, t, T, x):
        """P(t,T) given the factor x_t, broadcasting over paths (x) and maturities (T):
        P = A(t,T) exp(-B(t,T) r_t) with Hull's ln A = ln(P(0,T)/P(0,t)) + B f(0,t)
        - sigma^2/(4a) (1 - e^{-2at}) B^2."""
        c = self.curve
        B = self.B(t, T)
        lnA = np.log(c.df(T) / c.df(t)) + B * c.fwd(t) - self.sigma**2 / (4 * self.a) * (1 - np.exp(-2 * self.a * t)) * B**2
        return np.exp(lnA - B * (x + self.alpha(t)))

    def simulate(self, times, n_paths, seed=0):
        """Factor x and stochastic discount factors D(0,t) on the grid `times` (starting at 0)."""
        rng = np.random.default_rng(seed)
        x = np.zeros((n_paths, len(times)))
        integral = np.zeros((n_paths, len(times)))
        alpha = self.alpha(times)
        for i in range(1, len(times)):
            dt = times[i] - times[i - 1]
            decay = np.exp(-self.a * dt)
            sd = self.sigma * np.sqrt((1 - np.exp(-2 * self.a * dt)) / (2 * self.a))
            x[:, i] = x[:, i - 1] * decay + sd * rng.standard_normal(n_paths)
            r_prev, r_now = x[:, i - 1] + alpha[i - 1], x[:, i] + alpha[i]
            integral[:, i] = integral[:, i - 1] + 0.5 * (r_prev + r_now) * dt
        return x, np.exp(-integral)


@dataclass(frozen=True)
class Swap:
    notional: float
    fixed_rate: float
    maturity: float
    pay_fixed: bool = True       # True: we pay fixed (payer swap)
    freq: int = 1

    def pay_dates(self):
        return np.arange(1, int(round(self.maturity * self.freq)) + 1) / self.freq


def par_rate(curve: Curve, maturity, freq=1):
    pay = np.arange(1, int(round(maturity * freq)) + 1) / freq
    return float((1 - curve.df(maturity)) / (curve.df(pay).sum() / freq))


def swap_values(model: HullWhite, swap: Swap, times, x):
    """Mark-to-market of the swap on each path (rows) and date (columns), from our side.

    The floating leg is valued exactly: the coupon already fixed at the last reset is paid at
    the next payment date, and the remaining floating coupons are worth P(t, t_next) - P(t, T).
    The time grid must contain the reset dates.
    """
    pay = swap.pay_dates()
    delta = 1 / swap.freq
    resets = pay - delta
    fixing = np.full(x.shape[0], np.nan)
    out = np.zeros_like(x)
    for j, t in enumerate(times):
        xt = x[:, j:j + 1]
        if np.any(np.isclose(resets, t)):                                    # new floating rate fixed today
            fixing = (1 / model.zcb_from_x(t, t + delta, xt)[:, 0] - 1) / delta
        future = pay[pay > t + 1e-9]
        if len(future) == 0:
            continue
        P = model.zcb_from_x(t, future[None, :], xt)                          # (paths, payments)
        float_leg = P[:, 0] * (1 + fixing * delta) - P[:, -1]
        fixed_leg = swap.fixed_rate * delta * P.sum(axis=1)
        value = swap.notional * (float_leg - fixed_leg)
        out[:, j] = value if swap.pay_fixed else -value
    return out


def exposure_profile(values, quantile=0.975):
    pos = np.maximum(values, 0)
    return {"EE": pos.mean(axis=0), "ENE": np.minimum(values, 0).mean(axis=0),
            "PFE": np.quantile(values, quantile, axis=0), "EPE": pos.mean()}


def collateralised(values, times, threshold=0.0, mta=0.0, mpor=10 / 252):
    """Exposure under a CSA: collateral reflects the value one margin period of risk earlier
    (the time to detect a default and close out), net of the threshold."""
    lag = np.searchsorted(times, times - mpor, side="left")
    lagged = values[:, np.clip(lag, 0, None)]
    collateral = np.maximum(lagged - threshold, 0.0)
    collateral = np.where(collateral < mta, 0.0, collateral)
    return values - collateral


def cva(values, disc, times, hazard, recovery=0.4, wwr_beta=0.0, x=None):
    """Unilateral CVA. With wwr_beta != 0, the counterparty's intensity on each path is
    hazard x exp(beta z_t - beta^2/2), z_t the standardised rate factor (mean preserved):
    beta > 0 means it defaults more often when rates are high (wrong-way for a payer swap)."""
    dt = np.diff(times)
    if wwr_beta and x is not None:
        z = x / np.maximum(x.std(axis=0, keepdims=True), 1e-12)
        lam = hazard * np.exp(wwr_beta * z - 0.5 * wwr_beta**2)
    else:
        lam = np.full_like(values, hazard)
    cum = np.concatenate([np.zeros((values.shape[0], 1)), np.cumsum(0.5 * (lam[:, 1:] + lam[:, :-1]) * dt, axis=1)], axis=1)
    surv = np.exp(-cum)
    default_prob = surv[:, :-1] - surv[:, 1:]
    disc_exposure = disc[:, 1:] * np.maximum(values[:, 1:], 0)
    return float((1 - recovery) * (disc_exposure * default_prob).sum(axis=1).mean())


def dva(values, disc, times, own_hazard, recovery=0.4):
    return cva(-values, disc, times, own_hazard, recovery)


def cva_running_spread(cva_value, curve: Curve, maturity, notional, hazard):
    """CVA converted into a running spread (bp per year) on the swap notional, using the
    counterparty's risky annuity."""
    pay = np.arange(1, int(maturity) + 1)
    risky_annuity = (curve.df(pay) * np.exp(-hazard * pay)).sum()
    return cva_value / (notional * risky_annuity) * 1e4
