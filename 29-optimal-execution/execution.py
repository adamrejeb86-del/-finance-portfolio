"""
Optimal execution of a large order: Almgren-Chriss (2000), TWAP and VWAP.

Selling X shares quickly moves the price against you (market impact); selling slowly leaves
you exposed to the price moving anyway (timing risk). Almgren and Chriss minimise
    E[implementation shortfall] + lambda x Var[implementation shortfall]
with
  - permanent impact: each share sold lowers the price by gamma (linear);
  - temporary impact: selling at rate v costs eta x v per share on that trade;
  - arithmetic Brownian price with volatility sigma.

The optimal holdings follow x(t) = X sinh(kappa (T - t)) / sinh(kappa T): front-loaded when the
trader is risk-averse, a straight line (TWAP) when lambda -> 0.

Author: Adam Rejeb
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Market:
    X: float = 1_000_000          # shares to sell
    S0: float = 50.0              # EUR
    sigma: float = 0.95           # EUR per share per sqrt(day)  (about 30% annual vol)
    T: float = 5.0                # days
    N: int = 50                   # trading intervals
    eta: float = 2.5e-6           # temporary impact: EUR per share per (share/day)
    gamma: float = 2.5e-7         # permanent impact: EUR per share per share
    epsilon: float = 0.0625       # fixed cost per share traded (half spread + fees)

    @property
    def tau(self):
        return self.T / self.N


def ac_trajectory(m: Market, lam: float) -> np.ndarray:
    """Holdings x_0 = X, ..., x_N = 0 on the trading grid."""
    t = np.arange(m.N + 1) * m.tau
    if lam <= 0:
        return m.X * (1 - t / m.T)
    eta_t = m.eta - 0.5 * m.gamma * m.tau
    kappa_tilde2 = lam * m.sigma**2 / eta_t
    kappa = np.arccosh(1 + 0.5 * kappa_tilde2 * m.tau**2) / m.tau
    return m.X * np.sinh(kappa * (m.T - t)) / np.sinh(kappa * m.T)


def expected_cost_variance(m: Market, x: np.ndarray):
    """Closed-form mean and variance of the implementation shortfall for a trajectory."""
    n = -np.diff(x)                                    # shares sold in each interval
    eta_t = m.eta - 0.5 * m.gamma * m.tau
    E = 0.5 * m.gamma * m.X**2 + m.epsilon * np.abs(n).sum() + eta_t / m.tau * (n**2).sum()
    V = m.sigma**2 * m.tau * (x[1:] ** 2).sum()
    return E, V


def vwap_trajectory(m: Market, intraday_profile=None) -> np.ndarray:
    """Trade in proportion to a U-shaped intraday volume curve, the same every day."""
    per_day = m.N // int(m.T)
    if intraday_profile is None:
        u = np.linspace(0, 1, per_day)
        intraday_profile = 1 + 2.5 * (u - 0.5) ** 2 * 4      # heavy at the open and the close
    w = np.tile(intraday_profile / intraday_profile.sum(), int(m.T)) / m.T
    sold = np.concatenate([[0], np.cumsum(w)]) * m.X
    return m.X - sold


def simulate_shortfall(m: Market, x: np.ndarray, n_paths=20_000, seed=0) -> np.ndarray:
    """Implementation shortfall (EUR) = X S0 - cash received, on simulated price paths."""
    rng = np.random.default_rng(seed)
    n = -np.diff(x)
    S = np.full(n_paths, m.S0)
    cash = np.zeros(n_paths)
    for k in range(m.N):
        # Almgren-Chriss timing: slice k is executed at the price prevailing at the start of the
        # interval minus the temporary impact; then the price moves and keeps the permanent impact
        exec_price = S - m.epsilon * np.sign(n[k]) - m.eta * n[k] / m.tau
        cash += n[k] * exec_price
        S = S + m.sigma * np.sqrt(m.tau) * rng.standard_normal(n_paths) - m.gamma * n[k]
    return m.X * m.S0 - cash


def efficient_frontier(m: Market, lambdas) -> list:
    out = []
    for lam in lambdas:
        E, V = expected_cost_variance(m, ac_trajectory(m, lam))
        out.append({"lambda": lam, "expected cost": E, "std": np.sqrt(V)})
    return out
