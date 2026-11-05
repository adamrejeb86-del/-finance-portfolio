"""
Market making with inventory risk: the Avellaneda-Stoikov (2008) model.

A market maker quotes a bid and an ask around the mid price and earns the spread when
clients hit them. The risk is inventory: after buying from several sellers in a row, the
dealer is long and exposed to the price falling.

Avellaneda and Stoikov solve the dealer's problem with exponential utility (risk aversion
gamma) when the mid follows a Brownian motion (volatility sigma) and clients arrive as
Poisson processes whose intensity falls with the distance to the mid, lambda(delta) = A e^{-k delta}:

  reservation price   r = s - q gamma sigma^2 (T - t)
  optimal spread      delta_a + delta_b = gamma sigma^2 (T - t) + (2 / gamma) ln(1 + gamma / k)

The dealer skews both quotes against the inventory q: long inventory -> lower quotes, more
likely to sell. A symmetric strategy with the same spread, quoted around the mid, is the
benchmark.

Author: Adam Rejeb
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Params:
    s0: float = 100.0
    T: float = 1.0
    sigma: float = 2.0
    dt: float = 0.005
    A: float = 140.0
    k: float = 1.5
    gamma: float = 0.1
    max_inventory: int | None = None      # optional hard limit


def reservation_price(s, q, t, p: Params):
    return s - q * p.gamma * p.sigma**2 * (p.T - t)


def optimal_spread(t, p: Params):
    return p.gamma * p.sigma**2 * (p.T - t) + (2 / p.gamma) * np.log(1 + p.gamma / p.k)


def simulate(p: Params, strategy="inventory", n_paths=1000, seed=0, record=False):
    """Simulate the dealer on n_paths independent days.

    strategy: 'inventory' (Avellaneda-Stoikov quotes around the reservation price) or
              'symmetric' (same spread, centred on the mid).
    Returns final P&L (cash + inventory marked at the mid), final inventory, and optionally
    one recorded path.
    """
    rng = np.random.default_rng(seed)
    n_steps = int(round(p.T / p.dt))
    s = np.full(n_paths, p.s0)
    q = np.zeros(n_paths)
    cash = np.zeros(n_paths)
    n_trades = np.zeros(n_paths)
    rec = {k: [] for k in ("t", "mid", "bid", "ask", "inventory", "pnl")} if record else None
    for i in range(n_steps):
        t = i * p.dt
        spread = optimal_spread(t, p)
        centre = reservation_price(s, q, t, p) if strategy == "inventory" else s
        bid, ask = centre - spread / 2, centre + spread / 2
        delta_b, delta_a = np.maximum(s - bid, 0), np.maximum(ask - s, 0)
        hit_bid = rng.random(n_paths) < p.A * np.exp(-p.k * delta_b) * p.dt     # a client sells to us
        lift_ask = rng.random(n_paths) < p.A * np.exp(-p.k * delta_a) * p.dt    # a client buys from us
        if p.max_inventory is not None:
            hit_bid &= q < p.max_inventory
            lift_ask &= q > -p.max_inventory
        q += hit_bid.astype(float) - lift_ask.astype(float)
        cash += np.where(lift_ask, ask, 0.0) - np.where(hit_bid, bid, 0.0)
        n_trades += hit_bid.astype(int) + lift_ask.astype(int)     # (bool + bool would be a logical OR)
        if record:
            for k, v in zip(rec, (t, s[0], bid[0], ask[0], q[0], cash[0] + q[0] * s[0])):
                rec[k].append(v)
        s = s + p.sigma * np.sqrt(p.dt) * np.where(rng.random(n_paths) < 0.5, 1.0, -1.0)   # binomial mid, as in the paper
    pnl = cash + q * s
    return {"pnl": pnl, "inventory": q, "trades": n_trades, "path": rec}


def summary(res) -> dict:
    pnl, q = res["pnl"], res["inventory"]
    return {"Mean P&L": pnl.mean(), "Std P&L": pnl.std(), "Sharpe (per day)": pnl.mean() / pnl.std(),
            "Mean |final inventory|": np.abs(q).mean(), "Std final inventory": q.std(), "Trades per day": res["trades"].mean()}
