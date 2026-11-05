"""
Credit portfolio risk with the one-factor Gaussian copula (Vasicek), credit VaR, the Basel
IRB capital formula, and CDO tranche pricing.

Each obligor i defaults if its latent variable falls below a threshold:
      X_i = sqrt(rho) Z + sqrt(1 - rho) eps_i,     default  <=>  X_i < Phi^{-1}(PD_i)
Z is the common factor (the economy): defaults are independent given Z, which gives
  - fast Monte Carlo of portfolio losses,
  - the large-homogeneous-pool (LHP) closed form of the loss distribution, which is the
    basis of the Basel II/III IRB capital formula,
  - semi-analytic tranche pricing by integrating over Z.
A Student-t copula variant shows the effect of tail dependence.

Author: Adam Rejeb
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm, t as student_t


# ---------------------------------------------------------------- portfolio simulation
def simulate_losses(ead, pd_, lgd, rho, n_sims=100_000, seed=0, nu=None, chunk=20_000):
    """Portfolio loss for each scenario, shape (n_sims,).

    nu: degrees of freedom of a Student-t copula (None = Gaussian). With a t copula, all
    latent variables are scaled by the same random factor, so defaults cluster in the tails.
    """
    ead, pd_, lgd = map(np.asarray, (ead, pd_, lgd))
    rng = np.random.default_rng(seed)
    if nu is None:
        thresholds = norm.ppf(pd_)
    else:
        thresholds = student_t.ppf(pd_, nu)
    losses = []
    for start in range(0, n_sims, chunk):
        m = min(chunk, n_sims - start)
        Z = rng.standard_normal((m, 1))
        eps = rng.standard_normal((m, len(ead)))
        X = np.sqrt(rho) * Z + np.sqrt(1 - rho) * eps
        if nu is not None:
            X = X / np.sqrt(rng.chisquare(nu, (m, 1)) / nu)
        losses.append(((X < thresholds) * (ead * lgd)).sum(axis=1))
    return np.concatenate(losses)


def var_es(losses, alpha=0.999):
    var = np.quantile(losses, alpha)
    return float(var), float(losses[losses >= var].mean())


# ---------------------------------------------------------------- Vasicek closed forms
def conditional_pd(pd_, rho, z):
    """Default probability given the factor value z."""
    return norm.cdf((norm.ppf(pd_) - np.sqrt(rho) * z) / np.sqrt(1 - rho))


def lhp_cdf(x, pd_, rho):
    """P(loss fraction <= x) in a large homogeneous pool (LGD = 100%)."""
    x = np.clip(x, 1e-12, 1 - 1e-12)
    return norm.cdf((np.sqrt(1 - rho) * norm.ppf(x) - norm.ppf(pd_)) / np.sqrt(rho))


def lhp_quantile(alpha, pd_, rho):
    """Loss fraction not exceeded with probability alpha (LGD = 100%)."""
    return norm.cdf((norm.ppf(pd_) + np.sqrt(rho) * norm.ppf(alpha)) / np.sqrt(1 - rho))


def basel_corporate_correlation(pd_):
    w = (1 - np.exp(-50 * pd_)) / (1 - np.exp(-50))
    return 0.12 * w + 0.24 * (1 - w)


def basel_irb_capital(pd_, lgd, maturity=2.5):
    """Basel IRB capital requirement K per unit of exposure (corporates), 99.9% confidence,
    unexpected loss only, with the maturity adjustment."""
    rho = basel_corporate_correlation(pd_)
    b = (0.11852 - 0.05478 * np.log(pd_)) ** 2
    ma = (1 + (maturity - 2.5) * b) / (1 - 1.5 * b)
    return lgd * (lhp_quantile(0.999, pd_, rho) - pd_) * ma


# ---------------------------------------------------------------- CDO tranches
_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(80)      # Gauss-Hermite nodes for E[f(Z)], Z ~ N(0,1)
_GH_W = _GH_W / _GH_W.sum()


def expected_tranche_loss(pd_t, rho, attach, detach, recovery=0.4):
    """E[tranche loss] / tranche size at one date, large homogeneous pool, by integrating
    the conditional (deterministic) pool loss over the common factor."""
    pool_loss = (1 - recovery) * conditional_pd(pd_t, rho, _GH_X)
    tranche = (np.clip(pool_loss, attach, detach) - attach) / (detach - attach)
    return float((_GH_W * tranche).sum())


def tranche_par_spread(attach, detach, rho, hazard=0.012, recovery=0.4, r=0.03, maturity=5.0, freq=4):
    """Par spread of a tranche: quarterly premium on the outstanding notional against the
    increase in expected tranche loss (protection leg)."""
    times = np.arange(1, int(round(maturity * freq)) + 1) / freq
    el = np.array([0.0] + [expected_tranche_loss(1 - np.exp(-hazard * t), rho, attach, detach, recovery) for t in times])
    dfs = np.exp(-r * times)
    premium = (dfs * (1 - 0.5 * (el[1:] + el[:-1])) / freq).sum()     # outstanding notional, mid-period
    protection = (dfs * np.diff(el)).sum()
    return protection / premium
