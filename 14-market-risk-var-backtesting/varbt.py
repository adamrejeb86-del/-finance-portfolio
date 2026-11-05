"""
Market risk engine: four VaR models, their regulatory backtests, and stress tests.

VaR / ES models (one-day, rolling window)
  - Parametric normal (variance-covariance)
  - Historical simulation
  - EWMA / RiskMetrics (lambda = 0.94) with normal quantiles
  - Filtered historical simulation: historical returns rescaled by today's EWMA volatility

Backtests
  - Kupiec proportion-of-failures test (is the number of exceptions right?)
  - Christoffersen independence and conditional-coverage tests (do exceptions cluster?)
  - Basel traffic-light zones over 250 days and the capital multiplier they imply
  - Acerbi-Szekely Z2 test for Expected Shortfall (FRTB moved capital from VaR to ES)

Stress testing
  - Historical-style scenarios applied to factor exposures
  - Reverse stress test: the smallest move along a scenario direction that causes a given loss

Author: Adam Rejeb
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm


ASSETS = ["Equities", "Govt bonds", "Credit", "USD/EUR", "Commodities"]


# ---------------------------------------------------------------- synthetic market
def simulate_market(n_days=2500, seed=0) -> pd.DataFrame:
    """Daily returns of five asset classes with volatility clustering, fat tails and a
    stress regime in which volatilities jump and equities, credit and commodities fall together."""
    rng = np.random.default_rng(seed)
    base_vol = np.array([0.011, 0.003, 0.002, 0.005, 0.013])
    calm_corr = np.array([[1, -0.3, 0.5, 0.2, 0.3], [-0.3, 1, 0.2, -0.1, -0.1], [0.5, 0.2, 1, 0.1, 0.2],
                          [0.2, -0.1, 0.1, 1, 0.2], [0.3, -0.1, 0.2, 0.2, 1]])
    stress_corr = np.array([[1, -0.5, 0.8, 0.4, 0.6], [-0.5, 1, -0.2, -0.3, -0.4], [0.8, -0.2, 1, 0.3, 0.5],
                            [0.4, -0.3, 0.3, 1, 0.4], [0.6, -0.4, 0.5, 0.4, 1]])
    L_calm, L_stress = np.linalg.cholesky(calm_corr), np.linalg.cholesky(stress_corr)
    regime, var = 0, base_vol**2
    out = np.empty((n_days, 5))
    for t in range(n_days):
        if regime == 0 and rng.random() < 0.004:
            regime = 1
        elif regime == 1 and rng.random() < 0.03:
            regime = 0
        nu = 5
        z = rng.standard_normal(5) / np.sqrt(rng.chisquare(nu) / nu) * np.sqrt((nu - 2) / nu)   # unit-variance t
        L = L_stress if regime else L_calm
        level = 2.2 if regime else 1.0
        shock = np.sqrt(var) * (L @ z)
        drift = np.array([-0.0015, 0.0002, -0.0004, 0.0, -0.001]) if regime else np.array([0.0003, 0.00005, 0.00008, 0.0, 0.0001])
        out[t] = drift + shock
        target = (base_vol * level) ** 2
        var = 0.05 * target + 0.88 * var + 0.07 * shock**2                    # GARCH(1,1)-type clustering
    dates = pd.bdate_range("2016-01-04", periods=n_days)
    return pd.DataFrame(out, index=dates, columns=ASSETS)


# ---------------------------------------------------------------- VaR models
def rolling_var_es(pnl: pd.Series, method: str, alpha=0.99, window=250, lam=0.94, asset_returns=None, weights=None):
    """One-day VaR and ES forecasts (positive numbers = losses), made with data up to t-1."""
    x = pnl.values
    n = len(x)
    var_f, es_f = np.full(n, np.nan), np.full(n, np.nan)
    q = norm.ppf(1 - alpha)
    ewma_var = np.var(x[:window])
    for t in range(window, n):
        hist = x[t - window:t]
        ewma_var = lam * ewma_var + (1 - lam) * x[t - 1] ** 2
        if method == "normal":
            mu, sd = hist.mean(), hist.std(ddof=1)
            var_f[t] = -(mu + q * sd)
            es_f[t] = -(mu - sd * norm.pdf(q) / (1 - alpha))
        elif method == "historical":
            var_f[t] = -np.quantile(hist, 1 - alpha)
            es_f[t] = -hist[hist <= -var_f[t]].mean()
        elif method == "ewma":
            sd = np.sqrt(ewma_var)
            var_f[t] = -q * sd
            es_f[t] = sd * norm.pdf(q) / (1 - alpha)
        elif method == "fhs":
            # rescale each past return by the ratio of today's vol to the vol at that time
            vols = _ewma_path(x[:t], lam)[-window:]
            scaled = hist / vols * np.sqrt(ewma_var)
            var_f[t] = -np.quantile(scaled, 1 - alpha)
            es_f[t] = -scaled[scaled <= -var_f[t]].mean()
        else:
            raise ValueError(method)
    return pd.Series(var_f, pnl.index), pd.Series(es_f, pnl.index)


def _ewma_path(x, lam):
    v = np.empty(len(x))
    v[0] = np.var(x[: min(250, len(x))])
    for i in range(1, len(x)):
        v[i] = lam * v[i - 1] + (1 - lam) * x[i - 1] ** 2
    return np.sqrt(v)


# ---------------------------------------------------------------- backtests
def kupiec_pof(n_exceptions, n_obs, alpha=0.99):
    """Likelihood-ratio test that the exception rate equals 1 - alpha. Returns (LR, p-value)."""
    p, x, T = 1 - alpha, n_exceptions, n_obs
    phat = x / T
    def ll(prob):
        prob = np.clip(prob, 1e-12, 1 - 1e-12)
        return (T - x) * np.log(1 - prob) + x * np.log(prob)
    lr = -2 * (ll(p) - ll(phat))
    return float(lr), float(1 - chi2.cdf(lr, 1))


def christoffersen(exceptions, alpha=0.99):
    """Independence test (do exceptions follow exceptions?) and conditional coverage.
    Returns dict with LR_ind, p_ind, LR_cc, p_cc."""
    e = np.asarray(exceptions, dtype=int)
    n00 = np.sum((e[:-1] == 0) & (e[1:] == 0)); n01 = np.sum((e[:-1] == 0) & (e[1:] == 1))
    n10 = np.sum((e[:-1] == 1) & (e[1:] == 0)); n11 = np.sum((e[:-1] == 1) & (e[1:] == 1))
    pi0 = n01 / max(n00 + n01, 1); pi1 = n11 / max(n10 + n11, 1); pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    def safe(a, b):
        return b * np.log(a) if b > 0 else 0.0
    ll_null = safe(1 - pi, n00 + n10) + safe(pi, n01 + n11)
    ll_alt = safe(1 - pi0, n00) + safe(pi0, n01) + safe(1 - pi1, n10) + safe(pi1, n11)
    lr_ind = -2 * (ll_null - ll_alt)
    lr_pof, _ = kupiec_pof(e.sum(), len(e), alpha)
    lr_cc = lr_pof + lr_ind
    return {"LR_ind": lr_ind, "p_ind": 1 - chi2.cdf(lr_ind, 1), "LR_cc": lr_cc, "p_cc": 1 - chi2.cdf(lr_cc, 2)}


def traffic_light(n_exceptions_250):
    """Basel zones for a 99% VaR over 250 days and the multiplier on market-risk capital."""
    plus = {5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}
    if n_exceptions_250 <= 4:
        return "green", 3.0
    if n_exceptions_250 <= 9:
        return "yellow", 3.0 + plus[n_exceptions_250]
    return "red", 4.0


def acerbi_szekely_z2(pnl, var_f, es_f, alpha=0.975):
    """Z2 = sum(X_t 1{X_t + VaR_t < 0} / (T (1 - alpha) ES_t)) + 1, with X_t the P&L.
    Expected value 0 if ES is correct; negative values mean ES is underestimated."""
    X, v, s = np.asarray(pnl), np.asarray(var_f), np.asarray(es_f)
    hit = X < -v
    return float((X * hit / s).sum() / (len(X) * (1 - alpha)) + 1)


def z2_critical_value(alpha=0.975, n=1000, n_sims=5000, seed=0, level=0.05):
    """5% critical value of Z2 under a correct normal model, by simulation."""
    rng = np.random.default_rng(seed)
    q = norm.ppf(1 - alpha)
    v, s = -q, norm.pdf(q) / (1 - alpha)
    zs = [acerbi_szekely_z2(rng.standard_normal(n), np.full(n, v), np.full(n, s), alpha) for _ in range(n_sims)]
    return float(np.quantile(zs, level))


# ---------------------------------------------------------------- stress tests
SCENARIOS = {   # one-off moves in % for each asset class, stylised versions of past episodes
    "2008 Lehman (Sep-Oct)": [-0.30, 0.04, -0.08, 0.10, -0.35],
    "2020 Covid crash (Mar)": [-0.33, 0.02, -0.06, 0.03, -0.30],
    "2022 rates shock": [-0.20, -0.17, -0.12, 0.12, 0.25],
    "Euro sovereign crisis 2011": [-0.22, -0.05, -0.07, -0.06, -0.10],
}


def stress_pnl(exposures, scenario):
    return float(np.dot(exposures, scenario))


def reverse_stress(exposures, direction, loss_threshold):
    """Scale k of the scenario direction such that the loss equals the threshold."""
    unit = stress_pnl(exposures, direction)
    if unit >= 0:
        return np.inf
    return float(-loss_threshold / unit)
