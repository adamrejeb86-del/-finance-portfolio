import numpy as np
import pandas as pd
import pytest

from factors import (brinson_fachler, carino_link, factor_regression, ols_newey_west, performance_stats, rolling_betas,
                     simulate_factors, simulate_fund)

F = simulate_factors(600, seed=3)


def test_regression_recovers_true_exposures_and_alpha():
    betas = [1.0, 0.4, 0.6, -0.1, 0.2]
    fund = simulate_fund(F, betas, alpha_annual=0.03, idio_vol_annual=0.02, seed=4)
    reg = factor_regression(fund, F)
    assert reg["coef"].iloc[1:].values == pytest.approx(betas, abs=0.05)
    assert reg.loc["alpha (monthly)", "coef"] * 12 == pytest.approx(0.03, abs=0.01)


def test_hac_errors_exceed_ols_errors_with_autocorrelated_noise():
    fund = simulate_fund(F, [1, 0, 0, 0, 0], idio_vol_annual=0.05, seed=5)
    reg = factor_regression(fund, F, lags=6)
    assert reg.loc["alpha (monthly)", "se (HAC)"] > reg.loc["alpha (monthly)", "se (OLS)"]


def test_rolling_betas_detect_style_drift():
    fund = simulate_fund(F, [1, 0, 0.6, 0, 0], drift=(300, [1, 0, -0.2, 0, 0]), idio_vol_annual=0.01, seed=6)
    rb = rolling_betas(fund, F, 36)
    assert rb["HML"].iloc[250] == pytest.approx(0.6, abs=0.1) and rb["HML"].iloc[-1] == pytest.approx(-0.2, abs=0.1)


def test_brinson_effects_add_up_to_active_return():
    rng = np.random.default_rng(0)
    wb = rng.dirichlet(np.ones(10)); wp = rng.dirichlet(np.ones(10))
    rb, rp = rng.normal(0.01, 0.03, 10), rng.normal(0.012, 0.03, 10)
    eff = brinson_fachler(wp, wb, rp, rb)
    assert eff.values.sum() == pytest.approx(wp @ rp - wb @ rb, abs=1e-15)


def test_carino_linking_matches_compounded_active_return():
    rng = np.random.default_rng(1)
    effects, Rp, Rb = [], [], []
    for _ in range(12):
        wb = rng.dirichlet(np.ones(5)); wp = rng.dirichlet(np.ones(5))
        rb, rp = rng.normal(0.01, 0.04, 5), rng.normal(0.01, 0.04, 5)
        effects.append(brinson_fachler(wp, wb, rp, rb)); Rp.append(wp @ rp); Rb.append(wb @ rb)
    linked = carino_link(effects, Rp, Rb)
    assert linked.values.sum() == pytest.approx(np.prod(1 + np.array(Rp)) - np.prod(1 + np.array(Rb)), abs=1e-12)


def test_performance_stats_on_known_series():
    r = pd.Series([0.01] * 12)
    s = performance_stats(r)
    assert s["Annual return"] == pytest.approx(1.01**12 - 1) and s["Max drawdown"] == 0.0
