import numpy as np
import pytest

from garch import GarchFit, fit, forecast_variance, news_impact, qlike, simulate, variance_path

TRUE = dict(omega=2e-6, alpha=0.05, gamma=0.10, beta=0.88)
R = simulate(8000, **TRUE, nu=7, seed=1)


def test_parameter_recovery_gjr_t():
    f = fit(R, asym=True, student=True)
    assert f.alpha == pytest.approx(0.05, abs=0.03)
    assert f.gamma == pytest.approx(0.10, abs=0.04)
    assert f.beta == pytest.approx(0.88, abs=0.03)
    assert f.nu == pytest.approx(7, abs=2.5)
    assert f.persistence < 1


def test_richer_model_has_higher_likelihood():
    g = fit(R)
    gjr_t = fit(R, asym=True, student=True)
    assert gjr_t.loglik > g.loglik and gjr_t.aic < g.aic


def test_forecast_reverts_to_long_run_variance():
    f = GarchFit(0.0, 2e-6, 0.05, 0.10, 0.88, None, 0.0, 1)
    path = forecast_variance(f, last_var_next=4 * f.long_run_vol**2, horizon=2000)
    assert path[0] == pytest.approx(4 * f.long_run_vol**2)
    assert path[-1] == pytest.approx(f.long_run_vol**2, rel=1e-6)


def test_news_impact_is_asymmetric_with_gamma():
    f = GarchFit(0.0, 2e-6, 0.05, 0.10, 0.88, None, 0.0, 1)
    up, down = news_impact(f, np.array([0.02, -0.02]))
    assert down > up


def test_variance_recursion():
    eps = np.array([0.01, -0.02, 0.005])
    s2 = variance_path(eps, 1e-6, 0.1, 0.0, 0.8, init=1e-4)
    assert s2[1] == pytest.approx(1e-6 + 0.1 * 1e-4 + 0.8 * 1e-4)


def test_qlike_is_minimised_by_the_truth():
    rng = np.random.default_rng(0)
    true_var = rng.uniform(1e-4, 4e-4, 5000)
    realised = true_var * rng.chisquare(1, 5000)
    assert qlike(realised, true_var) < qlike(realised, true_var * 1.3) and qlike(realised, true_var) < qlike(realised, true_var * 0.7)
