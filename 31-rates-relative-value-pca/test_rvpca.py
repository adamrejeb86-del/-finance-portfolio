import numpy as np
import pandas as pd
import pytest

from rvpca import (curve_package, dv01_per_million, factor_exposure, fit_residuals, mean_reversion_backtest, package_pnl, pca,
                   simulate_curves)


@pytest.fixture(scope="module")
def data():
    curves, idio = simulate_curves(seed=0)
    changes = curves.diff().iloc[1:] * 100
    return curves, idio, changes, pca(changes.iloc[:504])


def test_pca_is_an_orthonormal_decomposition_of_the_covariance(data):
    *_, p = data
    L = p["loadings"].values
    assert np.allclose(L.T @ L, np.eye(len(L)), atol=1e-10)
    assert np.allclose(L @ np.diag(p["eigenvalues"]) @ L.T, p["cov"], atol=1e-8)
    assert p["explained"].sum() == pytest.approx(1.0)
    assert p["explained"][:3].sum() > 0.95


def test_dv01_of_a_par_swap():
    assert dv01_per_million(10, 0.0) == pytest.approx(1000.0)          # 10 years x EUR 100 per bp per million
    assert dv01_per_million(2, 3.0) < dv01_per_million(10, 3.0) < dv01_per_million(30, 3.0)


def test_dv01_neutral_steepener_is_immune_to_a_parallel_shift(data):
    *_, p = data
    pos = curve_package(["2y", "10y"], "dv01", p["loadings"])
    shift = pd.DataFrame([[5.0] * 8], columns=p["loadings"].index)
    assert package_pnl(pos, shift).iloc[0] == pytest.approx(0.0)


def test_pca_weighted_packages_are_factor_neutral(data):
    *_, p = data
    steep = curve_package(["2y", "10y"], "pca", p["loadings"])
    fly = curve_package(["2y", "10y", "30y"], "pca", p["loadings"])
    assert factor_exposure(steep, p)["PC1"] == pytest.approx(0.0, abs=1e-8)
    assert np.allclose(factor_exposure(fly, p)[["PC1", "PC2"]], 0.0, atol=1e-8)
    assert abs(factor_exposure(fly, p)["PC3"]) > 1_000          # the fly keeps its curvature risk


def test_residuals_recover_the_true_dislocations(data):
    curves, idio, _, p = data
    res = fit_residuals(curves, p["loadings"])
    corr = [np.corrcoef(res[c], idio[c])[0, 1] for c in curves.columns]
    assert np.mean(corr) > 0.6


def test_backtest_has_no_look_ahead_and_pays_costs(data):
    curves, *_, p = data
    fly = curve_package(["2y", "10y", "30y"], "pca", p["loadings"])
    r0, _, pos0 = mean_reversion_backtest(curves, fly, cost_bp=0.0)
    r1, _, _ = mean_reversion_backtest(curves, fly, cost_bp=0.25)
    assert r1.sum() < r0.sum()
    shocked = curves.copy()
    shocked.iloc[1500:] += 0.5
    _, _, pos_s = mean_reversion_backtest(shocked, fly, cost_bp=0.0)
    assert np.array_equal(pos0[:1500], pos_s[:1500])
