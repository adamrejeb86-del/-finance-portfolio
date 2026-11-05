import numpy as np
import pytest

from dispersion import (Regimes, dispersion_weights, expected_variances, implied_correlation, index_vol_from_corr, realised_vol,
                        run_dispersion, simulate_market, var_swap_pnl)

VOLS = np.array([0.2, 0.25, 0.3, 0.4])
W = np.full(4, 0.25)


def test_implied_correlation_inverts_the_index_formula():
    for rho in (0.0, 0.3, 0.8, 1.0):
        assert implied_correlation(index_vol_from_corr(VOLS, W, rho), VOLS, W) == pytest.approx(rho)
    assert index_vol_from_corr(VOLS, W, 1.0) == pytest.approx((W * VOLS).sum())
    assert index_vol_from_corr(VOLS, W, 0.0) == pytest.approx(np.sqrt(((W * VOLS) ** 2).sum()))


def test_variance_swap_payoff():
    assert var_swap_pnl(10_000, 0.20, 0.20) == 0
    assert var_swap_pnl(10_000, 0.20, 0.201) == pytest.approx(10_000 * 0.1, rel=0.01)   # ~ vega x 0.1 vol point
    assert var_swap_pnl(10_000, 0.20, 0.25) > -var_swap_pnl(10_000, 0.20, 0.15)          # convexity


def test_correlation_weights_neutralise_a_proportional_vol_move():
    rho = 0.5
    iv_index = index_vol_from_corr(VOLS, W, rho)
    bump = 1.01
    rv, rv_index = VOLS * bump, index_vol_from_corr(VOLS * bump, W, rho)
    pnl = {}
    for scheme in ("vega", "correlation"):
        v = dispersion_weights(10_000, iv_index, VOLS, W, scheme)
        pnl[scheme] = var_swap_pnl(v, VOLS, rv).sum() - var_swap_pnl(10_000, iv_index, rv_index)
    assert abs(pnl["correlation"]) < 0.02 * pnl["vega"]       # only a tiny convexity term is left
    assert pnl["vega"] > 0                                     # vega-flat dispersion is net long volatility


def test_expected_variances_mix_the_regimes():
    reg = Regimes()
    w = np.full(10, 0.1)
    calm_single, calm_index = expected_variances(0, reg, w)
    base = np.array(reg.base_vols)
    assert np.all(base**2 < calm_single) and np.all(calm_single < (base * reg.vol_mult[1]) ** 2)
    stress_single, _ = expected_variances(1, reg, w)
    assert np.all(stress_single > calm_single)


def test_realised_correlation_is_recovered_in_a_single_regime():
    reg = Regimes(stay=(1.0, 1.0))
    r, state = simulate_market(n_days=5040, seed=1, reg=reg)
    w = np.full(r.shape[1], 0.1)
    rho = implied_correlation(realised_vol(r.values @ w), realised_vol(r.values), w)
    assert state.max() == 0 and rho == pytest.approx(reg.corr[0], abs=0.03)


def test_strikes_use_only_past_information():
    r, state = simulate_market(seed=2)
    a = run_dispersion(r, state)
    shocked = r.copy(); shocked.iloc[1000:] *= 3
    b = run_dispersion(shocked, state)
    assert np.allclose(a["implied index vol"], b["implied index vol"])
    assert not np.allclose(a["realised index vol"], b["realised index vol"])
