import numpy as np
import pytest

from volsurface import (HestonParams, bs_call, calibrate_heston, durrleman_g, fit_svi, heston_call,
                        heston_monte_carlo, implied_vol_call, svi_total_variance)

S0, r, q = 100.0, 0.03, 0.01
K = np.array([80.0, 90.0, 100.0, 110.0, 120.0])


def test_heston_collapses_to_black_scholes():
    """With almost no vol-of-vol and v0 = theta, the variance is constant: Black-Scholes."""
    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=1e-4, rho=0.0)
    assert heston_call(S0, K, 1.0, r, q, p) == pytest.approx(bs_call(S0, K, 1.0, r, 0.2, q), abs=1e-5)


def test_heston_fourier_matches_monte_carlo():
    p = HestonParams()
    fourier = heston_call(S0, K, 1.0, r, q, p)
    mc, se = heston_monte_carlo(S0, K, 1.0, r, q, p, n_paths=200_000, n_steps=250)
    assert np.all(np.abs(fourier - mc) < 4 * se + 0.03)   # MC error plus Euler bias


def test_negative_rho_creates_a_downside_skew():
    p = HestonParams(rho=-0.7)
    vols = [implied_vol_call(c, S0, k, 1.0, r, q) for c, k in zip(heston_call(S0, K, 1.0, r, q, p), K)]
    assert vols[0] > vols[2] > vols[-1]


def test_svi_fit_recovers_parameters():
    true = np.array([0.02, 0.15, -0.6, 0.05, 0.2])
    k = np.linspace(-0.6, 0.4, 25)
    fitted = fit_svi(k, svi_total_variance(k, *true))
    assert fitted == pytest.approx(true, abs=1e-5)


def test_durrleman_flags_arbitrage():
    k = np.linspace(-1, 1, 201)
    assert durrleman_g(k, [0.02, 0.15, -0.6, 0.05, 0.2]).min() > 0      # well-behaved smile
    assert durrleman_g(k, [0.001, 2.0, -0.95, 0.0, 0.01]).min() < 0     # far too steep and thin


def test_calibration_recovers_heston_parameters():
    true = HestonParams()
    maturities = [0.5, 1.0, 2.0]
    strikes = np.linspace(75, 125, 9)
    vols = [np.array([implied_vol_call(c, S0, k, T, r, q) for c, k in
                      zip(heston_call(S0, strikes, T, r, q, true), strikes)]) for T in maturities]
    fitted, rmse = calibrate_heston(S0, r, q, maturities, strikes, vols)
    assert rmse < 1e-5
    assert fitted.rho == pytest.approx(true.rho, abs=0.02)
    assert fitted.v0 == pytest.approx(true.v0, abs=0.002)
