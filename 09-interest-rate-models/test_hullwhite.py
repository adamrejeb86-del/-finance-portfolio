import numpy as np
import pytest

from hullwhite import (HullWhite, InitialCurve, atm_cap_strike, calibrate, implied_cap_normal_vol,
                       market_cap_price)

HW = HullWhite()
C = HW.curve


def test_model_reproduces_todays_curve_exactly():
    T = np.array([0.5, 1, 2, 5, 10, 20])
    r0 = C.inst_forward(0.0)
    assert HW.zcb(0.0, T, r0) == pytest.approx(C.df(T), rel=1e-6)


def test_monte_carlo_reprices_zero_coupon_bonds():
    t, r, disc = HW.simulate(10.0, 400, 50_000, seed=1)
    for T in (2.0, 5.0, 10.0):
        i = int(round(T / 10 * 400))
        mc = disc[:, i]
        assert mc.mean() == pytest.approx(C.df(T), abs=4 * mc.std() / np.sqrt(len(mc)) + 2e-5)


def test_zcb_put_call_parity():
    T, S, K = 2.0, 5.0, 0.9
    assert HW.zcb_call(T, S, K) - HW.zcb_put(T, S, K) == pytest.approx(C.df(S) - K * C.df(T), abs=1e-12)


def test_caplet_matches_monte_carlo():
    T, delta, K = 3.0, 0.5, 0.03
    t, r, disc = HW.simulate(T, 300, 100_000, seed=2)
    P_TS = HW.zcb(T, T + delta, r[:, -1])
    L = (1 / P_TS - 1) / delta
    pv = disc[:, -1] * P_TS * delta * np.maximum(L - K, 0)      # paid at T + delta
    assert HW.caplet(T, delta, K) == pytest.approx(pv.mean(), abs=4 * pv.std() / np.sqrt(len(pv)))


def test_swaption_matches_monte_carlo():
    expiry, tenor, K = 2.0, 5.0, 0.03
    t, r, disc = HW.simulate(expiry, 200, 100_000, seed=3)
    pay = expiry + np.arange(1, 6)
    bonds = np.stack([HW.zcb(expiry, ti, r[:, -1]) for ti in pay], axis=1)
    swap_value = 1 - bonds[:, -1] - K * bonds.sum(axis=1)        # payer swap at expiry
    pv = disc[:, -1] * np.maximum(swap_value, 0)
    assert HW.payer_swaption(expiry, tenor, K) == pytest.approx(pv.mean(), abs=4 * pv.std() / np.sqrt(len(pv)))


def test_normal_vol_round_trip():
    K = atm_cap_strike(C, 5)
    price = market_cap_price(C, 5, K, 0.0085)
    assert implied_cap_normal_vol(C, 5, K, price) == pytest.approx(0.0085, abs=1e-9)


def test_calibration_recovers_parameters():
    true = HullWhite(0.08, 0.011, C)
    mats = [2, 3, 5, 7, 10]
    vols = [implied_cap_normal_vol(C, T, atm_cap_strike(C, T), true.cap(T, atm_cap_strike(C, T))) for T in mats]
    fitted, rmse = calibrate(C, mats, vols)
    assert rmse < 1e-6
    assert fitted.a == pytest.approx(0.08, abs=1e-4) and fitted.sigma == pytest.approx(0.011, abs=1e-6)
