import numpy as np
import pytest

from cva import HullWhite, Swap, collateralised, cva, exposure_profile, par_rate, swap_values

M = HullWhite()
TIMES = np.round(np.arange(0, 10 + 1e-9, 1 / 26), 10)
X, D = M.simulate(TIMES, 30_000, seed=1)
K = par_rate(M.curve, 10)
PAYER = Swap(1e8, K, 10, pay_fixed=True)
V = swap_values(M, PAYER, TIMES, X)


def test_discount_factors_reprice_the_curve():
    for T in (2.0, 5.0, 10.0):
        j = int(round(T * 26))
        assert D[:, j].mean() == pytest.approx(M.curve.df(T), abs=4 * D[:, j].std() / np.sqrt(len(D)) + 1e-4)


def test_par_swap_starts_at_zero_and_expires_at_zero():
    assert np.allclose(V[:, 0], 0, atol=1e-6) and np.allclose(V[:, -1], 0)


def test_discounted_value_is_a_martingale():
    j = int(round(3.0 * 26))
    pay = np.arange(4, 11.0)
    analytic = 1e8 * sum(M.curve.df(p - 1) - M.curve.df(p) - K * M.curve.df(p) for p in pay)
    y = D[:, j] * V[:, j]
    assert y.mean() == pytest.approx(analytic, abs=4 * y.std() / np.sqrt(len(y)))


def test_netting_reduces_exposure():
    receiver = swap_values(M, Swap(5e7, par_rate(M.curve, 5), 5, pay_fixed=False), TIMES, X)
    separate = exposure_profile(V)["EPE"] + exposure_profile(receiver)["EPE"]
    assert exposure_profile(V + receiver)["EPE"] < separate


def test_collateral_reduces_exposure():
    assert exposure_profile(collateralised(V, TIMES))["EPE"] < 0.2 * exposure_profile(V)["EPE"]


def test_cva_increases_with_spread_and_with_wrong_way_risk():
    c1, c2 = cva(V, D, TIMES, 0.01), cva(V, D, TIMES, 0.03)
    assert 0 < c1 < c2
    assert cva(V, D, TIMES, 0.02, wwr_beta=0.5, x=X) > cva(V, D, TIMES, 0.02) > cva(V, D, TIMES, 0.02, wwr_beta=-0.5, x=X)
