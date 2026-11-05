import numpy as np
import pytest

from copula import (basel_irb_capital, expected_tranche_loss, lhp_cdf, lhp_quantile, simulate_losses,
                    tranche_par_spread, var_es)


def test_expected_loss_is_independent_of_correlation():
    ead, pd_, lgd = np.full(200, 1.0), np.full(200, 0.02), np.full(200, 0.45)
    for rho in (0.0, 0.2, 0.5):
        L = simulate_losses(ead, pd_, lgd, rho, 100_000, seed=1)
        assert L.mean() == pytest.approx((ead * pd_ * lgd).sum(), rel=0.02)


def test_large_pool_matches_vasicek_closed_form():
    n, p, rho = 2000, 0.03, 0.15
    L = simulate_losses(np.full(n, 1 / n), np.full(n, p), np.ones(n), rho, 40_000, seed=2)
    for x in (0.02, 0.05, 0.10):
        assert (L <= x).mean() == pytest.approx(lhp_cdf(x, p, rho), abs=0.012)


def test_quantile_inverts_cdf():
    assert lhp_cdf(lhp_quantile(0.999, 0.01, 0.2), 0.01, 0.2) == pytest.approx(0.999, abs=1e-10)


def test_correlation_fattens_the_tail():
    ead, pd_, lgd = np.full(300, 1.0), np.full(300, 0.01), np.full(300, 0.45)
    low = var_es(simulate_losses(ead, pd_, lgd, 0.05, 100_000, seed=3))[0]
    high = var_es(simulate_losses(ead, pd_, lgd, 0.30, 100_000, seed=3))[0]
    assert high > 2 * low


def test_student_t_copula_has_heavier_tail():
    ead, pd_, lgd = np.full(300, 1.0), np.full(300, 0.01), np.full(300, 0.45)
    g = var_es(simulate_losses(ead, pd_, lgd, 0.15, 100_000, seed=4))[0]
    t = var_es(simulate_losses(ead, pd_, lgd, 0.15, 100_000, seed=4, nu=4))[0]
    assert t > g


def test_tranches_add_up_to_the_pool():
    p, rho, cuts = 0.05, 0.3, [0, 0.03, 0.07, 0.15, 0.3, 1.0]
    total = sum((b - a) * expected_tranche_loss(p, rho, a, b) for a, b in zip(cuts[:-1], cuts[1:]))
    assert total == pytest.approx(0.6 * p, rel=1e-3)          # pool expected loss = (1 - R) PD


def test_equity_tranche_is_long_correlation_senior_is_short():
    assert tranche_par_spread(0, 0.03, 0.6) < tranche_par_spread(0, 0.03, 0.1)
    assert tranche_par_spread(0.15, 0.3, 0.6) > tranche_par_spread(0.15, 0.3, 0.1)


def test_irb_capital_is_positive_and_increasing_with_pd():
    k = [basel_irb_capital(p, 0.45) for p in (0.001, 0.01, 0.05)]
    assert all(x > 0 for x in k) and k[0] < k[1] < k[2]
