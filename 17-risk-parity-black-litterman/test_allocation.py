import numpy as np
import pytest

from allocation import (MARKET_WEIGHTS, black_litterman, unconstrained_weights, covariance, implied_returns, inverse_volatility,
                        max_sharpe, mean_variance_weights, min_variance, risk_budget_portfolio, risk_contributions)

COV = covariance()
PI = implied_returns(COV, MARKET_WEIGHTS)


def test_risk_contributions_add_up_to_volatility():
    w = np.full(8, 1 / 8)
    assert risk_contributions(w, COV).sum() == pytest.approx(np.sqrt(w @ COV @ w))


def test_equal_risk_contribution():
    w = risk_budget_portfolio(COV)
    rc = risk_contributions(w, COV)
    assert w.sum() == pytest.approx(1) and np.all(w > 0)
    assert rc / rc.sum() == pytest.approx(np.full(8, 1 / 8), abs=1e-6)


def test_custom_risk_budgets():
    budgets = np.array([3, 1, 1, 1, 1, 1, 1, 1.0])
    rc = risk_contributions(risk_budget_portfolio(COV, budgets), COV)
    assert rc / rc.sum() == pytest.approx(budgets / budgets.sum(), abs=1e-6)


def test_min_variance_beats_alternatives():
    v = lambda w: w @ COV @ w
    assert v(min_variance(COV)) <= min(v(risk_budget_portfolio(COV)), v(inverse_volatility(COV)), v(MARKET_WEIGHTS)) + 1e-12


def test_reverse_optimisation_recovers_market_weights():
    w = mean_variance_weights(PI, COV, risk_aversion=2.5, long_only=False)
    assert w == pytest.approx(MARKET_WEIGHTS, abs=1e-4)


def test_black_litterman_without_strong_views_stays_at_equilibrium():
    P = np.zeros((1, 8)); P[0, 0] = 1
    mu, _ = black_litterman(COV, PI, P, [PI[0]])       # a view equal to equilibrium changes nothing
    assert mu == pytest.approx(PI, abs=1e-12)


def test_black_litterman_moves_towards_the_view():
    P = np.zeros((1, 8)); P[0, 1], P[0, 0] = 1, -1     # EM outperforms global equities
    q = PI[1] - PI[0] + 0.03
    mu_low, _ = black_litterman(COV, PI, P, [q], confidences=[0.2])
    mu_high, _ = black_litterman(COV, PI, P, [q], confidences=[0.9])
    spread = lambda m: m[1] - m[0]
    assert spread(PI) < spread(mu_low) < spread(mu_high) < q


def test_max_sharpe_is_on_the_frontier():
    w = max_sharpe(PI, COV)
    s = lambda w: (w @ PI - 0.02) / np.sqrt(w @ COV @ w)
    assert s(w) >= max(s(MARKET_WEIGHTS), s(risk_budget_portfolio(COV))) - 1e-9


def test_views_only_move_the_assets_they_involve():
    P = np.zeros((1, 8)); P[0, 7] = 1                  # absolute view on gold only
    mu, _ = black_litterman(COV, PI, P, [0.07])
    w = unconstrained_weights(mu, COV)
    others = [i for i in range(8) if i != 7]
    assert w[others] == pytest.approx(MARKET_WEIGHTS[others], abs=1e-10) and w[7] > MARKET_WEIGHTS[7]
