import numpy as np
import pytest

from refund import Fund, irr, liquidation, project, sample_fund

F = sample_fund()
NAMES = [a.name for a in F.assets]


def test_nav_identity_and_ratios():
    assert F.nav() == pytest.approx(F.portfolio_value() + F.cash - F.debt)
    assert F.ltv() == pytest.approx(F.debt / F.portfolio_value())


def test_projection_without_shocks_keeps_values_consistent():
    p = project(F, years=3, rent_growth=0.0)
    assert np.allclose(p["NAV"], p["portfolio value"] + p["cash"] - p["debt"])


def test_cap_rate_shock_lowers_nav_and_raises_ltv():
    base = project(F, years=3)
    stress = project(F, years=3, cap_shift={1: 0.01})
    assert stress["NAV per share"].iloc[-1] < base["NAV per share"].iloc[-1]
    assert stress["LTV"].iloc[0] > base["LTV"].iloc[0]


def test_fire_sale_returns_less_than_orderly_sale_in_flat_market():
    fire = liquidation(F, NAMES, [0] * len(NAMES))
    orderly = liquidation(F, NAMES, [1 + i % 3 for i in range(len(NAMES))])
    assert fire["total per share"] < orderly["total per share"]


def test_liquidation_cash_conservation_without_discounts():
    """Selling everything today at appraisal with no costs returns exactly the NAV."""
    res = liquidation(F, NAMES, [0] * len(NAMES), discounts={}, transaction_cost=0.0)
    assert res["total per share"] == pytest.approx(F.nav_per_share())


def test_irr_solver():
    assert irr([-100, 10, 10, 110]) == pytest.approx(0.10, abs=1e-8)
