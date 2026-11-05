import numpy as np
import pytest

from alm import (ZeroCurve, bond_universe, cash_flow_matching, par_bond_universe, liability_cashflows, match_key_rates,
                 match_pv_duration, parallel, portfolio_flows, pv, random_curve_shocks, redington,
                 risk_measures, surplus_change)

C = ZeroCurve()
LT, LCF = liability_cashflows()
BT, BCF, MATS = bond_universe()
PILLARS = [2, 5, 10, 15, 20, 30, 40]


def _assets(w):
    return portfolio_flows(w, BCF)


def test_duration_matching_constraints():
    w = match_pv_duration(LT, LCF, BT, BCF, C, (0, 6))
    A, L = risk_measures(BT, _assets(w), C), risk_measures(LT, LCF, C)
    assert A["pv"] == pytest.approx(L["pv"]) and A["duration"] == pytest.approx(L["duration"])


def test_duration_matched_surplus_is_second_order():
    w = match_pv_duration(LT, LCF, BT, BCF, C, (1, 5))
    small = surplus_change(w, BT, BCF, LT, LCF, C, parallel(1))
    assert abs(small) < 1e-3 * abs(pv(LT, LCF, C) * 1e-4 * risk_measures(LT, LCF, C)["duration"])


def test_redington_gains_on_parallel_moves_both_ways():
    w = redington(LT, LCF, BT, BCF, C, (0, 3, 6))
    assert surplus_change(w, BT, BCF, LT, LCF, C, parallel(100)) > 0
    assert surplus_change(w, BT, BCF, LT, LCF, C, parallel(-100)) > 0


def test_cash_flow_matching_covers_every_payment():
    t, flows, _ = par_bond_universe(C)
    w = cash_flow_matching(LCF, flows, C, t, reinvest=0.0)
    assert np.all(w >= 0)
    assert np.all(np.cumsum(portfolio_flows(w, flows)[: len(LCF)]) >= np.cumsum(LCF) - 1e-6)
    # with one bond per year the dedicated portfolio costs about the PV of the liabilities
    assert pv(t, portfolio_flows(w, flows), C) == pytest.approx(pv(LT, LCF, C), rel=0.01)


def test_key_rate_matching_beats_duration_matching_on_curve_twists():
    w_dur = match_pv_duration(LT, LCF, BT, BCF, C, (0, 6))
    w_krd = match_key_rates(LT, LCF, BT, BCF, C, PILLARS)
    shocks = random_curve_shocks(400, seed=1)
    sd = lambda w: np.std([surplus_change(w, BT, BCF, LT, LCF, C, s) for s in shocks])
    assert sd(w_krd) < 0.5 * sd(w_dur)
