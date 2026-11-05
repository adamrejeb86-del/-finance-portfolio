import numpy as np
import pandas as pd
import pytest
from dataclasses import replace

from salestool import (CapitalProtected, Market, Phoenix, ReverseConvertible, scenario_outcome, scenario_path, solve_barrier,
                       solve_coupon, term_sheet_pdf)

PAIR = Market(names=("European bank", "Luxury goods"))


def test_solved_coupon_reprices_the_note_to_target():
    for product in (Phoenix(), ReverseConvertible(), CapitalProtected()):
        q = solve_coupon(product, PAIR, sales_margin=0.015)
        assert q["value_check"] == pytest.approx(98.5, abs=1e-9)


def test_zero_volatility_matches_the_closed_form():
    m = Market(vol_shift=-0.17)                      # the index with zero vol: forward drifts down (r < q)
    p = Phoenix(maturity=3)
    q = solve_coupon(p, m, sales_margin=0.015, n_paths=2)
    df = np.exp(-(m.rate + m.funding_spread) * p.times)
    expected = (98.5 - 100 * df[-1]) / (100 * df.sum())   # never called, every coupon paid, 100 back
    assert q["coupon_per_period"] == pytest.approx(expected, rel=1e-10)


def test_coupon_rises_with_risk():
    base = solve_coupon(Phoenix(), PAIR)["coupon_pa"]
    assert solve_coupon(Phoenix(coupon_barrier=0.7, protection_barrier=0.7), PAIR)["coupon_pa"] > base
    assert solve_coupon(Phoenix(), replace(PAIR, vol_shift=0.05))["coupon_pa"] > base
    assert solve_coupon(Phoenix(), Market(names=PAIR.names + ("Chip maker",)))["coupon_pa"] > base
    assert solve_coupon(Phoenix(), replace(PAIR, funding_spread=0.0))["coupon_pa"] < base


def test_redemption_profile_is_a_distribution():
    q = solve_coupon(Phoenix(), PAIR)
    assert q["redemption_profile"].sum() == pytest.approx(1.0)
    assert q["prob_autocall"] == pytest.approx(1 - q["redemption_profile"].iloc[-1])


def test_scenarios():
    p = Phoenix()
    up = scenario_outcome(p, 0.03, scenario_path(p, 1.3))
    assert up["Redeemed after (years)"] == 1.0 and up["Capital returned (%)"] == 100
    assert up["Coupons received (%)"] == pytest.approx(4 * 3.0)
    crash = scenario_outcome(p, 0.03, scenario_path(p, 0.4))
    assert crash["Capital returned (%)"] == pytest.approx(40.0)
    w = scenario_path(p, 0.4)
    assert crash["Coupons received (%)"] == pytest.approx(3.0 * (w >= 0.6).sum())   # with memory, each paid date settles all


def test_solve_barrier_hits_the_target():
    b = solve_barrier(Phoenix(), PAIR, 0.10, n_paths=10_000)
    q = solve_coupon(Phoenix(coupon_barrier=b, protection_barrier=b), PAIR, n_paths=10_000)
    assert q["coupon_pa"] == pytest.approx(0.10, abs=0.002)


def test_term_sheet_is_one_page(tmp_path):
    from pypdf import PdfReader
    p = Phoenix()
    q = solve_coupon(p, PAIR)
    sc = pd.DataFrame([scenario_outcome(p, q["coupon_per_period"], scenario_path(p, f)) for f in (1.2, 0.8, 0.4)],
                      index=["Up", "Down", "Crash"])
    out = term_sheet_pdf(str(tmp_path / "ts.pdf"), p, PAIR, q, sc)
    assert len(PdfReader(out).pages) == 1
