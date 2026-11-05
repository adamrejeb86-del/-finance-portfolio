import numpy as np
import pytest

from fxopt import (atm_dns_strike, delta, forward, gk_price, implied_vol, pillar_vols, strike_from_delta,
                   vanna_volga_price, vanna_volga_vol)

S, T, RD, RF, VOL = 1.0850, 0.5, 0.040, 0.025, 0.08     # EUR/USD-like: domestic USD, foreign EUR


def test_put_call_parity_and_forward():
    K = 1.10
    assert gk_price(S, K, T, RD, RF, VOL) - gk_price(S, K, T, RD, RF, VOL, call=False) == pytest.approx(
        S * np.exp(-RF * T) - K * np.exp(-RD * T), abs=1e-14)
    assert forward(S, T, RD, RF) > S                              # higher domestic rate -> forward premium


@pytest.mark.parametrize("kind", ["spot", "forward"])
@pytest.mark.parametrize("pa", [False, True])
def test_strike_from_delta_round_trip(kind, pa):
    for call, target in [(True, 0.25), (False, -0.25)]:
        K = strike_from_delta(target, S, T, RD, RF, VOL, call, kind, pa)
        assert delta(S, K, T, RD, RF, VOL, call, kind, pa) == pytest.approx(target, abs=1e-9)


def test_atm_dns_straddle_is_delta_neutral():
    for pa in (False, True):
        K = atm_dns_strike(S, T, RD, RF, VOL, pa)
        total = delta(S, K, T, RD, RF, VOL, True, "spot", pa) + delta(S, K, T, RD, RF, VOL, False, "spot", pa)
        assert total == pytest.approx(0.0, abs=1e-10)


def test_vanna_volga_reproduces_the_three_pillars():
    vols = pillar_vols(0.08, -0.006, 0.0025)
    Ks = [strike_from_delta(-0.25, S, T, RD, RF, vols["25P"], False), atm_dns_strike(S, T, RD, RF, 0.08),
          strike_from_delta(0.25, S, T, RD, RF, vols["25C"], True)]
    sv = [vols["25P"], vols["ATM"], vols["25C"]]
    for K, v in zip(Ks, sv):
        assert vanna_volga_vol(K, Ks, sv) == pytest.approx(v, abs=1e-12)
        assert implied_vol(vanna_volga_price(S, K, T, RD, RF, Ks, sv), S, K, T, RD, RF) == pytest.approx(v, abs=1e-8)


def test_flat_quotes_give_a_flat_smile():
    Ks = [1.03, 1.09, 1.15]
    for K in (1.00, 1.08, 1.20):
        assert implied_vol(vanna_volga_price(S, K, T, RD, RF, Ks, [0.08] * 3), S, K, T, RD, RF) == pytest.approx(0.08, abs=1e-9)
