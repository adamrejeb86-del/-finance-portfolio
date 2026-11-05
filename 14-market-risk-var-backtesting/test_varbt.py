import numpy as np
import pandas as pd
import pytest

from varbt import (acerbi_szekely_z2, christoffersen, kupiec_pof, reverse_stress, rolling_var_es, simulate_market,
                   stress_pnl, traffic_light, z2_critical_value)


def test_kupiec_is_zero_at_the_expected_rate_and_rejects_too_many():
    assert kupiec_pof(10, 1000)[0] == pytest.approx(0.0, abs=1e-10)
    assert kupiec_pof(25, 1000)[1] < 0.01


def test_christoffersen_detects_clustering():
    spread = np.zeros(1000, int); spread[::100] = 1
    clustered = np.zeros(1000, int); clustered[500:510] = 1
    assert christoffersen(spread)["p_ind"] > 0.5
    assert christoffersen(clustered)["p_ind"] < 0.01


def test_traffic_light_zones():
    assert traffic_light(3) == ("green", 3.0)
    assert traffic_light(7) == ("yellow", 3.65)
    assert traffic_light(12) == ("red", 4.0)


def test_models_pass_on_iid_normal_data():
    rng = np.random.default_rng(1)
    pnl = pd.Series(rng.standard_normal(3000))
    var_f, _ = rolling_var_es(pnl, "normal")
    hits = (pnl < -var_f)[var_f.notna()]
    assert kupiec_pof(hits.sum(), len(hits))[1] > 0.05


def test_historical_var_is_the_empirical_quantile():
    rng = np.random.default_rng(2)
    pnl = pd.Series(rng.standard_t(4, 600))
    var_f, es_f = rolling_var_es(pnl, "historical")
    assert var_f.iloc[400] == pytest.approx(-np.quantile(pnl.values[150:400], 0.01))
    assert es_f.iloc[400] >= var_f.iloc[400]


def test_z2_is_centred_for_a_correct_model():
    rng = np.random.default_rng(3)
    from scipy.stats import norm
    q = norm.ppf(0.025)
    zs = [acerbi_szekely_z2(rng.standard_normal(2000), np.full(2000, -q), np.full(2000, norm.pdf(q) / 0.025)) for _ in range(300)]
    assert abs(np.mean(zs)) < 0.03
    assert z2_critical_value(n_sims=500) < 0


def test_market_has_fat_tails_and_stress_tests_are_linear():
    r = simulate_market(2000, seed=4)
    assert (r["Equities"].kurt()) > 1.0
    exp = np.array([50, 30, 10, 5, 5.0])
    assert stress_pnl(exp, [-0.1] * 5) == pytest.approx(-10.0)
    k = reverse_stress(exp, np.array([-0.1, 0, 0, 0, 0]), 25.0)
    assert stress_pnl(exp, k * np.array([-0.1, 0, 0, 0, 0])) == pytest.approx(-25.0)
