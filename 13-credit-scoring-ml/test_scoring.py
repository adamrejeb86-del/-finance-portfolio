import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from scoring import (WoEEncoder, assign_grade, calibration_table, generate_applications, gini, ks_statistic,
                     population_stability_index, score_from_pd, scorecard_points)

DF = generate_applications(20_000, seed=1)
X, Y = DF.drop(columns="default"), DF["default"]


def test_woe_sign_and_information_value():
    enc = WoEEncoder().fit(X, Y)
    for col in ("late_payments", "dti", "utilisation"):
        t = enc.tables[col]
        # a riskier bin (higher default rate) always has a lower WoE
        assert t.sort_values("default_rate")["woe"].is_monotonic_decreasing
    iv = enc.information_values()
    assert iv["late_payments"] > 0.1 and iv.min() >= 0


def test_missing_values_get_their_own_bin():
    enc = WoEEncoder().fit(X, Y)
    assert "missing" in enc.tables["income"].index
    assert enc.transform(X)["income"].notna().all()


def test_gini_and_ks_definitions():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 5000)
    p = rng.random(5000)
    assert gini(y, p) == pytest.approx(2 * roc_auc_score(y, p) - 1)
    assert gini(y, y) == pytest.approx(1.0) and ks_statistic(y, y) == pytest.approx(1.0)


def test_scorecard_reproduces_the_logistic_model():
    enc = WoEEncoder().fit(X, Y)
    W = enc.transform(X)
    lr = LogisticRegression(max_iter=1000).fit(W, Y)
    pts = scorecard_points(dict(zip(W.columns, lr.coef_[0])), lr.intercept_[0], enc)
    lookup = {(r.feature, r.bin): r.points for r in pts.itertuples()}
    total = sum(enc._bin(c, X[c]).map(lambda b, c=c: lookup[(c, b)]) for c in W.columns)
    expected = score_from_pd(lr.predict_proba(W)[:, 1])
    assert np.allclose(total, expected, atol=1e-6)


def test_model_is_calibrated_on_average():
    enc = WoEEncoder().fit(X, Y)
    lr = LogisticRegression(max_iter=1000).fit(enc.transform(X), Y)
    p = lr.predict_proba(enc.transform(X))[:, 1]
    assert p.mean() == pytest.approx(Y.mean(), abs=0.002)
    cal = calibration_table(Y, p)
    assert cal["observed_dr"].corr(cal["predicted_pd"], method="spearman") > 0.9


def test_psi_detects_a_population_shift():
    rng = np.random.default_rng(2)
    base = rng.normal(600, 50, 10_000)
    assert population_stability_index(base, rng.normal(600, 50, 10_000)) < 0.02
    assert population_stability_index(base, rng.normal(560, 50, 10_000)) > 0.25


def test_master_scale_covers_all_pds():
    pds = np.array([0.0001, 0.001, 0.01, 0.05, 0.3, 0.99])
    assert all(g is not None for g in assign_grade(pds))
