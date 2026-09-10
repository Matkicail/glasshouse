"""The panel always carries a naive row, and the naive row scores what 'no model' should."""

from __future__ import annotations

import json

import numpy as np
import pytest

from glasshouse.metrics import deviance_per_row
from glasshouse.scorecard import HIGHER_IS_BETTER, compare, naive_prediction, scorecard

rng = np.random.default_rng(5)
N = 2000
EXPOSURE = rng.uniform(0.1, 1.0, size=N)
RATE = rng.gamma(2.0, 0.1, size=N)
COUNT = rng.poisson(RATE * EXPOSURE).astype(float)
PROB = rng.beta(0.6, 3.0, size=N)
LABEL = (rng.uniform(size=N) < PROB).astype(float)


def test_naive_row_is_the_no_model_baseline_for_a_glm() -> None:
    card = scorecard(COUNT / EXPOSURE, RATE, family="poisson", sample_weight=EXPOSURE)
    assert card.naive["d2"] == pytest.approx(0.0, abs=1e-12)  # null model explains nothing
    assert card.naive["r2"] == pytest.approx(0.0, abs=1e-12)
    assert card.naive["gini"] == pytest.approx(0.0, abs=1e-12)  # a constant ranks nothing
    assert card.naive["balance"] == pytest.approx(1.0, rel=1e-12)  # the mean is balanced
    assert card.metrics["d2"] > card.naive["d2"]  # the true rate beats the mean
    assert card.naive_prediction == pytest.approx(np.average(COUNT / EXPOSURE, weights=EXPOSURE))
    assert set(card.metrics) == set(card.naive)


def test_naive_row_for_classification_is_the_class_prior() -> None:
    card = scorecard(LABEL, PROB, family="binomial")
    prior = LABEL.mean()
    assert card.naive_prediction == pytest.approx(prior)
    assert card.naive["roc_auc"] == pytest.approx(0.5)  # constant score = coin flip ranking
    assert card.naive["average_precision"] == pytest.approx(prior)  # AP baseline = prevalence
    assert card.naive["brier"] == pytest.approx(prior * (1 - prior))
    assert card.metrics["log_loss"] < card.naive["log_loss"]
    assert card.metrics["mcc"] > card.naive["mcc"] == 0.0  # never flags at 0.5 → undefined → 0


def test_gaussian_target_with_negatives_reports_nan_gini_not_an_error() -> None:
    y = rng.normal(size=500)
    card = scorecard(y, y + rng.normal(scale=0.5, size=500), family="gaussian")
    assert np.isnan(card.metrics["gini"])
    assert card.metrics["r2"] > 0.5


def test_deviance_per_row_is_the_cann_papers_formula_and_only_on_weighted_panels() -> None:
    """(2/n) sum [v mu - Y - Y log(v mu / Y)] on counts, Y log(.) = 0 at Y = 0 (W&M 5.28)."""
    v, mu = EXPOSURE, RATE
    expected = (
        v * mu
        - COUNT
        - np.where(COUNT > 0, COUNT * np.log(v * mu / np.where(COUNT > 0, COUNT, 1)), 0)
    )
    expected = 2.0 * expected.sum() / N
    got = deviance_per_row(COUNT / v, mu, family="poisson", sample_weight=v)
    assert got == pytest.approx(expected, rel=1e-12)
    card = scorecard(COUNT / v, mu, family="poisson", sample_weight=v)
    assert card.metrics["deviance_per_row"] == pytest.approx(expected, rel=1e-12)
    assert card.metrics["deviance_per_row"] == pytest.approx(card.metrics["deviance"] * v.sum() / N)
    assert "deviance_per_row" not in scorecard(COUNT, RATE * v, family="poisson").metrics
    assert deviance_per_row([0, 1, 2], [0.5, 1, 2.5], family="poisson") == pytest.approx(
        deviance_per_row([0, 1, 2], [0.5, 1, 2.5], family="poisson", sample_weight=[1, 1, 1])
    )


def test_compare_is_direction_aware_and_refuses_mismatches() -> None:
    good = scorecard(COUNT, RATE * EXPOSURE, family="poisson", label="good")
    bad = scorecard(COUNT, np.full(N, COUNT.mean()), family="poisson", label="bad")
    cmp = compare(good, bad)
    by_name = {name: winner for name, _, _, winner in cmp.rows}
    assert by_name["deviance"] == "good"  # lower is better and good is lower
    assert by_name["d2"] == "good"
    # the mean is exactly balanced; 'good' is a rate times exposure, off by fold noise only,
    # and a tenth of a percent is a tie, not a loss
    assert by_name["balance"] == ("tie" if abs(good.metrics["balance"] - 1) <= 1e-3 else "bad")
    far = scorecard(COUNT, 1.5 * RATE * EXPOSURE, family="poisson", label="far")
    assert {n: w for n, _, _, w in compare(far, bad).rows}["balance"] == "bad"
    assert "good" in str(cmp) and "metric" in str(cmp)
    with pytest.raises(ValueError, match="same family"):
        compare(good, scorecard(LABEL, PROB, family="binomial"))
    with pytest.raises(ValueError, match="same split"):
        compare(good, scorecard(COUNT[:10], RATE[:10], family="poisson"))


def test_card_prints_and_serialises() -> None:
    card = scorecard(COUNT, RATE * EXPOSURE, family="poisson", label="glm")
    text = str(card)
    assert "glm vs naive" in text and "better?" in text and "d2" in text
    payload = json.dumps(card.to_dict())
    back = json.loads(payload)
    assert back["metrics"]["d2"] == pytest.approx(card.metrics["d2"])
    assert len(back["calibration"]["actual"]) == len(card.calibration)


def test_every_scored_metric_has_a_direction() -> None:
    card = scorecard(COUNT, RATE * EXPOSURE, family="poisson")
    for name in card.metrics:
        assert name in HIGHER_IS_BETTER or name == "balance", name
    card = scorecard(LABEL, PROB, family="binomial")
    for name in card.metrics:
        assert name in HIGHER_IS_BETTER or name == "balance", name


def test_naive_prediction_is_weighted_mean() -> None:
    assert naive_prediction([1, 2, 3], family="gaussian", sample_weight=[1, 1, 2]) == 2.25
