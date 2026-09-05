"""Win sets and the tournament: golden vs a NumPy write-out, partition properties, plumbing."""

from __future__ import annotations

import numpy as np
import pytest

from glasshouse import report, tournament

rng = np.random.default_rng(17)
N = 4000
EXPO = rng.uniform(0.1, 1.0, size=N)
RATE = np.exp(rng.normal(-2.0, 0.5, size=N))
Y = rng.poisson(RATE * EXPO) / EXPO
PREDS = {
    "sharp": RATE * rng.lognormal(0.0, 0.05, size=N),
    "blunt": RATE * rng.lognormal(0.0, 0.5, size=N),
    "flat": np.full(N, RATE.mean()),
}


def _reference(
    y: np.ndarray, preds: dict[str, np.ndarray], w: np.ndarray
) -> dict[str, list[float]]:
    """The definition, written out: cheapest wins, exact ties split equally."""
    stack = np.column_stack(list(preds.values()))
    lowest = stack.min(axis=1, keepdims=True)
    wins = (stack == lowest).astype(float)
    wins /= wins.sum(axis=1, keepdims=True)
    return {
        "weight": (wins * w[:, None]).sum(axis=0).tolist(),
        "predicted": (wins * w[:, None] * stack).sum(axis=0).tolist(),
        "actual": (wins * w[:, None] * y[:, None]).sum(axis=0).tolist(),
    }


def test_golden_vs_numpy_write_out() -> None:
    t = tournament.tournament(Y, PREDS, EXPO)
    ref = _reference(Y, PREDS, EXPO)
    for key in ("weight", "predicted", "actual"):
        np.testing.assert_allclose(getattr(t, key), ref[key], rtol=1e-12)
    assert t.labels == ["sharp", "blunt", "flat"]


def test_win_sets_partition_the_market_and_ties_split() -> None:
    t = tournament.tournament(Y, PREDS, EXPO)
    assert t.share.sum() == pytest.approx(1.0)
    assert t.weight.sum() == pytest.approx(EXPO.sum())
    assert t.actual.sum() == pytest.approx((EXPO * Y).sum())
    # the sharp model wins less market than the blunt one but at a healthier A/E: the blunt
    # model wins exactly the rows it under-prices
    assert t.actual_over_expected[1] > t.actual_over_expected[0]
    # identical prices: every row is a tie and the market halves
    same = tournament.tournament(Y, {"a": PREDS["sharp"], "b": PREDS["sharp"]}, EXPO)
    np.testing.assert_allclose(same.share, [0.5, 0.5])
    np.testing.assert_allclose(same.actual, [(EXPO * Y).sum() / 2] * 2)


def test_pairwise_is_the_two_model_tournament_and_serialises() -> None:
    pair = tournament.win_sets(Y, PREDS["sharp"], PREDS["blunt"], EXPO, "sharp", "blunt")
    both = tournament.tournament(Y, {"sharp": PREDS["sharp"], "blunt": PREDS["blunt"]}, EXPO)
    np.testing.assert_allclose(pair.weight, both.weight)
    d = pair.to_dict()
    assert set(d) == {
        "labels",
        "share",
        "weight",
        "predicted",
        "actual",
        "profit",
        "actual_over_expected",
    }
    np.testing.assert_allclose(d["profit"], pair.predicted - pair.actual)
    assert "sharp" in str(pair)


def test_fails_early() -> None:
    with pytest.raises(ValueError, match="at least one model"):
        tournament.tournament(Y, {}, EXPO)
    with pytest.raises(ValueError, match="length"):
        tournament.tournament(Y, {"a": PREDS["sharp"][:-1]}, EXPO)
    with pytest.raises(ValueError, match="weights"):
        tournament.tournament(Y, PREDS, -EXPO)


def test_report_carries_the_block_for_priced_tasks_only() -> None:
    freq = report.build("frequency", Y, PREDS, weight=EXPO).to_dict()
    report.validate(freq)
    assert {p["labels"][0] + "|" + p["labels"][1] for p in freq["tournament"]["pairs"]} == {
        "sharp|blunt",
        "sharp|flat",
        "blunt|flat",
    }
    assert freq["tournament"]["overall"]["labels"] == ["sharp", "blunt", "flat"]
    assert sum(freq["tournament"]["overall"]["share"]) == pytest.approx(1.0)
    labels = (rng.uniform(size=N) < 0.1).astype(float)
    binary = report.build("binary", labels, {"p": np.clip(PREDS["sharp"], 0.01, 0.99)}).to_dict()
    assert "tournament" not in binary
