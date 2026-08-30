"""Curves are data that agree with the metrics, serialise, and render."""

from __future__ import annotations

import json

import numpy as np
import pytest

from glasshouse import curves, metrics, plots

rng = np.random.default_rng(9)
N = 3000
EXPO = rng.uniform(0.1, 1.0, size=N)
RATE = rng.gamma(2.0, 0.1, size=N)
Y = rng.poisson(RATE * EXPO).astype(float) / EXPO
MU_A = RATE * rng.lognormal(0, 0.2, size=N)
MU_B = np.full(N, RATE.mean())


def test_lorenz_integrates_to_the_gini_and_is_a_real_curve() -> None:
    c = curves.lorenz(Y, MU_A, EXPO, label="a")
    assert (c.x[0], c.y[0], c.x[-1], c.y[-1]) == (0.0, 0.0, 1.0, 1.0)
    assert np.all(np.diff(c.x) >= 0) and np.all(np.diff(c.y) >= 0)
    area = np.trapezoid(c.y, c.x)
    assert 1 - 2 * area == pytest.approx(c.gini, abs=1e-12)
    assert c.gini == pytest.approx(metrics.gini(Y, MU_A, EXPO))


def test_lift_and_calibration_are_the_calibration_table() -> None:
    t = metrics.calibration_table(Y, MU_A, sample_weight=EXPO)
    li = curves.lift(Y, MU_A, EXPO)
    ca = curves.calibration(Y, MU_A, EXPO)
    np.testing.assert_array_equal(li.actual, t.actual)
    np.testing.assert_array_equal(ca.actual_over_expected, t.actual_over_expected)
    assert li.weight.sum() == pytest.approx(EXPO.sum())


def test_double_lift_bins_partition_and_separate_the_models() -> None:
    d = curves.double_lift(Y, MU_A, MU_B, EXPO, label_a="glm", label_b="mean")
    assert d.weight.sum() == pytest.approx(EXPO.sum())
    assert np.all(np.diff(d.ratio) >= 0)  # sorted by a / b
    np.testing.assert_allclose(d.predicted_b, RATE.mean())  # b is constant everywhere
    # a tracks the truth better than the constant at both ends of the disagreement
    err_a = np.abs(d.predicted_a - d.actual)
    err_b = np.abs(d.predicted_b - d.actual)
    assert err_a[0] < err_b[0] and err_a[-1] < err_b[-1]
    with pytest.raises(ValueError, match="must be positive"):
        curves.double_lift(Y, MU_A, np.zeros(N))


def test_curves_serialise_to_the_json_contract() -> None:
    payload = [
        curves.lorenz(Y, MU_A, EXPO).to_dict(),
        curves.lift(Y, MU_A, EXPO).to_dict(),
        curves.double_lift(Y, MU_A, MU_B, EXPO).to_dict(),
        curves.calibration(Y, MU_A, EXPO).to_dict(),
    ]
    back = json.loads(json.dumps(payload))
    assert [p["kind"] for p in back] == ["lorenz", "lift", "double_lift", "calibration"]
    assert back[0]["gini"] == pytest.approx(metrics.gini(Y, MU_A, EXPO))


def test_plots_render_from_the_data() -> None:
    lo = curves.lorenz(Y, MU_A, EXPO, label="a")
    figs = [
        plots.lorenz(lo, curves.lorenz(Y, MU_B, EXPO, label="b")),
        plots.lift(curves.lift(Y, MU_A, EXPO)),
        plots.double_lift(curves.double_lift(Y, MU_A, MU_B, EXPO)),
        plots.calibration(curves.calibration(Y, MU_A, EXPO)),
    ]
    for fig in figs:
        assert len(fig.data) >= 2
        assert "<div" in fig.to_html()
