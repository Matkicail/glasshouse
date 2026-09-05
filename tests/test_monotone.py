"""Monotone spline terms: an exact golden by enumeration, the shape property, the plumbing.

The reference for the constrained fit is the constrained problem itself. For a gaussian
identity model the penalised IRLS fixed point is the quadratic programme
``min ||y - X b||^2 + lam b'Sb  s.t. A b >= 0``, and with a handful of constraints its
solution can be found exactly by trying every active set: solve the equality-constrained
problem for each subset of constraints and keep the one that is primal and dual feasible.
No approximation, no tolerance to argue about.
"""

from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd
import pytest

from glasshouse import GLM
from glasshouse.encoders import BSpline, Monotone, Smooth

rng = np.random.default_rng(31)


def _kkt_enumeration(
    design: np.ndarray, y: np.ndarray, penalty: np.ndarray, a_mat: np.ndarray
) -> np.ndarray:
    """Solve ``min ||y - Xb||^2 + b'Sb  s.t. A b >= 0`` exactly by trying every active set."""
    hess = 2.0 * (design.T @ design + penalty)
    grad0 = -2.0 * design.T @ y
    n_con = a_mat.shape[0]
    for size in range(n_con + 1):
        for active in itertools.combinations(range(n_con), size):
            rows = a_mat[list(active)]
            k = len(active)
            kkt = np.block([[hess, -rows.T], [rows, np.zeros((k, k))]]) if k else hess
            rhs = np.concatenate([-grad0, np.zeros(k)]) if k else -grad0
            try:
                sol = np.linalg.solve(kkt, rhs)
            except np.linalg.LinAlgError:
                continue
            beta, mu = sol[: len(grad0)], sol[len(grad0) :]
            if np.all(a_mat @ beta >= -1e-9) and np.all(mu >= -1e-9):
                return beta
    msg = "no active set satisfied the KKT conditions"
    raise AssertionError(msg)


def test_golden_by_active_set_enumeration() -> None:
    n = 400
    x = rng.uniform(0.0, 1.0, size=n)
    y = np.sin(2.5 * np.pi * x) + 0.4 * x + rng.normal(0.0, 0.3, size=n)  # rises, dips, rises
    lam = 2.0
    enc = Smooth(df=6, lam=lam, monotone="increasing", name="x").fit(x)
    basis, _ = enc.transform(x)
    design = np.column_stack([np.ones(n), basis])
    p = design.shape[1]
    penalty = np.zeros((p, p))
    penalty[1:, 1:] = lam * enc.penalty_matrix()
    # constraints: b_1 >= 0 (anchor), then b_{j+1} - b_j >= 0 over the smooth's columns
    a_mat = np.zeros((p - 1, p))
    a_mat[0, 1] = 1.0
    for j in range(1, p - 1):
        a_mat[j, j + 1], a_mat[j, j] = 1.0, -1.0
    expected = _kkt_enumeration(design, y, penalty, a_mat)
    assert np.any(a_mat @ expected < 1e-9), "the constraint should bind on this data"
    m = GLM(family="gaussian", terms={"x": Smooth(df=6, lam=lam, monotone="increasing")}).fit(
        pd.DataFrame({"x": x}), y
    )
    got = np.concatenate([[m.intercept_], m.coef_])
    np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-11)


def _curve(model: GLM, grid: np.ndarray) -> np.ndarray:
    return model.predict(pd.DataFrame({"x": grid}))


@pytest.mark.parametrize("direction", ["increasing", "decreasing"])
def test_fitted_curve_is_monotone_and_the_constraint_binds(direction: Monotone) -> None:
    n = 3000
    x = rng.uniform(0.0, 10.0, size=n)
    sign = 1.0 if direction == "increasing" else -1.0
    eta = -2.0 + sign * (0.25 * x - 0.6 * np.sin(x))  # mostly monotone with real dips
    y = rng.poisson(np.exp(eta)).astype(float)
    df = pd.DataFrame({"x": x})
    free = GLM(family="poisson", terms={"x": "smooth"}).fit(df, y)
    held = GLM(family="poisson", terms={"x": Smooth(monotone=direction)}).fit(df, y)
    grid = np.linspace(0.0, 10.0, 801)
    steps = sign * np.diff(_curve(held, grid))
    assert np.all(steps >= -1e-12), f"the fitted curve must be {direction}"
    assert np.any(sign * np.diff(_curve(free, grid)) < 0.0), "the free smooth does dip"
    assert held.deviance_ >= free.deviance_
    assert held.edf_ < free.edf_ + 1e-9
    assert held.lambda_["x"] > 0.0 and len(held.gcv_["x"]) == 32
    # the intercept is unconstrained, so the fit stays balanced
    np.testing.assert_allclose(held.predict(df).sum(), y.sum(), rtol=1e-8)


def test_constraint_is_inactive_when_the_free_fit_is_already_monotone() -> None:
    # a quadratic is exactly a cubic spline, so the free fit recovers it with monotone
    # coefficients (no noise: with noise the boundary coefficients can dip even when the
    # truth rises, and then the constraint rightly binds)
    x = np.linspace(0.0, 1.0, 300)
    y = 1.0 + 2.0 * x + 0.5 * x**2
    df = pd.DataFrame({"x": x})
    free = GLM(family="gaussian", terms={"x": BSpline(df=5)}).fit(df, y)
    assert np.all(np.diff(free.coef_) > 0.0) and free.coef_[0] > 0.0, free.coef_
    held = GLM(family="gaussian", terms={"x": BSpline(df=5, monotone="increasing")}).fit(df, y)
    np.testing.assert_allclose(held.coef_, free.coef_, rtol=1e-9, atol=1e-12)
    assert held.edf_ == pytest.approx(free.edf_)


def test_round_trips_and_fails_early() -> None:
    n = 500
    x = rng.uniform(0.0, 1.0, size=n)
    y = rng.poisson(np.exp(-1.0 + np.sin(4.0 * x))).astype(float)
    df = pd.DataFrame({"x": x})
    m = GLM(family="poisson", terms={"x": Smooth(df=7, monotone="decreasing")}).fit(df, y)
    back = GLM.from_dict(json.loads(json.dumps(m.to_dict())))
    assert back.encoders_["x"].to_dict()["monotone"] == "decreasing"
    np.testing.assert_allclose(back.predict(df), m.predict(df), rtol=1e-12)
    with pytest.raises(ValueError, match="monotone must be"):
        Smooth(monotone="up").fit(x)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="monotone must be"):
        GLM(family="poisson", terms={"x": BSpline(monotone="flat")}).fit(df, y)  # type: ignore[arg-type]
