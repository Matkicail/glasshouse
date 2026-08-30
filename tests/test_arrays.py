"""The one data door: every container comes through the same, and bad data is named."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from glasshouse.arrays import to_matrix, to_vector
from glasshouse.metrics import poisson_deviance

VALUES = [1.0, 2.5, 3.0]


@pytest.mark.parametrize(
    "container",
    [
        VALUES,
        np.array(VALUES),
        np.array(VALUES, dtype=np.float32),
        np.array([1, 2, 3]),
        pd.Series(VALUES, name="y"),
        pl.Series("y", VALUES),
        pa.array(VALUES),
        pa.chunked_array([VALUES[:1], VALUES[1:]]),
        pd.DataFrame({"y": VALUES}),
    ],
    ids=[
        "list",
        "numpy",
        "float32",
        "int",
        "pandas",
        "polars",
        "arrow",
        "arrow-chunked",
        "1col-frame",
    ],
)
def test_every_container_becomes_the_same_vector(container: object) -> None:
    v = to_vector(container, "y")
    assert v.dtype == np.float64 and v.flags.c_contiguous and v.ndim == 1
    np.testing.assert_array_equal(v, np.asarray(np.asarray(container), dtype=np.float64).ravel())


def test_metrics_go_through_the_same_door() -> None:
    y = pl.Series("y", [0.0, 1.0, 2.0])
    mu = pd.Series([0.5, 1.0, 2.5])
    assert poisson_deviance(y, mu) == poisson_deviance([0.0, 1.0, 2.0], [0.5, 1.0, 2.5])


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({"age": [20, 30], "power": [1.5, 2.5]}),
        pl.DataFrame({"age": [20, 30], "power": [1.5, 2.5]}),
        pa.table({"age": [20, 30], "power": [1.5, 2.5]}),
    ],
    ids=["pandas", "polars", "arrow"],
)
def test_frames_keep_their_column_names(frame: object) -> None:
    matrix, names = to_matrix(frame)
    assert names == ["age", "power"]
    assert matrix.shape == (2, 2) and matrix.dtype == np.float64 and matrix.flags.c_contiguous
    np.testing.assert_array_equal(matrix, [[20.0, 1.5], [30.0, 2.5]])


def test_numpy_matrix_gets_generated_names_and_1d_becomes_a_column() -> None:
    matrix, names = to_matrix(np.arange(6).reshape(3, 2))
    assert names == ["x0", "x1"] and matrix.shape == (3, 2)
    column, names1 = to_matrix([1.0, 2.0, 3.0])
    assert column.shape == (3, 1) and names1 == ["x0"]


@pytest.mark.parametrize(
    ("bad", "needle"),
    [
        (["a", "b"], "strings"),
        (pd.Series(["a", None]), "object"),
        (pd.Series(pd.to_datetime(["2024-01-01", "2024-01-02"])), "datetime"),
        ([1.0, np.nan], "1 missing value"),
        ([1.0, np.inf], "1 infinite value"),
        ([[1.0, 2.0], [3.0, 4.0]], "1-D"),
    ],
    ids=["str-list", "object", "datetime", "nan", "inf", "2d"],
)
def test_vector_refuses_with_reasons(bad: object, needle: str) -> None:
    with pytest.raises(ValueError, match=needle):
        to_vector(bad, "col")


def test_nan_can_be_allowed_explicitly_but_inf_never() -> None:
    v = to_vector([1.0, np.nan], "col", allow_nan=True)
    assert np.isnan(v[1])
    with pytest.raises(ValueError, match="infinite"):
        to_vector([1.0, np.inf], "col", allow_nan=True)


def test_matrix_names_every_bad_column_at_once() -> None:
    df = pd.DataFrame({"ok": [1.0, 2.0], "region": ["a", "b"], "amount": [1.0, np.nan]})
    with pytest.raises(ValueError, match="2 unusable column") as err:
        to_matrix(df)
    text = str(err.value)
    assert "region" in text and "amount" in text and "ok" not in text.split("\n", maxsplit=1)[0]


def test_polars_nulls_are_missing_values() -> None:
    with pytest.raises(ValueError, match="missing value"):
        to_vector(pl.Series("y", [1.0, None]), "y")
