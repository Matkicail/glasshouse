"""Data in: anything array-ish becomes a clean float64 array, or you are told exactly why not.

This is the one door every model and metric uses. It accepts lists, NumPy arrays, pandas
and Polars Series / DataFrames, and Arrow arrays / tables — by duck-typing on ``to_numpy``
and ``__array__``, so none of those libraries is a dependency. It never coerces silently:
strings, objects, NaNs and infinities are refused with the column name, the row count, and
what to do about it. Family-specific rules (``y >= 0`` for Poisson, and so on) are checked
in the Rust core, not here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

ArrayLike = Any  # anything ``np.asarray`` accepts: list, NumPy, pandas/Polars Series, pyarrow
F64 = npt.NDArray[np.float64]

_ALLOWED_KINDS = frozenset("fiub")  # float, signed int, unsigned int, bool


def to_vector(x: ArrayLike, name: str = "x", *, allow_nan: bool = False) -> F64:
    """Return ``x`` as a contiguous 1-D float64 array.

    Parameters
    ----------
    x : array-like
        A list, NumPy array, pandas or Polars Series, Arrow array, or a single-column frame.
    name : str
        What to call it in error messages (``"y"``, ``"Exposure"`` ...).
    allow_nan : bool
        Let NaN through (for a column that will be encoded or imputed later). Infinity is
        never allowed.

    Raises
    ------
    ValueError
        Not 1-D; non-numeric dtype (strings, objects, dates); NaN or inf, with the count.

    Examples
    --------
    >>> from glasshouse.arrays import to_vector
    >>> to_vector([1, 2, 3], "y")
    array([1., 2., 3.])
    >>> to_vector(["a", "b"], "Region")  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: Region has dtype str (strings): encode categoricals first ...
    """
    arr = _to_numpy(x, name)
    if arr.ndim == 2 and arr.shape[1] == 1:  # noqa: PLR2004 — a single-column frame
        arr = arr[:, 0]
    if arr.ndim != 1:
        msg = f"{name} must be 1-D, got shape {arr.shape}; pass one column at a time"
        raise ValueError(msg)
    out = np.ascontiguousarray(arr, dtype=np.float64)
    _check_finite(out, name, allow_nan=allow_nan)
    return out


def to_matrix(x: ArrayLike, name: str = "X", *, allow_nan: bool = False) -> tuple[F64, list[str]]:
    """Return ``x`` as a C-contiguous 2-D float64 array plus its column names.

    A DataFrame (pandas, Polars, Arrow table) keeps its column names; a NumPy array gets
    ``x0, x1, ...``. Every column must be numeric — the message names the offending columns.

    Examples
    --------
    >>> import numpy as np
    >>> from glasshouse.arrays import to_matrix
    >>> matrix, names = to_matrix(np.array([[1, 2], [3, 4]]))
    >>> names
    ['x0', 'x1']
    """
    columns_ = columns(x)
    if columns_ is None:  # not a frame: treat as an array
        arr = _to_numpy(x, name)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim != 2:  # noqa: PLR2004
            msg = f"{name} must be 2-D (rows x features), got shape {arr.shape}"
            raise ValueError(msg)
        out = np.ascontiguousarray(arr, dtype=np.float64)
        names = [f"x{i}" for i in range(out.shape[1])]
        _check_finite(out, name, allow_nan=allow_nan)
        return out, names

    names = [str(c) for c, _ in columns_]
    bad: list[str] = []
    vectors: list[F64] = []
    for col_name, col in columns_:
        try:
            vectors.append(to_vector(col, str(col_name), allow_nan=allow_nan))
        except ValueError as err:
            bad.append(f"{col_name}: {err}")
    if bad:
        msg = f"{name} has {len(bad)} unusable column(s):\n  " + "\n  ".join(bad)
        raise ValueError(msg)
    if not vectors:
        msg = f"{name} has no columns"
        raise ValueError(msg)
    return np.ascontiguousarray(np.column_stack(vectors), dtype=np.float64), names


def _to_numpy(x: ArrayLike, name: str) -> npt.NDArray[Any]:
    """Duck-typed conversion: ``to_numpy()`` (pandas, Polars, Arrow) or ``__array__`` / lists."""
    arr = np.asarray(x.to_numpy()) if hasattr(x, "to_numpy") else np.asarray(x)
    kind = arr.dtype.kind
    if kind not in _ALLOWED_KINDS:
        what = {
            "U": "str (strings)",
            "S": "bytes",
            "O": "object (mixed or strings)",
            "M": "datetime",
            "m": "timedelta",
        }.get(kind, str(arr.dtype))
        fix = (
            "encode categoricals first (one-hot or target encoding)"
            if kind in "USO"
            else "convert to a number (e.g. days since a date) before modelling"
        )
        msg = f"{name} has dtype {what}: {fix}"
        raise ValueError(msg)
    return arr


def columns(x: ArrayLike) -> list[tuple[str, Any]] | None:
    """Return the (name, column) pairs of a frame-like object, or ``None`` if not a frame."""
    if isinstance(x, np.ndarray):
        return None
    if hasattr(x, "column_names") and hasattr(x, "column"):  # pyarrow.Table
        return [(n, x.column(n)) for n in x.column_names]
    if hasattr(x, "columns") and hasattr(x, "__getitem__") and not hasattr(x, "to_frame"):
        names = list(x.columns)  # pandas / Polars DataFrame (a Series has no .columns)
        return [(n, x[n]) for n in names]
    return None


def _check_finite(arr: F64, name: str, *, allow_nan: bool) -> None:
    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())
    if n_inf:
        msg = (
            f"{name} has {n_inf} infinite value(s): infinities cannot be modelled; check divisions"
        )
        raise ValueError(msg)
    if n_nan and not allow_nan:
        msg = (
            f"{name} has {n_nan} missing value(s) (NaN / null): "
            "drop or impute them before modelling — nothing here fills them in silently"
        )
        raise ValueError(msg)


__all__ = ["F64", "ArrayLike", "columns", "to_matrix", "to_vector"]
