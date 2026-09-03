"""Fold-aware subsetting, in one place.

Every model wrapper needs the same three moves: take a fold's training rows from a column
(whatever container it arrived in), from a vector, or pass everything through when there is
no fold. They were copy-pasted into three modules before this file existed; DRY at the
numerics applies to plumbing too.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from glasshouse.arrays import F64, ArrayLike, to_vector

Rows = Any  # an integer index array, or None for "all rows"


def as_array(col: ArrayLike) -> npt.NDArray[Any]:
    """Return the column as a plain numpy array, whatever container it came in."""
    arr: npt.NDArray[Any] = np.asarray(col.to_numpy() if hasattr(col, "to_numpy") else col)
    return arr


def subset_column(col: ArrayLike, rows: Rows) -> Any:
    """Return the fold's rows of one column; the column untouched when ``rows`` is None."""
    return col if rows is None else as_array(col)[rows]


def subset_vector(values: ArrayLike, name: str, rows: Rows) -> F64:
    """Return the fold's rows of a vector, validated through the data door."""
    v = to_vector(values, name)
    return v if rows is None else v[rows]


__all__ = ["Rows", "as_array", "subset_column", "subset_vector"]
