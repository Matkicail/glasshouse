"""Datasets: fetched once, cleaned the documented way, cached, and described.

Every loader states its source, its citation, and every cleaning rule it applies — the
"steps required before you run" — so a benchmark number can be reproduced by someone who was
not in the room. Cleaning is a pure function of the raw frame (``clean_<name>``) so it can be
unit-tested without a download.

Needs the ``data`` extra: ``pip install "glasshouse[data]"`` (pandas, pyarrow, scikit-learn
for the OpenML fetch).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Dataset:
    """What a loader knows about its data."""

    name: str
    openml_id: int
    target: str
    task: str
    exposure: str | None
    citation: str
    cleaning: tuple[str, ...]
    clean: Callable[[Any], Any]


def clean_fremtpl2_freq(raw: Any) -> Any:
    """freMTPL2freq cleaning as in Noll, Schelldorfer & Wüthrich (2018) and Wüthrich & Merz (2023).

    Caps the handful of implausible extremes, logs density, orders Area, and tidies the
    quoted strings OpenML ships. Returns a new frame; the raw one is untouched.
    """
    import pandas as pd  # noqa: PLC0415 — optional extra

    df = pd.DataFrame(
        {
            "IDpol": raw["IDpol"].astype(np.int64),
            "ClaimNb": np.minimum(raw["ClaimNb"].astype(np.float64), 4.0),
            "Exposure": np.minimum(raw["Exposure"].astype(np.float64), 1.0),
            "Area": raw["Area"].astype(str).str.strip("'"),
            "VehPower": np.minimum(raw["VehPower"].astype(np.float64), 9.0),
            "VehAge": np.minimum(raw["VehAge"].astype(np.float64), 20.0),
            "DrivAge": np.minimum(raw["DrivAge"].astype(np.float64), 90.0),
            "BonusMalus": np.minimum(raw["BonusMalus"].astype(np.float64), 150.0),
            "VehBrand": raw["VehBrand"].astype(str).str.strip("'"),
            "VehGas": raw["VehGas"].astype(str).str.strip("'"),
            "Density": raw["Density"].astype(np.float64),
            "Region": raw["Region"].astype(str).str.strip("'"),
        }
    )
    df["LogDensity"] = np.log(df["Density"])
    df["AreaCode"] = df["Area"].map({a: i + 1 for i, a in enumerate("ABCDEF")}).astype(np.float64)
    df["Frequency"] = df["ClaimNb"] / df["Exposure"]
    return df


def clean_creditcard(raw: Any) -> Any:
    """ULB credit-card fraud: 284 807 transactions, 492 frauds. Only the label is tidied."""
    import pandas as pd  # noqa: PLC0415 — optional extra

    df = pd.DataFrame(raw)
    df["Class"] = df["Class"].astype(str).str.strip("'").astype(np.int64).astype(np.float64)
    return df


DATASETS: dict[str, Dataset] = {
    "fremtpl2_freq": Dataset(
        name="fremtpl2_freq",
        openml_id=41214,
        target="ClaimNb",
        task="poisson frequency (offset = log Exposure)",
        exposure="Exposure",
        citation=(
            "French MTPL claim frequency, 678 013 policies (CASdatasets freMTPL2freq; "
            "OpenML 41214). Cleaning follows Noll, Schelldorfer & Wüthrich, 'Case Study: French "
            "Motor Third-Party Liability Claims' (SSRN 3164764, 2018) and Wüthrich & Merz, "
            "'Statistical Foundations of Actuarial Learning and its Applications' "
            "(Springer 2023, §13.1)."
        ),
        cleaning=(
            "ClaimNb capped at 4 (a few rows report 5-16 claims in one period)",
            "Exposure capped at 1 (a few rows exceed one policy-year)",
            "VehPower capped at 9, VehAge at 20, DrivAge at 90, BonusMalus at 150",
            "LogDensity = log(Density) added; AreaCode = A..F -> 1..6 added",
            "Frequency = ClaimNb / Exposure added (the rate; model ClaimNb with offset "
            "log Exposure)",
            "quotes stripped from the string columns OpenML ships ('Diesel' -> Diesel)",
            "NOT applied: the Wüthrich-Merz Appendix A.1 de-duplication of near-identical policies",
        ),
        clean=clean_fremtpl2_freq,
    ),
    "creditcard": Dataset(
        name="creditcard",
        openml_id=1597,
        target="Class",
        task="binomial, rare event (0.17 % positives)",
        exposure=None,
        citation=(
            "Credit Card Fraud Detection, ULB Machine Learning Group (Dal Pozzolo et al., "
            "'Calibrating probability with undersampling for unbalanced classification', "
            "IEEE CIDM 2015). "
            "284 807 transactions over two days, 492 frauds; features V1-V28 are PCA components, "
            "plus Time and Amount. OpenML 1597."
        ),
        cleaning=("Class parsed to 0/1 float; nothing else touched",),
        clean=clean_creditcard,
    ),
}


def cache_dir() -> Path:
    """Where cleaned frames live: ``$GLASSHOUSE_CACHE`` or ``~/.cache/glasshouse``."""
    root = os.environ.get("GLASSHOUSE_CACHE")
    path = Path(root) if root else Path.home() / ".cache" / "glasshouse"
    path.mkdir(parents=True, exist_ok=True)
    return path


def describe(name: str) -> str:
    """Return the citation, the task, and every cleaning rule — print it into your report.

    Examples
    --------
    >>> from glasshouse.data import describe
    >>> print(describe("creditcard").splitlines()[0])
    creditcard — binomial, rare event (0.17 % positives); target Class
    """
    d = _dataset(name)
    lines = [
        f"{d.name} — {d.task}; target {d.target}"
        + (f", exposure {d.exposure}" if d.exposure else "")
    ]
    lines.append(f"source: {d.citation}")
    lines.append("cleaning:")
    lines += [f"  - {rule}" for rule in d.cleaning]
    return "\n".join(lines)


def load(name: str, *, refresh: bool = False) -> Any:
    """Return the cleaned frame, fetching from OpenML the first time and caching as parquet.

    Parameters
    ----------
    name : str
        One of ``DATASETS``.
    refresh : bool
        Ignore the cache and fetch + clean again.
    """
    d = _dataset(name)
    path = cache_dir() / f"{d.name}.parquet"
    try:
        import pandas as pd  # noqa: PLC0415
    except ImportError as err:  # pragma: no cover - environment dependent
        msg = 'datasets need the data extra: pip install "glasshouse[data]"'
        raise ImportError(msg) from err
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    try:
        from sklearn.datasets import fetch_openml  # noqa: PLC0415
    except ImportError as err:  # pragma: no cover
        msg = 'fetching needs scikit-learn: pip install "glasshouse[data]"'
        raise ImportError(msg) from err
    raw = fetch_openml(data_id=d.openml_id, as_frame=True, parser="auto").frame
    df = d.clean(raw)
    df.to_parquet(path, index=False)
    return df


def _dataset(name: str) -> Dataset:
    if name not in DATASETS:
        msg = f"unknown dataset {name!r}: one of {sorted(DATASETS)}"
        raise ValueError(msg)
    return DATASETS[name]


__all__ = [
    "DATASETS",
    "Dataset",
    "cache_dir",
    "clean_creditcard",
    "clean_fremtpl2_freq",
    "describe",
    "load",
]
