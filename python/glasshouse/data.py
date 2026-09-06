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
    clean: Callable[..., Any]
    requires: str | None = None  # another dataset the cleaner joins to (passed as its 2nd argument)


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


def clean_fremtpl2_sev(raw: Any, freq: Any) -> Any:
    """freMTPL2sev: one row per claim, summed per policy and joined to the cleaned frequency frame.

    ``Severity`` is the mean claim amount per claim on the policy and ``ClaimCount`` the number
    of claims in the severity file (the weight for a gamma model). Returns a new frame.
    """
    import pandas as pd  # noqa: PLC0415 — optional extra

    claims = pd.DataFrame(
        {
            "IDpol": raw["IDpol"].astype(np.int64),
            "ClaimAmount": np.minimum(raw["ClaimAmount"].astype(np.float64), 1_000_000.0),
        }
    )
    per_policy = claims.groupby("IDpol", as_index=False).agg(
        ClaimCount=("ClaimAmount", "size"), ClaimTotal=("ClaimAmount", "sum")
    )
    per_policy["ClaimCount"] = per_policy["ClaimCount"].astype(np.float64)
    df = per_policy.merge(freq, on="IDpol", how="inner")
    df["Severity"] = df["ClaimTotal"] / df["ClaimCount"]
    return df


def clean_bike_sharing(raw: Any) -> Any:
    """Bike sharing hourly counts: typed columns and a chronological index; nothing dropped.

    The source ships the rows in time order (hourly, 2011-01-01 to 2012-12-31, some hours
    missing); ``hour_index`` is that order, the column a time-ordered split cuts on. Refuses
    a frame whose year-month is not non-decreasing, because then the order is not time.
    """
    import pandas as pd  # noqa: PLC0415 — optional extra

    year_month = raw["year"].astype(np.int64) * 100 + raw["month"].astype(np.int64)
    if not np.all(np.diff(year_month.to_numpy()) >= 0):
        msg = "bike_sharing: rows are not in chronological order, so row order cannot be time"
        raise ValueError(msg)
    flags = {"False": 0.0, "True": 1.0}
    df = pd.DataFrame(
        {
            "hour_index": np.arange(len(raw), dtype=np.float64),
            "season": raw["season"].astype(str),
            "year": raw["year"].astype(np.float64) + 2011.0,
            "month": raw["month"].astype(np.float64),
            "hour": raw["hour"].astype(np.float64),
            "holiday": raw["holiday"].astype(str).map(flags).astype(np.float64),
            "weekday": raw["weekday"].astype(np.float64),
            "workingday": raw["workingday"].astype(str).map(flags).astype(np.float64),
            "weather": raw["weather"].astype(str),
            "temp": raw["temp"].astype(np.float64),
            "feel_temp": raw["feel_temp"].astype(np.float64),
            "humidity": raw["humidity"].astype(np.float64),
            "windspeed": raw["windspeed"].astype(np.float64),
            "count": raw["count"].astype(np.float64),
        }
    )
    return df


def clean_telco_churn(raw: Any) -> Any:
    """Telco customer churn: the label to 0/1, quotes stripped, blank charges to 0, no id."""
    import pandas as pd  # noqa: PLC0415 — optional extra

    df = pd.DataFrame({c: raw[c] for c in raw.columns if c != "customerID"})
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(str).str.strip("'")
    df["Churn"] = df["Churn"].map({"No": 0.0, "Yes": 1.0}).astype(np.float64)
    # "No internet service" on six add-on columns says exactly what InternetService == "No"
    # says, and "No phone service" what PhoneService == "No": collinear with an intercept,
    # so the add-ons become plain Yes/No and the two service columns keep the information
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype) == "str":
            df[c] = df[c].replace({"No internet service": "No", "No phone service": "No"})
    charges = pd.to_numeric(df["TotalCharges"].replace("", np.nan), errors="coerce")
    df["TotalCharges"] = charges.fillna(0.0).astype(np.float64)
    for c in ("SeniorCitizen", "tenure", "MonthlyCharges"):
        df[c] = df[c].astype(np.float64)
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
            "plus Amount (the OpenML copy carries no Time column). OpenML 1597."
        ),
        cleaning=("Class parsed to 0/1 float; nothing else touched",),
        clean=clean_creditcard,
    ),
    "fremtpl2_sev": Dataset(
        name="fremtpl2_sev",
        openml_id=41215,
        target="Severity",
        task="gamma severity (mean claim amount per claim, weight = ClaimCount)",
        exposure="ClaimCount",
        citation=(
            "French MTPL claim severity, 26 639 claims on 24 950 policies (CASdatasets "
            "freMTPL2sev; OpenML 41215), joined to the cleaned freMTPL2freq policies on IDpol. "
            "Same references as fremtpl2_freq (Noll, Schelldorfer & Wüthrich 2018; Wüthrich & "
            "Merz 2023, §13.1)."
        ),
        cleaning=(
            "ClaimAmount per claim capped at 1 000 000 (three claims exceed it)",
            "claims summed per policy: ClaimCount (the weight) and ClaimTotal; "
            "Severity = ClaimTotal / ClaimCount (the target)",
            "inner join to the cleaned frequency frame on IDpol: 6 claims with no policy row "
            "dropped; policies with ClaimNb > 0 but no severity record (about 9 100) are not "
            "in this frame, a known quirk of the source",
            "ClaimCount is the count in the severity file; it agrees with the capped ClaimNb on "
            "99.9 % of policies",
        ),
        clean=clean_fremtpl2_sev,
        requires="fremtpl2_freq",
    ),
    "bike_sharing": Dataset(
        name="bike_sharing",
        openml_id=42712,
        target="count",
        task="poisson counts, time-ordered (hourly rentals; split on hour_index)",
        exposure=None,
        citation=(
            "Bike Sharing Dataset, Capital Bikeshare, Washington DC (Fanaee-T & Gama, 'Event "
            "labeling combining ensemble detectors and background knowledge', Progress in AI, "
            "2014; UCI). 17 379 hourly rows, 2011-01-01 to 2012-12-31, weather and calendar "
            "features. OpenML 42712 (which omits the casual/registered split of the count)."
        ),
        cleaning=(
            "hour_index = row order added; the source is chronological (checked: year-month "
            "never decreases), so a time-ordered split cuts on it",
            "year 0/1 -> 2011/2012; holiday and workingday -> 0/1 floats",
            "season and weather kept as their level names; temperatures, humidity and "
            "windspeed are the source's normalised values, untouched",
            "nothing dropped, nothing capped",
        ),
        clean=clean_bike_sharing,
    ),
    "telco_churn": Dataset(
        name="telco_churn",
        openml_id=42178,
        target="Churn",
        task="binomial churn (26.5 % positives)",
        exposure=None,
        citation=(
            "Telco Customer Churn (IBM sample data set; Kaggle blastchar/telco-customer-churn; "
            "OpenML 42178). 7 043 customers, 19 features: contract, services, tenure, charges."
        ),
        cleaning=(
            "Churn Yes/No -> 1/0 float; customerID dropped",
            "quotes stripped from the string columns OpenML ships ('One year' -> One year)",
            "TotalCharges is blank for 11 customers with zero tenure; set to 0",
            "'No internet service' and 'No phone service' on the add-on columns -> No: they "
            "repeat InternetService = No / PhoneService = No exactly and would make the "
            "one-hot design rank-deficient",
            "SeniorCitizen, tenure, MonthlyCharges as floats; nothing dropped",
        ),
        clean=clean_telco_churn,
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
    df = d.clean(raw) if d.requires is None else d.clean(raw, load(d.requires))
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
    "clean_bike_sharing",
    "clean_creditcard",
    "clean_fremtpl2_freq",
    "clean_fremtpl2_sev",
    "clean_telco_churn",
    "describe",
    "load",
]
