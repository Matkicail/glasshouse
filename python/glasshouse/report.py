"""The report: one JSON document that says everything about several models on one dataset.

``build(...)`` takes the actuals, each model's predictions, the weights, the **declared task
type**, and optional features / time / provenance, and computes — in Python, from the Rust
metrics — every score, curve and residual table the report suite shows. The TypeScript side
only draws what is here; nothing is recomputed in a browser. The document conforms to
``report/schema.json`` (checked in) and a test validates it.

Task types and what they select (see ``docs/report-suite.md``):

| task          | family    | y            | naive          |
|---------------|-----------|--------------|----------------|
| frequency     | poisson   | rate         | weighted mean  |
| severity      | gamma     | amount       | weighted mean  |
| pure_premium  | tweedie   | amount w/ 0s | weighted mean  |
| binary        | binomial  | 0/1          | class prior    |
| regression    | gaussian  | real         | weighted mean  |
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from glasshouse import curves, residuals
from glasshouse.arrays import F64, ArrayLike, to_vector
from glasshouse.metrics import FamilyName
from glasshouse.scorecard import Scorecard, compare, scorecard

SCHEMA_VERSION = "glasshouse-report/1"
TaskType = Literal["frequency", "severity", "pure_premium", "binary", "regression"]
_FAMILY: dict[str, FamilyName] = {
    "frequency": "poisson",
    "severity": "gamma",
    "pure_premium": "tweedie",
    "binary": "binomial",
    "regression": "gaussian",
}
_PRIMARY: dict[str, str] = {
    "frequency": "deviance",
    "severity": "deviance",
    "pure_premium": "deviance",
    "binary": "average_precision",
    "regression": "mae",
}


@dataclass(frozen=True)
class Report:
    """The built document. ``to_dict()`` is the JSON; ``write`` saves it."""

    doc: dict[str, Any]
    cards: dict[str, Scorecard] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON document (a copy)."""
        copy: dict[str, Any] = json.loads(json.dumps(self.doc))
        return copy

    def write(self, path: str | Path) -> Path:
        """Write the JSON to ``path`` and return it."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.doc, indent=1))
        return p


def build(  # noqa: PLR0913 — the report's inputs are its recipe; all after `task` are optional
    task: TaskType,
    y: ArrayLike,
    predictions: dict[str, ArrayLike],
    *,
    weight: ArrayLike | None = None,
    features: dict[str, ArrayLike] | None = None,
    time: ArrayLike | None = None,
    power: float | None = None,
    threshold: float = 0.5,
    n_bins: int = 10,
    sample: int = 5000,
    seed: int = 0,
    dataset: str = "dataset",
    describe: str = "",
    split: dict[str, Any] | None = None,
) -> Report:
    """Compute everything the report shows for these models on this data.

    Parameters
    ----------
    task : {"frequency", "severity", "pure_premium", "binary", "regression"}
        Declared, never guessed. Picks the family, the panel, the naive baseline, the curves
        and the residual definition.
    y, predictions, weight
        Actuals on the response scale; ``{label: predictions}`` on the same rows and scale;
        exposure / claim count / none. For a rate task pass rates and exposure.
    features
        ``{name: column}`` to slice A/E and residuals by; numeric columns are binned by
        weighted decile, categoricals by level.
    time
        A time column for residuals over time (binned by weighted decile of time).
    power, threshold, n_bins
        Tweedie power; the binary decision threshold; bins for every binned table.
    sample, seed
        At most ``sample`` rows go into the residual scatter (seeded); every score, curve and
        table uses all rows. The document says which.
    dataset, describe, split
        Provenance: a name, the ``data.describe()`` text (or your own), the ``Splits`` spec.
    """
    if task not in _FAMILY:
        msg = f"unknown task {task!r}: one of {sorted(_FAMILY)}"
        raise ValueError(msg)
    if not predictions:
        msg = "predictions must name at least one model"
        raise ValueError(msg)
    family = _FAMILY[task]
    yy = to_vector(y, "y")
    w = None if weight is None else to_vector(weight, "weight")
    preds = {label: to_vector(p, label) for label, p in predictions.items()}
    for label, p in preds.items():
        if len(p) != len(yy):
            msg = f"{label} has {len(p)} predictions but y has {len(yy)} rows"
            raise ValueError(msg)
    labels = list(preds)

    cards = {
        label: scorecard(
            yy,
            p,
            family=family,
            sample_weight=w,
            power=power,
            n_bins=n_bins,
            threshold=threshold,
            label=label,
        )
        for label, p in preds.items()
    }
    comparisons = [
        {"a": a, "b": b, "rows": [list(r) for r in compare(cards[a], cards[b]).rows]}
        for i, a in enumerate(labels)
        for b in labels[i + 1 :]
    ]
    feats = dict(features or {})
    rng = np.random.default_rng(seed)
    n = len(yy)
    keep = (
        np.sort(rng.choice(n, size=min(sample, n), replace=False)) if n > sample else np.arange(n)
    )

    doc: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "task": {
            "type": task,
            "family": family,
            "power": power,
            "threshold": threshold if task == "binary" else None,
            "primary_metric": _PRIMARY[task],
            "n_bins": n_bins,
        },
        "provenance": {
            "dataset": dataset,
            "describe": describe,
            "split": split,
            "n_rows": int(n),
            "weight_sum": float(n if w is None else w.sum()),
            "sample_rows": len(keep),
            "sample_seed": seed,
        },
        "models": labels,
        "scorecards": {label: c.to_dict() for label, c in cards.items()},
        "naive": next(iter(cards.values())).naive,
        "comparisons": comparisons,
        "curves": _curves(task, yy, preds, w, n_bins),
        "residuals": _residuals(family, yy, preds, w, power, keep, feats, time, n_bins),
    }
    return Report(doc=doc, cards=cards)


def _curves(
    task: str, y: F64, preds: dict[str, F64], w: F64 | None, n_bins: int
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    labels = list(preds)
    for label, p in preds.items():
        if task == "binary":
            out.append(curves.roc(y, p, w, label=label).to_dict())
            out.append(curves.pr(y, p, w, label=label).to_dict())
        elif y.min() >= 0:
            out.append(curves.lorenz(y, p, w, label=label).to_dict())
        out.append(curves.lift(y, p, w, n_bins=n_bins, label=label).to_dict())
        out.append(curves.calibration(y, p, w, n_bins=n_bins, label=label).to_dict())
    for i, a in enumerate(labels):
        for b in labels[i + 1 :]:
            if preds[b].min() > 0:
                out.append(
                    curves.double_lift(
                        y, preds[a], preds[b], w, n_bins=n_bins, label_a=a, label_b=b
                    ).to_dict()
                )
    return out


def _residuals(  # noqa: PLR0913, PLR0917
    family: FamilyName,
    y: F64,
    preds: dict[str, F64],
    w: F64 | None,
    power: float | None,
    keep: Any,
    features: dict[str, ArrayLike],
    time: ArrayLike | None,
    n_bins: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    time_col = None if time is None else to_vector(time, "time")
    for label, mu in preds.items():
        dev = residuals.deviance(y, mu, family=family, sample_weight=w, power=power)
        pea = residuals.pearson(y, mu, family=family, sample_weight=w, power=power)
        hist, edges = np.histogram(dev, bins=40)
        entry: dict[str, Any] = {
            "definition": {
                "deviance": "sign(y - mu) * sqrt(w * d(y, mu))",
                "pearson": "(y - mu) * sqrt(w / V(mu))",
            },
            "summary": {
                "deviance": _summary(dev),
                "pearson": _summary(pea),
            },
            "histogram": {"edges": edges.tolist(), "counts": hist.tolist()},
            "scatter": {
                "fitted": mu[keep].tolist(),
                "deviance": dev[keep].tolist(),
                "actual": y[keep].tolist(),
            },
            "by_feature": [
                residuals.ae_by_feature(
                    col, y, mu, w, name=name, n_bins=n_bins, label=label
                ).to_dict()
                for name, col in features.items()
            ],
            "over_time": None
            if time_col is None
            else residuals.ae_by_feature(
                time_col, y, mu, w, name="time", n_bins=n_bins, label=label
            ).to_dict(),
        }
        out[label] = entry
    return out


def _summary(r: F64) -> dict[str, float]:
    q = np.quantile(r, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "mean": float(r.mean()),
        "std": float(r.std()),
        "q01": float(q[0]),
        "q05": float(q[1]),
        "q25": float(q[2]),
        "median": float(q[3]),
        "q75": float(q[4]),
        "q95": float(q[5]),
        "q99": float(q[6]),
    }


def validate(doc: dict[str, Any]) -> None:
    """Validate a document against ``report/schema.json`` (needs the ``jsonschema`` package).

    Raises ``ValueError`` with the first problem found.
    """
    try:
        import jsonschema  # noqa: PLC0415 — dev / optional
    except ImportError as err:  # pragma: no cover
        msg = "validate() needs jsonschema: uv add --group dev jsonschema"
        raise ImportError(msg) from err
    schema = json.loads(schema_path().read_text())
    try:
        jsonschema.validate(doc, schema)
    except jsonschema.ValidationError as err:
        where = list(err.absolute_path)
        msg = f"report does not match {schema_path().name}: {err.message} at {where}"
        raise ValueError(msg) from err


def schema_path() -> Path:
    """Where the checked-in JSON Schema lives."""
    return Path(__file__).resolve().parents[2] / "report" / "schema.json"


__all__ = ["SCHEMA_VERSION", "Report", "TaskType", "build", "schema_path", "validate"]
