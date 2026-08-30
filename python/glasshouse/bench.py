"""Bench: models by folds on one dataset, scored the same way, written down.

A benchmark is a *recipe* — which data, which task, which models, which split — that anyone
can rerun and get the same report. The result carries every fold's scorecard, the mean and
spread across folds, and the curves computed on the pooled out-of-fold predictions (every
row scored by a model that never saw it). ``to_dict()`` is the report JSON that the report
suite reads; ``to_markdown()`` is the human summary committed next to it.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from glasshouse import curves
from glasshouse.arrays import F64, to_vector
from glasshouse.metrics import FamilyName
from glasshouse.scorecard import HIGHER_IS_BETTER, Scorecard, scorecard
from glasshouse.splits import Fold, Splits


class Model(Protocol):
    """What a model must do to be benchmarked: fit on a fold, predict on rows."""

    def fit(
        self,
        x: Any,
        y: Any,
        sample_weight: Any = None,
        offset: Any = None,
        fold: Fold | None = None,
    ) -> Any:
        """Fit on ``fold.train_idx`` rows (or all rows when ``fold`` is None)."""

    def predict(self, X: Any, offset: Any = None) -> F64:  # noqa: N803
        """Predict the mean on the response scale."""


@dataclass(frozen=True)
class TaskSpec:
    """How a dataset is modelled and scored.

    ``exposure`` names the weight column; for a rate task (``rate=True``) the model gets
    ``offset = log(exposure)``, and scoring uses ``y / exposure`` vs ``mu / exposure`` weighted
    by exposure. For everything else the weight is passed as ``sample_weight`` and scoring is
    on the raw scale.
    """

    family: FamilyName
    target: str
    exposure: str | None = None
    rate: bool = False
    power: float | None = None
    threshold: float = 0.5


@dataclass(frozen=True)
class ModelSpec:
    """A model to benchmark: a label, a factory (fresh model per fold), and its input columns."""

    label: str
    make: Callable[[], Model]
    columns: list[str]


@dataclass(frozen=True)
class FoldResult:
    """One model on one fold."""

    label: str
    fold: int
    card: Scorecard
    seconds: float


@dataclass(frozen=True)
class BenchResult:
    """Everything a benchmark run produced. Serialisable, printable, comparable."""

    dataset: str
    describe: str
    task: TaskSpec
    splits_spec: dict[str, Any]
    n_rows: int
    folds: list[FoldResult]
    curves: list[dict[str, Any]]
    labels: list[str] = field(default_factory=list)

    def naive_summary(self) -> dict[str, tuple[float, float]]:
        """Return the naive baseline: metric -> (mean, std) across folds (same for every model)."""
        first = self.labels[0]
        cards = [r.card for r in self.folds if r.label == first]
        return {
            m: (float(np.mean(v)), float(np.std(v)))
            for m in cards[0].naive
            for v in [[c.naive[m] for c in cards]]
        }

    def summary(self) -> dict[str, dict[str, tuple[float, float]]]:
        """Per model: metric -> (mean, std) across folds."""
        out: dict[str, dict[str, tuple[float, float]]] = {}
        for label in self.labels:
            cards = [r.card for r in self.folds if r.label == label]
            out[label] = {
                m: (float(np.mean(v)), float(np.std(v)))
                for m in cards[0].metrics
                for v in [[c.metrics[m] for c in cards]]
            }
        return out

    def to_dict(self) -> dict[str, Any]:
        """Return the report JSON, version 0 of the contract in docs/report-suite.md."""
        return {
            "schema": "glasshouse-report/0",
            "dataset": self.dataset,
            "describe": self.describe,
            "task": {
                "family": self.task.family,
                "target": self.task.target,
                "exposure": self.task.exposure,
                "rate": self.task.rate,
                "power": self.task.power,
            },
            "splits": self.splits_spec,
            "n_rows": self.n_rows,
            "models": self.labels,
            "summary": {
                label: {m: {"mean": mu, "std": sd} for m, (mu, sd) in metrics.items()}
                for label, metrics in self.summary().items()
            },
            "naive": {m: {"mean": mu, "std": sd} for m, (mu, sd) in self.naive_summary().items()},
            "folds": [
                {"label": r.label, "fold": r.fold, "seconds": r.seconds, **r.card.to_dict()}
                for r in self.folds
            ],
            "curves": self.curves,
        }

    def to_markdown(self) -> str:
        """Return a short report: the recipe, then mean ± std per metric per model."""
        summ = self.summary()
        metrics = list(next(iter(summ.values())))

        lines = [
            f"# {self.dataset} — {self.task.family} ({self.task.target})",
            "",
            f"{self.describe}",
            "",
            f"Split: {self.splits_spec}. Rows: {self.n_rows:,}. Models: {', '.join(self.labels)}.",
            "Scores are held-out, mean ± std over folds. Best per metric in bold; "
            "`naive` is the weighted mean of y (class prior for binomial), same folds.",
            "",
            "| metric | " + " | ".join(self.labels) + " | naive |",
            "|---|" + "---|" * (len(self.labels) + 1),
        ]
        naive = self.naive_summary()
        for m in metrics:
            values = {label: summ[label][m][0] for label in self.labels}
            if m == "balance":
                best = min(values, key=lambda k: abs(values[k] - 1.0))
            elif m in HIGHER_IS_BETTER:
                sign = 1.0 if HIGHER_IS_BETTER[m] else -1.0
                best = max(values, key=lambda k: sign * values[k])
            else:
                best = ""
            cells = []
            for label in self.labels:
                mu, sd = summ[label][m]
                cell = f"{mu:.5g} ± {sd:.2g}"
                cells.append(f"**{cell}**" if label == best else cell)
            lines.append(f"| {m} | " + " | ".join(cells) + f" | {naive[m][0]:.5g} |")
        secs = {
            label: sum(r.seconds for r in self.folds if r.label == label) for label in self.labels
        }
        lines += [
            "",
            "Fit time (all folds): " + ", ".join(f"{k} {v:.1f}s" for k, v in secs.items()) + ".",
        ]
        return "\n".join(lines)

    def write(self, directory: str | Path) -> Path:
        """Write ``report.json`` and ``report.md`` into ``directory``; return it."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(json.dumps(self.to_dict(), indent=1))
        (out / "report.md").write_text(self.to_markdown() + "\n")
        return out


def run(  # noqa: PLR0913 — the recipe: data, task, models, folds, plus provenance
    frame: Any,
    task: TaskSpec,
    models: list[ModelSpec],
    folds: Splits,
    *,
    dataset: str = "dataset",
    describe: str = "",
    n_bins: int = 10,
) -> BenchResult:
    """Fit every model on every fold, score every held-out fold, pool the curves.

    Parameters
    ----------
    frame : DataFrame
        The full cleaned data; models see only ``fold.train_idx`` rows when fitting.
    task, models, folds
        The recipe. Folds must have been made on this frame (same row count).
    dataset, describe
        Provenance strings for the report.
    """
    y = to_vector(frame[task.target], task.target)
    if len(y) != folds.n_rows:
        msg = f"folds were made for {folds.n_rows} rows but the frame has {len(y)}"
        raise ValueError(msg)
    expo = None if task.exposure is None else to_vector(frame[task.exposure], task.exposure)
    offset = np.log(expo) if (task.rate and expo is not None) else None
    weight_fit = None if task.rate else expo  # a rate model carries exposure in the offset
    results: list[FoldResult] = []
    pooled: dict[str, F64] = {m.label: np.full(len(y), np.nan) for m in models}
    for spec in models:
        for fold in folds:
            t0 = time.perf_counter()
            model = spec.make()
            model.fit(frame[spec.columns], y, sample_weight=weight_fit, offset=offset, fold=fold)
            te = fold.test_idx
            mu = model.predict(
                frame[spec.columns].iloc[te], offset=None if offset is None else offset[te]
            )
            pooled[spec.label][te] = mu
            y_s, mu_s, w_s = _scoring_scale(task, y[te], mu, None if expo is None else expo[te])
            card = scorecard(
                y_s,
                mu_s,
                family=task.family,
                sample_weight=w_s,
                power=task.power,
                n_bins=n_bins,
                threshold=task.threshold,
                label=spec.label,
            )
            results.append(FoldResult(spec.label, fold.number, card, time.perf_counter() - t0))
    scored = ~np.isnan(next(iter(pooled.values())))  # rows that were in some test fold
    curve_data = _pooled_curves(
        task,
        y[scored],
        {k: v[scored] for k, v in pooled.items()},
        None if expo is None else expo[scored],
        n_bins,
    )
    return BenchResult(
        dataset=dataset,
        describe=describe,
        task=task,
        splits_spec={"kind": folds.kind, **folds.spec},
        n_rows=len(y),
        folds=results,
        curves=curve_data,
        labels=[m.label for m in models],
    )


def _scoring_scale(
    task: TaskSpec, y: F64, mu: F64, expo: F64 | None
) -> tuple[F64, F64, F64 | None]:
    """Rates per unit of exposure for a rate task; raw scale (with weight) otherwise."""
    if task.rate and expo is not None:
        return y / expo, mu / expo, expo
    return y, mu, expo


def _pooled_curves(
    task: TaskSpec, y: F64, preds: dict[str, F64], expo: F64 | None, n_bins: int
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    scaled = {k: _scoring_scale(task, y, v, expo) for k, v in preds.items()}
    for label, (ys, mus, ws) in scaled.items():
        if task.family != "binomial" and ys.min() >= 0:
            out.append(curves.lorenz(ys, mus, ws, label=label).to_dict())
        out.append(curves.lift(ys, mus, ws, n_bins=n_bins, label=label).to_dict())
        out.append(curves.calibration(ys, mus, ws, n_bins=n_bins, label=label).to_dict())
    labels = list(scaled)
    for i, a in enumerate(labels):
        for b in labels[i + 1 :]:
            ys, mu_a, ws = scaled[a]
            _, mu_b, _ = scaled[b]
            if mu_b.min() > 0:
                out.append(
                    curves.double_lift(
                        ys, mu_a, mu_b, ws, n_bins=n_bins, label_a=a, label_b=b
                    ).to_dict()
                )
    return out


__all__ = ["BenchResult", "FoldResult", "Model", "ModelSpec", "TaskSpec", "run"]
