"""Bench: models by folds on one dataset, scored the same way, written down.

A benchmark is a *recipe* — which data, which task, which models, which split — that anyone
can rerun and get the same report. The result carries every fold's scorecard, the mean and
spread across folds, and the curves computed on the pooled out-of-fold predictions (every
row scored by a model that never saw it). ``to_dict()`` is the report JSON that the report
suite reads; ``to_markdown()`` is the human summary committed next to it.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from glasshouse import report as report_mod
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


class Progress:
    """A tqdm-style one-line progress bar on stderr. Zero dependencies, honest about time.

    Renders ``bench 60%|████████░░| 6/10 [12.3s < 8.2s] glm_full fold 2 (2.1s)`` and rewrites
    the line in place on a terminal; on a non-terminal (CI logs) it prints one line per step.
    """

    def __init__(self, total: int, *, enabled: bool, label: str = "bench") -> None:
        self.total = max(total, 1)
        self.done = 0
        self.enabled = enabled
        self.label = label
        self.start = time.perf_counter()
        self.is_tty = enabled and sys.stderr.isatty()

    def step(self, message: str, seconds: float | None = None) -> None:
        """Mark one unit done and describe it."""
        self.done += 1
        if not self.enabled:
            return
        elapsed = time.perf_counter() - self.start
        eta = elapsed / self.done * (self.total - self.done)
        took = "" if seconds is None else f" ({seconds:.1f}s)"
        filled = round(10 * self.done / self.total)
        bar = "█" * filled + "░" * (10 - filled)
        line = (
            f"{self.label} {100 * self.done // self.total:3d}%|{bar}| "
            f"{self.done}/{self.total} [{elapsed:.1f}s < {eta:.1f}s] {message}{took}"
        )
        end = "\r" if (self.is_tty and self.done < self.total) else "\n"
        sys.stderr.write("\x1b[2K" + line + end if self.is_tty else line + "\n")
        sys.stderr.flush()

    def note(self, message: str) -> None:
        """Print a plain status line without advancing the bar."""
        if self.enabled:
            sys.stderr.write(message + "\n")
            sys.stderr.flush()


_TASK_OF_FAMILY: dict[str, report_mod.TaskType] = {
    "poisson": "frequency",
    "gamma": "severity",
    "tweedie": "pure_premium",
    "binomial": "binary",
    "gaussian": "regression",
}


@dataclass(frozen=True)
class BenchResult:
    """Everything a benchmark run produced. Serialisable, printable, renderable."""

    dataset: str
    describe: str
    task: TaskSpec
    splits_spec: dict[str, Any]
    n_rows: int
    folds: list[FoldResult]
    doc: dict[str, Any] = field(repr=False, default_factory=dict)
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
        """Return the report document (glasshouse-report/1) with the bench block attached."""
        out = dict(self.doc)
        out["bench"] = {
            "summary": {
                label: {m: {"mean": mu, "std": sd} for m, (mu, sd) in metrics.items()}
                for label, metrics in self.summary().items()
            },
            "naive_summary": {
                m: {"mean": mu, "std": sd} for m, (mu, sd) in self.naive_summary().items()
            },
            "folds": [
                {"label": r.label, "fold": r.fold, "seconds": round(r.seconds, 3)}
                for r in self.folds
            ],
        }
        return out

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
        """Write the suite into ``directory``.

        ``report.json`` and ``report.html`` are local artifacts (gitignored; rerun the recipe
        to reproduce them). ``report.md`` and ``pinned.json`` (the summary numbers the drift
        test checks) are small and committed.
        """
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        doc = self.to_dict()
        (out / "report.json").write_text(json.dumps(doc, indent=1))
        (out / "report.md").write_text(self.to_markdown() + "\n")
        (out / "pinned.json").write_text(json.dumps(doc["bench"], indent=1) + "\n")
        report_mod.to_html(doc, out / "report.html")
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
    features: list[str] | None = None,
    progress: bool = False,
) -> BenchResult:
    """Fit every model on every fold, score every held-out fold, build the report.

    Parameters
    ----------
    frame : DataFrame
        The full cleaned data; models see only ``fold.train_idx`` rows when fitting.
    task, models, folds
        The recipe. Folds must have been made on this frame (same row count).
    dataset, describe
        Provenance strings for the report.
    features
        Column names to slice A/E and residuals by in the report.
    progress
        Show a progress bar on stderr (one unit per model-fold, plus the report build).
    """
    have = {str(c) for c in getattr(frame, "columns", [])}
    wanted = {task.target, *((task.exposure and [task.exposure]) or []), *(features or [])}
    for spec in models:
        wanted.update(spec.columns)
    missing = sorted(wanted - have)
    if have and missing:
        msg = f"the frame is missing column(s) {missing}; it has {sorted(have)[:12]}..."
        raise ValueError(msg)
    y = to_vector(frame[task.target], task.target)
    if len(y) != folds.n_rows:
        msg = f"folds were made for {folds.n_rows} rows but the frame has {len(y)}"
        raise ValueError(msg)
    expo = None if task.exposure is None else to_vector(frame[task.exposure], task.exposure)
    offset = np.log(expo) if (task.rate and expo is not None) else None
    weight_fit = None if task.rate else expo  # a rate model carries exposure in the offset
    results, pooled, bar = _fit_folds(
        frame, task, models, folds, n_bins, weight_fit, offset, y, expo, progress=progress
    )
    scored = ~np.isnan(next(iter(pooled.values())))  # rows that were in some test fold
    y_s, _, w_s = _scoring_scale(task, y[scored], y[scored], None if expo is None else expo[scored])
    preds_s = {
        label: _scoring_scale(task, y[scored], p[scored], None if expo is None else expo[scored])[1]
        for label, p in pooled.items()
    }
    feature_cols = {
        name: np.asarray(
            frame[name].to_numpy() if hasattr(frame[name], "to_numpy") else frame[name]
        )[scored]
        for name in (features or [])
    }
    t_report = time.perf_counter()
    built = report_mod.build(
        _TASK_OF_FAMILY[task.family],
        y_s,
        preds_s,
        weight=w_s,
        features=feature_cols,
        power=task.power,
        threshold=task.threshold,
        n_bins=n_bins,
        dataset=dataset,
        describe=describe,
        split={"kind": folds.kind, **folds.spec},
    )
    bar.step("building the report", time.perf_counter() - t_report)
    return BenchResult(
        dataset=dataset,
        describe=describe,
        task=task,
        splits_spec={"kind": folds.kind, **folds.spec},
        n_rows=len(y),
        folds=results,
        doc=built.to_dict(),
        labels=[m.label for m in models],
    )


def _fit_folds(  # noqa: PLR0913, PLR0917 — internal plumbing shared by run()
    frame: Any,
    task: TaskSpec,
    models: list[ModelSpec],
    folds: Splits,
    n_bins: int,
    weight_fit: F64 | None,
    offset: F64 | None,
    y: F64,
    expo: F64 | None,
    *,
    progress: bool,
) -> tuple[list[FoldResult], dict[str, F64], Progress]:
    """Fit every model on every fold; return the fold scorecards and pooled OOF predictions."""
    results: list[FoldResult] = []
    pooled: dict[str, F64] = {m.label: np.full(len(y), np.nan) for m in models}
    bar = Progress(len(models) * len(folds) + 1, enabled=progress)
    for spec in models:
        for fold in folds:
            t0 = time.perf_counter()
            model = spec.make()
            model.fit(frame[spec.columns], y, sample_weight=weight_fit, offset=offset, fold=fold)
            te = fold.test_idx
            off_te = None if offset is None else offset[te]
            mu = model.predict(frame[spec.columns].iloc[te], offset=off_te)
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
            took = time.perf_counter() - t0
            results.append(FoldResult(spec.label, fold.number, card, took))
            bar.step(f"{spec.label} fold {fold.number}", took)
    return results, pooled, bar


def _check_columns(
    frame: Any, task: TaskSpec, models: list[ModelSpec], features: list[str] | None
) -> None:
    """Refuse with the missing columns named, before any fitting starts."""
    have = {str(c) for c in getattr(frame, "columns", [])}
    if not have:
        return
    wanted = {task.target, *([task.exposure] if task.exposure else []), *(features or [])}
    for spec in models:
        wanted.update(spec.columns)
    missing = sorted(wanted - have)
    if missing:
        msg = f"the frame is missing column(s) {missing}; it has {sorted(have)[:12]}..."
        raise ValueError(msg)


def _scoring_scale(
    task: TaskSpec, y: F64, mu: F64, expo: F64 | None
) -> tuple[F64, F64, F64 | None]:
    """Rates per unit of exposure for a rate task; raw scale (with weight) otherwise."""
    if task.rate and expo is not None:
        return y / expo, mu / expo, expo
    return y, mu, expo


__all__ = ["BenchResult", "FoldResult", "Model", "ModelSpec", "TaskSpec", "run"]
