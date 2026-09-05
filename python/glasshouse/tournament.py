"""Win sets and the tournament: several models price the same risks, the cheapest wins.

Every row goes to the model with the lowest prediction (a tie is split equally). Each model
is then judged on the business it won: its share of the market, what it charged there, what
that business actually cost. A model that wins business it under-prices shows an actual over
expected above 1 on its win set — adverse selection, which is what a pricing review is
trying to catch before the market does. This is the most decision-shaped view of a model
comparison, and the reason it sits on the report's Compare tab.

Rates and exposure follow the same convention as every metric: pass ``y`` and the
predictions on the same scale, with the exposure as ``sample_weight`` if they are rates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from glasshouse import _core
from glasshouse.arrays import F64, ArrayLike, to_vector


@dataclass(frozen=True)
class Tournament:
    """One row per model: what it won, what it charged there, what it cost."""

    labels: list[str]
    weight: F64
    predicted: F64
    actual: F64

    @property
    def share(self) -> F64:
        """Each model's share of the total weight."""
        return np.asarray(self.weight / self.weight.sum(), dtype=np.float64)

    @property
    def profit(self) -> F64:
        """``predicted - actual`` on each win set: positive is money made."""
        return np.asarray(self.predicted - self.actual, dtype=np.float64)

    @property
    def actual_over_expected(self) -> F64:
        """``actual / predicted`` on each win set; above 1 the model under-priced what it won."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.asarray(self.actual / self.predicted, dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready (the derived columns included, so the viewer computes nothing)."""
        return {
            "labels": list(self.labels),
            "share": self.share.tolist(),
            "weight": self.weight.tolist(),
            "predicted": self.predicted.tolist(),
            "actual": self.actual.tolist(),
            "profit": self.profit.tolist(),
            "actual_over_expected": self.actual_over_expected.tolist(),
        }

    def __str__(self) -> str:
        """Fixed-width table."""
        lines = [f"{'model':<16}{'share':>8}{'predicted':>14}{'actual':>14}{'A/E':>8}"]
        for i, label in enumerate(self.labels):
            lines.append(
                f"{label:<16}{self.share[i]:>8.3f}{self.predicted[i]:>14.6g}"
                f"{self.actual[i]:>14.6g}{self.actual_over_expected[i]:>8.3f}"
            )
        return "\n".join(lines)


def tournament(
    y: ArrayLike, predictions: dict[str, ArrayLike], sample_weight: ArrayLike | None = None
) -> Tournament:
    """Give every row to the cheapest model (ties split) and sum each model's win set.

    Examples
    --------
    >>> from glasshouse.tournament import tournament
    >>> t = tournament([1, 2, 3, 4], {"a": [0.5, 3.0, 2.0, 4.0], "b": [1.0, 2.0, 2.0, 5.0]})
    >>> t.share.tolist(), t.actual_over_expected.round(3).tolist()
    ([0.625, 0.375], [1.182, 1.167])
    """
    if not predictions:
        msg = "predictions must name at least one model"
        raise ValueError(msg)
    labels = list(predictions)
    preds = [to_vector(p, label) for label, p in predictions.items()]
    w = None if sample_weight is None else to_vector(sample_weight, "sample_weight")
    raw = _core.win_sets(to_vector(y, "y"), preds, w)
    return Tournament(
        labels=labels,
        weight=np.asarray(raw["weight"], dtype=np.float64),
        predicted=np.asarray(raw["predicted"], dtype=np.float64),
        actual=np.asarray(raw["actual"], dtype=np.float64),
    )


def win_sets(
    y: ArrayLike,
    mu_a: ArrayLike,
    mu_b: ArrayLike,
    sample_weight: ArrayLike | None = None,
    label_a: str = "a",
    label_b: str = "b",
) -> Tournament:
    """Run the two-model tournament: where A prices below B and vice versa.

    Read it next to the double lift chart: the double lift shows *where* the models disagree,
    the win sets show what that disagreement would cost in a market.

    Examples
    --------
    >>> from glasshouse.tournament import win_sets
    >>> print(win_sets([1, 2, 3, 4], [0.5, 3.0, 2.0, 4.0], [1.0, 2.0, 2.0, 5.0], label_a="glm"))
    model              share     predicted        actual     A/E
    glm                0.625           5.5           6.5   1.182
    b                  0.375             3           3.5   1.167
    """
    return tournament(y, {label_a: mu_a, label_b: mu_b}, sample_weight)


__all__ = ["Tournament", "tournament", "win_sets"]
