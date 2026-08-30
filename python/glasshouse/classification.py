"""Binary classification metrics, weighted. Three kinds, and it matters which you read.

- **Decision** metrics (accuracy, precision, recall, F1, MCC) score a hard yes/no at a
  threshold. They answer "how good is this policy" — not "how good is this model".
- **Ranking** metrics (ROC-AUC, average precision, KS) score the ordering of the scores, so
  they are threshold-free but blind to whether the probabilities are right.
- **Probability** metrics (log-loss, Brier) are proper scores: you cannot game them by
  miscalibrating. Read them with :func:`glasshouse.metrics.calibration_table`.

Labels must be exactly 0/1; ``score`` is a probability or any monotone score (for the
probability metrics it must be a probability in (0, 1)).
"""

from __future__ import annotations

from dataclasses import dataclass

from glasshouse import _core
from glasshouse.metrics import ArrayLike, _f64, _weights, deviance


@dataclass(frozen=True)
class ThresholdMetrics:
    """Everything you can read off the confusion counts at one threshold."""

    tp: float
    fp: float
    fn: float
    tn: float
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    mcc: float


def threshold_metrics(
    y: ArrayLike, score: ArrayLike, sample_weight: ArrayLike | None = None, threshold: float = 0.5
) -> ThresholdMetrics:
    """Compute the confusion counts and every decision metric at ``score >= threshold``.

    What it is for: judging a *decision rule*, which is what actually gets deployed. Read
    ``mcc`` first under imbalance: it is high only when all four cells are right, so it
    punishes both missing the rare class and crying wolf (Chicco & Jurman 2020). ``f1``
    ignores true negatives; ``accuracy`` is meaningless at 0.2 % positives.

    When it lies: it scores one threshold, not the model — a threshold tuned on the test set
    flatters everything here. Undefined ratios (nothing flagged, one class only) return 0,
    scikit-learn's convention, rather than NaN.

    Examples
    --------
    >>> from glasshouse.classification import threshold_metrics
    >>> m = threshold_metrics([1, 1, 0, 1, 0, 0], [0.9, 0.8, 0.7, 0.4, 0.3, 0.1])
    >>> (m.tp, m.fp, m.fn, m.tn)
    (2.0, 1.0, 1.0, 2.0)
    >>> round(m.mcc, 4)
    0.3333
    """
    w = _weights(sample_weight)
    raw = _core.threshold_metrics(_f64(y, "y"), _f64(score, "score"), w, threshold)
    return ThresholdMetrics(**raw)


def mcc(
    y: ArrayLike, score: ArrayLike, sample_weight: ArrayLike | None = None, threshold: float = 0.5
) -> float:
    """Matthews correlation (phi) at ``threshold``. See :func:`threshold_metrics`."""
    return threshold_metrics(y, score, sample_weight, threshold).mcc


def f1(
    y: ArrayLike, score: ArrayLike, sample_weight: ArrayLike | None = None, threshold: float = 0.5
) -> float:
    """F1 at ``threshold``. See :func:`threshold_metrics`."""
    return threshold_metrics(y, score, sample_weight, threshold).f1


def roc_auc(y: ArrayLike, score: ArrayLike, sample_weight: ArrayLike | None = None) -> float:
    """Area under the ROC curve: ranking quality across all thresholds. 0.5 is random.

    What it is for: balanced classes, comparing models broadly; the probability that a random
    positive outscores a random negative. Ties count half, so it is order-free.

    When it lies: under heavy imbalance a 0.95 can hide a useless precision, because the
    false-positive *rate* stays tiny when negatives are plentiful — read
    :func:`average_precision` there.

    Examples
    --------
    >>> from glasshouse.classification import roc_auc
    >>> roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    0.75
    """
    w = _weights(sample_weight)
    return float(_core.roc_auc(_f64(y, "y"), _f64(score, "score"), w))


def average_precision(
    y: ArrayLike, score: ArrayLike, sample_weight: ArrayLike | None = None
) -> float:
    """Area under the precision-recall curve, step-wise like scikit-learn.

    What it is for: fraud and rare events, where positives are 0.2 % and ROC-AUC looks great
    for everyone. It only looks at how the positives are ranked, so it is the honest ranking
    number under imbalance.

    When it lies: it ignores true negatives entirely, so when both classes matter equally it
    undersells a model; and its baseline moves with prevalence, so never compare it across
    datasets without saying what the positive rate was.

    Examples
    --------
    >>> from glasshouse.classification import average_precision
    >>> round(average_precision([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]), 4)
    0.8333
    """
    w = _weights(sample_weight)
    return float(_core.average_precision(_f64(y, "y"), _f64(score, "score"), w))


def ks(y: ArrayLike, score: ArrayLike, sample_weight: ArrayLike | None = None) -> float:
    """Kolmogorov-Smirnov statistic: largest gap between the two classes' score distributions.

    What it is for: credit scoring's favourite; 0 is no separation, 1 is perfect, and it
    names the single score cut where the classes are most different.

    When it lies: it is one point on the curve, so two very different models can tie, and it
    is as blind to calibration as every other ranking metric.

    Examples
    --------
    >>> from glasshouse.classification import ks
    >>> ks([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    0.5
    """
    w = _weights(sample_weight)
    return float(_core.ks(_f64(y, "y"), _f64(score, "score"), w))


def log_loss(y: ArrayLike, prob: ArrayLike, sample_weight: ArrayLike | None = None) -> float:
    """Mean negative log-likelihood of the labels under ``prob``: half the binomial deviance.

    What it is for: scoring the probabilities themselves. It is a proper score, so the only
    way to improve it is to be more right — you cannot game it by shifting a threshold.

    When it lies: a handful of confident mistakes dominate it (``-ln(1e-6)`` is 13.8), and it
    is hard to explain to a business owner. Pair it with :func:`brier`.

    Examples
    --------
    >>> from glasshouse.classification import log_loss
    >>> round(log_loss([0, 1], [0.2, 0.8]), 4)
    0.2231
    """
    return deviance(y, prob, family="binomial", sample_weight=sample_weight) / 2.0


def brier(y: ArrayLike, prob: ArrayLike, sample_weight: ArrayLike | None = None) -> float:
    """Brier score: mean squared error between labels and probabilities. 0 is perfect.

    What it is for: a proper score you can explain — "on average the probability was off by
    the square root of this". Decomposes into calibration and refinement.

    When it lies: under heavy imbalance the always-say-0 model scores the positive rate, which
    looks excellent; read it against that baseline, not against 0.

    Examples
    --------
    >>> from glasshouse.classification import brier
    >>> round(brier([0, 1], [0.2, 0.8]), 4)
    0.04
    """
    return deviance(y, prob, family="gaussian", sample_weight=sample_weight)


__all__ = [
    "ThresholdMetrics",
    "average_precision",
    "brier",
    "f1",
    "ks",
    "log_loss",
    "mcc",
    "roc_auc",
    "threshold_metrics",
]
