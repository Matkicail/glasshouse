"""glasshouse: interpretable, well-rounded ML with a Rust core.

Glass-box models and metrics that tell the truth — weighted, exposure-aware, and always
measured against a naive baseline so you know whether the model is actually helping.
"""

from glasshouse import (
    arrays,
    bench,
    classification,
    curves,
    data,
    encoders,
    explain,
    foss,
    gbdt,
    glm,
    metrics,
    profile,
    regression,
    report,
    residuals,
    scorecard,
    splits,
    tournament,
)
from glasshouse.glm import GLM

__all__ = [
    "GLM",
    "arrays",
    "bench",
    "classification",
    "curves",
    "data",
    "encoders",
    "explain",
    "foss",
    "gbdt",
    "glm",
    "metrics",
    "profile",
    "regression",
    "report",
    "residuals",
    "scorecard",
    "splits",
    "tournament",
]
__version__ = "0.0.1"
