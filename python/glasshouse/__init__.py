"""glasshouse: interpretable, well-rounded ML with a Rust core.

Glass-box models and metrics that tell the truth — weighted, exposure-aware, and always
measured against a naive baseline so you know whether the model is actually helping.
"""

from glasshouse import classification, metrics, regression, scorecard
from glasshouse.scorecard import compare
from glasshouse.scorecard import scorecard as scorecard_fn

__all__ = ["classification", "compare", "metrics", "regression", "scorecard", "scorecard_fn"]
__version__ = "0.0.1"
