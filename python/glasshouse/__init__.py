"""glasshouse: interpretable, well-rounded ML with a Rust core.

Glass-box models and metrics that tell the truth — weighted, exposure-aware, and always
measured against a naive baseline so you know whether the model is actually helping.
"""

from glasshouse import arrays, classification, glm, metrics, regression, scorecard
from glasshouse.glm import GLM

__all__ = ["GLM", "arrays", "classification", "glm", "metrics", "regression", "scorecard"]
__version__ = "0.0.1"
