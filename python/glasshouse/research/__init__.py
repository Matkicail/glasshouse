"""The research fence: neural models that must earn their place on the same report.

Everything here is optional (``pip install "glasshouse[research]"`` brings torch) and
fenced on purpose: a model enters the core only after it beats the spline plus monotone
GLM on held-out deviance *and* calibration on freMTPL2, on the stored splits. Until then it
is a challenger row on the leaderboard, scored by the same Rust deviance, drawn on the same
curves, explained by the same importances and partial dependence, and nothing more.

Order of the track, smallest step first: :class:`~glasshouse.research.cann.CANN` (the GLM
frozen as a skip connection, a small net learning the residual), :class:`AdditiveNet` (one
net per feature: corrections you can draw), :class:`LocalGLMnet` (coefficients that depend
on the row), then a KAN-style additive model with the numeric encodings compared side by
side. The first three share one class and one training loop. See ``docs/research.md``.
"""

from glasshouse.research.cann import CANN, AdditiveNet, LocalGLMnet

__all__ = ["CANN", "AdditiveNet", "LocalGLMnet"]
