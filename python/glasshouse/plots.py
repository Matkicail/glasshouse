"""Plotly renderers for the four curves.

Data comes from :mod:`glasshouse.curves`; nothing is computed here, so a plot can never show a
number the metrics would not.

Needs the ``plots`` extra: ``pip install "glasshouse[plots]"``. Every function returns a
``plotly.graph_objects.Figure``; call ``.show()`` in a notebook or ``.write_html(path)``.
"""

from __future__ import annotations

from typing import Any

from glasshouse.curves import Calibration, DoubleLift, Lift, Lorenz


def _go() -> Any:
    try:
        import plotly.graph_objects as go  # noqa: PLC0415 — optional extra
    except ImportError as err:  # pragma: no cover
        msg = 'plots need plotly: pip install "glasshouse[plots]"'
        raise ImportError(msg) from err
    return go


def lorenz(*curves: Lorenz) -> Any:
    """Lorenz curves (one or several models) against the diagonal, Gini in the legend."""
    go = _go()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="random", line={"dash": "dash", "color": "grey"}
        )
    )
    for c in curves:
        fig.add_trace(go.Scatter(x=c.x, y=c.y, mode="lines", name=f"{c.label} (Gini {c.gini:.3f})"))
    fig.update_layout(
        title="Lorenz curve — ranked by predicted risk, low to high",
        xaxis_title="cumulative share of exposure",
        yaxis_title="cumulative share of outcome",
        xaxis={"range": [0, 1]},
        yaxis={"range": [0, 1]},
    )
    return fig


def lift(*curves: Lift) -> Any:
    """Actual and predicted by prediction bin; a good model's two lines sit on top of each other."""
    go = _go()
    fig = go.Figure()
    for c in curves:
        fig.add_trace(
            go.Scatter(x=c.bin, y=c.predicted, mode="lines+markers", name=f"{c.label} predicted")
        )
        fig.add_trace(
            go.Scatter(
                x=c.bin,
                y=c.actual,
                mode="lines+markers",
                name=f"{c.label} actual",
                line={"dash": "dot"},
            )
        )
    fig.update_layout(
        title="Lift — actual vs predicted by prediction bin (equal weight)",
        xaxis_title="prediction bin (low → high)",
        yaxis_title="mean outcome",
    )
    return fig


def double_lift(c: DoubleLift) -> Any:
    """Two models where they disagree: bins by a/b; whichever line tracks 'actual' wins there."""
    go = _go()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=c.bin, y=c.actual, mode="lines+markers", name="actual", line={"color": "black"}
        )
    )
    fig.add_trace(go.Scatter(x=c.bin, y=c.predicted_a, mode="lines+markers", name=c.label_a))
    fig.add_trace(go.Scatter(x=c.bin, y=c.predicted_b, mode="lines+markers", name=c.label_b))
    fig.update_layout(
        title=(
            f"Double lift — bins by {c.label_a} / {c.label_b} "
            f"(left: {c.label_b} says more; right: {c.label_a} says more)"
        ),
        xaxis_title=f"bin of {c.label_a} / {c.label_b}, low → high",
        yaxis_title="mean outcome",
    )
    return fig


def calibration(*curves: Calibration) -> Any:
    """Mean prediction vs mean outcome per bin, with the perfect line; hover shows A/E."""
    go = _go()
    fig = go.Figure()
    hi = max(float(max(c.predicted.max(), c.actual.max())) for c in curves) * 1.05
    fig.add_trace(
        go.Scatter(
            x=[0, hi],
            y=[0, hi],
            mode="lines",
            name="perfect",
            line={"dash": "dash", "color": "grey"},
        )
    )
    for c in curves:
        fig.add_trace(
            go.Scatter(
                x=c.predicted,
                y=c.actual,
                mode="lines+markers",
                name=c.label,
                text=[
                    f"A/E {r:.3f}, weight {w:.0f}"
                    for r, w in zip(c.actual_over_expected, c.weight, strict=True)
                ],
                hovertemplate="predicted %{x:.4g}<br>actual %{y:.4g}<br>%{text}",
            )
        )
    fig.update_layout(
        title="Calibration — actual vs predicted per bin",
        xaxis_title="mean predicted",
        yaxis_title="mean actual",
    )
    return fig


__all__ = ["calibration", "double_lift", "lift", "lorenz"]
