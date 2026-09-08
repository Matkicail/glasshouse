"""CANN: the Combined Actuarial Neural Network (Wüthrich & Merz; Schelldorfer & Wüthrich 2019).

Keep the GLM exactly as it is and let a small network learn only what the GLM got wrong::

    link(mu) = [GLM linear predictor, frozen] + [network output] + offset

Three networks fit that slot, and they are the first three models of the research track:

- ``"mlp"`` (the CANN proper): one small net on all inputs; it can learn interactions and
  says so with one number per row.
- ``"additive"`` (a neural additive model, Agarwal et al. 2021): one small net per input
  column, summed, so each correction is a curve you can draw and no interaction is
  possible; a one-hot or spline block gets a linear correction per column.
- ``"localglm"`` (LocalGLMnet, Richman & Wüthrich 2023): a net that outputs one coefficient
  per design column *for each row*, ``sum_j beta_j(x) x_j``, so the GLM's coefficients
  become functions of the row; ``attention`` returns them, the paper's regression
  attentions.
- ``"kan"`` (Kolmogorov-Arnold network, Liu et al. 2024): two layers whose edges carry
  learnable one-dimensional functions, ``phi(x) = w_b silu(x) + w_s sum_i c_i B_i(x)`` on a
  cubic B-spline grid, and whose nodes only add. The first layer's edge functions are
  curves you can draw (``edge_curves``); the second layer is what lets it express an
  interaction, which a one-layer sum of curves cannot.

And three ways to hand the network a numeric feature (``encoding``), after Gorishniy,
Rubachev & Babenko, "On embeddings for numerical features in tabular deep learning" (2022):

- ``"design"`` (the default): the GLM's own encoded columns, so a smooth term's spline basis
  is what the net sees.
- ``"raw"``: the standardised value, one column.
- ``"piecewise"``: the piecewise linear encoding, ``bins`` quantile bins with the fill of the
  bin the value sits in; inside the training range it is the degree-1 B-spline
  ``piecewise_linear`` term (the paper's version extrapolates linearly beyond the range,
  ours holds the boundary value).
- ``"periodic"``: ``concat[sin(v), cos(v)]`` with ``v = 2 pi c x`` for ``frequencies``
  trainable ``c``, initialised from ``N(0, sigma)`` as in the paper, inside the network.
  The paper says ``sigma`` is the hyperparameter that matters; on standardised inputs 0.3
  trains well where 1.0 overfits, and it is a knob, not a constant.

Categorical and interaction terms are always the GLM's design columns.

All three start at the GLM (zero-initialised output layers), train on the same deviance,
stop early the same way, are re-balanced the same way, and show up on the report the same
way: ``term_contributions`` returns the GLM's terms plus the network's, so "explain a row"
reads either "GLM plus one correction" or "GLM plus a correction per feature".

The GLM is fitted first, on the training rows, then frozen. The network's last layer starts
at zero, so before training the CANN *is* the GLM to the last decimal; training minimises
the family deviance, the same one every score uses, and the network moves away from zero
only where that lowers it. On the link scale the network's output is one number per row,
and for a log link its exponential is the multiplicative correction to the GLM's price:
a column a committee can read, chart and argue about.

What is given up, stated plainly: the balance property (restored after training by one
shift of the correction, on the training rows), standard errors (none for the net part),
and a fit that can overfit if let, so training stops early on a seeded slice of the
training rows only, the way the LightGBM adapter does. The CANN is not a glass box; it is
a glass box plus one column, and ``term_contributions`` returns exactly that, so the
report's "explain a row" shows the GLM's terms and the network's correction side by side.

Needs the ``research`` extra: ``pip install "glasshouse[research]"`` (torch, CPU build).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from glasshouse._rows import as_array, subset_column, subset_vector
from glasshouse.arrays import F64, ArrayLike, columns, to_vector
from glasshouse.encoders import BSpline, piecewise_linear
from glasshouse.glm import GLM
from glasshouse.metrics import FamilyName
from glasshouse.splits import Fold


def _default_glm() -> GLM:
    return GLM(family="poisson")


@dataclass
class CANN:
    """A frozen GLM plus a small network on its residual, trained on the family deviance.

    Parameters
    ----------
    family, power
        The family the deviance is taken in; must match the GLM's.
    glm
        A factory for the GLM to freeze (a fresh one per fold), e.g.
        ``lambda: GLM(family="poisson", terms={...})``. Its encoders build the network's
        inputs too, so the net sees the same one-hot, spline or smooth columns the GLM does,
        standardised.
    hidden
        Widths of the hidden layers (tanh).
    epochs, learning_rate, batch_size
        Adam over mini-batches, at most ``epochs`` passes. The defaults (0.01, 1024) move a
        zero-initialised last layer far enough on a few thousand rows; on a few hundred
        thousand, early stopping does the rest.
    valid_fraction, patience
        A seeded slice of the training rows scores each epoch; training stops after
        ``patience`` epochs without improvement and keeps the best weights.
    seed
        For the validation slice, the shuffling and the weights.
    network
        ``"mlp"``, ``"additive"``, ``"localglm"`` or ``"kan"``: see the module docstring.
        For ``"kan"`` the first entry of ``hidden`` is the inner width and ``grid`` the
        number of spline intervals per edge.
    encoding, bins, frequencies, sigma
        How numeric features reach the network: see the module docstring. ``bins`` is for
        ``"piecewise"``; ``frequencies`` and ``sigma`` for ``"periodic"``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from glasshouse import GLM
    >>> from glasshouse.research import CANN
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"a": rng.uniform(-1, 1, 400), "b": rng.uniform(-1, 1, 400)})
    >>> y = rng.poisson(np.exp(0.5 * df.a * df.b))  # an interaction a GLM does not have
    >>> m = CANN(glm=lambda: GLM(family="poisson"), epochs=0).fit(df, y)
    >>> bool(np.allclose(m.predict(df), m.glm_.predict(df)))  # zero epochs: it is the GLM
    True
    """

    family: FamilyName = "poisson"
    power: float | None = None
    glm: Callable[[], GLM] = _default_glm
    hidden: tuple[int, ...] = (20, 15, 10)
    epochs: int = 100
    learning_rate: float = 0.01
    batch_size: int = 1024
    valid_fraction: float = 0.2
    patience: int = 10
    seed: int = 0
    network: str = "mlp"
    encoding: str = "design"
    bins: int = 8
    frequencies: int = 8
    sigma: float = 0.3
    grid: int = 5
    glm_: GLM = field(init=False, repr=False)
    layout_: list[tuple[str, int, int]] = field(init=False, repr=False, default_factory=list)
    numeric_: list[str] = field(init=False, repr=False, default_factory=list)
    ple_: dict[str, Any] = field(init=False, repr=False, default_factory=dict)
    slices_: list[tuple[str, int, int]] = field(init=False, repr=False, default_factory=list)
    net_: Any = field(init=False, repr=False, default=None)
    mean_: F64 = field(init=False, repr=False, default_factory=lambda: np.empty(0))
    std_: F64 = field(init=False, repr=False, default_factory=lambda: np.empty(0))
    shift_: float = field(init=False, repr=False, default=0.0)
    history_: list[tuple[float, float]] = field(init=False, repr=False, default_factory=list)
    best_epoch_: int = field(init=False, repr=False, default=0)

    # ------------------------------------------------------------------ fitting

    def fit(
        self,
        X: ArrayLike,  # noqa: N803 — scikit-learn's spelling
        y: ArrayLike,
        sample_weight: ArrayLike | None = None,
        offset: ArrayLike | None = None,
        fold: Fold | None = None,
    ) -> CANN:
        """Fit the GLM on the fold's training rows, freeze it, then train the correction."""
        torch = _torch()
        self.glm_ = self.glm()
        if self.glm_.family != self.family:
            msg = f"the GLM's family ({self.glm_.family!r}) must match the CANN's ({self.family!r})"
            raise ValueError(msg)
        self.glm_.fit(X, y, sample_weight=sample_weight, offset=offset, fold=fold)
        rows = None if fold is None else fold.train_idx
        self._fit_encoding(X, rows)
        inputs = self._inputs_all(X)
        eta_glm = self.glm_.predict_linear(X, offset)
        if rows is not None:
            inputs, eta_glm = inputs[rows], eta_glm[rows]
        yy = subset_vector(y, "y", rows)
        w = (
            np.ones(len(yy))
            if sample_weight is None
            else subset_vector(sample_weight, "sample_weight", rows)
        )
        self.mean_ = inputs.mean(axis=0)
        self.std_ = np.where(inputs.std(axis=0) > 0, inputs.std(axis=0), 1.0)
        x = (inputs - self.mean_) / self.std_

        torch.manual_seed(self.seed)
        self.net_ = self._build(torch, x.shape[1])
        self.shift_ = 0.0
        self.history_ = []
        if self.epochs > 0:
            self._train(torch, x, yy, w, eta_glm)
        self.shift_ = self._balance_shift(x, yy, w, eta_glm)
        return self

    def _train(self, torch: Any, x: F64, y: F64, w: F64, eta_glm: F64) -> None:
        """Adam over mini-batches; the seeded validation slice decides when to stop."""
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(y))
        n_valid = max(1, round(self.valid_fraction * len(y)))
        valid, train = torch.as_tensor(perm[:n_valid]), torch.as_tensor(perm[n_valid:])
        tensors = {  # torch.tensor copies: a read-only numpy view must not be shared
            k: torch.tensor(v, dtype=torch.float64)
            for k, v in (("x", x), ("y", y), ("w", w), ("eta", eta_glm))
        }
        net = self.net_.double()
        optimiser = torch.optim.Adam(net.parameters(), lr=self.learning_rate)
        best, best_state, since_best = float("inf"), None, 0
        for epoch in range(self.epochs):
            self._epoch(torch, net, optimiser, tensors, train, epoch)
            with torch.no_grad():
                train_dev = float(self._loss(torch, net, tensors, train))
                valid_dev = float(self._loss(torch, net, tensors, valid))
            self.history_.append((train_dev, valid_dev))
            if valid_dev < best - 1e-12:
                best, since_best = valid_dev, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                self.best_epoch_ = epoch + 1
            else:
                since_best += 1
                if since_best >= self.patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()

    def _epoch(
        self, torch: Any, net: Any, optimiser: Any, tensors: dict[str, Any], train: Any, epoch: int
    ) -> None:
        """One pass over the training rows in a seeded order, one Adam step per batch."""
        generator = torch.Generator().manual_seed(self.seed + epoch)
        order = train[torch.randperm(len(train), generator=generator)]
        net.train()
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            optimiser.zero_grad()
            self._loss(torch, net, tensors, idx).backward()
            optimiser.step()
        net.eval()

    def _loss(self, torch: Any, net: Any, t: dict[str, Any], idx: Any) -> Any:
        """Return the weighted mean deviance of the batch, on the response scale."""
        eta = t["eta"][idx] + net(t["x"][idx]).squeeze(-1)
        mu = _inverse_link(torch, self.glm_._link_name(), eta)
        return deviance_torch(torch, self.family, t["y"][idx], mu, t["w"][idx], self.power)

    def _balance_shift(self, x: F64, y: F64, w: F64, eta_glm: F64) -> float:
        """One constant on the correction so the training total adds up again.

        Multiplicative for a log link (a factor on every price), additive for identity;
        left at zero for logit, where a single constant would not restore the total exactly
        and the report's balance row says what remains.
        """
        link = self.glm_._link_name()
        mu = self._mu(eta_glm + self._net(x))
        if link == "log":
            return float(np.log(np.sum(w * y) / np.sum(w * mu)))
        if link == "identity":
            return float(np.sum(w * (y - mu)) / np.sum(w))
        return 0.0

    # ------------------------------------------------------------------ predicting

    def predict_linear(self, X: ArrayLike, offset: ArrayLike | None = None) -> F64:  # noqa: N803
        """Return the GLM's linear predictor plus the network's correction (and the offset)."""
        return np.asarray(
            self.glm_.predict_linear(X, offset) + self.correction(X), dtype=np.float64
        )

    def predict(self, X: ArrayLike, offset: ArrayLike | None = None) -> F64:  # noqa: N803
        """Predict the mean on the response scale."""
        return self._mu(self.predict_linear(X, offset))

    def correction(self, X: ArrayLike) -> F64:  # noqa: N803
        """Return the network's output per row on the link scale, balance shift included.

        For a log link ``exp`` of it is the factor the network multiplies the GLM's price by.
        """
        x = (self._inputs_all(X) - self.mean_) / self.std_
        return np.asarray(self._net(x) + self.shift_, dtype=np.float64)

    def term_contributions(self, X: ArrayLike) -> tuple[F64, list[str]]:  # noqa: N803
        """Return the GLM's per-term contributions plus the network's.

        One ``network`` column for the ``"mlp"`` net; one ``<term> (net)`` column per input
        column for the additive and local nets. Either way they add up to ``predict_linear``
        without the offset, so "explain a row" shows both sides.
        """
        parts, names = self.glm_.term_contributions(X)
        x = (self._inputs_all(X) - self.mean_) / self.std_
        torch = _torch()
        with torch.no_grad():
            terms = self.net_.double().terms(torch.tensor(x, dtype=torch.float64)).numpy()
        terms = np.asarray(terms, dtype=np.float64)
        terms[:, 0] += self.shift_  # the balance shift rides on the first column
        one_column = self.network in ("mlp", "kan")
        net_names = ["network"] if one_column else [f"{n} (net)" for n, _, _ in self.slices_]
        return np.column_stack([parts, terms]), [*names, *net_names]

    def attention(self, X: ArrayLike) -> tuple[F64, list[str]]:  # noqa: N803
        """Return LocalGLMnet's per-row coefficients, one per design column (standardised).

        The paper's regression attentions: a column whose coefficient is near zero on every
        row is not used; one whose coefficient moves with the row is an interaction.
        """
        if self.network != "localglm":
            msg = "attention is only defined for network='localglm'"
            raise ValueError(msg)
        x = (self._inputs_all(X) - self.mean_) / self.std_
        torch = _torch()
        with torch.no_grad():
            beta = self.net_.double().beta(torch.tensor(x, dtype=torch.float64)).numpy()
        names = (
            self.glm_.feature_names_in_[1:]
            if self.glm_.fit_intercept
            else self.glm_.feature_names_in_
        )
        return np.asarray(beta, dtype=np.float64), list(names)

    def _term_slices(self) -> list[tuple[str, int, int]]:
        """Return the design's columns per term, without the intercept."""
        slices = self.glm_._slices
        if slices:
            return [(name, lo, hi) for name, (lo, hi) in slices.items()]
        return [(name, i, i + 1) for i, name in enumerate(self.glm_.input_columns_)]

    def _fit_encoding(self, X: ArrayLike, rows: npt.NDArray[np.int64] | None) -> None:  # noqa: N803
        """Decide the network's input layout and fit the piecewise bins on the training rows.

        With ``encoding="design"`` the layout is the GLM's term slices. Otherwise every
        numeric input column becomes its raw value (one column, or ``bins`` piecewise
        columns) and everything else keeps its design block.
        """
        if self.encoding not in ("design", "raw", "piecewise", "periodic"):
            msg = f"encoding must be design, raw, piecewise or periodic, not {self.encoding!r}"
            raise ValueError(msg)
        self.ple_, self.numeric_ = {}, []
        if self.encoding == "design":
            self.layout_ = self._term_slices()
            return
        cols = columns(X)
        if cols is None:
            msg = "an encoding other than 'design' needs a DataFrame with named columns"
            raise ValueError(msg)
        layout: list[tuple[str, int, int]] = []
        for name, col in cols:
            at = layout[-1][2] if layout else 0
            layout.append((str(name), at, at + self._width(str(name), col, rows)))
        for key, (lo, hi) in self.glm_._slices.items():
            if "*" in key:  # interaction terms keep their design block
                at = layout[-1][2] if layout else 0
                layout.append((key, at, at + (hi - lo)))
        self.layout_ = layout

    def _width(self, name: str, col: Any, rows: npt.NDArray[np.int64] | None) -> int:
        """How many input columns this frame column becomes, fitting the bins if asked."""
        if as_array(col).dtype.kind not in "fiub":
            lo, hi = self.glm_._slices.get(name, (0, 1))
            return hi - lo
        self.numeric_.append(name)
        if self.encoding != "piecewise":
            return 1
        enc = piecewise_linear(name, df=self.bins).fit(subset_column(col, rows))
        self.ple_[name] = enc
        return int(enc.transform(subset_column(col, rows)[:2])[0].shape[1])

    def _build(self, torch: Any, n_in: int) -> Any:
        """Build the network on the layout, behind the periodic front layer when asked."""
        if self.encoding != "periodic":
            self.slices_ = list(self.layout_)
            return _network(torch, self.network, n_in, self.hidden, self.slices_, self.grid)
        positions = [lo for name, lo, _ in self.layout_ if name in self.numeric_]
        front = _periodic_front(torch, n_in, positions, self.frequencies, self.sigma)
        self.slices_ = front.slices(self.layout_)
        body = _network(torch, self.network, front.n_out, self.hidden, self.slices_, self.grid)
        return _with_front(torch, front, body)

    def _link_name(self) -> str:
        return self.glm_._link_name()

    def _mu(self, eta: F64) -> F64:
        return self.glm_._inverse_link(eta)

    def _net(self, x: F64) -> F64:
        torch = _torch()
        with torch.no_grad():
            out = self.net_.double()(torch.tensor(x, dtype=torch.float64)).squeeze(-1)
        return np.asarray(out.numpy(), dtype=np.float64)

    def _inputs_all(self, X: ArrayLike) -> F64:  # noqa: N803
        """Return the network's inputs for every row of ``X``, laid out as ``layout_``."""
        design = self.glm_._design_predict(X)
        design = design[:, 1:] if self.glm_.fit_intercept else design
        if self.encoding == "design":
            return np.asarray(design, dtype=np.float64)
        by_name = dict(columns(X) or [])
        blocks: list[F64] = []
        for name, _, _ in self.layout_:
            if name in self.ple_:
                blocks.append(self.ple_[name].transform(by_name[name])[0])
            elif name in self.numeric_:
                blocks.append(to_vector(by_name[name], name)[:, None])
            else:
                lo, hi = self.glm_._slices[name]
                blocks.append(design[:, lo:hi])
        return np.ascontiguousarray(np.column_stack(blocks), dtype=np.float64)

    def edge_curves(self, n_points: int = 41) -> dict[str, dict[str, list[Any]]]:
        """Return the KAN's first-layer edge functions, one curve per hidden unit per input.

        Each curve is ``phi_{q,p}`` on a grid over the input's standardised range, given
        back in the input's own units: the one-dimensional pieces the network is made of,
        which is the KAN's whole claim to being readable.
        """
        if self.network != "kan":
            msg = "edge_curves is only defined for network='kan'"
            raise ValueError(msg)
        torch = _torch()
        body = self.net_.body if hasattr(self.net_, "front") else self.net_
        names = self._input_names()
        grid = np.linspace(-3.0, 3.0, n_points)
        with torch.no_grad():
            curves = body.first.edge_functions(torch.tensor(grid, dtype=torch.float64)).numpy()
        out: dict[str, dict[str, list[Any]]] = {}
        for p, name in enumerate(names):
            mean, std = (self.mean_[p], self.std_[p]) if p < len(self.mean_) else (0.0, 1.0)
            out[name] = {
                "x": (mean + std * grid).tolist(),
                "curves": np.asarray(curves[:, p, :], dtype=np.float64).tolist(),
            }
        return out

    def _input_names(self) -> list[str]:
        """One name per column the first layer sees (design columns, or the periodic pairs)."""
        names: list[str] = []
        for name, lo, hi in self.layout_:
            names.extend([name] if hi - lo == 1 else [f"{name}[{i}]" for i in range(hi - lo)])
        if self.encoding != "periodic":
            return names
        expanded: list[str] = []
        for name in names:
            if name in self.numeric_:
                expanded.extend(f"{name} sin{k}" for k in range(self.frequencies))
                expanded.extend(f"{name} cos{k}" for k in range(self.frequencies))
            else:
                expanded.append(name)
        return expanded

    # ------------------------------------------------------------------ persistence

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready: the GLM, the network's weights as lists, the scaling and the shift."""
        return {
            "kind": "cann",
            "family": self.family,
            "power": self.power,
            "hidden": list(self.hidden),
            "network": self.network,
            "encoding": self.encoding,
            "bins": self.bins,
            "frequencies": self.frequencies,
            "sigma": self.sigma,
            "grid": self.grid,
            "slices": [list(t) for t in self.slices_],
            "layout": [list(t) for t in self.layout_],
            "numeric": list(self.numeric_),
            "ple": {k: v.to_dict() for k, v in self.ple_.items()},
            "glm": self.glm_.to_dict(),
            "weights": {k: v.tolist() for k, v in self.net_.state_dict().items()},
            "mean": self.mean_.tolist(),
            "std": self.std_.tolist(),
            "shift": self.shift_,
            "best_epoch": self.best_epoch_,
            "history": [list(h) for h in self.history_],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CANN:
        """Rebuild from :meth:`to_dict`; no pickle, ever."""
        torch = _torch()
        glm = GLM.from_dict(payload["glm"])
        model = cls(
            family=payload["family"],
            power=payload["power"],
            hidden=tuple(payload["hidden"]),
            network=payload.get("network", "mlp"),
            encoding=payload.get("encoding", "design"),
            bins=int(payload.get("bins", 8)),
            frequencies=int(payload.get("frequencies", 8)),
            sigma=float(payload.get("sigma", 0.3)),
            grid=int(payload.get("grid", 5)),
        )
        model.glm_ = glm
        model.mean_ = np.asarray(payload["mean"], dtype=np.float64)
        model.std_ = np.asarray(payload["std"], dtype=np.float64)
        model.layout_ = [(str(n), int(lo), int(hi)) for n, lo, hi in payload.get("layout", [])]
        model.numeric_ = [str(n) for n in payload.get("numeric", [])]
        model.ple_ = {k: BSpline.from_dict(v) for k, v in payload.get("ple", {}).items()}
        model.net_ = model._build(torch, len(model.mean_)).double()
        model.net_.load_state_dict(
            {k: torch.as_tensor(v, dtype=torch.float64) for k, v in payload["weights"].items()}
        )
        model.net_.eval()
        model.shift_ = float(payload["shift"])
        model.best_epoch_ = int(payload.get("best_epoch", 0))
        model.history_ = [tuple(h) for h in payload.get("history", [])]
        return model


# ---------------------------------------------------------------------- torch pieces


def _torch() -> Any:
    try:
        import torch  # noqa: PLC0415 — the research extra
    except ImportError as err:  # pragma: no cover — environment dependent
        msg = 'the research track needs torch: pip install "glasshouse[research]"'
        raise ImportError(msg) from err
    return torch


def _mlp(torch: Any, n_in: int, hidden: tuple[int, ...], n_out: int = 1) -> Any:
    """Tanh layers, then a linear output whose weights start at zero: the model is the GLM."""
    layers: list[Any] = []
    width = n_in
    for h in hidden:
        layers += [torch.nn.Linear(width, h), torch.nn.Tanh()]
        width = h
    out = torch.nn.Linear(width, n_out)
    torch.nn.init.zeros_(out.weight)
    torch.nn.init.zeros_(out.bias)
    layers.append(out)
    return torch.nn.Sequential(*layers)


def _network(
    torch: Any,
    kind: str,
    n_in: int,
    hidden: tuple[int, ...],
    slices: list[tuple[str, int, int]],
    grid: int = 5,
) -> Any:
    """Build the correction network of the asked kind.

    Every kind has ``forward`` (the total per row) and ``terms`` (per-column corrections that
    add up to it); ``localglm`` also has ``beta``; ``kan`` has ``first.edge_functions``.
    """
    builders = {"mlp": _mlp_net, "additive": _additive_net, "localglm": _localglm_net}
    if kind == "kan":
        return _kan_net(torch, n_in, hidden, grid)
    if kind not in builders:
        msg = f"network must be one of {[*sorted(builders), 'kan']}, not {kind!r}"
        raise ValueError(msg)
    return builders[kind](torch, n_in, hidden, slices)


def _mlp_net(torch: Any, n_in: int, hidden: tuple[int, ...], _: list[tuple[str, int, int]]) -> Any:
    class Mlp(torch.nn.Module):  # type: ignore[misc]
        """One net on every input: the CANN proper. Its correction is one column."""

        def __init__(self) -> None:
            super().__init__()
            self.body = _mlp(torch, n_in, hidden)

        def forward(self, x: Any) -> Any:
            return self.body(x).squeeze(-1)

        def terms(self, x: Any) -> Any:
            return self.body(x)

    return Mlp()


def _additive_net(
    torch: Any, _: int, hidden: tuple[int, ...], slices: list[tuple[str, int, int]]
) -> Any:
    class Additive(torch.nn.Module):  # type: ignore[misc]
        """One small net per numeric column; a zero-initialised linear map per block."""

        def __init__(self) -> None:
            super().__init__()
            parts = []
            for _, lo, hi in slices:
                if hi - lo == 1:
                    parts.append(_mlp(torch, 1, hidden))
                else:
                    linear = torch.nn.Linear(hi - lo, 1)
                    torch.nn.init.zeros_(linear.weight)
                    torch.nn.init.zeros_(linear.bias)
                    parts.append(linear)
            self.parts = torch.nn.ModuleList(parts)

        def terms(self, x: Any) -> Any:
            pieces = zip(self.parts, slices, strict=True)
            return torch.cat([part(x[:, lo:hi]) for part, (_, lo, hi) in pieces], dim=1)

        def forward(self, x: Any) -> Any:
            return self.terms(x).sum(dim=1)

    return Additive()


def _localglm_net(
    torch: Any, n_in: int, hidden: tuple[int, ...], slices: list[tuple[str, int, int]]
) -> Any:
    class LocalGlm(torch.nn.Module):  # type: ignore[misc]
        """A coefficient per design column, per row: ``sum_j beta_j(x) x_j``."""

        def __init__(self) -> None:
            super().__init__()
            self.body = _mlp(torch, n_in, hidden, n_out=n_in)

        def beta(self, x: Any) -> Any:
            return self.body(x)

        def terms(self, x: Any) -> Any:
            per_column = self.beta(x) * x
            return torch.stack([per_column[:, lo:hi].sum(dim=1) for _, lo, hi in slices], dim=1)

        def forward(self, x: Any) -> Any:
            return (self.beta(x) * x).sum(dim=1)

    return LocalGlm()


def _kan_layer(torch: Any, n_in: int, n_out: int, grid: int, zero: bool) -> Any:
    """Build one KAN layer: every edge ``phi(x) = w_b silu(x) + w_s sum_i c_i B_i(x)``.

    Cubic B-splines on ``grid`` intervals over the standardised range [-3, 3] (inputs
    outside are clamped), the spline coefficients starting at zero and, for the output
    layer, the base weights too, so the network starts as the GLM.
    """
    order = 3
    lo, hi = -3.0, 3.0
    step = (hi - lo) / grid
    knots = torch.linspace(
        lo - order * step, hi + order * step, grid + 2 * order + 1, dtype=torch.float64
    )
    n_basis = grid + order

    class KanLayer(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.base_weight = torch.nn.Parameter(torch.zeros(n_out, n_in, dtype=torch.float64))
            self.spline_weight = torch.nn.Parameter(
                torch.zeros(n_out, n_in, n_basis, dtype=torch.float64)
            )
            if not zero:
                torch.nn.init.kaiming_uniform_(self.base_weight, a=5**0.5)
            self.register_buffer("knots", knots)

        def bases(self, x: Any) -> Any:
            """B-spline basis values, shape (..., n_basis), by Cox-de Boor."""
            x = x.clamp(lo, hi).unsqueeze(-1)
            t = self.knots
            b = ((x >= t[:-1]) & (x < t[1:])).to(x.dtype)
            for k in range(1, order + 1):
                left = (x - t[: -(k + 1)]) / (t[k:-1] - t[: -(k + 1)]) * b[..., :-1]
                right = (t[k + 1 :] - x) / (t[k + 1 :] - t[1:-k]) * b[..., 1:]
                b = left + right
            return b

        def forward(self, x: Any) -> Any:
            base = torch.nn.functional.silu(x) @ self.base_weight.T
            spline = torch.einsum("rib,oib->ro", self.bases(x), self.spline_weight)
            return base + spline

        def edge_functions(self, grid_x: Any) -> Any:
            """``phi_{q,p}`` on a 1-D grid: shape (outputs, inputs, points)."""
            g = grid_x.unsqueeze(-1).expand(len(grid_x), n_in)
            spline = torch.einsum("pib,oib->oip", self.bases(g), self.spline_weight)
            base = torch.nn.functional.silu(grid_x)[None, None, :] * self.base_weight[:, :, None]
            return base + spline

    return KanLayer()


def _kan_net(torch: Any, n_in: int, hidden: tuple[int, ...], grid: int) -> Any:
    width = hidden[0] if hidden else 8

    class Kan(torch.nn.Module):  # type: ignore[misc]
        """Two KAN layers: curves on every edge, sums on every node; one output column."""

        def __init__(self) -> None:
            super().__init__()
            self.first = _kan_layer(torch, n_in, width, grid, zero=False)
            self.second = _kan_layer(torch, width, 1, grid, zero=True)

        def forward(self, x: Any) -> Any:
            return self.second(self.first(x)).squeeze(-1)

        def terms(self, x: Any) -> Any:
            return self.second(self.first(x))

    return Kan()


def _periodic_front(torch: Any, n_in: int, positions: list[int], k: int, sigma: float) -> Any:
    """Build the periodic embedding of Gorishniy et al. as a front layer.

    Each listed column becomes ``concat[sin(2 pi c x), cos(2 pi c x)]`` over ``k`` trainable
    frequencies ``c``, initialised from ``N(0, sigma)``; the other columns pass through.
    """
    numeric = set(positions)
    n_out = n_in + len(positions) * (2 * k - 1)

    class Periodic(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.c = torch.nn.Parameter(torch.randn(len(positions), k, dtype=torch.float64) * sigma)
            self.n_out = n_out

        def forward(self, x: Any) -> Any:
            pieces = []
            j = 0
            for col in range(n_in):
                if col in numeric:
                    v = 2.0 * torch.pi * x[:, col : col + 1] * self.c[j][None, :]
                    pieces += [torch.sin(v), torch.cos(v)]
                    j += 1
                else:
                    pieces.append(x[:, col : col + 1])
            return torch.cat(pieces, dim=1)

        def slices(self, layout: list[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
            """Return the layout after expansion: a periodic column becomes ``2k`` columns."""
            out, at = [], 0
            for name, lo, hi in layout:
                width = sum(2 * k if col in numeric else 1 for col in range(lo, hi))
                out.append((name, at, at + width))
                at += width
            return out

    return Periodic()


def _with_front(torch: Any, front: Any, body: Any) -> Any:
    """Put a network behind a front layer; ``terms`` and ``beta`` pass through it too."""

    class Fronted(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.front = front
            self.body = body

        def forward(self, x: Any) -> Any:
            return self.body(self.front(x))

        def terms(self, x: Any) -> Any:
            return self.body.terms(self.front(x))

        def beta(self, x: Any) -> Any:
            return self.body.beta(self.front(x))

    return Fronted()


def _inverse_link(torch: Any, link: str, eta: Any) -> Any:
    if link == "identity":
        return eta
    if link == "log":
        return torch.exp(eta)
    return torch.sigmoid(eta)


def deviance_torch(
    torch: Any, family: FamilyName, y: Any, mu: Any, w: Any, power: float | None = None
) -> Any:
    """Return the weighted mean unit deviance, differentiable, as ``metrics.deviance`` counts it.

    Written once here for the loss; a test checks it against the Rust deviance to rounding.
    ``y log y`` is kept apart from the ``mu`` terms: ``xlogy(y, y / mu)`` has a NaN gradient
    at ``y = 0``, and a frequency model is mostly zeros.
    """
    if family == "gaussian":
        unit = (y - mu) ** 2
    elif family == "poisson":
        unit = 2.0 * (torch.xlogy(y, y) - y * torch.log(mu) - (y - mu))
    elif family == "gamma":
        unit = 2.0 * ((y - mu) / mu - torch.log(y) + torch.log(mu))
    elif family == "binomial":
        unit = 2.0 * (
            torch.xlogy(y, y)
            + torch.xlogy(1.0 - y, 1.0 - y)
            - y * torch.log(mu)
            - (1.0 - y) * torch.log(1.0 - mu)
        )
    elif family == "tweedie":
        if power is None:
            msg = "tweedie needs a variance power"
            raise ValueError(msg)
        p = power
        unit = 2.0 * (
            torch.where(y > 0, y ** (2.0 - p), torch.zeros_like(y)) / ((1.0 - p) * (2.0 - p))
            - y * mu ** (1.0 - p) / (1.0 - p)
            + mu ** (2.0 - p) / (2.0 - p)
        )
    else:
        msg = f"unknown family {family!r}"
        raise ValueError(msg)
    return torch.sum(w * unit) / torch.sum(w)


@dataclass
class AdditiveNet(CANN):
    """The CANN with one small net per input column: corrections you can draw, no interactions."""

    network: str = "additive"


@dataclass
class LocalGLMnet(CANN):
    """The CANN whose network outputs a coefficient per design column for each row."""

    network: str = "localglm"


__all__ = ["CANN", "AdditiveNet", "LocalGLMnet", "deviance_torch"]
