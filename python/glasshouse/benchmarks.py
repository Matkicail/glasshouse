"""Named benchmarks: the recipes anyone can rerun with ``glasshouse bench <name>``.

Each entry says which dataset, which task, which models (as factories), and which split.
The committed ``benchmarks/<name>/report.{json,md}`` are what these produced; a test pins
them so a change in the numbers is a change someone has to explain.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from glasshouse import data, splits
from glasshouse.bench import BenchResult, ModelSpec, TaskSpec, run
from glasshouse.encoders import BSpline, Interaction, OneHot, Smooth
from glasshouse.foss import GlumPoisson, SklearnPoisson
from glasshouse.gbdt import LightGBM
from glasshouse.glm import GLM


@dataclass(frozen=True)
class Benchmark:
    """A named recipe."""

    name: str
    dataset: str
    task: TaskSpec
    models: list[ModelSpec]
    make_splits: Callable[[Any], splits.Splits]
    features: list[str] = ()  # type: ignore[assignment]
    time: str | None = None  # residuals over this column; set it whenever the split is time-ordered


def _fremtpl2_models() -> list[ModelSpec]:
    return [
        ModelSpec(
            "glm_simple",
            lambda: GLM(family="poisson", terms={"Area": "onehot", "VehGas": "onehot"}),
            ["Area", "VehGas", "DrivAge", "BonusMalus"],
        ),
        ModelSpec(
            "glm_full",
            lambda: GLM(
                family="poisson",
                terms={
                    "Area": "onehot",
                    "VehGas": "onehot",
                    "VehBrand": "onehot",
                    "Region": "target",
                },
            ),
            [
                "Area",
                "VehGas",
                "VehBrand",
                "Region",
                "DrivAge",
                "VehAge",
                "VehPower",
                "BonusMalus",
                "LogDensity",
            ],
        ),
    ]


_FOSS_ONEHOT = ["Area", "VehGas", "VehBrand", "Region"]
_FOSS_COLUMNS = [
    "Area",
    "VehGas",
    "VehBrand",
    "Region",
    "DrivAge",
    "VehAge",
    "VehPower",
    "BonusMalus",
    "LogDensity",
]


def _smooth_glm() -> GLM:
    """Build the bar the research track has to beat: GCV smooths, one held monotone."""
    return GLM(
        family="poisson",
        terms={
            "Area": "onehot",
            "VehGas": "onehot",
            "VehBrand": "onehot",
            "Region": "target",
            "DrivAge": "smooth",
            "VehAge": "smooth",
            # the business rule: a premium must not fall as the bonus-malus rises. Free, the
            # smooth wiggles where the high-BM data is thin (19 dips over 50..150 on fold 0);
            # held, it does not, on fewer edf and the same held-out deviance
            "BonusMalus": Smooth(monotone="increasing"),
            "LogDensity": "smooth",
        },
    )


def _interaction_glm() -> GLM:
    """Build the smooth GLM plus the one interaction the two-feature A/E grid pointed at."""
    glm = _smooth_glm()
    assert glm.terms is not None
    glm.terms = {**glm.terms, "DrivAge*BonusMalus": Interaction(df=4)}
    return glm


def _research(network: str, encoding: str = "design") -> Callable[[], Any]:
    """Build a research model: the smooth GLM frozen, this kind of net on its residual."""

    def make() -> Any:
        from glasshouse.research import CANN  # noqa: PLC0415 — the research extra (torch)

        hidden = (8,) if network == "kan" else (20, 15, 10)
        return CANN(
            family="poisson",
            glm=_smooth_glm,
            hidden=hidden,
            epochs=100,
            network=network,
            encoding=encoding,
        )

    return make


def _foss_models() -> list[ModelSpec]:
    """Ours vs glum vs scikit-learn on the identical one-hot design: a solver comparison."""
    terms = dict.fromkeys(_FOSS_ONEHOT, "onehot")
    return [
        ModelSpec(
            "glasshouse",
            lambda: GLM(family="poisson", terms=dict(terms)),
            list(_FOSS_COLUMNS),
        ),
        ModelSpec("glum", lambda: GlumPoisson(onehot=list(_FOSS_ONEHOT)), list(_FOSS_COLUMNS)),
        ModelSpec(
            "sklearn", lambda: SklearnPoisson(onehot=list(_FOSS_ONEHOT)), list(_FOSS_COLUMNS)
        ),
    ]


_SEV_COLUMNS = [
    "Area",
    "VehGas",
    "VehBrand",
    "Region",
    "DrivAge",
    "VehAge",
    "VehPower",
    "BonusMalus",
]
_BIKE_COLUMNS = [
    "season",
    "weather",
    "hour",
    "workingday",
    "holiday",
    "temp",
    "humidity",
    "windspeed",
]
_TELCO_CATEGORICAL = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
# TotalCharges is left out: it is tenure times MonthlyCharges to within rounding, so it says
# nothing the other two do not, and its near-collinearity with them is what a coordinate
# descent crawls on. The penalty is on the raw scale (glmnet's convention with standardisation
# off), so the numerics are standardised: otherwise a column in the tens sets alpha_max and
# the 0/1 columns are crushed at every alpha on the path. The plain logistic gets the same
# design, so the only difference between the two rows is the penalty.
_TELCO_COLUMNS = [*_TELCO_CATEGORICAL, "SeniorCitizen", "tenure", "MonthlyCharges"]
_TELCO_TERMS = {
    **dict.fromkeys(_TELCO_CATEGORICAL, "onehot"),
    **dict.fromkeys(["tenure", "MonthlyCharges"], "standardize"),
}

BENCHMARKS: dict[str, Benchmark] = {
    "fremtpl2_glm": Benchmark(
        name="fremtpl2_glm",
        dataset="fremtpl2_freq",
        task=TaskSpec(family="poisson", target="ClaimNb", exposure="Exposure", rate=True),
        models=_fremtpl2_models(),
        make_splits=lambda df: splits.stratified((df.ClaimNb > 0).astype(int), k=5, seed=0),
        features=["Region", "DrivAge", "VehBrand", "BonusMalus"],
    ),
    "fremtpl2_challengers": Benchmark(
        name="fremtpl2_challengers",
        dataset="fremtpl2_freq",
        task=TaskSpec(family="poisson", target="ClaimNb", exposure="Exposure", rate=True),
        models=[
            _fremtpl2_models()[1],  # glm_full
            ModelSpec(
                "glm_splines",
                lambda: GLM(
                    family="poisson",
                    terms={
                        "Area": "onehot",
                        "VehGas": "onehot",
                        "VehBrand": "onehot",
                        "Region": "target",
                        "DrivAge": BSpline(df=6),
                        "VehAge": BSpline(df=5),
                        "BonusMalus": BSpline(df=5),
                        "LogDensity": BSpline(df=4),
                    },
                ),
                [
                    "Area",
                    "VehGas",
                    "VehBrand",
                    "Region",
                    "DrivAge",
                    "VehAge",
                    "VehPower",
                    "BonusMalus",
                    "LogDensity",
                ],
            ),
            ModelSpec("glm_smooth", _smooth_glm, list(_FOSS_COLUMNS)),
            ModelSpec("glm_interaction", _interaction_glm, list(_FOSS_COLUMNS)),
            ModelSpec(
                "lightgbm",
                lambda: LightGBM(
                    family="poisson", categorical=["Area", "VehGas", "VehBrand", "Region"]
                ),
                [
                    "Area",
                    "VehGas",
                    "VehBrand",
                    "Region",
                    "DrivAge",
                    "VehAge",
                    "VehPower",
                    "BonusMalus",
                    "LogDensity",
                ],
            ),
        ],
        make_splits=lambda df: splits.stratified((df.ClaimNb > 0).astype(int), k=5, seed=0),
        features=["Region", "DrivAge", "VehBrand", "BonusMalus"],
    ),
    "creditcard_glm": Benchmark(
        name="creditcard_glm",
        dataset="creditcard",
        task=TaskSpec(family="binomial", target="Class"),
        models=[
            ModelSpec(
                "logistic",
                lambda: GLM(family="binomial"),
                [f"V{i}" for i in range(1, 29)] + ["Amount"],
            ),
        ],
        make_splits=lambda df: splits.stratified(df.Class.astype(int), k=5, seed=0),
        features=["Amount"],
    ),
    "fremtpl2_sev": Benchmark(
        name="fremtpl2_sev",
        dataset="fremtpl2_sev",
        task=TaskSpec(family="gamma", target="Severity", exposure="ClaimCount"),
        models=[
            ModelSpec(
                "glm_gamma",
                lambda: GLM(
                    family="gamma",
                    terms={
                        "Area": "onehot",
                        "VehGas": "onehot",
                        "VehBrand": "onehot",
                        "Region": "target",
                        "DrivAge": BSpline(df=5),
                        "VehAge": BSpline(df=4),
                        "BonusMalus": BSpline(df=4),
                    },
                ),
                list(_SEV_COLUMNS),
            ),
            ModelSpec(
                "lightgbm",
                lambda: LightGBM(
                    family="gamma", categorical=["Area", "VehGas", "VehBrand", "Region"]
                ),
                list(_SEV_COLUMNS),
            ),
        ],
        make_splits=lambda df: splits.kfold(len(df), k=5, seed=0),
        features=["Region", "DrivAge", "VehBrand", "BonusMalus"],
    ),
    "bike_sharing": Benchmark(
        name="bike_sharing",
        dataset="bike_sharing",
        task=TaskSpec(family="poisson", target="count"),
        models=[
            ModelSpec(
                "glm_poisson",
                lambda: GLM(
                    family="poisson",
                    # train strictly before test means a season the model has never seen
                    # (winter is not in the first 30 % of 2011): encode it as the reference
                    # rather than refuse, and let the report show what that costs
                    terms={
                        "season": OneHot(unknown="zero"),
                        "weather": OneHot(unknown="zero"),
                        "hour": BSpline(df=12),
                        "temp": BSpline(df=4),
                        "humidity": BSpline(df=4),
                    },
                ),
                list(_BIKE_COLUMNS),
            ),
            ModelSpec(
                "lightgbm",
                lambda: LightGBM(family="poisson", categorical=["season", "weather"]),
                list(_BIKE_COLUMNS),
            ),
        ],
        # train strictly before test: the first 30 % of hours, then five consecutive blocks
        make_splits=lambda df: splits.time_ordered(df.hour_index, n_folds=5),
        features=["hour", "temp", "weather", "workingday"],
        time="hour_index",
    ),
    "telco_churn": Benchmark(
        name="telco_churn",
        dataset="telco_churn",
        task=TaskSpec(family="binomial", target="Churn"),
        models=[
            ModelSpec(
                "logistic",
                lambda: GLM(family="binomial", terms=dict(_TELCO_TERMS)),
                list(_TELCO_COLUMNS),
            ),
            ModelSpec(
                "lasso_logistic",
                lambda: GLM(
                    family="binomial",
                    terms=dict(_TELCO_TERMS),
                    alpha="cv",
                    l1_ratio=1.0,
                    alpha_rule="min",  # the prediction rule; "1se" is for a sparser story
                ),
                list(_TELCO_COLUMNS),
            ),
            ModelSpec(
                # a factor leaves whole or stays whole: the review's own notion of dropping one
                "group_lasso_logistic",
                lambda: GLM(
                    family="binomial",
                    terms=dict(_TELCO_TERMS),
                    alpha="cv",
                    l1_ratio=1.0,
                    alpha_rule="1se",
                    group_lasso=True,
                ),
                list(_TELCO_COLUMNS),
            ),
        ],
        make_splits=lambda df: splits.stratified(df.Churn.astype(int), k=5, seed=0),
        features=["Contract", "tenure", "InternetService", "MonthlyCharges"],
    ),
    "fremtpl2_cann": Benchmark(
        # the research fence's go/no-go: a model must beat the smooth GLM on held-out
        # deviance AND calibration on these splits, or it stays a notebook. The three nets
        # share one class and one training loop; only the correction's shape differs
        name="fremtpl2_cann",
        dataset="fremtpl2_freq",
        task=TaskSpec(family="poisson", target="ClaimNb", exposure="Exposure", rate=True),
        models=[
            ModelSpec("glm_smooth", _smooth_glm, list(_FOSS_COLUMNS)),
            ModelSpec("glm_interaction", _interaction_glm, list(_FOSS_COLUMNS)),
            ModelSpec("cann", _research("mlp"), list(_FOSS_COLUMNS)),
            ModelSpec("additive", _research("additive"), list(_FOSS_COLUMNS)),
            ModelSpec("localglm", _research("localglm"), list(_FOSS_COLUMNS)),
            ModelSpec(
                "lightgbm",
                lambda: LightGBM(
                    family="poisson", categorical=["Area", "VehGas", "VehBrand", "Region"]
                ),
                list(_FOSS_COLUMNS),
            ),
        ],
        make_splits=lambda df: splits.stratified((df.ClaimNb > 0).astype(int), k=5, seed=0),
        features=["Region", "DrivAge", "VehBrand", "BonusMalus"],
    ),
    "fremtpl2_kan": Benchmark(
        # the fence's last stage, and the feature-encoding trade-off as a picture: the same
        # smooth GLM under an MLP and under a two-layer KAN, each fed the numeric features
        # raw, as piecewise linear bins, and as periodic (sin/cos) embeddings
        name="fremtpl2_kan",
        dataset="fremtpl2_freq",
        task=TaskSpec(family="poisson", target="ClaimNb", exposure="Exposure", rate=True),
        models=[
            ModelSpec("glm_smooth", _smooth_glm, list(_FOSS_COLUMNS)),
            ModelSpec("cann", _research("mlp"), list(_FOSS_COLUMNS)),
            ModelSpec("cann_piecewise", _research("mlp", "piecewise"), list(_FOSS_COLUMNS)),
            ModelSpec("cann_periodic", _research("mlp", "periodic"), list(_FOSS_COLUMNS)),
            ModelSpec("kan_raw", _research("kan", "raw"), list(_FOSS_COLUMNS)),
            ModelSpec("kan_piecewise", _research("kan", "piecewise"), list(_FOSS_COLUMNS)),
            ModelSpec("kan_periodic", _research("kan", "periodic"), list(_FOSS_COLUMNS)),
            ModelSpec(
                "lightgbm",
                lambda: LightGBM(
                    family="poisson", categorical=["Area", "VehGas", "VehBrand", "Region"]
                ),
                list(_FOSS_COLUMNS),
            ),
        ],
        make_splits=lambda df: splits.stratified((df.ClaimNb > 0).astype(int), k=5, seed=0),
        features=["Region", "DrivAge", "VehBrand", "BonusMalus"],
    ),
    "fremtpl2_vs_foss": Benchmark(
        name="fremtpl2_vs_foss",
        dataset="fremtpl2_freq",
        task=TaskSpec(family="poisson", target="ClaimNb", exposure="Exposure", rate=True),
        models=_foss_models(),
        make_splits=lambda df: splits.stratified((df.ClaimNb > 0).astype(int), k=5, seed=0),
        features=["Region", "DrivAge", "VehBrand", "BonusMalus"],
    ),
}


def run_named(name: str, *, progress: bool = False) -> BenchResult:
    """Load the data, build the split, run the recipe."""
    if name not in BENCHMARKS:
        msg = f"unknown benchmark {name!r}: one of {sorted(BENCHMARKS)}"
        raise ValueError(msg)
    b = BENCHMARKS[name]
    if progress:
        import sys  # noqa: PLC0415

        from glasshouse.data import cache_dir  # noqa: PLC0415

        cached = (cache_dir() / f"{b.dataset}.parquet").exists()
        note = "from cache" if cached else "from OpenML, roughly a minute the first time"
        sys.stderr.write(f"loading {b.dataset} ({note})\n")
    df = data.load(b.dataset)
    return run(
        df,
        b.task,
        b.models,
        b.make_splits(df),
        dataset=b.dataset,
        describe=data.describe(b.dataset),
        features=list(b.features),
        time=b.time,
        progress=progress,
    )


__all__ = ["BENCHMARKS", "Benchmark", "run_named"]
