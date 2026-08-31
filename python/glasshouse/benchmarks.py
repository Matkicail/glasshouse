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


BENCHMARKS: dict[str, Benchmark] = {
    "fremtpl2_glm": Benchmark(
        name="fremtpl2_glm",
        dataset="fremtpl2_freq",
        task=TaskSpec(family="poisson", target="ClaimNb", exposure="Exposure", rate=True),
        models=_fremtpl2_models(),
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
        progress=progress,
    )


__all__ = ["BENCHMARKS", "Benchmark", "run_named"]
