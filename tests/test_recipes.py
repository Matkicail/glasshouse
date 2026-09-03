"""Every registered benchmark recipe must be well-formed without touching the network.

A broken recipe (a misspelled column, a family with no task mapping, a factory that throws)
should fail here in milliseconds, not forty seconds into a data download.
"""

from __future__ import annotations

import pytest

from glasshouse import data
from glasshouse.bench import _TASK_OF_FAMILY
from glasshouse.benchmarks import BENCHMARKS


@pytest.mark.parametrize("name", sorted(BENCHMARKS))
def test_recipe_is_well_formed(name: str) -> None:
    b = BENCHMARKS[name]
    assert b.name == name
    assert b.dataset in data.DATASETS, "recipes must use registered datasets"
    assert b.task.family in _TASK_OF_FAMILY, "the family must map to a report task type"
    assert b.task.target == data.DATASETS[b.dataset].target
    assert all(isinstance(f, str) for f in b.features)
    labels = [m.label for m in b.models]
    assert len(labels) == len(set(labels)), "model labels must be unique"
    for spec in b.models:
        model = spec.make()  # the factory must build a fresh model without side effects
        assert callable(model.fit) and callable(model.predict)
        assert spec.columns and all(isinstance(c, str) for c in spec.columns)
    if b.task.exposure is not None:
        assert b.task.exposure == data.DATASETS[b.dataset].exposure


def test_factories_return_fresh_instances() -> None:
    for b in BENCHMARKS.values():
        for spec in b.models:
            assert spec.make() is not spec.make(), (b.name, spec.label)
