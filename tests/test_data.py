"""Cleaning is a pure function of the raw frame: test it on a raw-shaped sample, no download."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from glasshouse import data


def _raw_fremtpl2() -> pd.DataFrame:
    """Five rows shaped exactly like OpenML 41214 ships them, including the extremes."""
    return pd.DataFrame(
        {
            "IDpol": [1.0, 3.0, 5.0, 10.0, 11.0],
            "ClaimNb": [1, 0, 16, 2, 0],
            "Exposure": [0.10, 0.77, 2.01, 1.0, 0.5],
            "Area": pd.Categorical(["D", "D", "B", "F", "A"]),
            "VehPower": [5, 6, 15, 4, 7],
            "VehAge": [0, 2, 100, 20, 3],
            "DrivAge": [55, 52, 100, 90, 30],
            "BonusMalus": [50, 50, 230, 150, 60],
            "VehBrand": pd.Categorical(["B12", "B12", "B1", "B3", "B12"]),
            "VehGas": ["'Regular'", "'Regular'", "'Diesel'", "'Diesel'", "'Regular'"],
            "Density": [1217, 1217, 54, 27000, 1],
            "Region": pd.Categorical(["R82", "R82", "R22", "R11", "R91"]),
        }
    )


def test_fremtpl2_cleaning_applies_every_documented_rule() -> None:
    df = data.clean_fremtpl2_freq(_raw_fremtpl2())
    assert df.ClaimNb.tolist() == [1.0, 0.0, 4.0, 2.0, 0.0]  # capped at 4
    assert df.Exposure.tolist() == [0.10, 0.77, 1.0, 1.0, 0.5]  # capped at 1
    assert df.VehPower.max() == 9.0 and df.VehAge.max() == 20.0
    assert df.DrivAge.max() == 90.0 and df.BonusMalus.max() == 150.0
    assert df.VehGas.tolist()[:3] == ["Regular", "Regular", "Diesel"]  # quotes gone
    assert df.AreaCode.tolist() == [4.0, 4.0, 2.0, 6.0, 1.0]
    np.testing.assert_allclose(df.LogDensity, np.log([1217, 1217, 54, 27000, 1]))
    np.testing.assert_allclose(df.Frequency, df.ClaimNb / df.Exposure)
    assert df.IDpol.dtype == np.int64
    for col in ("ClaimNb", "Exposure", "VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"):
        assert df[col].dtype == np.float64, col


def test_cleaning_is_pure_and_documented() -> None:
    raw = _raw_fremtpl2()
    before = raw.copy()
    data.clean_fremtpl2_freq(raw)
    pd.testing.assert_frame_equal(raw, before)  # the raw frame is untouched
    text = data.describe("fremtpl2_freq")
    assert "capped at 4" in text and "Wüthrich" in text and "NOT applied" in text
    with pytest.raises(ValueError, match="unknown dataset"):
        data.describe("titanic")


def test_creditcard_cleaning_parses_the_label() -> None:
    raw = pd.DataFrame({"V1": [0.1, -0.2], "Amount": [1.0, 2.0], "Class": ["'0'", "'1'"]})
    df = data.clean_creditcard(raw)
    assert df.Class.tolist() == [0.0, 1.0] and df.Class.dtype == np.float64


@pytest.mark.skipif(
    not os.environ.get("GLASSHOUSE_NETWORK_TESTS"),
    reason="needs OpenML; set GLASSHOUSE_NETWORK_TESTS=1",
)
def test_load_fremtpl2_from_openml_and_cache() -> None:
    df = data.load("fremtpl2_freq")
    assert len(df) == 678013 and df.ClaimNb.max() == 4.0 and df.Exposure.max() == 1.0
    assert (data.cache_dir() / "fremtpl2_freq.parquet").exists()
