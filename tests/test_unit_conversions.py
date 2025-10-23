import os
import math
import numpy as np
import pandas as pd
import pytest
import importlib.util

import vtools.functions.unit_conversions as uc


# -----------------------------------------------------------------------------
# Linear / affine converters
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("x", [123.456, np.array([0.0, 1.0, 10.0, 123.456])])
def test_m_ft_roundtrip(x):
    ft = uc.m_to_ft(x)
    m = uc.ft_to_m(ft)
    np.testing.assert_allclose(m, x, rtol=0, atol=1e-12)


@pytest.mark.parametrize("x", [7.5, np.array([0.0, 1.0, 2.5, 1000.0])])
def test_cms_cfs_roundtrip(x):
    cfs = uc.cms_to_cfs(x)
    cms = uc.cfs_to_cms(cfs)
    np.testing.assert_allclose(cms, x, rtol=0, atol=1e-12)


def test_temperature_roundtrip():
    c = np.array([-40.0, 0.0, 20.0, 37.0, 100.0])
    f = uc.celsius_to_fahrenheit(c)
    c2 = uc.fahrenheit_to_celsius(f)
    np.testing.assert_allclose(c2, c, rtol=0, atol=1e-12)


# -----------------------------------------------------------------------------
# Series / DataFrame preservation
# -----------------------------------------------------------------------------
@pytest.fixture
def sample_series_df():
    idx = pd.date_range("2020-01-01", periods=4, freq="h")
    ser = pd.Series([0.0, 1.0, 2.0, 3.0], index=idx, name="flow")
    df = pd.DataFrame(
        {"a": [0.0, 1.0, 2.0, 3.0], "b": [10.0, 11.0, 12.0, 13.0]},
        index=idx,
    )
    return ser, df


def test_series_conversion(sample_series_df):
    ser, _ = sample_series_df
    out = uc.convert_units(ser, "m", "ft")
    assert isinstance(out, pd.Series)
    assert out.index.equals(ser.index)
    assert out.name == ser.name
    np.testing.assert_allclose(out.values, ser.values * uc.M2FT, atol=1e-12)


def test_dataframe_conversion(sample_series_df):
    _, df = sample_series_df
    out = uc.convert_units(df, "ft^3 s-1", "m^3 s-1")
    assert isinstance(out, pd.DataFrame)
    assert out.index.equals(df.index)
    assert (out.columns == df.columns).all()
    np.testing.assert_allclose(out.values, df.values * uc.CFS2CMS, atol=1e-12)


# ----------------------------------------------------------------import importlib.util


@pytest.mark.skipif(
    importlib.util.find_spec("cf_units") is None, reason="cf_units not installed"
)
def test_optional_cf_units_backend(monkeypatch):
    x = np.array([0.0, 1.0, 3.0])
    monkeypatch.setenv("VTOOLS_UNITS_BACKEND", "cf_units")
    out = uc.convert_units(x, "m", "ft")
    np.testing.assert_allclose(out, x * uc.M2FT, atol=1e-12)
    monkeypatch.delenv("VTOOLS_UNITS_BACKEND", raising=False)


# Aliases, backend consistency
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("alias", ["cfs", "ft3/s", "ft^3/s"])
def test_aliases_cfs_to_cms(alias):
    x = np.array([0.0, 35.31466621, 353.1466621])
    out = uc.convert_units(x, alias, "m^3 s-1")
    expected = x * uc.CFS2CMS
    np.testing.assert_allclose(out, expected, atol=1e-12)


@pytest.mark.parametrize("alias", ["cms", "m3/s", "m^3/s"])
def test_aliases_cms_to_cfs(alias):
    x = np.array([0.0, 1.0, 10.0])
    out = uc.convert_units(x, alias, "ft^3 s-1")
    expected = x * uc.CMS2CFS
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_aliases_temperature():
    x = np.array([32.0, 212.0])
    out = uc.convert_units(x, "deg F", "degC")
    expected = uc.fahrenheit_to_celsius(x)
    np.testing.assert_allclose(out, expected, atol=1e-12)


@pytest.mark.parametrize("alias", ["us/cm", "μs/cm", "micromhos/cm"])
def test_aliases_conductivity(alias):
    x = np.array([0.0, 500.0, 5000.0, 50000.0])
    out = uc.convert_units(x, alias, "psu")
    direct = uc.ec_psu_25c(x, hill_correction=True)
    np.testing.assert_allclose(out, direct, atol=1e-12)


def test_backend_consistency_with_constants():
    x = np.array([0.0, 1.23, 10.0, 123.45])

    # m <-> ft
    np.testing.assert_allclose(uc.convert_units(x, "m", "ft"), x * uc.M2FT, atol=1e-12)
    np.testing.assert_allclose(uc.convert_units(x, "ft", "m"), x * uc.FT2M, atol=1e-12)

    # cfs <-> cms
    np.testing.assert_allclose(
        uc.convert_units(x, "ft^3 s-1", "m^3 s-1"), x * uc.CFS2CMS, atol=1e-12
    )
    np.testing.assert_allclose(
        uc.convert_units(x, "m^3 s-1", "ft^3 s-1"), x * uc.CMS2CFS, atol=1e-12
    )


# @pytest.mark.skipif(
#    not pytest.importorskip("cf_units", reason="cf_units not installed"),
#    reason="cf_units not available",
# )
# def test_optional_cf_units_backend(monkeypatch):
#    x = np.array([0.0, 1.0, 3.0])
#    monkeypatch.setenv("VTOOLS_UNITS_BACKEND", "cf_units")
#    out = uc.convert_units(x, "m", "ft")
#    np.testing.assert_allclose(out, x * uc.M2FT, atol=1e-12)
#    monkeypatch.delenv("VTOOLS_UNITS_BACKEND", raising=False)


# -----------------------------------------------------------------------------
# EC↔PSU consistency
# -----------------------------------------------------------------------------
def test_convert_units_matches_ec_psu():
    ec = np.array([0.0, 100.0, 1000.0, 50000.0])
    a = uc.convert_units(ec, "us/cm", "psu")
    b = uc.ec_psu_25c(ec, hill_correction=True)
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_convert_units_matches_psu_ec():
    psu = np.array([0.0, 1.0, 10.0, 33.0])
    a = uc.convert_units(psu, "psu", "us/cm")
    b = uc.psu_ec_25c(psu, refine=True, hill_correction=True)
    np.testing.assert_allclose(a, b, atol=1e-9)  # root-finder tolerance


def test_negative_ec_behavior():
    # Scalar negative → NaN
    assert math.isnan(uc.ec_psu_25c(-5.0))
    assert math.isnan(uc.convert_units(-5.0, "us/cm", "psu"))

    # Array: negative entries become NaN
    ec = np.array([-5.0, 0.0, 10.0])
    out1 = uc.ec_psu_25c(ec)
    out2 = uc.convert_units(ec, "us/cm", "psu")
    assert np.isnan(out1[0]) and np.isnan(out2[0])
    np.testing.assert_allclose(out1[1:], out2[1:], atol=1e-12)
