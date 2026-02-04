#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
import pandas as pd
import numpy as np
from vtools.data.timeseries import is_regular

# Irregular DatetimeIndex Series
irreg_datetime_series = pd.Series(
    data=[1, 2, 3, 4],
    index=pd.DatetimeIndex(
        [
            pd.Timestamp("2025-01-01 00:00"),
            pd.Timestamp("2025-01-01 01:00"),
            pd.Timestamp("2025-01-01 03:00"),  # Gap: should be 02:00
            pd.Timestamp("2025-01-01 04:00"),
        ]
    ),
)


# Regular DatetimeIndex Series
reg_datetime_series = pd.Series(
    data=[1, 2, 3, 4],
    index=pd.DatetimeIndex(
        [
            pd.Timestamp("2025-01-01 00:00"),
            pd.Timestamp("2025-01-01 01:00"),
            pd.Timestamp("2025-01-01 02:00"),
            pd.Timestamp("2025-01-01 03:00"),
        ]
    ),
)

# Irregular Float Index Series
irreg_float_series = pd.Series(
    data=[1, 2, 3, 4],
    index=pd.Index(
        [0.0, 90.0, 180.0, 300.0], dtype="float64"
    ),  # Last diff is 120, not 90
)

# Regular Float Index Series
reg_float_series = pd.Series(
    data=[1, 2, 3, 4],
    index=pd.Index([0.0, 90.0, 180.0, 270.0], dtype="float64"),  # All diffs are 90.0
)


# Irregular Int Index Series
irreg_int_series = pd.Series(
    data=[1, 2, 3, 4],
    index=pd.Index([0, 100, 200, 350], dtype="int64"),  # Last diff is 150, not 100
)

# Regular Int Index Series
reg_int_series = pd.Series(
    data=[1, 2, 3, 4],
    index=pd.Index([0, 100, 200, 300], dtype="int64"),  # All diffs are 100
)


# ============================================================================
# Tests
# ============================================================================
def test_irregular_datetime_series():
    """Test that irregular datetime series is identified as not regular"""
    assert is_regular(irreg_datetime_series) is False


def test_regular_datetime_series():
    """Test that regular datetime series is identified as regular"""
    assert is_regular(reg_datetime_series) is True


def test_irregular_float_series():
    """Test that irregular float index series is identified as not regular"""
    assert is_regular(irreg_float_series) is False


def test_regular_float_series():
    """Test that regular float index series is identified as regular"""
    assert is_regular(reg_float_series) is True


def test_irregular_int_series():
    """Test that irregular int index series is identified as not regular"""
    assert is_regular(irreg_int_series) is False


def test_regular_int_series():
    """Test that regular int index series is identified as regular"""
    assert is_regular(reg_int_series) is True


def test_irregular_datetime_raises():
    """Test that raise_exception=True raises for irregular datetime"""
    try:
        is_regular(irreg_datetime_series, raise_exception=True)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_regular_datetime_raises():
    """Test that raise_exception=True does not raise for regular datetime"""
    result = is_regular(reg_datetime_series, raise_exception=True)
    assert result is True


def test_irregular_float_raises():
    """Test that raise_exception=True raises for irregular float"""
    try:
        is_regular(irreg_float_series, raise_exception=True)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_regular_float_raises():
    """Test that raise_exception=True does not raise for regular float"""
    result = is_regular(reg_float_series, raise_exception=True)
    assert result is True


def test_irregular_int_raises():
    """Test that raise_exception=True raises for irregular int"""
    try:
        is_regular(irreg_int_series, raise_exception=True)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_regular_int_raises():
    """Test that raise_exception=True does not raise for regular int"""
    result = is_regular(reg_int_series, raise_exception=True)
    assert result is True
