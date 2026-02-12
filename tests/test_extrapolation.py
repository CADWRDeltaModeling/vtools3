
import pandas as pd
import numpy as np
import pytest
from pandas.testing import assert_series_equal, assert_frame_equal

import pandas as pd
import numpy as np
import pytest
from pandas.testing import assert_series_equal
from vtools import extrapolate_ts

# Core test cases
def test_constant_forward():
    ts = pd.Series([1, 2, 3], index=pd.date_range("2020-01-01", periods=3, freq="D"))
    result = extrapolate_ts(ts, end="2020-01-05", method="constant", val=10)
    expected = pd.Series([1, 2, 3, 10, 10], index=pd.date_range("2020-01-01", periods=5, freq="D"))
    assert_series_equal(result, expected)

def test_constant_backward():
    ts = pd.Series([4, 5, 6], index=pd.date_range("2020-01-03", periods=3, freq="D"))
    result = extrapolate_ts(ts, start="2020-01-01", method="constant", val=0)
    expected = pd.Series([0, 0, 4, 5, 6], index=pd.date_range("2020-01-01", periods=5, freq="D"))
    assert_series_equal(result, expected)

def test_taper_forward():
    ts = pd.Series([5], index=pd.date_range("2020-01-01", periods=1, freq="D"))
    result = extrapolate_ts(ts, end="2020-01-04", method="taper", val=0)
    expected = pd.Series([5, 3.3333, 1.6667, 0], index=pd.date_range("2020-01-01", periods=4, freq="D"))
    assert_series_equal(result.round(4), expected.round(4), check_dtype=False)

def test_taper_backward():
    ts = pd.Series([5], index=pd.date_range("2020-01-04", periods=1, freq="D"))
    result = extrapolate_ts(ts, start="2020-01-01", method="taper", val=0)
    expected = pd.Series([0, 1.6667, 3.3333, 5], index=pd.date_range("2020-01-01", periods=4, freq="D"))
    assert_series_equal(result.round(4), expected.round(4), check_dtype=False)

def test_linear_slope_bidirectional():
    ts = pd.Series([2, 4], index=pd.date_range("2020-01-02", periods=2, freq="D"))
    result = extrapolate_ts(ts, start="2020-01-01", end="2020-01-04", method="linear_slope")
    expected = pd.Series([0, 2, 4, 6], index=pd.date_range("2020-01-01", periods=4, freq="D"))
    assert_series_equal(result.round(4), expected.round(4), check_dtype=False)

# Contract violations
def test_ffill_before_start_error():
    ts = pd.Series([1, 2, 3], index=pd.date_range("2020-01-03", periods=3, freq="D"))
    with pytest.raises(ValueError, match="ffill.*before start"):
        extrapolate_ts(ts, start="2020-01-01", method="ffill")

def test_bfill_after_end_error():
    ts = pd.Series([1, 2, 3], index=pd.date_range("2020-01-01", periods=3, freq="D"))
    with pytest.raises(ValueError, match="bfill.*after end"):
        extrapolate_ts(ts, end="2020-01-05", method="bfill")

def test_taper_without_val():
    ts = pd.Series([1], index=pd.date_range("2020-01-01", periods=1, freq="D"))
    with pytest.raises(ValueError, match="requires 'val'"):
        extrapolate_ts(ts, end="2020-01-02", method="taper")

def test_linear_with_val_error():
    ts = pd.Series([1, 2], index=pd.date_range("2020-01-01", periods=2, freq="D"))
    with pytest.raises(ValueError, match="does not use 'val'"):
        extrapolate_ts(ts, start="2019-12-30", end="2020-01-03", method="linear_slope", val=99)


def test_short_linear_error():
    ts = pd.Series([1], index=pd.date_range("2020-01-01", periods=1, freq="D"))
    with pytest.raises(ValueError, match="2 data points.*required"):
        extrapolate_ts(ts, start="2019-12-30", method="linear_slope")

# Regression test from thread
def test_dtype_preservation():
    ts = pd.Series([1, 2], dtype="int64", index=pd.date_range("2020-01-02", periods=2, freq="D"))
    extended = extrapolate_ts(ts, end="2020-01-04", method="ffill")
    expected = pd.Series([1, 2, 2], index=pd.date_range("2020-01-02", periods=3, freq="D"), dtype="int64")
    assert_series_equal(extended, expected)

def test_fill_preserves_original():
    ts = pd.Series([1, 2, 3], index=pd.date_range("2020-01-01", periods=3, freq="D"))
    result = extrapolate_ts(ts, end="2020-01-05", method="constant", val=5)
    assert result.loc["2020-01-01"] == 1
    assert result.loc["2020-01-04"] == 5



def generate_series(start, periods, freq, values=None):
    """Helper to generate a series with given freq and values."""
    idx = pd.date_range(start=start, periods=periods, freq=freq)
    vals = values if values is not None else np.arange(periods)
    return pd.Series(vals, index=idx)


def test_taper_across_frequencies():
    freqs = ["15min", "h", "D"]
    for freq in freqs:
        ts = generate_series("2020-01-01", periods=2, freq=freq, values=[10, 20])
        interval = ts.index[1] - ts.index[0]
        end_time = ts.index[-1] + 3 * interval
        result = extrapolate_ts(ts, end=end_time, method="taper", val=0.0)
        assert result.index[-1] == end_time


def test_taper_across_frequencies2():
    results = {}
    for freq in ["15min", "h", "D"]:
        ts = generate_series("2020-01-01", periods=2, freq=freq, values=[10, 10])  # Ensure 2+ points
        step = ts.index[1] - ts.index[0]
        end_time = ts.index[-1] + 3 * step
        result = extrapolate_ts(ts, end=end_time, method="taper", val=0.0)
        assert result.index.freqstr == freq
        results[freq] = result

