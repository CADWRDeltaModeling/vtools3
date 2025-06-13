import pytest
import pandas as pd
import numpy as np
from vtools import transition_ts


def pink_noise(size, alpha=1.0):
    f = np.fft.rfftfreq(size)
    f[0] = 1e-6
    spectrum = 1 / f ** (alpha / 2.0)
    phases = np.exp(2j * np.pi * np.random.rand(len(f)))
    signal = np.fft.irfft(spectrum * phases, n=size)
    return signal / np.std(signal)


@pytest.fixture
def test_series():
    freq = "15min"
    date0 = pd.date_range("2023-01-01", "2023-06-01", freq=freq)
    date1 = pd.date_range("2023-05-20", "2023-09-01", freq=freq)
    t0 = np.linspace(0, 1, len(date0))
    t1 = np.linspace(0, 1, len(date1))
    ts0 = pd.Series(
        1.0 + 0.5 * pink_noise(len(date0)) + 0.1 * np.sin(2 * np.pi * 3 * t0),
        index=date0,
    )
    ts1 = pd.Series(
        2.0 + 0.4 * pink_noise(len(date1)) + 0.15 * np.cos(2 * np.pi * 2 * t1),
        index=date1,
    )
    return ts0, ts1


def test_natural_gap_linear(test_series):
    ts0, ts1 = test_series
    # Artificially move ts1 earlier to reduce transition zone to < 2 steps
    ts1 = ts1.copy()
    ts1.index = ts1.index - pd.Timedelta("5d")

    with pytest.raises(ValueError, match="at least two steps"):
        transition_ts(ts0, ts1, method="linear", create_gap=None, return_type="series")


def test_explicit_gap_linear(test_series):
    ts0, ts1 = test_series
    gap = ["2023-05-15", "2023-06-10"]
    result = transition_ts(
        ts0, ts1, method="linear", create_gap=gap, return_type="series"
    )
    glue = transition_ts(ts0, ts1, method="linear", create_gap=gap, return_type="glue")
    assert glue.index[0] == pd.Timestamp(gap[0])
    assert glue.index[-1] == pd.Timestamp(gap[1])


def test_explicit_gap_pchip_with_overlap(test_series):
    ts0, ts1 = test_series
    gap = ["2023-05-15", "2023-06-10"]
    result = transition_ts(
        ts0, ts1, method="pchip", create_gap=gap, overlap=(10, 10), return_type="series"
    )
    assert result.index.is_monotonic_increasing
    assert not result.index.duplicated().any()


def test_gap_inside_natural_gap(test_series):
    ts0, ts1 = test_series
    gap = ["2023-05-10", "2023-05-15"]
    transition = transition_ts(
        ts0, ts1, method="linear", create_gap=gap, return_type="glue"
    )
    diff = np.diff(transition.values)
    assert np.std(diff) < 0.5


def test_overlapping_series_with_gap(test_series):
    ts0, ts1 = test_series
    gap = ["2023-05-25", "2023-05-28"]
    result = transition_ts(
        ts0, ts1, method="pchip", create_gap=gap, overlap=(12, 12), return_type="series"
    )
    assert result.index.is_monotonic_increasing
    assert not result.index.duplicated().any()


def test_glue_return_type(test_series):
    ts0, ts1 = test_series
    gap = ["2023-05-15", "2023-06-10"]
    glue = transition_ts(ts0, ts1, method="linear", create_gap=gap, return_type="glue")
    assert glue.index[0] >= pd.Timestamp(gap[0])
    assert glue.index[-1] <= pd.Timestamp(gap[1])


def test_string_overlap_pchip(test_series):
    ts0, ts1 = test_series
    gap = ["2023-05-20", "2023-06-01"]
    result = transition_ts(
        ts0,
        ts1,
        method="pchip",
        create_gap=gap,
        overlap=("3h", "4h"),
        return_type="series",
    )
    assert result.index.is_monotonic_increasing
    assert not result.index.duplicated().any()
    assert result.index[0] <= pd.Timestamp("2023-05-20")
    assert result.index[-1] >= pd.Timestamp("2023-06-01")


def test_mixed_overlap_types(test_series):
    ts0, ts1 = test_series
    gap = ["2023-05-18", "2023-06-03"]
    result = transition_ts(
        ts0, ts1, method="pchip", create_gap=gap, overlap=("1h", 8), return_type="glue"
    )
    assert pd.Timestamp(gap[0]) in result.index
    assert pd.Timestamp(gap[1]) in result.index
