from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from vtools.data.indexing import (
    resolve_common_freq,
    regular_index_from_valid_extent,
    reindex_to_continuous,
    inferred_regular_freq,
    compare_regular_freq,
)


def test_resolve_common_freq_same():
    i1 = pd.date_range("2024-01-01", periods=5, freq="D")
    i2 = pd.date_range("2024-02-01", periods=3, freq="D")
    f = resolve_common_freq([i1, i2], preserve_freq=True)
    assert f == i1.freq


def test_resolve_common_freq_none_when_preserve_false():
    i1 = pd.date_range("2024-01-01", periods=5, freq="D")
    i2 = pd.date_range("2024-02-01", periods=3, freq="2D")
    assert resolve_common_freq([i1, i2], preserve_freq=False) is None


def test_resolve_common_freq_raises_on_mismatch():
    i1 = pd.date_range("2024-01-01", periods=5, freq="D")
    i2 = pd.date_range("2024-02-01", periods=3, freq="2D")
    with pytest.raises(ValueError, match="inconsistent frequencies"):
        resolve_common_freq([i1, i2], preserve_freq=True)


def test_regular_index_from_valid_extent_series():
    idx1 = pd.date_range("2024-01-01", periods=5, freq="D")
    idx2 = pd.date_range("2024-01-03", periods=5, freq="D")
    s1 = pd.Series([np.nan, 1.0, 2.0, 3.0, np.nan], index=idx1)
    s2 = pd.Series([10.0, 11.0, 12.0, 13.0, np.nan], index=idx2)

    out = regular_index_from_valid_extent([s1, s2], idx1.freq)
    expected = pd.date_range("2024-01-02", "2024-01-06", freq="D")
    pd.testing.assert_index_equal(out, expected)


def test_regular_index_from_valid_extent_empty_valid_returns_empty_like_first():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    s1 = pd.Series([np.nan, np.nan, np.nan], index=idx)
    s2 = pd.Series([np.nan, np.nan, np.nan], index=idx)

    out = regular_index_from_valid_extent([s1, s2], idx.freq)
    assert len(out) == 0
    assert isinstance(out, pd.DatetimeIndex)


def test_reindex_to_continuous_regularizes_when_aligned():
    idx = pd.to_datetime(["2024-01-01", "2024-01-03"])
    s = pd.Series([1.0, 3.0], index=idx, name="x")

    out = reindex_to_continuous(s, pd.tseries.frequencies.to_offset("D"))
    expected = pd.Series(
        [1.0, np.nan, 3.0],
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
        name="x",
    )
    pd.testing.assert_series_equal(out, expected)


def test_reindex_to_continuous_returns_original_when_misaligned():
    idx = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:10", "2024-01-01 00:15"])
    s = pd.Series([1.0, 2.0, 3.0], index=idx, name="x")

    out = reindex_to_continuous(s, pd.tseries.frequencies.to_offset("15min"))
    pd.testing.assert_series_equal(out, s)


def test_inferred_regular_freq_datetime_ok():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    s = pd.Series(range(5), index=idx)
    freq, reason = inferred_regular_freq(s)
    assert reason == "ok"
    assert freq == idx.freq


def test_inferred_regular_freq_non_datetime():
    s = pd.Series([1, 2, 3], index=[1, 2, 3])
    freq, reason = inferred_regular_freq(s)
    assert freq is None
    assert reason == "not_datetime_like"


def test_inferred_regular_freq_not_monotonic():
    idx = pd.to_datetime(["2024-01-02", "2024-01-01", "2024-01-03"])
    s = pd.Series([1, 2, 3], index=idx)
    freq, reason = inferred_regular_freq(s)
    assert freq is None
    assert reason == "not_monotonic"


def test_inferred_regular_freq_duplicates():
    idx = pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"])
    s = pd.Series([1, 2, 3], index=idx)
    freq, reason = inferred_regular_freq(s)
    assert freq is None
    assert reason == "duplicates"


def test_inferred_regular_freq_irregular_infer_failed():
    idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-04"])
    s = pd.Series([1, 2, 3], index=idx)
    freq, reason = inferred_regular_freq(s)
    assert freq is None
    assert reason == "infer_failed"


def test_inferred_regular_freq_single_point_degenerate():
    idx = pd.date_range("2024-01-01", periods=1, freq="D")
    s = pd.Series([1.0], index=idx)
    freq, reason = inferred_regular_freq(s)
    assert reason == "degenerate"
    assert freq == pd.Timedelta(0)


def test_compare_regular_freq_both_regular_same():
    i1 = pd.date_range("2024-01-01", periods=5, freq="D")
    i2 = pd.date_range("2024-02-01", periods=3, freq="D")
    s1 = pd.Series(range(5), index=i1)
    s2 = pd.Series(range(3), index=i2)

    status, reason, sf, rf = compare_regular_freq(s1, s2)
    assert status == "both_regular_same"
    assert sf == i1.freq
    assert rf == i2.freq


def test_compare_regular_freq_both_regular_different():
    i1 = pd.date_range("2024-01-01", periods=5, freq="D")
    i2 = pd.date_range("2024-02-01", periods=3, freq="2D")
    s1 = pd.Series(range(5), index=i1)
    s2 = pd.Series(range(3), index=i2)

    status, _, sf, rf = compare_regular_freq(s1, s2)
    assert status == "both_regular_different"
    assert sf == i1.freq
    assert rf == i2.freq


def test_compare_regular_freq_src_irregular():
    s1 = pd.Series([1, 2, 3], index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-04"]))
    s2 = pd.Series([1, 2, 3], index=pd.date_range("2024-01-01", periods=3, freq="D"))

    status, reason, sf, rf = compare_regular_freq(s1, s2)
    assert status == "src_irregular"
    assert sf is None
    assert rf == s2.index.freq
    assert "staged not regular" in reason


def test_compare_regular_freq_dst_irregular():
    s1 = pd.Series([1, 2, 3], index=pd.date_range("2024-01-01", periods=3, freq="D"))
    s2 = pd.Series([1, 2, 3], index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-04"]))

    status, reason, sf, rf = compare_regular_freq(s1, s2)
    assert status == "dst_irregular"
    assert sf == s1.index.freq
    assert rf is None
    assert "repo not regular" in reason


def test_compare_regular_freq_both_irregular():
    s1 = pd.Series([1, 2, 3], index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-04"]))
    s2 = pd.Series([1, 2, 3], index=pd.to_datetime(["2024-02-01", "2024-02-03", "2024-02-04"]))

    status, reason, sf, rf = compare_regular_freq(s1, s2)
    assert status == "both_irregular"
    assert sf is None
    assert rf is None
    assert "both irregular" in reason


def test_compare_regular_freq_single_point_compatible():
    idx1 = pd.date_range("2024-01-01", periods=1, freq="D")
    idx2 = pd.date_range("2024-02-01", periods=1, freq="D")
    s1 = pd.Series([1.0], index=idx1)
    s2 = pd.Series([2.0], index=idx2)

    status, reason, sf, rf = compare_regular_freq(s1, s2)
    assert status == "both_regular_same"
    assert reason == "degenerate_single_point"
    assert sf is None
    assert rf is None