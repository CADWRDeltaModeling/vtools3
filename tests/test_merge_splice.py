import pytest
import numpy as np
import pandas as pd
from vtools.functions.merge import ts_merge, ts_splice
from vtools.data.vtime import hours, days
from vtools.functions.blend import ts_blend 


# ----------------------------------------------------------------------
# Fixtures and Helper Functions
# ----------------------------------------------------------------------

@pytest.fixture
def sample_data():
    """Create sample time series data for tests."""
    idx1 = pd.date_range("2023-01-01", periods=5, freq="d")
    idx2 = pd.date_range("2023-01-03", periods=5, freq="d")
    series1 = pd.Series([1, 2, np.nan, 4, 5], index=idx1, name="A")
    series2 = pd.Series([10, 20, 30, np.nan, 50], index=idx2, name="A")
    df1 = pd.DataFrame({"A": [1, np.nan, 3, 4, 5]}, index=idx1)
    df2 = pd.DataFrame({"A": [10, 20, np.nan, 40, 50]}, index=idx2)
    df_multi1 = pd.DataFrame({
        "A": [1, 2, np.nan, 4, 5],
        "B": [10, np.nan, 30, 40, 50]
    }, index=idx1)
    df_multi2 = pd.DataFrame({
        "A": [100, 200, 300, np.nan, 500],
        "B": [1000, 2000, np.nan, 4000, 5000]
    }, index=idx2)
    return {
        "series1": series1,
        "series2": series2,
        "df1": df1,
        "df2": df2,
        "df_multi1": df_multi1,
        "df_multi2": df_multi2
    }

@pytest.fixture
def irregular_sample_data():
    """Create irregularly spaced time series data for testing."""
    idx1 = pd.to_datetime(["2023-01-01", "2023-01-03", "2023-01-07", "2023-01-10"])
    idx2 = pd.to_datetime(["2023-01-02", "2023-01-04", "2023-01-08", "2023-01-11"])
    series1 = pd.Series([1., 2, 3, 4], index=idx1, name="A")
    series2 = pd.Series([10., 20, 30, 40], index=idx2, name="A")
    df1 = pd.DataFrame({"A": [1., 2, 3, 4]}, index=idx1)
    df2 = pd.DataFrame({"A": [10., 20, 30, 40]}, index=idx2)
    return {
        "series1": series1,
        "series2": series2,
        "df1": df1,
        "df2": df2
    }

# Helper functions to generate random time series/dataframe data.
# (These are kept as-is; you might later choose to make them fixtures if needed.)

def get_test_dataframes_irregular(names, ts_len):
    dfs = []
    last_end = 0
    last_interval = []
    s1 = pd.to_datetime('1970-01-01')
    for i in range(len(names)):
        if i > 0:
            start = last_end
            dr = pd.date_range(start=start, periods=ts_len)
        else:
            dr = pd.date_range(start=s1, periods=ts_len)
        last_end = dr[-1]
        intervals = np.random.randint(1, 10, size=ts_len)
        if i > 0 and intervals[0] <= last_interval[-1]:
            intervals[0] = last_interval[-1] + 1
        last_interval = intervals
        inter_lst = pd.to_timedelta(intervals, unit="h")
        dr = dr + inter_lst
        if isinstance(names[i], str):
            data = np.random.randint(1, 11, size=(ts_len, 1))
        else:
            data = np.random.randint(1, 11, size=(ts_len, len(names[i])))
        df = pd.DataFrame(index=dr, data=data, columns=[names[i]])
        dfs.append(df)
    return dfs

def get_test_series(names, ts_len, overlap):
    dfs = []
    last_end = 0
    for i in range(len(names)):
        if i > 0:
            dr = pd.date_range(last_end - hours(overlap - 1),
                               freq="h", periods=ts_len)
        else:
            dr = pd.date_range(pd.Timestamp(2000, 1, 1) + hours(i * ts_len),
                               freq="h", periods=ts_len)
        last_end = dr[-1]
        if isinstance(names[i], str):
            data = np.random.randint(1, 11, size=(ts_len, 1))
        else:
            data = np.random.randint(1, 11, size=(ts_len, len(names[i])))
        df = pd.DataFrame(index=dr, data=data, columns=[names[i]])
        dfs.append(df.squeeze(axis=1))
    return dfs

# ----------------------------------------------------------------------
# Tests for ts_merge
# ----------------------------------------------------------------------

class TestTsMerge:
    def test_series_merge(self, sample_data):
        s1, s2 = sample_data["series1"], sample_data["series2"]
        result = ts_merge((s1, s2), names="A")
        expected = pd.Series(
            [1, 2, 10, 4, 5, np.nan, 50],
            index=pd.date_range("2023-01-01", periods=7, freq="d"),
            name="A"
        )
        pd.testing.assert_series_equal(result, expected)

    def test_single_column_dataframe(self, sample_data):
        df1, df2 = sample_data["df1"], sample_data["df2"]
        result = ts_merge((df1, df2))
        expected = pd.DataFrame(
            {"A": [1., np.nan, 3., 4., 5., 40., 50.]},
            index=pd.date_range("2023-01-01", periods=7, freq="d")
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_multi_column_dataframe(self, sample_data):
        df_multi1, df_multi2 = sample_data["df_multi1"], sample_data["df_multi2"]
        result = ts_merge((df_multi1, df_multi2))
        expected = pd.DataFrame({
            "A": [1., 2., 100, 4, 5, np.nan, 500],
            "B": [10., np.nan, 30, 40, 50, 4000, 5000]
        }, index=pd.date_range("2023-01-01", periods=7, freq="d"))
        pd.testing.assert_frame_equal(result, expected)

    def test_series_merge_with_names(self, sample_data):
        s1, s2 = sample_data["series1"], sample_data["series2"]
        result = ts_merge((s1, s2), names="new_name")
        expected = pd.Series(
            [1., 2., 10, 4, 5, np.nan, 50],
            index=pd.date_range("2023-01-01", periods=7, freq="d"),
            name="new_name"
        )
        pd.testing.assert_series_equal(result, expected)

    def test_dataframe_merge_with_renaming(self, sample_data):
        df1, df2 = sample_data["df1"], sample_data["df2"]
        result = ts_merge((df1, df2), names="Renamed_A")
        expected = pd.DataFrame(
            {"Renamed_A": [1., np.nan, 3., 4., 5., 40, 50]},
            index=pd.date_range("2023-01-01", periods=7, freq="d")
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_dataframe_merge_with_custom_column_selection(self, sample_data):
        df_multi1, df_multi2 = sample_data["df_multi1"], sample_data["df_multi2"]
        result = ts_merge((df_multi1, df_multi2), names=["A"])
        expected = pd.DataFrame(
            {"A": [1, 2, 100, 4, 5, np.nan, 500]},
            index=pd.date_range("2023-01-01", periods=7, freq="d")
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_merge_irregular_series(self, irregular_sample_data):
        s1, s2 = irregular_sample_data["series1"], irregular_sample_data["series2"]
        result = ts_merge((s1, s2))
        expected_index = sorted(s1.index.union(s2.index))
        expected_values = [1., 10, 2, 20, 3, 30, 4, 40]
        expected = pd.Series(expected_values, index=expected_index, name="A")
        pd.testing.assert_series_equal(result, expected)

# ----------------------------------------------------------------------
# Tests for ts_splice
# ----------------------------------------------------------------------

class TestTsSplice:
    def test_splice_prefer_first(self, sample_data):
        s1, s2 = sample_data["series1"], sample_data["series2"]
        result = ts_splice((s1, s2), transition="prefer_first")
        expected = pd.Series(
            [1., 2, np.nan, 4, 5, np.nan, 50],
            index=pd.date_range("2023-01-01", periods=7, freq="d"),
            name="A"
        )
        pd.testing.assert_series_equal(result, expected)

    def test_splice_prefer_last(self, sample_data):
        s1, s2 = sample_data["series1"], sample_data["series2"]
        result = ts_splice((s1, s2), transition="prefer_last")
        expected = pd.Series(
            [1., 2, 10, 20., 30., np.nan, 50],
            index=pd.date_range("2023-01-01", periods=7, freq="d"),
            name="A"
        )
        pd.testing.assert_series_equal(result, expected)

    def test_splice_series_with_names(self, sample_data):
        s1, s2 = sample_data["series1"], sample_data["series2"]
        result = ts_splice((s1, s2), names="new_name", transition="prefer_last")
        expected = pd.Series(
            [1, 2, 10, 20, 30, np.nan, 50],
            index=pd.date_range("2023-01-01", periods=7, freq="d"),
            name="new_name"
        )
        pd.testing.assert_series_equal(result, expected)

    def test_splice_with_custom_transitions(self, sample_data):
        s1, s2 = sample_data["series1"], sample_data["series2"]
        transition_points = [pd.Timestamp("2023-01-04T01:01")]
        result = ts_splice((s1, s2), transition=transition_points)
        expected = pd.Series(
            [1, 2, np.nan, 4., 30., np.nan, 50],
            index=pd.date_range("2023-01-01", periods=7, freq="d"),
            name="A"
        )
        pd.testing.assert_series_equal(result, expected)

    def test_splice_dataframe_prefer_first(self, sample_data):
        df1, df2 = sample_data["df1"], sample_data["df2"]
        result = ts_splice((df1, df2), transition="prefer_first")
        expected = pd.DataFrame(
            {"A": [1, np.nan, 3, 4, 5, 40, 50]},
            index=pd.date_range("2023-01-01", periods=7, freq="d")
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_splice_dataframe_prefer_last(self, sample_data):
        df1, df2 = sample_data["df1"], sample_data["df2"]
        result = ts_splice((df1, df2), transition="prefer_last")
        expected = pd.DataFrame(
            {"A": [1., np.nan, 10, 20., np.nan, 40, 50]},
            index=pd.date_range("2023-01-01", periods=7, freq="d")
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_splice_multi_column_dataframe(self, sample_data):
        df_multi1, df_multi2 = sample_data["df_multi1"], sample_data["df_multi2"]
        result = ts_splice((df_multi1, df_multi2), transition="prefer_last")
        expected = pd.DataFrame({
            "A": [1., 2, 100, 200, 300., np.nan, 500],
            "B": [10., np.nan, 1000., 2000., np.nan, 4000, 5000]
        }, index=pd.date_range("2023-01-01", periods=7, freq="d"))
        pd.testing.assert_frame_equal(result, expected)

    def test_splice_with_renaming(self, sample_data):
        s1, s2 = sample_data["series1"], sample_data["series2"]
        result = ts_splice((s1, s2), names="Renamed_A", transition="prefer_last")
        expected = pd.Series(
            [1., 2, 10., 20., 30, np.nan, 50],
            index=pd.date_range("2023-01-01", periods=7, freq="d"),
            name="Renamed_A"
        )
        pd.testing.assert_series_equal(result, expected)

    def test_splice_multi_column_dataframe_with_column_selection(self, sample_data):
        df_multi1, df_multi2 = sample_data["df_multi1"], sample_data["df_multi2"]
        result = ts_splice((df_multi1, df_multi2), names=["A"], transition="prefer_last")
        expected = pd.DataFrame(
            {"A": [1., 2, 100, 200, 300, np.nan, 500]},
            index=pd.date_range("2023-01-01", periods=7, freq="d")
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_splice_floor_dates(self, sample_data):
        s1, s2 = sample_data["series1"], sample_data["series2"]
        result = ts_splice((s1, s2), transition="prefer_last", floor_dates=True)
        expected = pd.Series(
            [1., 2, 10, 20., 30., np.nan, 50],
            index=pd.date_range("2023-01-01", periods=7, freq="d"),
            name="A"
        )
        pd.testing.assert_series_equal(result, expected)

    def test_splice_irregular_series(self, irregular_sample_data):
        s1, s2 = irregular_sample_data["series1"], irregular_sample_data["series2"]
        result = ts_splice((s1, s2), transition="prefer_first")
        expected = pd.Series(
            [1., 2, 3, 4, 40],
            index=pd.to_datetime(["2023-01-01", "2023-01-03", "2023-01-07", "2023-01-10", "2023-01-11"]),
            name="A"
        )
        pd.testing.assert_series_equal(result, expected)

# ----------------------------------------------------------------------
# Tests for error conditions
# ----------------------------------------------------------------------

class TestErrorConditions:
    def test_mismatched_column_names(self, sample_data):
        df1 = pd.DataFrame({"X": [1, 2, np.nan, 4, 5]}, 
                           index=pd.date_range("2023-01-01", periods=5, freq="d"))
        df2 = pd.DataFrame({"Y": [10, 20, np.nan, 40, 50]}, 
                           index=pd.date_range("2023-01-03", periods=5, freq="d"))
        with pytest.raises(ValueError, match="All input columns must be identical when `names` is None"):
            ts_merge((df1, df2))

    def test_empty_series_list(self):
        with pytest.raises(ValueError, match="`series` must be a non-empty tuple or list"):
            ts_merge([])

    def test_non_datetime_index(self):
        df1 = pd.DataFrame({"A": [1, 2, 3]}, index=[1, 2, 3])
        df2 = pd.DataFrame({"A": [4, 5, 6]}, index=[2, 3, 4])
        with pytest.raises(ValueError, match="All input series must have a DatetimeIndex."):
            ts_merge((df1, df2))


# ----------------------------------------------------------------------
# Additional tests for strict_priority behavior in ts_merge
# ----------------------------------------------------------------------

def test_ts_merge_strict_priority_series_window(sample_data):
    s1, s2 = sample_data["series1"], sample_data["series2"]
    # s1 dominates Jan1..Jan5; its NaN on Jan3 remains NaN; s2 cannot fill it.
    result = ts_merge((s1, s2), strict_priority=True)
    expected_index = s1.index.union(s2.index, sort=False).sort_values()
    expected = pd.Series([1., 2., np.nan, 4., 5., np.nan, 50.], index=expected_index, name="A")
    pd.testing.assert_series_equal(result, expected)

def test_ts_merge_strict_priority_dataframe_per_column(sample_data):
    df1, df2 = sample_data["df1"], sample_data["df2"]
    # Create multi-column frames to exercise per-column dominance
    df1m = pd.concat([df1, df1.rename(columns={"A": "B"})], axis=1)
    df2m = pd.concat([sample_data["df2"], sample_data["df2"].rename(columns={"A": "B"})], axis=1)
    # Insert an interior NaN in higher-priority B column to ensure NaN is not backfilled
    df1m.loc[df1m.index[2], "B"] = np.nan
    result = ts_merge((df1m, df2m), strict_priority=True)
    expected_index = df1m.index.union(df2m.index, sort=False).sort_values()
    exp = pd.DataFrame(index=expected_index, columns=["A", "B"], dtype=float)
    # Column A: df1 covers first window fully; df2 only contributes after the window
    exp["A"] = [1., np.nan, 3., 4., 5., 40., 50.]
    # Column B: an interior NaN in df1's window must remain NaN
    exp["B"] = [1., np.nan, np.nan, 4., 5., 40., 50.]
    pd.testing.assert_frame_equal(result[["A", "B"]], exp)

def test_ts_merge_strict_priority_irregular(irregular_sample_data):
    s1 = irregular_sample_data["series1"]
    s2 = irregular_sample_data["series2"]
    # s1 window [first_valid, last_valid] excludes s2 within; s2 contributes only after.
    result = ts_merge((s1, s2), strict_priority=True)
    expected = pd.Series([1., 2., 3., 4., 40.],
                         index=pd.to_datetime(["2023-01-01","2023-01-03","2023-01-07","2023-01-10","2023-01-11"]),
                         name="A")
    pd.testing.assert_series_equal(result, expected)






# ----------------------------------------------------------------------
# Tests for ts_blend
# ----------------------------------------------------------------------


class TestTsBlend:
    def test_blend_series_no_blend_length_equiv_to_merge(self, sample_data):
        """
        With blend_length=None, ts_blend should behave like a simple
        priority merge (ts_merge with default settings).
        """
        s1, s2 = sample_data["series1"], sample_data["series2"]

        result = ts_blend((s1, s2), blend_length=None)

        # For reference, this is the same expectation as TestTsMerge.test_series_merge
        expected = pd.Series(
            [1., 2., 10., 4., 5., np.nan, 50.],
            index=pd.date_range("2023-01-01", periods=7, freq="d"),
            name="A",
        )
        pd.testing.assert_series_equal(result, expected)

    def test_blend_series_integer_window(self, sample_data):
        """
        Integer blend_length: we should see a soft blend near the main gap in
        the higher priority series, but otherwise keep the higher priority
        values (or pure lower-priority inside gaps).
        """
        s1, s2 = sample_data["series1"], sample_data["series2"]
        # s1: 2023-01-01..05, [1, 2, NaN, 4, 5]
        # s2: 2023-01-03..07, [10, 20, 30, NaN, 50]

        result = ts_blend((s1, s2), blend_length=2)

        # Union index: 2023-01-01..07
        idx = pd.date_range("2023-01-01", periods=7, freq="d")

        # Expected values (see derivation in chat):
        # high-priority gaps (NaN) at days 3, 6, 7 (in union),
        # blended shoulders at days 4 and 5.
        expected = pd.Series(
            [1.0, 2.0, 10.0, 8.0, 11.25, np.nan, 50.0],
            index=idx,
            name="A",
        )
        pd.testing.assert_series_equal(result, expected)

    def test_blend_dataframe_integer_window(self, sample_data):
        """
        DataFrame version: verify that blending works column-wise and that
        we still prefer the high-priority frame where possible.
        """
        df1, df2 = sample_data["df1"], sample_data["df2"]
        # df1 A: 2023-01-01..05, [1, NaN, 3, 4, 5]
        # df2 A: 2023-01-03..07, [10, 20, NaN, 40, 50]

        result = ts_blend((df1, df2), blend_length=2)

        idx = pd.date_range("2023-01-01", periods=7, freq="d")
        # Derived expectations for column "A":
        # - Fill high-priority gaps using low where possible
        # - Then blend near gaps using the same kernel as in the series test
        expected = pd.DataFrame(
            {"A": [1.0, np.nan, 4.75, 4.0, 5.0, 40.0, 50.0]},
            index=idx,
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_blend_hourly_series_timedelta_window(self):
        """
        Time-based blend_length (Timedelta-like) on a regular hourly index
        should behave analogously to the integer-window case.
        """
        idx_hi = pd.date_range("2023-01-01 00:00", periods=5, freq="h")
        idx_lo = pd.date_range("2023-01-01 02:00", periods=5, freq="h")

        # Similar pattern to sample_data but hourly
        s1 = pd.Series([1, 2, np.nan, 4, 5], index=idx_hi, name="A")
        s2 = pd.Series([10, 20, 30, np.nan, 50], index=idx_lo, name="A")

        result = ts_blend((s1, s2), blend_length="2H")

        idx = pd.date_range("2023-01-01 00:00", periods=7, freq="h")
        expected = pd.Series(
            [1.0, 2.0, 10.0, 8.0, 11.25, np.nan, 50.0],
            index=idx,
            name="A",
        )
        pd.testing.assert_series_equal(result, expected)

    def test_blend_series_with_names(self, sample_data):
        """
        Make sure the align_inputs_strict decorator still handles `names`
        consistently with ts_merge/ts_splice.
        """
        s1, s2 = sample_data["series1"], sample_data["series2"]
        result = ts_blend((s1, s2), names="new_name", blend_length=2)

        idx = pd.date_range("2023-01-01", periods=7, freq="d")
        expected = pd.Series(
            [1.0, 2.0, 10.0, 8.0, 11.25, np.nan, 50.0],
            index=idx,
            name="new_name",
        )
        pd.testing.assert_series_equal(result, expected)

    def test_blend_time_based_requires_regular_index(self, irregular_sample_data):
        """
        Time-based blend_length on an irregular index (no .freq) should
        raise a clear error.
        """
        s1 = irregular_sample_data["series1"]
        s2 = irregular_sample_data["series2"]

        with pytest.raises(ValueError, match="requires a regular index with a .freq"):
            ts_blend((s1, s2), blend_length="2D")

    def test_blend_non_datetime_index_raises(self):
        """
        Non-datetime index should error out (same spirit as ts_merge).
        """
        s1 = pd.Series([1, 2, np.nan], index=[1, 2, 3], name="A")
        s2 = pd.Series([10, 20, 30], index=[2, 3, 4], name="A")

        with pytest.raises(ValueError, match="DatetimeIndex or PeriodIndex"):
            ts_blend((s1, s2), blend_length=2)
