import pytest
import numpy as np
import pandas as pd
from vtools.functions.merge import ts_merge, ts_splice
from vtools.data.vtime import hours, days

# ----------------------------------------------------------------------
# Fixtures and Helper Functions
# ----------------------------------------------------------------------

@pytest.fixture
def sample_data():
    """Create sample time series data for tests."""
    idx1 = pd.date_range("2023-01-01", periods=5, freq="D")
    idx2 = pd.date_range("2023-01-03", periods=5, freq="D")
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
            index=pd.date_range("2023-01-01", periods=7, freq="D"),
            name="A"
        )
        pd.testing.assert_series_equal(result, expected)

    def test_single_column_dataframe(self, sample_data):
        df1, df2 = sample_data["df1"], sample_data["df2"]
        result = ts_merge((df1, df2))
        expected = pd.DataFrame(
            {"A": [1., np.nan, 3., 4., 5., 40., 50.]},
            index=pd.date_range("2023-01-01", periods=7, freq="D")
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_multi_column_dataframe(self, sample_data):
        df_multi1, df_multi2 = sample_data["df_multi1"], sample_data["df_multi2"]
        result = ts_merge((df_multi1, df_multi2))
        expected = pd.DataFrame({
            "A": [1., 2., 100, 4, 5, np.nan, 500],
            "B": [10., np.nan, 30, 40, 50, 4000, 5000]
        }, index=pd.date_range("2023-01-01", periods=7, freq="D"))
        pd.testing.assert_frame_equal(result, expected)

    def test_series_merge_with_names(self, sample_data):
        s1, s2 = sample_data["series1"], sample_data["series2"]
        result = ts_merge((s1, s2), names="new_name")
        expected = pd.Series(
            [1., 2., 10, 4, 5, np.nan, 50],
            index=pd.date_range("2023-01-01", periods=7, freq="D"),
            name="new_name"
        )
        pd.testing.assert_series_equal(result, expected)

    def test_dataframe_merge_with_renaming(self, sample_data):
        df1, df2 = sample_data["df1"], sample_data["df2"]
        result = ts_merge((df1, df2), names="Renamed_A")
        expected = pd.DataFrame(
            {"Renamed_A": [1., np.nan, 3., 4., 5., 40, 50]},
            index=pd.date_range("2023-01-01", periods=7, freq="D")
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_dataframe_merge_with_custom_column_selection(self, sample_data):
        df_multi1, df_multi2 = sample_data["df_multi1"], sample_data["df_multi2"]
        result = ts_merge((df_multi1, df_multi2), names=["A"])
        expected = pd.DataFrame(
            {"A": [1, 2, 100, 4, 5, np.nan, 500]},
            index=pd.date_range("2023-01-01", periods=7, freq="D")
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
            index=pd.date_range("2023-01-01", periods=7, freq="D"),
            name="A"
        )
        pd.testing.assert_series_equal(result, expected)

    def test_splice_prefer_last(self, sample_data):
        s1, s2 = sample_data["series1"], sample_data["series2"]
        result = ts_splice((s1, s2), transition="prefer_last")
        expected = pd.Series(
            [1., 2, 10, 20., 30., np.nan, 50],
            index=pd.date_range("2023-01-01", periods=7, freq="D"),
            name="A"
        )
        pd.testing.assert_series_equal(result, expected)

    def test_splice_series_with_names(self, sample_data):
        s1, s2 = sample_data["series1"], sample_data["series2"]
        result = ts_splice((s1, s2), names="new_name", transition="prefer_last")
        expected = pd.Series(
            [1, 2, 10, 20, 30, np.nan, 50],
            index=pd.date_range("2023-01-01", periods=7, freq="D"),
            name="new_name"
        )
        pd.testing.assert_series_equal(result, expected)

    def test_splice_with_custom_transitions(self, sample_data):
        s1, s2 = sample_data["series1"], sample_data["series2"]
        transition_points = [pd.Timestamp("2023-01-04T01:01")]
        result = ts_splice((s1, s2), transition=transition_points)
        expected = pd.Series(
            [1, 2, np.nan, 4., 30., np.nan, 50],
            index=pd.date_range("2023-01-01", periods=7, freq="D"),
            name="A"
        )
        pd.testing.assert_series_equal(result, expected)

    def test_splice_dataframe_prefer_first(self, sample_data):
        df1, df2 = sample_data["df1"], sample_data["df2"]
        result = ts_splice((df1, df2), transition="prefer_first")
        expected = pd.DataFrame(
            {"A": [1, np.nan, 3, 4, 5, 40, 50]},
            index=pd.date_range("2023-01-01", periods=7, freq="D")
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_splice_dataframe_prefer_last(self, sample_data):
        df1, df2 = sample_data["df1"], sample_data["df2"]
        result = ts_splice((df1, df2), transition="prefer_last")
        expected = pd.DataFrame(
            {"A": [1., np.nan, 10, 20., np.nan, 40, 50]},
            index=pd.date_range("2023-01-01", periods=7, freq="D")
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_splice_multi_column_dataframe(self, sample_data):
        df_multi1, df_multi2 = sample_data["df_multi1"], sample_data["df_multi2"]
        result = ts_splice((df_multi1, df_multi2), transition="prefer_last")
        expected = pd.DataFrame({
            "A": [1., 2, 100, 200, 300., np.nan, 500],
            "B": [10., np.nan, 1000., 2000., np.nan, 4000, 5000]
        }, index=pd.date_range("2023-01-01", periods=7, freq="D"))
        pd.testing.assert_frame_equal(result, expected)

    def test_splice_with_renaming(self, sample_data):
        s1, s2 = sample_data["series1"], sample_data["series2"]
        result = ts_splice((s1, s2), names="Renamed_A", transition="prefer_last")
        expected = pd.Series(
            [1., 2, 10., 20., 30, np.nan, 50],
            index=pd.date_range("2023-01-01", periods=7, freq="D"),
            name="Renamed_A"
        )
        pd.testing.assert_series_equal(result, expected)

    def test_splice_multi_column_dataframe_with_column_selection(self, sample_data):
        df_multi1, df_multi2 = sample_data["df_multi1"], sample_data["df_multi2"]
        result = ts_splice((df_multi1, df_multi2), names=["A"], transition="prefer_last")
        expected = pd.DataFrame(
            {"A": [1., 2, 100, 200, 300, np.nan, 500]},
            index=pd.date_range("2023-01-01", periods=7, freq="D")
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_splice_floor_dates(self, sample_data):
        s1, s2 = sample_data["series1"], sample_data["series2"]
        result = ts_splice((s1, s2), transition="prefer_last", floor_dates=True)
        expected = pd.Series(
            [1., 2, 10, 20., 30., np.nan, 50],
            index=pd.date_range("2023-01-01", periods=7, freq="D"),
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
                           index=pd.date_range("2023-01-01", periods=5, freq="D"))
        df2 = pd.DataFrame({"Y": [10, 20, np.nan, 40, 50]}, 
                           index=pd.date_range("2023-01-03", periods=5, freq="D"))
        with pytest.raises(ValueError, match="All input DataFrames must have the same columns when `names` is None."):
            ts_merge((df1, df2))

    def test_empty_series_list(self):
        with pytest.raises(ValueError, match="`series` must be a non-empty tuple or list"):
            ts_merge([])

    def test_non_datetime_index(self):
        df1 = pd.DataFrame({"A": [1, 2, 3]}, index=[1, 2, 3])
        df2 = pd.DataFrame({"A": [4, 5, 6]}, index=[2, 3, 4])
        with pytest.raises(ValueError, match="All input series must have a DatetimeIndex."):
            ts_merge((df1, df2))
