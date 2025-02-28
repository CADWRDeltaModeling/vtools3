

import pytest
import numpy as np
from vtools.functions.merge import ts_merge, ts_splice
import pandas as pd
from vtools.data.vtime import hours, days


@pytest.fixture
def sample_data():
    """Create sample time series data for tests."""
    idx1 = pd.date_range("2023-01-01", periods=5, freq="D")
    idx2 = pd.date_range("2023-01-03", periods=5, freq="D")

    series1 = pd.Series([1, 2, np.nan, 4, 5], index=idx1, name="A")
    series2 = pd.Series([10, 20, 30, np.nan, 50], index=idx2, name="A")

    df1 = pd.DataFrame({"A": [1, np.nan, 3, 4, 5]}, index=idx1)
    df2 = pd.DataFrame({"A": [10, 20, np.nan, 40, 50]}, index=idx2)

    # ✅ Add Multi-Column DataFrames
    df_multi1 = pd.DataFrame({
        "A": [1, 2, np.nan, 4, 5], 
        "B": [10, np.nan, 30, 40, 50]
    }, index=idx1)

    df_multi2 = pd.DataFrame({
        "A": [100, 200, 300, np.nan, 500], 
        "B": [1000, 2000, np.nan, 4000, 5000]
    }, index=idx2)

    return series1, series2, df1, df2, df_multi1, df_multi2
    
@pytest.fixture
def irregular_sample_data():
    """Create irregularly spaced time series data for testing."""
    idx1 = pd.to_datetime(["2023-01-01", "2023-01-03", "2023-01-07", "2023-01-10"])
    idx2 = pd.to_datetime(["2023-01-02", "2023-01-04", "2023-01-08", "2023-01-11"])

    series1 = pd.Series([1., 2, 3, 4], index=idx1, name="A")
    series2 = pd.Series([10., 20, 30, 40], index=idx2, name="A")

    df1 = pd.DataFrame({"A": [1., 2, 3, 4]}, index=idx1)
    df2 = pd.DataFrame({"A": [10., 20, 30, 40]}, index=idx2)

    return series1, series2, df1, df2
    
# --- TEST CASES ---

def test_series_merge(sample_data):
    """Test merging two Series with overlap."""
    series1, series2, _, _, _, _ = sample_data
    

    result = ts_merge((series1, series2),names="A")
    expected = pd.Series(
        [1, 2, 10, 4, 5, np.nan, 50], 
        index=pd.date_range("2023-01-01", periods=7, freq="d"), 
        name="A"
    )
    pd.testing.assert_series_equal(result, expected)


def test_single_column_dataframe(sample_data):
    """Test merging two single-column DataFrames."""
    _, _, df1, df2, _, _ = sample_data


    result = ts_merge((df1, df2))    

    expected = pd.DataFrame(
        {"A": [1., np.nan, 3., 4., 5., 40., 50.]},
        index=pd.date_range("2023-01-01", periods=7, freq="d")
    )
    pd.testing.assert_frame_equal(result, expected)


def test_multi_column_dataframe(sample_data):
    """Test merging two multi-column DataFrames with overlapping columns."""
    _, _, _, _, df_multi1, df_multi2 = sample_data
    
    result = ts_merge((df_multi1, df_multi2))
    expected = pd.DataFrame({
        "A": [1., 2., 100, 4, 5, np.nan, 500],
        "B": [10., np.nan, 30, 40, 50, 4000, 5000]
    }, index=pd.date_range("2023-01-01", periods=7, freq="D"))

    pd.testing.assert_frame_equal(result, expected)


def test_series_merge_with_names(sample_data):
    """Test merging Series with a renaming operation."""

    series1, series2, _, _, _, _ = sample_data
   
    result = ts_merge((series1, series2), names="new_name")
  
    
    expected = pd.Series(
        [1., 2., 10, 4, 5, np.nan, 50], 
        index=pd.date_range("2023-01-01", periods=7, freq="D"), 
        name="new_name"
    )
    pd.testing.assert_series_equal(result, expected)


def test_dataframe_merge_with_renaming(sample_data):
    """Test merging DataFrames and renaming the column."""
    _, _, df1, df2, _, _ = sample_data

   
    result = ts_merge((df1, df2), names="Renamed_A")

    
    expected = pd.DataFrame(
        {"Renamed_A": [1., np.nan, 3., 4., 5., 40, 50]},
        index=pd.date_range("2023-01-01", periods=7, freq="D")
    )
    pd.testing.assert_frame_equal(result, expected)


def test_dataframe_merge_with_custom_column_selection(sample_data):
    """Test merging multi-column DataFrames while selecting specific columns."""
    _, _, _, _, df_multi1, df_multi2 = sample_data

    result = ts_merge((df_multi1, df_multi2), names=["A"])

    expected = pd.DataFrame(
        {"A": [1, 2, 100, 4, 5, np.nan, 500]},
        index=pd.date_range("2023-01-01", periods=7, freq="D")
    )

    pd.testing.assert_frame_equal(result, expected)


def test_mismatched_column_names(sample_data):
    """Ensure ValueError is raised when columns do not match and names is None."""
    df1 = pd.DataFrame({"X": [1, 2, np.nan, 4, 5]}, index=pd.date_range("2023-01-01", periods=5, freq="D"))
    df2 = pd.DataFrame({"Y": [10, 20, np.nan, 40, 50]}, index=pd.date_range("2023-01-03", periods=5, freq="D"))
    
    with pytest.raises(ValueError, match="All input DataFrames must have the same columns when `names` is None."):
        ts_merge((df1, df2))

def test_empty_series_list():
    """Ensure ValueError is raised when an empty list is provided."""
    with pytest.raises(ValueError, match="`series` must be a non-empty tuple or list of pandas.Series or pandas.DataFrame."):
        ts_merge([])



def test_non_datetime_index():
    """Ensure ValueError is raised when inputs do not have a DatetimeIndex."""
    df1 = pd.DataFrame({"A": [1, 2, 3]}, index=[1, 2, 3])
    df2 = pd.DataFrame({"A": [4, 5, 6]}, index=[2, 3, 4])

    with pytest.raises(ValueError, match="All input series must have a DatetimeIndex."):
        ts_merge((df1, df2))





################################################# Splice
def test_splice_prefer_first(sample_data):
    """Test splicing two Series with 'prefer_first'."""
    series1, series2, _, _, _, _ = sample_data
    result = ts_splice((series1, series2), transition="prefer_first")


    expected = pd.Series(
        [1., 2, np.nan, 4, 5, np.nan, 50],
        index=pd.date_range("2023-01-01", periods=7, freq="D"),
        name="A",
    )

    pd.testing.assert_series_equal(result, expected)


def test_splice_prefer_last(sample_data):
    """Test splicing two Series with 'prefer_last'."""
    series1, series2, _, _, _, _ = sample_data
    result = ts_splice((series1, series2), transition="prefer_last")

    expected = pd.Series(
        [1., 2, 10, 20., 30., np.nan, 50],
        index=pd.date_range("2023-01-01", periods=7, freq="D"),
        name="A",
    )

    pd.testing.assert_series_equal(result, expected)


def test_splice_with_custom_transitions(sample_data):
    """Test splicing with custom transition timestamps."""
    series1, series2, _, _, _, _ = sample_data
    transition_points = [pd.Timestamp("2023-01-04T01:01")]
    result = ts_splice((series1, series2), transition=transition_points)
    print("cts1",series1,"\nts2\n",series2,"\nresult\n",result)
    expected = pd.Series(
        [1, 2, np.nan, 4., 30., np.nan, 50],
        index=pd.date_range("2023-01-01", periods=7, freq="D"),
        name="A",
    )

    pd.testing.assert_series_equal(result, expected)


def test_splice_dataframe_prefer_first(sample_data):
    """Test splicing DataFrames with 'prefer_first'."""
    _, _, df1, df2, _, _ = sample_data
    result = ts_splice((df1, df2), transition="prefer_first")

    expected = pd.DataFrame(
        {"A": [1, np.nan, 3, 4, 5, 40, 50]},
        index=pd.date_range("2023-01-01", periods=7, freq="D"),
    )

    pd.testing.assert_frame_equal(result, expected)


def test_splice_dataframe_prefer_last(sample_data):
    """Test splicing DataFrames with 'prefer_last'."""
    _, _, df1, df2, _, _ = sample_data
    result = ts_splice((df1, df2), transition="prefer_last")
   
    expected = pd.DataFrame(
        {"A": [1., np.nan,10, 20., np.nan, 40, 50]},
        index=pd.date_range("2023-01-01", periods=7, freq="D"),
    )

    pd.testing.assert_frame_equal(result, expected)


def test_splice_multi_column_dataframe(sample_data):
    """Test splicing multi-column DataFrames with 'prefer_last'."""
    _, _, _, _, df_multi1, df_multi2 = sample_data
    result = ts_splice((df_multi1, df_multi2), transition="prefer_last")

    expected = pd.DataFrame(
        {
            "A": [1., 2, 100, 200, 300., np.nan, 500],
            "B": [10., np.nan, 1000.,2000., np.nan, 4000, 5000],
        },
        index=pd.date_range("2023-01-01", periods=7, freq="D"),
    )

    pd.testing.assert_frame_equal(result, expected)


def test_splice_with_renaming(sample_data):
    """Test renaming column names during splicing."""
    series1, series2, _, _, _, _ = sample_data
    result = ts_splice((series1, series2), names="Renamed_A", transition="prefer_last")

    expected = pd.Series(
        [1., 2, 10.,20., 30, np.nan, 50],
        index=pd.date_range("2023-01-01", periods=7, freq="D"),
        name="Renamed_A",
    )

    pd.testing.assert_series_equal(result, expected)


def test_splice_multi_column_dataframe_with_column_selection(sample_data):
    """Test splicing multi-column DataFrames while selecting specific columns."""
    _, _, _, _, df_multi1, df_multi2 = sample_data
    result = ts_splice((df_multi1, df_multi2), names=["A"], transition="prefer_last")
    print("s1",df_multi1,"\nts2\n",df_multi2,"\nresult\n",result) 
    expected = pd.DataFrame(
        {"A": [1., 2, 100, 200, 300, np.nan, 500]},
        index=pd.date_range("2023-01-01", periods=7, freq="D"),
    )

    pd.testing.assert_frame_equal(result, expected)


def test_splice_floor_dates(sample_data):
    """Test splicing with floor_dates=True."""
    series1, series2, _, _, _, _ = sample_data
    result = ts_splice((series1, series2), transition="prefer_last", floor_dates=True)

    expected = pd.Series(
        [1., 2, 10, 20., 30., np.nan, 50],
        index=pd.date_range("2023-01-01", periods=7, freq="D"),
        name="A",
    )

    pd.testing.assert_series_equal(result, expected)






def test_merge_irregular_series(irregular_sample_data):
    """Test ts_merge with irregularly spaced Series."""
    series1, series2, _, _ = irregular_sample_data
    result = ts_merge((series1, series2))

    expected_index = sorted(series1.index.union(series2.index))
    expected_values = [1., 10, 2, 20, 3, 30, 4, 40]

    expected = pd.Series(expected_values, index=expected_index, name="A")

    pd.testing.assert_series_equal(result, expected)


def test_splice_irregular_series(irregular_sample_data):
    """Test ts_splice with irregularly spaced Series."""
    series1, series2, _, _ = irregular_sample_data
    result = ts_splice((series1, series2), transition="prefer_first")
    print("s1\n",series1,"\ns2\n",series2,"\nres\n",result)
    expected = pd.Series(
        [1., 2, 3, 4,  40],
        index=pd.to_datetime(["2023-01-01", "2023-01-03", "2023-01-07", "2023-01-10",  "2023-01-11"]),
        name="A",
    )

    pd.testing.assert_series_equal(result, expected)








if __name__ == "__main__":
    pytest.main()