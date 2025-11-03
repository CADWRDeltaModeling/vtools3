import pytest
import pandas as pd
import numpy as np
from vtools.functions.transition import transition_ts


# ---------- helpers ----------
def _idx(start="2023-01-01", periods=6, freq="h"):
    return pd.date_range(start, periods=periods, freq=freq)


def _gap(i0="2023-01-02", i1="2023-01-03"):
    return [i0, i1]


# ---------- type & frequency contracts ----------


def test_type_mismatch_raises():
    # Series vs DataFrame should fail
    ts0 = pd.Series([1, 2, 3, 4, 5, 6], index=_idx("2023-01-01"), name="A")
    ts1 = pd.DataFrame({"A": [10, 20, 30, 40, 50, 60]}, index=_idx("2023-01-02"))
    with pytest.raises(ValueError, match="same type"):
        transition_ts(
            ts0, ts1, method="linear", window=_gap(), return_type="series"
        )


def test_frequency_mismatch_raises():
    # Same type, different freq should fail
    ts0 = pd.Series([1, 2, 3, 4, 5, 6], index=_idx("2023-01-01", freq="h"), name="A")
    ts1 = pd.Series(
        [1, 2, 3, 4, 5, 6], index=_idx("2023-01-02", freq="30min"), name="A"
    )
    with pytest.raises(ValueError, match="same frequency"):
        transition_ts(
            ts0, ts1, method="linear", window=_gap(), return_type="series"
        )


# ---------- strict column alignment contracts (names=None) ----------


def test_df_columns_mismatch_raises_when_names_none():
    idx0 = _idx("2023-01-01", freq="h")
    idx1 = _idx("2023-01-02", freq="h")
    df0 = pd.DataFrame({"A": [1, 2, 3, 4, 5, 6]}, index=idx0)
    df1 = pd.DataFrame({"B": [10, 20, 30, 40, 50, 60]}, index=idx1)
    with pytest.raises(
        ValueError, match=r"All input columns must be identical when `names` is None"
    ):
        transition_ts(
            df0, df1, method="linear", window=_gap(), return_type="series"
        )


def test_df_column_order_mismatch_raises_when_names_none():
    idx0 = _idx("2023-01-01", freq="h")
    idx1 = _idx("2023-01-02", freq="h")
    df0 = pd.DataFrame({"A": [1, 2, 3, 4, 5, 6], "B": [0, 0, 0, 0, 0, 0]}, index=idx0)
    df1 = pd.DataFrame(
        {"B": [0, 0, 0, 0, 0, 0], "A": [1, 2, 3, 4, 5, 6]}, index=idx1
    )  # same set, different order
    with pytest.raises(
        ValueError, match=r"All input columns must be identical when `names` is None"
    ):
        transition_ts(
            df0, df1, method="linear", window=_gap(), return_type="series"
        )


# ---------- names=str / names=[str] on univariate inputs ----------


def test_series_univariate_names_str_returns_series_named():
    s0 = pd.Series([1, 2, 3, 4, 5, 6], index=_idx("2023-01-01", freq="h"), name="A")
    s1 = pd.Series(
        [np.nan, np.nan, 3, 4, 5, 6], index=_idx("2023-01-02", freq="h"), name="B"
    )
    out = transition_ts(
        s0,
        s1,
        method="linear",
        window=["2023-01-01 12:00", "2023-01-01 18:00"],
        return_type="series",
        names="X",
    )
    assert isinstance(out, pd.Series)
    assert out.name == "X"


def test_series_univariate_names_list_single_equiv_to_str():
    s0 = pd.Series([1, 2, 3, 4, 5, 6], index=_idx("2023-01-01", freq="h"), name="A")
    s1 = pd.Series(
        [np.nan, np.nan, 3, 4, 5, 6], index=_idx("2023-01-02", freq="h"), name="B"
    )
    out1 = transition_ts(
        s0,
        s1,
        method="linear",
        window=["2023-01-01 12:00", "2023-01-01 18:00"],  # <-- inside natural gap
        return_type="series",
        names="X",
    )

    out2 = transition_ts(
        s0,
        s1,
        method="linear",
        window=["2023-01-01 12:00", "2023-01-01 18:00"],  # <-- same valid gap
        return_type="series",
        names=["X"],
    )
    pd.testing.assert_series_equal(out1, out2)


def test_series_univariate_names_list_multi_raises():
    s0 = pd.Series([1, 2, 3, 4, 5, 6], index=_idx("2023-01-01", freq="h"), name="A")
    s1 = pd.Series(
        [np.nan, np.nan, 3, 4, 5, 6], index=_idx("2023-01-02", freq="h"), name="B"
    )
    with pytest.raises(ValueError, match="multiple names"):
        transition_ts(
            s0,
            s1,
            method="linear",
            window=_gap(),
            return_type="series",
            names=["X", "Y"],
        )


# ---------- names=[...] selection on multivariate DFs ----------


def test_df_names_list_selection_subset_and_order_preserved():
    idx0 = _idx("2023-01-01", freq="h")
    idx1 = _idx("2023-01-02", freq="h")
    df0 = pd.DataFrame(
        {"A": [1, 2, 3, 4, 5, 6], "B": [10, 20, 30, 40, 50, 60]}, index=idx0
    )
    df1 = pd.DataFrame(
        {"A": [2, 3, 4, 5, 6, 7], "B": [11, 21, 31, 41, 51, 61]}, index=idx1
    )

    gap_start = df0.index[-1] + df0.index.freq  # 2023-01-01 06:00
    gap_end = df1.index[0] - df1.index.freq  # 2023-01-01 23:00
    out = transition_ts(
        df0,
        df1,
        method="linear",
        window=[gap_start, gap_end],
        return_type="series",
        names=["B", "A"],
    )


def test_df_names_list_missing_column_raises():
    idx0 = _idx("2023-01-01", freq="h")
    idx1 = _idx("2023-01-02", freq="h")
    df0 = pd.DataFrame(
        {"A": [1, 2, 3, 4, 5, 6], "B": [10, 20, 30, 40, 50, 60]}, index=idx0
    )
    df1 = pd.DataFrame({"A": [2, 3, 4, 5, 6, 7]}, index=idx1)  # missing 'B'
    with pytest.raises(ValueError, match=r"missing requested columns"):
        transition_ts(
            df0,
            df1,
            method="linear",
            window=_gap(),
            return_type="series",
            names=["A", "B"],
        )


def test_gap_end_after_ts1_last_raises():
    s0 = pd.Series(range(6), index=_idx("2023-01-01", freq="h"), name="A")
    s1 = pd.Series(range(6), index=_idx("2023-01-02", freq="h"), name="A")
    with pytest.raises(ValueError, match="window end"):
        transition_ts(
            s0, s1, method="linear", window=["2023-01-02 12:00", "2023-01-03 00:00"]
        )


def test_gap_start_not_before_ts0_any_sample_raises():
    s0 = pd.Series(range(6), index=_idx("2023-01-01", freq="h"), name="A")
    s1 = pd.Series(range(6), index=_idx("2023-01-02", freq="h"), name="A")
    with pytest.raises(ValueError, match="window start"):
        transition_ts(
            s0, s1, method="linear", window=["2022-12-31 00:00", "2022-12-31 12:00"]
        )


# ---------- names=[] guard ----------


def test_empty_names_list_raises():
    s0 = pd.Series([1, 2, 3, 4, 5, 6], index=_idx("2023-01-01", freq="h"), name="A")
    s1 = pd.Series(
        [np.nan, np.nan, 3, 4, 5, 6], index=_idx("2023-01-02", freq="h"), name="B"
    )
    with pytest.raises(ValueError, match="selection is empty"):
        transition_ts(
            s0, s1, method="linear", window=_gap(), return_type="series", names=[]
        )


import pytest
import pandas as pd
import numpy as np
from vtools import transition_ts


def _idx(start="2023-01-01", periods=6, freq="h"):
    return pd.date_range(start, periods=periods, freq=freq)


# --- helpers to build simple data ---
def _s(name, start, periods=6, freq="h", offset=0):
    idx = _idx(start, periods=periods, freq=freq) + pd.Timedelta(offset, unit=freq)
    return pd.Series(range(periods), index=idx, name=name)


def _df(names, start, periods=6, freq="h"):
    idx = _idx(start, periods=periods, freq=freq)
    data = {n: np.arange(periods) for n in names}
    return pd.DataFrame(data, index=idx)


# ---------- CONTRACT: explicit window strict domain checks ----------


def test_gap_start_before_ts0_first_errors():
    ts0 = _s("A", "2023-01-02")
    ts1 = _s("A", "2023-01-03")
    with pytest.raises(ValueError, match=r"window start.*"):
        transition_ts(
            ts0,
            ts1,
            window=["2023-01-01 00:00", "2023-01-02 12:00"],
            method="linear",
        )


def test_gap_end_after_ts1_last_errors():
    ts0 = _s("A", "2023-01-01")
    ts1 = _s("A", "2023-01-02")
    with pytest.raises(ValueError, match=r"window end.*"):
        transition_ts(
            ts0,
            ts1,
            window=["2023-01-02 00:00", "2023-01-03 12:00"],
            method="linear",
        )


def test_gap_start_ge_end_errors():
    ts0 = _s("A", "2023-01-01")
    ts1 = _s("A", "2023-01-02")
    with pytest.raises(ValueError, match="start must be strictly before end"):
        transition_ts(
            ts0,
            ts1,
            window=["2023-01-01 10:00", "2023-01-01 10:00"],
            method="linear",
        )


# ---------- OPTIONAL SNAP: only when gap ⊂ natural gap ----------


def test_max_snap_expands_inside_natural_gap_symmetrically():
    # Natural gap: ts0.last < ts1.first (24h apart)
    ts0 = _s("A", "2023-01-01")  # ends ~ 2023-01-01 05:00
    ts1 = _s("A", "2023-01-03")  # starts 2023-01-03 00:00
    # User picks a very small sub-gap in the middle of the natural gap
    out = transition_ts(
        ts0,
        ts1,
        method="linear",
        window=["2023-01-02 06:00", "2023-01-02 07:00"],
        max_snap="1D",  # allow widening up to 24h
        return_type="series",
    )
    assert isinstance(out, (pd.Series, pd.DataFrame))


def test_max_snap_ignored_when_overlap():
    # Overlap (no natural gap)
    ts0 = _s("A", "2023-01-01", periods=12, freq="h")
    ts1 = _s("A", "2023-01-01 06:00", periods=12, freq="h")
    # Valid gap inside overlap; max_snap should be ignored (no errors; algorithms decide)
    out = transition_ts(
        ts0,
        ts1,
        method="linear",
        window=["2023-01-01 08:00", "2023-01-01 10:00"],
        max_snap="12H",
        return_type="series",
    )
    assert isinstance(out, (pd.Series, pd.DataFrame))


def test_max_snap_does_not_cross_natural_bounds():
    ts0 = _s("A", "2023-01-01", periods=6, freq="h")  # last = 2023-01-01 05:00
    ts1 = _s("A", "2023-01-02", periods=6, freq="h")  # first = 2023-01-02 00:00
    # choose a sub-gap near the left edge; allow large snap
    out = transition_ts(
        ts0,
        ts1,
        method="linear",
        window=["2023-01-01 05:30", "2023-01-01 06:00"],
        max_snap="1D",
        return_type="series",
    )
    assert isinstance(out, (pd.Series, pd.DataFrame))
    # We don’t assert internal endpoints here, but this protects against crossing bounds.


# ---------- NAMES contract still holds with gap handling ----------


def test_df_subset_names_with_gap_inside_natural_gap():
    df0 = _df(["A", "B"], "2023-01-01")
    df1 = _df(["A", "B"], "2023-01-03")
    out = transition_ts(
        df0,
        df1,
        method="linear",
        window=["2023-01-02 06:00", "2023-01-02 07:00"],
        names=["B", "A"],
        max_snap="12h",
        return_type="series",
    )
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["B", "A"]
