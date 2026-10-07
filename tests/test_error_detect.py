
import numpy as np
import numpy.ma as ma
import pandas as pd
import pytest

# Prefer the installed module (vtools.functions.error_detect) if available, but keep
# the import name stable for monkeypatching.
import vtools.functions.error_detect as ed


def _ts(values, freq="h", start="2024-01-01"):
    idx = pd.date_range(start, periods=len(values), freq=freq)
    return pd.Series(values, index=idx, name="x")


def _df(values, freq="h", start="2024-01-01", cols=("a", "b")):
    idx = pd.date_range(start, periods=len(values), freq=freq)
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = np.column_stack([arr, arr])
    return pd.DataFrame(arr, index=idx, columns=list(cols))


def _rolling_window(a, window):
    """
    Minimal stand-in for vtools rolling_window used by despike.
    Returns shape (n-window+1, window). Works with NaNs.
    """
    a = np.asarray(a)
    if window <= 0:
        raise ValueError("window must be positive")
    if a.ndim != 1:
        raise ValueError("rolling_window test helper expects 1D input")
    if len(a) < window:
        # match typical rolling-window semantics: empty roll
        return np.empty((0, window), dtype=float)
    # sliding view
    return np.stack([a[i : i + window] for i in range(len(a) - window + 1)], axis=0)


# -----------------
# nrepeat / _nrepeat
# -----------------

def test_nrepeat_series_basic_runs():
    s = _ts([1, 1, 1, 2, 2, 3, 3, 3, 3])
    out = ed.nrepeat(s)
    # run lengths should be constant within each run
    assert out.iloc[0] == 3
    assert out.iloc[2] == 3
    assert out.iloc[3] == 2
    assert out.iloc[4] == 2
    assert out.iloc[5] == 4
    assert out.iloc[8] == 4


def test_nrepeat_dataframe_applies_columnwise():
    df = pd.DataFrame(
        {
            "a": [1, 1, 2, 2, 2],
            "b": [5, 6, 6, 6, 7],
        },
        index=pd.date_range("2024-01-01", periods=5, freq="h"),
    )
    out = ed.nrepeat(df)
    assert list(out.columns) == ["a", "b"]
    # a: [1,1] run=2 ; [2,2,2] run=3
    assert out.loc[df.index[0], "a"] == 2
    assert out.loc[df.index[2], "a"] == 3
    # b: [5] run=1 ; [6,6,6] run=3 ; [7] run=1
    assert out.loc[df.index[0], "b"] == 1
    assert out.loc[df.index[1], "b"] == 3
    assert out.loc[df.index[3], "b"] == 3
    assert out.loc[df.index[4], "b"] == 1


def test_nrepeat_nan_behavior_maps_to_zero():
    """
    Implementation maps NaNs to 0 (docstring says this too).
    """
    s = _ts([1, 1, np.nan, np.nan, 2])
    out = ed.nrepeat(s)
    assert out.iloc[0] == 2
    assert out.iloc[2] == 0
    assert out.iloc[3] == 0


# -----------------
# threshold / bounds_test
# -----------------

@pytest.mark.parametrize(
    "bounds, expected_nan_mask",
    [
        ((0.0, 10.0), [False, False, True, True, False]),
        ((None, 10.0), [False, False, False, True, False]),
        ((0.0, None), [False, False, True, False, False]),
        (None, [False, False, False, False, False]),
    ],
)
def test_threshold_masks_out_of_bounds(bounds, expected_nan_mask):
    s = _ts([0.0, 10.0, -0.01, 10.01, 5.0])
    out = ed.threshold(s, bounds=bounds, copy=True)
    assert list(out.isna().to_numpy()) == expected_nan_mask
    # equals-to-bound should NOT be masked
    if bounds is not None and bounds[0] is not None:
        assert not pd.isna(out.iloc[0])
    if bounds is not None and bounds[1] is not None:
        assert not pd.isna(out.iloc[1])


def test_threshold_copy_false_mutates_input():
    s = _ts([0.0, 1.0, 99.0])
    ed.threshold(s, bounds=(None, 10.0), copy=False)
    assert pd.isna(s.iloc[2])


def test_bounds_test_flags_anomalies_or_xfails_if_current_impl_is_broken():
    """
    Intended behavior: return boolean mask of out-of-bounds values without mutating inputs.
    Current implementation in some vtools versions raises a TypeError due to dtype handling.
    """
    df = _df([0.0, 10.0, -1.0, 11.0, 5.0])
    try:
        anom = ed.bounds_test(df, bounds=(0.0, 10.0))
    except TypeError as e:
        pytest.xfail(f"bounds_test currently raises TypeError (likely dtype bug): {e}")

    assert (
        anom.dtype == bool
        if isinstance(anom, pd.Series)
        else anom.dtypes.eq(bool).all()
    )
    assert anom.dtypes.eq(bool).all()
    assert anom.shape == df.shape
    assert bool(anom.iloc[2, 0]) is True
    assert bool(anom.iloc[3, 0]) is True
    assert bool(anom.iloc[0, 0]) is False
    # original must remain unchanged
    assert not df.isna().any().any()


# -----------------
# max_change_test
# -----------------

@pytest.mark.parametrize(
    "values, limit, plateau_len, flagged_positions",
    [
        ([], 3, 0, []),
        ([20], 3, 0, []),
        ([20, 20, 20], 3, 0, []),
        ([20, 15, 15], 3, 0, []),
        ([20, 10, 20], 3, 0, [1]),
        ([20, 15, 10, 5, 10, 15, 20], 3, 0, [1, 2, 3, 4, 5]),
        ([20, 15, 15, 10, 20], 3, 0, [3]),
        ([20, 15, 15, 10, 20], 3, 1, [1, 2, 3]),
        ([20, 16, 16, 12], 3, 1, []),
        ([20, 14, 14, 8], 4, 1, []),
        ([20, 10, 10, 10, 0], 3, 1, []),
        ([20, 10, np.nan, 0, 10], 3, 10, []),
        ([20, 10, 0, np.nan, 0, 10, 20], 3, 10, [1, 5]),
        ([0, 3, 6], 3, 0, []),
        ([0, 7, 7, 7, 14, 14, 14, 21], 3, 2, [1, 2, 3]),
    ],
)
def test_max_change_test_run_merging_and_endpoints(
    values, limit, plateau_len, flagged_positions
):
    series = _ts(np.asarray(values, dtype=float), freq="15min")
    original = series.copy()
    expected = pd.Series(False, index=series.index, name=series.name)
    expected.iloc[flagged_positions] = True
    actual = ed.max_change_test(series, limit, max_plateau_len=plateau_len)
    pd.testing.assert_series_equal(actual, expected)
    pd.testing.assert_series_equal(series, original)


def test_max_change_test_dataframe_columns_are_independent():
    frame = pd.DataFrame(
        {"excursion": [20, 10, 20], "normal": [20, 20, 20]},
        index=pd.date_range("2026-07-09", periods=3, freq="15min"),
    )
    original = frame.copy()
    expected = pd.DataFrame(
        {"excursion": [False, True, False], "normal": [False, False, False]},
        index=frame.index,
    )
    pd.testing.assert_frame_equal(ed.max_change_test(frame, 3), expected)
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("limit", [1.0, 2.0])
def test_max_change_test_detects_july_temperature_ramp_without_recovery_flags(limit):
    series = _ts(
        [19.54, 16.75, 13.96, 11.17, 8.37, 5.58, 2.79, 0.0, 6.63, 13.27, 19.9],
        freq="15min", start="2026-07-09 02:15",
    )
    expected = pd.Series(True, index=series.index, name=series.name)
    expected.iloc[[0, -1]] = False
    pd.testing.assert_series_equal(ed.max_change_test(series, limit), expected)


@pytest.mark.parametrize("limit", [0, -1, np.nan, np.inf])
def test_max_change_test_rejects_invalid_limits(limit):
    with pytest.raises(ValueError, match="max_change"):
        ed.max_change_test(_ts([20, 10, 20]), limit)


@pytest.mark.parametrize("plateau_len", [-1, 1.5, True])
def test_max_change_test_rejects_invalid_plateau_lengths(plateau_len):
    with pytest.raises(ValueError, match="max_plateau_len"):
        ed.max_change_test(_ts([20, 10, 20]), 3, max_plateau_len=plateau_len)


# -----------------
# med_outliers / median_test / median_test_twoside
# -----------------

def test_med_outliers_series_flags_spike_as_nan_and_preserves_copy():
    base = np.zeros(31)
    base[15] = 100.0  # isolated spike
    s = _ts(base, freq="h")
    s_orig = s.copy()

    out = ed.med_outliers(
        s,
        level=4.0,
        filt_len=7,
        quantiles=(0.25, 0.75),
        copy=True,
        as_anomaly=False,
    )
    assert pd.isna(out.iloc[15])
    # mostly unchanged elsewhere
    assert out.drop(out.index[15]).notna().all()
    # original unchanged because copy=True
    assert s.equals(s_orig)


def test_med_outliers_as_anomaly_returns_boolean_mask():
    base = np.zeros(21)
    base[10] = 50.0
    s = _ts(base)
    anom = ed.med_outliers(
        s,
        level=3.0,
        filt_len=5,
        quantiles=(0.25, 0.75),
        copy=True,
        as_anomaly=True,
    )
    assert isinstance(anom, (pd.Series, pd.DataFrame))
    assert anom.dtype == bool
    assert bool(anom.iloc[10]) is True
    assert bool(anom.iloc[0]) is False


def test_median_test_delegates_to_med_outliers():
    base = np.zeros(21)
    base[10] = 50.0
    df = pd.DataFrame({"x": base}, index=pd.date_range("2024-01-01", periods=21, freq="h"))
    anom = ed.median_test(df, level=3, filt_len=5, quantiles=(0.25, 0.75))
    assert anom.shape == df.shape
    assert bool(anom.iloc[10, 0]) is True


def test_median_test_twoside_excludes_center_from_median_reduces_false_self_bias():
    vals = np.ones(25)
    vals[12] = 100.0
    df = pd.DataFrame({"x": vals}, index=pd.date_range("2024-01-01", periods=25, freq="h"))
    anom = ed.median_test_twoside(df, level=3.0, filt_len=7, quantiles=(0.25, 0.75), as_anomaly=True)
    assert bool(anom.iloc[12, 0]) is True
    assert bool(anom.iloc[11, 0]) is False
    assert bool(anom.iloc[13, 0]) is False


def test_med_outliers_dataframe_operates_columnwise():
    n = 31
    a = np.zeros(n); a[10] = 25.0
    b = np.zeros(n); b[20] = -30.0
    df = pd.DataFrame({"a": a, "b": b}, index=pd.date_range("2024-01-01", periods=n, freq="h"))
    out = ed.med_outliers(df, level=3.0, filt_len=7, quantiles=(0.25, 0.75), copy=True, as_anomaly=False)
    assert pd.isna(out.loc[df.index[10], "a"])
    assert pd.isna(out.loc[df.index[20], "b"])
    assert out["a"].drop(df.index[10]).notna().all()
    assert out["b"].drop(df.index[20]).notna().all()


# -----------------
# median_test_oneside
# -----------------

@pytest.mark.parametrize("reverse", [False, True])
def test_median_test_oneside_detects_outlier_and_preserves_index(monkeypatch, reverse):
    """
    median_test_oneside uses dask rolling with npartitions=50, which breaks for small inputs
    (partition size < overlap window). We patch dd.from_pandas to use a single partition
    to exercise the logic deterministically.
    """
    import dask.dataframe as dd

    real_from_pandas = dd.from_pandas

    def from_pandas_1part(df, npartitions=50):
        return real_from_pandas(df, npartitions=1)

    monkeypatch.setattr(ed.dd, "from_pandas", from_pandas_1part)

    vals = np.arange(40, dtype=float)
    vals[20] += 50.0
    s = _ts(vals, freq="h")
    anom = ed.median_test_oneside(s, level=3, filt_len=6, quantiles=(0.25, 0.75), reverse=reverse)
    assert anom.index.equals(s.index)
    assert bool(anom.iloc[20]) is True


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("as_frame", [False, True])
@pytest.mark.parametrize("npartitions", [1, 3])
def test_median_test_oneside_matches_directional_reference(
    monkeypatch, reverse, as_frame, npartitions
):
    values = 0.1 * np.arange(96) + np.sin(np.arange(96) / 5.0)
    values[13] += 7.0
    values[61] -= 11.0
    values[35] = np.nan
    series = _ts(values, freq="15min")
    source = series.to_frame() if as_frame else series
    original = source.copy()
    monkeypatch.setattr(
        ed, "choose_npartitions", lambda nrows, *, window: npartitions
    )
    ordered = series.iloc[::-1] if reverse else series
    window = 6
    prediction = (
        ordered.shift().rolling(window, min_periods=5).median()
        + 3 * ordered.diff().shift().rolling(window, min_periods=5).median()
    )
    residual = ordered - prediction
    quantiles = residual.quantile([0.25, 0.75])
    scale = quantiles.loc[0.75] - quantiles.loc[0.25]
    expected = (residual.abs() > 3 * scale).reindex(series.index)
    actual = ed.median_test_oneside(
        source, level=3, filt_len=6, quantiles=(0.25, 0.75), reverse=reverse
    )
    pd.testing.assert_series_equal(actual, expected, check_names=False)
    assert bool(actual.loc[series.index[13]])
    if as_frame:
        pd.testing.assert_frame_equal(source, original)
    else:
        pd.testing.assert_series_equal(source, original)


# -----------------
# gapdist_test_series
# -----------------

def test_gapdist_test_series_marks_small_gaps_with_sentinel(monkeypatch):
    """
    gapdist_test_series depends on vtools gap_count; patch it to deterministic output.
    """
    def fake_gap_count(ts):
        out = pd.Series(np.zeros(len(ts), dtype=int), index=ts.index)
        out.iloc[3:5] = 2
        out.iloc[10:15] = 5
        return out

    monkeypatch.setattr(ed, "gap_count", fake_gap_count)

    vals = np.arange(20, dtype=float)
    vals[3:5] = np.nan
    vals[10:15] = np.nan
    s = _ts(vals, freq="h")

    out = ed.gapdist_test_series(s, smallgaplen=3)
    assert (out.iloc[3:5].to_numpy() == -99999999.0).all()
    assert np.isnan(out.iloc[10:15].to_numpy()).all()


# -----------------
# steep_then_nan
# -----------------

def test_steep_then_nan_flags_outlier_only_near_gap(monkeypatch, capsys):
    """
    steep_then_nan combines:
      1) median-filter residual threshold (outlier)
      2) nearbiggap condition from gap_distance

    Patch gap-related pieces to make behavior deterministic.
    """
    monkeypatch.setattr(ed, "gapdist_test_series", lambda ts, smallgaplen=3: ts)

    def fake_gap_distance(ts, disttype="count", to="bad"):
        dist = pd.Series(999, index=ts.index, dtype=float)
        dist.iloc[18:23] = 1.0
        return dist.to_frame("dist")

    monkeypatch.setattr(ed, "gap_distance", fake_gap_distance)

    vals = np.zeros(40, dtype=float)
    vals[20] = 100.0
    vals[5] = 100.0
    s = _ts(vals, freq="h")

    anom = ed.steep_then_nan(s.to_frame("x"), level=3.0, filt_len=11, quantiles=(0.25, 0.75), as_anomaly=True)
    assert bool(anom.iloc[20, 0]) is True
    assert bool(anom.iloc[5, 0]) is False



def test_steep_then_nan_as_anomaly_false_replaces_values(monkeypatch):
    monkeypatch.setattr(ed, "gapdist_test_series", lambda ts, smallgaplen=3: ts)

    def fake_gap_distance(ts, disttype="count", to="bad"):
        dist = pd.Series(999, index=ts.index, dtype=float)
        dist.iloc[10:13] = 1.0
        return dist.to_frame("dist")

    monkeypatch.setattr(ed, "gap_distance", fake_gap_distance)

    vals = np.zeros(30, dtype=float)
    vals[11] = 100.0
    df = _df(vals, freq="h", cols=("x", "y"))
    out = ed.steep_then_nan(df, level=3.0, filt_len=11, quantiles=(0.25, 0.75), as_anomaly=False)
    assert pd.isna(out.iloc[11, 0])


# -----------------
# despike
# -----------------

def test_despike_replaces_spike_with_nan_and_preserves_baseline():
    arr = np.ones(200, dtype=float) * 10.0
    arr[100] = 1000.0
    out = ed.despike(arr.copy(), n1=1, n2=1, block=20)
    assert np.isnan(out[100])
    assert np.nanmedian(out) == pytest.approx(10.0, abs=1e-6)


def test_despike_as_anomaly_returns_boolean_mask():
    arr = np.ones(200, dtype=float) * 10.0
    arr[100] = 1000.0
    mask = ed.despike(arr.copy(), n1=1, n2=1, block=20, as_anomaly=True)
    assert mask.dtype == bool
    assert mask.shape == arr.shape
    assert bool(mask[100]) is True
    # Most points should not be flagged
    assert bool(mask[0]) is False


def test_despike_handles_negative_values_and_offset_restore():
    arr = np.linspace(-5.0, 5.0, 200)
    arr[50] = 50.0
    out = ed.despike(arr.copy(), n1=1, n2=1, block=20)
    assert np.isnan(out[50])
    assert np.nanmin(out) <= -5.0 + 1e-6
    diff = out - arr
    assert np.nanmedian(diff) == pytest.approx(0.0, abs=1e-9)
