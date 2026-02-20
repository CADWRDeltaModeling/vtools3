#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import dask

dask.config.set({"dataframe.query-planning": True})
import dask.dataframe as dd
import numpy as np
import numpy.ma as ma
import pandas as pd
import sys
import warnings
from scipy.signal import medfilt
from scipy.stats.mstats import mquantiles
from scipy.stats import iqr as scipy_iqr

from vtools.data.gap import *
from vtools.data.timeseries import to_dataframe


"""
    Functions to detect (and replace with NaN) any outliers or bad data
    
    norepeats: catches any repeated values. This will often occur if the sensor
    is not working properly, and since realistically the value will always be
    slightly changing, repeats are a clear sign of a malfunctioning device.
    
    Inputs: ts:         the time series
            threshold:  the minimum number of repeated values in a row that 
                        should be caught (default value is currently 20 but
                        should be changed)
            copy:       if True (default), a copy is made, leaving the original
                        series intact
    
    Output:             time series object with repeats replaced by NaNs

    
    
    med_outliers: uses a median filter on a rolling window to detect any
    outliers, making two passes. The function subtracts the median of a window
    centered on each value from the value itself, and then compares this
    difference to the interquartile range on said window. If the difference
    is too large relative to the IQR, the value is marked as an outlier. 
    The original concept comes from Basu & Meckesheimer (2007) although they 
    didn't use the interquartile range but rather expert judgment. This is
    an option here as well, if there is a known value that can be inserted,
    but otherwise the IQR is recommended.
    
    The function completes two passes, first a secondary pass on a large window
    (a couple days) and then the main pass on a window of default 7 values. The
    main pass is more accurate, but gets messed up when there are many outliers
    near each other, hence the secondary pass.
    
    Inputs: ts:             the time series
            secfilt:        boolean whether or not to make the secondary pass
                            (default True)
            level:          the number of times the IQR the distance between
                            the median and value can be before the value is
                            considered an outlier (default 3.0)
            scale:          the manual replacement for the IQR (default None)
            filt_len:       the length of the window (must be odd, default 7)
            quantiles:      a tuple of quantiles as the bounds for the IQR
                            (numbers between 0 and 100)
            seclevel:       same as level but for the larger pass
            secscale:       same as scale but for the larger pass
            secfilt_len:    same as filt_len but for the larger pass
            secquantiles:   same as quantiles but for the larger pass
            copy:           if True (default), a copy is made, leaving the
                            original series intact
            
            Output:         time series object with outliers replaced by NaNs
"""


def _nrepeat(ts):
    """Series-only version"""
    mask = ts.ne(ts.shift())
    counts = ts.groupby(mask.cumsum()).transform("count")
    return counts


def nrepeat(ts):
    """Return the length of consecutive runs of repeated values

    Parameters
    ----------

    ts:  DataFrame or series

    Returns
    -------
    Like-indexed series with lengths of runs. Nans will be mapped to 0

    """
    if isinstance(ts, pd.Series):
        return _nrepeat(ts)
    return ts.apply(_nrepeat, axis=0)


def threshold(ts, bounds, copy=True):
    ts_out = ts.copy() if copy else ts
    warnings.filterwarnings("ignore")
    if bounds is not None:
        if bounds[0] is not None:
            ts_out.mask(ts_out < bounds[0], inplace=True)
        if bounds[1] is not None:
            ts_out.mask(ts_out > bounds[1], inplace=True)
    return ts_out


def bounds_test(ts, bounds):
    # Make a boolean mask with exactly the same shape as ts
    if isinstance(ts, pd.Series):
        anomaly = pd.Series(False, index=ts.index, name=ts.name, dtype=bool)
    else:
        anomaly = pd.DataFrame(False, index=ts.index, columns=ts.columns, dtype=bool)

    if bounds is not None:
        lo, hi = bounds
        if lo is not None:
            anomaly |= (ts < lo)
        if hi is not None:
            anomaly |= (ts > hi)
            
    return anomaly


def median_test(ts, level=4, filt_len=7, quantiles=(0.005, 0.095), copy=True):
    return med_outliers(
        ts,
        level=level,
        filt_len=filt_len,
        quantiles=quantiles,
        copy=False,
        as_anomaly=True,
    )


def median_test_oneside(
    ts,
    scale=None,
    level=4,
    filt_len=6,
    quantiles=(0.005, 0.095),
    copy=True,
    reverse=False,
):
    if copy:
        ts = ts.copy()
    kappa = filt_len // 2
    if reverse:
        original_index = ts.index
        vals = ts[::-1]
    else:
        vals = ts
    vals = to_dataframe(vals)
    vals.columns = ["ts"]

    vals["z"] = vals.ts.diff()
    min_periods = kappa * 2 - 1

    dds = dd.from_pandas(vals, npartitions=50)
    dds["my"] = dds["ts"].shift().rolling(kappa * 2, min_periods=min_periods).median()
    dds["mz"] = dds.z.shift().rolling(kappa * 2, min_periods=min_periods).median()
    dds["pred"] = dds.my + kappa * dds.mz
    res = (dds.ts - dds.pred).compute()
    if scale is None:
        qq = res.quantile(q=quantiles)
        scale = qq.loc[quantiles[1]] - qq.loc[quantiles[0]]
    anomaly = (res.abs() > level * scale) | (res.abs() < -level * scale)

    if reverse:
        anomaly = anomaly[::-1]
        anomaly.index = original_index

    return anomaly


def med_outliers(
    ts,
    level=4.0,
    scale=None,
    filt_len=7,
    range=(None, None),
    quantiles=(0.01, 0.99),
    copy=True,
    as_anomaly=False,
):
    """
    Detect outliers by running a median filter, subtracting it
    from the original series and comparing the resulting residuals
    to a global robust range of scale (the interquartile range).
    Individual time points are rejected if the residual at that time point is more than level times the range of scale.

    The original concept comes from Basu & Meckesheimer (2007)
    Automatic outlier detection for time series: an application to sensor data
    although they didn't use the interquartile range but rather
    expert judgment. To use this function effectively, you need to
    be thoughtful about what the interquartile range will be. For instance,
    for a strongly tidal flow station it is likely to

    level: Number of times the scale or interquantile range the data has to be
           to be rejected.d

    scale: Expert judgment of the scale of maximum variation over a time step.
           If None, the interquartile range will be used. Note that for a
           strongly tidal station the interquartile range may substantially overestimate the reasonable variation over a single time step, in which case the filter will work fine, but level should be set to
           a number (less than one) accordingly.

    filt_len: length of median filter, default is 5

    quantiles : tuple of quantiles defining the measure of scale. Ignored
          if scale is given directly. Default is interquartile range, and
          this is almost always a reasonable choice.

    copy: if True, a copy is made leaving original series intact

    You can also specify rejection of  values based on a simple range

    Returns: copy of series with outliers replaced by nan
    """
    import warnings

    ts_out = ts.copy() if copy else ts
    warnings.filterwarnings("ignore")

    if range is not None:
        threshold(ts_out, range, copy=False)

    vals = ts_out.to_numpy()
    # if ts_out.ndim == 1:
    #    filt = medfilt(vals,filt_len)
    # else:
    #    filt = np.apply_along_axis(medfilt,0,vals,filt_len)
    filt = ts_out.rolling(filt_len, center=True).median()

    res = ts_out - filt

    if scale is None:
        qq = res.quantile(q=quantiles)
        scale = qq.loc[quantiles[1]] - qq.loc[quantiles[0]]

    anomaly = (res.abs() > level * scale) | (res.abs() < -level * scale)
    if as_anomaly:
        return anomaly
    # apply anomaly by setting values to nan
    values = np.where(anomaly, np.nan, ts_out.values)
    ts_out.iloc[:] = values
    warnings.resetwarnings()
    return ts_out


def median_test_twoside(
    ts,
    level=4.0,
    scale=None,
    filt_len=7,
    quantiles=(0.01, 0.99),
    copy=True,
    as_anomaly=True,
):
    """
    Detect outliers by running a median filter, subtracting it
    from the original series and comparing the resulting residuals
    to a global robust range of scale (the interquartile range).
    Individual time points are rejected if the residual at that time point is more than level times the range of scale.

    The original concept comes from Basu & Meckesheimer (2007)
    Automatic outlier detection for time series: an application to sensor data
    although they didn't use the interquartile range but rather
    expert judgment. To use this function effectively, you need to
    be thoughtful about what the interquartile range will be. For instance,
    for a strongly tidal flow station it is likely to

    level: Number of times the scale or interquantile range the data has to be
           to be rejected.d

    scale: Expert judgment of the scale of maximum variation over a time step.
           If None, the interquartile range will be used. Note that for a
           strongly tidal station the interquartile range may substantially overestimate the reasonable variation over a single time step, in which case the filter will work fine, but level should be set to
           a number (less than one) accordingly.

    filt_len: length of median filter, default is 5

    quantiles : tuple of quantiles defining the measure of scale. Ignored
          if scale is given directly. Default is interquartile range, and
          this is almost always a reasonable choice.

    copy: if True, a copy is made leaving original series intact

    You can also specify rejection of  values based on a simple range

    Returns: copy of series with outliers replaced by nan
    """
    import warnings

    ts_out = ts.copy() if copy else ts
    warnings.filterwarnings("ignore")

    vals = ts_out.to_numpy()
    # if ts_out.ndim == 1:
    #    filt = medfilt(vals,filt_len)
    # else:
    #    filt = np.apply_along_axis(medfilt,0,vals,filt_len)

    def mseq(flen):
        halflen = flen // 2
        a = np.arange(0, halflen)
        b = np.arange(halflen + 1, flen)
        return np.concatenate((a, b))

    medseq = mseq(filt_len)

    dds = dd.from_pandas(ts_out, npartitions=1)
    filt = (
        dds.rolling(filt_len, center=True)
        .apply(lambda x: np.nanmedian(x[medseq]), raw=True, engine="numba")
        .compute()
    )
    res = ts_out - filt

    if scale is None:
        qq = res.quantile(q=quantiles)
        scale = qq.loc[quantiles[1]] - qq.loc[quantiles[0]]

    anomaly = (res.abs() > level * scale) | (res.abs() < -level * scale)
    if as_anomaly:
        return anomaly
    # apply anomaly by setting values to nan
    values = np.where(anomaly, np.nan, ts_out.values)
    ts_out.iloc[:] = values
    warnings.resetwarnings()
    return ts_out


def gapdist_test_series(ts, smallgaplen=0):
    test_gap = ts.copy()
    gapcount = gap_count(ts)
    testgapnull = test_gap.isnull()
    is_small_gap = gapcount <= smallgaplen
    smallgap = testgapnull & is_small_gap
    test_gap.where(~smallgap, -99999999.0, inplace=True)
    return test_gap


def steep_then_nan(
    ts,
    level=4.0,
    scale=None,
    filt_len=11,
    range=(None, None),
    quantiles=(0.01, 0.99),
    copy=True,
    as_anomaly=True,
    *,
    gap_aggregation="any",
    smallgaplen=3,
    neargapdist=5,
):
    """Detect outliers via a median-filter residual test, gated by proximity to large gaps.

    This is a Basu/median-style residual detector (median filter + robust global scale)
    with an additional *gap proximity gate*: only points close to a *large* gap are
    eligible to be flagged. "Small gaps" are treated as ignorable by converting them
    to a sentinel before computing gap distance.

    Parameters
    ----------
    ts : pandas.Series or pandas.DataFrame
        Input time series. For DataFrames, residual/outlier detection is performed
        per column.
    level : float, default 4.0
        Threshold multiplier applied to the robust scale. A point is an outlier if
        ``abs(residual) > level * scale``.
    scale : float or pandas.Series, optional
        If None, compute a robust global scale from residual quantiles.
        If provided, it is used directly (scalar or per-column Series).
    filt_len : int, default 11
        Median filter window length passed to ``scipy.signal.medfilt``.
    range : tuple, default (None, None)
        Optional (lo, hi) bounds applied to the input before residual calculation.
        If lo or hi is None, that side is not applied.
    quantiles : tuple[float, float], default (0.01, 0.99)
        Quantiles defining the robust range used as the scale when ``scale`` is None.
    copy : bool, default True
        If True, work on a copy of ``ts``.
    as_anomaly : bool, default True
        If True, return a boolean anomaly mask (same shape as ``ts``).
        If False, return a copy of ``ts`` with anomalies replaced by NaN.
    gap_aggregation : {"any", "all"}, default "any"
        How to combine gap proximity across columns when ``ts`` is a DataFrame.

        - "any": a time is considered near a large gap if ANY column is near a gap.
        - "all": a time is considered near a large gap only if ALL columns are near a gap.

        This only affects the near-gap gate; residual outliers are still computed
        per column.
    smallgaplen : int, default 3
        Gaps of length <= ``smallgaplen`` are treated as "small" and replaced by a
        sentinel before gap distance is computed.
    neargapdist : int, default 5
        A time is considered "near a large gap" if its distance-to-gap (in samples)
        is < ``neargapdist``.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        If ``as_anomaly=True``, returns a boolean mask with the same shape as ``ts``.
        If ``as_anomaly=False``, returns a copy of ``ts`` with flagged points set to NaN.

    Notes
    -----
    - Gap proximity is computed using ``gapdist_test_series`` and ``gap_distance``.
      For DataFrames, the per-column gap distance is reduced to a *time mask*
      using ``gap_aggregation``.
    - This function is intended for offline QC where gap-adjacent behavior is
      considered more suspect than behavior in continuous data.
    """
    import warnings
    import numpy as np
    import pandas as pd
    from scipy.signal import medfilt

    ts_out = ts.copy() if copy else ts
    warnings.filterwarnings("ignore")

    # Optional hard bounds (mutates ts_out)
    if range is not None:
        threshold(ts_out, range, copy=False)

    # ---- Gap proximity gate (time mask) ----
    test_gap = gapdist_test_series(ts_out, smallgaplen=smallgaplen)
    gapdist = gap_distance(test_gap, disttype="count", to="bad")

    near = gapdist < neargapdist
    if isinstance(near, pd.DataFrame):
        if gap_aggregation == "any":
            near_t = near.any(axis=1)
        elif gap_aggregation == "all":
            near_t = near.all(axis=1)
        else:
            raise ValueError(f"gap_aggregation must be 'any' or 'all', got {gap_aggregation!r}")
    else:
        near_t = near

    # ---- Median filter + residuals (per column) ----
    vals = ts_out.to_numpy()
    if ts_out.ndim == 1:
        filt = medfilt(vals, filt_len)
        filt = pd.Series(filt, index=ts_out.index, name=ts_out.name)
    else:
        filt = np.apply_along_axis(medfilt, 0, vals, filt_len)
        filt = pd.DataFrame(filt, index=ts_out.index, columns=ts_out.columns)

    res = ts_out - filt

    # ---- Robust scale ----
    if scale is None:
        qq = res.quantile(q=quantiles)
        # Series input -> qq is Series (indexed by quantiles)
        # DataFrame input -> qq is DataFrame (index=quantiles, columns=ts_out.columns)
        scale = qq.loc[quantiles[1]] - qq.loc[quantiles[0]]

    # ---- Outlier mask (per column / per series) ----
    outlier = res.abs() > (level * scale)

    # Gate by near-gap time mask
    if isinstance(outlier, pd.DataFrame):
        outlier = outlier & near_t.to_numpy()[:, None]
    else:
        outlier = outlier & near_t

    if as_anomaly:
        warnings.resetwarnings()
        return outlier

    # Apply via pandas alignment (avoids shape/broadcast pitfalls)
    ts_out = ts_out.mask(outlier)

    warnings.resetwarnings()
    return ts_out

def despike(arr, n1=2, n2=20, block=10, *, as_anomaly=False):
    """Detect and optionally remove isolated spikes using overlapping-window statistics.

    This is a legacy-style *spike scrubber* rather than a pure anomaly detector.
    It runs two passes:

    1) Compute a rolling (left-aligned) window mean/std and flag values whose
       deviation exceeds ``n1 * std``. Those flagged values are temporarily set
       to NaN.
    2) Recompute rolling window mean/std on the NaN-masked signal and flag values
       whose deviation exceeds ``n2 * std``. These flags are the final spikes.

    Parameters
    ----------
    arr : array-like
        1D numeric signal.
    n1 : float, default 2
        Pass-1 threshold multiplier on the rolling standard deviation.
    n2 : float, default 20
        Pass-2 threshold multiplier on the rolling standard deviation.
    block : int, default 10
        Rolling window length (number of samples). Windows are **left-aligned**
        (i.e., window i is arr[i:i+block]). The final mean/std arrays are extended
        by repeating the last window's stats for the last ``block-1`` samples,
        matching the historical behavior.
    as_anomaly : bool, default False
        If False (default), return a cleaned array where spikes are replaced with NaN.
        If True, return a boolean mask (True where spikes are detected by pass-2).

    Returns
    -------
    np.ndarray
        If ``as_anomaly`` is False: a float array (same length) with spikes set to NaN.
        If ``as_anomaly`` is True: a boolean array (same length) marking spikes.

    Notes
    -----
    - The implementation uses NumPy's ``sliding_window_view`` instead of an external
      ``rolling_window`` helper to make the behavior explicit and self-contained.
    - Any NaNs in the input are treated as missing data via masked arrays.
    - This function copies the input and does **not** modify it in-place.
    """
    import numpy as np
    import numpy.ma as ma
    from numpy.lib.stride_tricks import sliding_window_view

    x = np.asarray(arr, dtype=float).copy()
    if x.ndim != 1:
        raise ValueError("despike expects a 1D array")

    if not isinstance(block, (int, np.integer)) or block <= 0:
        raise ValueError("block must be a positive integer")

    if x.size < block:
        raise ValueError(f"block={block} is larger than input length {x.size}")

    # Offset to reduce negative values; ignore NaNs (treated as missing).
    offset = np.nanmin(x)
    if not np.isfinite(offset):
        raise ValueError("input contains no finite values")

    x0 = x - offset  # offset-normalized signal

    def _roll_stats(sig, mult):
        # shape: (n-block+1, block)
        roll = sliding_window_view(sig, window_shape=block)
        roll = ma.masked_invalid(roll)
        std = mult * roll.std(axis=1)
        mean = roll.mean(axis=1)

        # Extend to length n by repeating last window's stats (legacy behavior).
        if block > 1:
            std = np.r_[std, np.tile(std[-1], block - 1)]
            mean = np.r_[mean, np.tile(mean[-1], block - 1)]
        return mean.filled(np.nan), std.filled(np.nan)

    # Pass 1: conservative flagging, then mask flagged values.
    mean1, std1 = _roll_stats(x0, n1)
    mask1 = np.abs(x0 - mean1) > std1

    x1 = x0.copy()
    x1[mask1] = np.nan

    # Pass 2: aggressive flagging on the masked signal.
    mean2, std2 = _roll_stats(x1, n2)
    mask2 = np.abs(x0 - mean2) > std2

    if as_anomaly:
        return mask2.astype(bool)

    out = x0.copy()
    out[mask2] = np.nan
    return out + offset


def example():
    station = sys.argv[1]
    ts = read_cdec("cdec_download/%s.csv" % station, start=None, end=None)

    filt = medfilt(ts.data, kernel_size=5)
    ts, filt = med_outliers(ts, quantiles=[0.2, 0.8], range=[120.0, None], copy=False)

    plt.plot(ts.times, ts.data, label="data")
    plt.plot(ts.times, filt.data, label="filt")
    plt.plot(ts.times, ts.data - filt.data, label="res")
    plt.legend()
    plt.show()


def example2():
    data = np.arange(32) * 2.0 + np.cos(np.arange(32) * 2 * np.pi / 24.0)
    ndx = pd.date_range(pd.Timestamp(2000, 1, 1), periods=len(data), freq="15min")
    df = pd.DataFrame(data=data, index=ndx)
    median_test_oneside(df, quantiles=(0.25, 0.75), level=2)


if __name__ == "__main__":
    example2()
    # Just an example
