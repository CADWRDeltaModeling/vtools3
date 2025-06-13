#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Time series module
Helpers for creating regular and irregular time series, transforming irregular to regular
and analyzing gaps.
"""
import sys
import numpy as np
import pandas as pd
from vtools.data.vtime import *
import datetime as _datetime


__all__ = [
    "to_dataframe",
    "time_overlap",
    "rts",
    "rts_formula",
    "extrapolate_ts",
    "datetime_elapsed",
    "elapsed_datetime",
    "to_dataframe",
    "is_regular",
]


def to_dataframe(ts):
    if isinstance(ts, pd.DataFrame):
        return ts
    else:
        return ts.to_frame()


def time_overlap(ts0, ts1, valid=True):
    """Check for overlapping time coverage between series
    Returns a tuple of start and end of overlapping periods. Only considers
    the time stamps of the start/end, possibly ignoring NaNs at the beginning
    if valid=True, does not check for actual time stamp alignment
    """
    if valid:
        start = max(ts0.first_valid_index(), ts1.first_valid_index())
        end = min(ts0.last_valid_index(), ts1.last_valid_index())

    else:
        start = max(ts0.index[0], ts1.index[0])
        end = min(ts0.index[-1], ts1.index[-1])
    return (start, end) if end > start else None


def rts(data, start, freq, columns=None, props=None):
    """Create a regular or calendar time series from data and time parameters

    Parameters
    ----------
    data : array_like
        Should be a array/list of values. There is no restriction on data
         type, but not all functionality like addition or interpolation will work on all data.

    start : :class:`Pandas.Timestamp`
        Timestamp or a string or type that can be coerced to one.

    interval : _time_interval
        Can also be a string representing a pandas `freq`.

    Returns
    -------
    result :  :class:`Pandas.DataFrame`
        A regular time series with the `freq` attribute set
    """

    if type(data) == list:
        data = np.array(data)
    if not props is None:
        raise NotImplementedError(
            "Props reserved for future implementation using xarray"
        )
    tslen = data.shape[0]
    ndx = pd.date_range(start, freq=freq, periods=tslen)
    ts = pd.DataFrame(data, index=ndx, columns=columns)
    return ts


def rts_formula(start, end, freq, valfunc=np.nan):
    """Create a regular time series filled with constant value or formula based on elapsed seconds

    Parameters
    ----------

    start : :class:`Pandas.Timestamp`
        Starting Timestamp or a string or type that can be coerced to one.

    end : :class:`Pandas.Timestamp`
        Ending Timestamp or a string or type that can be coerced to one.

    freq : _time_interval
        Can also be a string representing an interval.

    valfunc : dict
        Constant or dictionary that maps column names to lambdas based on elapsed time from the starts of the series. An example would be {"value": lambda x: np.nan}

    Returns
    -------
    result :  :class:`Pandas.DataFrame`
        A regular time series with the `freq` attribute set

    """
    from numbers import Number as numtype

    ndx = pd.date_range(start=start, end=end, freq=freq)
    secs = (ndx - ndx[0]).total_seconds()

    if isinstance(valfunc, numtype):
        data = np.array([valfunc for x in secs])
        cols = ["value"]
    else:
        data = np.array([valfunc[x](secs) for x in valfunc]).T
        cols = valfunc.keys()
    ts = rts(data, start, freq, columns=cols)

    return ts


def extrapolate_ts(ts, start=None, end=None, method="ffill", val=None):
    """
    Extend a regular time series to a new start and/or end using a specified extrapolation method.

    Parameters
    ----------
    ts : pandas.Series or pandas.DataFrame
        The input time series with a DateTimeIndex and a regular frequency.

    start : datetime-like, optional
        The new starting time. If None, no extension is done before the existing data.

    end : datetime-like, optional
        The new ending time. If None, no extension is done after the existing data.

    method : {'ffill', 'bfill', 'linear_slope', 'taper', 'constant'}, default 'ffill'
        The method used to fill new values outside the original time range:

        - 'ffill' : Forward-fill after the original data using its last value.
        - 'bfill' : Backward-fill before the original data using its first value.
        - 'linear_slope' : Bidirectional linear extrapolation using the first/last two points.
        - 'taper' : One-sided linear interpolation to/from a specified value (`val`).
        - 'constant' : One-sided constant value fill with `val`.

    val : float, optional
        Required for 'taper' and 'constant'. Specifies the value to use.

    Returns
    -------
    extended : pandas.Series or pandas.DataFrame
        The time series extended and filled using the selected method.

    Raises
    ------
    ValueError
        - If extrapolation rules are violated based on the method.
        - If method requires or forbids `val` and it's misused.
        - If frequency cannot be inferred.

    """
    if not isinstance(ts, (pd.Series, pd.DataFrame)):
        raise TypeError("Input must be a pandas Series or DataFrame")

    freq = ts.index.freq or pd.infer_freq(ts.index)
    if freq is None:
        raise ValueError("Time series must have a regular frequency")

    start = pd.to_datetime(start) if start else ts.index[0]
    end = pd.to_datetime(end) if end else ts.index[-1]

    full_index = pd.date_range(start=start, end=end, freq=freq)
    ts_full = ts.reindex(full_index)

    print(start, ts.index[0], end, ts.index[-1])
    if method == "ffill":
        if start < ts.index[0]:

            raise ValueError("'ffill' not allowed when extending before start of data")
        ts_full.loc[ts.index[-1] :] = ts_full.loc[ts.index[-1] :].ffill()
        return ts_full.astype(ts.dtype)

    elif method == "bfill":
        if end > ts.index[-1]:
            raise ValueError("'bfill' not allowed when extending after end of data")
        ts_full.loc[: ts.index[0]] = ts_full.loc[: ts.index[0]].bfill()
        return ts_full.astype(ts.dtype)

    elif method == "linear_slope":
        if val is not None:
            raise ValueError("'linear_slope' does not use 'val'")
        if len(ts) < 2:
            raise ValueError(
                "At least 2 data points are required for slope-based extrapolation."
            )

        result = ts_full.copy().astype(float).interpolate(method="time")
        idx = ts.index

        # Forward extrapolation
        slope_end = (ts.iloc[-1] - ts.iloc[-2]) / (idx[-1] - idx[-2]).total_seconds()
        late = result.index[result.index > idx[-1]]
        seconds_late = (late - idx[-1]).total_seconds()
        result.loc[late] = ts.iloc[-1] + slope_end * seconds_late

        # Backward extrapolation
        slope_start = (ts.iloc[1] - ts.iloc[0]) / (idx[1] - idx[0]).total_seconds()
        early = result.index[result.index < idx[0]]
        seconds_early = (idx[0] - early).total_seconds()
        result.loc[early] = ts.iloc[0] - slope_start * seconds_early

        return result if isinstance(ts, pd.Series) else result.to_frame(ts.columns[0])

    elif method == "taper":
        if val is None:
            raise ValueError("Taper method requires 'val' to be specified.")
        if start < ts.index[0] and end > ts.index[-1]:
            raise ValueError("Taper method only supports one-sided extrapolation.")

        result = ts_full.copy().astype(float)

        if start < ts.index[0]:
            ramp_index = result.index[result.index < ts.index[0]]
            temp = pd.Series([val, ts.iloc[0]], index=[ramp_index[0], ts.index[0]])
            filled = (
                temp.reindex(ramp_index.union(temp.index))
                .interpolate(method="time")
                .loc[ramp_index]
            )
            result.loc[ramp_index] = filled.values

        elif end > ts.index[-1]:
            ramp_index = result.index[result.index > ts.index[-1]]
            temp = pd.Series([ts.iloc[-1], val], index=[ts.index[-1], ramp_index[-1]])
            filled = (
                temp.reindex(ramp_index.union(temp.index))
                .interpolate(method="time")
                .loc[ramp_index]
            )
            result.loc[ramp_index] = filled.values

        return result if isinstance(ts, pd.Series) else result.to_frame(ts.columns[0])

    elif method == "constant":
        if val is None:
            raise ValueError("Constant method requires 'val' to be specified.")
        if start < ts.index[0] and end > ts.index[-1]:
            raise ValueError("Constant method only supports one-sided extrapolation.")

        result = ts_full.copy()
        if start < ts.index[0]:
            result.loc[result.index < ts.index[0]] = val
        if end > ts.index[-1]:
            result.loc[result.index > ts.index[-1]] = val

        return result.astype(ts.dtype) if not result.isna().any().any() else result

    else:
        raise ValueError(f"Unknown method: {method}")


def datetime_elapsed(index_or_ts, reftime=None, dtype="d", inplace=False):
    """Convert a time series or DatetimeIndex to an integer/double series of elapsed time

    Parameters
    ----------

    index_or_ts : :class:`DatatimeIndex <pandas:pandas.DatetimeIndex> or :class:`DataFrame <pandas:pandas.DataFrame>`
        Time series or index to be transformed

    reftime :  :class:`DatatimeIndex <pandas:pandas.Timestamp>` or something convertible
        The reference time upon which elapsed time is measured. Default of None means start of
        series

    dtype : str like 'i' or 'd' or type like `int` (`Int64`) or `float` (`Float64`)
        Data type for output, which starts out as a Float64 ('d') and gets converted, typically to Int64 ('i')

    inplace : `bool`
        If input is a data frame, replaces the index in-place with no copy

    Returns
    -------
    result :
        A new index using elapsed time from `reftime` as its value and of type `dtype`

    """
    try:
        ndx = index_or_ts.index
        input_index = False
    except AttributeError as e:
        ndx = index_or_ts
        input_index = True

    if reftime is None:
        ref = ndx[0]
    else:
        ref = pd.Timestamp(reftime)

    elapsed = (ndx - ref).total_seconds().astype(dtype)
    if input_index:
        return elapsed
    if inplace:
        index_or_ts.index = elapsed
        return index_or_ts
    else:
        result = index_or_ts.copy()
        # Not sure of the merits of this relative to
        # result.index = ["elapsed"] = elapsed; result.reindex(key = 'elapsed',drop=True)
        result.index = elapsed
    return result


def elapsed_datetime(index_or_ts, reftime=None, time_unit="s", inplace=False):
    """Convert a time series or numerical Index to a Datetime index or series

    Parameters
    ----------

    index_or_ts : :class:`DatatimeIndex <pandas:pandas.Int64Index> or float or TimedeltaIndex :class:`DataFrame <pandas:pandas.DataFrame>`
        Time series or index to be transformed with index in elapsed seconds from `reftime`

    reftime :  :class:`DatatimeIndex <pandas:pandas.Timestamp>` or something convertible
        The reference time upon which datetimes are to be evaluated.

    inplace : `bool`
        If input is a data frame, replaces the index in-place with no copy

    Returns
    -------
    result :
        A new index using DatetimeIndex inferred from elapsed time from `reftime` as its value and of type `dtype`

    """

    try:
        ndx = index_or_ts.index
        input_index = False
    except AttributeError as e:
        ndx = index_or_ts
        input_index = True

    if isinstance(ndx, pd.TimedeltaIndex):
        dtndx = reftime + ndx
    else:
        if time_unit.lower() == "h":
            ndx = ndx * 3600.0
        elif time_unit.lower() == "d":
            ndx = ndx * 86400.0
        elif time_unit.lower() == "s":
            pass
        else:
            raise ValueError("time unit must be 's','h',or 'd'")
        dtndx = reftime + pd.to_timedelta(ndx, unit="s")

    if input_index:
        return dtndx
    if inplace:
        index_or_ts.index = dtndx
        return index_or_ts
    else:
        result = index_or_ts.copy()
        # Not sure of the merits of this relative to
        # result.index = ["elapsed"] = elapsed; result.reindex(key = 'elapsed',drop=True)
        result.index = dtndx
    return result


import pandas as pd


def is_regular(ts, raise_exception=False):
    """
    Check if a pandas DataFrame, Series, or xarray object with a time axis (axis 0)
    has a regular time index.

    Regular means:
      - The index is unique.
      - The index equals a date_range spanning from the first to the last value with
        the inferred frequency.

    Parameters:
      ts : DataFrame, Series, or xarray object.
      raise_exception (bool): If True, raises a ValueError when the index is not regular.
                              Otherwise, returns False.

    Returns:
      bool : True if the time index is regular; False otherwise.
    """
    # Determine the index from the object
    if hasattr(ts, "index"):
        idx = ts.index
    # For xarray objects, assume the first dimension is time.
    elif hasattr(ts, "coords") and ts.dims:
        time_dim = ts.dims[0]
        # Try to convert coordinate to a pandas Index
        coord = ts.coords[time_dim]
        if hasattr(coord, "to_index"):
            idx = coord.to_index()
        else:
            idx = pd.Index(coord.values)
    else:
        msg = "The provided object does not have an accessible time index."
        if raise_exception:
            raise ValueError(msg)
        return False

    # An empty or single-element index is considered regular.
    if len(idx) == 0 or len(idx) == 1:
        return True

    # Check if the index has duplicate values.
    if not idx.is_unique:
        msg = "Index contains duplicate values."
        if raise_exception:
            raise ValueError(msg)
        return False

    # Ensure we are working with a DatetimeIndex. If not, attempt conversion.
    if not isinstance(idx, pd.DatetimeIndex):
        try:
            idx = pd.to_datetime(idx)
        except Exception as e:
            msg = "Index could not be converted to datetime."
            if raise_exception:
                raise ValueError(msg) from e
            return False

    # Attempt to get the frequency. First check the .freq attribute.
    freq = idx.freq
    # If not set, try to infer it. This can often produce false negatives with messy data
    # but will not fail in this case because every timestamp is checked
    if freq is None:
        freq = pd.infer_freq(idx)
    if freq is None:
        msg = "Could not infer a frequency from the index; it may not be regular."
        if raise_exception:
            raise ValueError(msg)
        return False

    # Build the expected index using the determined frequency.
    expected_index = pd.date_range(start=idx[0], end=idx[-1], freq=freq)
    if not expected_index.equals(idx):
        msg = "Index is not regular based on the inferred frequency."
        if raise_exception:
            raise ValueError(msg)
        return False
    return True


from scipy.interpolate import PchipInterpolator


def transition_ts(
    ts0, ts1, method="linear", create_gap=None, overlap=(0, 0), return_type="series"
):
    if not isinstance(ts0, (pd.Series, pd.DataFrame)) or not isinstance(ts1, type(ts0)):
        raise ValueError("ts0 and ts1 must be of the same type (Series or DataFrame).")
    if ts0.index.freq != ts1.index.freq:
        raise ValueError("ts0 and ts1 must have the same frequency.")

    freq = ts0.index.freq

    # Determine transition interval
    if create_gap:
        trans_start = pd.Timestamp(create_gap[0])
        trans_end = pd.Timestamp(create_gap[1])

        # Start anchor
        if ts0.index[-1] < trans_start:
            start_time = ts0.index[-1]
            start_val = ts0.iloc[-1]
        else:
            start_time = ts0.loc[:trans_start].index[-1]
            start_val = ts0.loc[:trans_start].iloc[-1]

        # End anchor
        if ts1.index[0] > trans_end:
            end_time = ts1.index[0]
            end_val = ts1.iloc[0]
        else:
            end_time = ts1.loc[trans_end:].index[0]
            end_val = ts1.loc[trans_end:].iloc[0]

    else:
        trans_start = ts0.index[-1] + freq
        trans_end = ts1.index[0] - freq
        start_time = ts0.index[-1]
        start_val = ts0.iloc[-1]
        end_time = ts1.index[0]
        end_val = ts1.iloc[0]

    trans_index = pd.date_range(start=trans_start, end=trans_end, freq=freq)
    if len(trans_index) < 2:
        raise ValueError("Transition zone must have at least two steps.")

    # Interpolation
    elif method == "linear":
        total_duration = (end_time - start_time).total_seconds()
        rel_pos = [
            (t - start_time).total_seconds() / total_duration for t in trans_index
        ]

        if isinstance(ts0, pd.DataFrame):
            interpolated = pd.DataFrame(
                np.outer(1 - rel_pos, start_val) + np.outer(rel_pos, end_val),
                index=trans_index,
                columns=ts0.columns,
            )
        else:
            interpolated = pd.Series(
                [(1 - p) * start_val + p * end_val for p in rel_pos],
                index=trans_index,
                name=ts0.name,
            )

    elif method == "pchip":
        n_before, n_after = overlap

        seg0 = (
            ts0.loc[:trans_start].iloc[-n_before:]
            if n_before > 0
            else ts0.loc[[ts0.index[-1]]]
        )
        seg1 = (
            ts1.loc[trans_end:].iloc[:n_after]
            if n_after > 0
            else ts1.loc[[ts1.index[0]]]
        )
        all_data = pd.concat([seg0, seg1])
        all_data = all_data[~all_data.index.duplicated()].sort_index()

        if isinstance(ts0, pd.Series):
            interp = PchipInterpolator(all_data.index.astype(np.int64), all_data.values)
            interpolated = pd.Series(
                interp(trans_index.astype(np.int64)), index=trans_index, name=ts0.name
            )
        else:
            interpolated = pd.DataFrame(index=trans_index, columns=ts0.columns)
            for col in ts0.columns:
                interp = PchipInterpolator(
                    all_data.index.astype(np.int64), all_data[col].values
                )
                interpolated[col] = interp(trans_index.astype(np.int64))
    else:
        raise ValueError("Only 'linear' and 'pchip' methods are supported.")

    # Final output
    if return_type == "glue":
        return interpolated
    elif return_type == "series":
        ts0_trunc = ts0.loc[: trans_start - freq]
        ts1_trunc = ts1.loc[trans_end + freq :]
        return pd.concat([ts0_trunc, interpolated, ts1_trunc])
    else:
        raise ValueError("return_type must be either 'glue' or 'series'.")


def example():
    ndx = pd.date_range(pd.Timestamp(2017, 1, 1, 12), freq="15min", periods=10)
    out = datetime_elapsed(ndx, dtype="i")
    print(out)
    print(type(out))
    vals = np.arange(0.0, 10.0, dtype="d")
    df = pd.DataFrame({"vals": vals}, index=ndx.copy())
    ref = pd.Timestamp(2017, 1, 1, 11, 59)
    df2 = datetime_elapsed(df, reftime=ref, dtype=int)
    print(elapsed_datetime(df2, reftime=ref) - df)


if __name__ == "__main__":
    example()
