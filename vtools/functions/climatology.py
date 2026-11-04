#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


__all__ = ["climatology", "apply_climatology", "climatology_quantiles"]


HOURS_PER_YEAR = 365 * 24


def _to_hourly(ts):
    """Average ts into hourly bins labeled at the start of the hour, [hh:00, hh+1:00).

    Data that is already hourly is returned unchanged (apart from regularization).
    Data coarser than hourly raises ValueError.
    """
    if len(ts) > 1:
        step = pd.Series(ts.index).diff().median()
        if step > pd.Timedelta("1h"):
            raise ValueError("Hourly climatology requires data at hourly or finer time step")
    return ts.resample("h").mean()


def _hour_of_year(index):
    """Hour of year (1..8760) on a 365-day calendar for a DatetimeIndex.

    Feb 29 is mapped onto Feb 28 so that leap years share the 365-day climatology.
    """
    doy = np.asarray(index.dayofyear)
    leap = np.asarray(index.is_leap_year)
    doy = np.where(leap & (doy > 59), doy - 1, doy)
    return (doy - 1) * 24 + np.asarray(index.hour) + 1


def _circular_mean(values, window, axis):
    """NaN-aware centered moving average with wraparound along axis.

    Positions with no valid values in the window remain NaN.
    """
    if window is None or window == 1:
        return values
    if window < 1 or window % 2 == 0:
        raise ValueError("Smoothing windows must be odd positive integers")
    if window > values.shape[axis]:
        raise ValueError("Smoothing window is longer than the cycle being smoothed")
    half = window // 2
    valid = ~np.isnan(values)
    filled = np.where(valid, values, 0.0)
    total = np.zeros_like(filled)
    count = np.zeros(filled.shape)
    for shift in range(-half, half + 1):
        total += np.roll(filled, shift, axis=axis)
        count += np.roll(valid, shift, axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


def _climatology_hourly(ts, nsmooth, nday):
    """Hour-of-year climatology (8760 values) with circular smoothing; see climatology."""
    if not isinstance(ts.index, pd.DatetimeIndex):
        raise ValueError("ts must have a DatetimeIndex")
    hourly = _to_hourly(ts)
    # Feb 29 is dropped, consistent with the daily climatology
    hourly = hourly[~((hourly.index.month == 2) & (hourly.index.day == 29))]
    is_series = isinstance(hourly, pd.Series)
    frame = hourly.to_frame() if is_series else hourly

    hoy = _hour_of_year(frame.index)
    mean = frame.groupby(hoy).mean().reindex(range(1, HOURS_PER_YEAR + 1))
    vals = mean.to_numpy(dtype=float)

    # Pool the same hour of neighboring days, then neighboring hours
    ncol = vals.shape[1]
    vals = _circular_mean(vals.reshape(365, 24, ncol), nday, axis=0)
    vals = _circular_mean(vals.reshape(HOURS_PER_YEAR, ncol), nsmooth, axis=0)

    out = pd.DataFrame(vals, index=mean.index, columns=mean.columns)
    out.index.name = "hourofyear"
    return out.iloc[:, 0].rename(hourly.name) if is_series else out


def climatology(ts, freq, nsmooth=None, nday=None):
    """ " Create a climatology on the columns of ts

    Parameters
    ----------

    ts: DataFrame or Series
    DataStructure to be analyzed. Must have a length of at least 2*freq

    freq: period ["hour","day","month"]
    Period over which the climatology is analyzed. "hour" produces one value for each hour of
    a 365-day year (8760 values); Feb 29 is ignored. Data are first averaged into hourly bins
    labeled at the start of the hour, so ts must be hourly or finer.

    nsmooth: int
       For "day" and "month": window size (number of values) of pre-smoothing of the input. This may not make sense for series that are not approximately regular. An odd number is usually best.
       For "hour": odd window size in hours (e.g. 3 for +/- 1 hour) applied to the hourly climatology after
       averaging across years, wrapping around the end of the year.

    nday: int
       Only for "hour". Odd number of days (e.g. 5 for +/- 2 days) over which the same hour of day is pooled
       to strengthen the estimate, wrapping around the end of the year. None means no pooling across days.

    Returns:
       out: DataFrame or Series
       Data structure of the same type as ts, with Integer index representing month (Jan=1), day of year (1:365)
       or hour of year (1:8760, Jan 1 00:00 = 1). Hourly values with no data in their window are NaN.

    """
    if freq == "hour":
        return _climatology_hourly(ts, nsmooth, nday)
    if nday is not None:
        raise ValueError("nday is only valid for freq='hour'")

    if nsmooth is None:
        ts_mean = ts
    else:
        ts_mean = ts.rolling(
            nsmooth, center=True, min_periods=nsmooth // 2
        ).mean()  # moving average

    by = [ts_mean.index.month, ts_mean.index.day]
    if freq == "month":
        by = [ts_mean.index.month]
    elif not (freq == "day"):
        raise ValueError("invalid frequency, must be 'month', 'day' or 'hour'")

    mean_data = []
    mean_data_size = []
    for name, group in ts_mean.groupby(by):
        if len(by) == 2:
            (mo, dy) = name
            if not ((mo == 2) and (dy == 29)):
                mean_data.append(group.mean(axis=0))
                mean_data_size.append(group.count())
        else:
            mean_data.append(group.mean(axis=0))
            mean_data_size.append(group.count())

    climatology_data = pd.concat(mean_data, axis=1).transpose()
    indexvals = list(
        range(1, 13) if freq == "month" else range(1, len(climatology_data) + 1)
    )

    climatology_data.index = list(indexvals)
    climatology_data.index.name = "month" if freq == "month" else "dayofyear"
    return climatology_data


def climatology_quantiles(
    ts,
    min_day_year,
    max_day_year,
    window_width,
    quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
):
    """ " Create windowed quantiles across years on a time series

    Parameters
    ----------

    ts: DataFrame or Series
    DataStructure to be analyzed.

    min_day_year: int
    Minimum Julian day to be considered

    freq: period ["day","month"]
    Maximum Julian day to be considered

    window_width: int
    Number of days to include, including the central day and days on each side. So for instance window_width=15 would span the central date and 7 days on each side

    quantiles: array-like
       quantiles requested

    Returns:
       out: DataFrame or Series
       Data structure with Julian day as the index and quantiles as columns.

    """

    if (min_day_year < window_width) or (max_day_year > (365 - window_width)):
        raise NotImplementedError(
            "Time brackets that cross January 1 not implemented yet"
        )
    if window_width % 2 == 0:
        raise ValueError("window_width must be odd")
    window_half = (window_width - 1) / 2
    day_year = ts.index.dayofyear
    nquant = len(quantiles)
    clim = pd.DataFrame(columns=quantiles, index=range(min_day_year, max_day_year))

    for imid in range(min_day_year, max_day_year):
        iend = (
            imid + window_half + 1
        )  # The plus one centers the estimate, equal on each side plus one for center
        istart = imid - window_half
        usets = ts[(day_year > istart) & (day_year < iend)]
        qs = usets.quantile(quantiles)
        clim.loc[imid, :] = qs.values.flatten()
    clim.index.name = "day_of_year"
    return clim


SUBSTEP_OPTIONS = ("fill", "linear", "rhistinterp")
_PERIOD_UNIT = {"month": "M", "day": "D", "hour": "h"}


def _climatology_kind(climate):
    n = len(climate)
    if n not in [12, 365, 366, HOURS_PER_YEAR]:
        raise ValueError("Length of climatology must be 12, 365, 366 or 8760")
    return {12: "month", HOURS_PER_YEAR: "hour"}.get(n, "day")


def _fill_climatology(climate, kind, index):
    """Value of the climatology period (month, day or hour) containing each timestamp."""
    if kind == "hour":
        out = climate.loc[_hour_of_year(index)]
        out.index = index
    elif kind == "day":
        dayofyear = index.dayofyear
        # Ensure day 366 is handled
        dayofyear = np.where(dayofyear > 365, 366, dayofyear)
        out = climate.loc[dayofyear]
    else:
        out = climate.loc[index.month]
    out.index = index
    return out


def _interpolate_climatology(climate, kind, index, method, **kwargs):
    """Treat climatology values as period averages and interpolate them to index."""
    unit = _PERIOD_UNIT[kind]
    # One extra period on each side so that every requested time is interior
    periods = pd.period_range(
        index.min().to_period(unit) - 1, index.max().to_period(unit) + 1
    )
    starts = periods.to_timestamp()
    values = _fill_climatology(climate, kind, starts)

    if method == "linear":
        # period averages are anchored at the middle of their period
        middles = starts + (((periods + 1).to_timestamp() - starts) / 2)
        values.index = middles
        union = middles.union(index)
        return values.reindex(union).interpolate(method="time").reindex(index)

    if np.isnan(values.to_numpy(dtype=float)).any():
        raise ValueError("rhistinterp requires a climatology with no missing values")
    from vtools.functions.rhistinterp import rhistinterp

    values.index = periods
    out = rhistinterp(values, index, **kwargs)
    if isinstance(values, pd.Series):
        out.name = values.name
    return out


def apply_climatology(
    climate, index=None, start=None, end=None, freq=None, substep_interp="fill", **kwargs
):
    """Apply hourly, daily or monthly climatology to a new index or generate index from start/end/freq

    Parameters
    ----------
    climate : DataFrame or Series
        DataFrame with integer index representing month of year (Jan=1), day of year or hour of year. Must be of size 12, 365, 366 or 8760. Day 366 will be inferred from day 365 value. An hourly (8760) climatology uses a 365-day calendar: Feb 29 takes the Feb 28 values. The values are treated as averages over their period (month, day or hour beginning at the label).
    index : pandas.DatetimeIndex, optional
        Locations to be inferred. If not provided, must specify start, end, and freq.
    start : str or datetime-like, optional
        Start date for generating index (used if index is None).
    end : str or datetime-like, optional
        End date for generating index (used if index is None).
    freq : str, optional
        Frequency string for generating index (used if index is None). E.g., 'D' for daily, 'M' for monthly.
    substep_interp : {"fill", "linear", "rhistinterp"}, default "fill"
        How values are obtained at times within a climatology period (e.g. a 15min index with a daily climatology).

        - "fill": each timestamp takes the value of the month, day or hour containing it, giving a step function.
        - "linear": linear interpolation in time between the middles of the periods.
        - "rhistinterp": rational histospline (see :func:`vtools.rhistinterp`) that preserves the period averages. Requires a climatology without missing values.
    **kwargs
        Only for ``substep_interp="rhistinterp"``; passed to :func:`vtools.rhistinterp`. Notably ``p``, the spline tension (default 2.0; larger values are closer to monotonic and strictly positive), and ``lowbound``, a lower bound for the interpolated values.

    Returns
    -------
    DataFrame or Series
        Values extracted from climatology for the month, day or hour at the specified index.

    Notes
    -----
    - If index is not provided, start, end, and freq must be specified to generate a DatetimeIndex using pandas.date_range.
    """
    kind = _climatology_kind(climate)
    if substep_interp not in SUBSTEP_OPTIONS:
        raise ValueError(
            f"substep_interp must be one of {SUBSTEP_OPTIONS}, got {substep_interp!r}"
        )
    if kwargs and substep_interp != "rhistinterp":
        raise ValueError(
            f"Unexpected arguments {sorted(kwargs)}: extra keyword arguments are only used with "
            "substep_interp='rhistinterp'"
        )
    if len(climate) == 365:
        climate.loc[366, :] = climate.loc[365, :]

    if index is None:
        if start is None or end is None or freq is None:
            raise ValueError("If index is not provided, start, end, and freq must be specified")
        index = pd.date_range(start=start, end=end, freq=freq)

    if substep_interp == "fill":
        return _fill_climatology(climate, kind, index)
    return _interpolate_climatology(climate, kind, index, substep_interp, **kwargs)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from dms_datastore.read_ts import read_ts

    fname = "//cnrastore-bdo/Modeling_Data/continuous_station_repo/raw/des_twi_405_turbidity_*.csv"
    fname = "//cnrastore-bdo/Modeling_Data/continuous_station_repo/raw/usgs_lib*turbidity*.rdb"
    selector = "16127_63680"
    ts = read_ts(fname, selector=selector)
    window = 19  # central day plus 9 on each side
    clim = climatology_quantiles(ts, 182, 305, window)
    clim.plot()
    plt.show()
