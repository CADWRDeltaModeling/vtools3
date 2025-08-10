#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


__all__ = ["climatology", "apply_climatology", "climatology_quantiles"]


def climatology(ts, freq, nsmooth=None):
    """ " Create a climatology on the columns of ts

    Parameters
    ----------

    ts: DataFrame or Series
    DataStructure to be analyzed. Must have a length of at least 2*freq

    freq: period ["day","month"]
    Period over which the climatology is analyzed

    nsmooth: int
       window size (number of values) of pre-smoothing. This may not make sense for series that are not approximately regular. An odd number is usually best.

    Returns:
       out: DataFrame or Series
       Data structure of the same type as ts, with Integer index representing month (Jan=1) or day of year (1:365).

    """
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
        raise ValueError("invalid frequency, must be 'month' or 'day'")

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


def apply_climatology(climate, index):
    """Apply daily or monthly climatology to a new index

    Parameters
    ----------

    climate: DataFrame with integer index representing month of year (Jan=1) or day of year. Must be of size 12 365,366. Day 366 will be inferred from day 365 value

    index: DatetimeIndex representing locations to be inferred

    Returns
    -------
    DataFrame or Series as given by climate with values extracted from climatology for the month or day
    """


    if len(climate) not in [12, 365, 366]:
        raise ValueError("Length of climatology must be 12,365 or 366")
    if len(climate) == 365:
        climate.loc[366, :] = climate.loc[365, :]

    freq = "month" if len(climate) == 12 else "day"

    # Vectorized lookup
    if freq == "day":
        dayofyear = index.dayofyear
        # Ensure day 366 is handled
        dayofyear = np.where(dayofyear > 365, 366, dayofyear)
        out = climate.loc[dayofyear].set_index(index)
    else:
        month = index.month
        out = climate.loc[month].set_index(index)

    return out
