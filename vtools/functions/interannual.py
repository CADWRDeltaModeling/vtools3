#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import calendar

all = ["interannual", "interannual_ticks_labels"]


def interannual(ts):
    """Pivots years of multiyear series to columns and convert index to elapsed time of year

    Parameters
    ----------
    ts : series  Univariate series

    Returns
    -------
    annual : DataFrame with year in the columns and elapsed time of year as index
    """
    out = ts.copy()
    try:
        out = out.to_frame()
    except:
        pass
    out.columns = ["value"]
    out["year"] = ts.index.year
    out["secondofyear"] = (
        86400.0 * (ts.index.dayofyear - 1)
        + ts.index.hour * 3600.0
        + ts.index.minute * 60.0
        + +ts.index.second
    )
    out = out.pivot(index="secondofyear", columns="year", values="value")
    return out


# These labels and ticks are leap-unaware
def interannual_ticks_labels(freq):
    """Convenient ticks and labels for interannual

    Parameters
    ----------
    freq : Frequency string for desired spacing. Must be "Q","QS","M","MS" for quarterly or monthly

    Returns
    -------
    ticks_labels : tuple of tick locations and labels


    """

    if freq == "Q" or freq == "QS":
        quarterly_ticks = [
            (pd.Timestamp(2001, x, 1).dayofyear - 1) * 86400 for x in [1, 4, 7, 10]
        ]
        quarterly_labels = ["Jan", "Apr", "Jul", "Oct"]
        return (quarterly_ticks, quarterly_labels)
    elif freq == "M" or freq == "MS":
        monthly_ticks = [
            (pd.Timestamp(2001, x, 1).dayofyear - 1) * 86400 for x in range(1, 13)
        ]
        monthly_labels = [calendar.month_abbr[i] for i in range(1, 13)]
        return (monthly_ticks, monthly_labels)


def interannual_sample():
    dr = pd.date_range(start="2008-02-06", end="2018-11-02", freq="15T")
    df = pd.DataFrame(index=dr, data=np.arange(0, len(dr)) / 1000.0)
    return df
