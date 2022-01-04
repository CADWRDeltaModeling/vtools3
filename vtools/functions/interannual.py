#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import calendar


def interannual(ts):
    """ Pivots years of multiyear series to columns and convert index to elapsed time of year
    
    Parameters
    ----------
    ts : series  Univariate series
    
    Returns
    -------
    annual : DataFrame with year in the columns and elapsed time of year as index
    """
    out = ts.copy()
    try: 
        out=out.to_frame()
    except:
        pass
    out.columns=["value"]
    out["year"]=ts.index.year
    out["secondofyear"] = 86400. * (ts.index.dayofyear - 1) + ts.index.hour*3600. + ts.index.minute*60. +    + ts.index.second
    out = out.pivot(index="secondofyear",columns="year",values="value")
    return out

def interannual_sample():
    dr = pd.date_range(start = "2008-02-06",end="2018-11-02",freq="15T")
    df = pd.DataFrame(index = dr, data = np.arange(0,len(dr))/1000.)
    return df

# These labels and ticks are leap-unaware
quarterly_ticks = [(pd.Timestamp(2001,x,1).dayofyear-1)*86400 for x in [1,4,7,10]]
quarterly_labels = ["Jan","Apr","Jul","Oct"]
monthly_ticks = [(pd.Timestamp(2001,x,1).dayofyear-1)*86400 for x in range(1,13)]
monthly_labels = [calendar.month_abbr[i] for i in range(1,13)]
