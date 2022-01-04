from sklearn.linear_model import LinearRegression
import numpy as np

import pandas as pd
import statsmodels.api as sm
import statsmodels as sm
import matplotlib.pyplot as plt
import matplotlib

from scipy.signal import boxcar
from vtools.data.vtime import hours,minutes,days,months

def period_op(ts,period="D",agg="mean",max_absent_frac=0.):
    period = pd.tseries.frequencies.to_offset(period)
    # resample getting a sum and non-null count
    ts1 = ts.resample(period).agg([agg, 'count']).copy()

    # determine required number data which is exact for period of "D" or smaller
    agged = ts1.loc[:,(slice(None),agg)].droplevel(level=1,axis=1)
    if max_absent_frac is None: 
        return agged
    counts = ts1.loc[:,(slice(None),'count')].droplevel(level=1,axis=1)
    try:
        na_tol = (period/ts.index.freq)*(1. - max_absent_frac)
        invalid = counts < na_tol
    except:
        if period == months(1):
            # determine invalid months
            na_tol = max_absent_frac*ts.index.days_in_month*(days(1)/ts.index.freq)
            invalid = counts < na_tol  
        else:
            raise ValueError("Offset {} not supported. Only absolute offsets (<= 1day) or one month averages supported with max_absent_frac option".format(period))
    agged.mask(invalid,inplace=True)
    return agged


def window_op(ts,window,period = "D",agg="mean",max_absent_frac=0.):
    
    period = pd.tseries.frequencies.to_offset(period)

    # resample getting a sum and non-null count
    ts1 = ts.resample(period).agg([agg, 'count'])
    
    # determine required number data which is exact for period of "D" or smaller
    if max_absent_frac is not None:
        try:
            na_tol = (ts1.index.freq/ts.index.freq)*(1. - max_absent_frac)
            invalid = ts1['count'] <  na_tol
        except:
            if period == months(1):
                # determine invalid months
                invalid = ts1['count'] < max_absent_frac * ts.index.days_in_month
            else:
                raise ValueError("Offset {} not supported. Only absolute offsets (<= 1day) or one month averages supported with max_absent_frac option".format(period))
    else:
        invalid = ts1['count'] < 0   # None are invalid

    ts1 = ts1[agg]
    ts1[invalid] = np.nan
    return ts1

import numpy