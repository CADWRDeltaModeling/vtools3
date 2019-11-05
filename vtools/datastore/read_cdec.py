from sklearn.linear_model import LinearRegression
import numpy as np

import pandas as pd
import statsmodels.api as sm
import statsmodels as sm
import matplotlib.pyplot as plt
import matplotlib

from scipy.signal import boxcar
from read_scalar import *
from vtools3.functions.filter import cosine_lanczos
from vtools3.data.vtime import hours,minutes,days


from unit_conversions import ec_psu_25c


# 1970.2058633  3798.60021635 5561.45924409 7279.506588
print psu_ec_25c(np.arange(1,5))
print ec_psu_25c(2640)



def period_op(ts,period = "D",agg="mean",max_absent_frac=0.):
    
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

def smooth(x,window_len=99,window='flat'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=numpy.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='same')
    return y


def read_des(fpat):
    start = dtm.datetime(2008,1,1)
    end = dtm.datetime(2019,5,1)
    mdir = "//cnrastore-bdo/Modeling_Data/des_emp/raw"
    
    ts=csv_retrieve_ts(fpat,mdir,start,end,selector="VALUE",qaqc_selector="QAQC Flag",
                    parsedates=["DATETIME"],
                    indexcol=["DATETIME"],
                    skiprows=2,
                    dateparser=None,
                    comment = None,
                    extra_na=[""],
                    prefer_age="new",                    
                    tz_adj=hours(0))  
    return ts




def read_cdec(fpat):
    start = dtm.datetime(2008,1,1)
    end = dtm.datetime(2019,5,1)    
    fdir="."
    
    ts = csv_retrieve_ts(fpat,fdir,start,end,selector="VALUE",
                    qaqc_selector="DATA_FLAG",
                    qaqc_accept=['',' ',u' ',u'e'],
                    parsedates=["DATE TIME"],
                    indexcol="DATE TIME",
                    skiprows=0,
                    dateparser=None,
                    comment = None,
                    prefer_age="new",                    
                    tz_adj=hours(0))
    return ts
    
    
