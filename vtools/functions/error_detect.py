#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
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


'''
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
'''

def _nrepeat(ts):
    """ Series-only version"""
    mask = ts.ne(ts.shift())
    counts = ts.groupby(mask.cumsum()).transform('count')
    return counts


def nrepeat(ts):
    """ Return the length of consecutive runs of repeated values
    
    Parameters
    ----------
   
    ts:  DataFrame or series 

    Returns
    -------
    Like-indexed series with lengths of runs. Nans will be mapped to 0    
    
    """
    if isinstance(ts,pd.Series): return _nrepeat(ts)
    return ts.apply(_nrepeat,axis=0)

def threshold(ts,bounds,copy=True):
    ts_out = ts.copy() if copy else ts
    warnings.filterwarnings("ignore")
    if bounds is not None:
        if bounds[0] is not None:
            ts_out.mask(ts_out < bounds[0],inplace=True)
        if bounds[1] is not None:
            ts_out.mask(ts_out > bounds[1],inplace=True)  
    return ts_out

def bounds_test(ts,bounds):
    anomaly = pd.DataFrame(dtype=bool).reindex_like(ts)
    anomaly[:] = False
    if bounds is not None:
        if bounds[0] is not None:
            anomaly |= ts < bounds[0]
        if bounds[1] is not None:
            anomaly |= ts > bounds[1]           
    return anomaly


def median_test(ts, level = 4, filt_len = 7, quantiles=(0.005,0.095),copy = True):
    return med_outliers(ts,level=level,filt_len=filt_len, quantiles=quantiles,copy=False,as_anomaly=True)


def median_test_oneside(ts, scale=None,level = 4, filt_len = 6, quantiles=(0.005,0.095),
                        copy = True,reverse=False):    
    if copy: 
        ts=ts.copy()
    kappa = filt_len//2
    if reverse:
        original_index = ts.index
        vals = ts[::-1] 
    else: 
        vals = ts
    vals = to_dataframe(vals)
    vals.columns=["ts"]

    vals["z"]=vals.ts.diff()
    min_periods = kappa*2 - 1
    
    dds = dd.from_pandas(vals,npartitions=50)
    dds['my'] = dds['ts'].shift().rolling(kappa*2,min_periods=min_periods).median()
    dds['mz'] = dds.z.shift().rolling(kappa*2,min_periods=min_periods).median()
    dds['pred'] = dds.my + kappa*dds.mz    
    res = (dds.ts - dds.pred).compute()
    if scale is None:
        qq = res.quantile( q=quantiles)
        scale = qq.loc[quantiles[1]] - qq.loc[quantiles[0]]
    anomaly = (res.abs() > level*scale) | (res.abs() < -level*scale)    
    
    if reverse: 
        anomaly = anomaly[::-1]
        anomaly.index = original_index

    return anomaly



def med_outliers(ts,level=4.,scale = None,\
                 filt_len=7,range=(None,None),
                 quantiles = (0.01,0.99),
                 copy = True,as_anomaly=False):
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
        threshold(ts_out,range,copy=False)

    vals = ts_out.to_numpy()
    #if ts_out.ndim == 1:
    #    filt = medfilt(vals,filt_len)
    #else:
    #    filt = np.apply_along_axis(medfilt,0,vals,filt_len)
    filt = ts_out.rolling(filt_len,center=True,axis=0).median()

    res = ts_out - filt


    if scale is None:
        qq = res.quantile( q=quantiles)
        scale = qq.loc[quantiles[1]] - qq.loc[quantiles[0]]

    anomaly = (res.abs() > level*scale) | (res.abs() < -level*scale)
    if as_anomaly: 
        return anomaly   
    # apply anomaly by setting values to nan
    values = np.where(anomaly,np.nan,ts_out.values)
    ts_out.iloc[:]= values
    warnings.resetwarnings()
    return ts_out



def median_test_twoside(ts,level=4.,scale = None,\
                 filt_len=7,
                 quantiles = (0.01,0.99),
                 copy = True,as_anomaly=True):
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
    #if ts_out.ndim == 1:
    #    filt = medfilt(vals,filt_len)
    #else:
    #    filt = np.apply_along_axis(medfilt,0,vals,filt_len)
    
    def mseq(flen):
        halflen = flen//2
        a = np.arange(0,halflen)
        b = np.arange(halflen+1,flen)
        return np.concatenate((a,b))
    medseq = mseq(filt_len)
    
    dds = dd.from_pandas(ts_out,npartitions=50)
    filt = dds.rolling(filt_len,center=True,axis=0).apply(lambda x: np.nanmedian(x[medseq]),raw=True,engine='numba').compute()
    res = (ts_out - filt)

    if scale is None:
        qq = res.quantile( q=quantiles)
        scale = qq.loc[quantiles[1]] - qq.loc[quantiles[0]]

    anomaly = ((res.abs() > level*scale) | (res.abs() < -level*scale))
    if as_anomaly: 
        return anomaly   
    # apply anomaly by setting values to nan
    values = np.where(anomaly,np.nan,ts_out.values)
    ts_out.iloc[:]= values
    warnings.resetwarnings()
    return ts_out


def gapdist_test_series(ts,smallgaplen=0):
    test_gap = ts.copy()
    gapcount = gap_count(ts)
    testgapnull = test_gap.isnull()
    is_small_gap = (gapcount <= smallgaplen)
    smallgap = testgapnull & is_small_gap
    test_gap.where(~smallgap,-99999999.,inplace=True)
    return test_gap

def steep_then_nan(ts,level=4.,scale = None,\
                 filt_len=11,range=(None,None),
                 quantiles = (0.01,0.99),
                 copy = True,as_anomaly = True):
    """
    Detect outliers by running a median filter, subtracting it
    from the original series and comparing the resulting residuals
    to a global robust range of scale (the interquartile range).
    Individual time points are rejected if the residual at that time point is more than level times the range of scale. 

    The original concept comes from Basu & Meckesheimer (2007)
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
    
    test_gap = gapdist_test_series(ts,smallgaplen=3)
    gapdist = gap_distance(test_gap, disttype="count", to = "bad").squeeze()
    neargapdist = 5
    nearbiggap = gapdist.squeeze() < neargapdist  
    vals = ts_out.to_numpy()
    if ts_out.ndim == 1:
        filt = medfilt(vals,filt_len)
    else:
        filt = np.apply_along_axis(medfilt,0,vals,filt_len)
        
    res = ts_out - filt


    if scale is None:
        qq = res.quantile( q=quantiles)
        scale = qq.loc[quantiles[1]] - qq.loc[quantiles[0]]

    outlier = (np.absolute(res) > level*scale) | (np.absolute(res) < -level*scale)
    diag_plot = False
    if diag_plot:
        fig,(ax0,ax1) = plt.subplots(2,sharex=True)

        print(outlier)
        outliernum = (ts_out*0.).fillna(0.)
        outliernum[outlier] = 1.
        nearbiggapnum = (ts_out*0.).fillna(0.)
        print(nearbiggap)
        nearbiggapnum[nearbiggap] = 1.
        outliernum.plot(ax=ax0)
        nearbiggapnum.plot(ax=ax1)
        gapdist.plot(ax=ax1)
        plt.show()
  
    outlier = outlier.squeeze() & nearbiggap.squeeze()  
    print("Any outliers?")
    print(outlier.any())
    print(ts_out[outlier])
    print("Near big gap")
    print(nearbiggap[nearbiggap.values])
    print("OK")
    
    if not as_anomaly:
        values = np.where(outlier,np.nan,ts_out.values)
        ts_out.iloc[:]= values
    warnings.resetwarnings()
    return outlier if as_anomaly else ts_out






def despike(arr, n1=2, n2=20, block=10):
    offset = arr.min()
    arr -= offset
    data = arr.copy()
    roll = rolling_window(data, block)
    roll = ma.masked_invalid(roll)
    std = n1 * roll.std(axis=1)
    mean = roll.mean(axis=1)
    # Use the last value to fill-up.
    std = np.r_[std, np.tile(std[-1], block - 1)]
    mean = np.r_[mean, np.tile(mean[-1], block - 1)]
    mask = (np.abs(data - mean.filled(fill_value=np.NaN)) >
            std.filled(fill_value=np.NaN))
    data[mask] = np.NaN
    # Pass two: recompute the mean and std without the flagged values from pass
    # one now removing the flagged data.
    roll = rolling_window(data, block)
    roll = ma.masked_invalid(roll)
    std = n2 * roll.std(axis=1)
    mean = roll.mean(axis=1)
    # Use the last value to fill-up.
    std = np.r_[std, np.tile(std[-1], block - 1)]
    mean = np.r_[mean, np.tile(mean[-1], block - 1)]
    mask = (np.abs(arr - mean.filled(fill_value=np.NaN)) >
            std.filled(fill_value=np.NaN))
    arr[mask] = np.NaN
    return arr + offset

def example():
    station = sys.argv[1]
    ts = read_cdec("cdec_download/%s.csv"%station,start=None,end=None)

    filt = medfilt(ts.data, kernel_size=5)
    ts,filt = med_outliers(ts,quantiles=[0.2,0.8],range=[120.,None],copy=False)

    plt.plot(ts.times,ts.data,label="data")
    plt.plot(ts.times,filt.data,label="filt")
    plt.plot(ts.times,ts.data-filt.data,label="res")
    plt.legend()
    plt.show()

def example2():
    data = np.arange(32)*2. + np.cos(np.arange(32)*2*np.pi/24.)
    ndx = pd.date_range(pd.Timestamp(2000,1,1),periods = len(data),freq="15min")
    df = pd.DataFrame(data=data,index=ndx)
    median_test_oneside(df,quantiles=(0.25,0.75),level=2)

if __name__ == '__main__':
    example2()
    # Just an example
