#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Time series module
Helpers for creating regular and irregular time series, transforming irregular to regular
and analyzing gaps.
"""
import sys
import numpy as np
import pandas as pd
from vtools.data.vtime import *
import datetime as _datetime


__all__ = ["to_dataframe","time_overlap","rts","rts_formula","extrapolate_ts","datetime_elapsed","elapsed_datetime","to_dataframe"]

def to_dataframe(ts):
    if isinstance(ts,pd.DataFrame):
        return ts
    else:
        return ts.to_frame()
      

def time_overlap(ts0,ts1,valid=True):
    """Check for overlapping time coverage between series
       Returns a tuple of start and end of overlapping periods. Only considers
       the time stamps of the start/end, possibly ignoring NaNs at the beginning
       if valid=True, does not check for actual time stamp alignment
    """
    if valid:
        start = max(ts0.first_valid_index(),ts1.first_valid_index())
        end = min(ts0.last_valid_index(),ts1.last_valid_index())

    else:
        start = max(ts0.index[0],ts1.index[0])
        end = min(ts0.index[-1], ts1.index[-1])
    return (start,end)  if end > start else None
    
def rts(data,start,freq,columns=None,props=None):
    """ Create a regular or calendar time series from data and time parameters

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

    if type(data)==list:
        data=np.array(data)
    if (not props is None):
        raise NotImplementedError("Props reserved for future implementation using xarray")
    tslen = data.shape[0]
    ndx = pd.date_range(start,freq=freq,periods=tslen)
    ts = pd.DataFrame(data,index=ndx,columns=columns)
    return ts


def rts_formula(start,end,freq,valfunc=np.nan):    
    """ Create a regular time series filled with constant value or formula based on elapsed seconds

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
    ndx = pd.date_range(start=start,end=end,freq=freq)
    secs = (ndx - ndx[0]).total_seconds()

    if isinstance(valfunc,numtype):
        data = np.array([valfunc for x in secs])
        cols = ["value"]
    else:    
        data = np.array([valfunc[x](secs) for x in valfunc]).T
        cols = valfunc.keys()
    ts=rts(data,start,freq,columns=cols)

    return ts


def extrapolate_ts(ts,start=None,end=None,method="constant",val=np.nan):
    
    """ Extend a regular time series with newer start/end 

        Parameters
        ----------
        start : :py:class:`datetime.datetime`
            The starting time to be extended to.Optional, default no extension.
        
        end : :py:class:`datetime.datetime`
           The ending time to be extended to.Optional,default no extension.
        
        method : :string
            Method to fill the extended part of resulting series. either
            "taper" or"constant". default constant.
            
        val : float
            Constant will be filled or the value that the last non-nan
            gets tapered to. Only constant is supported for multivariate data
            
        Returns
        -------
        result :class:`Pandas.DataFrame`
           An new time series extended to the input start and end.
           This function will only extend, not clip, so if you want to be
           sure the returned function is the correct size you need to apply
           a window afterwards.
    """

    raise NotImplementedError("Placeholder for vtools3")


    if start is None:
        start=ts.start
    if end is None:
        end=ts.end

    if start < ts.start:
        head_extended = number_intervals(start,ts.start,ts.interval)
        eff_start = start
    else:
        head_extended = 0
        eff_start = ts.start
    if ts.end < end:
        tail_extended = number_intervals(ts.end,end,ts.interval)
    else:
        tail_extended = 0
    new_len=len(ts)+head_extended+tail_extended
    shp = list(ts.data.shape)
    shp[0] = new_len
    data=np.zeros(tuple(shp))
    data[:] = np.nan
    old_len=len(ts.times)
    if len(shp) == 1:
        data[head_extended:head_extended+old_len]=ts.data[:]
    else:
        data[head_extended:head_extended+old_len,]=ts.data
        

    if method=="constant":
        data[0:head_extended,]=val
        data[head_extended+old_len:new_len,]=val
    elif method=="taper":
        if np.any(np.isnan(val)):
            raise ValueError("You must input a valid value for taper method")        
        if data.ndim > 1:
            raise ValueError("Extrapolate with taper not implemented for ndim > 1")
        ## find out first and last non val
        temp=ts.data[~np.isnan(ts.data)]
        begin_taper_to=temp[0]
        end_taper_from=temp[-1]
        
        head_taper_step=(begin_taper_to-val)/(head_extended)
        tail_taper_step=(val-end_taper_from)/(tail_extended)
        data[0:head_extended]=np.arange(start=val,\
                                stop=begin_taper_to,
                                step=head_taper_step)
        data[old_len+head_extended:new_len]=np.arange(start=end_taper_from+tail_taper_step,\
                                stop=val+tail_taper_step,step=tail_taper_step)
        
    else:
        raise ValueError("Unkonw filling method:"+method)
        
    new_ts=rts(data,eff_start,ts.interval,{})
    return new_ts
        
    
    
def datetime_elapsed(index_or_ts,reftime=None,dtype="d",inplace=False):
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

def elapsed_datetime(index_or_ts,reftime=None,time_unit='s',inplace=False):
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
    
    if isinstance(ndx,pd.TimedeltaIndex):
        dtndx = reftime + ndx
    else: 
        if time_unit.lower()=="h": ndx=ndx*3600.
        elif time_unit.lower()=="d": ndx=ndx*86400.
        elif time_unit.lower()=="s": pass
        else: raise ValueError("time unit must be 's','h',or 'd'")
        dtndx = reftime + pd.to_timedelta(arg=ndx, unit="s")

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

def example():
    ndx = pd.date_range(pd.Timestamp(2017,1,1,12),freq='15min',periods=10)
    out = datetime_elapsed(ndx,dtype='i')
    print(out)
    print(type(out))
    vals = np.arange(0.,10.,dtype='d')
    df = pd.DataFrame({'vals':vals},index = ndx.copy())
    ref = pd.Timestamp(2017,1,1,11,59)
    df2 = datetime_elapsed(df,reftime=ref,dtype=int)    
    print(elapsed_datetime(df2,reftime=ref) - df)


    
if __name__=="__main__":
    example()    
    
    
