""" Time series module
Helpers for creating regular and irregular time series, transforming irregular to regular
and analyzing gaps.
"""
import sys
from vtools.data.vtime import *
import datetime as _datetime
import numpy as np

all = ["gap_size","rts","rts_constant","its","extrapolate_ts"]


    
def rts(data,start,interval,props=None):
    """ Create a regular or calendar time series from data and time parameters

        Parameters
        ----------
        data : array_like
            Should be a array/list of values. There is no restriction on data 
             type, but not all functionality like addition or interpolation will work on all data.
        
        start : :py:class:`datetime.datetime`
            Can also be a string representing a datetime.
        
        interval : :ref:`time_interval<time_intervals>`
            Can also be a string representing an interval that is
            understood by :func:`vools.vtime.parse_interval`. 

        Returns
        -------
        result :  :class:`~vtools.data.timeseries.TimeSeries`
            A regular time series.
    """
    
    if type(start)==type(' '):
        start=parse_time(start)
    if type(interval)==type(' '):
        interval=parse_interval(interval)
    timeseq=time_sequence(start,interval,len(data))
    if type(data)==list:
        data=scipy.array(data)
    if (props is None):
        props = {}
    elif not(type(props)==type({})):
        raise TypeError("input props must be a dictionary")
    
    
    ts=TimeSeries(timeseq,data,props)
    ts.interval=interval
    return ts

def its(times,data,props=None):
    """ Create an irregular time series from time and data sequences
        
        Parameters
        ----------
        times : :ref:`time_sequence<time_sequences>`
            An array or list of datetimes or ticks
        
        data : array or list of values
            An array/list of values. No restriction on data type,
            but not all functionality will work on every data type.
        
        Returns
        -------
        result :  :class:`~vtools.data.timeseries.TimeSeries`
            An irregular TimeSeries instance
    """
    # convert times to a tick sequence
    if type(data)==list:
        data=scipy.array(data)
    if (props is None): props = {}        
    ts=TimeSeries(times,data,props)

    return ts


def its2rts(its,interval,original_dates=True):
   """ Convert an irregular time series to a regular.
   
       .. note::
       This function assumes observations were taken at "almost regular" intervals with some 
       variation due to clocks/recording. It nudges the time to "neat" time points to obtain the
       corresponding regular index, allowing gaps. There is no correctness checking, 
       The dates are stored at the original "imperfect" time points if original_dates == True,
       otherwise at the "nudged" regular times.
       
        Parameters
        ----------
        its : :class:`~vtools.data.timeseries.TimeSeries`
             A irregular time series.
             
        
        interval : :ref:`time_interval<time_intervals>`
            Interval of resulting regular timeseries,Can also be a string 
            representing an interval that is understood by :func:`vools.vtime.parse_interval`. 

        original_dates:boolean,optional
            Use original datetime or nudged regular times.
        
        Returns
        -------
        result :  :class:`~vtools.data.timeseries.TimeSeries`
            A regular time series.
       
   """
   import warnings
   if not isinstance(interval, _datetime.timedelta): 
       raise ValueErrror("Only exact regular intervals (secs, mins, hours, days)\
                        accepted in its2rts")
   start = round_ticks(its.ticks[0],interval)
   stime = ticks_to_time(start)
   end = round_ticks(its.ticks[-1],interval)
   interval_seconds = ticks(interval)
   its_ticks = its.ticks   
   n = (end - start)/interval_seconds
   tseq = time_sequence(stime, interval, n+1)   # todo: Changed from n+1 to n+2 a bit casually
   outsize = list(its.data.shape)
   outsize[0] = n+1
   
   data = np.full(tuple(outsize),fill_value=np.nan,dtype=its.data.dtype)
   vround = np.vectorize(round_ticks)
   tround = vround(its.ticks,interval)   
   ndx = np.searchsorted(tseq,tround)
   conflicts = np.equal(ndx[1:],ndx[:-1])
   if any(conflicts):
        badndx = np.extract(conflicts,ndx[1:])
        badtime = tseq[badndx]
        # todo: use warnings.warn()
        for t in badtime[0:10]:
            warnings.warn("Warning multiple time steps map to a single neat output step near: %s " % ticks_to_time(t))
   data[ndx,]=its.data
   if original_dates:
       tseq[ndx]=its.ticks
   newprops = {} if its.props is None else its.props
   ts = TimeSeries(tseq,data,newprops)
   ts.interval = interval
   return ts


def rts_constant(start,end,interval,val=np.nan):
    
    """ Create a regular or calendar time series filled with constant value

        Parameters
        ----------
        start : :py:class:`datetime.datetime`
            Starting time, can also be a string representing a datetime.
        
        end : :py:class:`datetime.datetime`
            Ending time,can also be a string representing a datetime.
        
        interval : :ref:`time_interval<time_intervals>`
            Can also be a string representing an interval that is 
            understood by :func:`vools.vtime.parse_interval`. 
            
        val : float,int
            Constant will be filled into the time series.
            Optional, default is nan.
            
        Returns
        -------
        result :  :class:`~vtools.data.timeseries.TimeSeries`
            A regular time series wiht constant values
    """
    
    
    num_data=number_intervals(start,end,interval)+1
    data=np.empty(num_data)
    data.fill(val)
    ts=rts(data,start,interval,{})
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
        result :  :class:`~vtools.data.timeseries.TimeSeries`
           An new time series extended to the input start and end.
           This function will only extend, not clip, so if you want to be
           sure the returned function is the correct siae you need to apply
           a window afterwards.
           
           
    """
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
        
    
def gap_size(ts):
    """ Assess the size of local gaps
    Identifies gaps (runs of missing data) and quantifies the
    length of the gap. Each time point receives the length of the run
    in terms of seconds or number of values in the time dimension,
    with non-missing data returning zero.     
    
    Parameters
    -----------
    
    ts : :class:`DataFrame <pandas:pandas.DataFrame>`
  
    Returns
    -------
    result : :class:`DataFrame <pandas:pandas.DataFrame>`
        A new regular time series with the same freq as the argument
        holding the size of the gap. 
        
        
    """
    

    s = ts.index.to_series()
    for c in ts.columns:
        #test missing values
        miss = ts[c].isna()
        #create consecutive groups
        g = miss.ne(miss.shift()).cumsum()
        #aggregate minimal 
        m1 = s.groupby(g).min()
        #get minimal of next groups, last value is replaced last value of index
        m2 = m1.shift(-1).fillna(ts.index[-1])
        #get difference, convert to minutes
        out = m2.sub(m1).dt.total_seconds().div(60).astype(int)
        #map to column
        ts[c] = g.map(out)
        ts.loc[~miss,c] = 0.       
    return ts
    
def datetime_elapsed(index_or_ts,reftime=None,dtype="d",inplace=False):
    """Convert a time series or DatetimeIndex to an integer/double series of elapsed time

    Parameters
    -----------
    
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

def elapsed_datetime(index_or_ts, time_unit='s', reftime=None,inplace=False):
    """Convert a time series or numerical Index to a Datetime index or series

    Parameters
    -----------
    
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
    print("inside",index_or_ts)
    try:
        ndx = index_or_ts.index
        input_index = False
    except AttributeError as e:
        ndx = index_or_ts
        input_index = True
    
    if isinstance(ndx,pd.TimedeltaIndex):
        dtndx = reftime + ndx
    else: 
        dtndx = reftime + pd.TimedeltaIndex(data=ndx,unit=time_unit)

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

def example_gap():
    ndx = pd.date_range(pd.Timestamp(2017,1,1,12),freq='15min',periods=10)
    vals0 = np.arange(0.,10.,dtype='d')
    vals1 = vals0.copy()
    vals2 = vals0.copy()
    vals0[0:3] = np.nan
    vals0[7:-1] = np.nan
    vals1[2:4] = np.nan
    vals1[6] = np.nan

    df = pd.DataFrame({'vals0':vals0,'vals1':vals1,'vals2':vals2},index = ndx)
    print(df)
    print(gap_size(df))

    
if __name__=="__main__":
    example_gap()    
    
    
