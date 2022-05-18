#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from vtools.datastore.read_ts import *

import pandas as pd
import numpy as np



def gap_count(ts,state="gap",dtype=int):        
    """ Count missing data
    Identifies gaps (runs of missing or non-missing data) and quantifies the
    length of the gap in terms of number of samples, which works better for
    regular series. Each time point receives the length of the run. 
    
    Parameters
    ----------
    
    ts : :class:`DataFrame <pandas:pandas.DataFrame>`
        Time series to analyze
        
    state : `str` one of 'gap'|'good'|'both'
        State to count. If state is gap, block size of missing data are counted 
        and reported for time points in the gap (every point in a given gap will
        receive the same value). Non missing data will have a size of zero. 
        Setting state to 'good' inverts this -- missing blocks are reported as
        zero and good data are counted. 
        
    dtype : `str` or `type`
        Data type of output, should be acceptable to
        pandas :meth:`astype <pandas:pandas.DataFrame.astype>`
    
    """   
    def column_gap_count(ser):
        s = ser.index.to_series()
        tsout = ser.fillna(0).astype(dtype)
        miss = ser.isna()
        #create consecutive groups that increment each time the "is missing state" (na or not na) changes
        g = miss.ne(miss.shift()).cumsum()

        # identify beginning (min time) of each state
        count = s.groupby(g).count()
                
        # g contains a group index for each member of out, and here
        # we map g to out which has cumulative time
        tsout = g.map(count)
        if state == "gap":
            tsout.loc[~miss] = 0
        elif state == "good":
            tsout.loc[miss] = 0
        return tsout
    
    if hasattr(ts,"columns"):
        return ts.apply(column_gap_count,axis=0,result_type='broadcast').astype(dtype)
    else:
        return column_gap_count(ts).astype(dtype)




def gap_size(ts):
    """
    Identifies gaps (runs of missing data) and quantifies the
    length of the gap. Each time point receives the length of the run
    in terms of seconds or number of values in the time dimension,
    with non-missing data returning zero. Time is measured from the time the
    data first started being missing to when the data first starts being not missing
    .
    
    Parameters
    -----------
    
    ts : :class:`DataFrame <pandas:pandas.DataFrame>`
  
    Returns
    -------
    result : :class:`DataFrame <pandas:pandas.DataFrame>`
        A new regular time series with the same freq as the argument
        holding the size of the gap. 
        
    Examples
    --------
    >>> ndx = pd.date_range(pd.Timestamp(2017,1,1,12),freq='15min',periods=10)
    >>> vals0 = np.arange(0.,10.,dtype='d')
    >>> vals1 = np.arange(0.,10.,dtype='d')
    >>> vals2 =  np.arange(0.,10.,dtype='d')
    >>> vals0[0:3] = np.nan
    >>> vals0[7:-1] = np.nan
    >>> vals1[2:4] = np.nan>>> 
    >>> vals1[6] = np.nan
    >>> vals1[9] = np.nan

    >>> df = pd.DataFrame({'vals0':vals0,'vals1':vals1,'vals2':vals2},index = ndx)
    >>> out = gap_size(df)
    >>> print(df)
                             vals0  vals1  vals2
    2017-01-01 12:00:00    NaN    0.0    0.0
    2017-01-01 12:15:00    NaN    1.0    1.0
    2017-01-01 12:30:00    NaN    NaN    2.0
    2017-01-01 12:45:00    3.0    NaN    3.0
    2017-01-01 13:00:00    4.0    4.0    4.0
    2017-01-01 13:15:00    5.0    5.0    5.0
    2017-01-01 13:30:00    6.0    NaN    6.0
    2017-01-01 13:45:00    NaN    7.0    7.0
    2017-01-01 14:00:00    NaN    8.0    8.0
    2017-01-01 14:15:00    9.0    NaN    9.0
    >>> print(out)    
                             vals0  vals1  vals2
    2017-01-01 12:00:00   45.0    0.0    0.0
    2017-01-01 12:15:00   45.0    0.0    0.0
    2017-01-01 12:30:00   45.0   30.0    0.0
    2017-01-01 12:45:00    0.0   30.0    0.0
    2017-01-01 13:00:00    0.0    0.0    0.0
    2017-01-01 13:15:00    0.0    0.0    0.0
    2017-01-01 13:30:00    0.0   15.0    0.0
    2017-01-01 13:45:00   30.0    0.0    0.0
    2017-01-01 14:00:00   30.0    0.0    0.0
    2017-01-01 14:15:00    0.0    0.0    0.0    
        
    """
    
    ts_out = ts*0.
    
    s = ts.index.to_series()
    for c in ts.columns:
        #test missing values
        miss = ts[c].isna()
        #create consecutive groups that increment each time the "is missing state" (na or not na) changes
        g = miss.ne(miss.shift()).cumsum()
        # identify beginning (min time) of each state
        m1 = s.groupby(g).min()
        
        #get beginning of next groups, last value is replaced last value of index
        m2 = m1.shift(-1).fillna(ts.index[-1])

        #get difference, convert to minutes
        diffs = m2.sub(m1).dt.total_seconds().div(60).astype(int)
        
        # g contains a group index for each member of out, and here
        # we map g to out which has cumulative time
        ts_out[c] = g.map(diffs)
        ts_out.loc[~miss,c] = 0.       
    return ts_out



def gap_distance(ts, disttype="count", to = "good"):
    
    """
    For each element of ts, count the distance to the nearest good data/or bad data.
      
    Parameters
    -----------
    
    ts : :class:`DataFrame <pandas:pandas.DataFrame>`
    
    disttype : `str` one of 'bad'|'good'
    If disttype = "count" this is the number of values. If dist_type="freq" it is in the units of ts.freq
    (so if freq == "15T" it is in minutes")
    
    to : `str` one of 'bad'|'good'
    
    If to = "good" this is the distance to the nearest good data (which is 0 for good data).
    If to = "bad", this is the distance to the nearest nan (which is 0 for nan). 

    Returns
    -------
    result : :class:`DataFrame <pandas:pandas.DataFrame>`
        A new regular time series with the same freq as the argument
        holding the distance of good/bad data. 
        
    """
    si = ts.index.to_series()
    ts_out = ts.to_frame() if isinstance(ts,pd.Series) else ts.copy()
    cols = ts_out.columns
    for col in cols:
        id_key=True
        #test missing values
        miss = ts_out[col].isna()
        if to=="good":
            ts_out.at[~miss,col]=0
        elif to=="bad":
            ts_out.at[miss,col]=0
            id_key=False
        else:
            raise ValueError("invalid input to, must be good or bad")

      
        if np.any(miss==(id_key)):
            mm=si.groupby(miss).indices
            for i in mm[id_key]:
                #ts_out.iloc[i][col]=np.min(np.abs(i-mm[not(id_key)]))
                ts_out.at[si[i],col]=np.min(np.abs(i-mm[not(id_key)]))
                
                
    if disttype=="count":
        return ts_out
    elif disttype=="freq":
        return ts_out*ts.index.freq
    else:
        raise ValueError("invalid input disttype, must be count or freq")



def example_gap():
    import numpy as np
    ndx = pd.date_range(pd.Timestamp(2017,1,1,12),freq='15min',periods=10)
    vals0 = np.arange(0.,10.,dtype='d')
    vals1 = vals0.copy()
    vals2 = vals0.copy()
    vals0[0:3] = np.nan
    vals0[7:-1] = np.nan
    vals1[2:4] = np.nan
    vals1[6] = np.nan
    vals1[9] = np.nan

    df = pd.DataFrame({'vals0':vals0,'vals1':vals1,'vals2':vals2},index = ndx)
    out = gap_count(df)
    print(df)
    print(out)
    
    out = gap_distance(df)
    print("**")
    print(out)
    
if __name__=="__main__":
    example_gap()    
    