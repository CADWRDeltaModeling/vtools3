""" Basic ops for creating, testing and manipulating times and time intervals.
    This module contains factory and helper functions for working with
    times and time intervals.

    The module will import the name datetime and is
    designed to work exclusively with python datetimes. In addition, datetimes
    are convertible to long integer timestamps called ticks. The resolution
    of 1 tick in ticks per second may be obtained using the resolution()
    function or certain utility constants such as ticks_per_day. Never
    code with hard wired numbers.

    For time intervals (or deltas), VTools requires a time and time
    interval system that is consistent (e.g. time+n*interval makes sense)
    and that can be applied to both calendar dependent
    and calendar-independent intervals. Because this requirement
    is not met by any one implementation it is recommended that
    you always use the factory functions in this module
    for creating intervals or testing whether an interval is valid.
"""

import numpy as np
import pandas as pd






def seconds(s):
    """ Create a time interval representing s seconds"""
    return pd.tseries.offsets.Second(s)

def minutes(m):
    """ Create a time interval representing m minutes"""
    return pd.tseries.offsets.Minute(m)

def hours(h):
    """ Create a time interval representing h hours"""
    return pd.tseries.offsets.Hour(h)

def days(d):
    """ Create a time interval representing d days"""    
    return pd.tseries.offsets.Day(d)

def months(m):
    """ Create a time interval representing m months"""    
    return pd.tseries.offsets.MonthOffset(m)

def years(y):
    """ Create a time interval representing y years"""    
    return pd.tseries.offsets.YearOffset(y)





def dst_to_standard_naive(ts, dst_zone = "US/Pacific", standard_zone="Etc/GMT+8"):
    """ Convert timezone-unaware series from a local (with daylight) time to standard time
        This would be useful, say, for converting a series that is PDT during summer to one that is not.
        The routine is mainly to treat cases where the time stamps at DST interfaces are not redundant -- if they 
        are you can probably use tz_convert and tz_localize with the ambiguous = 'infer' option and do the job more 
        efficiently, but lots of databases don't store data this way.
        
        The choice of the standard_zone is, it seems, buggy. The defaults are supposed to convert from PST/PDT to pure PST,
        and the latter should be GMT-8. In a sense, this function is included before the behavior is really understood.
        
        Only regular series are accepted ... this is a quirk of the implementation
        """


    # Create a date range that spans the possible times and is same refinement
    try:
        dt = ts.freq 
    except:
        dt = pd.offsets.Minute(15)
    hr = pd.offsets.Hour(1)
    ndx2 = pd.date_range(start=ts.index[0] - hr, end=ts.index[-1] + hr,freq=dt,tz=dst_zone)

    # Here is the determination of whether it is dst
    isdst = [bool(x.dst()) for x in ndx2.to_pydatetime()]

    # Use DataFrame indexing to perform the lookup for values in my original index
    

    df2 = pd.DataFrame({"isdst":isdst},index=ndx2.tz_localize(None))    
    df2 = df2.loc[~df2.index.duplicated(keep="last"),:]

    print(ts.index)
    print(df2)
    ambig = df2.loc[ts.index,"isdst"].values

    # Here is the real work        
    ts2 = ts.tz_localize("US/Pacific",ambiguous=ambig).tz_convert(standard_zone).tz_localize(None) 
    return ts2    
