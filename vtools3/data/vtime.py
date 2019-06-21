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




    
    
    
    
    


            
    
    

    

