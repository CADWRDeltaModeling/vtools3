#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

def ts_merge(series):
    """ merge a number of timeseries together and return a new ts.
    Similar to concatenate, but provides more control over order in cases of overlap

    Parameters
    ----------
    series  :  tuple(:class:`DataFrame <pandas:pandas.DataFrame>`) or tuple(:class:`DataArray) <xarray:xarray.DataArray>`
        Series ranked from hight to low priority              
    Returns
    -------    
    merged : :class:`DataFrame <pandas:pandas.DataFrame>`
        A new time series with time interval same as the inputs, time extent
        the union of the inputs, and filled first with ts1, then with remaining
        gaps filled with ts2, then ts3....

    """

    # this concatenates, leaving redundant indices in df0, df1, df2
    # we are not doing the real work yet, just getting the right indexes
    # this doesn't seem super efficient, but good enough for a start     
    dfmerge = pd.concat(series, sort=True)
    
    # This populates with the values from the highest series 
    # It is a bug for the first series to have a duplicate index
    series0 =series[0][~series[0].index.duplicated(keep='first')]
    dfmerge = series0.reindex(dfmerge.index)  

    # drop duplicate indices
    dfmerge = dfmerge.loc[~dfmerge.index.duplicated(keep='last')]
    # Now apply, in priority order, each of the original dataframes to fill the original
    for ddf in series[1:]:
        dfmerge = dfmerge.combine_first(ddf)
    return dfmerge
    
def ts_splice(tss,transition="prefer_last",floor_dates=False):
    """ splice a number of timeseries together in a non-overlapping way. 
    The function supports three methods of positioning the breakpoints
    between series.

    Parameters
    ----------
    series  :  tuple(:class:`DataFrame <pandas:pandas.DataFrame>`)
        Series ranked from first to last in time. Must have identical column structure
        or unexpected results may occur during concatenate.

    transition : 'prefer_first' | 'prefer_last' | list(pd.Datetime)
        Description of how to switch between series. If prefer_first, the
        earlier series will be used until their end point.  
        If prefer_last, the first time stamp of the 
        later series will be used. 

    floor_dates: bool
        Floor the transition dates that are inferred with 'prefer_first' or 'prefer_last'.
        Note that this can produce nans if the input series are regular with a freq attribute
        or a big gap otherwise unless there is overlapping data covering the days that are rounded.
        
    Returns
    -------    
    spliced : :class:`DataFrame <pandas:pandas.DataFrame>`
        A new time series with freq same as the inputs if they are all the same,
        time extent the union of the inputs.
    """
    if isinstance(transition,list):
        if floor_dates:
            raise ValueError("Floor dates option not compatible with transition that is a list of Timestamps")
        dup_keep='first'
    else:
        if transition == 'prefer_first':
            dup_keep = 'first'
            transition = [ts.last_valid_index() for ts in tss[:-1]]
            if floor_dates:
                transition = [dt.floor('D') for dt in transition]
        if transition == 'prefer_last':
            dup_keep = 'last'
            transition = [ts.first_valid_index() for ts in tss[1:]]
            if floor_dates:
                transition = [dt.floor('D') for dt in transition]                
    transition = [None] + transition + [None]
    print("transition\n",transition)
    sections = []
    f=tss[0].index.freq
    for ts,start,end in zip(tss,transition[:-1],transition[1:]): 
        print("here",start,end,"ts\n",ts)
        f = f if (f is not None) and ts.index.freq == f else None      
        sections.append(ts.loc[slice(start,end),:])
    tsout = pd.concat(sections,axis=0)
    duplicatetime = tsout.index.duplicated(keep=dup_keep)
    nduplicatetime = duplicatetime.sum()
    if nduplicatetime >= 1:
        tsout = tsout[~duplicatetime]
    if f is not None: tsout = tsout.asfreq(f)
    return tsout

def main():
    import pandas as pd
    import numpy as np
    from vtools.data.vtime import hours,days

    dfs = []    
    for i in range(4):
        dr = pd.date_range(pd.Timestamp(2000,1,1)+hours(i*8),freq="H",periods=9)
        data = np.arange(1.,10.) + float(i*100.) 
        df = pd.DataFrame(index=dr,data=data)
        dfs.append(df)
        print(df)
    ts_long = ts_splice(dfs,transition='prefer_first',floor_dates=False)
    print(ts_long)
    print("**")
    dfs = []    
    for i in range(4):
        dr = pd.date_range(pd.Timestamp(2000,1,1)+hours(i*8),freq="H",periods=11)
        data = np.arange(1.,12.) + float(i*100.) 
        df = pd.DataFrame(index=dr,data=data)
        dfs.append(df)
        print(df)
    ts_long = ts_splice(dfs,transition='prefer_last',floor_dates=False)
    print(ts_long)
    
    transitions = [pd.Timestamp(2000,1,1,9,1),pd.Timestamp(2000,1,1,17,1),pd.Timestamp(2000,1,1,22,1)]
    print("***************")
    dfs = []    
    for i in range(4):
        dr = pd.date_range(pd.Timestamp(2000,1,1)+days(i*3)-hours(2),freq="H",periods=77)
        data = np.arange(1.,78.) + float(i*100.) 
        df = pd.DataFrame(index=dr,data=data)
        df.index.freq=None
        dfs.append(df)
        print(df)
    ts_long = ts_splice(dfs,transition='prefer_last',floor_dates=True)
    print(ts_long.to_string())

    
if __name__ == "__main__":
    main()