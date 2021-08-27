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
    dfmerge = series[0].reindex(dfmerge.index)  

    # drop duplicate indices
    dfmerge = dfmerge.loc[~dfmerge.index.duplicated(keep='last')]
    # Now apply, in priority order, each of the original dataframes to fill the original
    for ddf in series[1:]:
        dfmerge = dfmerge.combine_first(ddf)
    return dfmerge
