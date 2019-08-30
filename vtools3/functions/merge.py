


def ts_merge(series):
    """ merge a number of timeseries together and return a new ts.
    Similar to concatenate, but provides more control over order in cases of overlap
    
    Parameters
    ----------
    series  :  tuple(time series)`
        Series ranked from hight to low priority              

    Returns
    -------    
    merged : :class:`~vtools.data.timeseries.TimeSeries`
        A new time series with time interval same as the inputs, time extent
        the union of the inputs, and filled first with ts1, then with remaining
        gaps filled with ts2, then ts3....
        
    """

    # this concatenates, leaving redundant indices in df0, df1, df2
    dfmerge = pd.concat(series,sort=True)

    # finally, drop duplicate indices
    dfmerge = dfmerge.loc[~dfmerge.index.duplicated(keep='last')]    
    # Now apply, in priority order, each of the original dataframes to fill the original
    for ddf in series:
        dfmerge = dfmerge.combine_first(ddf)
    return dfmerge

