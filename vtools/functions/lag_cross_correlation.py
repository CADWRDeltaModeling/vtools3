import pandas as pd
import numpy as np

__all__=['calculate_lag']

def icrosscorr(lag,ts0, ts1):
    """ Lag-N cross correlation. 
    Shifted data filled with NaNs 
    
    Parameters
    ----------
    lag : int, default 0
    ts0, ts1 : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    """
    return ts0.corr(ts1.shift(int(lag)))

def mincrosscorr(lag,ts0,ts1):
    return -icrosscorr(lag,ts0,ts1)


def calculate_lag(lagged,base,max_lag,res,
                  interpolate_method = "linear"):

    """ Calculate shift in lagged, that maximizes cross-correlation with base.

        Parameters
        ----------
        base,lagged: :class:`Pandas.Series`
            time series to compare. The result is relative to base

        max_lag: interval
            Maximum pos/negative time shift to consider in cross-correlation 
            (ie, from -max_lag to +max_lag). Required windows in lagged will
            account for this bracket. For series dominated by a single 
            frequency (eg 1c/12.5 hours for tides), the algorithm can tolerate
            a range of 180 degrees (6 hours)
            
        res: interval
            Resolution of analysis. The series lagged will
            be interpolated to this resolution using interpolate_method. Unit
            used here will determine the type of the output. See documentation
            of the interval concept which is most compatible with 
            pandas.tseries.offsets, not timeDelta, because of better math
            properties in things like division -- vtime helpers like minutes(1) 
            may be helpful


        interpolate_method: str, optional
            Interpolate method to refine lagged to res. Must be compatible with 
            pandas interpolation method names (and hence scipy)

        Returns
        -------

        lag : interval
            Shift as a pandas.tseries.offsets subtype that matches units with res
            This shift is the apparent lateness (pos) or earliness (neg). It must be
            applied to base or removed to lagged to align the features.
        
    """
    from scipy.optimize import brent
    if not isinstance(base, pd.Series):
        raise ValueError("base and lagged arguments must be Series")
    if not isinstance(lagged, pd.Series):
        raise ValueError("base and lagged arguments must be Series")

    


    lagged_hr = lagged.resample(res).interpolate(interpolate_method)
    do_plot=False
    if do_plot:
        lagstride=6
        ilag = np.arange(-max_lag/res,max_lag/res,lagstride)
        ccors = []
        for i in ilag:
            ccors.append(icrosscorr(i,base,lagged_hr))
            
        dflag = pd.DataFrame(data=ccors,index=ilag)
        dflag.plot()
    
    # This uses a simple heuristic for bracketing a maximum
    # that is guaranteed to work if there is a single local maximum
    # in the interior. This allows the use of a 3-point bracket in brent, which
    # won't drift outside of the stipulated interval (the 2-point bracket 
    # tends to do this for max_lag > 0.25 period, wh ereas we generally
    # want max_lag to be about half a period)
    mlag = max_lag/res
    bracket = [-mlag, -mlag//3,mlag//3] 
    if icrosscorr(-mlag//3,base,lagged_hr) >icrosscorr(mlag//3,base,lagged_hr):
        bracket = [-mlag, -mlag//3,mlag//3] 
    else:
        bracket = [-mlag//3,mlag//3, mlag] 
  
    try:
        lagres = brent(mincrosscorr,args=(base,lagged_hr),brack=bracket)
    except ValueError as e0:
        if "bracketing" in str(e0).lower():
            arglist = list(e0.args)
            arglist[0] = e0.args[0]+" Argument max_interval may not contain a maximizing lag"
            e0.args = tuple(arglist)
            raise
    return int(lagres)*res


