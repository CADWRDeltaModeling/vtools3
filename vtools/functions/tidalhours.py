
################################################################
# Three related functions for analyzing tidal cycles from time series data:
# 1. `find_slack(jd, u, leave_mean=False, which='both')`

# - **Purpose:**  
#   Identifies the times of "slack water"—the moments when tidal current velocity (`u`) crosses zero (i.e., transitions between ebb and flood tides).
# - **Inputs:**  
#   - `jd`: Array of time values (Julian days or similar).
#   - `u`: Array of velocity values (flood-positive).
#   - `leave_mean`: If `False`, removes the mean (low-frequency) component from `u`.
#   - `which`: `'both'`, `'high'`, or `'low'`—specifies which zero-crossings to return.
# - **Process:**  
#   - Applies a lowpass filter to smooth `u`.
#   - Optionally removes the mean (longer-term trend).
#   - Interpolates missing values.
#   - Finds indices where `u` crosses zero (from negative to positive or vice versa).
#   - Interpolates to estimate the exact time of zero-crossing.
#   - Determines whether the series starts with ebb or flood.
# - **Returns:**  
#   - `jd_slack`: Array of times when slack water occurs.
#   - `start`: String, either `'ebb'` or `'flood'`, indicating the initial state.

# ---

# ### 2. `hour_tide(jd, u=None, h=None, jd_new=None, leave_mean=False)`

# - **Purpose:**  
#   Calculates the "tidal hour" for each time point, i.e., the phase of the tidal cycle (0–12, where 0 is slack before ebb).
# - **Inputs:**  
#   - `jd`: Array of time values.
#   - `u`: Velocity time series (optional).
#   - `h`: Water level time series (optional).
#   - `jd_new`: Optional new time points to evaluate.
#   - `leave_mean`: See above.
# - **Process:**  
#   - If only `h` is given, computes velocity as the time derivative of `h`.
#   - Calls `hour_tide_fn` to get a function mapping time to tidal hour.
#   - Evaluates this function at `jd_new` (or `jd` if not provided).
# - **Returns:**  
#   - Array of tidal hour values (0–12) for each time point.

# ---

# ### 3. `hour_tide_fn(jd, u, leave_mean=False)`

# - **Purpose:**  
#   Returns a function that computes tidal hour for arbitrary time points, based on the provided time/velocity series.
# - **Inputs:**  
#   - `jd`: Time array.
#   - `u`: Velocity array.
#   - `leave_mean`: See above.
# - **Process:**  
#   - Finds slack water times using `find_slack`.
#   - Interpolates between slacks to assign a tidal hour (0–12) to each time.
#   - Uses angular interpolation to handle the cyclic nature of the tidal hour.
#   - Returns a function that maps new time points to tidal hour.
# - **Returns:**  
#   - Function: `fn(jd_new)` → tidal hour array.

# ---

# **Summary:**  
# These functions are used to analyze tidal time series, identifying slack water times and mapping
# any time to its position within the tidal cycle (tidal hour). This is useful for tidal phase analysis
# in estuarine and coastal studies.
#################################################################
import numpy as np
from scipy.interpolate import interp1d
from vtools.functions.filter import cosine_lanczos5
import pandas as pd
from scipy.signal import hilbert
from typing import Union

__all__=['find_slack', 'hour_tide', 'hour_tide_fn',
         'tidal_hour_signal', 'diff_h']

def find_slack(jd,u,leave_mean=False,which='both'):
    # returns ([jd, ...], 'high'|'low')
    dt=jd[1]-jd[0]
    dtdelta=pd.Timedelta(dt,unit='D')
    dayindex=pd.timedelta_range(start=jd[0],periods=len(jd), freq=dtdelta)
    u_ts = pd.Series(u,index=dayindex)
    # ~1 hour lowpass
    u_ts=cosine_lanczos5(u_ts,cutoff_period="1h")
    if not leave_mean:
        u_ts-=cosine_lanczos5(u_ts,cutoff_period="16h")
    u=u_ts.values
    missing=np.isnan(u)
    u[missing]=np.interp(jd[missing],
                         jd[~missing],u[~missing])

    # transition from ebb/0 to flood, or the other way around
    sel_low=(u[:-1]<=0) & (u[1:]>0)
    sel_high=(u[:-1]>0) & (u[1:]<=0)
    if which=='both':
        sel=sel_low|sel_high
    elif which=='high':
        sel=sel_high
    elif which=='low':
        sel=sel_low
    else:
        assert(False)

    b=np.nonzero(sel)[0]
    jd_slack=jd[b]-u[b]/(u[b+1]-u[b])*dt
    if u[0]<0:
        start='ebb'
    else:
        start='flood'
    return jd_slack,start


def hour_tide(jd,u=None,h=None,jd_new=None,leave_mean=False,start_datum="ebb"):
    """
    Calculate tide-hour from a time series of u (velocity, flood-positive)
    or h (water level, i.e. positive-up).

    jd: time in days, i.e. julian day, datenum, etc.
    u: velocity, flood-positive
    h: water level, positive up.
    jd_new: if you want to evaluate the tidal hour at a different
      (but overlapping) set of times.
    leave_mean: by default, the time series mean is removed, but this 
     can be disabled by passing True here.

    the return values are between 0 and 12, with 0 being slack before
    ebb (hmm - may need to verify that.).
    """
    assert (u is None) != (h is None),"Must specify one of u,h"
    if h is not None:
        # abuse cdiff to avoid concatenate code here
        dh_dt=cdiff(h) / cdiff(jd)
        dh_dt[-1]=dh_dt[-2]
        dh_dt=np.roll(dh_dt,1) # better staggering to get low tide at h=0
        u=dh_dt
        
    fn=hour_tide_fn(jd,u,leave_mean=leave_mean,start_datum=start_datum)

    if jd_new is None:
        jd_new=jd
    return fn(jd_new)


def cdiff(a,n=1,axis=-1):
    """
    Like np.diff, but include difference from last element back
    to first.
    """
    assert n==1 # not ready for higher order
    # assert axis==-1 # also not ready for weird axis

    result=np.zeros_like(a)
    d=np.diff(a,n=n,axis=axis)

    # using [0] instead of 0 means that axis is preserved
    # so the concatenate is easier
    last=np.take(a,[0],axis=axis) - np.take(a,[-1],axis=axis)

    # this is the part where we have to assume axis==-1
    #return np.concatenate( [d,last[...,None]] )

    return np.concatenate( [d,last] )


def hour_tide_fn(jd,u,start_datum="ebb",leave_mean=False):
    """ Return a function for extracting tidal hour
    from the time/velocity given.
    Use the _fn version if making repeated calls with different jd_new,
    but the same jd,u
    """
    #function hr_tide=hour_tide(jd,u,[jd_new],[leave_mean]);
    #  translated from rocky's m-files
    #   generates tidal hours starting at slack water, based on
    #   u is a vector, positive is flood-directed velocity
    #   finds time of "slack" water
    #   unless leave_mean=True, removes low-pass velocity from record

    jd_slack,start=find_slack(jd,u,leave_mean=leave_mean,which='both')
    tide_period =12 # semidiurnal tide period in hours
    half_tide = tide_period / 2.0
    # left/right here allow for one more slack crossing
    hr_tide=np.interp(jd,
                      jd_slack,np.arange(len(jd_slack))*half_tide,
                      left=-0.01,right=len(jd_slack)-0.99)
    if start!=start_datum: 
           # if we start on flood while input start_datum is ebb, then we need to shift
           # the tidal hour by half a tide period. vice versa.
           hr_tide += half_tide # starting on an ebb
    #if start=='ebb':
    #       hr_tide += half_tide # starting on an flood

    # print("start is",start)
    hr_tide %= tide_period
    
    # angular interpolation - have to use scipy interp1d for complex values
    arg=np.exp(1j*hr_tide*np.pi/half_tide)
    def fn(jd_new):
        argi=interp1d(jd,arg,bounds_error=False)(jd_new)
        hr_tide=(np.angle(argi)*6/np.pi) % tide_period
        return hr_tide
    return fn

def tidal_hour_signal2(ts: Union[pd.Series, pd.DataFrame], filter: bool = True) -> Union[pd.Series, pd.DataFrame]:
    """
    Calculate the tidal hour from a semidiurnal tidal signal.

    Parameters
    ----------
    ts : pd.Series or pd.DataFrame
        Input time series of water levels (must have datetime index)
    filter : bool, optional
        If True, apply Lanczos filter to remove low-frequency components (default True)
        Note this is opposite of 'leave_mean' in original implementation

    Returns
    -------
    pd.Series or pd.DataFrame
        Tidal hour in datetime format (same shape as input)

    Notes
    -----
    The tidal hour represents the phase of the semidiurnal tide in temporal units.
    The calculation uses complex interpolation for smooth phase estimation:
    1. The Hilbert transform creates an analytic signal
    2. The angle gives the instantaneous phase
    3. Complex interpolation avoids phase jumps at 0/2π boundaries
    4. This provides continuous phase evolution even during slack tides

    If h/u distinction is needed, consider applying `diff_h` to separate
    flood/ebb phases. The derivative was likely included in original code
    to identify phase reversals during tidal current analysis.
    """
    # Ensure we're working with a DataFrame to handle both Series and DataFrame inputs
    if isinstance(ts, pd.Series):
        df = ts.to_frame()
        return_series = True
    else:
        df = ts.copy()
        return_series = False

    # Apply filtering if requested (opposite sense of original leave_mean)
    if filter:
        filtered = cosine_lanczos5(df, '40h')  
    else:
        filtered = df

    # Remove mean if filtered (since cosine_lanczos may preserve it)
    if filter:
        detrended = filtered - filtered.mean()
    else:
        detrended = filtered

    # Hilbert transform to get analytic signal
    #analytic_signal =  detrended + 1j * hilbert(detrended)
    analytic_signal = hilbert(detrended, axis=0)

    # Get instantaneous phase (angle) - using complex interpolation
    phase = np.unwrap(np.angle(analytic_signal))

    # Convert phase to tidal hours (semidiurnal cycle = 12.42 hours)
    tidal_period_hours = 12.4206
    tidal_hours = (phase * tidal_period_hours / (2 * np.pi)) % tidal_period_hours

    # Convert to timedelta and add to original index
    result = pd.to_timedelta(tidal_hours, unit='h') + df.index

    # Return same type as input
    if return_series:
        return result.iloc[:, 0]
    return result


def tidal_hour_signal(ts, filter=True):
    """
    Compute the tidal hour of a semidiurnal signal.

    This function returns the instantaneous phase-based tidal hour for a time series, 
    assuming a semidiurnal signal. Optionally applies a cosine Lanczos low-pass 
    filter (e.g., 40h) to isolate tidal components from subtidal or noisy fluctuations.

    The tidal hour is computed using the phase of the analytic signal obtained 
    via the Hilbert transform. This phase is then scaled to range from 0 to 12 hours 
    to represent one semidiurnal tidal cycle. The output is a pandas Series aligned 
    with the input time index.

    Parameters
    ----------
    ts : pandas.Series
        Time series of water level or other semidiurnal signal. Must have a datetime index.
    filter : bool, default True
        Whether to apply a 40-hour cosine Lanczos filter to the input signal. If False, 
        uses the raw signal.

    Returns
    -------
    pandas.Series
        Tidal hour as a float (range [0, 12)), indexed by datetime.

    Notes
    -----
    The tidal hour is derived from the instantaneous phase of the analytic signal.
    This signal is computed as:

        `analytic_signal = ts + 1j * hilbert(ts)`

    The phase (angle) of this complex signal varies smoothly over time and reflects
    the oscillatory nature of the tide, allowing us to construct a continuous 
    representation of "tidal time" even between extrema.

    The use of the Hilbert transform provides a smooth interpolation of the signal's 
    phase progression, since it yields the narrow-band envelope and instantaneous phase 
    of the dominant frequency component (assumed to be semidiurnal here).

    See Also
    --------
    diff_h : Compute the derivative (rate of change) of tidal hour.
    cosine_lanczos : External function used to apply low-pass filtering.
    """
    if not isinstance(ts, pd.Series):
        raise ValueError("Input `ts` must be a pandas Series.")
    if filter:
        ts -= cosine_lanczos5(ts, cutoff_period='40h')

    analytic = ts + 1j * hilbert(ts)
    phase = np.angle(analytic)
    # Shift from [-π, π] to [0, 2π]
    phase = (phase + 2 * np.pi) % (2 * np.pi)

    # Map phase in [0, 2π) to tidal hour in [0, 12)
    tidal_hour = (phase / (2 * np.pi)) * 12

    return pd.Series(tidal_hour, index=ts.index, name='tidal_hour')


def diff_h(tidal_hour_series):
    """
    Compute the time derivative of tidal hour.

    This derivative is often included to capture how rapidly the tidal phase is changing,
    which can be important in modeling flow reversals, estuarine dynamics, or for detecting
    slack tide conditions where the rate of change is near zero.

    Parameters
    ----------
    tidal_hour_series : pandas.Series
        Output of `tidal_hour_signal`, indexed by datetime.

    Returns
    -------
    pandas.Series
        Time derivative of tidal hour (dH/dt) in hours/hour, indexed by datetime.
    """
    return tidal_hour_series.diff() / tidal_hour_series.index.to_series().diff().dt.total_seconds().div(3600)
