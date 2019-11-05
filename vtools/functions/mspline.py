
from _monotonic_spline import *
import pandas as pd
from vtools3.functions._monotonic_spline import _monotonic_spline
import matplotlib.pyplot as plt

def monotonic_spline(ts,dest):    
    """ Interpolating a regular time series (rts) to a finer rts by rational histospline.

    The rational histospline preserves area under the curve. This is a good choice of spline for period averaged data where an interpolant is desired
    that is 'conservative'. Note that it is the underlying continuous interpolant
    that will be 'conservative though, not the returned discrete time series which
    merely samples the underlying interpolant.    
    
    Parameters
    ----------

    ts : :class:`Pandas.DataFrame`
        Series to be interpolated, typically with DatetimeIndex

    dest : a pandas freq code (e.g. '16min' or 'D') or a DateTimeIndex
        
    Returns
    -------
    result : :class:`~Pandas.DataFrame`
        A regular time series with same columns as ts, populated with instantaneous values and with
        an index of type DateTimeIndex
    """
    # Get the source index including two endpoints
    # todo: easier way?    
    ndx = ts.index
    strt = ndx[0] 
    end = ndx[-1]
    x = (ndx - strt).total_seconds().to_numpy()   
 
    print(x)
    print(strt)
    print(end)
    if not isinstance(dest,pd.Index):
        end=ndx[-1].round(dest)
        dest = pd.date_range(start = strt, end = end, freq = dest)
        
    xnew = (dest-strt).total_seconds().to_numpy()
    print(len(xnew))
    cols = ts.columns
    result = pd.DataFrame([],index=dest)
    for col in cols:
        y = ts[col].to_numpy()
        result[col] = _monotonic_spline(x,y,xnew)
    return result    
    
    
    
if __name__ == "__main__":
    nper = 20
    xx = np.linspace(0,12.,nper)
    yy = np.cos(2.*np.pi*xx/6.)
    yy2 = yy + 1.3*np.sin(2.*np.pi*xx/6.1)
    ndx = pd.date_range(start=pd.Timestamp(2009,2,10),freq='H',periods=nper)
    ndx2 = pd.date_range(start=pd.Timestamp(2009,2,10),freq='15min',periods=90)    
    tsin = pd.DataFrame({"data":yy,"data2":yy2},index=ndx)
    tsout = monotonic_spline(tsin,ndx2)

    fig,ax = plt.subplots(1)
    tsin.plot(ax=ax)
    tsout.plot(ax=ax)
    plt.show()
