import vtools
from vtools.data.timeseries import *
from vtools.functions.interpolate import *
#from vtools.functions.shift import *
import numpy as np



def ts_data_arg(f):
    """ Decorator used to ensure that functions can be used on time series or arrays"""
    def b(predictions,targets,*args,**kwargs):
        if( isinstance(predictions,vtools.data.timeseries.TimeSeries) ):
            x = predictions.data
        else:
            x = predictions
        if( isinstance(targets,vtools.data.timeseries.TimeSeries) ):
            y = targets.data
        else:
            y = targets  
        return f(x,y,**kwargs) 
    b.__name__= f.__name__
    b.__doc__ = f.__doc__
    return b



@ts_data_arg
def mse(predictions, targets):
    """Mean squared error

       Parameters
       ----------
       predictions, targets : array_like
           Time series or arrays to analyze       
       
       Returns
       -------
       mse : vtools.data.timeseries.TimeSeries
           Mean squared error between predictions and targets
    """
    return np.nanmean((predictions - targets) ** 2.)

@ts_data_arg
def rmse(predictions, targets):
    """Root mean squared error
       
       Parameters
       ----------
       predictions, targets : array_like
           Time series or arrays to analyze          
       
       Returns
       -------
       mse : float
           Mean squared error
    """
    return np.sqrt(np.nanmean((predictions - targets) ** 2.))

@ts_data_arg    
def median_error(predictions, targets):
    """ Calculate the median error, discounting nan values
    
        Parameters
        ----------
        predictions, targets : array_like
        
            Time series or arrays to be analyzed
        
        Returns
        -------
        med : float
            Median error
    """
    return np.nanmedian(predictions - targets)

@ts_data_arg  
def skill_score(predictions,targets,ref=None):
    """Calculate a Nash-Sutcliffe-like skill score based on mean squared error
       
       As per the discussion in Murphy (1988) other reference forecasts (climatology, 
       harmonic tide, etc.) are possible.
       
       Parameters
       ----------
       predictions, targets : array_like
           Time series or arrays to be analyzed
       
       Returns
       -------
       rmse : float
           Root mean squared error
    """
    if not ref:
        ref = np.nanmean(targets)
    else:
        if isinstance(ref,vtools.data.timeseries.TimeSeries):
            ref = ref.data
            
    return 1.0 - (mse(predictions,targets)/mse(ref,targets))

@ts_data_arg     
def tmean_error(predictions,targets,limits=None,inclusive=[True,True]):
    """ Calculate the (possibly trimmed) mean error, discounting nan values
    
        Parameters
        ----------
        predictions, targets : array_like
            Time series or arrays to be analyzed
        
        limits : tuple(float)
            Low and high limits for trimming 
            
        inclusive : [boolean, boolean]
            True if clipping is inclusive on the low/high end
        
        Returns
        -------
        mean : float
            Trimmed mean error
    """
    import scipy
    y = numpy.ma.masked_invalid(predictions)
    z = numpy.ma.masked_invalid(targets)
    return scipy.stats.mstats.tmean(y-z,limits)

@ts_data_arg     
def corr_coefficient(predictions,targets,bias=False):
    """Calculates the correlation coefficient (the 'r' in '-squared' between two series.
   
    For time series where the targets are serially correlated and may span only a fraction
    of the natural variability, this statistic may not be appropriate and Murphy (1988) explains
    why caution should be exercised in using this statistic.
    
    Parameters
    ----------
    predictions, targets : array_like
        Time series to analyze
    
    bias : boolean
        Whether to use the biased (N) or unbiased (N-1) sample size for normalization
    
    Returns
    -------
    r : float
        Correlation coefficient
    """
    
    
    from numpy.ma import corrcoef
    y = numpy.ma.masked_invalid(predictions)
    z = numpy.ma.masked_invalid(targets)
    return corrcoef(y,z,bias)[0][1]
    
def _main():
    #from vtools.data.sample_series import arma
    start=datetime(2009,3,12)
    intvl = minutes(15)
    lag = minutes(37)
    tseq = time_sequence(start, intvl, n=4000).astype('d')/float(ticks_per_hour)
    trend = (tseq-tseq[0])/4000.
    noise = arma([0.2,0.1],[0.1,-0.1],sigma=0.1,n=len(tseq),discard=500)
    base = np.cos(tseq*2.*np.pi/12.4) + 0.4*np.cos(tseq*2.*np.pi/24. - 1.2)
    scale = 1.30
    y0 = base + noise
    y1 = scale*base + trend
    ts0 = rts(y0,start,intvl)
    ts1 = rts(y1,start,intvl)
    ts1 = shift(ts1,lag)
    
    lag_sec = calculate_lag(ts0,ts1, (datetime(2009,3,14),datetime(2009,4,14)),max_shift=minutes(60),period = days(21))
    print("lag = %s" % (lag_sec/60.))
    print("skill = %s" % skill_score(ts0.data, shift(ts1,seconds(lag_sec)).data))
    
    x = np.array([2.,4.,6.])
    y = x*1.5
    print(mse(x,y.mean()))
    print(rmse(x,x.mean()))
    print(skill_score(x,y))
    print(median_error(x,y))
    
    z0 = np.array([1.,np.nan,7.,5.,7.,10.,12.,14.])
    z1 = np.array([2.,3.,5.,6.,8.,11.,13.,36.])
    zts0 = rts(z0,start,intvl)
    zts1 = rts(z1,start,intvl)
    print("OK")
    print("tmean %s" % tmean_error(z0,z1,limits=(-1.,1.)))
    
    print("tmean %s" % tmean_error(zts0,zts1,limits=(-1.,1.)))
    print("r %s" % corr_coefficient(zts0,zts1))

 
    
all = [calculate_lag,mse,rmse,median_error,tmean_error,corr_coefficient,skill_score]
    
if __name__ == "__main__":
    _main()
