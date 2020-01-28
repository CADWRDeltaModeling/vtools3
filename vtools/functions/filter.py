""" Module contains filter used in tidal time series analysis. 
"""

## Python libary import.
from numpy import abs
import pandas as pd
import numpy as np
from scipy import array as sciarray
from scipy.signal import lfilter,firwin,filtfilt
from scipy.signal.filter_design import butter


#__all__=["boxcar","butterworth","daily_average","godin","cosine_lanczos",\
#         "lowpass_cosine_lanczos_filter_coef","ts_gaussian_filter"]




## This dic saves the missing points for different intervals. 
#first_point={time_interval(minutes=15):48,time_interval(hours=1):36}

## This dic store tide butterworth filter' cut-off frequencies for 
## tide time series analysis, here set cutoff frequency at 0.8 cycle/day
## ,but it is represented as a ratio to Nyquist frequency for two
## sample intervals.
#butterworth_cutoff_frequencies={time_interval(minutes=15):0.8/(48),\
#                               time_interval(hours=1):0.8/12}

## supported interval
#_butterworth_interval=[time_interval(minutes=15),time_interval(hours=1)]

########################################################################### 
## Public interface.
###########################################################################



def process_cutoff(cutoff_frequency,cutoff_period,freq):
    if cutoff_frequency is None:
        if cutoff_period is None:
            raise("One of cutoff_frequency or cutoff_period must be given")
        cp = pd.tseries.frequencies.to_offset(cutoff_period)
        return 2.*freq/cp
    else:
        if cutoff_frequency < 0 or cutoff_frequency > 1.:
            raise ValueError("cutoff frequency must be 0 < cf < 1)")
        return cutoff_frequency


def cosine_lanczos(ts,cutoff_period=None,cutoff_frequency=None,filter_len=None,
                   padtype=None,padlen=None,fill_edge_nan=True):
    """ squared low-pass cosine lanczos  filter on a regular time series.
      
        
    Parameters
    -----------
    
    ts : :class:`DataFrame <pandas:pandas.DataFrame>`
    
    filter_len  : int, time_interval
        Size of lanczos window, default is to number of samples within filter_period*1.25.
        
    cutoff_frequency: float,optional
        Cutoff frequency expressed as a ratio of a Nyquist frequency,
        should within the range (0,1). For example, if the sampling frequency
        is 1 hour, the Nyquist frequency is 1 sample/2 hours. If we want a
        36 hour cutoff period, the frequency is 1/36 or 0.0278 cycles per hour. 
        Hence the cutoff frequency argument used here would be
        0.0278/0.5 = 0.056.
                      
    cutoff_period : string  or  :ref:`time_interval<time_intervals>`
         Period of cutting off frequency. If input as a string, it must 
         be  convertible to :ref:`Time interval<time_intervals>`.
         cutoff_frequency and cutoff_period can't be specified at the same time.
         
     padtype : str or None, optional
         Must be 'odd', 'even', 'constant', or None. This determines the type
         of extension to use for the padded signal to which the filter is applied. 
         If padtype is None, no padding is used. The default is None.

     padlen : int or None, optional
          The number of elements by which to extend x at both ends of axis 
          before applying the filter. This value must be less than x.shape[axis]-1. 
          padlen=0 implies no padding. If padtye is not None and padlen is not
          given, padlen is be set to 6*m.
    
     fill_edge_nan: bool,optional
          If pading is not used and fill_edge_nan is true, resulting data on 
          the both ends are filled with nan to account for edge effect. This is
          2*m on the either end of the result. Default is true.
  
    Returns
    -------
    result : :class:`~vtools.data.timeseries.TimeSeries`
        A new regular time series with the same interval of ts. If no pading 
        is used the beigning and ending 4*m resulting data will be set to nan
        to remove edge effect.
        
    Raise
    --------
    ValueError
        If input timeseries is not regular, 
        or, cutoff_period and cutoff_frequency are given at the same time,
        or, neither cutoff_period nor curoff_frequence is given,
        or, padtype is not "odd","even","constant",or None,
        or, padlen is larger than data size
        
    """
    
    #if not ts.is_regular():
    #    raise ValueError("Only regular time series are supported.")
        
    
    freq=ts.index.freq
    if freq is None:
        raise ValueError("Time series has no frequency attribute")
    
    m=filter_len
    
    cf = process_cutoff(cutoff_frequency,cutoff_period,freq)
    
    m = int(1.25 * 2. /cf)
   
        
        
    ##find out nan location and fill with 0.0. This way we can use the
    ## signal processing filtrations out-of-the box without nans causing trouble
    idx=np.where(np.isnan(ts.values))[0]
    data=sciarray(ts.values).copy()
    
    ## figure out indexes that will be nan after the filtration,which
    ## will "grow" the nan region around the original nan by 2*m
    ## slots in each direction
    if  len(idx)>0:
        data[idx]=0.0
        shifts=np.arange(-2*m,2*m+1)
        result_nan_idx=np.clip(np.add.outer(shifts,idx),0,len(ts)-1).ravel()
    
    if m<1:
        raise ValueError("bad input cutoff period or frequency")
        
    if padtype is not None:
        if (not padtype in ["odd","even","constant"]):
            raise ValueError("unkown padtype :"+padtype)
    
    if (padlen is None) and (padtype is not None):
        padlen=6*m

    if padlen is not None:   # is None sensible? 
        if padlen>len(data):
            raise ValueError("Padding length is more  than data size")


        
    ## get filter coefficients. sizeo of coefis is 2*m+1 in fact
    coefs= lowpass_cosine_lanczos_filter_coef(cf,m)
    
    
    d2=filtfilt(coefs,[1.0],data,axis=0,padtype=padtype,padlen=padlen)



    if(len(idx)>0):
        d2[result_nan_idx]=np.nan
    
    ## replace edge points with nan if pading is not used

    if (padtype is None) and (fill_edge_nan==True):
        d2[0:2*m]=np.nan
        d2[len(d2)-2*m:len(d2)]=np.nan


    out = ts.copy(deep=True)
    out[:]=d2
        
    return out




def butterworth(ts,cutoff_period=None,cutoff_frequency=None,order=4):
    """ low-pass butterworth-squared filter on a regular time series.
      
        
    Parameters
    -----------
    
    
    ts : :class:`DataFrame <pandas:pandas.DataFrame>`
        Must be one or two dimensional, and regular.
    
    order: int ,optional
        The default is 4.
        
    cutoff_frequency: float,optional
        Cutoff frequency expressed as a ratio with Nyquist frequency,
        should within the range (0,1). For a discretely sampled system,
        the Nyquist frequency is the fastest frequency that can be resolved by that 
        sampling, which is half the sampling frequency. For example, if the sampling frequency
        is 1 sample/1 hour, the Nyquist frequency is 1 sample/2 hours. If we want a
        36 hour cutoff period, the frequency is 1/36 or 0.0278 cycles per hour. 
        Hence the cutoff frequency argument used here would be
        0.0278/0.5 = 0.056.
                      
    cutoff_period : string  or  :ref:`time_interval<time_intervals>`
         Period corresponding to cutoff frequency. If input as a string, it must 
         be  convertible to a regular interval using the same rules as a pandas frequency..
         cutoff_frequency and cutoff_period can't be specified at the same time.
           
    Returns
    -------
    result : 
        A new regular time series with the same interval as ts.
        
    Raise
    --------
    ValueError
        If input order is not even, or input timeseries is not regular, 
        or neither cutoff_period and cutoff_frequency is given while input
        time series interval is not 15min or 1 hour, or  cutoff_period and cutoff_frequency 
        are given at the same time.
        
    """
    
    if (order%2):
        raise ValueError("only even order is accepted")
        
    #if not ts.is_regular():
    #    raise ValueError("Only regular time series can be filtered.")
    
    freq=ts.index.freq
    
#    if (not (interval in _butterworth_interval)) and (cutoff_period is None) and (cutoff_frequency is None):
#        raise ValueError("time interval is not supported by butterworth if no cuttoff period/frequency given.")

    if (not (cutoff_frequency is None)) and (not(cutoff_period is None)):
        raise ValueError("cutoff_frequency and cutoff_period can't be specified simultaneously")

    if (cutoff_frequency is None) and (cutoff_period is None):
        raise ValueError("Either cutoff_frequency or cutoff_period must be given")
    
    
    cf=cutoff_frequency
    
    
    if (cf is None):
        if  (not(cutoff_period is None)):
            cutoff_period = pd.tseries.frequencies.to_offset(cutoff_period)
            cf = 2.*freq/cutoff_period
        else:
            cf=butterworth_cutoff_frequencies[interval]


    ## get butter filter coefficients.
    [b,a]=butter(order/2,cf)
    d2=filtfilt(b,a,ts.values,axis=0,padlen=90)
    out = ts.copy(deep=True)
    out.values=d2

#    prop={}
#    for key,val in ts.props.items():
#        prop[key]=val
#    prop[TIMESTAMP]=INST
#    prop[AGGREGATION]=INDIVIDUAL
#    time_interval
    return out
    

    
def lowpass_cosine_lanczos_filter_coef(cf,m,normalize=True):
    """return the convolution coefficients for low pass lanczos filter.
      
    Parameters
    -----------
    
    Cf: float
      Cutoff frequency expressed as a ratio of a Nyquist frequency.
                  
    M: int
      Size of filtering window size.
        
    Returns
    --------
    pdb.set_trace()
    Results: list
           Coefficients of filtering window.
    
    """
    
    coscoef=[cf*np.sin(np.pi*k*cf)/(np.pi*k*cf) for k in np.arange(1,m+1,1,dtype='d')]
    sigma=[np.sin(np.pi*k/m)/(np.pi*k/m) for k in np.arange(1,m+1,1,dtype='float')]
    prod= [c*s for c,s in zip(coscoef,sigma)]
    temp = prod[-1::-1]+[cf]+prod
    res=sciarray(temp)
    if normalize:
        res = res/res.sum()
    return res    
    
def generate_godin_fir(timeinterval):
    '''
    generate godin filter impulse response for given timeinterval
    timeinterval could be anything that pd.Timedelta can accept
    '''
    mins=pd.Timedelta(timeinterval).seconds/60
    wts24=np.zeros(round(24*60/mins))
    wts24[:]=1/wts24.size
    tidal_period=round(24.75*60/mins)
    if tidal_period%2==0: tidal_period=tidal_period+1
    wts25=np.zeros(tidal_period)
    wts25[:]=1.0/wts25.size
    return np.convolve(wts25,np.convolve(wts24,wts24))
    
def godin_filter(ts):
    """ Low-pass Godin filter a regular time series.
    Applies the :math:`\mathcal{A_{24}^{2}A_{25}}` Godin filter [1]_
    The filter is generalized to be the equivalent of one
    boxcar of the length of the lunar diurnal (~25 hours)
    constituent and two of the solar diurnal (~24 hours), though the
    implementation combines these steps.
    
    
    Parameters
    -----------
    
    ts : :class:`DataFrame <pandas:pandas.DataFrame>`
  
    Returns
    -------
    result : :class:`DataFrame <pandas:pandas.DataFrame>`
        A new regular time series with the same interval of ts. 
        
    Raise
    --------
    ValueError
        If input time series is not univariate
        
    References
    ----------
    .. [1] Godin (1972) Analysis of Tides
        
    """
    freqstr=ts.index.freqstr
    if freqstr == None:
        freqstr=pd.infer_freq(ts.index)
    if freqstr == None:
        raise Exception("""No regular frequency could be determined from the index of the data frame or from infer_freq method. Try on smaller slice of dataframe index""")
    godin_ir=generate_godin_fir(freqstr)
    if not (len(ts.columns) == 1):
        raise ValueError("Godin Filter not functional for multivariate series yet")
    dfg=pd.DataFrame(np.convolve(ts.iloc[:,0].values,godin_ir,mode='same'), 
        columns=ts.columns, index = ts.index)

    return dfg
#
