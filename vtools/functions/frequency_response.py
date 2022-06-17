# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 13:17:21 2014

@author: qshu
"""

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from vtools.functions.filter import *
from scipy.signal import freqz
import numpy as np
import pandas as pd
import pdb

plt.style.use(["seaborn-colorblind","seaborn-talk"])

def compare_response(cutoff_period):
    """ Generate a comparison plot of the frequency response of filters used in tidal work
        The comparison includes, squared cosine-Lanczos, Godin and boxcar.
    """
    interval=0.25 # sample interval 15min as ratio of one hour
    
    cf=2.0*interval/cutoff_period #cufoff frequency as ratio of Nyquist frequency
    
    ## C_L of size 70hours
    m0=int(70.0/interval) # cosine lanczos filer window size as number of smaples, here use size of 70hours
    a=1
    b0 = lowpass_cosine_lanczos_filter_coef(cf,m0,normalize=True)
    worN=4096
    b_cosine_lanczos_70=np.array([b0,b0])
    
    ## C_L of default size,which is 1.25 times number of 
    ## interval within cutoff period
    m2= int(1.25*2.0/cf)
    b2 = lowpass_cosine_lanczos_filter_coef(cf,m2,normalize=True)
    b_cosine_lanczos_default=np.array([b2,b2])
    
    ## godin response is computed by multiplying responses of 
    ## three boxcar filter on hourly data (23,23,24)
    ## 15
    l1=99
    l2=96
    l3=96
    b3_1 = [1.0/l1]*l1
    b3_2 = [1.0/l2]*l2
    b3_3 = [1.0/l3]*l3
    b_god=np.array([b3_1,b3_2+[0.0,0.0,0.0],b3_3+[0.0,0.0,0.0]])
      
    ## compute boxcar coefficients for 24 and 25 hours
    num_intervl=int(24/interval)
    b4=[1.0/num_intervl]*num_intervl
    b_box_24h=np.array([b4,b4])
    
    num_intervl=int(25/interval)
    b5=[1.0/num_intervl]*num_intervl
    b_box_25h=np.array([b5,b5])    
    response={}
    

    fig = plt.figure(figsize=(6,6),dpi=300)
    
    ax = fig.add_subplot(1,1,1)
    ax.set_ylim(-.2,1.5)
    
    for b,label in [(b_cosine_lanczos_70,"C_L,size=70 hours"),(b_cosine_lanczos_default,"C_L,size=%s default"%m2),(b_god,"godin"),(b_box_24h,"boxcar 24hours"),(b_box_25h,"boxcar 25hours")]:
        w,h =freqz(b.T[...,np.newaxis],worN=worN)
        pw=w[1:]
        ## convert frequence to period in hours
        period=1.0/pw
        period=2.0*np.pi*period*interval
        hh=np.abs(np.prod(h,axis=0))
        response[label]=hh
        ax.plot(period,hh[1:],linewidth=1,label=label)   
    ax.set_xlim(0.1,400)
    ax.axvline(x=cutoff_period,ymin=-0.2,linewidth=1,color='r')
    ax.annotate("cutoff period=%f h"%cutoff_period,(cutoff_period,1.2),xycoords='data',\
                xytext=(50, 50), textcoords='offset points',\
               arrowprops=dict(arrowstyle="->"))
    ax.set_ylabel(r'Magnitude')
    ax.set_xlabel(r'Period(hours)')
    plt.grid(b=True, which='both', color='0.9', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    ax.legend(loc="lower right")
    
def unit_impulse_ts(size,interval):
    """
    Parameters
    ----------
    size : int
        Length of impluse time series, odd number.
    interval : string
        time series interval, such as "15min"

    Returns
    -------
    pandas.dataframe
        time sereies value 0.0 except 1 at the middle of time series.

    """
    idx=pd.date_range("2001-1-1",periods=size,freq=interval)
    val=np.zeros(size)
    mid_idx=int(size/2)
    val[mid_idx]=1.0
    #return pd.Series(val,index=idx) # series doesn't works with godin
    return pd.DataFrame(index=idx,data=val)
    

def compare_response2(unit_impulse,cutoff_period):
    
    fig = plt.figure(figsize=(6,6),dpi=300)
    ax = fig.add_subplot(1,1,1)
    ax.set_ylim(-.2,1.5)
    worN=4096
    dt=unit_impulse.index[1]-unit_impulse.index[0]
    cf=2.0*dt.seconds/(cutoff_period*3600) #cuf off frequency as ratio of Nyquist frequency
    dt_hours_ratio=dt.seconds/3600.0
    ## set consine_lanczos as 70h
    cl_size_hours1=70
    cl_size1=int(cl_size_hours1*3600/dt.seconds)
    ##  default consine_lanczos size
    default_cl_size= int(1.25*2.0/cf)
    for (unit_impulse_response,label) in [(cosine_lanczos(unit_impulse,cutoff_period="40h",filter_len=cl_size1),"C_L,size=%s hours"%cl_size_hours1),\
                         (cosine_lanczos(unit_impulse,cutoff_period="40h"),"C_L,size=%s default"%default_cl_size),\
                         (unit_impulse.rolling(96,center=True,min_periods=96).mean(),"boxcar24"),\
                         (unit_impulse.rolling(99,center=True,min_periods=99).mean(),"boxcar25"),\
                         (godin(unit_impulse),"godin")]: 
        b=unit_impulse_response.fillna(0.0)
        w,h =freqz(b.values,worN=worN)
        pw=w[1:]
        ## convert frequence to period in hours
        period=1.0/pw
        period=2.0*np.pi*period*dt_hours_ratio
        hh=np.abs(h)
        ax.plot(period,hh[1:],linewidth=1,label=label)   
    ax.set_xlim(0.1,400)
    ax.axvline(x=cutoff_period,ymin=-0.2,linewidth=1,color='r')
    ax.annotate("cutoff period=%f h"%cutoff_period,(cutoff_period,1.2),xycoords='data',\
                xytext=(50, 50), textcoords='offset points',\
               arrowprops=dict(arrowstyle="->"))
    ax.set_ylabel(r'Magnitude')
    ax.set_xlabel(r'Period(hours)')
    plt.grid(visible=True, which='both', color='0.9', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    ax.legend(loc="lower right")
    
if __name__=="__main__":
    ## compare response for data with 15min interval 
    unit_impulse=unit_impulse_ts(5001,"15min")
    ## 
    compare_response2(unit_impulse,40)
    #compare_response(40)
    plt.savefig('frequency_response',bbox_inches=0)
    
