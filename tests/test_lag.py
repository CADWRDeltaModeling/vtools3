
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import pytest
from vtools.data.timeseries import datetime_elapsed
from vtools.data.vtime import hours,minutes,seconds
from statsmodels.tsa.arima_process import arma_generate_sample
from .lag_cross_correlation import calculate_lag
import matplotlib.pyplot as plt
#pd.plotting.register_matplotlib_converters()

def lag_samples(n):
    d0 = pd.date_range(start="2009-02-10",
                       end="2009-04-20T00:30",freq="15min")
    nval = len(d0)
    d0_elapsed = datetime_elapsed(d0)
    vals = np.cos(2*np.pi*d0_elapsed.to_numpy()/44000.)
    df = pd.Series(data=vals,index=d0)
    arparams = np.array([0.975])
    maparams = np.array([.5,0.25,0.25])
    ar = np.r_[1, -arparams] # add zero-lag and negate
    ma = np.r_[1, maparams] # add zero-lag
    arma1 = arma_generate_sample(ar, ma,nsample=nval) 
    ex = 10.*df + 0.4*arma1
    ex0 = pd.Series(ex.to_numpy(),index =d0.copy())
    ex1 =  0.5*(10.1*df + 0.45*arma1).shift(n) \
         + 0.5*(10.0*df + 0.3*arma1).shift(n+1) \
         + 0.3*arma_generate_sample(ar,ma,nsample=len(ex))
    ex0.iloc[120] = np.nan
    ex0.iloc[2002:2223] = np.nan
    ex0=ex.iloc[0:len(ex0-44)]
    ex1.iloc[222:225] = np.nan
    ex1.iloc[333:334] = np.nan
    return ex0,ex1




def test_lag_cross_correlation():
    base,lagged = lag_samples(7)
    base.to_frame().plot()
    #lagged.plot()
    #print(lagged.info())
    #print(base.info())
    #base.plot()
    #plt.show()
    max_lag=minutes(300)
    res = minutes(1)

    lag_calc = calculate_lag(base,lagged,max_lag,res,
                  interpolate_method = "linear")

    max_lag=minutes(100)
    res = minutes(1)

    with pytest.raises(ValueError):
        lag_calc = calculate_lag(base,lagged,max_lag,res,
                  interpolate_method = "linear")
    
    base,lagged = lag_samples(20)
    max_lag = minutes(300)
    with pytest.raises(ValueError):
        lag_calc = calculate_lag(base,lagged,max_lag,res,
                  interpolate_method = "linear")

if __name__ == "__main__":
    test_lag_cross_correlation()

 