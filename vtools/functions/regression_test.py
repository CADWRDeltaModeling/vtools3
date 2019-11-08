
from sklearn.linear_model import LinearRegression
import numpy as np

import pandas as pd
import statsmodels.api as sm
import statsmodels as sm
import matplotlib.pyplot as plt
import matplotlib

from scipy.signal import boxcar
from read_scalar import *
from vtools3.functions.filter import cosine_lanczos
from vtools3.data.vtime import hours,minutes,days


from unit_conversions import ec_psu_25c
#from montezuma_flow import *    # Don't want to copy and pste functions



def regression_coef(series):
    na = np.isnan(series)
    n = len(series)
    num_nan = np.count_nonzero(na)
    count = n - num_nan
    y=series[~na]
    ymean = np.mean(y)
    if count < 2 or count/n < 0.5: 
        #print("Too many nans")
        return np.array([np.nan],dtype='d')
    #y = series.copy().dropna()
    try:
        X = (np.arange(n,dtype='d')-float(n))[~na,np.newaxis]
    except:
        return np.array([np.nan],dtype='d')
     
    reg = 10000.*LinearRegression().fit( X, y).coef_/ymean
    #reg = LinearRegression().fit( X, y).intercept_ #_/ymean
    #print "toot"
    return reg


def regression_intercept(series):
    na = np.isnan(series)
    n = len(series)
    num_nan = np.count_nonzero(na)
    count = n - num_nan
    y=series[~na]
    ymean = np.mean(y)
    if count < 2 or count/n < 0.5: 
        #print("Too many nans")
        return np.array([np.nan],dtype='d')
    #y = series.copy().dropna()
    try:
        X = (np.arange(n,dtype='d')-float(n))[~na,np.newaxis]
    except:
        return np.array([np.nan],dtype='d')
     
    #reg = 10000.*LinearRegression().fit( X, y).coef_/ymean
    reg = LinearRegression().fit( X, y).intercept_ #_/ymean
    #print "toot"
    return reg


def read_des(fpat):
    start = dtm.datetime(2008,1,1)
    end = dtm.datetime(2019,5,1)
    mdir = "//cnrastore-bdo/Modeling_Data/des_emp/raw"
    
    ts=csv_retrieve_ts(fpat,mdir,start,end,selector="VALUE",qaqc_selector="QAQC Flag",
                    parsedates=["DATETIME"],
                    indexcol=["DATETIME"],
                    skiprows=2,
                    dateparser=None,
                    comment = None,
                    extra_na=[""],
                    prefer_age="new",                    
                    tz_adj=hours(0))  
    return ts



data=pd.read_csv("data_full.csv",index_col=0,parse_dates=[0])["2008-01-01":"2018-12-01"].asfreq("15min")
cse_stage = read_des("anh_antioch_anh_stage_inst_2008_2019.csv")["2008-01-01":"2018-12-01"]
data["cse_stage"]= cse_stage.resample('15min').interpolate(limit=12)

log_ndo = np.log(data["outflow"])
reg = log_ndo.rolling(14*96).apply(regression_coef,raw=True)
fit = log_ndo.rolling(14*96).apply(regression_intercept,raw=True)

#cse_stage = read_des("c2b_collinsville_cse_stage_inst_20*.csv")
#cse_stage = cse_stage.asfreq('15min')
cse_stage = cosine_lanczos(data["cse_stage"],hours(40))
cse_mean14 = cse_stage.rolling(15*96).mean()

stage_diff = (cse_stage - cse_mean14).abs()
ndo_fit_diff = (log_ndo - fit).abs()

print(type(stage_diff))
print(type(ndo_fit_diff))

ok = (stage_diff  < 0.3) & ( ndo_fit_diff < 0.4) & (reg.abs() < 0.3)
#ok_flag = np.where(ok,5.,0.)

usept = cse_stage.copy()*0. + 5
usept.where(ok,0.,inplace=True)


fig,ax=plt.subplots(1)
reg.plot(ax=ax)
fit.plot(ax=ax)
(log_ndo).plot(ax=ax)
cse_stage.plot(ax=ax)
cse_mean14.plot(ax=ax)
#ok_flag.plot(ax=ax)
stage_diff.plot(ax=ax)
ndo_fit_diff.plot(ax=ax)
usept.plot(ax=ax)
ax.legend(["reg","fit","log_ndo","stage","stage(14)","stage_diff","ndo_fit_diff","usept"])
sampled = data.loc[ok,:]
print("Number of samples")
print((sampled.shape))
sampled.to_csv("sampled.csv")



ax.grid()

#data["outflow"].plot()
plt.show()