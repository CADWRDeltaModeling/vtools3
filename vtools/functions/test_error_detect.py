
from vtools.data.api import minutes as vmin
from vtools.data.api import rts
import pandas as pd
import numpy as np
import datetime as dtm
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters


from vtools3.data.vtime import minutes as vmin3
from error_detect import med_outliers as med_outliers7
from vtools3.functions.error_detect3 import med_outliers as  med_filter3          



def test_med_filter():

    # sign wave wtih outliers 
    t = np.arange(0.,2000.)
    sin0 = np.cos(t*2.*np.pi/1000.)*6.
    sin1 = np.cos(t*2.*np.pi/25.)*2.
   
    np.random.seed(1)
    r = np.random.rand(2000,2)
    
    
    # Add some outliers
    r[400:402,:] += 19.
    r[800:803,:] += 2.5
    r[1200,:] += 4.
    r[1600,:] += 9
    
    signal = sin0+sin1
    signal = np.tile(signal[:,np.newaxis],2)
    signal +=r

    ts = rts(signal,dtm.datetime(2000,1,1),vmin(60))    
    ndx = pd.date_range(start="2001-01-01",freq="H",periods=2000)
    df = pd.DataFrame(signal,index=ndx)
    
    ts,filt = med_outliers7(ts,level=5,quantiles=(0.15,0.85))
    
    df = med_filter3(df,level=5,quantiles=(0.15,0.85))
    
    fig,ax = plt.subplots(1)
    ax.plot(df.index,df.values[:,1])
    ax.plot(df.index,ts.data[:,1]+0.4)
    plt.show()
    
    
if __name__=="__main__":
    test_med_filter()