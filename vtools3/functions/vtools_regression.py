

import pandas as pd
from vtools.data.api import *
from vtools.functions.api import *
import matplotlib.pyplot as plt
import datetime as dtm

def df_vtools(df):
    stime = dtm.datetime(2000,1,1)
    interval = minutes(15)
    return rts(df.values,stime,interval)


def small_subtide(subtide_scale=0.,add_nan=False):
    """Inspired by large tidal flow with small Qr undercurrent with 72hr period
    This is a tough lowpass filtering job because the diurnal band is large and 
    must be supressed in order to see the more subtle subtidal amplitude"""
    freqmult=np.pi/180./3600.   # converts cycles/hour to rad/sec
    discharge_tide = {
    "O1": (13.943035*freqmult ,0.5*0.755 ,96),\
    "K1": (15.041069*freqmult,0.5*1.2,105.),\
    "M2": (28.984104*freqmult,0.75*1.89,336.),\
    "S2": (30.*freqmult,0.75*0.449,336.)}

    month_nsec = 30*86400
    t = np.arange(0,month_nsec,900)
    nsample = len(t)
    nanstart=nsample//3
    numnan = nsample//10
    FLOW_SCALE = 100000.
    tide = t*0.
    for key,(freq,amp,phase) in  discharge_tide.items():
        print(key,freq,amp,phase)
        tide+=FLOW_SCALE*amp*np.cos(freq*t-phase*np.pi/180.)
        
    
    subtide_freq = 2.*np.pi/(3.*86400.) #one cycle per 3 days
    # Add a subtide that is very small compared to the tidal amplitude
    tide += subtide_scale*FLOW_SCALE*np.cos(subtide_freq*t)

    tide[nanstart:(nanstart+numnan)] = np.nan
    dr = pd.date_range(start="2000-01-01",periods=nsample,freq="15min")
    return pd.DataFrame({"values":tide},dr)
    
    
    
if __name__=="__main__":
    vts=df_vtools(small_subtide(.03,True))
    
    vts_filt1 = cosine_lanczos(vts,hours(40))
    #vts_filt2 = butterworth(vts,4,hours(40))
    
    fig,ax0 = plt.subplots(1)
    ax0.plot(vts_filt1.times,vts_filt1.data)
    #ax0.plot(vts_filt2.times,vts_filt2.data)
    
    with open("vtools_cosine_lanczos_small_subtide_03_True_out.csv","w") as f:
        for el in vts_filt1:
            f.write("{},{}\n".format( el.time,el.value[0]))
            

    plt.show()






    