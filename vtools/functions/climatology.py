
""" 
Module to apply climatology data pattern of historical record to user desired date time index, 
one usage is to extroplate temperature time series to future.

# here is a exampe of usage
temp_file="temp_new.th" ## historical temperature csv file with header
temp_data=pd.read_csv(temp_file,header=0,parse_dates=True,index_col=0,sep="\s+")## 15min interval
tt=climatology(temp_data,"day",480) ## 5 day moving aveaged, daily climatology 
## extend to 2024 daily data
new_range=pd.date_range('2024-01-01', '2024-12-31', freq='D') 
ext_tt=apply_climatology(tt,new_range)

"""

import numpy as np
import pandas as pd


def climatology(ts,freq,nsmooth):

    """" Create a climatology on the columns of ts

        

        Parameters

        ----------

 

        ts: DataFrame or Series

        DataStructure to be analyzed. Must have a length of at least 2*freq

 

        freq: period ["day","month"]

        Period over which the climatology is analyzed

 

        nsmooth: int

           window size (number of values) of pre-smoothing. This may not make sense for series that are not approximately regular.

         
        Returns:

           out: DataFrame or Series

           Data structure of the same type as ts, with Integer index representing month (Jan=1) or day of year (1:365).
           
    """
    ts_mean=ts.rolling(nsmooth,center=True).mean() # moving average
    
    by=[ts_mean.index.month, ts_mean.index.day]
    if(freq=="month"):
        by=[ts_mean.index.month]
    elif not(freq=="day"):
        raise ValueError("invalid frequency, must be 'month' or 'day'") 
        
    
    mean_data=[]
    mean_data_size=[]
    for name, group in ts_mean.groupby(by): 
        if (len(by)==2):
            (mo, dy) = name
            if not((mo==2) and (dy==29)):
                mean_data.append(group.mean(axis=0))
                mean_data_size.append(group.count())
        else:
             mean_data.append(group.mean(axis=0))
             mean_data_size.append(group.count())
             
    climatology_data=pd.concat(mean_data,axis=1).transpose()
    return climatology_data
            
    
              

def apply_climatology(climate, index):

    """ Apply daily or monthly climatology to a new index 

    

         Parameters

           ----------

           climate: DataFrame with integer index representing month of year (Jan=1) or day of year. Must be of size 12 365,366. Day 366 will be inferred from day 365 value

          

           index: DatetimeIndex representing locations to be inferred

          

           Returns

           -------

           DataFrame or Series as given by climate with values extracted from climatology for the month or day
           
    
    """
    
    
    doy=index.dayofyear-1
    moy=index.month-1
    
    extend_data_dic={}
    extend_data_dic["time"]=index
    extend_data=pd.DataFrame(extend_data_dic)
    extend_data=extend_data.set_index("time")
    for sid in climate.columns:
        extend_data[sid]=[np.nan]*len(index)
    
    if(index.freqstr=="M"):
        extend_data.at[index]=climate.iloc[moy].values
    elif(index.freqstr=="D"):
        if(365 in doy):    ##fix leap year 366 point to 365
            new_data = pd.DataFrame(climate[-1:].values, index=[365], columns=climate.columns)
            ct=climate.append(new_data)
            extend_data.at[index]=ct.iloc[doy].values
        else:
            extend_data.at[index]=climate.iloc[doy].values
    else:
        print("Warning: index frequency is not supported, only daily or monthly supported")
    
    return extend_data


