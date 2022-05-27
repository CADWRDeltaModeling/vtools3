#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import pytest


from vtools.functions.climatology import *
import matplotlib.pyplot as plt


def test_climatology_daily():
    
    data_time=pd.date_range("2007-01-20","2012-03-21",freq="15min")
    temp_dic={}
    temp_dic["time"]=data_time
    for staion in ["a1","a2","a3","a4","a5","a6"]:
        temp_dic[staion]=np.random.random_sample(size=len(data_time))*25.0
    temp_data=pd.DataFrame(temp_dic)
    temp_data=temp_data.set_index("time")
    
    tt=climatology(temp_data,"day",480) ##  daily climatology 480 for 5 dya moving average of raw data
    assert(len(tt)==365)
    ## extend to 2024 daily data
    new_range=pd.date_range('2024-01-01', '2024-12-31', freq='D') 
    ext_tt=apply_climatology(tt,new_range)
    
    tt=climatology(temp_data,"month",480) ##  daily climatology 480 for 5 dya moving average of raw data
    assert(len(tt)==12)
    
    with pytest.raises(ValueError):
        tt = climatology(temp_data,"year",480) 

def test_climatology_round_trip():
    """Tests round trip of series whose climatology matches the index"""
    for freq in ["day","month"]:  
        datetime=pd.date_range("2009-01-01","2012-02-21",freq="15min")
        if freq=="month": 
            data = datetime.month.astype(float)
        else: data = datetime.dayofyear.astype(float)
        df = pd.DataFrame(index=datetime,data=data)
        df.to_csv("testc.csv")
        climate = climatology(df,freq)
        print("calling")
        round_trip = apply_climatology(climate,datetime)

        print("round")
        print(round_trip)
        #climate = climatology(df,"day")
        print(climate)
    
    
    

    

if __name__ == "__main__":
    test_climatology_round_trip()

 