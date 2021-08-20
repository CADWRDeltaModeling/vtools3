# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 13:18:17 2021

""" 
all=["get_medianAD"]

from adtk.detector import ThresholdAD,PersistAD,SeasonalAD, AutoregressionAD,CustomizedDetector1D
from vtools.functions.error_detect import med_outliers

# This is a wrapper around the median_outliers function in vtools that converts the output to
# anomaly detection. This is the best detector I was able to construct, and it only takes one function.
def median_wrapper(ts,level=8,scale=None,filter_len=7,quantiles=(0.02,0.98)):
    test = med_outliers(ts,level=level,scale=scale,filt_len=filter_len,quantiles=quantiles)
    return test.isnull() & (~ts.isnull())



def get_medianAD(level=8,scale=None,filter_len=7,quantiles=(0.02,0.98)):
    detect_func_params={"level":level,"scale":scale,"filter_len":filter_len,"quantiles":quantiles}
    median_ad = CustomizedDetector1D(detect_func=median_wrapper, detect_func_params= detect_func_params)
    return median_ad