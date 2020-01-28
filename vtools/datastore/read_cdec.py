import numpy as np
import pandas as pd
from .read_scalar import *
from vtools.data.vtime import hours, minutes, days

def read_des(fpat):
    start = dtm.datetime(2008, 1, 1)
    end = dtm.datetime(2019, 5, 1)
    mdir = "//cnrastore-bdo/Modeling_Data/des_emp/raw"

    ts = csv_retrieve_ts(fpat, mdir, start, end, selector="VALUE", qaqc_selector="QAQC Flag",
                         parsedates=["DATETIME"],
                         indexcol=["DATETIME"],
                         skiprows=2,
                         dateparser=None,
                         comment=None,
                         extra_na=[""],
                         prefer_age="new",
                         tz_adj=hours(0))
    return ts


def read_cdec(fpat):
    start = dtm.datetime(2008, 1, 1)
    end = dtm.datetime(2019, 5, 1)
    fdir = "."

    ts = csv_retrieve_ts(fpat, fdir, start, end, selector="VALUE",
                         qaqc_selector="DATA_FLAG",
                         qaqc_accept=['', ' ', ' ', 'e'],
                         parsedates=["DATE TIME"],
                         indexcol="DATE TIME",
                         skiprows=0,
                         dateparser=None,
                         comment=None,
                         prefer_age="new",
                         tz_adj=hours(0))
    return ts

def read_wdl(fpat):
    raise NotImplementedError()
    start = pd.Timestamp(2018,6,1)
    end  = pd.Timestamp(2018,12,1)
    fdir = "."
    ts = csv_retrieve_ts(fpat, fdir, start, end, selector="VALUE",
                         qaqc_selector="DATA_FLAG",
                         qaqc_accept=['', ' ', ' ', 'e'],
                         parsedates=["DATE TIME"],
                         indexcol="DATE TIME",
                         skiprows=0,
                         dateparser=None,
                         comment=None,
                         prefer_age="new",
                         tz_adj=hours(0))    
