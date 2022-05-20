#!/usr/bin/env python
# -*- coding: utf-8 -*-
import yaml
import pandas as pd
from vtools.datastore.write_ts import *
from vtools.datastore.read_ts import *

def test_block_comment():
    teststr = \
    """
    This is
    a test"""
    print(block_comment(teststr))

def test_prep_header():
    header0 = {"format": "something else","unit":"feet"}
    out0 = prep_header(header0,format_version="dwr-dms-1.0")

    header1 = {"apple":"orange","unit":"feet","format": "something else"}
    out1 = prep_header(header1,format_version="dwr-dms-1.0")

    header2 = {"apple":"orange","unit":"feet"}
    out2 = prep_header(header2,format_version="dwr-dms-1.0")

    header3 = "# format : dwr-dms-1.0\n# unit : feet"
    out3 = prep_header(header3,format_version="dwr-dms-1.0")

    header4 = "# unit : feet\n# format : dwr-dms-1.0\n"
    out4 = prep_header(header4,format_version="dwr-dms-1.0")

    assert out1 == out2
    assert out3.strip() == out4.strip()

def max_diff(ts0,ts1):
    out = (ts0 - ts1).abs().max(axis=1).max()
    return out

def test_write_ts():
    import numpy as np
    nperiod=20000
    ndx = pd.date_range(start=pd.Timestamp(2009,2,10),periods=nperiod,freq='H',name="datetime")
    data = np.arange(2*nperiod).reshape(nperiod,2)
    ts = pd.DataFrame(index=ndx,data=data,columns=["value","other"],dtype=float)
    header = {"format": "something else","unit":"feet"}
    print(prep_header(header,format_version="dwr-dms-1.0"))
    # purpose of float_format is to test **kwargs
    write_ts_csv(ts,"test_write_ts_csv.csv",metadata=header,chunk_years = True,float_format="%.2f")
    ts1=read_ts("test_write_ts_csv*.csv")
    assert max_diff(ts,ts1) < 1e-15
    
if __name__ == "__main__":
    test_prep_header()
    test_write_ts()