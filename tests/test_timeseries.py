from vtools.data.timeseries import *
import pandas as pd
import numpy as np
import pytest

def test_rts():
    arr = np.array([[1.,2.],[3.,4.],[5.,6.]])
    start = pd.Timestamp(2009,2,10)
    ts = rts(arr,start,"15min")
    assert len(ts) == 3
    assert ts.iloc[2,0] == 5.
    assert ts.loc[pd.Timestamp(2009,2,10,0),1] == 2. 
    assert ts.columns[1] == 1
    arr = np.array([[1.,2.],[3.,4.],[5.,6.]])
    start = pd.Timestamp(2009,2,10)
    ts = rts(arr,start,"15min",columns=["v1","v2"])
    assert len(ts) == 3
    assert ts.iloc[2,0] == 5.
    assert ts.loc[pd.Timestamp(2009,2,10,0),"v2"] == 2. 
    assert ts.columns[1] == "v2"
    with pytest.raises(NotImplementedError):
        ts = rts(arr,start,"15min",columns=["v1","v2"],props={"units":"cfs"})

def test_rts_formula():
    valfunc={"sin": lambda x: np.sin(2.*np.pi*x/86400.), "cos": lambda x: np.cos(2.*np.pi*x/86400.)}
    start = pd.Timestamp(2009,2,10)
    end = pd.Timestamp(2009,2,16)
    freq="H"
    ts = rts_formula(start,end,freq,valfunc)
    test_stamp = pd.Timestamp(2009,2,11)
    assert np.isclose(ts.loc[test_stamp,"cos"], 1.0)
    
    valfunc1 = 0.
    ts1 = rts_formula(start,end,freq,valfunc1)
    assert ts1.loc[test_stamp,"value"] == 0.
    
def test_extrapolate_ts():
    valfunc={"sin": lambda x: np.sin(2.*np.pi*x/86400.), "cos": lambda x: np.cos(2.*np.pi*x/86400.)}
    start = pd.Timestamp(2009,2,10)
    end = pd.Timestamp(2009,2,16)
    freq="H"
    ts = rts_formula(start,end,freq,valfunc)    
    new_start = pd.Timestamp(2009,2,9)
    ndxnew = pd.date_range(start=new_start,end=ts.index[-1],freq=ts.index.freq)
    ts=ts.reindex(ndxnew,axis=0)
    ts.iloc[0,:]=0.
    front = ts.loc[new_start:start,:].interpolate()
    ts.loc[new_start:start,:] = front 
    print("hello")
    print(ts)
    
    
    
if __name__ == "__main__":
    test_rts()
    test_rts_formula()
    test_extrapolate_ts()