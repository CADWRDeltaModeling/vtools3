import pandas as pd
import vtools.functions.filter as filter
import pytest

def test_godin():
    '''
    Test for godin filter by comparing with vtools answer
    The first timeseries is all zeros except for a single one
    The godin filtering of that series yields the coefficients that are being used
    This is compared with the vtools result based on the same.
    '''
    ts=pd.read_csv('godintest1.csv', parse_dates=True, index_col=0)
    ts.index.freq=ts.index.inferred_freq #FIXME: better way to do this on parse?
    tsg=filter.godin_filter(ts)
    tsg_vtools=pd.read_csv('godintest-vtools.csv', parse_dates=True, index_col=0)
    tsg_vtools.index.freq=tsg_vtools.index.inferred_freq
    pytest.approx(tsg_vtools['05JAN1990':'15FEB1990'].values,tsg['05JAN1990':'15FEB1990'].values)
