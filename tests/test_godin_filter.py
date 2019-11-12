import os
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
    fname_input = os.path.join(os.path.dirname(__file__),
                               'test_data/godintest1.csv')
    ts = pd.read_csv(fname_input, parse_dates=True, index_col=0)
    # FIXME: better way to do this on parse?
    ts.index.freq = ts.index.inferred_freq
    tsg = filter.godin_filter(ts)
    fname_expected = os.path.join(os.path.dirname(__file__),
                                  'test_data/godintest-vtools.csv')
    tsg_vtools = pd.read_csv(fname_expected, parse_dates=True, index_col=0)
    tsg_vtools.index.freq = tsg_vtools.index.inferred_freq
    pytest.approx(tsg_vtools['05JAN1990':'15FEB1990'].values,
                  tsg['05JAN1990':'15FEB1990'].values)
