import pandas as pd
import numpy as np
import pytest
from vtools.data.gap import *

@pytest.fixture
def gappy_reg_df():
    ndx = pd.date_range(pd.Timestamp(2017, 1, 1, 12), freq='15min', periods=10)
    vals0 = np.arange(0., 10., dtype='d')
    vals1 = vals0.copy()
    vals2 = vals0.copy()
    vals0[0:3] = np.nan
    vals0[7:-1] = np.nan
    vals1[2:4] = np.nan
    vals1[6] = np.nan
    vals1[9] = np.nan
    df = pd.DataFrame({'vals0': vals0, 'vals1': vals1, 'vals2': vals2}, index=ndx)
    return df    
    
@pytest.fixture
def gappy_irreg_df(gappy_reg_df):
    irreg_index = pd.DatetimeIndex([
        "2017-01-01T12:00",
        "2017-01-01T01:30",
        "2017-01-01T1:45",
        "2017-01-01T02:45",
        "2017-01-01T01:30",
        "2017-01-01T5:30",
        "2017-01-01T05:45",
        "2017-01-01T00:45",
        "2017-01-02T01:45",
        "2017-01-01T02:00"
    ])
    gappy_irr = gappy_reg_df.copy()
    gappy_irr.index = irreg_index
    return gappy_irr
                                   
    
def test_gap_count(gappy_reg_df):
    out = gap_count(gappy_reg_df)
    # Use .iloc for positional indexing.
    assert out['vals0'].iloc[0] == 3
    assert out['vals0'].iloc[-1] == 0
    assert out['vals1'].iloc[0] == 0 
    assert out['vals1'].iloc[1] == 0
    assert out['vals1'].iloc[-2] == 0
    assert out['vals1'].iloc[-1] == 1
    assert (out['vals2'] == 0).all()
    
def test_gap_count_good(gappy_reg_df):
    out = gap_count(gappy_reg_df, state="good")
    assert out['vals0'].iloc[0] == 0
    assert out['vals0'].iloc[2] == 0    
    assert out['vals0'].iloc[-1] == 1
    assert out['vals0'].iloc[3] == 4
    assert out['vals1'].iloc[0] == 2 
    assert out['vals1'].iloc[1] == 2
    assert out['vals1'].iloc[-2] == 2
    assert out['vals1'].iloc[-1] == 0
    assert (out['vals2'] == 10).all()
    
def test_gap_count_reg_irreg_agree(gappy_reg_df, gappy_irreg_df):
    """Gap counts shouldn't depend on the time index."""
    assert np.all(np.equal(gap_count(gappy_reg_df).to_numpy(),
                           gap_count(gappy_irreg_df).to_numpy()))
