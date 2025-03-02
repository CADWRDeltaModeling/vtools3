#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import pytest

from numpy.testing import (
    assert_array_almost_equal,
    assert_array_equal,
    assert_almost_equal,
)
from numpy import pi
from vtools.data.timeseries import rts
from vtools.data.vtime import minutes, hours, days
from vtools.data.sample_series import *
from vtools.functions.filter import (
    butterworth,
    godin,
    cosine_lanczos,
    ts_gaussian_filter,
    lowpass_cosine_lanczos_filter_coef,
)

# Fixture for common filtering parameters.
@pytest.fixture
def filter_params():
    return {
        "num_ts": 1000,
        "two_to_ten": 2**10,
    }

def test_butterworth(filter_params):
    """Test Butterworth filter on a 1-hour interval series with four frequencies."""
    st = pd.Timestamp(1990, 2, 3, 11, 15)
    num = filter_params["two_to_ten"]
    delta = hours(1)
    f1, f2, f3, f4 = 0.76, 0.44, 0.95, 1.23
    pi = np.pi
    av1 = f1 * pi / 12.0
    av2 = f2 * pi / 12.0
    av3 = f3 * pi / 12.0
    av4 = f4 * pi / 12.0
    data = [np.sin(av1 * k) + 0.7 * np.cos(av2 * k) +
            2.4 * np.sin(av3 * k) + 0.1 * np.sin(av4 * k)
            for k in np.arange(num)]
    ts0 = rts(data, st, delta)
    ts_filt = butterworth(ts0, cutoff_period=hours(40))
    assert ts_filt.index.freq == ts0.index.freq

def test_butterworth_noevenorder():
    """Test that a Butterworth filter with non-even order raises a ValueError."""
    start = pd.Timestamp(2000, 2, 3)
    freq = hours(1)
    data = np.arange(100)
    ts0 = rts(data, start, freq)
    with pytest.raises(ValueError):
        butterworth(ts0, order=7)

def test_godin(filter_params):
    """Test Godin filter on a 1-hour interval series with four frequencies."""
    st = pd.Timestamp(1990, 2, 3, 11, 15)
    num = filter_params["two_to_ten"]
    delta = hours(1)
    f1, f2, f3, f4 = 0.76, 0.44, 0.95, 1.23
    av1 = f1 * np.pi / 12
    av2 = f2 * np.pi / 12
    av3 = f3 * np.pi / 12
    av4 = f4 * np.pi / 12
    data = [np.sin(av1 * k) + 0.7 * np.cos(av2 * k) +
            2.4 * np.sin(av3 * k) + 0.1 * np.sin(av4 * k)
            for k in range(num)]
    ts0 = rts(data, st, delta)
    ts_filt = godin(ts0)
    assert ts_filt.index.freq == ts0.index.freq

def test_godin_15min():
    """Test Godin filtering on a 15-minute series with a NaN."""
    data = [1.0] * 800 + [2.0] * 400 + [1.0] * 400
    data = np.array(data)
    data[336] = np.nan
    st = pd.Timestamp(1990, 2, 3, 11, 15)
    delta = minutes(15)
    ts = rts(data, st, delta)
    nt3 = godin(ts)
    npout = nt3.to_numpy().ravel()
    assert np.all(np.isnan(npout[0:144]))
    assert_array_almost_equal(npout[144:192], [1] * 48, decimal=12)
    assert np.all(np.isnan(npout[192:481]))
    assert_array_almost_equal(npout[481:656], [1.] * 175, decimal=12)
    assert np.all(np.greater(nt3.to_numpy()[656:944], 1))
    assert np.isclose(npout[868], 1.916618441)
    assert_array_almost_equal(npout[944:1056], [2.] * 112, decimal=12)
    assert np.all(np.greater(npout[1056:1344], 1))
    assert np.isclose(npout[1284], 1.041451845)
    assert_array_almost_equal(npout[1344:1456], [1.] * 112, decimal=12)
    assert np.all(np.isnan(npout[1456:1600]))

def test_godin_2d():
    """Test Godin filter on a 2-dimensional dataset."""
    st = pd.Timestamp(1990, 2, 3, 11, 15)
    ndx = pd.date_range(start=st, freq='15min', periods=1600)
    d1 = [1.0] * 800 + [2.0] * 400 + [1.0] * 400
    d2 = [1.0] * 800 + [2.0] * 400 + [1.0] * 400
    df = pd.DataFrame({"x": d1, "y": d2}, index=ndx)
    df.iloc[336, :] = np.nan
    nt3 = godin(df)
    arr = nt3.to_numpy()
    d1_out = arr[:, 0]
    d2_out = arr[:, 1]
    assert_array_almost_equal(d1_out[144:192], [1] * 48, decimal=12)
    assert np.all(np.isnan(d1_out[192:481]))
    assert_array_almost_equal(d1_out[481:656], [1] * 175, decimal=12)
    assert np.all(np.greater(d1_out[656:944], 1))
    assert np.isclose(d1_out[868], 1.916618441)
    assert_array_almost_equal(d1_out[944:1056], [2] * 112, decimal=12)
    assert np.all(np.greater(d1_out[1056:1344], 1))
    assert np.isclose(d1_out[1284], 1.041451845)
    assert_array_almost_equal(d1_out[1344:1456], [1] * 112, decimal=12)
    assert np.all(np.isnan(d1_out[1456:1600]))
    assert_array_almost_equal(d2_out[144:192], [1] * 48, decimal=12)
    assert np.all(np.isnan(d2_out[192:481]))
    assert_array_almost_equal(d2_out[481:656], [1] * 175, decimal=12)
    assert np.all(np.greater(d2_out[656:944], 1))
    assert np.isclose(d2_out[868], 1.916618441)
    assert_array_almost_equal(d2_out[944:1056], [2] * 112, decimal=12)
    assert np.all(np.greater(d2_out[1056:1344], 1))
    assert np.isclose(d2_out[1284], 1.041451845)
    assert_array_almost_equal(d2_out[1344:1456], [1] * 112, decimal=12)
    assert np.all(np.isnan(d2_out[1456:1600]))

def test_godin_csv():
    """
    Test Godin filter against expected results from CSV files.
    The CSV files 'godintest1.csv' and 'godintest-vtools.csv' contain the input
    and expected output (from vtools) respectively.
    """
    fname_input = os.path.join(os.path.dirname(__file__), 'test_data/godintest1.csv')
    ts = pd.read_csv(fname_input, parse_dates=True, index_col=0)
    ts.index.freq = ts.index.inferred_freq
    tsg = godin(ts)
    fname_expected = os.path.join(os.path.dirname(__file__), 'test_data/godintest-vtools.csv')
    tsg_vtools = pd.read_csv(fname_expected, parse_dates=True, index_col=0)
    tsg_vtools.index.freq = tsg_vtools.index.inferred_freq
    assert_array_almost_equal(
        tsg_vtools.loc['05JAN1990':'15FEB1990'].values,
        tsg.loc['05JAN1990':'15FEB1990'].values
    )

def test_lanczos_cos_filter_coef():
    """Test the sum of cosine Lanczos filter coefficients."""
    cf = 0.2
    m = 10
    coef = lowpass_cosine_lanczos_filter_coef(cf, m, False)
    coef = np.array(coef)
    coefsum = np.sum(coef)
    assert_almost_equal(np.abs(1.0 - coefsum), 0.0, decimal=1)
    m = 40
    coef = lowpass_cosine_lanczos_filter_coef(cf, m, False)
    coef = np.array(coef)
    coefsum = np.sum(coef)
    assert_almost_equal(np.abs(1.0 - coefsum), 0.0, decimal=3)

def test_lanczos_cos_filter_phase_neutral():
    """Test the phase neutrality of the cosine Lanczos filter."""
    t = np.linspace(0, 2000, 2001)
    xlow = np.cos(2 * np.pi * t / (4. * 24))
    xhigh = np.cos(2 * np.pi * t / 6.)
    st = pd.Timestamp(1990, 2, 3, 11, 15)
    delta = hours(1)
    ts = rts(xlow + xhigh, st, delta)
    nt1 = cosine_lanczos(ts, cutoff_period=hours(30), filter_len=200)
    nt1 = nt1.rename(columns={nt1.columns[0]: "nt1"})
    nt1["nt2"] = xlow
    absdiff = (nt1["nt1"] - nt1["nt2"]).abs().max()
    assert_almost_equal(absdiff, 0, decimal=1)

def test_lanczos_cos_filter_period_freq_api():
    """Test the cutoff period and frequency API of the cosine Lanczos filter."""
    t = np.linspace(0, 1.0, 2001)
    xlow = np.sin(2 * np.pi * 5 * t)
    xhigh = np.sin(2 * np.pi * 250 * t)
    st = pd.Timestamp(1990, 2, 3, 11, 15)
    delta = hours(1)
    ts = rts(xlow + xhigh, st, delta)
    nt1 = cosine_lanczos(ts, cutoff_period=hours(30), filter_len=20, padtype="even")
    cutoff_frequency = 2.0 / 30
    nt2 = cosine_lanczos(ts, cutoff_frequency=cutoff_frequency, filter_len=20, padtype="even")
    assert np.abs(nt1.to_numpy() - nt2.to_numpy()).max() == 0

def test_lanczos_cos_filter_len_api():
    """Test the filter length API of the cosine Lanczos filter."""
    t = np.linspace(0, 1.0, 2001)
    xlow = np.sin(2 * np.pi * 5 * t)
    xhigh = np.sin(2 * np.pi * 250 * t)
    st = pd.Timestamp(1990, 2, 3, 11, 15)
    delta = hours(1)
    ts = rts(xlow + xhigh, st, delta)
    nt1 = cosine_lanczos(ts, cutoff_period=hours(40), padtype="even")
    assert nt1.index.freq == ts.index.freq
    nt2 = cosine_lanczos(ts, cutoff_period=hours(40), filter_len=50, padtype="even")
    assert np.abs(nt1.to_numpy() - nt2.to_numpy()).max() == 0

def test_lanczos_cos_filter_len():
    """Test the cosine Lanczos filter length parameter."""
    data = [
        2.0 * np.cos(2 * pi * i / 5 + 0.8) +
        3.0 * np.cos(2 * pi * i / 45 + 0.1) +
        7.0 * np.cos(2 * pi * i / 55 + 0.3)
        for i in range(1000)
    ]
    data = np.array(data)
    st = pd.Timestamp(1990, 2, 3, 11, 15)
    delta = hours(1)
    ts = rts(data, st, delta)
    t1 = cosine_lanczos(ts, cutoff_period=hours(30), filter_len=24)
    t2 = cosine_lanczos(ts, cutoff_period=hours(30), filter_len=days(1))
    assert_array_equal(t1.to_numpy(), t2.to_numpy())
    with pytest.raises(TypeError):
        cosine_lanczos(ts, cutoff_period=hours(30), filter_len="invalid")

def test_lanczos_cos_filter_nan():
    """Test that the cosine Lanczos filter handles NaN values correctly."""
    data = [
        2.0 * np.cos(2 * pi * i / 5 + 0.8) +
        3.0 * np.cos(2 * pi * i / 45 + 0.1) +
        7.0 * np.cos(2 * pi * i / 55 + 0.3)
        for i in range(1000)
    ]
    data = np.array(data)
    nanloc = 336
    data[nanloc] = np.nan
    st = pd.Timestamp(1990, 2, 3, 11, 15)
    delta = hours(1)
    ts = rts(data, st, delta)
    m = 20
    nt1 = cosine_lanczos(ts, cutoff_period=hours(30), filter_len=m, padtype="even")
    nanidx = np.where(np.isnan(nt1.to_numpy()))[0]
    nanidx_should_be = np.arange(nanloc - 2 * m, nanloc + 2 * m + 1)
    assert_array_equal(nanidx, nanidx_should_be)

def test_gaussian_filter():
    """Test the Gaussian filter on data with a NaN for various orders."""
    data = [
        2.0 * np.cos(2 * pi * i / 5 + 0.8) +
        3.0 * np.cos(2 * pi * i / 45 + 0.1) +
        7.0 * np.cos(2 * pi * i / 55 + 0.3)
        for i in range(1000)
    ]
    data = np.array(data)
    nanloc = 336
    data[nanloc] = np.nan
    st = pd.Timestamp(1990, 2, 3, 11, 15)
    delta = hours(1)
    ts = rts(data, st, delta)
    sigma = 2
    for order in range(4):
        filtered = ts_gaussian_filter(ts, sigma, order=order)
        assert filtered is not None

if __name__ == "__main__":
    pytest.main([__file__])
