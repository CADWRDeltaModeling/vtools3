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
        climate = climatology(df,freq)
        round_trip = apply_climatology(climate,datetime)

def _hourly_signal(index):
    """Deterministic function of (month, day, hour) only"""
    return ((index.month * 31 + index.day) * 24 + index.hour).astype(float)


def test_climatology_hourly_round_trip():
    # 2011 and 2013 are non-leap, 2012 is leap
    index = pd.date_range("2010-01-01", "2013-12-31 23:00", freq="h")
    df = pd.DataFrame({"a": _hourly_signal(index), "b": -_hourly_signal(index)}, index=index)
    clim = climatology(df, "hour")
    assert len(clim) == 8760
    assert clim.index.name == "hourofyear"
    assert clim.index[0] == 1 and clim.index[-1] == 8760
    assert not clim.isna().any().any()

    out = apply_climatology(clim, index)
    assert out.index.equals(index)
    # Feb 29 has no climatology of its own and takes Feb 28 values
    day = np.where((index.month == 2) & (index.day == 29), 28, index.day)
    expected = ((index.month * 31 + day) * 24 + index.hour).astype(float)
    np.testing.assert_allclose(out["a"].to_numpy(), expected)
    np.testing.assert_allclose(out["b"].to_numpy(), -expected)


def test_climatology_hourly_series_and_subhourly_input():
    index15 = pd.date_range("2010-01-01", "2012-12-31 23:45", freq="15min")
    ts = pd.Series(_hourly_signal(index15), index=index15, name="temp")
    clim = climatology(ts, "hour")
    assert isinstance(clim, pd.Series)
    assert clim.name == "temp"
    assert len(clim) == 8760
    # Hourly bins start on the hour, and the signal is constant within each hour of 15 min data
    t = pd.DatetimeIndex(["2011-06-15 12:00"])
    assert apply_climatology(clim, t).iloc[0] == pytest.approx((6 * 31 + 15) * 24 + 12)
    out = apply_climatology(clim, pd.date_range("2013-03-01", periods=48, freq="h"))
    assert isinstance(out, pd.Series)
    assert len(out) == 48


def test_climatology_hourly_smoothing_hours():
    index = pd.date_range("2010-01-01", "2012-12-31 23:00", freq="h")
    ts = pd.Series(0.0, index=index)
    # spike on the first hour of every non-leap year, so smoothing must wrap to the end of the year
    ts[(index.month == 1) & (index.day == 1) & (index.hour == 0)] = 3.0
    clim = climatology(ts, "hour", nsmooth=3)
    assert clim.loc[1] == pytest.approx(1.0)
    assert clim.loc[2] == pytest.approx(1.0)
    assert clim.loc[8760] == pytest.approx(1.0)
    assert clim.loc[3] == pytest.approx(0.0)
    assert clim.sum() == pytest.approx(3.0)


def test_climatology_hourly_pooling_days():
    index = pd.date_range("2010-01-01", "2012-12-31 23:00", freq="h")
    ts = pd.Series(0.0, index=index)
    ts[(index.month == 6) & (index.day == 15) & (index.hour == 9)] = 5.0
    clim = climatology(ts, "hour", nday=5)
    # same hour on +/- 2 days shares the value; other hours are unaffected
    pooled = apply_climatology(clim, pd.date_range("2011-06-13 09:00", periods=5, freq="D"))
    np.testing.assert_allclose(pooled.to_numpy(), 1.0)
    other = apply_climatology(clim, pd.DatetimeIndex(["2011-06-15 10:00"]))
    assert other.iloc[0] == 0.0
    assert clim.sum() == pytest.approx(5.0)


def test_climatology_hourly_gaps():
    # Data only covers part of the year: uncovered hours are NaN, covered hours are valid
    index = pd.date_range("2010-01-01", "2010-03-31 23:00", freq="h")
    ts = pd.Series(1.0, index=index)
    clim = climatology(ts, "hour", nsmooth=3, nday=3)
    assert clim.loc[1] == pytest.approx(1.0)
    assert clim.isna().sum() > 0
    assert clim.loc[4380:8000].isna().all()
    # smoothing wraps around the end of the year, so the last day borrows from Jan 1
    assert clim.loc[8760] == pytest.approx(1.0)


def test_climatology_hourly_errors():
    index = pd.date_range("2010-01-01", "2012-12-31 23:00", freq="h")
    ts = pd.Series(1.0, index=index)
    with pytest.raises(ValueError):
        climatology(ts, "hour", nsmooth=2)
    with pytest.raises(ValueError):
        climatology(ts, "hour", nday=4)
    with pytest.raises(ValueError):
        climatology(ts, "day", nday=3)
    daily = pd.Series(1.0, index=pd.date_range("2010-01-01", "2012-12-31", freq="D"))
    with pytest.raises(ValueError):
        climatology(daily, "hour")


def test_climatology_hourly_apply_floors_to_hour():
    index = pd.date_range("2010-01-01", "2012-12-31 23:00", freq="h")
    clim = climatology(pd.Series(_hourly_signal(index), index=index), "hour")
    # values vary hour to hour, so flooring is distinguishable from rounding
    fine = pd.date_range("2013-01-01", periods=8, freq="15min")
    out = apply_climatology(clim, fine)
    assert out.index.equals(fine)
    np.testing.assert_allclose(out.to_numpy(), apply_climatology(clim, fine.floor("h")).to_numpy())
    np.testing.assert_allclose(out.iloc[[0, 3, 4, 7]].to_numpy(), _hourly_signal(fine.floor("h"))[[0, 3, 4, 7]])
    assert out.iloc[3] != out.iloc[4]


def _smooth_hourly_clim():
    index = pd.date_range("2010-01-01", "2013-12-31 23:00", freq="h")
    ts = pd.Series(10.0 + 5.0 * np.sin(2 * np.pi * index.hour / 24), index=index, name="temp")
    return climatology(ts, "hour", nday=3)


def test_apply_climatology_substep_options_validated():
    clim = _smooth_hourly_clim()
    fine = pd.date_range("2014-06-01", periods=8, freq="15min")
    # default is fill
    pd.testing.assert_series_equal(apply_climatology(clim, fine), apply_climatology(clim, fine, substep_interp="fill"))
    with pytest.raises(ValueError, match="linear"):
        apply_climatology(clim, fine, substep_interp=None)
    with pytest.raises(ValueError, match="rhistinterp"):
        apply_climatology(clim, fine, substep_interp="cubic")
    with pytest.raises(ValueError, match="rhistinterp"):
        apply_climatology(clim, fine, substep_interp="linear", p=3.0)


def test_apply_climatology_linear_hourly():
    clim = _smooth_hourly_clim()
    fine = pd.date_range("2014-06-01", "2014-06-03", freq="15min")
    out = apply_climatology(clim, fine, substep_interp="linear")
    assert out.index.equals(fine)
    assert out.name == "temp"
    hourly = apply_climatology(clim, pd.date_range("2014-06-01", "2014-06-03", freq="h"))
    # hour values sit at the middle of their hour and linear is exact there
    mids = pd.date_range("2014-06-01 00:30", "2014-06-02 23:30", freq="h")
    np.testing.assert_allclose(
        apply_climatology(clim, mids, substep_interp="linear").to_numpy(),
        hourly.iloc[:-1].to_numpy(),
    )
    # between hour middles the value lies between its neighbors
    quarter = apply_climatology(clim, pd.DatetimeIndex(["2014-06-01 04:45"]), substep_interp="linear").iloc[0]
    v4, v5 = hourly.loc["2014-06-01 04:00"], hourly.loc["2014-06-01 05:00"]
    assert quarter == pytest.approx(0.75 * v4 + 0.25 * v5)


def test_apply_climatology_linear_monthly_and_daily():
    monthly = pd.DataFrame({"x": np.arange(1.0, 13.0)}, index=pd.RangeIndex(1, 13, name="month"))
    day_index = pd.date_range("2013-01-01", "2013-12-31", freq="D")
    out = apply_climatology(monthly, day_index, substep_interp="linear")
    assert out.index.equals(day_index)
    # the year edges interpolate against the neighboring (cyclic) December and January
    assert out["x"].iloc[0] == pytest.approx(6.5, abs=0.1)
    assert not out.isna().any().any()
    inner = out.loc["2013-01-17":"2013-12-15", "x"]
    assert inner.is_monotonic_increasing
    assert out["x"].between(1.0, 12.0).all()
    # Value mid-July (middle of the month) is the July value
    assert out.loc["2013-07-16", "x"] == pytest.approx(7.0, abs=0.1)

    daily = pd.DataFrame({"x": np.arange(1.0, 366.0)}, index=pd.RangeIndex(1, 366, name="dayofyear"))
    fine = pd.date_range("2013-03-01", "2013-03-03", freq="6h")
    out = apply_climatology(daily, fine, substep_interp="linear")
    assert out["x"].is_monotonic_increasing
    assert out["x"].iloc[0] == pytest.approx(59.5)


def test_apply_climatology_rhistinterp_conserves_monthly_means():
    values = np.array([5.0, 6.0, 8.0, 11.0, 15.0, 19.0, 22.0, 22.0, 19.0, 14.0, 9.0, 6.0])
    monthly = pd.Series(values, index=pd.RangeIndex(1, 13, name="month"), name="temp")
    index = pd.date_range("2013-01-01", "2013-12-31 23:00", freq="h")
    out = apply_climatology(monthly, index, substep_interp="rhistinterp")
    assert out.index.equals(index)
    assert out.name == "temp"
    assert not out.isna().any()
    np.testing.assert_allclose(out.groupby(out.index.month).mean().to_numpy(), values, rtol=0.02)
    # not a step function
    assert out.nunique() > 300


def test_apply_climatology_rhistinterp_kwargs_and_dataframe():
    monthly = pd.DataFrame(
        {"a": [1.0, 0.0, 0.0, 10.0, 0.0, 0.0, 5.0, 0.0, 0.0, 8.0, 0.0, 0.0], "b": np.arange(1.0, 13.0)},
        index=pd.RangeIndex(1, 13, name="month"),
    )
    index = pd.date_range("2013-01-01", "2013-12-31", freq="D")
    default = apply_climatology(monthly, index, substep_interp="rhistinterp")
    assert list(default.columns) == ["a", "b"]
    assert default.index.equals(index)
    tight = apply_climatology(monthly, index, substep_interp="rhistinterp", p=20.0, lowbound=0.0)
    assert (tight["a"] >= -1e-6).all()
    assert not np.allclose(default["a"], tight["a"])


def test_apply_climatology_rhistinterp_hourly_and_daily():
    clim = _smooth_hourly_clim()
    fine = pd.date_range("2014-06-01", "2014-06-04 23:45", freq="15min")
    out = apply_climatology(clim, fine, substep_interp="rhistinterp")
    hourly = apply_climatology(clim, pd.date_range("2014-06-01", "2014-06-04 23:00", freq="h"))
    np.testing.assert_allclose(
        out.groupby(out.index.floor("h")).mean().to_numpy(), hourly.to_numpy(), rtol=0.02
    )

    index = pd.date_range("2010-01-01", "2013-12-31 23:00", freq="h")
    ts = pd.DataFrame({"x": np.cos(2 * np.pi * index.dayofyear / 365) + 3.0}, index=index)
    daily = climatology(ts, "day")
    dfine = apply_climatology(daily, pd.date_range("2014-03-01", "2014-03-10 23:00", freq="h"), substep_interp="rhistinterp")
    dday = apply_climatology(daily, pd.date_range("2014-03-01", "2014-03-10", freq="D"))
    np.testing.assert_allclose(dfine.groupby(dfine.index.floor("D")).mean().to_numpy(), dday.to_numpy(), rtol=0.01)


def test_apply_climatology_rhistinterp_rejects_missing():
    index = pd.date_range("2010-01-01", "2010-03-31 23:00", freq="h")
    clim = climatology(pd.Series(1.0, index=index), "hour")
    with pytest.raises(ValueError):
        apply_climatology(clim, pd.date_range("2014-01-01", periods=96, freq="15min"), substep_interp="rhistinterp")
    # linear and fill do not apply that restriction
    apply_climatology(clim, pd.date_range("2014-01-01", periods=96, freq="15min"), substep_interp="fill")


if __name__ == "__main__":
    test_climatology_round_trip()

 