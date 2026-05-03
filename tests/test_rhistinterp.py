import sys
import types

import numpy as np
import pandas as pd

# Stub vtools dependency required by legacy interpolate.py import.
vtools_mod = types.ModuleType("vtools")
functions_mod = types.ModuleType("vtools.functions")
mono_mod = types.ModuleType("vtools.functions._monotonic_spline")
mono_mod._monotonic_spline = lambda x, y, xnew: np.interp(xnew, x, y)
sys.modules.setdefault("vtools", vtools_mod)
sys.modules.setdefault("vtools.functions", functions_mod)
sys.modules.setdefault("vtools.functions._monotonic_spline", mono_mod)

from vtools.functions.rhistinterp import find_runs, rhistinterp
from vtools.functions.interpolate import rhistinterp as legacy_rhistinterp


def test_find_runs_empty():
    mask = np.array([], dtype=bool)
    assert find_runs(mask) == []


def test_find_runs():
    mask = np.array([False, True, True, False, True, False, True, True, True])
    assert find_runs(mask) == [(1, 3), (4, 5), (6, 9)]


def test_rhistinterp_thresh_none_matches_legacy_series():
    ndx = pd.period_range(start='2001-01-01', periods=6, freq='M')
    ts = pd.Series([1.0, 3.0, 2.5, 4.0, 3.5, 2.0], index=ndx, name='x')
    dest = pd.date_range(start=ndx[0].start_time, end=ndx[-1].end_time.round('s').floor('D'), freq='D')

    got = rhistinterp(ts, dest, p=1.5, lowbound=0.0, thresh=None)
    want = legacy_rhistinterp(ts, dest, p=1.5, lowbound=0.0)

    pd.testing.assert_series_equal(got, want)


def test_rhistinterp_protected_middle_interval_is_constant():
    ndx = pd.period_range(start='2001-01-01', periods=3, freq='M')
    ts = pd.Series([1.0, 0.02, 1.0], index=ndx, name='x')
    dest = pd.date_range(start=ndx[0].start_time, end=ndx[-1].end_time.round('s').floor('D'), freq='D')

    got = rhistinterp(ts, dest, p=1.5, lowbound=0.0, thresh=0.05)

    left = ndx[1].start_time
    right = ndx[1].end_time.round('s')
    mask = (got.index >= left) & (got.index < right)
    vals = got.loc[mask].to_numpy()
    assert vals.size > 0
    assert np.allclose(vals, 0.02)
    assert np.nanmin(got.to_numpy()) >= 0.0
