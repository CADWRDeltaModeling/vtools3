"""Conservative period-to-instantaneous interpolation with optional threshold protection.

This module provides a new implementation of ``rhistinterp`` that can reproduce the
legacy behavior exactly when ``thresh=None`` and can switch to a hybrid mode when a
threshold is supplied. In hybrid mode, source intervals with values at or below the
threshold are reconstructed as constant plateaus, while contiguous runs of remaining
intervals are interpolated conservatively.
"""

from __future__ import annotations

import numbers
from typing import Iterable

import numpy as np
import pandas as pd

__all__ = ["rhistinterp"]


def _load_legacy_functions():
    from vtools.functions.interpolate import rhistinterp as legacy_rhistinterp
    from vtools.functions.interpolate import rhist_bound as legacy_rhist_bound

    return legacy_rhistinterp, legacy_rhist_bound



def classify_input_versus_threshold(y, lowbound, thresh):
    """Classify source intervals relative to a threshold.

    Parameters
    ----------
    y : ndarray, shape (n_intervals,)
        Period-averaged values for each source interval.
    lowbound : float
        Lower bound enforced by the interpolation.
    thresh : float
        Threshold above ``lowbound``. Any interval with value less than or equal
        to ``thresh`` is treated as protected and reconstructed as a constant
        plateau.

    Returns
    -------
    protected : ndarray of bool, shape (n_intervals,)
        Boolean mask where True indicates a protected interval.
    """
    if lowbound is None:
        raise ValueError("lowbound must be provided when thresh is not None")
    if thresh is None:
        raise ValueError("thresh must not be None in classify_input_versus_threshold")
    if thresh <= lowbound:
        raise ValueError("thresh must be strictly greater than lowbound")
    y = np.asarray(y, dtype=float)
    return y <= thresh



def find_runs(mask):
    """Identify contiguous runs of True values in a boolean mask.

    Parameters
    ----------
    mask : ndarray of bool, shape (n,)
        Boolean array.

    Returns
    -------
    runs : list of tuple[int, int]
        List of ``(start, stop)`` index pairs for contiguous True regions. Each
        run includes indices ``[start, stop)``.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError("mask must be one-dimensional")
    if mask.size == 0:
        return []

    padded = np.concatenate(([False], mask, [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    starts = changes[0::2]
    stops = changes[1::2]
    return list(zip(starts.tolist(), stops.tolist()))



def select_destination_points_in_interval(xnew, left, right, include_right=False):
    """Select destination points belonging to a source interval.

    Parameters
    ----------
    xnew : ndarray
        Destination coordinates.
    left : float
        Left boundary of the interval.
    right : float
        Right boundary of the interval.
    include_right : bool, optional
        If True, include points exactly equal to ``right``.

    Returns
    -------
    mask : ndarray of bool
        Boolean mask selecting destination points in the interval.
    """
    xnew = np.asarray(xnew, dtype=float)
    if include_right:
        return (xnew >= left) & (xnew <= right)
    return (xnew >= left) & (xnew < right)




def _interp_single_interval(x0, x1, y, y0, y1, xnew, lowbound=None):
    """
    Conservative quadratic interpolant over a single interval.
    """
    dx = x1 - x0
    t = (xnew - x0) / dx

    a = 3.0 * (y1 + y0 - 2.0 * y)
    b = (y1 - y0) - a
    c = y0

    out = a * t * t + b * t + c


    return out

def _build_source_coordinates(ts, dest):
    """Build source and destination coordinates for interpolation."""
    try:
        ndx_left = ts.index.to_timestamp(how="s")
    except AttributeError as exc:
        raise ValueError("Time series to interpolate must have PeriodIndex") from exc
    ndx_right = ts.index.to_timestamp(how="e").round("s")
    ndx = ndx_left.union(ndx_right)

    strt = ndx[0]
    x = (ndx - strt).total_seconds().to_numpy(dtype=float)

    if not isinstance(dest, pd.Index):
        end = ndx[-1].floor(dest)
        dest = pd.date_range(start=strt, end=end, freq=dest)
    xnew = (dest - strt).total_seconds().to_numpy(dtype=float)
    return dest, x, xnew



def _validate_1d_input(y):
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("input must be one-dimensional")
    if np.isnan(y).any():
        raise ValueError("input contains NaN values")
    return y



def _normalize_p_for_run(p, run_start, run_stop, n_intervals):
    if isinstance(p, numbers.Real):
        return float(p)
    parr = np.asarray(p, dtype=float)
    if parr.ndim != 1 or len(parr) != n_intervals:
        raise ValueError("array-valued p must have one value per source interval")
    return parr[run_start:run_stop].copy()



def _single_interval_conservative_values(x_run, ybar, xnew_run, y0, yn):
    """Quadratic reconstruction for a single interval.

    The quadratic satisfies the endpoint values and the exact interval mean.
    """
    left = x_run[0]
    width = x_run[1] - x_run[0]
    if width <= 0.0:
        raise ValueError("source interval width must be positive")
    t = (xnew_run - left) / width
    a = 3.0 * y0 - 6.0 * ybar + 3.0 * yn
    b = -4.0 * y0 + 6.0 * ybar - 2.0 * yn
    c = y0
    return a * t * t + b * t + c



def _interpolate_active_run(
    x,
    y,
    xnew,
    run_start,
    run_stop,
    p,
    lowbound,
    tolbound,
    maxiter,
):
    """Interpolate a contiguous active run using conservative reconstruction."""
    span_left = x[run_start]
    span_right = x[run_stop]
    include_right = run_stop == len(y)
    mask_run = select_destination_points_in_interval(
        xnew, span_left, span_right, include_right=include_right
    )
    xnew_run = xnew[mask_run]
    x_run = x[run_start : run_stop + 1]
    y_run = y[run_start:run_stop]

    if run_start > 0:
        y0 = y[run_start - 1]
    else:
        y0 = y[run_start]

    if run_stop < len(y):
        yn = y[run_stop]
    else:
        yn = y[run_stop - 1]

    nrun = len(y_run)

    if nrun == 1:
        vals = _single_interval_conservative_values(
            x_run, y_run[0], xnew_run, y0, yn
        )
        if lowbound is not None:
            vals = np.maximum(vals, lowbound)
        return mask_run, vals

    if nrun == 2:
        vals = _two_interval_conservative_values(
            x_run, y_run, xnew_run, y0, yn
        )
        if lowbound is not None:
            vals = np.maximum(vals, lowbound)
        return mask_run, vals

    p_run = _normalize_p_for_run(p, run_start, run_stop, len(y))
    vals = _load_legacy_functions()[1](
        x_run,
        y_run,
        xnew_run,
        y0=y0,
        yn=yn,
        p=p_run,
        lbound=lowbound,
        maxiter=maxiter,
        floor_eps=tolbound,
    )
    return mask_run, vals

def _two_interval_conservative_values(x_run, y_run, xnew_run, y0, y2):
    """
    Conservative cubic reconstruction over two intervals.

    Enforces:
      - endpoint values
      - exact mean on each interval
      - smooth shape (no spike)
    """
    x0, x1, x2 = x_run
    y0_bar, y1_bar = y_run

    L = x2 - x0
    t = (xnew_run - x0) / L

    # Solve coefficients
    d = y0

    # Precompute integrals
    # ∫ t³ dt = t⁴/4
    # ∫ t² dt = t³/3
    # ∫ t dt = t²/2

    def I(a, b, c, d, t):
        return a * t**4 / 4 + b * t**3 / 3 + c * t**2 / 2 + d * t

    # Build linear system for a, b, c
    # unknowns: a, b, c

    # Constraint 2
    # a + b + c + d = y2
    eq1 = [1, 1, 1]

    rhs1 = y2 - d

    # Constraint 3
    # I(0.5) - I(0) = 0.5 * y0_bar
    t0 = 0.5
    eq2 = [
        t0**4 / 4,
        t0**3 / 3,
        t0**2 / 2,
    ]
    rhs2 = 0.5 * y0_bar - d * t0

    # Constraint 4
    # I(1) - I(0.5) = 0.5 * y1_bar
    I1 = [1/4, 1/3, 1/2]
    I05 = [
        t0**4 / 4,
        t0**3 / 3,
        t0**2 / 2,
    ]
    eq3 = [
        I1[0] - I05[0],
        I1[1] - I05[1],
        I1[2] - I05[2],
    ]
    rhs3 = 0.5 * y1_bar - d * (1 - t0)

    A = np.array([eq1, eq2, eq3], dtype=float)
    bvec = np.array([rhs1, rhs2, rhs3], dtype=float)

    a, b, c = np.linalg.solve(A, bvec)

    vals = a * t**3 + b * t**2 + c * t + d
    return vals

def _rhistinterp_1d(x, y, xnew, p, lowbound, thresh, tolbound, maxiter):
    """Hybrid conservative interpolation for a single series."""
    y = _validate_1d_input(y)

    if thresh is None:
        return _load_legacy_functions()[1](
            x,
            y,
            xnew,
            y0=y[0],
            yn=y[-1],
            p=p,
            lbound=lowbound,
            maxiter=maxiter,
        )

    protected = classify_input_versus_threshold(y, lowbound=lowbound, thresh=thresh)
    active = ~protected
    out = np.full(len(xnew), np.nan, dtype=float)

    for i in np.flatnonzero(protected):
        mask = select_destination_points_in_interval(
            xnew,
            x[i],
            x[i + 1],
            include_right=(i == len(y) - 1),
        )
        out[mask] = y[i]

    for run_start, run_stop in find_runs(active):
        mask_run, vals = _interpolate_active_run(
            x=x,
            y=y,
            xnew=xnew,
            run_start=run_start,
            run_stop=run_stop,
            p=p,
            lowbound=lowbound,
            tolbound=tolbound,
            maxiter=maxiter,
        )
        out[mask_run] = vals

    if np.isnan(out).any():
        raise RuntimeError("internal error: incomplete interpolation coverage")

    return out



def rhistinterp(ts, dest, p=2.0, lowbound=None, thresh=None, tolbound=1.0e-3, maxiter=5):
    """Interpolate a regular time series to a finer series by conservative reconstruction.

    Parameters
    ----------
    ts : pandas.Series or pandas.DataFrame
        Input series with PeriodIndex.
    dest : str or pandas.DatetimeIndex
        Target frequency or explicit destination index.
    p : float or ndarray, optional
        Initial spline tension parameter.
    lowbound : float, optional
        Lower bound of interpolated values.
    thresh : float, optional
        Threshold above ``lowbound``. Intervals with values at or below this
        threshold are reconstructed as constant plateaus. If None, legacy
        behavior is used.
    tolbound : float, optional
        Tolerance for lower-bound enforcement.
    maxiter : int, optional
        Maximum iterations for local tension tightening.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        Interpolated instantaneous time series.
    """
    if thresh is None:
        legacy_rhistinterp, _ = _load_legacy_functions()
        return legacy_rhistinterp(
            ts,
            dest,
            p=p,
            lowbound=lowbound,
            tolbound=tolbound,
            maxiter=maxiter,
        )

    if lowbound is None:
        raise ValueError("lowbound must be provided when thresh is not None")

    dest, x, xnew = _build_source_coordinates(ts, dest)

    if isinstance(ts, pd.Series):
        y = ts.to_numpy(dtype=float)
        out = _rhistinterp_1d(
            x=x,
            y=y,
            xnew=xnew,
            p=p,
            lowbound=lowbound,
            thresh=thresh,
            tolbound=tolbound,
            maxiter=maxiter,
        )
        return pd.Series(out, index=dest, name=ts.name)

    if not isinstance(ts, pd.DataFrame):
        raise ValueError("ts must be a pandas Series or DataFrame")

    non_numeric = [col for col in ts.columns if not pd.api.types.is_numeric_dtype(ts[col])]
    if non_numeric:
        raise ValueError(f"all columns must be numeric, got non-numeric columns: {non_numeric}")

    res = {}
    for col in ts.columns:
        y = ts[col].to_numpy(dtype=float)
        res[col] = _rhistinterp_1d(
            x=x,
            y=y,
            xnew=xnew,
            p=p,
            lowbound=lowbound,
            thresh=thresh,
            tolbound=tolbound,
            maxiter=maxiter,
        )
    return pd.DataFrame(res, index=dest)
