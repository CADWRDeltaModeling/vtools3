"""Basic ops for creating, testing and manipulating times and time intervals.
This module contains factory and helper functions for working with
times and time intervals.

For time intervals (or deltas), VTools uses classes that are compatible
with the "freq" argument of

requires a time and time
interval system that is consistent (e.g. time+n*interval makes sense)
and that can be applied to both calendar dependent
and calendar-independent intervals. Because this requirement
is not met by any one implementation it is recommended that
you always use the factory functions in this module
for creating intervals or testing whether an interval is valid.
"""

import numpy as np
import pandas as pd


__all__ = ["seconds", "minutes", "hours", "days", "months", "years","to_timedelta", "divide_interval"]


def seconds(s):
    """Create a time interval representing s seconds"""
    return pd.offsets.Second(s)


def minutes(m):
    """Create a time interval representing m minutes"""
    return pd.offsets.Minute(m)


def hours(h):
    """Create a time interval representing h hours"""
    return pd.offsets.Hour(h)


def days(d):
    """Create a time interval representing d days"""
    return pd.offsets.Day(d)


def months(m):
    """Create a time interval representing m months"""
    return pd.offsets.DateOffset(months=m)


def years(y):
    """Create a time interval representing y years"""
    return pd.offsets.DateOffset(years=y)


def dst_to_standard_naive(ts, dst_zone="US/Pacific", standard_zone="Etc/GMT+8"):
    """Convert timezone-unaware series from a local (with daylight) time to standard time
    This would be useful, say, for converting a series that is PDT during summer to one that is not.
    The routine is mainly to treat cases where the time stamps at DST interfaces are not redundant -- if they
    are you can probably use tz_convert and tz_localize with the ambiguous = 'infer' option and do the job more
    efficiently, but lots of databases don't store data this way.

    The choice of the standard_zone is, it seems, buggy. The defaults are supposed to convert from PST/PDT to pure PST,
    and the latter should be GMT-8. In a sense, this function is included before the behavior is really understood.

    Only regular series are accepted ... this is a quirk of the implementation
    """

    # Create a date range that spans the possible times and is same refinement
    try:
        dt = ts.freq
    except:
        dt = pd.offsets.Minute(15)
    hr = pd.offsets.Hour(1)
    ndx2 = pd.date_range(
        start=ts.index[0] - hr, end=ts.index[-1] + hr, freq=dt, tz=dst_zone
    )

    # Here is the determination of whether it is dst
    isdst = [bool(x.dst()) for x in ndx2.to_pydatetime()]

    # Use DataFrame indexing to perform the lookup for values in my original index

    df2 = pd.DataFrame({"isdst": isdst}, index=ndx2.tz_localize(None))
    df2 = df2.loc[~df2.index.duplicated(keep="last"), :]

    ambig = df2.loc[ts.index, "isdst"].values

    # Here is the real work
    ts2 = (
        ts.tz_localize("US/Pacific", ambiguous=ambig)
        .tz_convert(standard_zone)
        .tz_localize(None)
    )
    return ts2


_FIXED_OFFSET_CLASSES = (
    pd.offsets.Day,
    pd.offsets.Hour,
    pd.offsets.Minute,
    pd.offsets.Second,
    pd.offsets.Milli,
    pd.offsets.Micro,
    pd.offsets.Nano,
)

def to_timedelta(x):
    """
    Convert x to pandas.Timedelta if and only if it represents
    a fixed-length duration.

    Notes
    -----
    Numeric values are rejected because they are unit-ambiguous.
    """
    if isinstance(x, (int, np.integer, float)):
        raise TypeError(
            "Numeric values are ambiguous as time intervals; "
            "use a Timedelta, offset (hours(1)), or a string like '1h'."
        )

    if isinstance(x, pd.Timedelta):
        return x

    # FIRST: try Timedelta parsing (handles '1h', '1D', etc.)
    try:
        if isinstance(x, str):
            x = x.replace("H", "h").replace("d", "D").replace("T", "t")
        return pd.Timedelta(x)
    except Exception:
        pass

    # FALLBACK: try fixed pandas offsets
    try:
        off = pd.tseries.frequencies.to_offset(x)
    except Exception as e:
        raise TypeError(f"Cannot interpret interval {x!r}") from e

    if not isinstance(off, _FIXED_OFFSET_CLASSES):
        raise TypeError(
            f"Offset {type(off).__name__} is calendar-dependent "
            "and not a fixed-length interval"
        )

    return pd.Timedelta(off.nanos, unit="ns")


def divide_interval(a, b, *, tol=1e-12, require_int=True):
    """
    Divide two intervals (or two scalars) safely.

    - interval / interval -> float ratio (or int if require_int)
    - scalar / scalar     -> numeric ratio
    - mixed scalar/interval -> TypeError
    """
    a_is_num = isinstance(a, (int, np.integer, float))
    b_is_num = isinstance(b, (int, np.integer, float))

    if a_is_num and b_is_num:
        if b == 0:
            raise ZeroDivisionError("Division by zero")
        r = a / b
        if require_int:
            r_int = int(round(r))
            if abs(r - r_int) > tol:
                raise ValueError(f"Scalars are not evenly divisible: {a!r} / {b!r} = {r}")
            return r_int
        return float(r)

    if a_is_num or b_is_num:
        raise TypeError(
            "divide_interval does not support scalar/interval division; "
            "use explicit Timedelta scaling instead."
        )

    td_a = to_timedelta(a)
    td_b = to_timedelta(b)

    if td_b == pd.Timedelta(0):
        raise ZeroDivisionError("Division by zero interval")

    ratio = td_a / td_b

    if require_int:
        r_int = int(round(ratio))
        if abs(ratio - r_int) > tol:
            raise ValueError(
                f"Intervals are not evenly divisible: {a!r} / {b!r} = {ratio}"
            )
        return r_int

    return float(ratio)
