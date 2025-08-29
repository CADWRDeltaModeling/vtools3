"""
Daylight Savings Time Conversion
================================

This module provides the function :func:`dst_st` for converting a pandas Series/Dataframe
with a naive DatetimeIndex that observes daylight savings time (DST) to a fixed
standard time zone (e.g., PST) using POSIX conventions.

See the automatic API documentation for details:
:py:func:`vtools.data.dst.dst_st`
"""

import pandas as pd

__all__ = ["dst_st"]

def dst_st(
    ts,
    src_tz: str = "US/Pacific",
    target_tz: str = "Etc/GMT+8"
):
    """
    Convert a pandas Series with a datetime index from a timezone-unaware index
    that observes DST (e.g., US/Pacific) to a fixed standard time zone (e.g., Etc/GMT+8)
    using POSIX conventions.

    Parameters
    ----------
    ts : pandas.Series
        Time series with a naive (timezone-unaware) DatetimeIndex.
    src_tz : str, optional
        Source timezone name (default is 'US/Pacific').
    target_tz : str, optional
        Target standard timezone name (default is 'Etc/GMT+8').

    Returns
    -------
    pandas.Series
        Time series with index converted to the target standard timezone and made naive.

    Notes
    -----
    - The function assumes the index is not already timezone-aware.
    - 'Etc/GMT+8' is the correct tz name for UTC-8 (PST) in pytz; note the sign is reversed from what
      might be expected.
    - Handles ambiguous/nonexistent times due to DST transitions.
    - The returned index is naive (timezone-unaware) but represents the correct standard time.
    - If the input index is already timezone-aware, this function will raise an error.

    Examples
    --------
    >>> import pandas as pd
    >>> from vtools import dst_st
    >>> rng = pd.date_range("2023-11-05 00:00", "2023-11-05 04:00", freq="30min")
    >>> ts = pd.Series(range(len(rng)), index=rng)
    >>> converted = dst_st(ts)
    >>> print(converted)
    2023-11-05 00:00:00    0
    2023-11-05 00:30:00    1
    2023-11-05 01:00:00    2
    2023-11-05 01:30:00    3
    2023-11-05 02:30:00    5
    2023-11-05 03:00:00    6
    2023-11-05 03:30:00    7
    2023-11-05 04:00:00    8
    dtype: int64

    """
    ts = ts.copy()
    orig_freq = getattr(ts.index, 'freq', None)
    ts.index = ts.index.tz_localize(
        src_tz,
        nonexistent="shift_backward",  # Handle nonexistent times (e.g., spring forward)
        ambiguous="NaT"                # Mark ambiguous times (e.g., fall back) as NaT
    )
    ts.index = ts.index.tz_convert(target_tz)
    ts.index = ts.index.tz_localize(None)
    # Drop NaT values in the index (from ambiguous times)
    mask = ~ts.index.isna()
    ts = ts[mask]
    # Try to restore original frequency if possible
    if orig_freq is not None:
        try:
            ts = ts.asfreq(orig_freq)
        except Exception:
            pass
    return ts

if __name__ == "__main__":

    # Create a DatetimeIndex that spans the PDT to PST transition (first Sunday in November)
    rng = pd.date_range("2023-11-05 00:00", "2023-11-05 04:00", freq="30min")
    ts = pd.Series(range(len(rng)), index=rng)

    print("Original (naive, US/Pacific):")
    print(ts)

    converted = dst_st(ts)

    print("\nConverted to standard time (Etc/GMT+8, naive):")
    print(converted)
