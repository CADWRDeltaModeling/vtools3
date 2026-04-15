# vtools/data/indexing.py

import pandas as pd
"""
Index construction and frequency-handling utilities for time series operations.

This module provides low-level helpers for:
- enforcing frequency consistency across multiple time series,
- constructing regular time indexes from valid data extents, and
- reindexing existing series onto continuous grids when possible.

These functions are used by higher-level operations such as merging,
splicing, and blending of time series.
"""

def resolve_common_freq(indexes, preserve_freq=True):
    """
    Determine a common frequency across a collection of pandas indexes.

    This function inspects the `.freq` attribute of each index and, if
    frequency preservation is requested, verifies that all non-null
    frequency attributes are identical. If so, that frequency is returned.
    Otherwise, an error is raised.

    Parameters
    ----------
    indexes : sequence of pandas.Index
        Sequence of pandas Index objects (typically DatetimeIndex or
        PeriodIndex). Each index may or may not have a `.freq` attribute.

    preserve_freq : bool, default True
        If True, enforce that all indexes with a non-null `.freq` attribute
        have identical frequencies. If any mismatch is detected, a
        ValueError is raised.

        If False, no checking is performed and the function always returns
        None.

    Returns
    -------
    freq : pandas offset or None
        The common frequency if one can be determined and
        `preserve_freq=True`. Returns None if:
        - `preserve_freq=False`, or
        - no index has a non-null `.freq` attribute.

    Raises
    ------
    ValueError
        If `preserve_freq=True` and multiple indexes define conflicting
        `.freq` attributes.

    Notes
    -----
    - This function relies only on the `.freq` attribute and does not
      attempt to infer frequency using `pandas.infer_freq` or the more robust
      vtools functions for that purpose.
    - It is intended for use in routines that require strict consistency
      of sampling intervals across inputs.

    See Also
    --------
    regular_index_from_valid_extent : Construct a regular index once a
        common frequency has been established.
    """    
    freqs = [idx.freq for idx in indexes if getattr(idx, "freq", None) is not None]

    if not preserve_freq or not freqs:
        return None

    first = freqs[0]
    for f in freqs[1:]:
        if f != first:
            raise ValueError(
                "Input series have inconsistent frequencies; cannot preserve frequency."
            )
    return first


def regular_index_from_valid_extent(series, freq):
    """
    Construct a regular index spanning the valid data extent of input series.

    This function creates a regular (fixed-frequency) index that spans
    from the earliest first valid timestamp to the latest last valid
    timestamp across a collection of time series.

    Parameters
    ----------
    series : sequence of pandas.Series or pandas.DataFrame
        Input time series objects. Each must have an index of the same
        type (DatetimeIndex or PeriodIndex). The index values are used
        to determine the overall time extent of valid data.

    freq : pandas offset
        Frequency to use when constructing the regular index (e.g.,
        pandas offset such as Hour, Day, etc.). Typically obtained from
        `resolve_common_freq`.

    Returns
    -------
    index : pandas.DatetimeIndex or pandas.PeriodIndex
        A regular index spanning from the minimum first valid timestamp
        to the maximum last valid timestamp across all input series.

        If no valid timestamps are found in any series, an empty index
        of the same type as the first input is returned.

    Raises
    ------
    ValueError
        If the index type is not supported (i.e., not DatetimeIndex or
        PeriodIndex).

    Notes
    -----
    - The function uses `first_valid_index()` and `last_valid_index()`
      for each series, so leading and trailing NaNs are ignored when
      determining the time extent.
    - The returned index includes both endpoints.
    - No validation is performed to ensure that input series conform
      to the specified frequency.

    See Also
    --------
    resolve_common_freq : Determine whether a shared frequency exists
        across input indexes.
    """
    firsts = [s.first_valid_index() for s in series if s.first_valid_index() is not None]
    lasts  = [s.last_valid_index()  for s in series if s.last_valid_index()  is not None]

    if not firsts or not lasts:
        return series[0].index[:0]

    start = min(firsts)
    end   = max(lasts)

    idx0 = series[0].index
    if isinstance(idx0, pd.DatetimeIndex):
        return pd.date_range(start=start, end=end, freq=freq)
    elif isinstance(idx0, pd.PeriodIndex):
        return pd.period_range(start=start, end=end, freq=freq)
    else:
        raise ValueError("Unsupported index type for frequency preservation.")


def reindex_to_continuous(result, freq):
    """
    Reindex a time series onto a regular grid if possible.

    This function attempts to map an existing time series onto a
    continuous, fixed-frequency index spanning its full time extent.
    If the existing timestamps are not compatible with the target
    regular grid, the input is returned unchanged.

    Parameters
    ----------
    result : pandas.Series or pandas.DataFrame
        Time series to be reindexed. Must have a DatetimeIndex or
        PeriodIndex.

    freq : pandas offset or None
        Target frequency for the regular index. If None, no action is
        taken and `result` is returned unchanged.

    Returns
    -------
    out : pandas.Series or pandas.DataFrame
        Reindexed time series if all existing timestamps align with
        the target regular grid. Otherwise, the original input is
        returned unchanged.

    Notes
    -----
    - The function first constructs a regular index from the minimum
      to maximum timestamps of `result` using the provided `freq`.
    - If any existing timestamps are not present in the constructed
      regular index, the function does not reindex and instead clears
      the `.freq` attribute (if possible) before returning the original
      data.
    - This behavior is intentionally conservative to avoid silently
      dropping or shifting data.

    - When reindexing succeeds:
      - Missing timestamps are filled with NaN.
      - The `.freq` attribute is set on the resulting index if possible.

    Limitations
    -----------
    - This function assumes that `result.index` is monotonic and
      comparable with the generated regular index.
    - It does not attempt to infer or repair irregular spacing.

    See Also
    --------
    regular_index_from_valid_extent : Construct a regular index prior
        to composition operations.
    resolve_common_freq : Determine if a shared frequency can be enforced.
    """
    if freq is None:
        return result

    start = result.index.min()
    end   = result.index.max()

    if isinstance(result.index, pd.DatetimeIndex):
        cont = pd.date_range(start=start, end=end, freq=freq)
    elif isinstance(result.index, pd.PeriodIndex):
        cont = pd.period_range(start=start, end=end, freq=freq)
    else:
        return result

    try:
        if not pd.Index(result.index).isin(cont).all():
            try:
                result.index.freq = None
            except ValueError:
                pass
            return result
    except Exception:
        return result

    result = result.reindex(cont)

    try:
        result.index.freq = freq
    except ValueError:
        result.index.freq = None
    return result
