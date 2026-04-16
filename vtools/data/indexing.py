# vtools/data/indexing.py

import pandas as pd
__all__ = ["resolve_common_freq", "regular_index_from_valid_extent",
           "reindex_to_continuous", "infer_freq_robust", "inferred_regular_freq", "compare_regular_freq"]

"""
Index construction and frequency-handling utilities for time series operations.

This module provides low-level helpers for:
- enforcing frequency consistency across multiple time series,
- constructing regular time indexes from valid data extents, and
- reindexing existing series onto continuous grids when possible.

These functions are used by higher-level operations such as merging,
splicing, and blending of time series.
"""

def infer_freq_robust(
    index, preferred=["h", "15min", "6min", "10min", "h", "d"], **kwargs
):
    """
    Infer the frequency of a time series index using multiple strategies.
    
    Attempts to detect the frequency of a DatetimeIndex by trying various
    approaches: direct inference on the full index, on subsets at different
    positions, and finally by testing against a list of preferred frequencies.
    This robust approach handles indices with irregular intervals or gaps better
    than pandas' native infer_freq.
    
    Parameters
    ----------
    index : pd.DatetimeIndex
        The time series index to analyze.
    preferred : list of str, optional
        Preferred frequency strings to test as fallback (default: ["h", "15min", "6min", "10min", "h", "d"]).
        Each string should be a valid pandas frequency alias.
    **kwargs
        Additional keyword arguments (currently unused, reserved for future use).
    
    Returns
    -------
    str
        A pandas frequency alias string (e.g., "h", "D", "15min").
    
    Raises
    ------
    ValueError
        If all frequency inference attempts fail.
    
    Notes
    -----
    - Index is rounded to 1-minute resolution before analysis.
    - For short indices (< 8 points), direct inference is attempted.
    - For longer indices, inference is tried on the last 7 points, then at
      75% and 50% positions, then at the start.
    - Finally, preferred frequencies are tested by checking if at least 98%
      of index values round correctly to that frequency.
    """
    index = index.round("1min")
    if len(index) < 8:
        # not enough to quibble, use the 8 points
        f = pd.infer_freq(index)
    else:
        f = pd.infer_freq(index[-7:-1])

        if f is None:
            index = index.round("1min")
            istrt = 3 * len(index) // 4
            f = pd.infer_freq(index[istrt : istrt + 7])
        if f is None:
            # Give it one more shot halfway through
            istrt = len(index) // 2
            f = pd.infer_freq(index[istrt : istrt + 7])
        if f is None:
            f = pd.infer_freq(index[0:7])
        if f is None:
            for p in preferred:
                freq = to_timedelta(p)
                tester = index.round(p)

                diff = abs(index - tester) < (freq / 5)
                frac = diff.mean()
                if frac > 0.98:
                    return p
        if f is None:
            raise ValueError(
                "read_ts set to infer frequency, but multiple attempts failed. Set to string to manually "
            )
        return f



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




def inferred_regular_freq(ts):
    """
    Determine whether a time series has a regular inferred frequency.

    This function is a strict checker for already-read time series. It does
    not coerce, round, or repair timestamps. It inspects the existing index
    and reports whether a regular frequency can be established.

    Parameters
    ----------
    ts : pandas.Series or pandas.DataFrame
        Time series with a DatetimeIndex or PeriodIndex.

    Returns
    -------
    freq : pandas offset or None
        The inferred or attached frequency if the series is regular.
        Returns None if the series is not regular.

    reason : str
        Diagnostic reason code. One of:
        - ``"ok"``
        - ``"degenerate"``
        - ``"not_datetime_like"``
        - ``"not_monotonic"``
        - ``"duplicates"``
        - ``"infer_failed"``

    Notes
    -----
    This checker is intentionally strict and non-repairing. It is intended
    for validation and reconcile policy, not ingestion recovery.

    Frequency is determined by:
    1. Using ``index.freq`` if present.
    2. Otherwise trying ``pandas.infer_freq(index)``.

    See Also
    --------
    compare_regular_freq : Compare the inferred regular frequencies of two
        time series for reconcile compatibility.
    """
    idx = ts.index

    if not isinstance(idx, (pd.DatetimeIndex, pd.PeriodIndex)):
        return None, "not_datetime_like"

    if len(idx) < 2:
        return pd.Timedelta(0), "degenerate"

    if not idx.is_monotonic_increasing:
        return None, "not_monotonic"

    if not idx.is_unique:
        return None, "duplicates"

    freq = getattr(idx, "freq", None)
    if freq is not None:
        return freq, "ok"

    try:
        freq = pd.infer_freq(idx)
    except Exception:
        freq = None

    if freq is None:
        return None, "infer_failed"

    try:
        freq = pd.tseries.frequencies.to_offset(freq)
    except Exception:
        pass

    return freq, "ok"


def compare_regular_freq(ts_src, ts_dst):
    """
    Compare source and destination time series for frequency compatibility.

    This function supports reconcile-policy decisions by checking whether two
    already-read time series are regular and, if so, whether their inferred
    frequencies match.

    Parameters
    ----------
    ts_src : pandas.Series or pandas.DataFrame
        Source (staged) time series.
    ts_dst : pandas.Series or pandas.DataFrame
        Destination (repo) time series.

    Returns
    -------
    status : str
        Compatibility classification. One of:
        - ``"both_regular_same"``
        - ``"both_regular_different"``
        - ``"src_irregular"``
        - ``"dst_irregular"``
        - ``"both_irregular"``

    reason : str
        Human-readable explanation of the classification.

    src_freq : pandas offset or None
        Inferred source frequency, if regular.
    dst_freq : pandas offset or None
        Inferred destination frequency, if regular.

    Notes
    -----
    This helper is asymmetric only in naming: ``src`` and ``dst`` are reported
    separately because reconcile policy may differ depending on which side is
    irregular.

    It does not modify the inputs and does not attempt to reconcile or repair
    them.

    See Also
    --------
    inferred_regular_freq : Check whether a single time series is regular and
        determine its inferred frequency.
    """
    src_freq, src_reason = inferred_regular_freq(ts_src)
    dst_freq, dst_reason = inferred_regular_freq(ts_dst)

    src_ok = src_freq is not None
    dst_ok = dst_freq is not None



    if src_ok and dst_ok:
        if len(ts_src.index) < 2 and len(ts_dst.index) < 2:
            return "both_regular_same", "degenerate_single_point", None, None
        
        if src_freq == dst_freq:
            return "both_regular_same", f"both regular with matching frequency: {src_freq}", src_freq, dst_freq
        return "both_regular_different", f"frequency mismatch: staged={src_freq} repo={dst_freq}", src_freq, dst_freq

    if (not src_ok) and (not dst_ok):
        return "both_irregular", f"both irregular: staged={src_reason}, repo={dst_reason}", src_freq, dst_freq

    if not src_ok:
        return "src_irregular", f"staged not regular: {src_reason}", src_freq, dst_freq

    return "dst_irregular", f"repo not regular: {dst_reason}", src_freq, dst_freq

