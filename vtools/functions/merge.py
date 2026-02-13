#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = ["ts_merge", "ts_splice"]

import pandas as pd
import numpy as np
from functools import reduce
from vtools.functions.colname_align import align_inputs_strict


def _reindex_to_continuous(result, first_freq):
    if first_freq is None:
        return result

    start = result.index.min()
    end   = result.index.max()

    if isinstance(result.index, pd.DatetimeIndex):
        cont = pd.date_range(start=start, end=end, freq=first_freq)
    elif isinstance(result.index, pd.PeriodIndex):
        cont = pd.period_range(start=start, end=end, freq=first_freq)
    else:
        return result  # unknown index type; do nothing

    # --- NEW: never drop existing stamps
    # If any current stamp isn't on 'cont', skip reindexing
    try:
        if not pd.Index(result.index).isin(cont).all():
            # preserve data; just avoid forcing a conflicting freq
            try:
                result.index.freq = None
            except ValueError:
                pass
            return result
    except Exception:
        return result

    result = result.reindex(cont)

    if isinstance(result.index, pd.PeriodIndex):
        result.index = pd.PeriodIndex(result.index, freq=first_freq)
    else:
        try:
            result.index.freq = first_freq
        except ValueError:
            result.index.freq = None
    return result

#####################
@align_inputs_strict(seq_arg=0, names_kw="names") 
def ts_merge(series,
             names=None,
             strict_priority = False):
    """
    Merge multiple time series together, prioritizing series in order.

    Parameters
    ----------
    series : sequence of pandas.Series or pandas.DataFrame
        Higher priority first. All indexes must be DatetimeIndex.
    names : None, str, or iterable of str, optional
        - If `None` (default), inputs must share compatible column names.
        - If `str`, the output is univariate and will be named accordingly.
        - If iterable, it is used as a subset/ordering of columns.
    strict_priority : bool, default False
        If False (default): lower-priority data may fill NaNs in higher-priority
        series anywhere (traditional merge/overlay).
        If True: for each column, within the window
        [first_valid_index, last_valid_index] of any higher-priority series,
        lower-priority data are masked out — even if the higher-priority value is NaN.
        Outside those windows, behavior is unchanged.

    Returns
    -------
    pandas.Series or pandas.DataFrame
    """
    # --- Input validation (messages match tests) ---
    if not isinstance(series, (list, tuple)) or len(series) == 0:
        raise ValueError("`series` must be a non-empty tuple or list")

    if not all(isinstance(getattr(s, "index", None), pd.DatetimeIndex) for s in series):
        raise ValueError("All input series must have a DatetimeIndex.")

    # Preserve first series freq (may be None)
    first_freq = getattr(series[0].index, "freq", None)

    # If any DataFrame is present, convert Series->DataFrame to unify shapes
    any_df = any(isinstance(s, pd.DataFrame) for s in series)
    if any_df:
        series = [s.to_frame(name=s.name) if isinstance(s, pd.Series) else s for s in series]

    # Column compatibility checks (messages match tests)
    all_df     = all(isinstance(s, pd.DataFrame) for s in series)
    any_df     = any(isinstance(s, pd.DataFrame) for s in series)
    any_series = any(isinstance(s, pd.Series)     for s in series)

    if all_df:
        if names is None:
            cols0 = list(series[0].columns)
            for s in series[1:]:
                if list(s.columns) != cols0:
                    raise ValueError(
                        "All input DataFrames must have the same columns when `names` is None."
                    )
    elif any_df and any_series:
        if names is None:
            df_cols = {c for s in series if isinstance(s, pd.DataFrame) for c in s.columns}
            for s in series:
                if isinstance(s, pd.Series) and s.name not in df_cols:
                    raise ValueError(
                        "Mixed Series and DataFrames require Series names to match DataFrame columns."
                    )
    # else: all Series → no column checks needed

    # Build the union index, sorted in time order
    full_index = series[0].index
    for s in series[1:]:
        full_index = full_index.union(s.index, sort=False)
    full_index = full_index.sort_values()

    # Align to union index and keep DataFrame shape for column-wise masking
    aligned = []
    for s in series:
        a = s.reindex(full_index)
        if isinstance(a, pd.Series):
            a = a.to_frame(name=a.name)
        aligned.append(a.copy())

    # --- Strict priority dominance windows ---
    if strict_priority:
        # For each column, mask lower-priority data only within the dominance window
        for col in set(c for a in aligned for c in a.columns):
            # Find the highest-priority series that contains this column
            for i, hi in enumerate(aligned):
                if col in hi.columns:
                    s_col = hi[col]
                    lo_idx = s_col.first_valid_index()
                    hi_idx = s_col.last_valid_index()
                    break
            else:
                continue  # column not present in any input

            if lo_idx is None or hi_idx is None:
                continue  # no dominance window for this column

            mask = (full_index >= lo_idx) & (full_index <= hi_idx)
            for j in range(i + 1, len(aligned)):
                if col in aligned[j].columns:
                    aligned[j].loc[mask, col] = np.nan  # keep numeric dtype

    # Combine in priority order via combine_first
    merged = aligned[0]
    for a in aligned[1:]:
        merged = merged.combine_first(a)

    # NEW: ensure the full union index is present (including all-NaN rows)
    merged = merged.reindex(full_index)
    # --- keep union, but drop masked lower-only rows inside dominance window ---
    if strict_priority:
        # Use the highest-priority input to define the dominance window (per-column union)
        top = aligned[0]  # DataFrame (we normalized earlier)
        lo_candidates = [top[c].first_valid_index() for c in top.columns]
        hi_candidates = [top[c].last_valid_index()  for c in top.columns]
        lo0 = min([x for x in lo_candidates if x is not None], default=None)
        hi0 = max([x for x in hi_candidates if x is not None], default=None)

        if lo0 is not None and hi0 is not None:
            idx = merged.index
            in_window   = (idx >= lo0) & (idx <= hi0)
            in_top_idx  = idx.isin(series[0].index)  # keep rows that are from the top series' index
            # rows that are entirely NaN after strict masking
            all_nan = merged.isna().all(axis=1) if isinstance(merged, pd.DataFrame) else merged.isna()
            # Drop only those that are NaN, within the window, and not in the top index (i.e., lower-only timestamps)
            drop_mask = in_window & (~in_top_idx) & all_nan
            if drop_mask.any():
                merged = merged.loc[~drop_mask]


    # If all inputs were univariate Series, return a Series
    all_series = all(isinstance(s, pd.Series) for s in series)
    if all_series:
        merged = merged.squeeze()
    else:
        if isinstance(merged, pd.Series):
            merged = merged.to_frame()

    
    # Reindex to a continuous index using the first series' freq (your helper)
    merged = _reindex_to_continuous(merged, first_freq)
    # Name alignment will be added by decorator

    return merged

@align_inputs_strict(seq_arg=0, names_kw="names") 
def ts_splice(series, names=None, transition="prefer_last", floor_dates=False):
    """
    Splice multiple time series together, prioritizing series in patches of time.

    Unlike `ts_merge`, which blends overlapping data points, `ts_splice` stitches
    together time series without overlap. The function determines when to switch
    between series based on a transition strategy.

    Parameters
    ----------
    series : tuple or list of pandas.DataFrame or pandas.Series
        A tuple or list of time series.
        Each series must have a `DatetimeIndex` and consistent column structure.

    names : None, str, or iterable of str, optional
        - If `None` (default), all input series must share common column names,
          and the output will merge common columns.
        - If a `str`, all input series must have a **single column**, and the
          output will be a DataFrame with this name as the column name.
        - If an iterable of `str`, all input DataFrames must have the same
          number of columns matching the length of `names`, and these will be
          used for the output.

    transition : {'prefer_first', 'prefer_last'} or list of pandas.Timestamp
        Defines how to determine breakpoints between time series:
        - `'prefer_first'`: Uses the earlier series on the list during until its valid timestamp.
        - `'prefer_last'`: Uses the later series starting from its first valid timestamp.
        - A list of specific timestamps can also be provided as transition points.

    floor_dates : bool, optional, default=False
        If `True`, inferred transition timestamps (`prefer_first` or `prefer_last`)
        are floored to the beginning of the day. This can introduce NaNs if the
        input series are regular with a `freq` attribute.

    Returns
    -------
    pandas.DataFrame or pandas.Series
        - If the input contains multi-column DataFrames, the output is a DataFrame
          with the same column structure.
        - If a collection of single-column `Series` is provided, the output will be
          a Series.
        - The output retains a `freq` attribute if all inputs share the same frequency.

    Notes
    -----
    - The output time index is the union of input time indices.
    - If `transition` is `'prefer_first'`, gaps may appear in the final time series.
    - If `transition` is `'prefer_last'`, overlapping data is resolved in favor
      of later series.

    See Also
    --------
    ts_merge : Merges series by filling gaps in order of priority.
    """
    series = [s.copy() for s in series]

    if not isinstance(series, (tuple, list)) or len(series) == 0:
        raise ValueError(
            "`series` must be a non-empty tuple or list of pandas.Series or pandas.DataFrame."
        )
    if not all(
        isinstance(s.index, pd.DatetimeIndex) or isinstance(s.index, pd.PeriodIndex)
        for s in series
    ):
        raise ValueError("All input series must have a DatetimeIndex or PeriodIndex.")

    # Ensure all input series have the same type of index.
    index_type = type(series[0].index)
    if not all(isinstance(s.index, index_type) for s in series):
        raise ValueError(
            f"All input series must have indexes of type {index_type.__name__}."
        )

    if transition not in ["prefer_last", "prefer_first"] and not isinstance(
        transition, list
    ):
        raise ValueError(
            "`transition` must be 'prefer_last', 'prefer_first', or a list of timestamps."
        )

    # Determine if frequency can be preserved.
    first_freq = series[0].index.freq
    same_freq = all(
        s.index.freq == first_freq for s in series if s.index.freq is not None
    )

    # If names is a string, pre-rename each series for consistency.

    if names is not None and isinstance(names, str):
        series = [
            (
                s.rename(columns={s.columns[0]: names})
                if isinstance(s, pd.DataFrame)
                else s.rename(names)
            )
            for s in series
        ]

    # Compute transition points.
    if isinstance(transition, list):
        transition_points = transition
        duplicate_keep = "first"
    else:
        if transition == "prefer_first":
            transition_points = [s.last_valid_index() for s in series[:-1]]
            duplicate_keep = "first"
        elif transition == "prefer_last":
            transition_points = [s.first_valid_index() for s in series[1:]]
            duplicate_keep = "last"
        if floor_dates:
            transition_points = [dt.floor("D") for dt in transition_points]
    transition_points = [None] + transition_points + [None]

    # Extract sections based on transition points.
    sections = []
    for ts_obj, start, end in zip(
        series, transition_points[:-1], transition_points[1:]
    ):
        section = ts_obj.loc[start:end]
        if not section.empty:
            sections.append(section)
    spliced = pd.concat(sections, axis=0).sort_index() if sections else pd.DataFrame()

    # If all inputs were univariate, ensure output remains univariate.
    univariate = all(
        (s.name is not None if isinstance(s, pd.Series) else s.shape[1] == 1)
        for s in series
    )
    if univariate and isinstance(spliced, pd.DataFrame):
        spliced = spliced.iloc[:, 0]

    # Normalize output type.
    if all(isinstance(s, pd.Series) for s in series):
        spliced = spliced.squeeze()
    else:
        if isinstance(spliced, pd.Series):
            spliced = spliced.to_frame()

    # Remove duplicate timestamps based on transition preference.
    dup = spliced.index.duplicated(keep=duplicate_keep)
    if dup.any():
        spliced = spliced[~dup]

    # Reindex to a continuous index *****
    spliced = _reindex_to_continuous(spliced, first_freq)
    return spliced

