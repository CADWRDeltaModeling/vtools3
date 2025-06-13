#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = ["ts_merge", "ts_splice"]

import pandas as pd
from functools import reduce


def ts_merge(series, names=None):
    """
    Merge multiple time series together, prioritizing series in order.

    This function merges a tuple of time series, filling missing values based on
    a priority order. The first series is used as the base, with subsequent series
    filling in missing values where available. If time series have irregular timestamps,
    consider using `ts_splice` instead.

    Parameters
    ----------
    series : tuple or list of pandas.DataFrame or pandas.Series
        A tuple of time series ranked from highest to lowest priority. Each series
        must have a `DatetimeIndex` and compatible column names.

    names : None, str, or iterable of str, optional
        - If `None` (default), all input series must share common column names,
          and the output will merge common columns or series names.
        - If a `str`, all input series must have a **single column** (or a matching
          column name if mixed types are provided), and the output will be a DataFrame
          with this name as the column name.
        - If an iterable of `str`, all input DataFrames must have the same
          number of columns matching the length of the names argument and these will be used
          for the output.

    Returns
    -------
    pandas.DataFrame
        - If the input contains DataFrames with multiple columns, the output is a
          DataFrame with the same time extent as the union of inputs and columns.
        - If a collection of single-column `Series` is specified, the output will be
          converted into a single-column DataFrame.

    Notes
    -----
    - The output time index is the union of input time indices.
    - If a duplicate index exists in the first series, only the first occurrence is used.
    - Lower-priority series only fill gaps in higher-priority series and do not override
      existing values.

    See Also
    --------
    ts_splice : Alternative merging method for irregular time series.
    """
    # Make defensive copies of the input series.
    series = [s.copy() for s in series]

    if not isinstance(series, (tuple, list)) or len(series) == 0:
        raise ValueError(
            "`series` must be a non-empty tuple or list of pandas.Series or pandas.DataFrame."
        )

    # Ensure all input series have an index of the same type.
    index_type = type(series[0].index)
    if not all(isinstance(s.index, index_type) for s in series):
        raise ValueError(
            f"All input series must have indexes of type {index_type.__name__}."
        )

    if not all(isinstance(s.index, (pd.DatetimeIndex, pd.PeriodIndex)) for s in series):
        raise ValueError("All input series must have a DatetimeIndex or PeriodIndex.")

    # Determine if frequency can be preserved.
    first_freq = series[0].index.freq
    same_freq = all(
        s.index.freq == first_freq for s in series if s.index.freq is not None
    )

    # For mixed types, convert Series to DataFrame.
    has_series = any(isinstance(s, pd.Series) for s in series)
    has_dataframe = any(isinstance(s, pd.DataFrame) for s in series)
    if has_series and has_dataframe:
        if isinstance(names, str):
            series = [
                s.to_frame(name=names) if isinstance(s, pd.Series) else s
                for s in series
            ]
        elif names is None:
            df_cols = {
                col for s in series if isinstance(s, pd.DataFrame) for col in s.columns
            }
            for s in series:
                if isinstance(s, pd.Series) and s.name not in df_cols:
                    raise ValueError(
                        "Mixed Series and DataFrames require Series names to match DataFrame columns."
                    )
            series = [
                s.to_frame(name=s.name) if isinstance(s, pd.Series) else s
                for s in series
            ]

    # For DataFrame inputs, validate column consistency.
    if isinstance(series[0], pd.DataFrame):
        if names is None:
            common = set(series[0].columns)
            for df in series:
                if set(df.columns) != common:
                    raise ValueError(
                        "All input DataFrames must have the same columns when `names` is None."
                    )
        elif hasattr(names, "__iter__") and not isinstance(names, str):
            for df in series:
                if not all(col in df.columns for col in names):
                    raise ValueError(
                        f"An input DataFrame does not contain all specified columns: {names}."
                    )

    # If names is a string, pre-rename each series for consistency.
    if names and isinstance(names, str):
        series = [
            (
                s.rename(columns={s.columns[0]: names})
                if isinstance(s, pd.DataFrame)
                else s.rename(names)
            )
            for s in series
        ]

    # Compute the union of all time indices.
    full_index = series[0].index
    for s in series[1:]:
        full_index = full_index.union(s.index, sort=False)
    full_index = full_index.sort_values()

    # Merge series by reindexing and using combine_first.
    merged = series[0].reindex(full_index)
    for s in series[1:]:
        merged = merged.combine_first(s.reindex(full_index))

    # If all inputs were univariate, ensure output remains univariate.
    univariate = all(
        (s.name is not None if isinstance(s, pd.Series) else s.shape[1] == 1)
        for s in series
    )
    if univariate and isinstance(merged, pd.DataFrame):
        merged = merged.iloc[:, 0]

    # Normalize output type.
    if all(isinstance(s, pd.Series) for s in series):
        merged = merged.squeeze()
    else:
        if isinstance(merged, pd.Series):
            merged = merged.to_frame()

    # Apply renaming/column selection.
    merged = _apply_names(merged, names)

    merged = _reindex_to_continuous(merged, first_freq)

    return merged


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
    if names and isinstance(names, str):
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
            transition_points = [dt.floor("d") for dt in transition_points]
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

    # Apply renaming/column selection.
    spliced = _apply_names(spliced, names)

    # ***** NEW STEP: Reindex to a continuous index *****
    spliced = _reindex_to_continuous(spliced, first_freq)
    return spliced


def _apply_names(result, names):
    """Helper to apply renaming and column selection based on `names`."""
    if names:
        if isinstance(result, pd.Series):
            result.name = names
        elif isinstance(names, str):
            result = result.rename(columns={result.columns[0]: names})
        elif hasattr(names, "__iter__"):
            result = result[names]
    return result


def _reindex_to_continuous(result, first_freq):
    """
    Reindex the given result (DataFrame or Series) to a continuous index spanning
    from the minimum to maximum timestamp of the current index using the provided frequency.

    Parameters
    ----------
    result : pandas.DataFrame or pandas.Series
        The merged or spliced result whose index will be reindexed.
    first_freq : frequency
        The frequency (e.g. 'D' for daily) to use for the continuous index, taken from the first input series.

    Returns
    -------
    result : pandas.DataFrame or pandas.Series
        The result reindexed to a continuous index. Any gaps in the timeline will be filled with NaN.
        If the index is a PeriodIndex, the index is rebuilt with the proper frequency.
    """
    if first_freq is None:
        return result

    start = result.index.min()
    end = result.index.max()

    if isinstance(result.index, pd.DatetimeIndex):
        continuous_index = pd.date_range(start=start, end=end, freq=first_freq)
    elif isinstance(result.index, pd.PeriodIndex):
        continuous_index = pd.period_range(start=start, end=end, freq=first_freq)
    else:
        continuous_index = result.index  # For other types, leave unchanged

    result = result.reindex(continuous_index)

    if isinstance(result.index, pd.PeriodIndex):
        # For PeriodIndex, rebuild the index because .freq is read-only.
        result.index = pd.PeriodIndex(result.index, freq=first_freq)
    else:
        try:
            result.index.freq = first_freq
        except ValueError:
            result.index.freq = None
    return result
