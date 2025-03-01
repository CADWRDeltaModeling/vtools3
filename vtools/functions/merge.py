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
    # Make defensive copies and validate inputs.
    series = [s.copy() for s in series]
    if not isinstance(series, (tuple, list)) or len(series) == 0:
        raise ValueError("`series` must be a non-empty tuple or list of pandas.Series or pandas.DataFrame.")
    if not all(isinstance(s.index, pd.DatetimeIndex) for s in series):
        raise ValueError("All input series must have a DatetimeIndex.")
    
    # Determine if freq can be preserved.
    first_freq = series[0].index.freq
    same_freq = all(s.index.freq == first_freq for s in series if s.index.freq is not None)
    
    # For mixed types, convert Series to DataFrame.
    has_series = any(isinstance(s, pd.Series) for s in series)
    has_dataframe = any(isinstance(s, pd.DataFrame) for s in series)
    if has_series and has_dataframe:
        if isinstance(names, str):
            series = [s.to_frame(name=names) if isinstance(s, pd.Series) else s for s in series]
        elif names is None:
            # Ensure each Series has a name matching one of the DataFrame columns.
            df_cols = {col for s in series if isinstance(s, pd.DataFrame) for col in s.columns}
            for s in series:
                if isinstance(s, pd.Series) and s.name not in df_cols:
                    raise ValueError("Mixed Series and DataFrames require Series names to match DataFrame columns.")
            series = [s.to_frame(name=s.name) if isinstance(s, pd.Series) else s for s in series]
    
    # For DataFrame inputs, validate column consistency or selection.
    if isinstance(series[0], pd.DataFrame):
        if names is None:
            common = set(series[0].columns)
            for df in series:
                if set(df.columns) != common:
                    raise ValueError("All input DataFrames must have the same columns when `names` is None.")
        elif hasattr(names, "__iter__") and not isinstance(names, str):
            for df in series:
                if not all(col in df.columns for col in names):
                    raise ValueError(f"An input DataFrame does not contain all specified columns: {names}.")
    
    # If names is a string, pre-rename each series.
    if names and isinstance(names, str):
        series = [s.rename(columns={s.columns[0]: names}) if isinstance(s, pd.DataFrame) else s.rename(names) for s in series]

    # Compute the union of all time indices.
    full_index = reduce(lambda i1, i2: i1.union(i2), (s.index for s in series))
    
    # Merge series by reindexing and using combine_first.
    merged = series[0].reindex(full_index)
    for s in series[1:]:
        merged = merged.combine_first(s.reindex(full_index))
    
    # If all inputs were univariate, keep output univariate.
    univariate = all((s.name is not None if isinstance(s, pd.Series) else s.shape[1] == 1) for s in series)
    if univariate and isinstance(merged, pd.DataFrame):
        merged = merged.iloc[:, 0]
    
    # Normalize output type: if all inputs were Series, squeeze the result.
    if all(isinstance(s, pd.Series) for s in series):
        merged = merged.squeeze()
    else:
        if isinstance(merged, pd.Series):
            merged = merged.to_frame()
    
    # Apply renaming/column selection.
    merged = _apply_names(merged, names)
    
    # Set index frequency if possible.
    if same_freq and first_freq is not None:
        try:
            merged.index.freq = first_freq
        except ValueError:
            merged.index.freq = None

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
    # Make defensive copies and validate inputs.
    series = [s.copy() for s in series]
    if not isinstance(series, (tuple, list)) or len(series) == 0:
        raise ValueError("`series` must be a non-empty tuple or list of pandas.Series or pandas.DataFrame.")
    if not all(isinstance(s.index, pd.DatetimeIndex) for s in series):
        raise ValueError("All input series must have a DatetimeIndex.")
    if transition not in ["prefer_last", "prefer_first"] and not isinstance(transition, list):
        raise ValueError("`transition` must be 'prefer_last', 'prefer_first', or a list of timestamps.")
    
    # Determine if freq can be preserved.
    first_freq = series[0].index.freq
    same_freq = all(s.index.freq == first_freq for s in series if s.index.freq is not None)
    
    # If names is a string, pre-rename each series for consistency.
    if names and isinstance(names, str):
        series = [s.rename(columns={s.columns[0]: names}) if isinstance(s, pd.DataFrame) else s.rename(names) for s in series]
    
    # Compute transition points.
    if isinstance(transition, list):
        transition_points = transition
        duplicate_keep = "first"
    else:
        if transition == "prefer_first":
            transition_points = [ts.last_valid_index() for ts in series[:-1]]
            duplicate_keep = "first"
        elif transition == "prefer_last":
            transition_points = [ts.first_valid_index() for ts in series[1:]]
            duplicate_keep = "last"
        if floor_dates:
            transition_points = [dt.floor("D") for dt in transition_points]
    transition_points = [None] + transition_points + [None]
    
    # Extract sections from each series based on transition points.
    sections = []
    for ts_obj, start, end in zip(series, transition_points[:-1], transition_points[1:]):
        section = ts_obj.loc[start:end]
        if not section.empty:
            sections.append(section)
    spliced = pd.concat(sections, axis=0).sort_index() if sections else pd.DataFrame()
    
    # If all inputs are univariate, squeeze output to Series.
    univariate = all((s.name is not None if isinstance(s, pd.Series) else s.shape[1] == 1) for s in series)
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
    
    # Set index frequency if possible.
    if same_freq and first_freq is not None:
        try:
            spliced.index.freq = first_freq
        except ValueError:
            spliced.index.freq = None
    
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


if __name__ == "__main__":
    # For manual testing if needed.
    pass
