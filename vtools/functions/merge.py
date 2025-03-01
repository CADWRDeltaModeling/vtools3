#!/usr/bin/env python
# -*- coding: utf-8 -*-


__all__ = ["ts_merge", "ts_splice"]

import pandas as pd

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

    if not isinstance(series, (tuple, list)) or len(series) == 0:
        raise ValueError("`series` must be a non-empty tuple or list of pandas.Series or pandas.DataFrame.")

    if not all(isinstance(s.index, pd.DatetimeIndex) for s in series):
        raise ValueError("All input series must have a DatetimeIndex.")

    first_freq = series[0].index.freq
    same_freq = all(s.index.freq == first_freq for s in series)

    has_series = any(isinstance(s, pd.Series) for s in series)
    has_dataframe = any(isinstance(s, pd.DataFrame) for s in series)

    if has_series and has_dataframe:
        if isinstance(names, str):
            series = [s.to_frame(name=names) if isinstance(s, pd.Series) else s for s in series]
        elif names is None:
            col_names = {col for s in series if isinstance(s, pd.DataFrame) for col in s.columns}
            if not all(s.name in col_names for s in series if isinstance(s, pd.Series)):
                raise ValueError("Mixed Series and DataFrames require Series names to match DataFrame columns.")
            series = [s.to_frame(name=s.name) if isinstance(s, pd.Series) else s for s in series]

    if isinstance(series[0], pd.DataFrame):
        if names is None:
            common_columns = set(series[0].columns)
            for df in series:
                if set(df.columns) != common_columns:
                    raise ValueError("All input DataFrames must have the same columns when `names` is None.")
            output_columns = list(common_columns)

        elif isinstance(names, str):
            output_columns = [series[0].columns[0]]

        elif hasattr(names, "__iter__"):
            names = list(names)
            for df in series:
                if not all(name in df.columns for name in names):
                    raise ValueError(f"An input DataFrame does not contain all specified columns: {names}.")
            output_columns = names

        else:
            raise ValueError("`names` must be None, a string, or an iterable of strings.")

    else:
        output_columns = [series[0].name if names is None else names]

    merged = pd.concat(series, axis=0, sort=True)
    merged = merged.loc[~merged.index.duplicated(keep='first')]

    for s in series[1:]:
        merged = merged.combine_first(s)

    # ✅ Only set `freq` if it's not None
    if same_freq and first_freq is not None:
        merged.index.freq = first_freq

    # ✅ Fix: Rename correctly for Series and DataFrames
    if isinstance(merged, pd.Series):
        if names:
            merged.name = names  # ✅ Correct way to rename a Series
    else:
        if isinstance(names, str):
            merged = merged.rename(columns={output_columns[0]: names})  # ✅ Correct way to rename a DataFrame

        # ✅ Ensure final DataFrame only contains requested columns
        if isinstance(names, list):
            merged = merged[names]  # ✅ Drop unwanted columns

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
        A tuple or list of time series ranked from first to last in time.
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
        - `'prefer_first'`: Uses the earlier series until its last valid timestamp.
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
        - If a collection of single-column `Series` is provided, the output will 
          be a Series.
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


    if not isinstance(series, (tuple, list)) or len(series) == 0:
        raise ValueError("`series` must be a non-empty tuple or list of pandas.Series or pandas.DataFrame.")

    if not all(isinstance(s.index, pd.DatetimeIndex) for s in series):
        raise ValueError("All input series must have a DatetimeIndex.")

    if transition not in ["prefer_last", "prefer_first"] and not isinstance(transition, list):
        raise ValueError("`transition` must be 'prefer_last', 'prefer_first', or a list of timestamps.")

    first_freq = series[0].index.freq
    same_freq = all(s.index.freq is not None and s.index.freq == first_freq for s in series)

    # ✅ Define output columns properly
    if isinstance(series[0], pd.DataFrame):
        output_columns = list(series[0].columns)
    else:
        output_columns = [series[0].name]

    # ✅ Determine transition points properly
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

    transition_points = [None] + transition_points + [None]  # Add start and end

    # ✅ Apply correct transition logic
    sections = []
    for ts, start, end in zip(series, transition_points[:-1], transition_points[1:]):
        if isinstance(ts, pd.Series):
            sections.append(ts.loc[start:end])  # Keep original timestamps
        else:
            sections.append(ts.loc[start:end])  # Preserve column structure

    spliced = pd.concat(sections, axis=0)

    # ✅ Remove duplicates based on transition preference
    duplicated_times = spliced.index.duplicated(keep=duplicate_keep)
    if duplicated_times.sum() >= 1:
        spliced = spliced[~duplicated_times]

    if same_freq and first_freq is not None:
        spliced.index.freq = first_freq

    # ✅ Fix: Ensure proper renaming of Series and DataFrame columns
    if isinstance(spliced, pd.Series):
        if names:
            spliced = spliced.rename(names)  # ✅ Ensure renaming is correctly applied
    else:
        if isinstance(names, list):
            spliced = spliced[names]  # ✅ Drop extra columns
        elif isinstance(names, str):
            spliced = spliced.rename(columns={output_columns[0]: names})
        else:
            spliced = spliced[output_columns]  # ✅ Preserve column order

    return spliced


if __name__ == "__main__":
    main()
