#!/usr/bin/env python
# -*- coding: utf-8 -*-


__all__ = ["ts_merge", "ts_splice"]

import pandas as pd

def ts_merge(series, names=None):
    """
    Merge multiple time series together, prioritizing series in order.
    If all input series have the same frequency and are aligned, the output retains this frequency.
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



def ts_splice(tss, names=None, transition="prefer_last", floor_dates=False):
    """ splice a number of timeseries together in a chronological,
    non-overlapping way.The function supports three methods of positioning
    the breakpoints between series.

    Parameters
    ----------
    series  :  tuple(:class:`DataFrame <pandas:pandas.DataFrame>`)
    Series ranked from first to last in time. Must have identical column
    structure or unexpected results may occur during concatenate.
    See `names` below

    names : str
        names : str or list or iterable of str representing column names.
    If None, the default, the input series must share common names across all
    columns and not doing so will raise a ValueError If a string, all the
    DataFrame inputs must have the same number of columns as `names`, the
    outputs will be merged based on position as if these were the column
    names. Note that this may be an inherently dangerous operation if you
    aren't sure the columns line up. It is a big convenience for
    univariate series

    transition : 'prefer_first' | 'prefer_last' | list(pd.Datetime)
        Description of how to switch between series. If prefer_first, the
        earlier series will be used until their end point.If prefer_last,
        the first time stamp of the later series will be used.

    floor_dates: bool
        Floor the transition dates that are inferred with 'prefer_first' or
        'prefer_last'.Note that this can produce nans if the input series are
        regular with a freq attribute or a big gap otherwise unless there
        is overlapping data covering the days that are rounded.

    Returns
    -------
    spliced : :class:`DataFrame <pandas:pandas.DataFrame>`
        A new time series with freq same as the inputs if they are all the
        same,time extent the union of the inputs.
    """
    same_type = all(isinstance(item, type(tss[0])) for item in tss[1:])
    
    if not transition in ["prefer_last","prefer_first"]:
        raise ValueError("input transition must be prefer_last or \
                         prefer_first")
    
    if not same_type:
        raise ValueError("mixed input dataframe and series not supported")
    old_colname = []
    if not (names is None):
        if type(names) is str:  # all input series must have only one column
            i = 0
            for ddf in tss:
                if not(type(ddf) is pd.Series):
                    if (ddf.shape[1] != 1):
                        raise ValueError(f"{i}th input series have 0 or more\
                                         than 1 column")
                try:
                    if not(type(ddf) is pd.Series):
                        old_colname.append(ddf.columns[0])
                        ddf.rename(columns={old_colname[i]:names}, inplace=True)
                    else:
                        old_colname.append(ddf.name)
                        ddf.rename(names,inplace=True)
                except Exception as inst:
                    print(inst)
                    raise ValueError(f"fail to rename {i}th input to {names}")
                i = i + 1
        elif hasattr(names, "__iter__"):
            i = 0
            for ddf in tss:
                for name in names:
                    if not (name in ddf.columns):
                        raise ValueError(f"{i}th input series doesn't have \
                                         column {name}")
        else:
            raise ValueError("input names must be str or list or iteratble \
                             of strings")
    else:
        # input series must have the same column names
        names = tss[0].columns.to_list()
        i = 1
        for ddf in tss[1:]:
            if not (ddf.columns.to_list() == names):
                raise ValueError("{i}th series has different columns \
                                 than {names}")
            i = i + 1
    if isinstance(transition, list):
        if floor_dates:
            raise ValueError(
                "Floor dates option not compatible with transition that \
                    is a list of Timestamps")
        dup_keep = 'first'
    else:
        if transition == 'prefer_first':
            dup_keep = 'first'
            transition = [ts.last_valid_index() for ts in tss[:-1]]
            if floor_dates:
                transition = [dt.floor('D') for dt in transition]
        if transition == 'prefer_last':
            dup_keep = 'last'
            transition = [ts.first_valid_index() for ts in tss[1:]]
            if floor_dates:
                transition = [dt.floor('D') for dt in transition]
    transition = [None] + transition + [None]
    #print("transition\n", transition)
    sections = []
    f = tss[0].index.freq
    for ts, start, end in zip(tss, transition[:-1], transition[1:]):
        #print("here", start, end, "ts\n", ts)
        f = f if (f is not None) and ts.index.freq == f else None
        if type(ts) is pd.Series:
            sections.append(ts.loc[slice(start, end)])
        else:
            sections.append(ts[names].loc[slice(start, end)])
       
    tsout = pd.concat(sections, axis=0)
    duplicatetime = tsout.index.duplicated(keep=dup_keep)
    nduplicatetime = duplicatetime.sum()
    if nduplicatetime >= 1:
        tsout = tsout[~duplicatetime]
    if f is not None:
        tsout = tsout.asfreq(f)
    if type(names) is str:  # recover old name of input ts
        for ddf, name in zip(tss, old_colname):
            if not(type(ddf) is pd.Series):
                ddf.rename(columns={names:name}, inplace=True)
            else:
                ddf.rename(name,inplace=True)
        if not (type(tsout) is pd.Series):
            tsout=tsout.squeeze(axis=1)
            tsout.rename(names,inplace=True)
    return tsout


def main():
    import pandas as pd
    import numpy as np
    from vtools.data.vtime import hours, days

    dfs = []
    for i in range(4):
        dr = pd.date_range(pd.Timestamp(2000, 1, 1) +
                           hours(i*8), freq="H", periods=9)
        data = np.arange(1., 10.) + float(i*100.)
        df = pd.DataFrame(index=dr, data=data)
        dfs.append(df)
        print(df)
    ts_long = ts_splice(dfs, transition='prefer_first', floor_dates=False)
    print(ts_long)
    print("**")
    dfs = []
    for i in range(4):
        dr = pd.date_range(pd.Timestamp(2000, 1, 1) +
                           hours(i*8), freq="H", periods=11)
        data = np.arange(1., 12.) + float(i*100.)
        df = pd.DataFrame(index=dr, data=data)
        dfs.append(df)
        print(df)
    ts_long = ts_splice(dfs, transition='prefer_last', floor_dates=False)
    print(ts_long)

    transitions = [pd.Timestamp(2000, 1, 1, 9, 1), pd.Timestamp(
        2000, 1, 1, 17, 1), pd.Timestamp(2000, 1, 1, 22, 1)]
    print("***************")
    dfs = []
    for i in range(4):
        dr = pd.date_range(pd.Timestamp(2000, 1, 1) +
                           days(i*3)-hours(2), freq="H", periods=77)
        data = np.arange(1., 78.) + float(i*100.)
        df = pd.DataFrame(index=dr, data=data)
        df.index.freq = None
        dfs.append(df)
        print(df)
    ts_long = ts_splice(dfs, transition='prefer_last', floor_dates=True)
    print(ts_long.to_string())


if __name__ == "__main__":
    main()
