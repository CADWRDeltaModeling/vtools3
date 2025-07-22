import numpy as np
import pandas as pd
import numba
from vtools.functions.filter import cosine_lanczos

__all__ = [
    "get_tidal_hl",
    "get_tidal_amplitude",
    "get_tidal_hl_zerocrossing",
    "get_tidal_phase_diff",
]


def get_smoothed_resampled(
    df, cutoff_period="2h", resample_period="1min", interpolate_method="pchip"
):
    """Resample the dataframe (indexed by time) to the regular period of resample_period using the interpolate method

    Furthermore the cosine lanczos filter is used with a cutoff_period to smooth the signal to remove high frequency noise

    Args:

        df (DataFrame): A single column dataframe indexed by datetime

        cutoff_period (str, optional): cutoff period for cosine lanczos filter. Defaults to '2h'.

        resample_period (str, optional): Resample to regular period. Defaults to '1min'.

        interpolate_method (str, optional): interpolation for resampling. Defaults to 'pchip'.

    Returns:

        DataFrame: smoothed and resampled dataframe indexed by datetime
    """
    dfb = df.resample(resample_period).bfill()
    df = df.resample(resample_period).interpolate(method=interpolate_method)
    df[dfb.iloc[:, 0].isna()] = np.nan
    return cosine_lanczos(df, cutoff_period)


@numba.jit(nopython=True)
def lmax(arr):
    """Local maximum: Returns value only when centered on maximum"""
    idx = np.argmax(arr)
    if idx == len(arr) / 2:
        return arr[idx]
    else:
        return np.nan


@numba.jit(nopython=True)
def lmin(arr):
    """Local minimum: Returns value only when centered on minimum"""
    idx = np.argmin(arr)
    if idx == len(arr) / 2:
        return arr[idx]
    else:
        return np.nan


def periods_per_window(moving_window_size: str, period_str: str) -> int:
    """Number of period size in moving window

    Args:

        moving_window_size (str): moving window size as a string e.g 7H for 7 hour

        period_str (str): period as str e.g. 1T for 1 min

    Returns:

        int: number of periods in the moving window rounded to an integer
    """

    return int(
        pd.Timedelta(moving_window_size)
        / pd.to_timedelta(pd.tseries.frequencies.to_offset(period_str))
    )


def tidal_highs(df, moving_window_size="7h"):
    """Tidal highs (could be upto two highs in a 25 hr period)

    Args:

        df (DataFrame): a time series with a regular frequency

        moving_window_size (str, optional): moving window size to look for highs within. Defaults to '7h'.

    Returns:

        DataFrame: an irregular time series with highs at resolution of df.index
    """
    period_str = df.index.freqstr
    periods = periods_per_window(moving_window_size, period_str)
    dfmax = df.rolling(moving_window_size, min_periods=periods).apply(lmax, raw=True)
    dfmax = dfmax.shift(periods=-(periods // 2 - 1))
    dfmax = dfmax.dropna()
    dfmax.columns = ["max"]
    return dfmax


def tidal_lows(df, moving_window_size="7h"):
    """Tidal lows (could be upto two lows in a 25 hr period)

    Args:

        df (DataFrame): a time series with a regular frequency

        moving_window_size (str, optional): moving window size to look for lows within. Defaults to '7h'.

    Returns:

        DataFrame: an irregular time series with lows at resolution of df.index
    """
    period_str = df.index.freqstr
    periods = periods_per_window(moving_window_size, period_str)
    dfmin = df.rolling(moving_window_size, min_periods=periods).apply(lmin, raw=True)
    dfmin = dfmin.shift(periods=-(periods // 2 - 1))
    dfmin = dfmin.dropna()
    dfmin.columns = ["min"]
    return dfmin


def get_tidal_hl(
    df,
    cutoff_period="2h",
    resample_period="1min",
    interpolate_method="pchip",
    moving_window_size="7h",
):
    """Get Tidal highs and lows

    Args:

        df (DataFrame): A single column dataframe indexed by datetime

        cutoff_period (str, optional): cutoff period for cosine lanczos filter. Defaults to '2h'.

        resample_period (str, optional): Resample to regular period. Defaults to '1min'.

        interpolate_method (str, optional): interpolation for resampling. Defaults to 'pchip'.

        moving_window_size (str, optional): moving window size to look for lows within. Defaults to '7h'.

    Returns:

        tuple of DataFrame: Tidal high and tidal low time series
    """
    dfs = get_smoothed_resampled(df, cutoff_period, resample_period, interpolate_method)
    return tidal_highs(dfs), tidal_lows(dfs)


get_tidal_hl_rolling = get_tidal_hl  # for older refs. #FIXME


def get_tidal_amplitude(dfh, dfl):
    """Tidal amplitude given tidal highs and lows

    Args:

        dfh (DataFrame): Tidal highs time series

        dfl (DataFrame): Tidal lows time series

    Returns:

        DataFrame: Amplitude timeseries, at the times of the low following the high being used for amplitude calculation
    """
    dfamp = pd.concat([dfh, dfl], axis=1)
    dfamp = dfamp[["min"]].dropna().join(dfamp[["max"]].ffill())
    return pd.DataFrame(dfamp["max"] - dfamp["min"], columns=["amplitude"])


def get_tidal_amplitude_diff(dfamp1, dfamp2, percent_diff=False, tolerance="4h"):
    """Get the difference of values within +/- 4H of values in the two amplitude arrays

    Args:

        dfamp1 (DataFrame): Amplitude time series

        dfamp2 (DataFrame): Amplitude time series

        percent_diff (bool, optional): If true do percent diff. Defaults to False.

    Returns:

        DataFrame: Difference dfamp1-dfamp2 or % Difference (dfamp1-dfamp2)/dfamp2*100  for values within +/- 4H of each other
    """
    dfamp = pd.merge_asof(
        dfamp1,
        dfamp2,
        left_index=True,
        right_index=True,
        direction="nearest",
        tolerance=pd.Timedelta(tolerance),
    )
    if percent_diff:
        dfdiff = 100.0 * (dfamp.iloc[:, 0] - dfamp.iloc[:, 1]) / dfamp.iloc[:, 1]
    else:
        dfdiff = dfamp.iloc[:, 0] - dfamp.iloc[:, 1]
    return pd.DataFrame(dfdiff, columns=["amplitude_diff"])


def get_phase_diff(df1, df2, tolerance="4h"):
    df1["time"] = df1.index
    df2["time"] = df2.index
    df21 = pd.merge_asof(
        df2,
        df1,
        left_index=True,
        right_index=True,
        direction="nearest",
        tolerance=pd.Timedelta(tolerance),
    )
    return (df21["time_x"] - df21["time_y"]).apply(lambda x: x.total_seconds() / 60)


def get_tidal_phase_diff(dfh2, dfl2, dfh1, dfl1, tolerance="4h"):
    """Calculates the phase difference between df2 and df1 tidal highs and lows

    Scans +/- 4 hours in df1 to get the highs and lows in that windows for df2 to
    get the tidal highs and lows at the times of df1


    Args:

        dfh2 (DataFrame): Timeseries of tidal highs

        dfl2 (DataFrame): Timeseries of tidal lows

        dfh1 (DataFrame): Timeseries of tidal highs

        dfl1 (DataFRame): Timeseries of tidal lows

    Returns:

        DataFrame: Phase difference (dfh2-dfh1) and (dfl2-dfl1) in minutes
    """
    high_phase_diff = get_phase_diff(dfh2, dfh1, tolerance)
    low_phase_diff = get_phase_diff(dfl2, dfl1, tolerance)
    merged_diff = pd.merge(
        pd.DataFrame(high_phase_diff, index=dfh1.index),
        pd.DataFrame(low_phase_diff, index=dfl1.index),
        how="outer",
        left_index=True,
        right_index=True,
    )
    return merged_diff.iloc[:, 0].fillna(merged_diff.iloc[:, 1])


def get_tidal_hl_zerocrossing(df, round_to="1min"):
    """
    Finds the tidal high and low times using zero crossings of the first derivative.

    This works for all situations but is not robust in the face of noise and perturbations in the signal
    """
    zc, zi = zerocross(df)
    if round_to:
        zc = pd.to_datetime(zc).round(round_to)
    return zc


def zerocross(df):
    """
    Calculates the gradient of the time series and identifies locations where gradient changes sign
    Returns the time rounded to nearest minute where the zero crossing happens (based on linear derivative assumption)
    """
    diffdfv = pd.Series(np.gradient(df[df.columns[0]].values), index=df.index)
    indi = np.where((np.diff(np.sign(diffdfv))) & (diffdfv[1:] != 0))[0]
    # Find the zero crossing by linear interpolation
    zdb = diffdfv[indi].index
    zda = diffdfv[indi + 1].index
    x = diffdfv.index
    y = diffdfv.values
    dx = x[indi + 1] - x[indi]
    dy = y[indi + 1] - y[indi]
    zc = -y[indi] * (dx / dy) + x[indi]
    return zc, indi


##---- FUNCTIONS CACHED BELOW THIS LINE PERHAPS TO USE LATER? ---#


def where_changed(df):
    """ """
    diff = np.diff(df[df.columns[0]].values)
    wdiff = np.where(diff != 0)[0]
    wdiff = np.insert(wdiff, 0, 0)  # insert the first value i.e. zero index
    return df.iloc[wdiff + 1, :]


def where_same(dfg, df):
    """
    return dfg only where its value is the same as df for the same time stamps
    i.e. the interesection locations with df
    """
    dfall = pd.concat([dfg, df], axis=1)
    return dfall[dfall.iloc[:, 0] == dfall.iloc[:, 1]].iloc[:, 0]


def limit_to_indices(df, si, ei):
    return df[(df.index > si) & (df.index < ei)]


def filter_where_na(df, dfb):
    """
    remove values in df where dfb has na values
    """
    dfx = dfb.loc[df.index]
    return df.loc[dfx.dropna().index, :]
