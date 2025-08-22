import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks, savgol_filter
from statsmodels.nonparametric.smoothers_lowess import lowess


__all__ = ["generate_simplified_mixed_tide", "tidal_envelope"]


def chunked_loess_smoothing(ts, window_hours=1.25, chunk_days=10, overlap_days=1):
    """
    Apply LOESS smoothing in overlapping chunks to reduce computation time.

    Parameters
    ----------
    ts : pd.Series
        Time series with datetime index and possible NaNs.
    window_hours : float
        LOESS smoothing window size in hours.
    chunk_days : int
        Core chunk size (e.g., 10 days).
    overlap_days : int
        Overlap added before and after each chunk to avoid edge effects.

    Returns
    -------
    pd.Series
        Smoothed series, NaNs where input is NaN or unsupported.
    """
    freq = ts.index.freq or pd.infer_freq(ts.index)
    assert freq is not None, "Time index must have a frequency."
    dt = pd.to_timedelta(freq)
    points_per_hour = pd.Timedelta("1h") / dt

    # frac = window_hours * points_per_hour / len(ts)

    result = pd.Series(index=ts.index, dtype=float)

    chunk_len = int(chunk_days * 24 * points_per_hour)
    overlap_len = int(overlap_days * 24 * points_per_hour)

    for start in range(0, len(ts), chunk_len):
        i0 = max(0, start - overlap_len)
        i1 = min(len(ts), start + chunk_len + overlap_len)

        sub = ts.iloc[i0:i1]
        mask = sub.notna()
        if mask.sum() < 2:
            continue
        chunk_len_pts = sub[mask].shape[0]
        frac = min(1.0, (window_hours * points_per_hour) / max(chunk_len_pts, 1))
        smoothed = lowess(
            sub[mask], sub[mask].index.view("int64"), frac=frac, return_sorted=False
        )
        sub_smoothed = pd.Series(index=sub[mask].index, data=smoothed)

        # keep only the central region
        j0 = start
        j1 = min(start + chunk_len, len(ts))
        result.iloc[j0:j1] = sub_smoothed.reindex(ts.iloc[j0:j1].index)

    return result


def generate_pink_noise(n, seed=None, scale=1.0):
    """
    Generate pink (1/f) noise using the Voss-McCartney algorithm.

    Parameters
    ----------
    n : int
        Number of samples to generate.
    seed : int or None
        Random seed for reproducibility.
    scale : float
        Standard deviation scaling factor for the noise.

    Returns
    -------
    np.ndarray
        Pink noise signal of length n.
    """
    if seed is not None:
        np.random.seed(seed)

    n_rows = int(np.ceil(np.log2(n)))
    array = np.random.randn(n_rows, n)
    array = np.cumsum(array, axis=1)
    weight = 2 ** np.arange(n_rows)[:, None]
    pink = np.sum(array / weight, axis=0)

    pink -= pink.mean()
    pink /= pink.std()
    return scale * pink[:n]


def generate_simplified_mixed_tide(
    start_time="2022-01-01",
    ndays=40,
    freq="15min",
    A_M2=1.0,
    A_K1=0.5,
    A_O1=0.5,
    phase_D1=3.14159 / 2.0,
    noise_amplitude=0.08,
    return_components=False,
):
    """
    Generate a simplified synthetic mixed semidiurnal/diurnal tide with explicit O1 and K1.

    Parameters
    ----------
    start_time : str
        Start time for the series.
    ndays : int
        Number of days.
    freq : str
        Sampling interval.
    A_M2 : float
        Amplitude of M2.
    A_K1 : float
        Amplitude of K1.
    A_O1 : float
        Amplitude of O1.
    phase_D1 : float
        Common phase shift for O1 and K1.
    return_components : bool
        Whether to return individual components.

    Returns
    -------
    pd.Series or pd.DataFrame
        Combined tide or components with time index.
    """
    index = pd.date_range(
        start=start_time,
        periods=int(ndays * 24 * 60 / pd.Timedelta(freq).seconds * 60),
        freq=freq,
    )
    t_hours = (index - index[0]) / pd.Timedelta(hours=1)
    M2 = A_M2 * np.sin(2 * np.pi * t_hours / 12.42)
    K1 = A_K1 * np.sin(2 * np.pi * t_hours / 23.93 + phase_D1)
    O1 = A_O1 * np.sin(2 * np.pi * t_hours / 25.82 + phase_D1)
    K1 = A_K1 * np.sin(2 * np.pi * t_hours / 23.93 + phase_D1)
    O1 = A_O1 * np.sin(2 * np.pi * t_hours / 25.82 + phase_D1)
    noise = generate_pink_noise(len(index), scale=noise_amplitude) + np.random.normal(
        scale=noise_amplitude / 4, size=len(index)
    )

    tide = M2 + K1 + O1 + noise

    if return_components:
        return pd.DataFrame(
            {"M2": M2, "K1": K1, "O1": O1, "noise": noise, "tide": tide}, index=index
        )
    else:
        return pd.Series(tide, index=index, name="tide")


def smooth_series(ts, window_hours=1.75):
    return chunked_loess_smoothing(
        ts, window_hours=window_hours, chunk_days=10, overlap_days=1
    )


def smooth_series2(series, window_pts=25, method="lowess", **kwargs):
    """
    Smooth a time series using the specified method.
    Currently supports 'lowess', 'moving_average', or 'savgol'.
    """
    if method == "lowess":
        x = (series.index - series.index[0]).total_seconds()
        y = series.values
        frac = min(1.0, window_pts / len(series))
        y_smooth = lowess(y, x, frac=frac, return_sorted=False)
        return pd.Series(y_smooth, index=series.index)

    elif method == "moving_average":
        return series.rolling(window=window_pts, min_periods=1, center=True).mean()

    elif method == "savgol":
        # window length must be odd and greater than polyorder
        polyorder = kwargs.get("polyorder", 2)
        window = window_pts | 1  # make it odd
        if window <= polyorder:
            window = polyorder + 3 if (polyorder % 2 == 0) else polyorder + 2
        y = series.bfill().values
        y_smooth = savgol_filter(y, window_length=window, polyorder=polyorder)
        return pd.Series(y_smooth, index=series.index)

    else:
        raise ValueError(f"Unknown smoothing method: {method}")


def find_raw_extrema(smoothed, prominence=0.01):
    """
    Find raw peaks and troughs using scipy.signal.find_peaks.
    Returns DataFrames for peaks and troughs.
    """
    y = smoothed.values
    peaks, _ = find_peaks(y, prominence=prominence)
    troughs, _ = find_peaks(-y, prominence=prominence)
    df_peaks = pd.DataFrame(
        {"time": smoothed.index[peaks], "value": y[peaks], "type": "high"}
    )
    df_troughs = pd.DataFrame(
        {"time": smoothed.index[troughs], "value": y[troughs], "type": "low"}
    )
    return df_peaks, df_troughs


def filter_extrema_ngood(
    extrema_df, smoothed, series, loess_window_pts=25, n_good=3, sig_gap_minutes=45
):
    """
    Filter extrema based on local and contextual data quality criteria.

    Parameters
    ----------
    extrema_df : pd.DataFrame
        DataFrame with columns 'time' and 'value' for candidate extrema.
    smoothed : pd.Series
        Smoothed version of the signal used for extrema detection.
    series : pd.Series
        Original time series (with gaps).
    loess_window_pts : int
        Number of points in the LOESS window.
    n_good : int
        Minimum number of non-NaN points required.
    sig_gap_minutes : float
        Threshold for detecting significant gaps (in minutes).

    Returns
    -------
    pd.DataFrame
        Filtered extrema DataFrame.
    """
    sig_gap = pd.Timedelta(minutes=sig_gap_minutes)
    idx = series.index
    not_nan = ~series.isna()
    valid_times = idx[not_nan]

    # Identify significant gaps
    time_diffs = valid_times.to_series().diff()
    gap_starts = valid_times[time_diffs > sig_gap].tolist()
    gap_ends = valid_times[time_diffs.shift(-1) > sig_gap].tolist()

    idx_map = {t: i for i, t in enumerate(smoothed.index)}
    keep = []

    for t in extrema_df["time"]:
        i = idx_map.get(t, None)
        if i is None:
            keep.append(False)
            continue

        # 1. Local LOESS support
        hw = loess_window_pts // 2
        lo = max(0, i - hw)
        hi = min(len(smoothed), i + hw + 1)
        window_values = smoothed.iloc[lo:hi]
        local_ok = window_values.notna().sum() >= n_good

        # 2. Contextual support: enough valid data between nearest gap or boundary
        prev_gap = max(
            [g for g in gap_ends + [series.index[0]] if g < t], default=series.index[0]
        )
        next_gap = min(
            [g for g in gap_starts + [series.index[-1]] if g > t],
            default=series.index[-1],
        )
        n_before = valid_times[(valid_times > prev_gap) & (valid_times < t)]
        n_after = valid_times[(valid_times > t) & (valid_times < next_gap)]
        context_ok = (len(n_before) >= n_good) and (len(n_after) >= n_good)

        keep.append(local_ok and context_ok)

    return extrema_df[keep].reset_index(drop=True)


def select_salient_extrema(extrema, typ, spacing_hours=14, envelope_type="outer"):
    """
    Select salient extrema (HH/LL or HL/LH) using literal spacing-based OR logic.

    Parameters
    ----------
    extrema : pd.DataFrame with columns ["time", "value"]
        Candidate extrema.
    typ : str
        Either "high" or "low" (for peak or trough selection).
    spacing_hours : float
        Time window for neighbor comparison.
    envelope_type : str
        Either "outer" (default) or "inner" to switch saliency logic.

    Returns
    -------
    pd.DataFrame
        Extrema that passed the saliency test.
    """
    extrema = extrema.copy().sort_values("time").reset_index(drop=True)
    spacing = pd.Timedelta(hours=spacing_hours)

    passed_left = []
    passed_right = []
    kept = []

    for i, row in extrema.iterrows():
        t, v = row["time"], row["value"]

        left = extrema[(extrema["time"] < t) & (extrema["time"] >= t - spacing)]
        right = extrema[(extrema["time"] > t) & (extrema["time"] <= t + spacing)]

        if typ == "high":
            if envelope_type == "outer":
                left_ok = (left["value"] < v).any()
                right_ok = (right["value"] < v).any()
            else:
                left_ok = (left["value"] > v).any()
                right_ok = (right["value"] > v).any()
        else:
            if envelope_type == "outer":
                left_ok = (left["value"] > v).any()
                right_ok = (right["value"] > v).any()
            else:
                left_ok = (left["value"] < v).any()
                right_ok = (right["value"] < v).any()

        passed_left.append(left_ok)
        passed_right.append(right_ok)
        kept.append(left_ok or right_ok)

    extrema["passed_left"] = passed_left
    extrema["passed_right"] = passed_right
    extrema["kept"] = kept

    return (
        extrema[extrema["kept"]]
        .drop(columns=["passed_left", "passed_right", "kept"])
        .reset_index(drop=True)
    )


def interpolate_envelope(anchor_df, series, max_anchor_gap_hours=36):
    """
    Interpolate envelope using PCHIP, breaking if anchor points are too far apart.
    """
    if anchor_df.empty:
        return pd.Series(np.nan, index=series.index)
    times = pd.to_datetime(anchor_df["time"])
    values = anchor_df["value"].values
    t0 = series.index[0]
    x = (times - t0).dt.total_seconds().values
    xi = (series.index - t0).total_seconds()
    breaks = np.where(np.diff(x) > max_anchor_gap_hours * 3600)[0]
    env = np.full(len(series.index), np.nan)
    seg_starts = np.concatenate(([0], breaks + 1))
    seg_ends = np.concatenate((breaks, [len(x) - 1]))
    for s, e in zip(seg_starts, seg_ends):
        if e - s < 1:
            continue
        pchip = PchipInterpolator(x[s : e + 1], values[s : e + 1])
        mask = (xi >= x[s]) & (xi <= x[e])
        env[mask] = pchip(xi[mask])
    env_series = pd.Series(env, index=series.index)
    env_series[series.isna()] = np.nan
    return env_series


# --- Main envelope extraction pipeline ---


def tidal_envelope(
    series,
    smoothing_window_hours=2.5,
    n_good=3,
    peak_prominence=0.05,
    saliency_window_hours=14,
    max_anchor_gap_hours=36,
    envelope_type="outer",
):
    """
    Compute the tidal envelope (high and low) of a time series using smoothing, extrema detection, and interpolation.
    This function processes a time series to extract its tidal envelope by smoothing the data, identifying significant peaks and troughs, filtering out unreliable extrema, selecting salient extrema within a specified window, and interpolating between anchor points to generate continuous envelope curves.
    Parameters
    ----------
    series : pandas.Series
        Time-indexed series of water levels or similar data.
    smoothing_window_hours : float, optional
        Window size in hours for smoothing the input series (default is 2.5).
    n_good : int, optional
        Minimum number of good points required for an extremum to be considered valid (default is 3).
    peak_prominence : float, optional
        Minimum prominence of peaks/troughs to be considered as extrema (default is 0.05).
    saliency_window_hours : float, optional
        Window size in hours for selecting salient extrema (default is 14).
    max_anchor_gap_hours : float, optional
        Maximum allowed gap in hours between anchor points for interpolation (default is 36).
    envelope_type : str, optional
        Type of envelope to compute, e.g., "outer" (default is "outer").
    Returns
    -------
    env_high : pandas.Series
        Interpolated high (upper) envelope of the input series.
    env_low : pandas.Series
        Interpolated low (lower) envelope of the input series.
    anchor_highs : pandas.DataFrame
        DataFrame of selected anchor points for the high envelope.
    anchor_lows : pandas.DataFrame
        DataFrame of selected anchor points for the low envelope.
    smoothed : pandas.Series
        Smoothed version of the input series.
    Notes
    -----
    This function assumes regular time intervals in the input series. If the frequency cannot be inferred, it is estimated from the first two timestamps.
    """

    # Smoothing
    freq = pd.infer_freq(series.index)
    if freq is None:
        freq = pd.Timedelta(series.index[1] - series.index[0])
    else:
        freq = pd.Timedelta(freq)
    window_pts = int(smoothing_window_hours * 60 / (freq.total_seconds() / 60))
    # smoothed = smooth_series(series, window_hours=smoothing_window_hours)
    smoothed = smooth_series2(
        series, window_pts=window_pts, method="savgol", polyorder=2
    )

    # Raw extrema
    df_peaks, df_troughs = find_raw_extrema(smoothed, prominence=peak_prominence)

    # Eliminate one-sided extrema near gaps/boundaries
    df_peaks = filter_extrema_ngood(
        df_peaks, smoothed, series, loess_window_pts=window_pts, n_good=n_good
    )
    df_troughs = filter_extrema_ngood(
        df_troughs, smoothed, series, loess_window_pts=window_pts, n_good=n_good
    )

    # Saliency selection
    anchor_highs = select_salient_extrema(
        df_peaks, "high", saliency_window_hours, envelope_type
    )
    anchor_lows = select_salient_extrema(
        df_troughs, "low", saliency_window_hours, envelope_type
    )

    # Interpolation
    env_high = interpolate_envelope(anchor_highs, series, max_anchor_gap_hours)
    env_low = interpolate_envelope(anchor_lows, series, max_anchor_gap_hours)

    return env_high, env_low, anchor_highs, anchor_lows, smoothed


def main():
    components = generate_simplified_mixed_tide(return_components=True)
    tide = components["tide"].copy()
    tide.iloc[500:600] = np.nan
    tide.iloc[2000:2100] = np.nan

    env_high, env_low, anchor_highs, anchor_lows, smooth = tidal_envelope(
        tide, envelope_type="outer"
    )
    env_high_in, env_low_in, anchor_highs_in, anchor_lows_in, _ = tidal_envelope(
        tide, envelope_type="inner"
    )

    plt.figure(figsize=(12, 6))
    plt.plot(tide, label="Noisy Gappy Tide", alpha=0.5)
    plt.plot(smooth, label="LOESS Smooth", color="gray", linestyle="--")
    plt.plot(env_high, label="Outer Upper Envelope (HHW)", color="red")
    plt.plot(env_low, label="Outer Lower Envelope (LLW)", color="blue")
    plt.plot(env_high_in, label="Inner Upper Envelope (HLW)", color="orange")
    plt.plot(env_low_in, label="Inner Lower Envelope (LHW)", color="purple")
    plt.scatter(
        anchor_highs["time"],
        anchor_highs["value"],
        color="red",
        marker="^",
        label="HHW",
    )
    plt.scatter(
        anchor_lows["time"], anchor_lows["value"], color="blue", marker="v", label="LLW"
    )
    plt.scatter(
        anchor_highs_in["time"],
        anchor_highs_in["value"],
        color="orange",
        marker="^",
        label="HLW",
    )
    plt.scatter(
        anchor_lows_in["time"],
        anchor_lows_in["value"],
        color="purple",
        marker="v",
        label="LHW",
    )
    plt.legend()
    plt.title("Tidal Envelope Extraction: Outer vs Inner")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
