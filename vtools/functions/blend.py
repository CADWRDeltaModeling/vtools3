
import pandas as pd
import numpy as np
from functools import reduce
from vtools import to_timedelta
from vtools.functions.colname_align import align_inputs_strict
from vtools.data.gap import gap_distance
from vtools.functions.merge import _reindex_to_continuous

__all__ = ["ts_blend"]



def _distance_to_gap(hi_col: pd.Series, mode: str = "count") -> pd.Series:
    """
    Distance to nearest gap (NaN) in hi_col.

    Parameters
    ----------
    hi_col : Series
        Higher-priority series.
    mode : {'count', 'freq'}
        'count' -> distance in # of samples (0 at gaps).
        'freq'  -> distance as Timedelta, using hi_col.index.freq.

    Returns
    -------
    Series
        Same index as hi_col, distance to nearest NaN.
    """
    idx = hi_col.index
    n = len(idx)
    mask = hi_col.isna().to_numpy()

    # No gaps -> everything is effectively "far away"
    if not mask.any():
        dist = np.full(n, np.inf, dtype=float)
        return pd.Series(dist, index=idx)

    dist = np.full(n, np.inf, dtype=float)

    # Forward pass: distance from the last gap
    last_gap = None
    for i in range(n):
        if mask[i]:
            dist[i] = 0.0
            last_gap = i
        elif last_gap is not None:
            dist[i] = float(i - last_gap)

    # Backward pass: distance from the next gap
    last_gap = None
    for i in range(n - 1, -1, -1):
        if mask[i]:
            last_gap = i
        elif last_gap is not None:
            dist[i] = min(dist[i], float(last_gap - i))

    dist_s = pd.Series(dist, index=idx)

    if mode == "count":
        return dist_s

    if mode == "freq":
        freq = idx.freq
        if freq is None:
            raise ValueError("Time-based blending requires a regular index with .freq set.")
        # counts * freq → Timedelta
        return dist_s * to_timedelta(freq)

    raise ValueError("mode must be 'count' or 'freq'")


def _normalize_blend_length(blend_length, index):
    """
    Interpret blend_length as sample count or time span.

    Returns
    -------
    (mode, L)
        mode : {'count', 'freq'} or None
        L    : numeric (count) or Timedelta
    """
    if blend_length is None:
        return None, None
    if isinstance(blend_length, str):
        blend_length = blend_length.replace("H", "h")
        blend_length = blend_length.replace("d", "D")

    # Integer: number of samples
    if isinstance(blend_length, (int, np.integer)):
        if blend_length <= 0:
            return None, None
        return "count", float(blend_length)

    # Timedelta-like: e.g. '2h', '30min'
    td = pd.to_timedelta(blend_length)
    if not isinstance(index, (pd.DatetimeIndex, pd.PeriodIndex)):
        raise ValueError("Time-based blend_length requires a DatetimeIndex or PeriodIndex.")
    if index.freq is None:
        raise ValueError("Time-based blend_length requires a regular index with a .freq attribute.")
    if td <= pd.Timedelta(0):
        return None, None

    return "freq", td


def _blend_two(
    aligned_hi: pd.DataFrame,
    aligned_lo: pd.DataFrame,
    blend_mode: str,
    blend_L,
) -> pd.DataFrame:
    """
    Blend a lower-priority DataFrame into a higher-priority DataFrame.

    Parameters
    ----------
    aligned_hi, aligned_lo : DataFrame
        Same index. Higher priority is 'aligned_hi'.
    blend_mode : {'count', 'freq'} or None
    blend_L : float or Timedelta

    Returns
    -------
    DataFrame
        Blended result.
    """
    # No blending requested → just do priority overlay
    if blend_mode is None or blend_L is None:
        return aligned_hi.combine_first(aligned_lo)

    idx = aligned_hi.index
    out = aligned_hi.copy()
    cols = sorted(set(aligned_hi.columns) | set(aligned_lo.columns))

    for col in cols:
        hi_col = aligned_hi[col] if col in aligned_hi.columns else pd.Series(index=idx, dtype=float)
        lo_col = aligned_lo[col] if col in aligned_lo.columns else pd.Series(index=idx, dtype=float)

        hi_nan = hi_col.isna()
        lo_nan = lo_col.isna()

        # Priority baseline: hi where present, otherwise lo
        merged = hi_col.copy()
        fill_mask = hi_nan & (~lo_nan)
        merged[fill_mask] = lo_col[fill_mask]

        # Distance to nearest gap in the *high-priority* series
        dist_to_gap = _distance_to_gap(
            hi_col,
            mode="count" if blend_mode == "count" else "freq",
        )

        # Candidate points for blending on the shoulders of gaps:
        # - hi has data
        # - lo has data
        near_gap = (~hi_nan) & (~lo_nan)

        if blend_mode == "count":
            near_gap &= (dist_to_gap > 0) & (dist_to_gap <= blend_L)
            if not near_gap.any():
                out[col] = merged
                continue
            d = dist_to_gap[near_gap].astype(float)
            t = (blend_L - d) / blend_L
        else:  # 'freq' mode (Timedelta)
            near_gap &= (dist_to_gap > pd.Timedelta(0)) & (dist_to_gap <= blend_L)
            if not near_gap.any():
                out[col] = merged
                continue
            d = dist_to_gap[near_gap]
            t = 1.0 - (d / blend_L)

        t = t.clip(lower=0.0, upper=1.0)

        # Kernel: lower-priority gets up to 0.5 weight at the gap edge,
        # tapering to 0 at distance >= blend_L.
        w_lo = 0.5 * t
        w_hi = 1.0 - w_lo

        hi_vals = hi_col[near_gap].astype(float)
        lo_vals = lo_col[near_gap].astype(float)

        blended_vals = (
            w_hi.to_numpy() * hi_vals.to_numpy()
            + w_lo.to_numpy() * lo_vals.to_numpy()
        )

        # IMPORTANT: use .loc with a boolean mask, not .at, so we never hit
        # DataFrame._set_value with a non-scalar index.
        merged.loc[near_gap] = blended_vals

        out[col] = merged

    return out


@align_inputs_strict(seq_arg=0, names_kw="names")
def ts_blend(
    series,
    names=None,
    blend_length=None,
):
    """
    Blend multiple time series together, using higher priority where possible,
    but ramping in lower-priority data near gaps in the higher-priority series.

    Parameters
    ----------
    series : sequence of pandas.Series or pandas.DataFrame
        Higher priority first. All indexes must be DatetimeIndex or PeriodIndex.
    names : None, str, or iterable of str, optional
        Same semantics as ts_merge / ts_splice.
    blend_length : int or Timedelta-like, optional
        Controls the width of the blending zone around gaps in the
        higher-priority series:

        - If an integer `N` is given, then up to `N` samples on each side of
          any gap in the higher-priority series will be blended using a kernel
          based on the distance to the gap edge (in sample counts).
        - If a Timedelta-like value (e.g. '2H', pd.Timedelta('30min')), then
          a regular DatetimeIndex with `.freq` is required, and distances are
          measured in time.

        If None or non-positive, ts_blend behaves like a hard-priority merge
        (equivalent to ts_merge with strict_priority=False).

    Returns
    -------
    pandas.Series or pandas.DataFrame
        A time series combining all inputs, with soft transitions near gaps.
    """
    if not isinstance(series, (list, tuple)) or len(series) == 0:
        raise ValueError("`series` must be a non-empty tuple or list")

    if not all(
        isinstance(getattr(s, "index", None), (pd.DatetimeIndex, pd.PeriodIndex))
        for s in series
    ):
        raise ValueError("All input series must have a DatetimeIndex or PeriodIndex.")

    # Preserve first series freq (may be None)
    first_freq = getattr(series[0].index, "freq", None)

    # If any DataFrame is present, normalize all to DataFrame
    any_df = any(isinstance(s, pd.DataFrame) for s in series)
    if any_df:
        series = [s.to_frame(name=s.name) if isinstance(s, pd.Series) else s for s in series]

    all_df     = all(isinstance(s, pd.DataFrame) for s in series)
    any_series = any(isinstance(s, pd.Series)     for s in series)

    # Column compatibility checks similar to ts_merge
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

    # Normalize blend_length against the union index
    blend_mode, blend_L = _normalize_blend_length(blend_length, full_index)

    # Align all to union index and normalize to DataFrames
    aligned = []
    for s in series:
        a = s.reindex(full_index)
        if isinstance(a, pd.Series):
            a = a.to_frame(name=a.name)
        aligned.append(a.copy())

    # Start from top priority and fold in lower priorities with blending
    blended = aligned[0]
    for lo in aligned[1:]:
        blended = _blend_two(blended, lo, blend_mode, blend_L)

    # If all inputs were univariate Series, return a Series
    all_series = all(isinstance(s, pd.Series) for s in series)
    if all_series:
        blended = blended.squeeze()
    elif isinstance(blended, pd.Series):
        blended = blended.to_frame()

    # Reindex to a continuous index using the first series' freq
    blended = _reindex_to_continuous(blended, first_freq)

    return blended
