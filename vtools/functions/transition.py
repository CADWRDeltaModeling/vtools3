from scipy.interpolate import PchipInterpolator
import pandas as pd
import numpy as np
from pandas.tseries.frequencies import to_offset
from vtools.functions.colname_align import align_inputs_pair_strict

__all__ = ["transition_ts"]


def _parse_max_snap(max_snap):
    if max_snap is None:
        return pd.Timedelta(0), pd.Timedelta(0)
    if isinstance(max_snap, (pd.Timedelta, str)):
        t = pd.Timedelta(max_snap)
        return t, t
    if isinstance(max_snap, (tuple, list)) and len(max_snap) == 2:
        left = pd.Timedelta(max_snap[0])
        right = pd.Timedelta(max_snap[1])
        return left, right
    raise ValueError("max_snap must be None, a Timedelta-like, or a (left,right) pair.")


def _resolve_gap_endpoints_subset_snap(ts0, ts1, window, max_snap=None):
    """
    Contract:
      - If window is None:
          * If there's a natural gap (ts0.last < ts1.first), use that full gap.
          * Otherwise (overlap/abut), return None to signal 'no explicit gap' (algorithms decide).
      - If window is provided:
          * Enforce: start < end; ts0 has <= start; ts1 has >= end. Else: ValueError.
          * If there is a natural gap AND (start,end) is a strict subset of it,
            expand start left and end right by up to max_snap (default 0) but never beyond
            the natural gap bounds. Otherwise, ignore max_snap.
          * Always snap endpoints to data: start_time = last ts0 sample <= effective start,
            end_time = first ts1 sample >= effective end.
    Returns:
        (start_time, end_time) or None if no explicit gap is to be used.
    """
    last0 = ts0.index.max()
    first1 = ts1.index.min()
    natural_gap = last0 < first1

    # No explicit gap
    if window is None:
        if natural_gap:
            # full natural gap
            start_time = ts0.loc[:last0].index[-1]  # == last0
            end_time = ts1.loc[first1:].index[0]  # == first1
            return start_time, end_time
        return None  # overlap/abut: algorithms handle

    # Explicit gap provided
    start = pd.Timestamp(window[0])
    end = pd.Timestamp(window[1])
    if start >= end:
        raise ValueError("window start must be strictly before end.")

    # Strict domain checks (no salvage for out-of-bounds)
    if ts0.loc[:start].empty:
        first0 = ts0.index.min()
        raise ValueError(
            f"window start {start} is before the first ts0 sample ({first0}); "
            f"no ts0 samples at or before start."
        )
    if ts1.loc[end:].empty:
        last1 = ts1.index.max()
        raise ValueError(
            f"window end {end} is after the last ts1 sample ({last1}); "
            f"ts1 must have a sample at or after end."
        )

    # Compute effective window
    eff_start, eff_end = start, end
    if natural_gap and (last0 < start < end < first1):
        left_snap, right_snap = _parse_max_snap(max_snap)
        # widen but never exceed natural gap bounds
        eff_start = max(start - left_snap, last0)
        eff_end = min(end + right_snap, first1)

    # Snap to actual data samples
    start_time = ts0.loc[:eff_start].index[-1]  # last <= eff_start
    end_time = ts1.loc[eff_end:].index[0]  # first >= eff_end
    if start_time >= end_time:
        # Shouldn't happen for legitimate gaps; guard just in case
        raise ValueError("Effective gap has no extent; adjust window or max_snap.")
    return start_time, end_time


@align_inputs_pair_strict(ts0_kw="ts0", ts1_kw="ts1", names_kw="names")
def transition_ts(
    ts0,
    ts1,
    method="linear",
    window=None,  # [start, end] or None
    overlap=(0, 0),  # as you already have
    return_type="series",
    names=None,
    max_snap=None,  # NEW: None (0h), "1D", or (left,right)
):
    """Create a smooth transition between two aligned time series.


    Parameters
    ----------
    ts0 : pandas.Series or pandas.DataFrame
        The initial time series segment. Must share the same frequency and type as `ts1`.

    ts1 : pandas.Series or pandas.DataFrame
        The final time series segment. Must share the same frequency and type as `ts0`.

    method : {"linear", "pchip", "blend"}, default="linear"
        The interpolation strategy:
        - "linear": interpolate across a gap using endpoints from ts0/ts1.
        - "pchip": shape-preserving interpolation using nearby points (see `overlap`).
        - "blend": requires an explicit `window=(start, end)` where both ts0 and ts1
          have values on every timestamp; returns a linear combination
          (1 - w(t)) * ts0(t) + w(t) * ts1(t) with w(start)=0 → w(end)=1.

    window : [start, end] or None
        - For "linear"/"pchip": If None and there's a natural gap (ts0.last < ts1.first),
          that full gap is used. If provided, start<end, ts0 must have a sample at/before
          start and ts1 at/after end; optional widening to a natural gap via `max_snap`.
        - For "blend": **Required.** Both series must cover every timestamp in
          [start, end] with non-missing values; no widening or gap logic is applied.

    names : None, str, or iterable of str, optional
        - If `None` (default), inputs must share compatible column names.
        - If `str`, the output is univariate and will be named accordingly.
        - If iterable, it is used as a subset/ordering of columns.

    overlap : tuple of int or str, default=(0, 0)
        Amount of overlap to use for interpolation anchoring in `pchip` mode.
        Each entry can be:
        - An integer: number of data points before/after to use.
        - A pandas-compatible frequency string: e.g., "2h" or "45min".

    max_snap : None | Timedelta-like | (Timedelta-like, Timedelta-like)
        Optional widening ONLY when window is strictly inside the natural gap.
        Expands start earlier and end later by up to max_snap, but never past
        (ts0.last, ts1.first). Default None = no widening.

    return_type : {"series", "glue"}, default="series"
        - "series": returns the full merged series including ts0, transition, ts1.
        - "glue": returns only the interpolated transition segment.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        The resulting time series segment, either the full merged series or just the transition zone.

    Raises
    ------
    ValueError
        If ts0 and ts1 have mismatched types or frequencies, or if overlap exists but `window` is not specified.
    """
    if not isinstance(ts0, (pd.Series, pd.DataFrame)) or not isinstance(ts1, type(ts0)):
        raise ValueError("ts0 and ts1 must be of the same type (Series or DataFrame).")
    freq0 = ts0.index.freq or ts0.index.inferred_freq
    freq1 = ts1.index.freq or ts1.index.inferred_freq
    if freq0 is None or freq1 is None:
        raise ValueError("ts0 and ts1 must be regular time series with a resolvable frequency.")

    freq0 = to_offset(freq0)
    freq1 = to_offset(freq1)
    if freq0 != freq1:
        raise ValueError("ts0 and ts1 must have the same frequency.")

    freq = freq0


    # --- BLEND mode: explicit overlap with non-missing values in both series ---
    if method == "blend":
        if window is None:
            raise ValueError("method='blend' requires window=(start, end).")
        start = pd.Timestamp(window[0])
        end = pd.Timestamp(window[1])
        if start >= end:
            raise ValueError("blend window start must be strictly before end.")
        # exact inclusive grid for the blend interval
        trans_index = pd.date_range(start=start, end=end, freq=freq)
        if len(trans_index) < 2:
            raise ValueError("blend window must contain at least two timestamps.")
        # require full coverage with no NaNs
        try:
            seg0 = ts0.loc[trans_index]
            seg1 = ts1.loc[trans_index]
        except KeyError:
            raise ValueError("Both series must cover every timestamp in the blend window.")
        if isinstance(ts0, pd.DataFrame):
            if seg0.isna().any().any() or seg1.isna().any().any():
                raise ValueError("NaNs found within blend window in ts0/ts1.")
            w = np.linspace(0.0, 1.0, len(trans_index))[:, None]
            blended_vals = (1.0 - w) * seg0.to_numpy(dtype=float) + w * seg1.to_numpy(dtype=float)
            blended = pd.DataFrame(blended_vals, index=trans_index, columns=ts0.columns)
        else:
            if seg0.isna().any() or seg1.isna().any():
                raise ValueError("NaNs found within blend window in ts0/ts1.")
            w = np.linspace(0.0, 1.0, len(trans_index))
            blended_vals = (1.0 - w) * seg0.to_numpy(dtype=float) + w * seg1.to_numpy(dtype=float)
            blended = pd.Series(blended_vals, index=trans_index, name=ts0.name)
        # splice: ts0 before start, blend in [start,end], ts1 after end
        if return_type == "glue":
            return blended
        elif return_type == "series":
            left = ts0.loc[ts0.index < start]
            right = ts1.loc[ts1.index > end]
            return pd.concat([left, blended, right])
        else:
            raise ValueError("return_type must be either 'glue' or 'series'.")



    # `resolved` is either:
    #   • (start_time, end_time): data-aligned gap anchors computed from `window`
    #     (and, if applicable, widened inside the natural gap by `max_snap`), where
    #     start_time = last ts0 sample ≤ start and end_time = first ts1 sample ≥ end.
    #   • None: no `window` given and no natural gap (the series overlap/abut).
    # In the None case we fall back to adjacent endpoints (ts0.last, ts1.first) and let
    # the width guard (`len(trans_index) < 2`) decide if there’s room to transition.
    resolved = _resolve_gap_endpoints_subset_snap(
        ts0, ts1, window, max_snap=max_snap
    )

    if resolved is None:
        # Fall back to adjacent endpoints even if series abut/overlap;
        # the width check below will emit the expected “at least two steps” error.
        start_time = ts0.index[-1]
        end_time = ts1.index[0]
    else:
        start_time, end_time = resolved

    # Interior of the gap (exclusive of anchors)
    trans_start = start_time + freq
    trans_end = end_time - freq
    trans_index = pd.date_range(start=trans_start, end=trans_end, freq=freq)
    # ONLY error on short width for natural-gap (window is None)
    require_two_steps = window is None
    if require_two_steps and len(trans_index) < 2:
        raise ValueError("Transition zone must have at least two steps.")

    # Anchor values
    start_val = ts0.loc[start_time]
    end_val = ts1.loc[end_time]

    # Interpolation
    if method == "linear":
        total_duration = (end_time - start_time).total_seconds()
        rel = np.asarray(
            [(t - start_time).total_seconds() / total_duration for t in trans_index],
            dtype=float,
        )

        if isinstance(ts0, pd.DataFrame):
            start_vec = start_val.to_numpy(dtype=float)
            end_vec = end_val.to_numpy(dtype=float)
            mat = np.outer(1.0 - rel, start_vec) + np.outer(rel, end_vec)
            interpolated = pd.DataFrame(mat, index=trans_index, columns=ts0.columns)
        else:
            vals = (1.0 - rel) * float(start_val) + rel * float(end_val)
            interpolated = pd.Series(vals, index=trans_index, name=ts0.name)

    elif method == "pchip":
        n_before, n_after = overlap
        if isinstance(n_before, str):
            n_before = int(pd.Timedelta(n_before) / freq)
        if isinstance(n_after, str):
            n_after = int(pd.Timedelta(n_after) / freq)

        seg0 = (
            ts0.loc[:trans_start].iloc[-n_before:]
            if n_before > 0
            else ts0.loc[[ts0.index[-1]]]
        )
        seg1 = (
            ts1.loc[trans_end:].iloc[:n_after]
            if n_after > 0
            else ts1.loc[[ts1.index[0]]]
        )
        all_data = pd.concat([seg0, seg1])

        if isinstance(ts0, pd.Series):
            interp = PchipInterpolator(all_data.index.astype(np.int64), all_data.values)
            interpolated = pd.Series(
                interp(trans_index.astype(np.int64)), index=trans_index, name=ts0.name
            )
        else:
            interpolated = pd.DataFrame(index=trans_index, columns=ts0.columns)
            for col in ts0.columns:
                interp = PchipInterpolator(
                    all_data.index.astype(np.int64), all_data[col].values
                )
                interpolated[col] = interp(trans_index.astype(np.int64))
    else:
        raise ValueError("Only 'linear' and 'pchip' and 'blend' methods are supported.")

    # Final output
    if return_type == "glue":
        # include anchors at start_time and end_time
        if isinstance(ts0, pd.DataFrame):
            start_df = start_val.to_frame().T
            start_df.index = pd.DatetimeIndex([start_time])
            end_df = end_val.to_frame().T
            end_df.index = pd.DatetimeIndex([end_time])
            return pd.concat([start_df, interpolated, end_df])
        else:
            start_s = pd.Series([start_val], index=[start_time], name=ts0.name)
            end_s = pd.Series([end_val], index=[end_time], name=ts0.name)
            return pd.concat([start_s, interpolated, end_s])

    elif return_type == "series":
        ts0_trunc = ts0.loc[:start_time]
        ts1_trunc = ts1.loc[end_time:]
        return pd.concat([ts0_trunc, interpolated, ts1_trunc])

    else:
        raise ValueError("return_type must be either 'glue' or 'series'.")
