from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------


def _prepare_df(df: pd.DataFrame):
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df must have a DatetimeIndex")
    if df.index.has_duplicates:
        raise ValueError("Duplicate timestamps in input")

    t = df.index.to_numpy(dtype="datetime64[ns]")
    order = np.argsort(t, kind="mergesort")
    df = df.iloc[order].copy()
    return df, df.index.to_numpy(dtype="datetime64[ns]"), df.to_numpy(dtype=float)


def _hysteretic_quantize(X: np.ndarray, q: float, hyst: float) -> np.ndarray:
    if q <= 0:
        raise ValueError("qwidth must be > 0")
    if hyst <= 0:
        raise ValueError("hyst must be > 0")

    half = hyst * q
    Xq = np.empty_like(X, float)

    Xq[0] = np.round(X[0] / q) * q
    for i in range(1, len(X)):
        y = Xq[i - 1]
        d = X[i] - y
        n = np.zeros_like(y)

        up = d > half
        dn = d < -half
        n[up] = np.ceil((d[up] - half) / q)
        n[dn] = -np.ceil(((-d[dn]) - half) / q)

        Xq[i] = y + n * q

    return Xq


def _segments_from_bool(b: np.ndarray):
    if len(b) == 0:
        return [], [], []
    change = np.empty(len(b), bool)
    change[0] = True
    change[1:] = b[1:] != b[:-1]
    s = np.flatnonzero(change)
    e = np.r_[s[1:], len(b)]
    return s, e, b[s]


def _stable_state_from_raw(t, raw, enter, exit):
    s, e, v = _segments_from_bool(raw)
    stable = np.zeros_like(raw, bool)

    dur = (t[e - 1] - t[s]).astype("timedelta64[ns]")
    for si, ei, vi, di in zip(s, e, v, dur):
        if vi and di >= enter:
            stable[si:ei] = True

    s2, e2, v2 = _segments_from_bool(stable)
    dur2 = (t[e2 - 1] - t[s2]).astype("timedelta64[ns]")
    for k in range(1, len(s2) - 1):
        if (not v2[k]) and v2[k - 1] and v2[k + 1] and dur2[k] < exit:
            stable[s2[k] : e2[k]] = True

    return stable


def _change_points_mask(X):
    keep = np.zeros(len(X), bool)
    if len(X):
        keep[0] = True
        keep[1:] = np.any(X[1:] != X[:-1], axis=1)
    return keep


def _preserve_boundary_mask(stable):
    keep = np.zeros(len(stable), bool)
    if len(stable) == 0:
        return keep

    keep[0] = True

    b = np.flatnonzero(stable[1:] != stable[:-1]) + 1  # boundary indices
    keep[b] = True
    keep[b - 1] = True  # NEW: also keep the sample just before the boundary

    return keep


def _heartbeat_mask(t, freq):
    if freq is None:
        return np.zeros(len(t), bool)
    key = t.floor(freq).to_numpy()
    change = np.empty(len(t), bool)
    change[0] = True
    change[1:] = key[1:] != key[:-1]
    s = np.flatnonzero(change)
    e = np.r_[s[1:], len(t)]
    keep = np.zeros(len(t), bool)
    keep[e - 1] = True
    return keep


def _emit_df(t, cols, X, keep, force_keep=None):
    idx = np.flatnonzero(keep)
    out = pd.DataFrame(X[idx], index=t[idx], columns=cols)

    if len(out) <= 1:
        return out

    v = out.to_numpy()
    km = np.ones(len(out), dtype=bool)

    # Drop consecutive duplicates by value
    km[1:] = np.any(v[1:] != v[:-1], axis=1)

    # But never drop explicitly forced rows (e.g. exit points)
    if force_keep is not None:
        fk = force_keep[idx]
        km[1:] |= fk[1:]

    return out.loc[km]



def _grid_force_pre_exit_from_raw(
    t_raw: np.ndarray,
    stable_raw: np.ndarray,
    t_grid: pd.DatetimeIndex,
    grid: str,
) -> np.ndarray:
    """Mark pre-exit + exit ticks on the grid based on *raw* stable transitions.

    In grid mode, exits can occur inside the final grid bin, so a grid-derived stable mask
    (from ffill/bfill) may never show the exit. This helper maps each raw True->False
    transition to:

      exit_tick     = floor(exit_time, grid)
      pre_exit_tick = exit_tick - one grid interval

    Any such ticks that exist in t_grid are marked True.
    """
    keep = np.zeros(len(t_grid), dtype=bool)
    if len(stable_raw) < 2:
        return keep

    off = pd.tseries.frequencies.to_offset(grid)
    exit_i = np.flatnonzero(stable_raw[:-1] & (~stable_raw[1:]))
    if len(exit_i) == 0:
        return keep

    for i in exit_i:
        exit_time = pd.Timestamp(t_raw[i + 1])
        g_exit = exit_time.floor(grid)
        g_pre = g_exit - off
        for tt in (g_pre, g_exit):
            j = t_grid.get_indexer([tt])[0]
            if j != -1:
                keep[j] = True
    return keep


# -----------------------------------------------------------------------------
# public API
# -----------------------------------------------------------------------------


def ts_coarsen(
    df: pd.DataFrame,
    *,
    grid: str | None = "1min",
    qwidth: float | None = None,
    use_original_vals: bool = True,
    heartbeat_freq: str | None = "120min",
    preserve_vals: Sequence[float] = (),
    preserve_eps: float | None = None,
    preserve_enter_dwell: str = "2min",
    preserve_exit_dwell: str = "30s",
    hyst: float = 1.0,
) -> pd.DataFrame:
    """
    Coarsen a multivariate time series by gridding, preserving semantic states,
    hysteretic quantization, and thinning.

    Parameters
    ----------
    df : pd.DataFrame
        Input time series with DatetimeIndex and numeric columns.
    grid : str or None
        Pandas resample frequency. If None, operate on original timestamps.
        The grid roughly defines the output resolution. It will be refined based
        on preservation and coarasened based on quantization, but loss due to the grid
        is not recoverable.
    qwidth : float or None
        Post-grid quantization width. None disables quantization.
    use_original_vals : bool
        If True, emit raw gridded values at kept timestamps after gridding; otherwise emit
        quantized values used for change detection. Setting this to False will give a more "digitize" feel.
        Recomment True for most use cases, and required with gridding,
        but False can be useful for debugging or if you want to quantize without gridding.
    heartbeat_freq : str or None
        Periodic keep-alive frequency preventing long gaps. None disables heartbeat.
    preserve_vals : sequence of float
        Values (e.g. 0.0, 1.0) defining semantic states to preserve, kind of like 
        a combination of snapping and preservation of entry/exit points. 
        If a preserve value is within preserve_eps of the raw value, it is considered 
        to be in that state for preservation purposes.
    preserve_eps : float or None
        Tolerance around preserve values. Defaults to 0.5*qwidth or 0.0.
    preserve_enter_dwell : str
        Minimum dwell time to enter a preserved state.
    preserve_exit_dwell : str
        Maximum gap to bridge between preserved state segments.
    hyst : float
        Hysteresis factor for quantization (multiplier on qwidth). Default of 0.5
        is intermediate -- quantization only occurs if value changes by more than half qwidth. 
        hyst > 1, as it will give a choppy feel. Low is fine and 0 disbles.

    Returns
    -------
    pd.DataFrame
        Coarsened time series with preserved semantic information.
    """

    if grid is not None and not use_original_vals:
        raise ValueError(
            "use_original_vals=False is not supported when grid is set; "
            "quantization is only used for thinning in grid mode."
        )

    df, t, X = _prepare_df(df)

    enter = np.timedelta64(pd.Timedelta(preserve_enter_dwell).value, "ns")
    exit = np.timedelta64(pd.Timedelta(preserve_exit_dwell).value, "ns")

    do_preserve = len(preserve_vals) > 0
    if preserve_eps is None:
        preserve_eps = 0.5 * qwidth if qwidth is not None else 0.0

    if do_preserve:
        stable_by_v = []
        for v in preserve_vals:
            raw = np.all(np.abs(X - v) <= preserve_eps, axis=1)
            stable_by_v.append(_stable_state_from_raw(t, raw, enter, exit))
        stable_any = np.logical_or.reduce(stable_by_v)
    else:
        stable_any = np.zeros(len(t), bool)
        stable_by_v = []

    if grid is not None:
        dfg = df.resample(grid).ffill().bfill()
        t2 = dfg.index
        Xraw = dfg.to_numpy(float, copy=True)

        if do_preserve:
            masks = []
            for sb in stable_by_v:
                s = pd.Series(sb, index=df.index).astype("int8")
                masks.append(
                    s.resample(grid)
                    .ffill()
                    .fillna(0)
                    .astype("int8")
                    .astype(bool)
                    .to_numpy()
                )
            for v, m in zip(preserve_vals, masks):
                Xraw[m, :] = v

            stable_any_g = np.zeros(len(t2), dtype=bool)
            for v in preserve_vals:
                stable_any_g |= np.all(np.abs(Xraw - v) <= preserve_eps, axis=1)
        else:
            stable_any_g = np.zeros(len(t2), bool)

        # boundary + pre-exit (grid semantics)
        boundary = _preserve_boundary_mask(stable_any_g)

        # Pre-exit semantics must consider raw transitions: the raw exit can occur
        # within the last grid bin, so stable_any_g may not show an exit at all.
        pre_exit_raw = np.zeros(len(t2), dtype=bool)
        if do_preserve:
            for sb in stable_by_v:
                pre_exit_raw |= _grid_force_pre_exit_from_raw(t, sb, t2, grid)


        Xdec = _hysteretic_quantize(Xraw, qwidth, hyst) if qwidth is not None else Xraw

        must_keep = boundary | pre_exit_raw | _heartbeat_mask(t2, heartbeat_freq)
        keep = _change_points_mask(Xdec) | must_keep

        # Always emit working (gridded) values; quantization is only for thinning.
        return _emit_df(t2, dfg.columns, Xraw, keep, force_keep=must_keep)




        # ------------------------------------------------------------------
        # no-grid path
        # ------------------------------------------------------------------
    Xdec = _hysteretic_quantize(X, qwidth, hyst) if qwidth is not None else X

    boundary = _preserve_boundary_mask(stable_any)
    hb = _heartbeat_mask(df.index, heartbeat_freq)

    keep = _change_points_mask(Xdec) | boundary | hb
    keep &= (~stable_any) | boundary

    Xw = X if use_original_vals else Xdec
    force_keep = boundary | hb

    return _emit_df(df.index, df.columns, Xw, keep, force_keep=force_keep)


