
from scipy.interpolate import PchipInterpolator
import pandas as pd
import numpy as np

__all__ = ["transition_ts"]

def transition_ts(
    ts0, ts1, method="linear", create_gap=None, overlap=(0, 0), return_type="series"
):
    if not isinstance(ts0, (pd.Series, pd.DataFrame)) or not isinstance(ts1, type(ts0)):
        raise ValueError("ts0 and ts1 must be of the same type (Series or DataFrame).")
    if ts0.index.freq != ts1.index.freq:
        raise ValueError("ts0 and ts1 must have the same frequency.")

    freq = ts0.index.freq

    # Determine transition interval
    if create_gap:
        trans_start = pd.Timestamp(create_gap[0])
        trans_end = pd.Timestamp(create_gap[1])

        # Start anchor
        if ts0.index[-1] < trans_start:
            start_time = ts0.index[-1]
            start_val = ts0.iloc[-1]
        else:
            start_time = ts0.loc[:trans_start].index[-1]
            start_val = ts0.loc[:trans_start].iloc[-1]

        # End anchor
        if ts1.index[0] > trans_end:
            end_time = ts1.index[0]
            end_val = ts1.iloc[0]
        else:
            end_time = ts1.loc[trans_end:].index[0]
            end_val = ts1.loc[trans_end:].iloc[0]

    else:
        trans_start = ts0.index[-1] + freq
        trans_end = ts1.index[0] - freq
        start_time = ts0.index[-1]
        start_val = ts0.iloc[-1]
        end_time = ts1.index[0]
        end_val = ts1.iloc[0]

    trans_index = pd.date_range(start=trans_start, end=trans_end, freq=freq)
    if len(trans_index) < 2:
        raise ValueError("Transition zone must have at least two steps.")

    # Interpolation
    elif method == "linear":
        total_duration = (end_time - start_time).total_seconds()
        rel_pos = [
            (t - start_time).total_seconds() / total_duration for t in trans_index
        ]

        if isinstance(ts0, pd.DataFrame):
            interpolated = pd.DataFrame(
                np.outer(1 - rel_pos, start_val) + np.outer(rel_pos, end_val),
                index=trans_index,
                columns=ts0.columns,
            )
        else:
            interpolated = pd.Series(
                [(1 - p) * start_val + p * end_val for p in rel_pos],
                index=trans_index,
                name=ts0.name,
            )

    elif method == "pchip":
        n_before, n_after = overlap

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
        all_data = all_data[~all_data.index.duplicated()].sort_index()

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
        raise ValueError("Only 'linear' and 'pchip' methods are supported.")

    # Final output
    if return_type == "glue":
        return interpolated
    elif return_type == "series":
        ts0_trunc = ts0.loc[: trans_start - freq]
        ts1_trunc = ts1.loc[trans_end + freq :]
        return pd.concat([ts0_trunc, interpolated, ts1_trunc])
    else:
        raise ValueError("return_type must be either 'glue' or 'series'.")