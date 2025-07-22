#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from vtools.functions.filter import *
from scipy.signal import freqz
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import numpy as np
import pandas as pd

# "compare_response",
__all__ = ["compare_response", "unit_impulse_ts"]

plt.style.use(["seaborn-v0_8-deep", "seaborn-v0_8-talk"])


def unit_impulse_ts(size, interval):
    """

    Parameters
    ----------

    size : int
        Length of impluse time series, odd number.

    interval : string
        time series interval, such as "15min"

    Returns
    -------
    pandas.Dataframe
        time series value 0.0 except 1 at the middle of time series.

    """
    idx = pd.date_range("2001-1-1", periods=size, freq=interval)
    val = np.zeros(size)
    mid_idx = int(size / 2)
    val[mid_idx] = 1.0
    # return pd.Series(val,index=idx) # series doesn't works with godin
    return pd.DataFrame(index=idx, data=val)


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


def compare_response(cutoff_period):
    """
    Generate frequency response plot of low-pass filters: cosine_lanczos, boxcar 24h, boxcar 25h, and godin.

    Parameters
    ----------
    cutoff_period : int
        Low-pass filter cutoff period in number of hours.

    Returns
    -------
    None.
    """

    unit_impulse = unit_impulse_ts(5001, "15min")
    fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
    ax.set_ylim(-0.2, 1.2)
    worN = 4096
    dt = unit_impulse.index[1] - unit_impulse.index[0]
    cf = 2.0 * dt.seconds / (cutoff_period * 3600)
    dt_hours_ratio = dt.seconds / 3600.0
    cl_size_hours1 = 70
    cl_size1 = int(cl_size_hours1 * 3600.0 / dt.seconds)
    default_cl_size = int(1.25 * 2.0 / cf)
    w = default_cl_size
    ts = godin(unit_impulse)

    for unit_impulse_response, label in [
        (
            cosine_lanczos(unit_impulse, cutoff_period="40h", filter_len=cl_size1),
            f"cos-Lanczos (width={cl_size1}h)",
        ),
        (
            cosine_lanczos(unit_impulse, cutoff_period="40h"),
            f"cos-Lanczos (width={w}h)",
        ),
        (lanczos(unit_impulse, cutoff_period="40h"), f"Lanczos (width={w}h)"),
        (unit_impulse.rolling(96, center=True, min_periods=96).mean(), "Boxcar 24h"),
        (unit_impulse.rolling(99, center=True, min_periods=99).mean(), "Boxcar 25h"),
        (godin(unit_impulse), "Godin 25-24-24 (width=72h)"),
    ]:
        b = unit_impulse_response.fillna(0.0)
        if (b.iloc[0, 0] != 0.0) or (b.iloc[-1, 0] != 0.0):
            print(f"Warning: Unit impulse length for {label} is not long enough")
        w, h = freqz(b.values, worN=worN)
        pw = w[1:]
        period = 1.0 / pw
        period = 2.0 * np.pi * period * dt_hours_ratio
        hh = np.abs(h)
        ax.plot(period, hh[1:], linewidth=1, label=label)

    # Main plot settings
    ax.set_xlim(0.1, 360)
    ax.axvline(x=cutoff_period, ymin=-0.2, linewidth=1, color="0.25", linestyle=":")

    # Vertical annotation: "Cutoff 40h" to the left of the line
    ax.text(
        cutoff_period - 3,  # Shift left
        0.7,  # Center vertically along the line
        "Cutoff 40h",
        fontsize=10,
        rotation=90,  # Vertical text
        ha="right",  # Left-align to be on the left side of the line
        va="center",
        color="0.25",
        bbox=dict(
            facecolor="white", edgecolor="none", alpha=0.6
        ),  # Background for readability
    )

    ax.set_ylabel("Magnitude")
    ax.set_xlabel("Period (hours)")
    ax.grid(visible=True, which="both", color="0.9", linestyle="-", linewidth=0.5)

    # Move legend **slightly left** to avoid overlapping the cutoff period text
    ax.legend(
        loc="upper center",
        fontsize=10,
        framealpha=1,
        edgecolor="gray",
        ncol=2,  # Two-column layout
        bbox_to_anchor=(0.5, 1.15),  # Shift slightly left
    )

    # Create an inset axis at **compromise position**, but 0.08 lower on the y-axis
    ax_inset = inset_axes(
        ax,
        width="45%",
        height="35%",
        loc="lower left",
        bbox_to_anchor=(0.38, 0.22, 1.2, 1.2),
        bbox_transform=ax.transAxes,
    )
    ax_inset.set_xlim(0, 40)
    ax_inset.set_ylim(-0.05, 0.23)

    # Plot the same response curves in the inset
    for unit_impulse_response, label in [
        (
            cosine_lanczos(unit_impulse, cutoff_period="40h", filter_len=cl_size1),
            f"cos-Lanczos, width={cl_size1}h",
        ),
        (
            cosine_lanczos(unit_impulse, cutoff_period="40h"),
            f"cos-Lanczos (width={w}h)",
        ),
        (lanczos(unit_impulse, cutoff_period="40h"), f"Lanczos (width={w}h)"),
        (unit_impulse.rolling(96, center=True, min_periods=96).mean(), "Boxcar 24h"),
        (unit_impulse.rolling(99, center=True, min_periods=99).mean(), "Boxcar 25h"),
        (godin(unit_impulse), "Godin 25-24-24"),
    ]:
        b = unit_impulse_response.fillna(0.0)
        w, h = freqz(b.values, worN=worN)
        pw = w[1:]
        period = 1.0 / pw
        period = 2.0 * np.pi * period * dt_hours_ratio
        hh = np.abs(h)
        ax_inset.plot(period, hh[1:], linewidth=1, label=label)

    # Tidal constituent markers in the inset (adjusted for spacing)
    labels = ["$M_2$", "$K_1$", "$O_1$"]
    pers = [12.42, 23.93, 25.82]
    shifts = [0.0, -1.0, 1.0]  # K1 shifted left, O1 shifted right

    for mx, label, shift in zip(pers, labels, shifts):
        ax_inset.scatter(
            mx, -0.012, marker="^", color="purple", s=45, zorder=3
        )  # Raised a bit more
        ax_inset.text(
            mx + shift, -0.025, label, ha="center", va="top", fontsize=8, color="purple"
        )  # Smaller font

    # Hide inset tick labels for clarity
    ax_inset.tick_params(axis="both", which="both", labelsize=8)

    # Connect inset to main plot **with a proper box**
    mark_inset(ax, ax_inset, loc1=2, loc2=3, fc="none", ec="0.5", lw=1.2)


def main():
    # compare response for data with 15min interval

    compare_response(40)
    plt.savefig("frequency_response.png", bbox_inches=0)


if __name__ == "__main__":
    main()
