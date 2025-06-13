import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta


def generate_sample_data():
    freq = '15T'
    start = pd.Timestamp("2010-01-01")
    end = pd.Timestamp("2012-01-01")
    index = pd.date_range(start, end, freq=freq)
    n = len(index)
    cols = [f"Series{i+1}" for i in range(12)]
    
    data = np.random.randn(n, 12)
    df = pd.DataFrame(data, index=index, columns=cols)
    
    rng = np.random.default_rng(seed=0)
    for col in df.columns:
        for _ in range(20):
            start_idx = rng.integers(0, n - 1)
            gap_length = rng.integers(1, 201)
            end_idx = min(start_idx + gap_length, n)
            df.loc[df.index[start_idx:end_idx], col] = np.nan
    
    gap_end_candidates = rng.choice(df.columns, size=6, replace=False)
    intervals_per_day = int((24*60) / 15)
    for col in gap_end_candidates:
        gap_length = rng.integers(1, 2 * intervals_per_day + 1)
        df.loc[df.index[-gap_length:], col] = np.nan
        
    df.loc[df.index < (start + pd.DateOffset(years=1)), df.columns[0]] = np.nan

    return df


def plot_missing_data(df, ax, min_gap_duration, overall_start, overall_end):
    overall_start_num = mdates.date2num(overall_start)
    overall_end_num = mdates.date2num(overall_end)
    
    overall_color = 'skyblue'
    gap_color = 'orange'
    boundary_gap_color = 'darkorange'
    bar_height = 0.8

    ax.cla()

    y_ticks = []
    y_labels = []
    
    for i, col in enumerate(df.columns):
        # Full-range background bar
        ax.broken_barh([(overall_start_num, overall_end_num - overall_start_num)],
                       (i - bar_height/2, bar_height),
                       facecolors=overall_color, alpha=0.6)
        
        series = df[col]
        mask = series.isna()
        if mask.any():
            groups = (mask != mask.shift()).cumsum()
            for _, group in mask.groupby(groups):
                if group.iloc[0]:  # missing segment
                    gs = group.index[0]
                    ge = group.index[-1] + pd.Timedelta(minutes=15)
                    actual = ge - gs
                    if actual < min_gap_duration:
                        extra = min_gap_duration - actual
                        gs = max(gs - extra/2, overall_start)
                        ge = min(ge + extra/2, overall_end)
                    color = boundary_gap_color if (gs <= overall_start or ge >= overall_end) else gap_color
                    ax.broken_barh([(mdates.date2num(gs), mdates.date2num(ge) - mdates.date2num(gs))],
                                   (i - bar_height/2, bar_height), facecolors=color)
        
        y_ticks.append(i)
        # Missing percentage label
        perc = series.isna().mean() * 100
        if perc == 0:
            lbl = f"{col} (0%)"
        elif perc < 0.01:
            lbl = f"{col} (<0.01%)"
        else:
            lbl = f"{col} ({perc:.2f}%)"
        y_labels.append(lbl)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Time")
    ax.set_title(f"Missing Data (Min gap = {min_gap_duration})")
    ax.xaxis_date()
    ax.figure.autofmt_xdate()
    # Fix y-limits so zooming on x-axis only
    ax.set_ylim(-0.5, len(df.columns)-0.5)


def interactive_gap_plot(df):
    overall_start = df.index[0]
    overall_end = df.index[-1] + pd.Timedelta(minutes=15)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    prev_xlim = [None, None]

    # Initial draw
    default_gap = timedelta(days=4)
    plot_missing_data(df, ax, default_gap, overall_start, overall_end)

    def on_draw(event):
        # Preserve user zoom x-limits
        x0, x1 = ax.get_xlim()
        if prev_xlim[0] == x0 and prev_xlim[1] == x1:
            return
        prev_xlim[0], prev_xlim[1] = x0, x1

        # Determine new min-gap based on visible span
        d0 = mdates.num2date(x0)
        d1 = mdates.num2date(x1)
        years_view = (d1 - d0).total_seconds() / (365.25 * 24 * 3600)
        if years_view >= 12:
            mg = timedelta(days=12)
        elif years_view >= 8:
            mg = timedelta(days=4)            
        elif years_view >= 4:
            mg = timedelta(hours=18)
        elif years_view >=1:
            mg = timedelta(hours=4)
        else:
            mg = timedelta(hours=1)

        # Redraw bars then restore zoom
        plot_missing_data(df, ax, mg, overall_start, overall_end)
        ax.set_xlim(x0, x1)

    # Use draw_event for reliable detection after toolbar zoom/pan
    fig.canvas.mpl_connect('draw_event', on_draw)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    df = generate_sample_data()
    interactive_plot(df)