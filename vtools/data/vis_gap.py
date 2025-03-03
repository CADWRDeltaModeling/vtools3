

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta


all = ["plot_missing_data"]

def generate_sample_data():
    """
    Generate sample time series data with:
      - 12 series over a 2-year period at 15-minute intervals.
      - 20 random gaps (length 1 to 200 intervals) per series.
      - In 6 randomly chosen series, a gap at the end (up to 2 days missing).
      - In the first series, a gap covering the first year.
    """
    freq = '15min'
    start = pd.Timestamp("2010-01-01")
    end = pd.Timestamp("2012-01-01")
    index = pd.date_range(start, end, freq=freq)
    n = len(index)
    cols = [f"Series{i+1:02d}" for i in range(12)]
    
    # Create DataFrame with random data
    data = np.random.randn(n, 12)
    df = pd.DataFrame(data, index=index, columns=cols)
    
    # For reproducibility of gaps:
    rng = np.random.default_rng(seed=0)
    
    # Introduce 20 large and 20 small scattered gaps per series (randomly chosen start and gap length)
    for col in df.columns:
        for _ in range(20):
            start_idx = rng.integers(0, n - 1)
            gap_length = rng.integers(1, 201)  # gap length between 1 and 200 intervals
            end_idx = min(start_idx + gap_length, n)
            df.loc[df.index[start_idx:end_idx], col] = np.nan
            start_idx = rng.integers(0, n - 1)
            gap_length = rng.integers(1, 5)  # gap length between 1 and 5 intervals
            end_idx = min(start_idx + gap_length, n)
            df.loc[df.index[start_idx:end_idx], col] = np.nan

            
    # For 6 randomly chosen series, remove a gap at the end (up to 2 days missing)
    gap_end_candidates = rng.choice(df.columns, size=6, replace=False)
    intervals_per_day = int((24*60) / 15)
    for col in gap_end_candidates:
        gap_length = rng.integers(1, 2 * intervals_per_day + 1)
        df.loc[df.index[-gap_length:], col] = np.nan
        
    # For the first series, remove the first year of data
    df.loc[df.index < (start + pd.DateOffset(years=1)), df.columns[0]] = np.nan
    df.iloc[0:(n-400),7] = 0.         
    df.iloc[:,8] = 0. 
    return df[cols]

def plot_missing_data(df, ax, min_gap_duration, overall_start, overall_end):
    """
    Plot missing data onto the provided axis with a given minimum gap duration.
    """
    overall_start_num = mdates.date2num(overall_start)
    overall_end_num = mdates.date2num(overall_end)
    
    # Colors for the bars
    overall_color = 'skyblue'
    gap_color = 'orange'
    boundary_gap_color = 'indianred'
    
    bar_height = 0.8  # thickness for each horizontal bar
    
    # Clear current content on ax
    ax.cla()
    
    # Prepare lists for y-ticks and labels with annotations.
    y_ticks = []
    y_labels = []
    
    # Loop over each series (each column in the DataFrame)
    for i, col in enumerate(df.columns):
        # Draw a light blue bar covering the entire time span for this series.
        ax.broken_barh([(overall_start_num, overall_end_num - overall_start_num)],
                       (i - bar_height/2, bar_height),
                       facecolors=overall_color, alpha=0.6)
        
        # Extract the series and find the missing (NaN) segments.
        series = df[col]
        mask = series.isna()
        if mask.any():
            groups = (mask != mask.shift()).cumsum()
            for group_id, group in mask.groupby(groups):
                if group.iloc[0]:  # missing segment
                    gap_start = group.index[0]
                    gap_end = group.index[-1] + pd.Timedelta(minutes=15)
                    
                    # Expand gap if too short
                    actual_gap = gap_end - gap_start
                    if actual_gap < min_gap_duration:
                        extra = min_gap_duration - actual_gap
                        gap_start_adj = gap_start - extra/2
                        gap_end_adj = gap_end + extra/2
                        gap_start = max(gap_start_adj, overall_start)
                        gap_end = min(gap_end_adj, overall_end)
                    
                    # Use a distinct color if the gap touches either end.
                    if gap_start <= overall_start or gap_end >= overall_end:
                        current_gap_color = boundary_gap_color
                    else:
                        current_gap_color = gap_color
                        
                    gap_start_num = mdates.date2num(gap_start)
                    gap_end_num = mdates.date2num(gap_end)
                    ax.broken_barh([(gap_start_num, gap_end_num - gap_start_num)],
                                   (i - bar_height/2, bar_height),
                                   facecolors=current_gap_color)
        
        y_ticks.append(i)
        # Compute percentage of missing data for annotation.
        perc = series.isna().mean() * 100
        if perc == 0:
            label = f"{col} (0%)"
        elif perc > 0 and perc < 0.01:
            label = f"{col} (<0.01%)"
        else:
            label = f"{col} ({perc:.2f}%)"
        y_labels.append(label)
    
    # Format axes
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.xaxis_date()
    ax.set_xlabel("Time")
    ax.set_title(f"Missing Data Visualization (Min gap = {min_gap_duration})")
    ax.figure.autofmt_xdate()
    ax.figure.canvas.draw_idle()

def interactive_gap_plot(df):
    """
    Create an interactive plot that updates the minimum gap duration based on the current x-axis view.
    The mapping used here is:
       - >=20 years view: min gap = 1 day
       - >=10 years view: min gap = 12 hours
       - Otherwise:      min gap = 1 hour
    """
    # Overall full time range (fixed)
    overall_start = df.index[0]
    overall_end = df.index[-1] + pd.Timedelta(minutes=15)
    
    # Create figure and initial plot with a default min_gap_duration.
    default_min_gap = timedelta(hours=20)  # starting default
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_missing_data(df, ax, default_min_gap, overall_start, overall_end)
    
    def on_xlim_change(event_ax):
        print("xlim changed")
        # Determine current view duration
        xlim = event_ax.get_xlim()
        dt0 = mdates.num2date(xlim[0])
        dt1 = mdates.num2date(xlim[1])
        view_duration = dt1 - dt0
        years_view = view_duration.total_seconds() / (365.25 * 24 * 3600)
        
        # Adjust min_gap_duration based on view span
        if years_view >= 20:
            new_min_gap = timedelta(days=1)
        elif years_view >= 10:
            new_min_gap = timedelta(hours=12)
        else:
            new_min_gap = timedelta(hours=1)
        
        # Redraw the missing data visualization with the new min_gap_duration.
        plot_missing_data(df, ax, new_min_gap, overall_start, overall_end)
    
    # Connect the x-axis limits change event to our callback.
    ax.callbacks.connect('xlim_changed', on_xlim_change)
    
    plt.tight_layout()
    plt.show()

# --- Main ---
if __name__ == '__main__':
    df_sample = generate_sample_data()
    interactive_plot(df_sample)
