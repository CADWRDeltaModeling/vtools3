
def inspect_duplicate_index(ts, label=None, max_times=5, max_rows=5):
    """
    Print a concise diagnostic for duplicate timestamps in a time series.

    Parameters
    ----------
    ts : pandas.DataFrame or Series
        Time series with DatetimeIndex.
    label : str, optional
        Context label (e.g., filename or pattern).
    max_times : int
        Number of duplicate timestamps to report.
    max_rows : int
        Number of rows per timestamp to print.
    """
    import pandas as pd

    idx = ts.index
    if not isinstance(idx, pd.DatetimeIndex):
        print(f"[dup-debug] Index is not DatetimeIndex ({type(idx)})")
        return

    dup_mask = idx.duplicated(keep=False)
    if not dup_mask.any():
        print("[dup-debug] No duplicate timestamps detected.")
        return

    dup_times = idx[dup_mask].unique()

    print("\n" + "="*60)
    print("[dup-debug] Duplicate timestamp diagnostic")
    if label:
        print(f"[dup-debug] Context: {label}")
    print(f"[dup-debug] Unique duplicate timestamps: {len(dup_times)}")
    print(f"[dup-debug] Total duplicate rows: {dup_mask.sum()}")

    # show first few duplicate timestamps
    for t in dup_times[:max_times]:
        print(f"\n[dup-debug] Timestamp: {t}")
        rows = ts.loc[t]
        if isinstance(rows, pd.Series):
            # single row (unlikely here, but safe)
            print(rows)
        else:
            print(rows.head(max_rows))

    print("="*60 + "\n")
 