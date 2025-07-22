def main():
    import pandas as pd
    import numpy as np

    tndx = pd.DatetimeIndex(
        [
            "2019-01-04 00:00",
            "2019-01-04 03:00",
            "2019-01-04 10:17",
            "2019-01-04 19:00",
            "2019-01-05 09:00",
            "2019-01-06 10:00",
            "2019-01-07 16:00",
            "2019-01-09 14:50",
        ]
    )

    data = {
        "op": ["a", "a", "b", "b", "b", np.nan, np.nan, np.nan],
        "setting": [1.0, 2.0, 1.0, np.nan, np.nan, 2.0, 1.0, 1.0],
    }
    df = pd.DataFrame(data=data, index=tndx)
    miss = df.setting.isna()
    g = miss.ne(miss.shift()).cumsum()
    print(g)
    elapsed = (df.index - df.index[0]).total_seconds() / 60.0
    print(elapsed)
    print((np.diff(elapsed) + [0]))
    df["per"] = 0.0
    df.loc[0:-1, "per"] = np.diff(elapsed)
    print(df)
    df2 = df.groupby(g).sum()
    print(df2)
    s = df.index.to_series()
    cols = ["op", "setting"]
    for c in cols:
        miss = df[c].isna()
        g = miss.ne(miss.shift()).cumsum()
        m1 = s.groupby(g).min()
        m2 = m1.shift(-1).fillna(df.index[-1])

        out = m2.sub(m1).dt.total_seconds().div(60).astype(int)
        df[c] = g.map(out)
    print(df)


if __name__ == "__main__":
    main()
