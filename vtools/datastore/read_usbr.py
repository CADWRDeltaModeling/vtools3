

import pandas as pd
import matplotlib.pyplot as plt


tndx = pd.DatetimeIndex(
    ["2016-11-06 01:00", "2016-11-06 02:00", "2016-11-06 03:00"])
tndx.tz_localize('US/Pacific', ambiguous="infer").tz_convert('Etc/GMT+8')
print(tndx[df.loc[df.index.dropna()]])

fname = "WQ_Hourly_2016-05-01 to 2019-06-30.txt"
fname = "test.txt"
ts = pd.read_csv(fname, sep="\s+", header=None, index_col=[2, 1, 0], parse_dates=[
                 ["date", "time"]], skiprows=[0, 1], names=["site", "shef", "date", "time", "value"])

ts2 = ts.query("shef == 'WC' and site=='RPN'")
# ts2 = ts.query("site == 'RPN'")
ts2 = ts2.set_index(ts2.index.droplevel("site"))
# ts2 = ts2.set_index(ts2.index.droplevel("shef"))

# ts2 = ts2.xs("WC",level="shef",drop_level=True)


# ts2 = ts2.set_index(ts2.index.droplevel("shef").droplevel("site"))

# ts2 = ts2.pivot(index = "date_time", columns = "shef", values = "values")

print(ts2.head())
ts2.columns = [""]
ts2 = ts2.unstack("shef")

print(ts2.head())
print("cols")
print(ts2.columns)
print(ts2.loc[:, (slice(None), "WC")])

ts2 = ts2.droplevel(1, axis=1)
print(ts2)

print("bleh")
print(ts2.loc["20161104":"20161105", :])

ts2.to_csv("out.csv")


ts3 = ts2.copy()
# .tz_localize(None,ambiguous="infer")
convert_index = ts3.index.tz_localize(
    'US/Pacific', ambiguous="infer", nonexistent="shift_backward").tz_convert('Etc/GMT+8')
ts3 = ts3.set_index(convert_index)
ts3 = ts3.loc[ts3.index.dropna()]

print("blah")
print(ts3.loc["20161104":"20161105", :])


# df.loc[:, (slice(None), 'A')]


# print ts.loc[pd.IndexSlice["WC","SAL",:],:]
# ts.plot()
# plt.show()
# .loc[(slice(None), 'one'), :]
