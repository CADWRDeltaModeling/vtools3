

import pytest
import numpy as np
from vtools.functions.merge import ts_merge, ts_splice
import pandas as pd
from vtools.data.vtime import hours, days

import pdb


# all result are dataframe irregualr interval
# guarentee no overlap
def get_test_dataframes_irregular(names, dlen):
    dfs = []
    last_end = 0
    last_interval = []
    s1 = pd.to_datetime('1970-01-01')
    s2 = s1 + pd.Timedelta(days=dlen)
    for i in range(len(names)):
        if i > 0:
            start = last_end
            dr = pd.date_range(start=start, periods=dlen)
        else:
            dr = pd.date_range(start=s1, periods=dlen)
        last_end = dr[-1]
        intervals = np.random.randint(1, 10, size=dlen)
        if (i > 0):
            if (intervals[0] <= last_interval[-1]):
                intervals[0] = last_interval[-1] + 1
        last_interval = intervals
        inter_lst = pd.to_timedelta(intervals, unit='h')
        dr = dr + inter_lst
        if (type(names[i]) is str):
            data = np.random.randint(1, 11, size=(ts_len, 1))
        else:
            data = np.random.randint(1, 11, size=(ts_len, len(names[i])))
        df = pd.DataFrame(index=dr, data=data, columns=[names[i]])
        dfs.append(df)
    return dfs

# all result are dataframe irregualr interval
# guarentee one overlap between two neibor series


def get_test_dataframes_irregular_1_overlap(names, dlen):
    dfs = []
    last_end = 0
    last_interval = []
    s1 = pd.to_datetime('1970-01-01')
    s2 = s1 + pd.Timedelta(days=dlen)
    for i in range(len(names)):
        if i > 0:
            start = last_end
            dr = pd.date_range(start=start, periods=dlen)
        else:
            dr = pd.date_range(start=s1, periods=dlen)
        last_end = dr[-1]
        intervals = np.random.randint(1, 10, size=dlen)
        if (i > 0):
            intervals[0] = last_interval[-1]
        last_interval = intervals
        inter_lst = pd.to_timedelta(intervals, unit='h')
        dr = dr + inter_lst
        if (type(names[i]) is str):
            data = np.random.randint(1, 11, size=(ts_len, 1))
        else:
            data = np.random.randint(1, 11, size=(ts_len, len(names[i])))
        df = pd.DataFrame(index=dr, data=data, columns=[names[i]])
        dfs.append(df)
    return dfs


# all result are dataframe irregualr interval
# guarentee the start of next series is
# within the end of last series
def get_test_dataframes_irregular_1_interweaved(names, dlen):
    dfs = []
    last_end = 0
    last_interval = []
    s1 = pd.to_datetime('1970-01-01')
    for i in range(len(names)):
        if i > 0:
            start = last_end
            dr = pd.date_range(start=start, periods=dlen)
        else:
            dr = pd.date_range(start=s1, periods=dlen)
        last_end = dr[-1]
        intervals = np.random.randint(2, 11, size=dlen)
        if (i > 0):
            intervals[0] = last_interval[-1]-1
        last_interval = intervals
        inter_lst = pd.to_timedelta(intervals, unit='h')
        dr = dr + inter_lst
        if (type(names[i]) is str):
            data = np.random.randint(1, 11, size=(dlen, 1))
        else:
            data = np.random.randint(1, 11, size=(dlen, len(names[i])))
        df = pd.DataFrame(index=dr, data=data, columns=[names[i]])
        dfs.append(df)
    return dfs

# all result are dataframe


def get_test_dataframes(names, dlen, overlap):
    dfs = []
    last_end = 0
    for i in range(len(names)):
        if i > 0:
            dr = pd.date_range(last_end-hours(overlap-1),
                               freq="H", periods=dlen)
        else:
            dr = pd.date_range(pd.Timestamp(2000, 1, 1)+hours(i*dlen), freq="H",
                               periods=dlen)
        last_end = dr[-1]

        if (type(names[i]) is str):
            data = np.random.randint(1, 11, size=(dlen, 1))
        else:
            data = np.random.randint(1, 11, size=(dlen, len(names[i])))
        df = pd.DataFrame(index=dr, data=data, columns=[names[i]])
        dfs.append(df)
    return dfs

# all output are series


def get_test_series(names, dlen, overlap):
    dfs = []
    last_end = 0
    for i in range(len(names)):
        if i > 0:
            dr = pd.date_range(last_end-hours(overlap-1),
                               freq="H", periods=dlen)
        else:
            dr = pd.date_range(pd.Timestamp(2000, 1, 1)+hours(i*dlen), freq="H",
                               periods=dlen)
        last_end = dr[-1]
        if (type(names[i]) is str):
            data = np.random.randint(1, 11, size=(dlen, 1))
        else:
            data = np.random.randint(1, 11, size=(dlen, len(names[i])))
        df = pd.DataFrame(index=dr, data=data, columns=[names[i]])
        dfs.append(df.squeeze(axis=1))
    return dfs

# return mixed series and dataframe


def get_test_series_dataframes(names, dlen, overlap):
    dfs = []
    for i in range(len(names)):
        dr = pd.date_range(pd.Timestamp(2000, 1, 1) +
                           hours(i*dlen), freq="H", periods=dlen)
        if i > 0:
            dr = pd.date_range(pd.Timestamp(2000, 1, 1) +
                               hours(i*dlen-overlap), freq="H", periods=dlen)
        if type(names[i] is str):
            data = np.random.randint(1, 11, size=(dlen, 1))
        else:
            data = np.random.randint(1, 11, size=(dlen, len(names[i])))
        df = pd.DataFrame(index=dr, data=data, columns=[names[i]])
        if i == 2:
            dfs.append(df.squeeze(axis=1))
        else:
            dfs.append(df)
    return dfs


dfs = []
names = ["a", "value", "value"]
ts_len = 9
overlap = 2
dfs = get_test_dataframes(names, ts_len, overlap)
ts_long = ts_merge(dfs, names="value")
# original input data should  not changed
for i in range(len(names)):
    assert dfs[i].columns.to_list()[0] == names[i]

assert ts_long.name == "value"
assert len(ts_long) == (len(names)*ts_len-(len(names)-1)*overlap)
# result should honor input priority when duplicate record
# exist in input series
for i in range(1, len(names)):
    overlap_index = dfs[i-1].index[-overlap:-1]
    n = 1
    overlap_index = overlap_index.union(overlap_index.shift(n)[-n:])
    compare = ts_long.loc[overlap_index] == \
        dfs[i-1][names[i-1]].loc[overlap_index]
    assert compare.all()

ts_long = ts_splice(dfs, names="value", transition="prefer_first")
assert ts_long.name == "value"
assert len(ts_long) == (len(names)*ts_len-(len(names)-1)*overlap)
# result should chose first input when duplicate record
# exist in input series
for i in range(1, len(names)):
    overlap_index = dfs[i-1].index[-overlap:-1]
    n = 1
    overlap_index = overlap_index.union(overlap_index.shift(n)[-n:])
    compare = ts_long.loc[overlap_index] == \
        dfs[i-1][names[i-1]].loc[overlap_index]
    assert compare.all()

with pytest.raises(ValueError):
    ts_long = ts_splice(dfs, names="value", transition="badinput")

ts_long = ts_splice(dfs, names="value", transition="prefer_last")
assert ts_long.name == "value"
assert len(ts_long) == (len(names)*ts_len-(len(names)-1)*overlap)
# result should choose last input when duplicate record
# exist in input series
for i in range(1, len(names)):
    overlap_index = dfs[i-1].index[-overlap:-1]
    n = 1
    overlap_index = overlap_index.union(overlap_index.shift(n)[-n:])
    compare = ts_long.loc[overlap_index] == \
        dfs[i][names[i]].loc[overlap_index]
    assert compare.all()


names = ["a", "value", "value"]
dfs = get_test_series(names, ts_len, overlap)
ts_long = ts_merge(dfs, names="value")
assert ts_long.name == "value"
assert type(ts_long) is pd.Series
# original input data should not changed
for i in range(len(names)):
    assert dfs[i].name == names[i]
# result should honor input priority when duplicate record
# exist in input series
for i in range(1, len(names)):
    overlap_index = dfs[i-1].index[-overlap:-1]
    n = 1
    overlap_index = overlap_index.union(overlap_index.shift(n)[-n:])
    compare = ts_long.loc[overlap_index] == dfs[i-1].loc[overlap_index]
    assert compare.all()

ts_long = ts_splice(dfs, names="value", transition="prefer_first")
assert ts_long.name == "value"
assert type(ts_long) is pd.Series
# result should chose first when duplicate record
# exist in input series
for i in range(1, len(names)):
    overlap_index = dfs[i-1].index[-overlap:-1]
    n = 1
    overlap_index = overlap_index.union(overlap_index.shift(n)[-n:])
    compare = ts_long.loc[overlap_index] == dfs[i-1].loc[overlap_index]
    assert compare.all()

ts_long = ts_splice(dfs, names="value", transition="prefer_last")
assert ts_long.name == "value"
assert type(ts_long) is pd.Series
# result should chose first when duplicate record
# exist in input series
for i in range(1, len(names)):
    overlap_index = dfs[i-1].index[-overlap:-1]
    n = 1
    overlap_index = overlap_index.union(overlap_index.shift(n)[-n:])
    compare = ts_long.loc[overlap_index] == dfs[i].loc[overlap_index]
    assert compare.all()


# mixed series and dataframe not allowed
dfs = get_test_series_dataframes(names, ts_len, overlap)
with pytest.raises(ValueError):
    ts_long = ts_merge(dfs, names="value")


dfs = get_test_series_dataframes(names, ts_len, overlap)
with pytest.raises(ValueError):
    ts_long = ts_splice(dfs, names="value")

############################################################
dfs = []
names = [["a", "b", "c"], ["a", "b", "c", "d"],
         ["a", "b", "c"], ["a", "b", "c", "f"]]
ts_len = 18
overlap = 5
dfs = get_test_dataframes(names, ts_len, overlap)
long_name = ["a", "b", "c"]
ts_long = ts_merge(dfs, names=long_name)
# original input data should  not changed
for i in range(len(names)):
    assert [a[0] for a in dfs[i].columns.to_list()] == names[i]
assert [a[0] for a in ts_long.columns.to_list()] == long_name
assert len(ts_long) == (len(names)*ts_len-(len(names)-1)*overlap)
# result should honor input priority when duplicate record
# exist in input series
for i in range(1, len(names)):
    overlap_index = dfs[i-1].index[-overlap:-1]
    n = 1
    overlap_index = overlap_index.union(overlap_index.shift(n)[-n:])
    compare = ts_long[long_name].loc[overlap_index] == \
        dfs[i-1][long_name].loc[overlap_index]
    assert compare.all().all()

ts_long = ts_splice(dfs, names=["a", "b", "c"],
                    transition='prefer_last', floor_dates=False)
for i in range(len(names)):
    assert [a[0] for a in dfs[i].columns.to_list()] == names[i]
assert len(ts_long) == (len(names)*ts_len-(len(names)-1)*overlap)
assert [a[0] for a in ts_long.columns.to_list()] == long_name
for i in range(1, len(names)):
    overlap_index = dfs[i-1].index[-overlap:-1]
    n = 1
    overlap_index = overlap_index.union(overlap_index.shift(n)[-n:])
    compare = ts_long[long_name].loc[overlap_index] == \
        dfs[i][long_name].loc[overlap_index]
    assert compare.all().all()

##################################################################
# test irregular data
##################################################################
dfs = []
names = ["a", "value", "value"]
ts_len = 9
dfs = get_test_dataframes_irregular(names, ts_len)
ts_long = ts_merge(dfs, names="value")
# original input data should  not changed
for i in range(len(names)):
    assert dfs[i].columns.to_list()[0] == names[i]
assert ts_long.name == "value"
assert len(ts_long) == len(names)*ts_len

ts_long = ts_splice(dfs, names="value")
# original input data should  not changed
for i in range(len(names)):
    assert dfs[i].columns.to_list()[0] == names[i]
assert ts_long.name == "value"
assert len(ts_long) == len(names)*ts_len


dfs = get_test_dataframes_irregular_1_overlap(names, ts_len)
ts_long = ts_merge(dfs, names="value")
# original input data should  not changed
for i in range(len(names)):
    assert dfs[i].columns.to_list()[0] == names[i]
assert ts_long.name == "value"
assert len(ts_long) == len(names)*ts_len-len(names)+1

ts_long = ts_splice(dfs, names="value")
# original input data should  not changed
for i in range(len(names)):
    assert dfs[i].columns.to_list()[0] == names[i]
assert ts_long.name == "value"
assert len(ts_long) == len(names)*ts_len-len(names)+1

dfs = get_test_dataframes_irregular_1_interweaved(names, ts_len)
ts_long = ts_merge(dfs, names="value")
# original input data should  not changed
for i in range(len(names)):
    assert dfs[i].columns.to_list()[0] == names[i]
assert ts_long.name == "value"
assert len(ts_long) == len(names)*ts_len

ts_long = ts_splice(dfs, names="value")
# original input data should  not changed
for i in range(len(names)):
    assert dfs[i].columns.to_list()[0] == names[i]
assert ts_long.name == "value"
assert len(ts_long) == len(names)*ts_len-len(names)+1


ts_long = ts_splice(dfs, names="value", transition='prefer_first')
# original input data should  not changed
for i in range(len(names)):
    assert dfs[i].columns.to_list()[0] == names[i]
assert ts_long.name == "value"
assert len(ts_long) == len(names)*ts_len-len(names)+1

dfs = []
names = [["a", "b", "c"], ["a", "b", "c", "d"],
         ["a", "b", "c"], ["a", "b", "c", "f"]]
ts_len = 18
dfs = get_test_dataframes_irregular_1_interweaved(names, ts_len)
ts_long = ts_splice(dfs, names=["a", "b", "c"], transition='prefer_first')
for i in range(len(names)):
    assert [a[0] for a in dfs[i].columns.to_list()] == names[i]
assert [a[0] for a in ts_long.columns.to_list()] == long_name
assert len(ts_long) == len(names)*ts_len-(len(names)-1)


ts_long = ts_splice(dfs, names=["a", "b", "c"], transition='prefer_last')
for i in range(len(names)):
    assert [a[0] for a in dfs[i].columns.to_list()] == names[i]
assert [a[0] for a in ts_long.columns.to_list()] == long_name
assert len(ts_long) == len(names)*ts_len-(len(names)-1)


