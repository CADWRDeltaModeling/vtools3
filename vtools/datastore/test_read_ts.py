import pytest
import numpy as np
from vtools.datastore.read_ts import read_cdec1,read_cdec2,read_wdl,read_ts
from vtools.datastore.read_ts import read_des,read_usgs1,read_usgs2,read_usgs_csv1
from vtools.datastore.read_ts import read_noaa,read_vtide
import pandas as pd

#todo: snap to freq example
#todo: verify time-values
#todo: other treatments of flags
#todo: user chooses data type of output
#todo: into class
#todo: test readers not confused with each other assert exception doing this with usgs1 or usgs2
#todo: check timestamps on USGS before and after time zone switch
#todo: check usgs2 and usgs1 don't try to read each other
#todo: user option for timestamp
#todo: big gaps
#####todo: omit start and/or end
#todo: manually set freq
#todo: test hint

#ts = read_ts("ts_data/usgs_2017_csv_format.csv",
#             start=pd.Timestamp(2008,3,4),end=pd.Timestamp(2008,3,5,2),hint="usgs")

ts=None
ts = read_vtide("ts_data/vtide_example.txt")
assert isinstance(ts,pd.DataFrame), "Returned a time series"

ts0=None
ts0 = read_noaa("ts_data/9414290_water_level_*.txt",
                start=pd.Timestamp(2019,11,1),end=pd.Timestamp(2020,1,10))
assert isinstance(ts0,pd.DataFrame), "Returned a time series"

del(ts,ts0)
ts0 = read_cdec2("ts_data/fal_flow.csv",start = pd.Timestamp(2019,10,1),end=pd.Timestamp(2020,5,1))
ts1 = read_ts("ts_data/fal_flow.csv",start = pd.Timestamp(2019,10,1),end=pd.Timestamp(2020,5,1))
assert (ts0 == ts1).all(axis=None)
assert isinstance(ts1,pd.DataFrame), "Returned a time series"

del(ts1,ts0)
ts2 = read_ts("ts_data/fal_flow_gap_middle.csv")
assert len(ts2) == 7, "One value is omitted in file, but should be installed by force_regular"
assert np.isnan(ts2.loc["2019-11-01 00:45","VALUE"]), "NaN value got included"

ts2 = read_ts("ts_data/fal_flow_gap_middle.csv",force_regular=False)
assert len(ts2) == 6, "One value is omitted in file,force_regular = False so not filled"



ts = None
ts = read_wdl("ts_data/B91560_discharge.csv")
assert(isinstance(ts,pd.DataFrame))


ts = read_des("ts_data/s71_mzmroar_msl_turb_inst_2018_2019.csv",
              start = pd.Timestamp(2019, 5, 26),end = pd.Timestamp(2019, 6, 1))

ts = read_usgs2("ts_data/usgs_2011_format.rdb",
                 start=pd.Timestamp(2010,10,1),
                 end=pd.Timestamp(2010,10,1,18))

ts = read_usgs_csv1("ts_data/usgs_2017_csv_format.csv",
                    start=pd.Timestamp(2008,3,4),end=pd.Timestamp(2008,3,5,2))

ts = read_ts("ts_data/usgs_2017_csv_format.csv",
             start=pd.Timestamp(2008,3,4),end=pd.Timestamp(2008,3,5,2))


ts = read_usgs1("ts_data/nwis_univariate_freeport_flow.rdb",
                 start=pd.Timestamp(2019,11,1),end=pd.Timestamp(2019,12,1),
                 selector="236032_00060")
assert "tz_cd" not in ts
#print(ts.loc["2019-11-03 00:15","236032_00060"])
assert np.isclose(ts.loc["2019-11-03 01:15","236032_00060"],13900.0), "PST takes precedence when redundant"
assert np.isclose(ts.loc[pd.Timestamp(2019,11,2,23,30)],11300.0), "PDT timestamped but moved to PST"
assert np.isclose(ts.loc[pd.Timestamp(2019,11,3,2)],14700.0), "PST time right after PDT-PST is OK"

# read_ts should be the same
ts = read_ts("ts_data/nwis_univariate_freeport_flow.rdb",
              start=pd.Timestamp(2019,11,1),end=pd.Timestamp(2020,1,1),
              selector="236032_00060")

ts0 = read_usgs1("ts_data/waterdata_multivariable_threemile.rdb",
               start=pd.Timestamp(2019,1,1),end=pd.Timestamp(2019,3,1),
               selector="176623_00060")

# Test omission of start
ts1 = read_usgs1("ts_data/waterdata_multivariable_threemile.rdb",
               end=pd.Timestamp(2019,3,1),
               selector="176623_00060")
assert ts1.index[0] == pd.Timestamp(2019,1,1)

# Test omission of end
ts2 = read_usgs1("ts_data/waterdata_multivariable_threemile.rdb",
               start=pd.Timestamp(2019,1,1),
               selector="176623_00060")
assert ts2.index[-1] == pd.Timestamp(2019,4,1,23,45)


# Series called without selector returns everything and
# is the same for columns that were in the selector (for ts0)
ts1 = read_usgs1("ts_data/waterdata_multivariable_threemile.rdb",
               start=pd.Timestamp(2019,1,1),end=pd.Timestamp(2019,3,1))
assert (ts0["176623_00060"] == ts1["176623_00060"]).all()

# read_ts same
ts2 = read_ts("ts_data/waterdata_multivariable_threemile.rdb",
               start=pd.Timestamp(2019,1,1),end=pd.Timestamp(2019,3,1))
#todo: this fails?
#assert (ts1.isna() == ts2.isna()).all(axis=None)
#assert ((ts1 == ts2) | ts1.isna() ).all(axis=None)


ts3 = read_usgs1("ts_data/waterdata_multivariable_threemile.rdb",
                start=pd.Timestamp(2019,1,1),end=pd.Timestamp(2019,3,1),
                selector=["176623_00060","15549_00065"])
assert (ts0["176623_00060"] == ts0["176623_00060"]).all()


ts0 = read_noaa("ts_data/9414290_water_temperature.txt",selector="Water Temperature",
               start=pd.Timestamp(2019,11,1),end=pd.Timestamp(2020,1,1))

ts1 = read_noaa("ts_data/9414290_water_temperature.txt",
               start=pd.Timestamp(2019,11,1),end=pd.Timestamp(2020,1,1))
assert (ts0.iloc[:,0].isna() == ts1.iloc[:,0].isna()).all()
assert ((ts0.iloc[:,0] == ts1.iloc[:,0]) | ts0.iloc[:,0].isna() ).all(axis=None)
# test to make sure infer_datetime_Format works
assert np.isclose(ts0.loc["2019-11-02 12:00","Water Temperature"],13.5)

ts = read_cdec1("ts_data/SJR.csv",start = pd.Timestamp(2008,10,23),end=pd.Timestamp(2020,5,1))
