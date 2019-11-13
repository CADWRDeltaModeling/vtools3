#import csv
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from vtools.data.vtime import hours

#from error_detect import *
import datetime as dtm
#from read_ts import read_ts
#from write_ts import *
import glob
from vtools.functions.merge import *
#from unit_conversions import *


def remove_isolated(ts, thresh):
    goodloc = np.where(np.isnan(ts.data), 0, 1)
    repeatgood = np.apply_along_axis(rep_size, 0, goodloc)
    isogood = (goodloc == 1) & (repeatgood < thresh)
    out = ts.copy()
    out.data[isogood] = np.nan
    return out


def rep_size(x):
    from itertools import groupby
    isrep = np.zeros_like(x)
    diff = np.ediff1d(x, to_begin=1, to_end=1)
    lrep = (diff[:-1] == 0) | (diff[1:] == 0)
    isrep[lrep] = 1
    reps = []

    xx = x.copy()
    xx[xx == 0.] = -0.123456789001
    for a, b in groupby(isrep*xx):
        lb = list(b)
        if a == 0:  # Where the value is 0, simply append to the list
            l = len(lb)
            reps.extend(lb)

        else:  # Where the value is one, replace 1 with the number of sequential 1's
            l = len(lb)
            reps.extend([l]*l)
            # print reps
    outy = np.array(reps)
    return outy


def cdec_date_parser(*args):
    if len(args) == 2:
        x = args[0] + args[1]
        return dtm.datetime.strptime(x, "%Y%m%d%H%M")
    else:
        return dtm.datetime.strptime(args, "%Y%m%d%H%M")


def csv_retrieve_ts(fpat, fdir, start, end, selector=":",
                    qaqc_selector=None,
                    qaqc_accept=["G", "U"],
                    parsedates=None,
                    indexcol=0,
                    skiprows=0,
                    dateparser=None,
                    comment=None,
                    extra_na=["m", "---", "", " "],
                    prefer_age="new",
                    tz_adj=hours(0)):
    import os
    import fnmatch
    matches = []
    for root, dirnames, filenames in os.walk(fdir):
        for filename in fnmatch.filter(filenames, fpat):
            matches.append(os.path.join(root, filename))

    head = skiprows
    column_names = None

    #parsetime = lambda x: pd.datetime.strptime(x, '%Y-%m-%d%H%M')
    tsm = []

    if prefer_age != "new":
        raise NotImplementedError("Haven't implemented prefer = 'old' yet")

    # The matches are in lexicogrphical order. Reversing them puts the newer ones
    # higher priority than the older ones for merging
    matches.reverse()
    if len(matches) == 0:
        raise ValueError(
            "No matches to file pattern {} in directory {}".format(fpat, fdir))
    for m in matches:
        dargs = {}
        if dateparser is not None:
            dargs["date_parser"] = dateparser
        if comment is not None:
            dargs["comment"] = comment
        # if not na_values is None: dargs["na_values"] = na_values
        dset = pd.read_csv(m, index_col=indexcol, header=head,
                           parse_dates=parsedates, na_values=extra_na,
                           keep_default_na=True, **dargs)
        if column_names is None:
            dset.columns = [x.strip() for x in dset.columns]
        else:
            dset.columns = column_names
        if dset.shape[0] == 0:
            # empty file
            continue
        # todo: may have to make this robust if it is a list
        if type(selector) == str:
            if selector.startswith("Var"):
                selector = dset.columns[int(selector[3:])]
        if qaqc_selector is None:
            rowok = None
        else:
            # print dset.head()
            # print dset.columns
            # print qaqc_selector
            # print dset[qaqc_selector].isin(qaqc_accept).head()
            # print dset.loc[~dset[qaqc_selector].isin(qaqc_accept),selector].count()
            # print dset.count()
            dset.loc[~dset[qaqc_selector].isin(qaqc_accept), selector] = np.nan

#            anyok = None
#            qa_flag = dset[qaqc_selector].as_matrix()
#            print("QAQC SEction: {}".format(qaqc_selector))
#            print qa_flag.shape
#            for okflag in qaqc_accept:
#                isok = np.equal(qa_flag,okflag)        #np.apply_along_axis(np.equal,0,qa_flag,okflag)
#                if anyok is None:
#                    anyok = isok
#                else:
#                    anyok |= isok
#            assert anyok.ndim <= 2
#            rowok =  anyok   #np.all(anyok,axis=anyok.ndim-1).flatten()
#            print("number flagged: {}".format(np.count_nonzero(rowok)))
#            print rowok.shape

        tsm.append(dset[selector].astype(float))
        big_ts = ts_merge(tsm)  # pd.concat(tsm)
    return big_ts.to_frame()  # .to_xarray()


if __name__ == "__main__":
    start = dtm.datetime(2003, 1, 1)
    end = dtm.datetime(2008, 5, 29)
    mdir = "//cnrastore-bdo/Modeling_Data/des_emp/raw"
    fpat = "s33_cygnus_cyg_ec_inst_*.csv"
    # fpat="bll_blacklockriver_bll_ec_inst_2009_2019.csv"
    # fpat="s71_mzmroar_msl_ec_inst_*.csv"
    ts = csv_retrieve_ts(fpat, mdir, start, end, selector="VALUE",
                         qaqc_selector="QAQC Flag",
                         parsedates=["DATETIME"],
                         indexcol=["DATETIME"],
                         skiprows=2,
                         dateparser=None,
                         comment=None,
                         prefer_age="new",
                         tz_adj=hours(0))
    ts2 = ts.asfreq("15min")
    fig, ax = plt.subplots(1)
    (ts-10.).plot(ax=ax)
    ts2.plot(ax=ax)
    plt.legend(["Original", "Regular"])
    plt.show()
    ts2.to_csv("//cnrastore-bdo/BDO_HOME/SCHISM/fielddata/emp_ec_20190802/cyg.csv",
               date_format="%Y%m%d,%H%M", float_format="%.1f", na_rep="m")
