#import csv
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import datetime as dtm
import glob
from vtools.functions.merge import *



def is_des(fname):
    with open(fname,"r") as f:
        for i,line in enumerate(f):
            if i > 6: return False
            if "result_id" in line.lower(): 
                return True 
    


def read_des(fpath_pattern,start=None,end=None,selector=None,force_regular=True):

    if selector is not None:
        raise ValueError("selector argument is for API compatability. This is not a multivariate format, selector not allowed")
    ts = csv_retrieve_ts(fpath_pattern, 
                         start, end, force_regular,
                         format_compatible_fn=is_des,
                         selector="VALUE",
                         qaqc_selector="QAQC Flag",
                         qaqc_accept=["U","G","A"],
                         parsedates=["DATETIME"],
                         indexcol="DATETIME",
                         skiprows=4,
                         sep=",",
                         header=0,
                         dateparser=None,
                         comment=None,
                         extra_na=[""],
                         prefer_age="new")
    return ts

################################################33
def cdec2_date_parser(arg):
    return dtm.datetime.strptime(arg, "%Y%m%d %H%M")

def is_cdec_csv2(fname):
    with open(fname,"r") as f:
        title_line = f.readline()
        return title_line.lower().startswith("station_id,duration,sensor_number")


def read_cdec2(fpath_pattern,start=None,end=None,selector=None,force_regular=True):
    if selector is not None:
        raise ValueError("selector argument is for API compatability. This is not a multivariate format, selector not allowed")
    ts = csv_retrieve_ts(fpath_pattern, start, end, force_regular,
                         selector="VALUE",
                         format_compatible_fn=is_cdec_csv2,
                         qaqc_selector="DATA_FLAG",
                         qaqc_accept=['', ' ', ' ', 'e'],
                         parsedates=["OBS DATE"],
                         indexcol="OBS DATE",
                         skiprows=0,
                         sep=",",
                         dateparser=cdec2_date_parser,
                         comment=None,
                         prefer_age="new")
    return ts

##################################### 


def cdec1_date_parser(*args):
    if len(args) == 2:
        x = args[0] + args[1]
        return dtm.datetime.strptime(x, "%Y%m%d%H%M")
    else:
        return dtm.datetime.strptime(args, "%Y%m%d%H%M")


def is_cdec_csv1(fname):
    with open(fname,"r") as f:
        title_line = f.readline()
        return title_line.lower().startswith("title:")
        

def read_cdec1(fpath_pattern,start=None,end=None,selector=None, force_regular=True):
    if selector is not None:
        raise ValueError("selector argument is for API compatability. This is not a multivariate format, selector not allowed")
    
    def cdec_date_parser(*args):
        if len(args) == 2: 
            x = args[0] + args[1]     
            return dtm.datetime.strptime(x,"%Y%m%d%H%M")
        else:
            return dtm.datetime.strptime(args,"%Y%m%d%H%M")
    
    
    ts = csv_retrieve_ts(fpath_pattern, start, end, force_regular,
                         selector="value",
                         format_compatible_fn=is_cdec_csv1,
                         qaqc_selector=None,
                         column_names = ["date","time","value"],
                         parsedates=[["date","time"]],
                         indexcol="date_time",
                         skiprows=2,
                         header=None,
                         sep=",",
                         dateparser=cdec1_date_parser,
                         comment=None,
                         prefer_age="new")
    return ts


def is_wdl(fname):
    with open(fname,"r") as f:
        first_line = f.readline()
        parts = first_line.split(",")
        if not len(parts) == 3: return false
        try:
            pd.to_datetime(parts[0])
            return True
        except: 
            return False


def read_wdl(fpath_pattern,start=None,end=None,selector=None,force_regular=True):
    if selector is not None:
        raise ValueError("selector argument is for API compatability. This is not a multivariate format, selector not allowed")
    
    ts = csv_retrieve_ts(fpath_pattern, start, end, force_regular,
                         selector="value",
                         format_compatible_fn=is_wdl,
                         qaqc_selector="qaqc_flag",
                         qaqc_accept=['', ' ', ' ', 'e',"1"],
                         parsedates=["datetime"],
                         indexcol="datetime",
                         sep=',',
                         skiprows=0,
                         column_names=["datetime","value","qaqc_flag"],
                         header=None,
                         dateparser=None,
                         comment=None)    
    return ts

############################################################
def is_usgs1(fname):
    MAX_SCAN_LINE = 20
    with open(fname,"r") as f:
        for i,line in enumerate(f):
            if i >MAX_SCAN_LINE: return False
            linelower = line.lower()
            if "waterdata.usgs.gov" in linelower: 
                return True 
            if "nwisweb" in linelower:
                return True
            if linelower.startswith("usgs"):
                return True
    return False

def usgs_data_columns1(fname):
    MAX_SCAN_LINE=60
    import re
    description_re = re.compile(r"#\s+(ts|ts_id)\s*(parameter)\s*description")
    colnames=[]
    description={}
    with open(fname,"r") as f:
        reading_cols = False
        for i,line in enumerate(f):
            if i > MAX_SCAN_LINE: return False
            linelower = line.lower()
            if not linelower.startswith("#"): 
                raise ValueError("Column names could not be inferred in file: {}".format(fname))
            if description_re.match(linelower):
                reading_cols = True
                continue
            if reading_cols:
                import string
                try: 
                    comment, ts_id, param, describe = linelower.split(maxsplit=3)
                    col_id = ts_id+"_"+param
                    colnames.append(col_id)
                    description[col_id]=describe.strip()
                except:
                    return colnames


def read_usgs1(fpath_pattern,start=None,end=None,selector=None,force_regular=True):
    TZCOL = "tz_cd"
    if selector is None: 
        selector = usgs_data_columns1
        qaselect = lambda x,y: x+"_cd"

    else:   
        selector = listify(selector)
        qaselect = [x+"_cd" for x in selector]

    dtypes = {TZCOL : str}
        
    # Now tack on time zone at the end
    ts = csv_retrieve_ts(fpath_pattern, start, end, force_regular, 
                         selector=selector,
                         format_compatible_fn = is_usgs1,
                         qaqc_selector=qaselect,
                         qaqc_accept=['', ' ', ' ', 'A','P'],
                         extra_cols="tz_cd",
                         parsedates=["datetime"],
                         indexcol="datetime",
                         header=0,
                         sep="\t",
                         skiprows="count",
                         dateparser=None,
                         comment="#",
                         dtypes=dtypes)    
    #todo: hardwired from PST (though the intent is easily generalized)
    # note there is some bugginess to this. See SO post:
    # https://stackoverflow.com/questions/57714830/convert-from-naive-local-daylight-time-to-naive-local-standard-time-in-pandas
    dst = ts[TZCOL] == "PDT"
    if dst.any():
        ts.index = ts.index.tz_localize('US/Pacific',ambiguous=ts["tz_cd"]=="PDT" ).tz_convert('Etc/GMT+8')
        ts.index = ts.index.tz_localize(None)
    
    # Get rid of the time zone column
    ts = ts.drop(TZCOL,axis=1)
    
    # Get rid of redundant entries caused by the time zone
    ts = ts.loc[~ts.index.duplicated(keep='first')]    
    return ts

################################################################
def is_usgs2(fname):
    MAX_SCAN_LINE = 20
    with open(fname,"r") as f:
        for i,line in enumerate(f):
            if i > MAX_SCAN_LINE: return False
            linelower = line.lower()
            if "# //united states geological survey" in linelower: 
                return True 
            if "nwis-i unit-values" in linelower:
                return True
    return False

def read_usgs2(fpath_pattern,start=None,end=None,selector=None,force_regular=True):
    tzcol = "TZCD"
    if selector is None:
        selector = "VALUE"
    selector = listify(selector)
    qaselect = listify("QA")
    dtypes = dict.fromkeys(selector,float)
    dtypes.update(dict.fromkeys(qaselect,str))
    selector.append(tzcol)
    dtypes[tzcol] = "str"    
        
    # Now tack on time zone at the end
    ts = csv_retrieve_ts(fpath_pattern, start, end, force_regular, 
                         selector=selector,
                         format_compatible_fn = is_usgs2,
                         qaqc_selector=qaselect,
                         qaqc_accept=['', ' ', ' ', 'A','P'],
                         parsedates=[["DATE","TIME"]],
                         indexcol="DATE_TIME",
                         header=0,
                         sep="\t",
                         skiprows="count",
                         comment="#",
                         dtypes=dtypes)    
    #todo: hardwired from PST (though the intent is easily generalized)
    # note there is some bugginess to this. See SO post:
    # https://stackoverflow.com/questions/57714830/convert-from-naive-local-daylight-time-to-naive-local-standard-time-in-pandas
    dst = ts[tzcol] == "PDT"
    if dst.any():
        #todo: hardwire
        ts.index = ts.index.tz_localize('US/Pacific',ambiguous=ts["tz_cd"]=="PDT" ).tz_convert('Etc/GMT+8')
        ts.index = ts.index.tz_localize(None)
    
    # Get rid of the time zone column
    ts = ts.drop(tzcol,axis=1)
    
    # Get rid of redundant entries caused by the time zone
    ts = ts.loc[~ts.index.duplicated(keep='first')]
    
    return ts

######################
def is_usgs_csv1(fname):
    MAX_SCAN = 32
    sampleline = False
    with open(fname,"r") as f:
        for i,line in enumerate(f):
            if i > MAX_SCAN and not sampleline: return False
            linelower = line.lower()
            commented = linelower.startswith("#")
            if not commented:
                if linelower.startswith("iso"):
                    return sampleline
                else:
                    return False
            if "csv data starts at line" in linelower: 
                sampleline = True
    return sampleline

def read_usgs_csv1(fpath_pattern,start=None,end=None,selector=None,force_regular=True):

    if selector is None:
        selector = "Value"
    qaselect = ["Approval Level"]
        
    # Now tack on time zone at the end
    ts = csv_retrieve_ts(fpath_pattern, start, end, force_regular, 
                         selector=selector,
                         format_compatible_fn = is_usgs_csv1,
                         qaqc_selector=qaselect,
                         qaqc_accept=['', ' ', ' ', 'A','P','Approved','Working'],
                         parsedates=[0,1],
                         indexcol=1,
                         header=0,
                         sep=", ",
                         engine="python",
                         skiprows="count",
                         comment="#",
                         dtypes={"Value":float,"Approval Level":str})
    return ts
    

##############################################


def is_noaa_file(fname):
    MAX_SCAN_LINE = 14
    with open(fname,"r") as f:
        for i,line in enumerate(f):
            if i > MAX_SCAN_LINE: return False
            linelower = line.lower()
            if "agency" in linelower and "noaa" in linelower: 
                return True 
            if "x, n, r" in linelower:
                return True
    return False
    
def noaa_data_column(fname):
    MAX_SCAN_LINE=60
    with open(fname,"r") as f:
        reading_cols = False
        for i,line in enumerate(f):
            if i > MAX_SCAN_LINE: 
                raise ValueError("Could not determine data columns within MAX_SCAN lines")
            if not line.startswith("#"):
                parts = [p.strip() for p in line.split(",")]
                return [parts[1]]               

def noaa_qaqc_selector(selector,fname):
    MAX_SCAN_LINE=60
    with open(fname,"r") as f:
        reading_cols = False
        for i,line in enumerate(f):
            if i > MAX_SCAN_LINE: 
                raise ValueError("Could not determine qaqc columns within MAX_SCAN lines")
            if not line.startswith("#"):
                if line.startswith("Date"):
                    parts = [p.strip() for p in line.split(",")]
                    if parts[2]=="X": return "X"
                    if parts[-1]=="Quality": return "Quality"
                else:
                    raise ValueError("Quality labels in file not known: {}".format(fname))


def read_noaa(fpath_pattern,start=None,end=None,selector=None,force_regular=True):
    
    ts = csv_retrieve_ts(fpath_pattern, 
                         start, end, force_regular, 
                         selector=noaa_data_column, 
                         format_compatible_fn=is_noaa_file,
                         qaqc_selector=None,
                         qaqc_accept=['',' ','0'],
                         parsedates=["Date Time"],
                         indexcol="Date Time",
                         header=0,
                         sep=",",
                         comment="#")    
    return ts



def vtide_date_parser(*args):
    x = args[0] + "T" + args[1] if len(args) == 2 else args
    return dtm.datetime.strptime(x, "%Y%m%dT%H%M")


def read_vtide(fpath_pattern,start=None,end=None,selector=None,force_regular=False):
    ts = csv_retrieve_ts(fpath_pattern, start, end, force_regular,
                             selector=selector,
                             format_compatible_fn=lambda x: True,
                             qaqc_selector=None,
                             parsedates=[0],
                             indexcol=0,
                             header=None,
                             sep="\s+",
                             comment="#")    
        
    return ts



def read_ts(fpath, start=None, end=None, force_regular=True, selector = None,hint=None):
    """ Read a time series from a text file in various formats.
        This function asks readers for different file formats to attempt to read the file.
        The first reader that confirms its appropriateness will be attempted. The order of this
        is not guaranteed..

        Parameters
        ----------
        fpath: str
            a file path to read in
        start: datetime.datetime, optional
            time to start reading
        end: datetime.datetime, optional
            time to end reading

        Returns
        -------
        vtools.data.timeseries.TimeSeries
            the time series from the file
        dict
            metadata of the time series
    """
    from os.path import split as op_split
    readers = [read_usgs1,read_usgs2,read_usgs_csv1,
               read_noaa,read_wdl,read_des,
               read_cdec1,read_cdec2]
    ts = None
    reader_count = 0
    for reader in readers:
        if hint is not None:
            if hint not in reader.__name__: continue
        try:
            ts = reader(fpath,start=None,end=None,selector=None,force_regular=force_regular)
            return ts
        except Exception as e:
            continue
    
    if ts is None:
        raise ValueError("File format not supported or error during read: {}\n" .format(fpath))



def write_ts():
    pass

def write_vtide():
    pass






















def listify(inp):
    if isinstance(inp, str): inp = [inp]
    else:
      try: iter(inp)
      except TypeError: inp = [inp]
      else: inp = list(inp)
    return inp


def remove_isolated(ts, thresh):
    goodloc = np.where(np.isnan(ts.data), 0, 1)
    repeatgood = np.apply_along_axis(rep_size, 0, goodloc)
    isogood = (goodloc == 1) & (repeatgood < thresh)
    out = ts.copy()
    out.data[isogood] = np.nan
    return out


def count_comments(fname,comment):
    ncomment = 0
    with open(fname,"r") as f:
        while f.readline().startswith("#"):
            ncomment += 1
    return ncomment


def path_pattern(path_pattern):
    from os.path import split as opsplit
    if isinstance(path_pattern,str): 
        fdir,fpat = opsplit(path_pattern)
    else:
        fdir,fpat = path_pattern
    return fdir,fpat

def csv_retrieve_ts(fpath_pattern,start, end, force_regular=True,selector=None,
                    format_compatible_fn = lambda x: False,
                    qaqc_selector=None,
                    qaqc_accept=["G", "U"],
                    extra_cols = None,
                    parsedates=None,
                    indexcol=0,
                    skiprows=0,
                    header=0,
                    dateparser=None,
                    comment=None,
                    sep="/s+",
                    extra_na=["m", "---", "", " "],
                    blank_qaqc_good=True,
                    prefer_age="new",
                    column_names=None,
                    replace_names=None,
                    dtypes = None,
                    freq='infer',
                    **kwargs):
    import os
    import fnmatch
    fdir,fpat = path_pattern(fpath_pattern)
    matches = []
    for root, dirnames, filenames in os.walk(fdir):
        for filename in fnmatch.filter(filenames, fpat):
            matches.append(os.path.join(root, filename))


    if len(matches)==0:
        raise IOError("No matches found for pattern: {}".format(fpat))

    if not format_compatible_fn(matches[0]):
        raise IOError("Format not compatible for file {} ({})".format(matches[0],
                      format_compatible_fn.__name__))

    #parsetime = lambda x: pd.datetime.strptime(x, '%Y-%m-%d%H%M')
    tsm = []

    if prefer_age != "new":
        raise NotImplementedError("Haven't implemented prefer = 'old' yet")

    if callable(selector):
        selector = selector(matches[0])
    selector = listify(selector)
    
    if callable(qaqc_selector):
        # todo: this is not efficient for noaa because it keep opening files
        qaqc_selector = [qaqc_selector(x,matches[0]) for x in selector]
    elif qaqc_selector is not None:
        qaqc_selector = listify(qaqc_selector)
    if extra_cols is not None:  selector.append(extra_cols)

    # This essentially forces the client code to define dtypes for all
    # complex cases. It correctly handles the situation where all the
    # items in selector are floats, all the items in qaqc_selector are alphanumeric
    if dtypes == None: 
        dtypes = {}

    # By default, all data selections are float, all flags are str
    if qaqc_selector is None:        
        for s in selector:
            if not s in dtypes:
                dtypes[s] = float
    else:
        # This behaves OK if there are extras in selector as with USGS and tz_cd
        for s,qs in zip(selector,qaqc_selector):
            if not s in dtypes:
                dtypes[s] = float
            if not qs in dtypes:
                dtypes[qs] = str

    # The matches are in lexicographical order. Reversing them puts the newer ones
    # higher priority than the older ones for merging
    matches.reverse()
    if len(matches) == 0:
        raise ValueError(
            "No matches to file pattern {} in directory {}".format(fpat, fdir))
    for m in matches:
        dargs = kwargs.copy()
        if dateparser is not None:
            dargs["date_parser"] = dateparser
        if comment is not None:
            dargs["comment"] = comment
        # if not na_values is None: dargs["na_values"] = na_values
        
        if skiprows=="count":
            ncomment = count_comments(m,comment)
            skiprows = list(range(ncomment)) + [ncomment+1]
        
        
        if column_names is None:
            dset = pd.read_csv(m, index_col=indexcol, header=header,
                           skiprows=skiprows,sep=sep,
                           parse_dates=parsedates, na_values=extra_na,
                           keep_default_na=True, dtype=dtypes,
                           infer_datetime_format=True,
                           **dargs)
            
            if header is None:
                # This is essentially a fixup for vtide, which I'm not
                # too happy about ... we should have a header and only one
                # choice of format for this tool
                # Assume the parser could not assign column names
                # and used Int64Index 
                dset.columns = [str(x-1) for x in dset.columns]
                dset.index.name = "datetime"
                if selector is None or selector==[None]: 
                    selector = dset.columns
            dset.columns = [x.strip() for x in dset.columns]

        else:
            dset = pd.read_csv(m, index_col=indexcol, header=header,
                           skiprows=skiprows,sep=sep,
                           parse_dates=parsedates, 
                           na_values=extra_na,
                           keep_default_na=True, dtype=dtypes,
                           names=column_names,
                           infer_datetime_format=True,
                           **dargs)

        if dset.shape[0] == 0:
            # empty file
            continue

        if qaqc_selector is not None:
            # It is costly to try to handle blanks differently for both data  
            # (for which we usually want blanks to be NaN and alphanumeric flags.
            if blank_qaqc_good: qaqc_accept += [np.NaN]
            try:
                dset.loc[~dset[:,qaqc_selector].isin(qaqc_accept), selector] = np.nan
            except:
                for v,f in zip(selector,qaqc_selector):
                    dset.loc[~dset[f].isin(qaqc_accept), v] = np.nan

        tsm.append(dset[selector])
    big_ts = ts_merge(tsm)  # pd.concat(tsm)
    if force_regular: 
        if freq == 'infer': 
            f = pd.infer_freq(big_ts.index)
            if f is None:
                # Give it one more shot halfway through
                istrt = len(big_ts)//2
                f = pd.infer_freq(big_ts.iloc[istrt:istrt+5,:].index)
            if f is None:
                raise ValueError("read_ts set to infer frequency, but two attempts failed. Set to string to manually ")
        else: 
            raise NotImplementedError("force_regular with prescribed frequency not implemented yet")
        # Round to neat times, which may cause duplicates
        big_ts.index = big_ts.index.round(f)
        big_ts = big_ts.loc[~big_ts.index.duplicated(keep='first')]
        # Now everything is on an expected timestamp, so subsample leaving uncovered times NaN
        big_ts = big_ts.asfreq(f,method=None)
    # This try/except Ensures frame rather than Series
    if start is None: start = big_ts.index[0]
    if end is None: end = big_ts.index[-1]
    try:
        return big_ts.to_frame().loc[start:end,:]
    except:
        return big_ts.loc[start:end,:]


