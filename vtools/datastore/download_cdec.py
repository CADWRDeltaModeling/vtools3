#!/usr/bin/env python
""" Download robot for water data library
    The main function in this file is cdec_download. 
    
    For help/usage:
    python cdec_download.py --help
"""
import sys                       # noqa
import argparse

if sys.version_info[0] == 2:
    import urllib2
else:
    import urllib.request
import re
import zipfile
import os
import string
import datetime as dt
import numpy as np
import pandas as pd
from vtools.datastore.process_station_variable import process_station_list,stationfile_or_stations
from vtools.datastore import station_config
      
cdec_base_url = "cdec.water.ca.gov"        
       
        
def create_arg_parser():
    parser = argparse.ArgumentParser()
    paramhelp = 'Variable to download'
    
    parser.add_argument('--dest', dest = "dest_dir", default="cdec_download", help = 'Destination directory for downloaded files.')
    parser.add_argument('--id_col', default = "id", type = str, help = 'Column in station file representing CDEC ID. IDs with > 3 characters will be ignored.')
    parser.add_argument('--param_col', type = str, default=None, help = 'Column in station file representing the parameter to download.')
    parser.add_argument('--start',required=True,help = 'Start time, format 2009-03-31 14:00')    
    parser.add_argument('--end',default = None,help = 'Start time, format 2009-03-31 14:00')    

    parser.add_argument('--param',help=paramhelp)
    parser.add_argument('--stations', default=None, nargs="*", required=False,
                        help='Id or name of one or more stations.')
    parser.add_argument('stationfile',nargs="*", help = 'CSV-format station file.')
    parser.add_argument('--overwrite', action='store_true',default=False,help = 'Overwrite existing files (if False they will be skipped, presumably for speed')
    parser.add_argument('--freq', default=None, help = 'specify the frequency. Otherwise proceeds from even to hour to day. must be H or D')
    return parser


def cdec_download(stations,dest_dir,start,end=None,param=None,overwrite=False,freq=None):
    """ Download robot for CDEC
    Requires a list of stations, destination directory and start/end date
    These dates are passed on to CDEC ... actual return dates can be
    slightly different
    """
      
    if end is None: 
        end = dt.datetime.now()
        endfile = 9999
    else: 
        endfile = end.year

    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)       
    failures = []
    skips = []

    # This is a small hardwired section to cull ec values 
    # from the wrong sublocation/program
    # CDEC uses a different variable code for each
    subloc_inconsist = (stations.subloc.isin(["default","nan","upper","top"]))\
               & (stations.src_var_id.isin([92,102]))
    stations = stations.loc[~subloc_inconsist,:]    
    subloc_inconsist = stations.subloc.isin(["lower","bot","bottom"]) & ~stations.src_var_id.isin([92,102])   # at present, only EC has bottom sensor listed
    stations = stations.loc[~subloc_inconsist,:] 
    
    for index,row in stations.iterrows():
        station = row.station_id
        try: 
            cdec_id = row.cdec_id.lower()
        except:
            cdec_id = station
            
        agency_id = row.agency_id
        p = row.param
        z = row.src_var_id
        subloc = row.subloc
        yearname = f"{start.year}_{endfile}"  # if start.year != end.year else f"{start.year}"

        if (subloc == "default"):
            path = os.path.join(dest_dir,f"cdec_{station}_{agency_id}_{p}_{yearname}.csv").lower()
        else:
            path = os.path.join(dest_dir,f"cdec_{station}@{subloc}_{agency_id}_{p}_{yearname}.csv").lower()
                
        if os.path.exists(path) and overwrite is False:
            print("Skipping existing station because file exists: %s" % path)
            skips.append(path)
            continue
        stime=start.strftime("%m-%d-%Y")
        etime=end if end == "Now" else end.strftime("%m-%d-%Y")
        found = False
        
        zz = [z]
        for code in zz:
            dur_codes = ["E","H","D","M"] if freq is None else [freq]
            for dur in dur_codes:
                station_query = f"http://{cdec_base_url}/dynamicapp/req/CSVDataServletPST?Stations={cdec_id}&SensorNums={code}&dur_code={dur}&Start={stime}&End={etime}"
                if sys.version_info[0] == 2:
                    response = urllib2.urlopen(station_query)
                else:
                    response = urllib.request.urlopen(station_query)
                station_html = response.read().decode().replace("\r","")
                if (station_html.startswith("Title") and len(station_html) > 16) or\
                   (station_html.startswith("STATION_ID") and len(station_html)>90):
                    found = True
                    with open(path,"w") as f:
                        f.write(station_html)
                    print("Found, duration code: %s" % dur)
                    break
            if found: break
        if not found: 
            failures.append((station,p))
    
    if len(failures) == 0:
        print("No failed stations")
    else:
        print("Failed query stations: ")
        for failure in failures:
            print(failure)


def process_station_list2(file,cdec_ndx,param_ndx=None):
    stations = []
    variables = [] if param_ndx else None
    for line in open(file,'r'):
        if not line or line.startswith("#") or len(line) < (param_ndx+1): continue
        elements = line.strip().split(",")
        cdec_id = elements[cdec_ndx]
        param = elements[param_ndx] if param_ndx else None
        if len(cdec_id.strip()) == 3:
            stations.append(cdec_id)
            if param_ndx: variables.append(param)            
    return stations,variables
     

def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    cdec_column = args.id_col
    param_column = args.param_col
    destdir = args.dest_dir
    stationfile = args.stationfile
    overwrite = args.overwrite
    param = args.param
    start = args.start
    end = args.end
    freq = args.freq
    stime = dt.datetime(*list(map(int, re.split(r'[^\d]', start))))
    if end is not None:
        etime = dt.datetime(*list(map(int, re.split(r'[^\d]', end))))
    else:
        etime = pd.Timestamp.now()
    if param_column is not None and param is not None:
        raise ValueError("param_col and param cannot both be specified")
    if param_column is None and param is None:
        param_column = 'param'
 
    stationfile=stationfile_or_stations(args.stationfile,args.stations)
    slookup = station_config.config_file("station_dbase")
    vlookup = station_config.config_file("variable_mappings")            
    df = process_station_list(stationfile,param=param,station_lookup=slookup,
                                  agency_id_col="agency_id",param_lookup=vlookup,source='cdec') 
    
    #stations,variables = process_station_list(stationfile,cdec_column,param_column)
    #if not variables: variables = [param]*len(stations)
    cdec_download(df,destdir,
                  stime,
                  etime,
                  param,
                  overwrite,
                  freq)

if __name__ == '__main__':
    main()
