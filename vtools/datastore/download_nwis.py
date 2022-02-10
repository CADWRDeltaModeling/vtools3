#!/usr/bin/env python
""" Download robot for Nationla Water Informaton System (NWIS)
    The main function in this file is nwis_download. 
    
    For help/usage:
    python nwis_download.py --help
"""
import argparse
import sys
import pandas as pd
if sys.version_info[0] == 2:
    import urllib2
elif sys.version_info[0] == 3:
    import urllib.request
else:
    raise RuntimeError("Not recognizable Python version")

import re
import zipfile
import os
import string
import datetime as dt
import numpy as np
from vtools.datastore.process_station_variable import process_station_list,stationfile_or_stations
from vtools.datastore import station_config

def create_arg_parser():
    parser = argparse.ArgumentParser()
   
    parser.add_argument('--dest', dest = "dest_dir", default="nwis_download", help = 'Destination directory for downloaded files.')
    parser.add_argument('--start',required=True,help = 'Start time, format 2009-03-31 14:00')    
    parser.add_argument('--end',default = None,help = 'End time, format 2009-03-31 14:00')
    parser.add_argument('--param',default = None, help = 'Parameter(s) to be downloaded, e.g. \
    00065 = gage height (ft.), 00060 = streamflow (cu ft/sec) and 00010 = water temperature in degrees Celsius. \
    See "http://help.waterdata.usgs.gov/codes-and-parameters/parameters" for complete listing. \
    (if not specified, all the available parameters will be downloaded)')
    parser.add_argument('--stations', default=None, nargs="*", required=False,
                        help='Id or name of one or more stations.')
    parser.add_argument('stationfile',nargs="*", help = 'CSV-format station file.') 
    parser.add_argument('--overwrite', action="store_true", default = False, help =  
    'Overwrite existing files (if False they will be skipped, presumably for speed)')
    return parser


def nwis_download(stations,dest_dir,start,end=None,param=None,overwrite=False):
    """ Download robot for NWIS
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
    for ndx,row in stations.iterrows():
        agency_id = row.agency_id
        station = row.station_id
        param = row.src_var_id
        paramname = row.param
        subloc = row.subloc

        yearname = f"{start.year}_{endfile}" #if start.year != end.year else f"{start.year}"
        outfname = f"usgs_{station}_{agency_id}_{paramname}_{yearname}.rdb"
        outfname = outfname.lower()
        path = os.path.join(dest_dir,outfname)
        if os.path.exists(path) and not overwrite:
            print("\nSkipping existing station because file exists: %s" % station)
            skips.append(path)
            continue
        else:
            print(f"\nAttempting to download station: {station} variable {param}")        
        stime=start.strftime("%Y-%m-%d")
        etime=end.strftime("%Y-%m-%d")
        found = False
        station_query_base = f"http://nwis.waterservices.usgs.gov/nwis/iv/?sites={agency_id}&startDT={stime}&endDT={etime}&format=rdb"
        if param:
            station_query = station_query_base + f'&variable={int(param):05}'
            #station_query = station_query_base % (station,stime,etime,param)
        else:
            station_query = station_query_base
        print(station_query)
        try: 
            if sys.version_info[0] == 2:
                response = urllib2.urlopen(station_query)
            elif sys.version_info[0] == 3:
                response = urllib.request.urlopen(station_query)
        except:
            failures.append(station)
        else:
            try:
                station_html = response.read().decode().replace("\r","")
            except:
                station_html = "" # Catches incomplete read error
            if len(station_html) > 30 and not "No sites found matching" in station_html:
                found = True
                with open(path,"w") as f:
                    f.write(station_html)
            if not found: 
                print("Station %s query failed or produced no data" % station)
                failures.append(station)
    
    if len(failures) == 0:
        print("No failed stations")
    else:
        print("Failed query stations: ")
        for failure in failures:
            print(failure)

def parse_start_year(txt):
    date_re = re.compile(r"(19|20)\d\d[- /.](0[1-9]|1[012])[- /.](0[1-9]|[12][0-9]|3[01])")
    if os.path.exists(txt):
        # assume file
        for iline,line in enumerate(open(txt,'r')):
            if iline > 1000: return None
            if line.startswith("#"): continue
            m = date_re.search(line)
            if m is not None:
                return int(m.group(0)[0:4])
        return None
    else:
        for iline,line in enumerate(iter(txt.splitlines())):
            if iline > 1000: return None
            if line.startswith("#"): continue           
            m = date_re.search(line)
            if m is not None:
                return int(m.group(0)[0:4])
    return None


def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    destdir = args.dest_dir
    stationfile = args.stationfile
    overwrite = args.overwrite
    start = args.start
    end = args.end
    param = args.param
    stime = dt.datetime(*list(map(int, re.split(r'[^\d]', start))))
    if end:
        etime = dt.datetime(*list(map(int, re.split(r'[^\d]', end))))
    else:
        etime = dt.datetime.now()
    
    stationfile=stationfile_or_stations(args.stationfile,args.stations)
    slookup = station_config.config_file("station_dbase")
    vlookup = station_config.config_file("variable_mappings")            
    df = process_station_list(stationfile,param=param,station_lookup=slookup,
                                  agency_id_col="agency_id",param_lookup=vlookup,source='usgs') 
    nwis_download(df,destdir,stime,etime,param,overwrite)  
        

if __name__ == '__main__':
    main()
