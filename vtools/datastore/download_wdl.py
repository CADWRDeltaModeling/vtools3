#!/usr/bin/env python
""" Download robot for water data library
    The main function in this file is wdl_download. Currently takes a list of stations, downloads data.
 
"""
import sys                       # noqa

if sys.version_info[0] == 2:
    import urllib2
else:
    import urllib.request as urllib2
import re
import zipfile
import os
import string
import datetime as dtm
import numpy as np
import argparse
import pandas as pd

items2 = {"FLOW":"discharge",
         "STAGE_15-MINUTE":"gageheight",
         "CONDUCTIVITY":"conductance",
         "VELOCITY":"velocity",
         "ADCP_WATER_TEMPERATURE":"temperature",
         "WATER_TEMPERATURE":"temperature",
         "TURBIDITY":"turbidity"}

items = {"flow": "FLOW_15",
         "stage": "STAGE_15",
         "conductivity": "CONDUCTIVITY_POINT",
         "velocity" : "VELOCITY",
         "water_temperature": "WATER_TEMPERATURE",
         "adcp_water_temperature": "ADCP_WATER_TEMPERATURE",
         "turbidity": "TURBIDITY",
         "ph": "PH_POINT",
         "do": "OXYGEN_DISSOLVED_POINT",
         "cholorophyll": "TOTAL_CHLOROPHYLL_POINT"
         }


def create_arg_parser():
    parser = argparse.ArgumentParser()
    paramhelp = 'Variable to download, should be in list:\n%s' % list(items.keys())
    
    parser.add_argument('--dest', dest = "dest_dir", default="wdl_download", help = 'Destination directory for downloaded files.')
    parser.add_argument('--id_col',default = None, help = 'Column in station file representing station. IDs not matched will trigger failure. If id_col or param_col is given, a header is assumed')
    parser.add_argument('--param_col',default = None, help = 'Column representing the parameter to download.')
    #parser.add_argument('--start',required=True,help = 'Start time, format 2009-03-31 14:00')    
    #parser.add_argument('--end',default = None,help = 'Start time, format 2009-03-31 14:00')    
    parser.add_argument('--syear',required=True,help = 'Start year')    
    parser.add_argument('--eyear',default = None,help = 'End year, if blank current year')
    parser.add_argument('--param',help=paramhelp)
    parser.add_argument('--overwrite',  action="store_true", help = 'Overwrite existing files (if False they will be skipped)')
    parser.add_argument('stationfile', help = 'CSV-format station file.')    
    return parser


def process_station_list(file,id_col=None,param_col=None,param=None):
    import pandas as pd
    station_list = pd.read_csv(file,sep=',',header=0,dtype=str)
    if param_col is not None:
        station_list=station_list[[id_col,param_col]]
    else:
        station_list = station_list[[id_col]]
        station_list.loc[:,'parameter'] = param
    station_list.columns = ['station_id','parameter']
    return station_list

def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    id_column = args.id_col
    #param_column = args.param_col
    param_column = 'param'
    destdir = args.dest_dir
    stationfile = args.stationfile
    overwrite = args.overwrite
    param = args.param
    syear = int(args.syear)
    eyear = args.eyear
    if eyear is None: 
        today = dtm.datetime.today()
        eyear = today.year
    eyear = int(eyear)
    # start = args.start
    # end = args.end
    # stime = dtm.datetime(*list(map(int, re.split(r'[^\d]', start))))
    # if end:
        # etime = dtm.datetime(*list(map(int, re.split(r'[^\d]', end))))
    # else:
        # etime = "Now"
    if param_column and param:
        raise ValueError("param_col and param cannot both be specified")
    if not (param_column or param):
        raise ValueError("Either param_col or param must be specified")
    if os.path.exists(stationfile):
        station_list = process_station_list(stationfile,id_column,param_column,param)
        wdl_download(station_list,destdir,
                      destdir,syear,
                      eyear,
                      overwrite)
    else:
        print("Station list does not exist")



def main1():
    import sys
    stationlist = sys.argv[1]
    firstyr = int(sys.argv[2])
    stopyr = int(sys.argv[3]) + 1
    f = open(stationlist,"r")
    stations = [x.strip().split(",")[2] for x in f.readlines() if x and len(x) > 2]
    years = range(firstyr,stopyr)
    destdir = 'wdl_data'
    overwrite = False    
    wdl_download(stations,years,destdir,overwrite)

aliases = None
alias_path = 'C:/Delta/vtools3/vtools/datastore/wdl_continuous_files_alias.csv'

def wdl_download(station_list,years,dest_dir,syear,eyear,overwrite=False):
    """ Download robot for Water Data Library
    First loads the stations page to see which variables are available 
    from a hardwired list (conductance, gage elevation, flow, etc). 
    Then it downloads all the years from years that are available for 
    those variables and adds the data to txt files one per station-variable. 
    """
    global aliases
    # prior url
    #base_url = "http://wdl.water.ca.gov/waterdatalibrary/docs/Hydstra/docs"
    base_url = "https://wdlstorageaccount.blob.core.windows.net/continuousdata/docs"
    work_dir = "tempfiles"
    retain_zip = False


    if aliases is None:
        aliases = pd.read_csv(alias_path)
    years = np.arange(int(syear),int(eyear)+1)
    

    if not os.path.exists(work_dir):
        os.mkdir(work_dir)

    for ndx,item in station_list.iterrows():
        station=item.station_id        
        param = item.parameter    

        alias_sub = aliases.loc[(aliases.STATION == item.station_id) & 
                                 aliases.FILE_NAME_BASE.str.startswith(items[param]) ,:]


        station_param = "%s_%s" % (station,param)
        newname = "%s.csv" % station_param
        newname = newname.lower()
        dest_path = os.path.join(dest_dir,newname)
        if (os.path.exists(dest_path)):
            if overwrite:
                os.remove(dest_path)
            else:
                print("Skipping station/param/year because file exists: %s" % newname) 
                continue
        file_created = False
        ndata = 0
        data_years = []
        alias_sub = alias_sub.loc[alias_sub.POR_YEAR_FOLDER != 'POR',:]
        alias_sub = alias_sub.astype({'POR_YEAR_FOLDER':int})     

        for year in years:
            yrdf = alias_sub.loc[alias_sub.POR_YEAR_FOLDER == year,:]
            if len(yrdf) == 0: continue
            filename = yrdf.iloc[0]['TABLE_FILE_NAME']
            station_query="{}/{}/{}/{}".format(base_url,station,year,filename)
            response = urllib2.urlopen(station_query)
            fname = station+"_"+str(year)+"_"+filename
            fname = fname.lower()
            localname = os.path.join(work_dir,fname)
            with open(localname, "wb") as local_file:
                local_file.write(response.read())
            f = open(localname,"r")
            mode = "a" if file_created else "w"
            destfile = open(dest_path,mode)
            if not file_created:
                destfile.write("\"Date\",\"Point\",\"Qual\",\n")
                file_created = True
            print("Working on {} {} {}".format(station,filename,year))
            for iline,line in enumerate(f.readlines()[3:]):
                if line and len(line) > 5:
                    ndata=ndata+1
                    stampstr0,val,flag,comment = line.strip().split(",")[:4]
                    # This is here for adding more exact start time 
                    # stamp = dtm.datetime.strptime(stampstr0,"%m/%d/%Y %H:%M:%S")
                    if val and not "\"" in val:
                        val = float(val)
                    else:
                        val = np.nan
                    if flag and not "\"" in flag:
                        flag = int(flag)
                    else:
                        flag = "no_flag"
                    if comment is None: comment = ''
                    destfile.write("{},{},{},{}\n".format(stampstr0,val,flag,comment))
            f.close()
            destfile.close()
            os.remove(localname)
            data_years.append(str(year))
            
                    

if __name__ == '__main__':
    main()