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
from vtools.datastore.process_station_variable import process_station_list
from vtools.datastore import station_config


items = {"flow": "FLOW_15-MINUTE",
         "elev": "STAGE_15-MINUTE",
         "ec": "CONDUCTIVITY_POINT",
         "velocity" : "VELOCITY",
         "temp": "WATER_TEMPERATURE_POINT",
         #"adcp_water_temperature": "ADCP_WATER_TEMPERATURE",
         "turbidity": "TURBIDITY",
         "ph": "PH_POINT",
         "do": "OXYGEN_DISSOLVED_POINT",
         "chl": "TOTAL_CHLOROPHYLL_POINT"
         }


def create_arg_parser():
    parser = argparse.ArgumentParser()
    paramhelp = 'Variable to download, should be in list:\n%s' % list(items.keys())
    
    parser.add_argument('--dest', dest = "dest_dir", default="wdl_download", help = 'Destination directory for downloaded files.')
    parser.add_argument('--id_col',default = None, help = 'Column in station file representing station. IDs not matched will trigger failure. Header is assumed to represent column names')
    parser.add_argument('--expand_id',default = True, help = 'Expand station ids to include all NCRO programs (try suffices with 00 and Q and blank ends)')
    parser.add_argument('--param_col',default = None, help = 'Column representing the parameter to download. Parameter can be ommitted if it is all the same and given in the --param argument')
    #parser.add_argument('--start',required=True,help = 'Start time, format 2009-03-31 14:00')    
    #parser.add_argument('--end',default = None,help = 'Start time, format 2009-03-31 14:00')    
    parser.add_argument('--syear',required=True,help = 'Start year (these are water years starting previous October')    
    parser.add_argument('--eyear',default = None,help = 'End year, if blank current year (water year)')
    parser.add_argument('--param',help=paramhelp)
    parser.add_argument('--overwrite',  action="store_true", help = 'Overwrite existing files (if False they will be skipped)')
    parser.add_argument('stationfile', help = 'CSV-format station file.')    
    return parser


def process_station_list2(file,id_col=None,param_col=None,param=None):
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
    param_column = args.param_col
    destdir = args.dest_dir
    stationfile = args.stationfile
    overwrite = args.overwrite
    param = args.param
    expand_id = args.expand_id
    syear = int(args.syear)
    eyear = args.eyear
    if eyear is None: 
        today = dtm.datetime.today()
        eyear = today.year
    eyear = int(eyear)

    if param_column and param:
        raise ValueError("param_col and param cannot both be specified")
    if not (param_column or param):
        raise ValueError("Either param_col or param must be specified")
    if os.path.exists(stationfile):
        slookup = station_config.config_file("station_dbase")
        vlookup = station_config.config_file("variable_mappings")
        df = process_station_list(stationfile,param=param,station_lookup=slookup,
                                  agency_id_col="agency_id",param_lookup=vlookup,source='wdl')
        wdl_download(df,destdir,
                      destdir,syear,
                      eyear,
                      overwrite,expand_id)
    else:
        print("Station list does not exist")


def candidate_id(orig,param):
    base = orig
    if len(base) > 6:
        if base.endswith('00'): base = base[:-2]
        if base.lower().endswith('Q'): base = base[:-1]
    candidates = [base]
    if param in ['elev']: 
        pass
    elif param in ['flow',"adcp_water_temperature","velocity","temperature"]: 
        candidates.append(base+'Q')
    else: 
        candidates.append(base+'00')
    return candidates

aliases = None
alias_path = os.path.join(os.path.split(__file__)[0],'wdl_continuous_files_alias.csv')


def wdl_download(station_list,years,dest_dir,syear,eyear,overwrite=False,expand_id=True):
    """ Download robot for Water Data Library
    First loads the stations page to see which variables are available 
    from a hardwired list (conductance, gage elevation, flow, etc). 
    Then it downloads all the years from years that are available for 
    those variables and adds the data to txt files one per station-variable. 
    """
    #station_list = list(station_list)
    print(station_list)
    
    global aliases
    # prior url
    #base_url = "http://wdl.water.ca.gov/waterdatalibrary/docs/Hydstra/docs"
    base_url = "https://wdlstorageaccount.blob.core.windows.net/continuousdata/docs"
    work_dir = "tempfiles"
    retain_zip = False


    if aliases is None:
        aliases = pd.read_csv(alias_path)
    years = np.arange(int(syear),int(eyear)+1)
    skips = []
    fails = []

    if not os.path.exists(work_dir):
        os.mkdir(work_dir)

    for ndx,row in station_list.iterrows():
        agency_id = row.agency_id
        station = row.station_id
        param = row.src_var_id
        paramname = row.param
        subloc = row.subloc  

        yearname = f"{syear}_{eyear}" if syear != eyear else f"{syear}"
        
        if expand_id:
            station_candidates = candidate_id(agency_id,paramname)        
        else:
            station_candidates = [agency_id]
        for agency_id in station_candidates:
            outfname = f"ncro_{station}_{agency_id}_{paramname}_{yearname}.csv"
            outfname = outfname.lower()
            path = os.path.join(dest_dir,outfname)
            if os.path.exists(path):
                if overwrite:
                    os.remove(dest_path)           
                else:
                    print("\nSkipping existing station because file exists: %s" % outfname)
                    skips.append(path)
                    continue
            else:
                print(f"\nAttempting to download station: {station} {agency_id} variable {paramname}")  
            alias_sub = aliases.loc[aliases.FILE_NAME_BASE.str.startswith(items[paramname]) ,:]
            if alias_sub.shape[0] != 1: 
                print(alias_sub)
                raise ValueError(f"Lookup failure on WDL alias table (nonunique) for station {agency_id} and param {param}")
            file_created = False
            ndata = 0
            data_years = []
            #alias_sub = alias_sub.loc[alias_sub.POR_YEAR_FOLDER != 'POR',:]
            #alias_sub = alias_sub.astype({'POR_YEAR_FOLDER':int})     

            for year in years:
                #yrdf = alias_sub.loc[alias_sub.POR_YEAR_FOLDER == year,:]
                #if len(yrdf) == 0: continue
                filename = alias_sub['TABLE_FILE_NAME'].iloc[0]
                station_query="{}/{}/{}/{}".format(base_url,agency_id.upper(),year,filename)
                print(station_query)
                try:
                    response = urllib2.urlopen(station_query)
                except Exception as e:
                    if 'blob does not exist' in e.reason: 
                        print("No data found")
                        continue
                    
                fname = station+"_"+str(year)+"_"+filename
                fname = fname.lower()
                localname = os.path.join(work_dir,fname)
                with open(localname, "wb") as local_file:
                    local_file.write(response.read())

                f = open(localname,"r")             
                mode = "a" if file_created else "w"
                destfile = open(path,mode)

                if not file_created:
                    destfile.write("\"Date\",\"Point\",\"Qual\",\n")
                    file_created = True                
                    
                print("Working on {} {} {}".format(station,filename,year))
                for iline,line in enumerate(f.readlines()[3:]):
                    if line and len(line) > 5:
                        ndata=ndata+1
                        try:
                            stampstr0,val,flag,comment = line.strip().split(",")[:4]
                        except:
                            comment = None
                            stampstr0,val,flag = line.strip().split(",")[:3]                         
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
            if not file_created: fails.append((station,paramname))
    print("Fails:")
    for fl in fails: print(fl)
        
                    

if __name__ == '__main__':
    main()