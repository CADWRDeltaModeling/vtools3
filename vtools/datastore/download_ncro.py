#!/usr/bin/env python
import urllib.request
import pandas as pd
import re
import zipfile
import os
import string
import datetime as dt
import numpy as np
from vtools.datastore.process_station_variable import process_station_list,stationfile_or_stations
from vtools.datastore import station_config

ncro_inventory_file = "ncro_por_inventory.txt"

def station_dbase():
    dbase_fname=station_config.config_file("station_dbase")
    dbase_df = pd.read_csv(dbase_fname,header=0,comment="#",index_col="id")
    is_ncro = dbase_df.agency.str.lower().str.contains("ncro")
    print(is_ncro[is_ncro.isnull()])
    return dbase_df.loc[is_ncro,:]


def download_ncro_inventory(dest,cache=True):
    url = "https://data.cnra.ca.gov/dataset/fcba3a88-a359-4a71-a58c-6b0ff8fdc53f/resource/cdb5dd35-c344-4969-8ab2-d0e2d6c00821/download/station-trace-download-links.csv"
    idf = pd.read_csv(url,header=0,parse_dates=["first_measurement_date","last_measurement_date"])
    print(idf)
    idf = idf.loc[(idf.station_type != "Groundwater") & (idf.output_interval == "Raw"),:]
    idf.to_csv(os.path.join(dest,ncro_inventory_file),sep=",",index=False,date_format="%Y-%d%-mT%H:%M")    
    return idf
    
def ncro_variable_map():
    varmap = pd.read_csv("variable_mappings.csv",header=0,comment="#")
    return varmap.loc[varmap.src_name=="wdl",:]
    
#station_number,station_type,first_measurement_date,last_measurement_date,parameter,output_interval,download_link

mappings = {
    'Water Temperature':'temp',
    'Stage':'elev',
    'Conductivity':'ec',
    'Electrical Conductivity at 25C':'ec',
    'Fluorescent Dissolved Organic Matter':'fdom',    
    'Water Temperature ADCP':'temp',
    'Dissolved Oxygen':'do',
    'Chlorophyll':'cla',
    'Dissolved Oxygen (%)': None,
    'Dissolved Oxygen Percentage': None,    
    'Velocity':'velocity',
    'pH':'ph',
    'Turbidity':'turbidity',
    'Flow':'flow',
    'Salinity':'salinity'
    }

def download_ncro_period_record(inventory,dbase,dest,variables=["flow","elev","ec","temp","do","ph","turbidity","cla"]):
    global mappings    
    #mappings = ncro_variable_map()
    print(mappings)
    failures = []
    for ndx,row in inventory.iterrows():
        agency_id = row.station_number
        param = row.parameter
        if param in mappings.keys(): 
            var = mappings[param]
            if var is None: continue
        else:
            print("Problem on row:",row)
            if type(param) == float:
                if np.isnan(param): 
                    continue # todo: this is a fix for an NCRO-end bug. Really the ValueError is best
            raise ValueError(f"No standard mapping for NCRO parameter {param}.")

        #printed.add(param)
        var = mappings[param]
        link_url = row.download_link
        sdate = row.first_measurement_date
        edate = row.last_measurement_date        
        entry = None
        ndx = ""        
        for suffix in ['','00','Q']:
            full_id = dbase.agency_id+suffix
            entry = dbase.index[full_id==agency_id]
            if len(entry) > 1:
                raise ValueError(f"multiple entries for agency id {agency_id} in database")
            elif not entry.empty:
               station_id = str(entry[0])
                    
        if station_id is "":
            raise ValueError(f"Item {agency_id} not found in station database after accounting for Q and 00 suffixes")
            
        fname = f"ncro_{station_id}_{agency_id}_{var}_{sdate.year}_{edate.year}.csv".lower()
        fpath = os.path.join(dest,fname)
        print(f"Processing: {agency_id} {param} {sdate} {edate}")
        print(link_url)

        try:
            response = urllib.request.urlopen(link_url)
        except:
            failures.append((station_id,agency_id,var,param))
        else:    
            station_html = response.read().decode().replace("\r","")
            if len(station_html) > 30 and not "No sites found matching" in station_html:
                found = True
                with open(fpath,"w") as f:
                    f.write(station_html)
            if not found: 
                print("Station %s query failed or produced no data" % station)
                failures.append((station_id,agency_id,var,param))
        
        print(f"Writing {fname}")
        
    print("Failures")
    for f in failures: 
        print(f)

def download_ncro_por(dest):
    idf = download_ncro_inventory(dest)
    dbase = station_dbase()
    upper_station = idf.station_number.str.upper()
    is_in_dbase = upper_station.isin(dbase.agency_id) | \
                  upper_station.isin(dbase.agency_id+"00") | \
                  upper_station.isin(dbase.agency_id+"Q")
    
    download_ncro_period_record(idf.loc[is_in_dbase,:],dbase,dest,variables=["flow","elev","ec","temp","do","ph","turbidity","cla"])


def main():
    dest = '.'
    #dest = "//cnrastore-bdo/Modeling_Data/continuous_station_repo/raw/incoming/dwr_ncro"
    download_ncro_por(dest)
    

if __name__ == "__main__":
    main()    
