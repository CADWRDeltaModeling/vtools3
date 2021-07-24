#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd


def process_station_list(fname,id_col="id",agency_id_col="agency_id",
                         param_col=None,param=None,
                         subloc_col=None,subloc_lookup=None,subloc="default",
                         station_lookup=None,param_lookup=None,source = 'cdec'):
    """Process a csv list of station
    
    Parameters
    fname : str
    File containing the list of requests
    
    id_col : str
    Name of column containing station id. This is expected to be the agency id unless station_lookup is provided
    
    subloc_col : str
    Addional location or program name needed to identify the data stream. An example of location might be upper and lower sensor. An example of program having its own instrument would be the bgc (biogeochemistry) program for USGS in the Bay-Delta. 
    
    param_col : str
    Name of column containing the parameter for the query. This is expected to be an agency code/name unless station_lookup is provided. If param is provided, this should be None.
    
    param : str
    Requested parameter. Should be an agency code unless param_lookup is given. This will be used for every station.
    
    param_lookup: str
    Lookup file containing columns "variable_name" and "agency_variable_name"
    
    station_lookup
    Lookup file containing "id" and "agency_id" columns.
    
    
    """
    
    
    if param_col is not None and param is not None:
        raise ValueError("Cannot use both param_col and param arguments")
    
    
    
    station_df = pd.read_csv(fname,sep=",",comment="#",header=0)
    
    # Rename the parameter column as param to standardize
    # If there is no param column, create one and populate all its rows with param.
    if param is not None:          # only one parameter, append as column
        station_df["param"] = param
    else:
        if param_col is not None: 
            station_df.rename(columns={param_col:"param"})
    
    if not "subloc" in station_df.columns: 
        station_df["subloc"]="default"
    else:
        station_df["subloc"] = station_df.subloc.astype(str)
    if station_lookup:
        slookup = pd.read_csv(station_lookup,sep=",",comment="#",header=0,index_col=id_col,usecols=["id","agency_id","name"]).squeeze()
        station_df["id"] = station_df.id.str.lower()
        station_df = station_df.merge(slookup,on="id",how="left")
        station_df.loc[station_df.subloc.isin(['nan','']),'subloc'] = "default"
    else: station_df["agency_id"] = station_df.id

    # Replace parameters with lookup values from station_lookup. Failure will leave as-is, in case of mix.
    if param_lookup:
        vlookup = pd.read_csv(param_lookup,sep=",",comment="#",header=0,usecols=['var_name','src_var_id','src_name'],dtype=str)
        vlookup = vlookup.loc[vlookup.src_name == source,:]
        vlookup.rename(columns={"var_name":"param"},inplace=True)
        station_df = station_df.merge(vlookup,on="param",how="left")
        station_df = station_df.fillna(value={"src_var_id":station_df.param})

    station_df = station_df.rename(columns={"id" : "station_id"})
    # Any nans in the agency_id indicate a lookup failure. As a backup assume the agency_id was already provided in the id column
    station_df["agency_id"].fillna(station_df.station_id,inplace=True)
    
    return station_df[["station_id","agency_id","subloc","param","src_var_id"]]
            

        
if __name__ == '__main__':
    slookup = "d:/delta/BayDeltaSCHISM/data/stations_utm_new.csv"
    vlookup = "D:/Delta/data_tools/dms_data_tools/dms_data_tools/variable_mappings.csv"
    df = process_station_list("slist.csv",param="ec",station_lookup=slookup,agency_id_col="cdec_id",
                              param_lookup=vlookup)
    print("\n")
    print(df)



      


