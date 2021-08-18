#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import pandas as pd
from vtools.datastore import station_config

def station_info(search):
    station_lookup = station_config.config_file("station_dbase")
    #vlookup = station_config.config_file("variable_mappings")
    slookup = pd.read_csv(station_lookup,sep=",",comment="#",header=0,usecols=["id","agency","agency_id","name","x","y"]).squeeze()
    slookup["id"] = slookup.id.str.lower()
    lsearch = search.lower()
    match_id = slookup["id"].str.lower().str.contains(lsearch)
    match_name = slookup.name.str.lower().str.contains(lsearch)
    match_agency_id = slookup.agency_id.str.lower().str.contains(lsearch)
    matches = match_id | match_name | match_agency_id 
    print("Matches:")
    mlook =slookup.loc[matches,["id","agency","agency_id","name","x","y"]].sort_values(axis=0,by='id').set_index("id") 
    if mlook.shape[0] == 0: 
        print("None")
    else:
        print(mlook)
    return mlook
    

def main():
    searchphrase = sys.argv[1]
    if searchphrase is None:
        raise ValueError("Usage 'station_info searchphrase'")
    station_info(searchphrase)