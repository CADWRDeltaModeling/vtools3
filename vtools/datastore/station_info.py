#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import pandas as pd
import argparse
from vtools.datastore import station_config

def station_info(search):
    station_lookup = station_config.config_file("station_dbase")
    if search == "config":
        print(station_config.configuration())
        return
    #vlookup = station_config.config_file("variable_mappings")
    #slookup = pd.read_csv(station_lookup,sep=",",comment="#",header=0,usecols=["id","agency",
    #                                                                           "agency_id","name",
    #                                                                           "x","y","lat","lon"]).squeeze()
    slookup = station_config.station_dbase()[["agency","agency_id","name","x","y","lat","lon"]]
    slookup.loc[:,"station_id"] = slookup.index.str.lower()
    lsearch = search.lower()
    match_id = slookup.station_id.str.contains(lsearch)
    match_name = slookup.name.str.lower().str.contains(lsearch)
    match_agency_id = slookup.agency_id.str.lower().str.contains(lsearch)
    match_agency = slookup.agency.str.lower().str.contains(lsearch)
    matches = match_id | match_name | match_agency_id | match_agency
    print("Matches:")
    mlook =slookup.loc[matches,["station_id","agency","agency_id","name","x","y","lat","lon"]].sort_values(axis=0,by='station_id')  #.set_index("id") 
    if mlook.shape[0] == 0: 
        print("None")
    else:
        print(mlook.to_string())
    return mlook
    
    
def create_arg_parser():
    parser = argparse.ArgumentParser("Lookup station metadata by partial string match on id or name")
    parser.add_argument('--config',default=False,action ="store_true",help="Print configuration and location of lookup files")
    parser.add_argument('searchphrase',nargs='?',default="",help = 'Search phrase which can be blank if using --config')

    return parser    



def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    searchphrase = args.searchphrase
    if args.config:
        searchphrase = "config"
    if searchphrase is None and not args.config:
        raise ValueError("searchphrase required")
    station_info(searchphrase)