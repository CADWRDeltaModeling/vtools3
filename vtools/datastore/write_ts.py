#!/usr/bin/env python
# -*- coding: utf-8 -*-
import yaml
import pandas as pd
import warnings

def chunk_bounds(ts,block_size):
    firstyr = ts.first_valid_index().year
    lastyr = ts.last_valid_index().year
    # Get a number that is "neat" w.r.t. block size
    neat_lower_bound = int(block_size*(firstyr//block_size))
    neat_upper_bound = int(block_size*(lastyr//block_size)) 
    bounds = []
    for bound in range(neat_lower_bound,neat_upper_bound+1,block_size):
        lo = max(firstyr,bound)
        hi = min(lastyr,bound+block_size-1)
        hi = bound+block_size - 1
        bounds.append((lo,hi))
    return bounds



def block_comment(txt):
    text = txt.split("\n")
    text =  ["# "+x for x in text]
    text = [x.replace("# #","#  ") for x in text]
    return "\n".join(text)

def block_uncomment(txt):
    split = txt.split("\n")
    def uncomment(x):
        return x[1:] if x[0]=="#" else x
    split = [uncomment(x) for x in split  ]
    return "\n".join(split)


def prep_header(metadata,format_version):
    """ Prepares metadata in the form of a string or yaml data structure for inclusion
        Prep includes making sure that the lines are commented and start with the format: line
    """

    if isinstance(metadata,str):
        metadata = metadata.split("\n")
        if not "format" in metadata[0]:
            if metadata[0].startswith("#"):
                metadata = [f"# format: {format_version}",f"# date_formatted: {pd.Timestamp.now().strftime('%Y-%m-%dT%H:%M:%S')}"]+metadata  
            else:
                metadata = [f"format: {format_version}",f"date_formatted: {pd.Timestamp.now().strftime('%Y-%m-%dT%H:%M:%S')}"]+metadata
            # Get rid of conflicting line
            conflict = -1
            for i in range(1,len(metadata)):
                if "format" in metadata[i]: conflict = i
            if conflict > 0: 
                del(metadata[conflict])
        if not metadata[0].startswith("#"):
            metadata = ["# "+x for x in metadata]
            metadata = [x.replace("# #","#") for x in metadata]
        header = "\n".join(metadata)
    else: # yaml
        if "format" in metadata:
            del(metadata["format"])
        if "original_header" in metadata:
            metadata["original_header"] = block_uncomment(metadata["original_header"])
        header_no_comment = yaml.dump(metadata)
        header = block_comment(header_no_comment)
        if not "format" in metadata:
            header = "# format: dwr-dms-1.0\n" + header
    if not header.endswith("\n"): header=header+"\n"            
    return header
        

def write_ts_csv(ts,fpath,metadata=None,chunk_years=False,format_version="dwr-dms-1.0",
                 overwrite_conventions=False,block_size=1,**kwargs):
    """ Write time series to a csv file following a standard format
    Parameters:
    -----------
    ts : pandas.DataFrame 
    The time series to write
    
    fpath : string
    File name to write
    
    metadata : str or dict
    String that represents valid yaml or a yaml-style data structure of dicts and lists
    
    chunk_years : bool
    Break data into chunks by year
    
    dms_data_format : str
    Version number of format. Defaults to current default format
    
    **kwargs : other
    Other items that will be passed to write_csv
    """
    former_index=ts.index.name
    if former_index != "datetime" and not overwrite_conventions: 
        warnings.warn("Index will be renamed datetime in file according to specification. Copy made")
        ts = ts.copy()
        ts.index.name = "datetime"
    if metadata is None:
        meta_header = f"# format: {format_version}\ndate_formatted: {pd.Timestamp.now().strftime('%Y-%m-%dT%H:%M:%S')}"
    else:
        meta_header = prep_header(metadata,format_version)
    
    
    if chunk_years:
        bounds = chunk_bounds(ts,block_size=block_size)
        single_year_label = (block_size==1)
        for bnd in bounds:
            s = max(pd.Timestamp(bnd[0],1,1),ts.first_valid_index())
            e = min(pd.Timestamp(bnd[1],12,31,23,59,59),ts.last_valid_index())
            tssub = ts.loc[s:e]
            new_date_range_str = f"{bnd[0]}_{bnd[1]}"
            if single_year_label:
                if bnd[0] != bnd[1]: 
                    raise ValueError("Blocks not compatible with single_year")
                else: 
                    new_date_range_str = f"{bnd[0]}"
            newfname = fpath
            newfname = fpath.replace(".csv","_"+new_date_range_str + ".csv")  # coerces to csv
            with open(newfname,'w',newline="\n") as outfile:
                outfile.write(meta_header)
                tssub.to_csv(outfile,header=True,sep=",",date_format="%Y-%m-%dT%H:%M:%S",**kwargs)
    else: # not chunk_years
        with open(fpath,'w',newline="\n") as outfile:
            outfile.write(meta_header)
            ts.to_csv(outfile,header=True,sep=",",date_format="%Y-%m-%dT%H:%M:%S",**kwargs)
