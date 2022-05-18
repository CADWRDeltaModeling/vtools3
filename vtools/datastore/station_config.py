import yaml
import os
import pandas as pd

config = None    
localdir = os.path.split(__file__)[0]
    
with open(os.path.join(localdir,"station_config.yaml"), 'r') as stream:
    config = yaml.load(stream,Loader=yaml.FullLoader)


def station_dbase(dbase_name=None):
    if dbase_name is None:
        dbase_name = config_file("station_dbase")
    db = pd.read_csv(dbase_name,sep=",",comment="#",header=0,index_col="id",dtype={"agency_id":str})
    db["agency_id"] = db["agency_id"].str.replace("\'","",regex=True)
    
    dup = db.index.duplicated()
    db.index = db.index.str.replace("'","")
    if dup.sum(axis=0)> 0:
        print("Duplicates")
        print(db[dup])
        raise ValueError("Station database has duplicate id keys. See above")
    return db



def configuration():
    config_ret = config.copy()
    config_ret["config_file_location"] = __file__
    return config_ret

def config_file(label):
    fname = config[label]
    
    # is path in local directory?
    localpath = os.path.join(localdir,fname)
    if os.path.exists(localpath):  
        return localpath
    else:
        if os.path.exists(fname): 
            return fname
        else:
            raise ValueError(f"Path not found {fname} for label {label}")