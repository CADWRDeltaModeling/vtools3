import yaml
import os

config = None    
localdir = os.path.split(__file__)[0]
    
with open(os.path.join(localdir,"station_config.yaml"), 'r') as stream:
    config = yaml.load(stream,Loader=yaml.FullLoader)

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