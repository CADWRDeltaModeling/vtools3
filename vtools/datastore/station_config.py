import yaml
import os

config = None    

    
with open("station_config.yaml", 'r') as stream:
    config = yaml.load(stream,Loader=yaml.FullLoader)
    
def config_file(label):
    fname = config[label]
    localdir = os.path.split(__file__)[0]
    # is path in local directory?
    localpath = os.path.join(localdir,fname)
    if os.path.exists(localpath):  
        return localpath
    else:
        if os.path.exists(fname): 
            return fname
        else:
            raise ValueError(f"Path not found {fname} for label {label}")