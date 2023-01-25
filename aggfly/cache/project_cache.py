import os
from os.path import exists
import warnings
from functools import lru_cache
from hashlib import sha256
import json
import glob
import warnings
import json

import dill as pickle
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import pygeos
import dask
import dask.array
import rasterio
from rasterio.enums import Resampling
import rioxarray

class ProjectCache:
    
    def __init__(
            self,
            project_dir,
            module_type,
            module_dict,
            reset=False
        ):

        self.project_dir = project_dir
        self.module_type = module_type
        self.module_name = self.module_type.__name__
        self.module_dict = module_dict
        self.module_sha = f'mod-{generate_sha(self.module_dict)}'
        self.tmp_dir = f'{self.project_dir}/tmp/{self.module_name}/{self.module_sha}'
        
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
            with open(f"{self.tmp_dir}/mod.json", "w") as outfile:
                json.dump(dictionary, outfile)
            
        if reset:
            self.reset()
    
    def reset(self):     
        files = glob.glob(f'{self.tmp_dir}/*')
        for f in files:
            os.remove(f)
    
    def uncache(self, obj_dict, extension='.nc'): 
        obj_sha = generate_sha(obj_dict)
        file_name = f'{self.temp_dir}/{obj_sha}{extension}'
        if exists(file_name):
            return load(file_name)
        else
            return None
        
    def cache(self, obj, obj_dict, extension='.nc', replace=False):
        obj_sha = generate_sha(obj_dict)
        file_name = f'{self.temp_dir}/{obj_sha}{extension}'
        if exists(file_name):
            if replace:
                save(obj, file_name)
            else:
                raise RuntimeError(
                    "Cached file already exists! " +
                    "Set replace=True to overwrite")
        else
            save(obj, file_name)

def save(self, obj, file_name, extension):
    if extension == '.nc':
        obj.to_netcdf(file_name)
    elif name == 'Dataset':
        if 'xarray' in str(type(obj)):
            obj.to_netcdf(f'{file_name}.nc')
        else:
            pickle_save(obj, file_name)
    else:
        pickle_save(obj, file_name)

def load(self, file_name, extension):
    # name = type(obj).__name__
    if extension == 'nc':
        return xr.open_dataset(f'{file_name}.nc')
    elif extension == 'pickle':
        if 'xarray' in str(type(obj)):
            return xr.open_dataset(f'{file_name}.nc')
        else:
            return pickle_load(file_name)
    else:
        pickle_save(obj, file_name)
    
def generate_sha(in_dict, num=15):
    dump = json.dumps(str(in_dict),sort_keys=True).encode('utf8')
    return sha256(dump).hexdigest()[:num]
        
def pickle_save(obj, file_name):
    with open(f'{file_name}.pickle', 'wb') as handle:
        pickle.dump(obj, handle)     
        
def pickle_load(file_name):
    with open(f'{file_name}.pickle', 'rb') as handle:
        b = pickle.load(handle)
    return b

# def get
# def exists(obj_sha):
#     files = glob.glob(f'{self.tmp_dir}/{obj_sha}.*')
#     if len(files)==0:
#         return False
#     if len(files)==1:
#         return files[0]
#     else:
#         warnings.warn(
#             "Multiple files cached with different file extensions! " +
#             f"Consider resetting cache. Arbitrarily using {files[0]}. ")
#         return files[0]