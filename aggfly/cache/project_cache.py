import os
from os.path import exists
import warnings
from functools import lru_cache
from hashlib import sha256
import json
import glob
import warnings
import json
from pprint import pformat, pprint

import dill as pickle
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from rasterio.enums import Resampling
import yaml


class ProjectCache:
    def __init__(self, project_dir, module_type, module_dict, reset=False):
        self.project_dir = project_dir
        self.module_type = module_type
        self.module_name = self.module_type.__name__
        self.module_dict = module_dict
        self.module_sha = f"mod-{generate_sha(self.module_dict)}"
        self.tmp_dir = f"{self.project_dir}/tmp/{self.module_name}/{self.module_sha}"
        print(self.tmp_dir)
        # If cache directory doesn't exist, create it
        if not os.path.exists(self.tmp_dir):
            print(f"Creating new cache ({self.module_sha}) in {self.tmp_dir}")
            os.makedirs(self.tmp_dir)

            # This file summarizes parameters used to instance cache
            with open(f"{self.tmp_dir}/mod.yaml", "w") as outfile:
                yaml.dump(self.module_dict, outfile, default_flow_style=False)

        # Reset the cache quickly if needed
        if reset:
            self.reset()

    def reset(self):
        # Delete files in temp directory.
        files = glob.glob(f"{self.tmp_dir}/*")
        for f in files:
            os.remove(f)

    def uncache(self, obj_dict, extension=".nc"):
        # Generate unique ID for the object
        obj_sha = generate_sha(obj_dict)
        file_name = f"{self.tmp_dir}/{obj_sha}"

        # Check if the unique ID exists and load; otherwise None.
        if exists(f"{file_name}{extension}"):
            return load(file_name, extension)
        else:
            print(f"Cache file {file_name}{extension} not found.")
            pprint(obj_dict)
            return None

    def cache(self, obj, obj_dict, extension=".nc", replace=False):
        # Generate unique ID for the object
        obj_sha = generate_sha(obj_dict)
        file_name = f"{self.tmp_dir}/{obj_sha}"

        # If it exists raise error unless expected; otherwise save
        if exists(f"{file_name}{extension}"):
            if replace:
                save(obj, file_name, extension)
            else:
                raise RuntimeError(
                    "Cached file already exists! " + "Set replace=True to overwrite"
                )
        else:
            save(obj, file_name, extension)
        with open(f"{file_name}.yaml", "w") as outfile:
            yaml.dump(obj_dict, outfile, default_flow_style=False)


def save(obj, file_name, extension):
    if extension == ".nc":
        obj.to_netcdf(f"{file_name}.nc")
    elif extension == ".feather":
        obj.to_feather(f"{file_name}.feather")
    elif name == "Dataset":
        if "xarray" in str(type(obj)):
            obj.to_netcdf(f"{file_name}.nc")
        else:
            pickle_save(obj, file_name)
    else:
        pickle_save(obj, file_name)


def load(file_name, extension):
    # name = type(obj).__name__
    if extension == ".nc":
        ds = xr.open_dataset(f"{file_name}.nc")
        varns = list(ds.keys())
        if len(varns) == 1:
            return ds[varns[0]].load()
        else:
            return ds.load()
    elif extension == ".feather":
        return pd.read_feather(f"{file_name}.feather")
    elif extension == "pickle":
        return pickle_load(file_name)
    else:
        pickle_load(file_name)


def generate_sha(in_dict, num=15):
    dump = json.dumps(str(in_dict), sort_keys=True).encode("utf8")
    return sha256(dump).hexdigest()[:num]


def pickle_save(obj, file_name):
    with open(f"{file_name}.pickle", "wb") as handle:
        pickle.dump(obj, handle)


def pickle_load(file_name):
    with open(f"{file_name}.pickle", "rb") as handle:
        b = pickle.load(handle)
    return b


def initialize_cache(obj):
    if obj.project_dir is not None:
        return ProjectCache(obj.project_dir, type(obj), obj.cdict())
    else:
        return None


def clean_object(obj):
    out = {}
    dic = obj.__dict__
    for o in dic.keys():
        if o != "cache":
            out[o] = str(dic[o])
    return out


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
