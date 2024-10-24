# The following code provides functionality for caching data related to aggfly. 
# This is useful for saving time and resources when repeatedly accessing or processing 
# large datasets. It handles the creation, management, and usage of cached files,
# and the saving, loading, and hashing of objects.

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
        """
        Initialize the ProjectCache object.

        Parameters:
        -----------
        project_dir : str
            The directory where the project cache will be stored.
        module_type : type
            The type of the module being cached.
        module_dict : dict
            A dictionary representing the module's parameters.
        reset : bool, optional
            A flag to reset the cache (default is False).
        """
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
        """
        Reset the cache by deleting all files in the temporary directory.
        """
        # List all files in the cache directory
        files = glob.glob(f"{self.tmp_dir}/*")
        for f in files:
            # Remove each file
            os.remove(f)

    def uncache(self, obj_dict, extension=".nc"):
        """
        Load a cached object if it exists.

        Parameters:
        -----------
        obj_dict : dict
            A dictionary representing the object's parameters.
        extension : str, optional
            The file extension of the cached object (default is ".nc").

        Returns:
        --------
        The cached object if it exists, otherwise None.
        """
        # Generate unique ID for the object
        obj_sha = generate_sha(obj_dict)
        # Define the file name based on the hash  (ID)
        file_name = f"{self.tmp_dir}/{obj_sha}"

        # Check if the unique ID exists and load; otherwise None.
        if exists(f"{file_name}{extension}"):
            # Load the cached file
            return load(file_name, extension)
        else:
            # Print a message if the file is not found
            print(f"Cache file {file_name}{extension} not found.")
            pprint(obj_dict)
            return None

    def cache(self, obj, obj_dict, extension=".nc", replace=False):
        """
        Save an object to the cache.

        Parameters:
        -----------
        obj : object
            The object to be cached.
        obj_dict : dict
            A dictionary representing the object's parameters.
        extension : str, optional
            The file extension for saving the object (default is ".nc").
        replace : bool, optional
            A flag to replace the cached object if it already exists (default is False).
        """
        # Generate unique ID for the object
        obj_sha = generate_sha(obj_dict)
        # Define the file name based on the hash (ID)
        file_name = f"{self.tmp_dir}/{obj_sha}"

        # If the cached file exists raise error unless expected; otherwise save
        if exists(f"{file_name}{extension}"):
            if replace:
                # Save the object, replacing the existing file
                save(obj, file_name, extension)
            else:
                raise RuntimeError(
                    "Cached file already exists! " + "Set replace=True to overwrite"
                )
        else:
            # Save the object
            save(obj, file_name, extension)
        with open(f"{file_name}.yaml", "w") as outfile:
            # Save the object dictionary as a YAML file
            yaml.dump(obj_dict, outfile, default_flow_style=False)


def save(obj, file_name, extension):
    """
    Save an object to a file.

    Parameters:
    -----------
    obj : object
        The object to save.
    file_name : str
        The name of the file.
    extension : str
        The file extension.
    """
    if extension == ".nc":
        # Save as NetCDF
        obj.to_netcdf(f"{file_name}.nc")
    elif extension == ".feather":
        # Save as Feather
        obj.to_feather(f"{file_name}.feather")
    elif name == "Dataset":
        # Save xarray dataset as NetCDF
        if "xarray" in str(type(obj)):
            obj.to_netcdf(f"{file_name}.nc")
        else:
            # Save using pickle
            pickle_save(obj, file_name)
    else:
        # Save using pickle
        pickle_save(obj, file_name)


def load(file_name, extension):
    """
    Load an object from a file.

    Parameters:
    -----------
    file_name : str
        The name of the file.
    extension : str
        The file extension.

    Returns:
    --------
    object
        The loaded object.
    """
    if extension == ".nc":
        # Load NetCDF file
        ds = xr.open_dataset(f"{file_name}.nc") 
        varns = list(ds.keys())
        if len(varns) == 1:
            # Return single variable dataset
            return ds[varns[0]].load()
        else:
            # Return dataset with multiple variables
            return ds.load()
    elif extension == ".feather":
        # Load Feather file
        return pd.read_feather(f"{file_name}.feather")
    elif extension == "pickle":
        # Load using pickle
        return pickle_load(file_name)
    else:
        # Load using pickle
        pickle_load(file_name)


def generate_sha(in_dict, num=15):
    """
    Generate a SHA-256 hash for a dictionary.

    Parameters:
    -----------
    in_dict : dict
        The dictionary to hash.
    num : int, optional
        The length of the hash (default is 15).

    Returns:
    --------
    str
        The generated hash.
    """
    # Serialize the dictionary
    dump = json.dumps(str(in_dict), sort_keys=True).encode("utf8")
    # Return the hash
    return sha256(dump).hexdigest()[:num]


def pickle_save(obj, file_name):
    """
    Save an object to a pickle file.

    Parameters:
    -----------
    obj : object
        The object to save.
    file_name : str
        The name of the file.
    """
    # Open file in write-binary mode
    with open(f"{file_name}.pickle", "wb") as handle:
        # Save object using pickle
        pickle.dump(obj, handle)


def pickle_load(file_name):
    """
    Load an object from a pickle file.

    Parameters:
    -----------
    file_name : str
        The name of the file.

    Returns:
    --------
    object
        The loaded object.
    """
    # Open file in read-binary mode
    with open(f"{file_name}.pickle", "rb") as handle:
        # Load object using pickle
        b = pickle.load(handle)
    return b


def initialize_cache(obj):
    """
    Initialize a cache for an object if a project directory is specified.

    Parameters:
    -----------
    obj : object
        The object to cache.

    Returns:
    --------
    ProjectCache or None
        The initialized cache or None if no project directory is specified.
    """
    if obj.project_dir is not None:
        # Return a ProjectCache instance
        return ProjectCache(obj.project_dir, type(obj), obj.cdict())
    else:
        return None


def clean_object(obj):
    """
    Clean an object by creating a dictionary of its attributes.

    Parameters:
    -----------
    obj : object
        The object to clean.

    Returns:
    --------
    dict
        The cleaned dictionary.
    """
    out = {}
    dic = obj.__dict__
    for o in dic.keys():
        # Exclude the cache attribute
        if o != "cache":
            # Add attribute to the dictionary
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
