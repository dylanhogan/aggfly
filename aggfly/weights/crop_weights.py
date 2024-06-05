import os
import warnings
from functools import lru_cache
from hashlib import sha256
import json
from pprint import pformat, pprint

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from rasterio.enums import Resampling

from .secondary_weights import RasterWeights
from ..dataset import reformat_grid
from ..cache import *


class CropWeights(RasterWeights):
    """
    A class to represent crop-specific raster weights.
    
    Attributes
    ----------
    wtype : str
        The type of crop.
    feed : str, optional
        The type of feed.
    cache : dict
        The cache for the CropWeights object.
    """

    def __init__(
        self, raster, crop="corn", name=None, feed=None, path=None, project_dir=None
    ):
        """
        Initializes the CropWeights object.

        Parameters
        ----------
        raster : xarray.DataArray
            The raster data array.
        crop : str, optional
            The type of crop (default is "corn").
        name : str, optional
            The name of the CropWeights object (default is None).
        feed : str, optional
            The type of feed (default is None).
        path : str, optional
            The path to the raster file (default is None).
        project_dir : str, optional
            The project directory (default is None).
        """
        # Initialize the parent class (RasterWeights)
        super().__init__(raster, name, path, project_dir)
        # Set the type of crop
        self.wtype = crop
        # Set the type of feed
        self.feed = feed
        # Initialize the cache
        self.cache = initialize_cache(self)

    def cdict(self):
        """
        Returns a dictionary with the attributes of the CropWeights object.

        Returns
        -------
        dict
            A dictionary with the attributes of the CropWeights object.
        """
        gdict = {
            "name": self.name,
            "path": self.path,
            "feed": self.feed,
            "crop": self.wtype,
            "raster": pformat(self.raster),
        }
        return gdict


def crop_weights_from_path(
    path,
    crop="corn",
    grid=None,
    write=False,
    name=None,
    feed=None,
    project_dir=None,
    crs=None,
):
    """
    Creates a CropWeights object from a raster file.

    Parameters
    ----------
    path : str
        The path to the raster file.
    crop : str, optional
        The type of crop (default is "corn").
    grid : xarray.DataArray, optional
        The grid to use for reformatting the raster (default is None).
    write : bool, optional
        A flag indicating if the weights should be written to a file (default is False).
    name : str, optional
        The name of the CropWeights object (default is None).
    feed : str, optional
        The type of feed (default is None).
    project_dir : str, optional
        The project directory (default is None).
    crs : str, optional
        The coordinate reference system to use (default is None).

    Returns
    -------
    CropWeights
        The created CropWeights object.
    """
    # Open the raster file as an xarray DataArray
    da = open_crop_raster(path, crop)
    # Write the coordinate reference system if provided
    if crs is not None:
        da = da.rio.write_crs(crs)
    # Create a CropWeights object using the raster data array and other parameters
    weights = CropWeights(da, crop, name, feed, path, project_dir)

    return weights


def crop_weights_from_name(
    name="cropland",
    crop="corn",
    grid=None,
    feed=None,
    write=False,
    project_dir=None,
    crs=None,
):
    """
    Creates CropWeights object based on the given name and other parameters.

    Parameters
    ----------
    name : str, optional
        The name of the crop weights source (default is "cropland").
    crop : str, optional
        The type of crop (default is "corn").
    grid : xarray.DataArray, optional
        The grid to use for reformatting the raster (default is None).
    feed : str, optional
        The type of feed (default is None).
    write : bool, optional
        A flag indicating if the weights should be written to a file (default is False).
    project_dir : str, optional
        The project directory (default is None).
    crs : str, optional
        The coordinate reference system to use (default is None).

    Returns
    -------
    CropWeights
        The created CropWeights object.
    """
    if name == "cropland":
        # Specify the path to the cropland raster file
        # path = "/home3/dth2133/data/cropland/2021_crop_mask.zarr"
        path = "/home3/dth2133/data/cropland/avg_2008-2021_crop_mask.zarr"
        # preprocess =
        # Set the coordinate reference system
        crs = "EPSG:5070"
    elif name == "GAEZ":
        # Specify the path to the GAEZ raster file based on the feed
        path = f"/home3/dth2133/data/GAEZ/GAEZ_2015_all-crops_{feed}.nc"

    else:
        # Raise an error if the name is not supported
        raise NotImplementedError
        
    # Create and return the CropWeights object using the specified parameters
    return crop_weights_from_path(
        path,
        crop=crop,
        grid=grid,
        write=write,
        name=name,
        feed=feed,
        project_dir=project_dir,
        crs=crs,
    )


def open_crop_raster(path, crop, preprocess=None, **kwargs):
    """
    Opens a crop raster file and returns it as an xarray DataArray.

    Parameters
    ----------
    path : str
        The path to the raster file.
    crop : str
        The type of crop.
    preprocess : callable, optional
        A function to preprocess the raster data (default is None).

    Returns
    -------
    # Separate file path from file extension
    file, ex = os.path.splitext(path)

    if ex == ".tif":
        # Open the raster file as a GeoTIFF
        da = rioxarray.open_rasterio(path, chunks=True, lock=False, **kwargs)

        if preprocess is not None:
            # Apply the preprocessing function if provided
            da = preprocess(da, crop)
        else:
            # Format the raster data array for cropland GeoTIFF
            da = format_cropland_tif_da(da, crop)

    elif ex == ".zarr":
        # Open the raster file as a Zarr dataset
        da = xr.open_zarr(path, **kwargs)
        # Select the specified crop
        da = da.layer.sel(crop=crop)
    elif ex == ".nc":
        # Open the raster file as a NetCDF dataset
        da = xr.open_dataset(path, **kwargs)
        # Select the specified crop
        da = da.layer.sel(crop=crop)
    else:
        # Raise an error if the file extension is not supported
        raise NotImplementedError

    return da


def format_cropland_tif_da(da, crop):
    """
    Formats a cropland GeoTIFF data array for a specified crop.

    Parameters
    ----------
    da : xarray.DataArray
        The data array to format.
    crop : str
        The type of crop.

    Returns
    -------
    xarray.Dataset
        The formatted data array as a dataset.
    """
    return (
        da.isin([cropland_id(crop)]) # Check if the data array contains the specified crop ID
        .drop("band") # Drop the "band" dimension
        .squeeze()  # Remove singleton dimensions
        .expand_dims("crop") # Add the "crop" dimension
        .assign_coords(crop=("crop", np.array(self.crop_dict[num]).reshape(1))) # Assign the crop name as a coordinate
        .to_dataset(name="layer") # Convert to a dataset with the variable name "layer"
    )


def cropland_id(crop):
    """
    Returns the crop ID for a specified crop.

    Parameters
    ----------
    crop : str
        The type of crop.

    Returns
    -------
    int
        The crop ID.
    """
    # Dictionary mapping crop names to IDs
    crop_dict = {
        "corn": 1,
        "cotton": 2,
        "rice": 3,
        "sorghum": 4,
        "soybeans": 5,
        "spring wheat": 23,
    }
    return crop_dict[crop]
