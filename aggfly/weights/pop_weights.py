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
import dask.array
from rasterio.enums import Resampling
import rioxarray

from .secondary_weights import RasterWeights
from ..dataset import reformat_grid
from ..cache import *


class PopWeights(RasterWeights):
    def __init__(self, raster, name=None, path=None, project_dir=None):
        """
        Initialize a PopWeights object.

        Parameters
        ----------
        raster : xarray.DataArray
            The raster data array to be used for population weights.
        name : str, optional
            The name of the weights (default is None).
        path : str, optional
            The path to the raster file (default is None).
        project_dir : str, optional
            The project directory (default is None).
        """

        super().__init__(raster, name, path, project_dir)
        self.wtype = "pop" # Set the weight type to 'pop' for population
        self.cache = initialize_cache(self) # Initialize cache for the PopWeights object


def pop_weights_from_path(
    path, grid=None, write=False, name=None, feed=None, project_dir=None, crs=None
):
    """
    Create PopWeights from a file path.

    Parameters
    ----------
    path : str
        The path to the population raster file.
    grid : Grid, optional
        The grid to which the weights will be rescaled (default is None).
    write : bool, optional
        A flag indicating whether to write the weights to a file (default is False).
    name : str, optional
        The name of the weights (default is None).
    feed : str, optional
        The feed type (default is None).
    project_dir : str, optional
        The project directory (default is None).
    crs : str, optional
        The coordinate reference system to use (default is None).

    Returns
    -------
    PopWeights
        The created PopWeights object.
    """
    # Open the population raster file
    da = open_pop_raster(path)
    if crs is not None:
        # Write the CRS to the raster if provided
        da = da.rio.write_crs(crs)
    # Create a PopWeights object
    weights = PopWeights(da, name, path, project_dir)

    return weights


def from_name(name="landscan", grid=None, write=False, project_dir=None, crs=None):
    """
    Create PopWeights from a predefined name.

    Parameters
    ----------
    name : str
        The name of the population dataset (default is 'landscan').
    grid : Grid, optional
        The grid to which the weights will be rescaled (default is None).
    write : bool, optional
        A flag indicating whether to write the weights to a file (default is False).
    project_dir : str, optional
        The project directory (default is None).
    crs : str, optional
        The coordinate reference system to use (default is None).

    Returns
    -------
    PopWeights
        The created PopWeights object.
    """
    if name == "landscan":
        # Define the path to the LandScan dataset
        # path = "/home3/dth2133/data/cropland/2021_crop_mask.zarr"
        path = "/home3/dth2133/data/population/landscan-global-2016.tif"
        # preprocess =
        # crs = 'EPSG:5070'
    else:
        raise NotImplementedError
    return pop_weights_from_path(
        path, grid=grid, write=write, name=name, project_dir=project_dir, crs=crs
    )


def open_pop_raster(path, preprocess=None, **kwargs):
    """
    Open a population raster file.

    Parameters
    ----------
    path : str
        The path to the population raster file.
    preprocess : callable, optional
        A function to preprocess the data when loaded (default is None).

    Returns
    -------
    xarray.DataArray
        The opened population raster data array.
    """
    # Separate file path from file extension
    file, ex = os.path.splitext(path)

    if ex == ".tif":
        # Open the raster file using rioxarray
        da = rioxarray.open_rasterio(
            path, chunks=True, lock=False, masked=True, **kwargs
        )

        if preprocess is not None:
            # Apply preprocessing if provided
            da = preprocess(da)

    # elif ex =='.zarr':
    #     da = xr.open_zarr(path,  **kwargs)
    #     da = da.layer.sel(crop=crop)
    # elif ex == '.nc':
    #     da = xr.open_dataset(path,  **kwargs)
    #     da = da.layer.sel(crop=crop)
    else:
        raise NotImplementedError

    return da
