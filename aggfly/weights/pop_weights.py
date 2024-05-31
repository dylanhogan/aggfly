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
        super().__init__(raster, name, path, project_dir)
        self.wtype = "pop"
        self.cache = initialize_cache(self)


def pop_weights_from_path(
    path, grid=None, write=False, name=None, feed=None, project_dir=None, crs=None
):
    da = open_pop_raster(path)
    if crs is not None:
        da = da.rio.write_crs(crs)
    weights = PopWeights(da, name, path, project_dir)

    return weights


def from_name(name="landscan", grid=None, write=False, project_dir=None, crs=None):
    if name == "landscan":
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
    # Separate file path from file extension
    file, ex = os.path.splitext(path)

    if ex == ".tif":
        da = rioxarray.open_rasterio(
            path, chunks=True, lock=False, masked=True, **kwargs
        )

        if preprocess is not None:
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
