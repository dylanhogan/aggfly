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
import pygeos
import dask
import dask.array
import rasterio
from rasterio.enums import Resampling
import rioxarray

from ..dataset import reformat_grid
from ..cache import *


class PopWeights:
    def __init__(self, raster, name=None, path=None, project_dir=None):
        self.raster = raster
        self.wtype = "pop"
        self.name = name
        self.path = path
        self.project_dir = project_dir
        self.cache = initialize_cache(self)

    def rescale_raster_to_grid(
        self,
        grid,
        verbose=False,
        resampling=Resampling.average,
        nodata=0,
        return_raw=False,
    ):
        gdict = {"func": "rescale_raster_to_grid", "grid": clean_object(grid)}

        if self.cache is not None:
            cache = self.cache.uncache(gdict)
        else:
            cache = None

        if cache is not None:
            print(f"Loading rescaled pop weights from cache")
            self.raster = cache
            if verbose:
                print("Cache dictionary:")
                pprint(gdict)
        else:
            print(f"Rescaling pop weights to grid.")
            print("This might take a few minutes and use a lot of memory...")

            g = xr.DataArray(
                    data=np.empty_like(grid.centroids()),
                    dims=["y", "x"],
                    coords=dict(
                        x=(["x"], np.float64(grid.longitude.values)), 
                        y=(["y"], np.float64(grid.latitude.values))
                    )
                ).rio.write_crs("WGS84")

            dsw = self.raster.rio.reproject_match(g, nodata=nodata, resampling=resampling)
            
            dsw = dsw.rename({'x': 'longitude', 'y': 'latitude'}).squeeze()

            if return_raw:
                return dsw.values.squeeze()

            self.raster = dsw

            if self.cache is not None:
                self.cache.cache(self.raster, gdict)

    def cdict(self):
        gdict = {
            "wtype": self.wtype,
            "name": self.name,
            "path": self.path,
            "raster": pformat(self.raster),
        }
        return gdict


def pop_weights_from_path(
    path, grid=None, write=False, name=None, feed=None, project_dir=None, crs=None
):
    da = open_raster(path)
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


def open_raster(path, preprocess=None, **kwargs):
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
