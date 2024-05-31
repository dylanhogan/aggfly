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
    def __init__(
        self, raster, crop="corn", name=None, feed=None, path=None, project_dir=None
    ):
        super().__init__(raster, name, path, project_dir)
        self.wtype = crop
        self.feed = feed
        
        self.cache = initialize_cache(self)

    def cdict(self):
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
    da = open_crop_raster(path, crop)
    if crs is not None:
        da = da.rio.write_crs(crs)
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
    if name == "cropland":
        # path = "/home3/dth2133/data/cropland/2021_crop_mask.zarr"
        path = "/home3/dth2133/data/cropland/avg_2008-2021_crop_mask.zarr"
        # preprocess =
        crs = "EPSG:5070"
    elif name == "GAEZ":
        path = f"/home3/dth2133/data/GAEZ/GAEZ_2015_all-crops_{feed}.nc"

    else:
        raise NotImplementedError
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
    # Separate file path from file extension
    file, ex = os.path.splitext(path)

    if ex == ".tif":
        da = rioxarray.open_rasterio(path, chunks=True, lock=False, **kwargs)

        if preprocess is not None:
            da = preprocess(da, crop)
        else:
            da = format_cropland_tif_da(da, crop)

    elif ex == ".zarr":
        da = xr.open_zarr(path, **kwargs)
        da = da.layer.sel(crop=crop)
    elif ex == ".nc":
        da = xr.open_dataset(path, **kwargs)
        da = da.layer.sel(crop=crop)
    else:
        raise NotImplementedError

    return da


def format_cropland_tif_da(da, crop):
    return (
        da.isin([cropland_id(crop)])
        .drop("band")
        .squeeze()
        .expand_dims("crop")
        .assign_coords(crop=("crop", np.array(self.crop_dict[num]).reshape(1)))
        .to_dataset(name="layer")
    )


def cropland_id(crop):
    crop_dict = {
        "corn": 1,
        "cotton": 2,
        "rice": 3,
        "sorghum": 4,
        "soybeans": 5,
        "spring wheat": 23,
    }
    return crop_dict[crop]
