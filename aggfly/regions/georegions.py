import numpy as np
import pandas as pd
import xarray as xr

# import pygeos
import geopandas as gpd
import dask_geopandas
import os
import dask
import dask.array
from functools import lru_cache
from copy import deepcopy
import warnings

from .shp_utils import *


class GeoRegions:
    def __init__(
        self, shp=None, regionid="state", region_list=None, name=None, path=None
    ):
        self.shp = shp.reset_index(drop=True)
        self.regionid = regionid
        self.regions = self.shp[self.regionid]
        if region_list is not None:
            self.sel(region_list, update=True)
        self.name = name
        self.path = path

    # @lru_cache(maxsize=None)
    def poly_array(self, buffer=0, datatype="array", chunks=20):
        # poly = pygeos.from_shapely(self.shp.geometry)
        if buffer != 0:
            # bufferPoly = pygeos.buffer(poly, buffer)
            # print(len(poly))
            # dask_poly = dask.array.from_array(poly, chunks=max(int(len(poly) / 50), 1))
            # bufferPoly = dask_poly.map_blocks(pygeos.buffer, buffer, dtype=type(poly[0])).compute()
            ddf = dask_geopandas.from_geopandas(self.shp, npartitions=chunks)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                bufferPoly = ddf.buffer(buffer).compute()
        else:
            bufferPoly = self.shp.geometry

        if datatype == "dask":
            ar = dask.array.from_array(
                bufferPoly, chunks=int(len(bufferPoly) / chunks)
            ).reshape(len(bufferPoly), 1, 1)
            return ar
        elif datatype == "array":
            return bufferPoly
        else:
            raise NotImplementedError

    def plot_region(self, region, **kwargs):
        geo = self.shp.loc[self.regions == region].geometry
        return geo.boundary.plot(**kwargs)

    def sel(self, region_list, update=False):
        region_list = (
            [region_list] if not isinstance(region_list, list) else region_list
        )
        if update:
            shp = self
        else:
            shp = deepcopy(self)

        m = np.in1d(shp.regions, region_list)
        shp.shp = shp.shp[m]
        shp.regions = shp.regions[m]
        shp.shp = shp.shp.reset_index(drop=True)
        return shp


def from_path(path, regionid, region_list=None):
    shp = gpd.read_file(path)
    return GeoRegions(shp, regionid, region_list)


def from_name(name="usa", region_list=None):
    if name == "usa":
        return GeoRegions(open_usa_shp(), "state", region_list, name=name)
    elif name == "counties":
        return GeoRegions(open_counties_shp(), "fips", region_list, name=name)
    elif name == "global":
        return GeoRegions(open_global_shp(), "OBJECTID", region_list, name=name)
    else:
        raise NotImplementedError
