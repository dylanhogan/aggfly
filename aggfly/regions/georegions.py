import numpy as np
import pandas as pd
import xarray as xr
import pygeos
import geopandas as gpd
import os
import dask
import dask.array
from functools import lru_cache
from copy import deepcopy

from .shp_utils import *

class GeoRegions:
    
    def __init__(self, shp=None, regionid='state', region_list=None):
        self.shp = shp
        self.regions = self.shp[regionid]
        if region_list is not None:
            self.sel(region_list, update=True)
        
    # @lru_cache(maxsize=None)
    def poly_array(self, buffer=0, datatype='dask', chunks=1):
        poly = pygeos.from_shapely(self.shp.geometry)
        if buffer != 0:
            # bufferPoly = pygeos.buffer(poly, buffer)
            dask_poly = dask.array.from_array(poly, chunks=int(len(poly) / 50))
            bufferPoly = dask_poly.map_blocks(pygeos.buffer, buffer, dtype=type(poly[0])).compute()
        else:
            bufferPoly = poly
            
        if datatype=='dask':
            ar = (dask.array
                    .from_array(
                        bufferPoly, 
                        chunks=int(len(bufferPoly) / chunks))
                    .reshape(len(bufferPoly), 1, 1))
            return ar
        elif datatype=='array':
            return bufferPoly
        else:
            raise NotImplementedError
            
    def plot_region(self, region, **kwargs):
        geo = self.shp.loc[self.regions==region].geometry
        return geo.boundary.plot(**kwargs)
    
    def sel(self, region_list, update=False):

        if update:
            shp = self
        else:
            shp = deepcopy(self)
            
        m = np.in1d(shp.regions, region_list)
        shp.shp = shp.shp[m]
        shp.regions = shp.regions[m] 
        
        return shp
    
def from_path(path, regionid, region_list=None):
    shp = gpd.read_file(path)
    return GeoRegions(shp, regionid, region_list)
                      
def from_name(name='usa', region_list=None):
    if name == 'usa':
        return GeoRegions(open_usa_shp(), 'state', region_list)
    elif name == 'counties':
        return GeoRegions(open_counties_shp(), 'fips', region_list)
    elif name == 'global':
        return GeoRegions(open_global_shp(), 'OBJECTID', region_list) 
    else:
        raise NotImplementedError
    