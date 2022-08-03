from functools import lru_cache, partial

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import pygeos
import dask
import dask.array
import rasterio
import rioxarray
from rasterio.enums import Resampling
import numba
from numba import prange
import zarr

from .aggregate_utils import *
from ..dataset.dataset import Dataset

class SpatialAggregator:
    
    def __init__(self, calc, poly=1):
        self.calc = calc
        # self.agg_from = agg_from
        self.poly = poly
        self.func = self.assign_func()
    
    def assign_func(self):
        if self.calc == 'avg':
            f = self._avg
            self.args = (self.poly,)
        return f
    
    def execute(self, arr, weight, **kwargs):
        return self.func(arr, weight, *self.args, **kwargs)
    
    def map_execute(self, clim, weights, update=True, **kwargs):

        if type(clim) == Dataset:
            da = clim.da.data
        else:
            da = clim
            update = False
            
        out = da.map_blocks(
                self.execute,
                weights.data,
                dtype=float,
                drop_axis=[0,1],
                new_axis=0,
                chunks=(weights.data.shape[0:1]+da.shape[2:]))
        if update:
            return clim.update(
                out,
                drop_dims=['latitude', 'longitude'],
                new_dims={'region': weights.region.values})
        else:
            return out
    @staticmethod
    @numba.njit(fastmath=True, parallel=True)
    def _avg(frame, weight, poly):

        frame_shp = frame.shape
        frame = nb_expander(frame)
        res = np.zeros((weight.shape[0],) + frame.shape[2:], dtype=np.float64)
        wes = np.zeros((weight.shape[0],) + frame.shape[2:], dtype=np.float64)
        for r in prange(weight.shape[0]):
            for a in prange(frame.shape[2]):
                for m in prange(frame.shape[3]):
                    for d in prange(frame.shape[4]):
                        for h in prange(frame.shape[5]):
                            for y in prange(frame.shape[0]):
                                for x in prange(frame.shape[1]):
                                    # I can't believe this was actually the solution
                                    # https://github.com/numba/numba/issues/2919
                                    if int(frame[y,x,a,m,d,h]) != -9223372036854775808:
                                        f = frame[y,x,a,m,d,h]
                                        w = weight[r,y,x]
                                        res[r,a,m,d,h] += f * w
                                        wes[r,a,m,d,h] += w
        out = res/wes
        return out.reshape((weight.shape[0],) + frame_shp[2:])

def from_name(name='era5l',
              calc=('avg', 1)):
    if name == 'era5l':
        return SpatialAggregator(calc=calc[0], poly=calc[1])
    else:
        raise NotImplementedError
