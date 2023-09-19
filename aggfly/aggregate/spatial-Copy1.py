from functools import lru_cache, partial
from copy import deepcopy

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
            f = _avg
            self.args = (self.poly,)
        return f
    
    def execute(self, arr, weight, **kwargs):
        # return **kwargs
        return self.func(arr, weight, *self.args, **kwargs)
    
    def map_execute(self, clim, weights, update=False, **kwargs):

        if update == False:
            clim = deepcopy(clim)
        
        if type(clim) == Dataset:
            da = clim.da.data
            # da.rechunk(-1)
        else:
            da = clim
            # da.rechunk(-1)
    
        out = da.map_blocks(
                self.execute,
                weights.data,
                dtype=float,
                drop_axis=[0,1],
                new_axis=0,
                chunks=(weights.data.chunks[0:1]+da.chunks[2:]))
        
        clim.update(
            out,
            drop_dims=['latitude', 'longitude'],
            new_dims={'region': weights.region.values})
        return clim
    

@numba.njit(fastmath=True, parallel=True)
def _avg(frame, weight, poly):

    frame_shp = frame.shape
    frame = nb_expander(frame)
    res = np.zeros((weight.shape[0],) + frame.shape[2:], dtype=np.float64)
    wes = np.zeros((weight.shape[0],) + frame.shape[2:], dtype=np.float64)
    for r in prange(weight.shape[0]):
        # print(r)
        yl, xl = np.nonzero(weight[r,:,:])
        for a in prange(frame.shape[2]):
            for m in prange(frame.shape[3]):
                for d in prange(frame.shape[4]):
                    for h in prange(frame.shape[5]):
                        if len(yl) == 0:
                            res[r,a,m,d,h] = np.nan
                            wes[r,a,m,d,h] = np.nan
                        else:
                            for l in prange(len(yl)):
                                y = yl[l]
                                x = xl[l]
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
