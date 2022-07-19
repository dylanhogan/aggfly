import zarr
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
from dataclasses import dataclass

class SpatialAggregator:
    
    def __init__(self, calc='avg', poly=1):
        self.calc = calc
        self.poly = poly
        self.func = self.assign_func()
    
    def assign_func(self):
        if self.calc == 'avg':
            f = self._avg
        return f
    
    def execute(self, weight, arr, **kwargs):
        if isinstance(arr, dask.array.Array):
            return self.func(weight, arr.values, self.poly, **kwargs)
        elif isinstance(arr, xr.core.dataarray.DataArray):
            return self.func(weight, arr.values,  self.poly, **kwargs)
        else:
            return self.func(weight, arr, self.poly, **kwargs) 
 
    @staticmethod
    @numba.njit(fastmath=True, parallel=True)
    def _avg(weight, frame, poly):
        res=np.zeros((weight.shape[0], frame.shape[2]), dtype=np.float64)
        wes = np.zeros((weight.shape[0], frame.shape[2]), dtype=np.float64)
        t=np.zeros((weight.shape[0], frame.shape[2]), dtype=np.float64)
        for r in prange(weight.shape[0]):
            for a in prange(frame.shape[2]):
                for y in prange(frame.shape[0]):
                    for x in prange(frame.shape[1]):
                        # I can't believe this was actually the solution
                        # https://github.com/numba/numba/issues/2919
                        if int(frame[y,x,a]) != -9223372036854775808:
                            t[r, a] += frame[y,x,a]
                            res[r,a] += frame[y,x,a] * weight[r,y,x]
                            wes[r,a] += weight[r,y,x]

        # s = res.shape
        # return res.reshape((s[0],s[1],1))
        # return res
        out = res/wes
        return t
        # return np.power(out, poly)

def from_name(name='era5l',
              calc=('avg', 1)):
    if name == 'era5l':
        return SpatialAggregator(calc=calc[0], poly=calc[1])
    else:
        raise NotImplementedError
        
    # @staticmethod
    # @numba.njit(fastmath=True, parallel=True)
    # def _avg(frame, weight, poly):
    #     res=np.zeros((frame.shape[0], frame.shape[3]), dtype=np.float64)
    #     wes = np.zeros((frame.shape[0], frame.shape[3]), dtype=np.float64)
    #     for r in prange(frame.shape[0]):
    #         for y in prange(frame.shape[1]):
    #             for x in prange(frame.shape[2]):
    #                 for a in prange(frame.shape[3]):
    #                     res[r, a]+=frame[r,y,x,a] * weight[r,y,x]
    #                     wes[r,a] += weight[r,y,x]
    #     if wes == 0:
    #         out = res
    #     else:
    #         out = res/wes
    #     s = res.shape
    #     return np.power(out, poly)


# @numba.guvectorize(
#     [(numba.float64[:,:,:,:,:], numba.float64[:, :], numba.int64, numba.float64[:,:,:,:,:])],
#     '(y,x,a,d,h), (y,x), () -> (y,x,a,d,h)'
# )
# def _guavg(arr, weight, poly, out):
#     out[:,:,:,:,:] = _avg(arr, weight, poly)
