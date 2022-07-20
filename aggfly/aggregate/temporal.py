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

class YearlyWeatherAggregator:
    
    def __init__(self, calc='sum', poly=1):
        self.calc = calc
        self.poly = poly
        self.func = self.assign_func()
    
    def assign_func(self):
        if self.calc == 'avg':
            f = self._avg
        elif self.calc == 'sum':
            f = self._sum
        return f
    
    def execute(self, arr, **kwargs):
        # print(f"Collapsing days to {self.calc}...")
        if isinstance(arr, dask.array.Array):
            return self.func(arr.values, self.poly, **kwargs)
        elif isinstance(arr, xr.core.dataarray.DataArray):
            return self.func(arr.values, self.poly, **kwargs)
        else:
            return self.func(arr, self.poly, **kwargs)
    
    @staticmethod
    @numba.njit(fastmath=True, parallel=True)
    def _avg(frame, poly):
        res=np.zeros(frame.shape[0:3], dtype=np.float64)
        for y in prange(frame.shape[0]):
            for x in prange(frame.shape[1]):
                for a in prange(frame.shape[2]):
                    for d in prange(frame.shape[3]):
                            res[y,x,a]+=frame[y,x,a,d]
        out = res/(frame.shape[3])
        return np.power(out, poly)

    @staticmethod
    @numba.njit(fastmath=True, parallel=True)
    def _sum(frame, poly):
        res=np.zeros(frame.shape[0:3], dtype=np.float64)
        for y in prange(frame.shape[0]):
            for x in prange(frame.shape[1]):
                for a in prange(frame.shape[2]):
                    for d in prange(frame.shape[3]):
                            res[y,x,a]+=frame[y,x,a,d]
        return np.power(res, poly)


class DailyWeatherAggregator:
    
    def __init__(self, calc='avg', poly=1, ddargs=[20,30,0]):
        self.calc = calc
        self.poly = poly
        self.ddargs = np.array(ddargs)
        self.func = self.assign_func()
    
    def assign_func(self):
        if self.calc == 'avg':
            f = self._avg
            self.args = (self.poly,)
        elif self.calc == 'dd':
            f = self._dd
            self.args = (self.poly, self.ddargs)
        return f

    def execute(self, arr, **kwargs):
        # print(f"Collapsing hours to {self.calc}...")
        if isinstance(arr, dask.array.Array):
            return self.func(arr.values, *self.args, **kwargs)
        elif isinstance(arr, xr.core.dataarray.DataArray):
            return self.func(arr.values, *self.args, **kwargs)
        else:
            return self.func(arr, *self.args, **kwargs)
    
    @staticmethod
    @numba.njit(fastmath=True, parallel=True)
    def _avg(frame, poly):
        res=np.zeros(frame.shape[0:4], dtype=np.float64)
        for y in prange(frame.shape[0]):
            for x in prange(frame.shape[1]):
                for a in prange(frame.shape[2]):
                    for d in prange(frame.shape[3]):
                        for h in prange(frame.shape[4]):
                            res[y,x,a,d]+=frame[y,x,a,d,h]
        out = res/frame.shape[4]
        return np.power(out, poly)
    
    @staticmethod
    @numba.njit(fastmath=True, parallel=True)
    def _dd(frame, poly, ddargs):
        res=np.zeros(frame.shape[0:4], dtype=np.float64)
        for y in prange(frame.shape[0]):
            for x in prange(frame.shape[1]):
                for a in prange(frame.shape[2]):
                    for d in prange(frame.shape[3]):
                        for h in prange(frame.shape[4]):
                            f = frame[y,x,a,d,h]
                            if f > ddargs[0] and f < ddargs[1]:
                                res[y,x,a,d]+=np.absolute(f-ddargs[ddargs[2]])/24
        return np.power(res, poly)


class TemporalAggregator:
    
    def __init__(self, 
                 daily = DailyWeatherAggregator(),
                 yearly = YearlyWeatherAggregator()):
        self.daily = daily
        self.yearly = yearly
    
    def execute(self, arr):
        # print("Processing temporal aggregation!")
        if self.daily is not None:
            res = self.yearly.execute(
                self.daily.execute(arr))
        else:
            res = self.yearly.execute(arr)
            
        if isinstance(arr, dask.array.Array):
            return dask.array.from_array(res)
        elif isinstance(arr, xr.core.dataarray.DataArray):
            return xr.DataArray(res)
        else:
            return res
        
def from_name(name='era5l',
              calc={'daily':('dd', 1, [30,999,0]),
                    'yearly':('sum', 1)}):
    daily = calc['daily']
    yearly = calc['yearly']
    if name == 'era5l':
        return TemporalAggregator(
            DailyWeatherAggregator(*daily),
            YearlyWeatherAggregator(*yearly))
    else:
        raise NotImplementedError