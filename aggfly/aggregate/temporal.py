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

class TemporalAggregator:
    
    def __init__(self, calc, agg_from, agg_to, poly=1, ddargs=[20,30,0]):
        self.calc = calc
        self.agg_from = agg_from
        self.agg_to = agg_to
        self.temp = self.get_temp_array()
        self.poly = poly
        self.ddargs = np.array(ddargs)
        self.func = self.assign_func()
    
    def assign_func(self):
        if self.calc == 'avg':
            f = _avg
            self.args = (self.temp, self.poly,)
        if self.calc == 'sum':
            f = _sum
            self.args = (self.temp, self.poly,)
        elif self.calc == 'dd':
            f = _dd
            self.args = (self.temp, self.poly, self.ddargs)
        elif self.calc == 'time':
            f = _time
            self.args = (self.temp, self.poly, self.ddargs)
        return f

    def get_temp_array(self):
        from_ndims = get_time_dim(self.agg_from) + 1
        to_ndims = get_time_dim(self.agg_to) + 1
        assert from_ndims > to_ndims
        return np.zeros(tuple([1 for x in range(to_ndims)]))
    
    def execute(self, arr, **kwargs):
        return self.func(arr, *self.args, **kwargs)
    
    def map_execute(self, clim, **kwargs):
        time_list = ['year', 'month', 'day', 'hour']
        ind_to = time_list.index(self.agg_to)
        ind_from = time_list.index(self.agg_from)
        drop_dims = time_list[ind_to+1:ind_from+1]
        drop_axis = tuple([get_time_dim(x) for x in drop_dims])
        clim.update(clim.da.data.map_blocks(
                self.execute,
                dtype=float,
                drop_axis=drop_axis),
            drop_dims=drop_dims)
        return clim
            
               
@numba.njit(fastmath=True, parallel=True)
def _avg(frame, temp, poly):
    
    frame_shp = frame.shape
    res_shp = nb_contractor(frame, temp) 
    res_empty = np.zeros_like(np.empty(res_shp))
    res = nb_expander(res_empty)
    frame = nb_expander(frame)
    res_ndim = temp.ndim
    
    w = 0
    for y in prange(frame.shape[0]):
        for x in prange(frame.shape[1]):
            for a in prange(frame.shape[2]):
                for m in prange(frame.shape[3]):
                    for d in prange(frame.shape[4]):
                        for h in prange(frame.shape[5]):
                            i=(y,x,a,m,d,h)
                            # print(i)
                            w += 1
                            if res_ndim == 2:
                                ind = (i[0],i[1],0,0,0,0)
                            elif res_ndim == 3:
                                ind = (i[0],i[1],i[2],0,0,0) 
                            if res_ndim == 4:
                                ind = (i[0],i[1],i[2],i[3],0,0) 
                            elif res_ndim == 5:
                                ind = (i[0],i[1],i[2],i[3],i[4],0)
                            res[ind] += frame[i]
    for s in res.shape:
        w = w / s
    return np.power(res / w, poly).reshape(res_shp)
               
@numba.njit(fastmath=True, parallel=True)
def _sum(frame, temp, poly):
        
    frame_shp = frame.shape
    res_shp = nb_contractor(frame, temp) 
    res_empty = np.zeros_like(np.empty(res_shp))
    res = nb_expander(res_empty)
    frame = nb_expander(frame)
    res_ndim = temp.ndim
    
    for y in prange(frame.shape[0]):
        for x in prange(frame.shape[1]):
            for a in prange(frame.shape[2]):
                for m in prange(frame.shape[3]):
                    for d in prange(frame.shape[4]):
                        for h in prange(frame.shape[5]):
                            i=(y,x,a,m,d,h)
                            if res_ndim == 2:
                                ind = (i[0],i[1],0,0,0,0)
                            elif res_ndim == 3:
                                ind = (i[0],i[1],i[2],0,0,0) 
                            if res_ndim == 4:
                                ind = (i[0],i[1],i[2],i[3],0,0) 
                            elif res_ndim == 5:
                                ind = (i[0],i[1],i[2],i[3],i[4],0)
                            res[ind] += frame[i]
    return np.power(res, poly).reshape(res_shp)
               
@numba.njit(fastmath=True, parallel=True)
def _dd(frame, temp, poly, ddargs):
    
    frame_shp = frame.shape
    res_shp = nb_contractor(frame, temp) 
    res_empty = np.zeros_like(np.empty(res_shp))
    res = nb_expander(res_empty)
    frame = nb_expander(frame)
    res_ndim = temp.ndim
    
    w=0
    for y in prange(frame.shape[0]):
        for x in prange(frame.shape[1]):
            for a in prange(frame.shape[2]):
                for m in prange(frame.shape[3]):
                    for d in prange(frame.shape[4]):
                        for h in prange(frame.shape[5]):
                            i=(y,x,a,m,d,h)
                            w += 1
                            if res_ndim == 2:
                                ind = (i[0],i[1],0,0,0,0)
                            elif res_ndim == 3:
                                ind = (i[0],i[1],i[2],0,0,0) 
                            if res_ndim == 4:
                                ind = (i[0],i[1],i[2],i[3],0,0) 
                            elif res_ndim == 5:
                                ind = (i[0],i[1],i[2],i[3],i[4],0)
                            f = frame[i]
                            if f > ddargs[0] and f < ddargs[1]:
                                res[ind]+=np.absolute(f-ddargs[ddargs[2]])
    for s in res.shape:
        w = w / s
    out = res / w
    return np.power(out, poly).reshape(res_shp)
               
@numba.njit(fastmath=True, parallel=True)
def _time(frame, res, poly, ddargs):
    
    frame_shp = frame.shape
    res_shp = nb_contractor(frame, temp) 
    res_empty = np.zeros_like(np.empty(res_shp))
    res = nb_expander(res_empty)
    frame = nb_expander(frame)
    res_ndim = temp.ndim
    
    w=0
    for y in prange(frame.shape[0]):
        for x in prange(frame.shape[1]):
            for a in prange(frame.shape[2]):
                for m in prange(frame.shape[3]):
                    for d in prange(frame.shape[4]):
                        for h in prange(frame.shape[5]):
                            i=(y,x,a,m,d,h)
                            w += 1
                            if res_ndim == 2:
                                ind = (i[0],i[1],0,0,0,0)
                            elif res_ndim == 3:
                                ind = (i[0],i[1],i[2],0,0,0) 
                            if res_ndim == 4:
                                ind = (i[0],i[1],i[2],i[3],0,0) 
                            elif res_ndim == 5:
                                ind = (i[0],i[1],i[2],i[3],i[4],0)
                            f = frame[i]
                            if f > ddargs[0] and f < ddargs[1]:
                                res[ind] += 1
    for s in res.shape:
        w = w / s
    return np.power(res / w).reshape(res_shp)
        
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