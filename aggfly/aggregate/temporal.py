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
            if isinstance(self.ddargs[0], list):
                f = _dd_multi
                self.args = (self.temp, self.poly, self.ddargs)
            else:
                f = _dd
                self.args = (self.temp, self.poly, self.ddargs) 
        elif self.calc == 'time':
            f = _time
            self.args = (self.temp, self.poly, self.ddargs)
        elif self.calc == 'sine':
            assert self.agg_from == 'hour'
            f = _sine_dd
            self.args = (self.temp, self.poly, self.ddargs)
        return f

    def get_temp_array(self):
        from_ndims = get_time_dim(self.agg_from) + 1
        to_ndims = get_time_dim(self.agg_to) + 1
        assert from_ndims > to_ndims
        return np.zeros(tuple([1 for x in range(to_ndims)]))
    
    def execute(self, arr, **kwargs):
        return self.func(arr, *self.args, **kwargs)
    
    def map_execute(self, clim, update=True, **kwargs):
        time_list = ['year', 'month', 'day', 'hour']
        ind_to = time_list.index(self.agg_to)
        ind_from = time_list.index(self.agg_from)
        drop_dims = time_list[ind_to+1:ind_from+1]
        drop_axis = tuple([get_time_dim(x) for x in drop_dims])
        if type(clim) == Dataset:
            da = clim.da.data
        else:
            da = clim
            update = False
        out = da.map_blocks(
                self.execute,
                dtype=float,
                drop_axis=drop_axis)
        if update:
            return clim.update(out, drop_dims=drop_dims)
        else:
            return out
            
               
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
                            elif res_ndim == 4:
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
                            elif res_ndim == 4:
                                ind = (i[0],i[1],i[2],i[3],0,0) 
                            elif res_ndim == 5:
                                ind = (i[0],i[1],i[2],i[3],i[4],0)
                            if int(frame[i]) != -9223372036854775808:
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
                            elif res_ndim == 4:
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
def _dd_multi(frame, temp, poly, ddargs):
    out = list()
    for i in prange(len(ddargs)):
        out.append(_dd(frame, temp, poly, ddargs[i]))
    return out
               
@numba.njit(fastmath=True, parallel=True)
def _time(frame, temp, poly, ddargs):
    
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
                            elif res_ndim == 4:
                                ind = (i[0],i[1],i[2],i[3],0,0) 
                            elif res_ndim == 5:
                                ind = (i[0],i[1],i[2],i[3],i[4],0)
                            f = frame[i]
                            if f > ddargs[0] and f < ddargs[1]:
                                res[ind]+=1
    for s in res.shape:
        w = w / s
    out = res / w
    return np.power(out, poly).reshape(res_shp)

@numba.njit(fastmath=True, parallel=True)
def _sine_dd(frame, temp, poly, ddargs):
    
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
                        i=(y,x,a,m,d)
                        # w += 1
                        if res_ndim == 2:
                            ind = (i[0],i[1],0,0,0,0)
                        elif res_ndim == 3:
                            ind = (i[0],i[1],i[2],0,0,0) 
                        elif res_ndim == 4:
                            ind = (i[0],i[1],i[2],i[3],0,0) 
                        elif res_ndim == 5:
                            ind = (i[0],i[1],i[2],i[3],i[4],0)
                        fmax = np.max(frame[y,x,a,m,d,:])
                        fmin = np.min(frame[y,x,a,m,d,:])
                        favg = (fmax + fmin) / 2
                        if fmax <= ddargs[0]:
                            res[ind]+=0
                        elif ddargs[0] < fmin:
                            res[ind]+= favg - ddargs[0]
                        elif fmin < ddargs[0] and ddargs[0] < fmax:
                            tempSave = np.arccos( (2 * ddargs[0] - fmax - fmin) / (fmax - fmin))
                            res[ind] += ( (favg - ddargs[0])*tempSave + (fmax - fmin) * np.sin(tempSave)/2) / np.pi
    return np.power(res, poly)

@numba.njit(fastmath=True, parallel=True)
def _sine_time(frame, temp, poly, ddargs):
    
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
                        i=(y,x,a,m,d)
                        # w += 1
                        if res_ndim == 2:
                            ind = (i[0],i[1],0,0,0,0)
                        elif res_ndim == 3:
                            ind = (i[0],i[1],i[2],0,0,0) 
                        elif res_ndim == 4:
                            ind = (i[0],i[1],i[2],i[3],0,0) 
                        elif res_ndim == 5:
                            ind = (i[0],i[1],i[2],i[3],i[4],0)
                        fmax = np.max(frame[y,x,a,m,d,:])
                        fmin = np.min(frame[y,x,a,m,d,:])
                        M = (fmax + fmin) / 2
                        W = (fmax - fmin) / 2
                        if fmax <= ddargs[0]:
                            res[ind]+=0
                        elif ddargs[0] < fmin:
                            res[ind]+= 1
                        elif fmin < ddargs[0] and ddargs[0] < fmax:
                            xmin = np.arcsin((ddargs[0] - M)/ W)
                            res[ind] = (np.pi - 2*xmin) / (2*np.pi)
                        res[ind] = res[ind] / (favg)
    return np.power(res, poly)

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