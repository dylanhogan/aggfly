from functools import lru_cache, partial
from copy import deepcopy
import datetime

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
    
    def __init__(self, calc, groupby, poly=1, ddargs=[20,30]):
        self.calc = calc
        self.groupby = groupby
        # self.agg_to = agg_to
        # self.temp = self.get_temp_array()
        self.poly = poly
        self.ddargs = np.array(ddargs)
        self.func = self.assign_func()
    
    def assign_func(self):
        if self.calc == 'avg':
            f = _avg
            self.args = (self.poly,)
            self.input_core_dims = (['time'],['y'],['x'])
        if self.calc == 'sum':
            f = _sum
            self.args = (self.poly,)
            self.input_core_dims = (['time'],['y'],['x'])
        elif self.calc == 'dd':
            if isinstance(self.ddargs[0], list):
                f = _dd_multi
                self.args = (self.poly, *self.ddargs)
            else:
                f = _dd
                self.args = (self.poly, *self.ddargs)
                self.input_core_dims = (['time'],['y'],['x'])
        elif self.calc == 'time':
            f = _time
            self.args = (self.poly, *self.ddargs)
            self.input_core_dims = (['time'],['y'],['x'])
        elif self.calc == 'sine-dd':
            f = _sine_dd
            self.args = (self.poly, *self.ddargs)
            self.input_core_dims = (['time'],['y'],['x'])
        elif self.calc == 'sine-time':
            f = _sine_time
            self.args = (self.poly, *self.ddargs)
            self.input_core_dims = (['time'],['y'],['x'])
        if self.calc == 'min':
            f = _min
            self.args = (self.poly,)
            self.input_core_dims = (['time'],['y'],['x'])
        if self.calc == 'max':
            f = _max
            self.args = (self.poly,)
            self.input_core_dims = (['time'],['y'],['x'])
        return f

    def get_temp_array(self):
        from_ndims = get_time_dim(self.agg_from) + 1
        to_ndims = get_time_dim(self.agg_to) + 1
        assert from_ndims > to_ndims
        return np.zeros(tuple([1 for x in range(to_ndims)]))
    
    def execute(self, arr,  y_ind, x_ind, **kwargs):
        return self.func(arr,  y_ind, x_ind, *self.args, **kwargs)
    
    # def groupby_execute(self, clim, nzw_ind, update= **kwargs):
    
    def map_execute(self, clim, nzw_ind, update=False, **kwargs):
        
        # Update the object or return a copy.
        if update == False:
            clim = deepcopy(clim)
        
        y_ind = xr.DataArray(
            data = nzw_ind[0],
            dims = ['y']
        )
        x_ind = xr.DataArray(
            data = nzw_ind[1],
            dims = ['x']
        )
        
        out = xr.apply_ufunc(
            self.execute, clim.da.groupby(self.groupby), y_ind, x_ind,
            dask='parallelized',
            input_core_dims=self.input_core_dims,
            exclude_dims={'time'},
            output_dtypes=float,
            dask_gufunc_kwargs=dict(allow_rechunk=True)
        )
        out = out.rename({out.dims[-1]:'time'})
        if type(out.time.values[0]) == datetime.date:
            out['time'] = [np.datetime64(x, 'ns') for x in out.time.values]
        
        # return out
        # Update object and return result
        if type(clim) == Dataset:
            clim.update(out)
            return clim
        else:
            return out

    def map_blocks_execute(self, da, nzw_ind, collapsed, drop_axis):
        # If collapsed, need to add a fake 1-dimension for the loops
        # in the numba functions
        if collapsed:
            da = da[None,...]
        
        # Let dask do its thing
        out = da.map_blocks(
                self.execute,
                nzw_ind,
                dtype=float,
                drop_axis=drop_axis)
        
        # Get rid of that pesky fake dimension.
        if collapsed:
            out = out[0,...]
        
        return out
    
    def groupby_execute(self, xda, groupby, clim, nzw_ind, collapsed, drop_axis, drop_dims):
        da = xda.unstack('grp').transpose('latitude', 'longitude', *groupby, ...).data
        clim = deepcopy(clim)
        # out = self.map_blocks_execute(da, nzw_ind, collapsed, drop_axis)
        out = self.execute(da.compute(), nzw_ind)
        clim.update(out, drop_dims=drop_dims)
        return clim.da
        
@numba.njit(fastmath=True, parallel=True)
def _avg(frame, y_ind, x_ind, poly):
    
    frame_shp = frame.shape
    res = np.zeros_like(np.empty((frame_shp[0], frame_shp[1])))
    w = res.copy()
    yl, xl = y_ind, x_ind
    # for y in prange(frame.shape[0]):
    #     for x in prange(frame.shape[1]):
    for l in prange(len(yl)):
        y = yl[l]
        x = xl[l]
        for t in range(frame.shape[2]):
            i=(y,x,t)
            if int(frame[i]) != -9223372036854775808:
                res[y,x] += frame[i]
                w[y,x] += 1
    return np.power(res/w, poly)

@numba.njit(fastmath=True, parallel=True)
def _max(frame, y_ind, x_ind, poly):
    
    frame_shp = frame.shape
    res = np.zeros_like(np.empty((frame_shp[0], frame_shp[1])))
    # w = res.copy()
    yl, xl = y_ind, x_ind
    for l in prange(len(yl)):
        y = yl[l]
        x = xl[l]
        res[y,x] += np.nanmax(frame[y,x,:])
    return np.power(res, poly)

@numba.njit(fastmath=True, parallel=True)
def _min(frame, y_ind, x_ind, poly):
    
    frame_shp = frame.shape
    res = np.zeros_like(np.empty((frame_shp[0], frame_shp[1])))
    # w = res.copy()
    yl, xl = y_ind, x_ind
    for l in prange(len(yl)):
        y = yl[l]
        x = xl[l]
        res[y,x] += np.nanmin(frame[y,x,:])
    return np.power(res, poly)

@numba.njit(fastmath=True, parallel=True)
def _sum_dep(frame, temp, poly):
        
    frame_shp = frame.shape
    res_shp = nb_contractor(frame, temp) 
    res_empty = np.zeros_like(np.empty(res_shp))
    res = nb_expander(res_empty)
    # isna_dim = np.ones_like(res)
    frame = nb_expander(frame)
    res_ndim = temp.ndim
    isna_dim = frame.shape[2]*frame.shape[3]*frame.shape[4]*frame.shape[5]
    
    for y in prange(frame.shape[0]):
        for x in prange(frame.shape[1]):
            isna = 0
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
                            else:
                                isna += 1
            if isna == isna_dim:
                res[y,x,:,:,:,:] = np.nan
    return np.power(res, poly).reshape(res_shp)

@numba.njit(fastmath=True, parallel=True)
def _sum(frame, y_ind, x_ind, poly):
    
    frame_shp = frame.shape
    res = np.zeros_like(np.empty((frame_shp[0], frame_shp[1])))
    w = res.copy()
    yl, xl = y_ind, x_ind

    nv = -9223372036854775808
    isna_dim = frame.shape[2]

    for l in prange(len(yl)):
        isna = 0
        y = yl[l]
        x = xl[l]
        w[y,x] = 1
        for t in range(frame.shape[2]):
            i=(y,x,t)
            f = frame[i]
            if int(f) != nv:
                res[y,x] += f
            else:
                isna += 1
        if isna == isna_dim:
            res[y,x] = np.nan
    return np.power(res/w, poly)

@numba.njit(fastmath=True, parallel=True)
def _dd(frame, y_ind, x_ind, poly, dd_low, dd_high, dd_by):
    
    ddargs=[dd_low,dd_high,dd_by]
    frame_shp = frame.shape
    res = np.zeros_like(np.empty((frame_shp[0], frame_shp[1])))
    w = res.copy()
    yl, xl = y_ind, x_ind

    nv = -9223372036854775808
    isna_dim = frame.shape[2]

    for l in prange(len(yl)):
        isna = 0
        y = yl[l]
        x = xl[l]
        for t in range(frame.shape[2]):
            i=(y,x,t)
            f = frame[i]
            if int(f) != nv:
                if f > ddargs[0] and f < ddargs[1]:
                    res[y,x]+=np.absolute(f-ddargs[ddargs[2]])
                w[y,x] += 1
    return np.power(res/w, poly)


@numba.njit(fastmath=True, parallel=True)
def _sine_dd(frame, y_ind, x_ind, poly, dd_low, dd_high, dd_by):
    
    ddargs=[dd_low,dd_high,dd_by]
    frame_shp = frame.shape
    res = np.zeros_like(np.empty((frame_shp[0], frame_shp[1])))
    w = res.copy()
    yl, xl = y_ind, x_ind

    nv = -9223372036854775808
    isna_dim = frame.shape[2]
    
    for l in prange(len(yl)):
        isna = 0
        y = yl[l]
        x = xl[l]
        
        fmax = np.max(frame[y,x,:])
        fmin = np.min(frame[y,x,:])
        favg = (fmax + fmin) / 2
        if fmax <= ddargs[0]:
            res[y,x]+=0
        elif ddargs[0] < fmin:
            res[y,x]+= favg - ddargs[0]
        elif fmin < ddargs[0] and ddargs[0] < fmax:
            tempSave = np.arccos( (2 * ddargs[0] - fmax - fmin) / (fmax - fmin))
            res[y,x] += ( (favg - ddargs[0])*tempSave + (fmax - fmin) * np.sin(tempSave)/2) / np.pi
    return np.power(res, poly)

@numba.njit(fastmath=True, parallel=True)
def _sine_time(frame, y_ind, x_ind, poly, dd_low, dd_high, dd_by):
    
    ddargs=[dd_low,dd_high,dd_by]
    frame_shp = frame.shape
    res = np.zeros_like(np.empty((frame_shp[0], frame_shp[1])))
    w = res.copy()
    yl, xl = y_ind, x_ind
    
    for l in prange(len(yl)):
        isna = 0
        y = yl[l]
        x = xl[l]
        
        fmax = np.max(frame[y,x,:])
        fmin = np.min(frame[y,x,:])
        M = (fmax + fmin) / 2
        W = (fmax - fmin) / 2
        if fmax <= ddargs[0]:
            res[y,x]+=0
        elif ddargs[0] < fmin:
            res[y,x]+= 1
        elif fmin < ddargs[0] and ddargs[0] < fmax:
            xmin = np.arcsin((ddargs[0] - M)/ W)
            res[y,x] = (np.pi - 2*xmin) / (2*np.pi)
        # res[ind] = res[ind] / (M)
    return np.power(res, poly)

@numba.njit(fastmath=True, parallel=True)
def _dd_dep(frame, temp, poly, ddargs):
    
    frame_shp = frame.shape
    res_shp = nb_contractor(frame, temp) 
    res_empty = np.zeros_like(np.empty(res_shp))
    res = nb_expander(res_empty)
    w = res.copy()
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
                            f = frame[i]
                            if int(f) != -9223372036854775808:
                                if f > ddargs[0] and f < ddargs[1]:
                                    res[y,x]+=np.absolute(f-ddargs[ddargs[2]])
                                w[y,x] += 1
    return np.power(res/w, poly).reshape(res_shp)


               
@numba.njit(fastmath=True, parallel=True)
def _time(frame, y_ind, x_ind, poly, dd_low, dd_high):
    
    frame_shp = frame.shape
    res = np.zeros_like(np.empty((frame_shp[0], frame_shp[1])))
    # w = res.copy()
    yl, xl = y_ind, x_ind

    nv = -9223372036854775808
    isna_dim = frame.shape[2]

    for l in prange(len(yl)):
        isna = 0
        y = yl[l]
        x = xl[l]
        for t in range(frame.shape[2]):
            i=(y,x,t)
            f = frame[i]
            if int(f) != nv:
                if f > dd_low and f < dd_high:
                    res[y,x]+=1
                # w[y,x] +=1
            else:
                isna += 1
            if isna == isna_dim:
                res[y,x] = np.nan
    return np.power(res, poly)


@numba.njit(fastmath=True, parallel=True)
def _sine_dd_dep(frame, temp, poly, ddargs):
    
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
def _sine_time_dep(frame, temp, poly, ddargs):
    
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
                        # res[ind] = res[ind] / (M)
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