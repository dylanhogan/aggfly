import numpy as np
import pandas as pd
import xarray as xr
import pygeos
import os
import dask
import dask.array
from functools import lru_cache

from ..utils import autochunk

def grid_centroids(lon_bound, lat_bound, res):
    longitude, latitude = np.meshgrid(
        np.arange(lon_bound[0], lon_bound[1], res),
        np.arange(lat_bound[0], lat_bound[1], res))
    return longitude, latitude

@lru_cache(maxsize=None)
def prism_grid_centroids(datatype='array', chunks=30):
    res = 0.041666666666
    lon_bound = [-125, -66.5]
    lat_bound = [24 + 1/12, 49.9375 - 1/48] 
    longitude, latitude = grid_centroids(lon_bound, lat_bound, res)
    centroids = reformat_grid(longitude, latitude, datatype, chunks)
    return centroids, np.unique(longitude), np.unique(latitude)
    
@lru_cache(maxsize=None)
def era5l_grid_centroids(datatype='array', chunks=30, usa=False):
    example = ( "/home3/dth2133/data/ERA5")
    clim = xr.open_zarr(example).t2m[:,:,0,0,0]
    # clim.coords['longitude'] = (clim.coords['longitude'] + 180) % 360 - 180
    # with dask.config.set(**{'array.slicing.split_large_chunks': True}):
    #     clim = clim.sortby(clim.longitude)
    if usa:
        clim = clim.sel(longitude = slice(-126, -67), latitude = slice(50, 24))
    if datatype=='coords':
        return clim.longitude.values, clim.latitude.values
    else:
        longitude, latitude = np.meshgrid(clim.longitude.values, clim.latitude.values)
        centroids = reformat_grid(longitude, latitude, datatype, chunks)
        return centroids

def reformat_grid(longitude, latitude, datatype='array', chunks=30):
    if datatype=='points':
        centroids = pygeos.points(longitude, y=latitude)
    elif datatype=='dask':
        if type(longitude) is not dask.array.core.Array:
            longitude, latitude = [dask.array.from_array(x, chunks='auto') 
                          for x in [longitude, latitude]]
        centroids = longitude.map_blocks(pygeos.points, y=latitude, dtype=float)
        centroids = centroids.rechunk(autochunk(centroids))
    elif datatype=='array':
        centroids = (longitude, latitude)
    elif datatype=='empty':
        return None
    else:
        raise NotImplementedError
    return centroids

def preprocess_era5l(array):
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):        
        # array = array.sortby(array.time)
        array.coords['longitude'] = (array.coords['longitude'] + 180) % 360 - 180
        array = array.sortby(array.longitude)
        # array['year'] = array.time.dt.year
        # array['month'] = array.time.dt.month
        # array['day'] = array.time.dt.day
        # array['hour'] = array.time.dt.hour
        # array = array.set_index(time=("year", "month", "day", "hour")).unstack('time')
        # array['t2m'] = array.t2m - 273.15
    return array

def timefix_era5l(array):
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):   
        array['year'] = array.time.dt.year
        array['month'] = array.time.dt.month
        array['day'] = array.time.dt.day
        array['hour'] = array.time.dt.hour
        array = array.set_index(time=("year", "month", "day", "hour")).unstack('time')
        array = array - 273.15
    return array