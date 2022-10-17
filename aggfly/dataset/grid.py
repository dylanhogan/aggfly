import numpy as np
import pandas as pd
import xarray as xr
import pygeos
import geopandas as gpd
import os
import dask
import dask.array
from functools import lru_cache
import warnings

from .grid_utils import *

class Grid:
    
    def __init__(self, longitude, latitude):
        self.longitude = longitude
        self.latitude = latitude
        self.resolution = self.get_resolution()
        self.cell_area = self.get_cell_area()


    @lru_cache(maxsize=None)
    def centroids(self, datatype='dask', chunks=30):
        longitude, latitude = np.meshgrid(self.longitude, self.latitude)
        centroids = reformat_grid(longitude, latitude, datatype, chunks)
        return centroids
    
    def get_resolution(self):
        return abs(np.diff(self.latitude).mean())

    def get_cell_area(self):
        cell = pygeos.box(0, 0, self.resolution, self.resolution)
        return pygeos.area(cell)

    @lru_cache(maxsize=None)
    def clip_grid_to_georegions_extent(self, georegions):
        bounds = georegions.shp.total_bounds
        inlon = np.logical_and(
            self.longitude >= bounds[0] - self.resolution / 2,
            self.longitude <= bounds[2] + self.resolution / 2)
        inlon_b = [self.longitude[inlon].min(), self.longitude[inlon].max()]

        inlat = np.logical_and(
            self.latitude >= bounds[1] - self.resolution / 2,
            self.latitude <= bounds[3] + self.resolution / 2)
        inlat_b = [self.latitude[inlat].min(), self.latitude[inlat].max()]
        
        longitude, latitude = grid_centroids(inlon_b, inlat_b, self.resolution)
        
        self.longitude = self.longitude[inlon]
        self.latitude = self.latitude[inlat]
        
    @lru_cache(maxsize=None)
    def mask(self, georegions, buffer=0, chunksize=100, compute=True):
        
        mask = (georegions.poly_array(buffer, 'dask').rechunk(chunksize)
                .map_blocks(pygeos.contains, self.centroids(), dtype=float))
        if compute:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # m = np.moveaxis(mask.compute().squeeze(), -1, 0)
                m = mask.compute().squeeze()
                da = xr.DataArray(
                    data = m,
                    dims = ['region', 'latitude', 'longitude'],
                    coords=dict(
                        lon = (['longitude'], self.longitude.values),
                        lat = (['latitude'], self.latitude.values),
                        region = (['region'], georegions.regions.values)
                    )
                )
                return da
        else:
            return mask 
        
    def centroids_to_cell(self, georegions, buffer=0, chunksize=100, datatype='xarray', 
                          compute=True, intersect_cells=False): 
        
        # Generate dask array of geometries
        pol = (georegions.poly_array(buffer=buffer).rechunk((chunksize, -1, -1)))
        
        # Generate mask array contining T/F for centroids within (buffered) geometry
        msk = self.mask(georegions, buffer=buffer)
        X, Y = np.meshgrid(msk.lon, msk.lat)
        cmask = xr.DataArray(
            data = msk.data,
            dims = ['region', 'latitude', 'longitude'],
            coords = dict(
                lat = (['latitude', 'longitude'], Y),
                lon = (['latitude', 'longitude'], X),
                region = ('region', msk.region.values)
            )
        )
        
        #
        mlon = np.moveaxis(np.array(cmask.lon.where(cmask)), -1, 0)
        mlat = np.moveaxis(np.array(cmask.lat.where(cmask)), -1, 0)
        lonpoints = [dask.array.from_array(mlon + x, chunks=(chunksize, -1, -1)) 
                         for x in [-self.resolution/2, self.resolution/2]]
        latpoints = [dask.array.from_array(mlat + x, chunks=(chunksize, -1, -1))
                         for x in [-self.resolution/2, self.resolution/2]]
        boxes = lonpoints[0].map_blocks(pygeos.box, latpoints[0], 
                                        lonpoints[1], latpoints[1], dtype=float)
        
        # Return full cells or intersection of cell and polygon (e.g. for area weights)
        if not intersect_cells:
            
            tr = boxes.compute()
            # tr = np.moveaxis(tr, 0, -1)
            return tr
        
        result = pol.map_blocks(pygeos.intersection, boxes, dtype=float)
        if compute:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tr = result.compute()
                # tr = np.moveaxis(result.compute(), 0, -1)
            if datatype == 'xarray':
                return xr.DataArray(data=tr, dims=msk.dims, coords=msk.coords)
            elif datatype == 'dask':
                return dask.array.from_array(tr, chunks=chunksize)
            elif datatype == 'array':
                return tr    
            # elif datatype == 'dataframe':
            else:
                raise NotImplementedError
        else:
            return result