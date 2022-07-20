import os
import warnings
from functools import lru_cache

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import pygeos
import dask
import dask.array
import rasterio
from rasterio.enums import Resampling
import rioxarray

from . import crop_weights
from ..dataset import *

class GridWeights:
    
    def __init__(self, grid, georegions, raster_weights, ncpu=55):
        
        self.grid = grid  
        self.georegions = georegions      
        self.grid.clip_grid_to_georegions_extent(georegions)
        self.raster_weights = raster_weights
        self.rchunk = int(len(self.georegions.regions)/ncpu)
    
    @lru_cache(maxsize=None)
    def mask(self, buffer=0, compute=True):
        
        mask = (self.georegions.poly_array(buffer, 'dask').rechunk(self.rchunk)
                .map_blocks(pygeos.contains, self.grid.centroids(), dtype=float))
        if compute:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # m = np.moveaxis(mask.compute().squeeze(), -1, 0)
                m = mask.compute().squeeze()
                da = xr.DataArray(
                    data = m,
                    dims = ['region', 'latitude', 'longitude'],
                    coords=dict(
                        lon = (['longitude'], self.grid.longitude.values),
                        lat = (['latitude'], self.grid.latitude.values),
                        region = (['region'], self.georegions.regions.values)
                    )
                )
                return da
        else:
            return mask

    @lru_cache(maxsize=None)    
    def grid_geo_from_centroids(self, region, buffer=0, geo=True):
        gres = self.grid.resolution
        m = self.mask(buffer).sel(region=region)
        X, Y = np.meshgrid(self.grid.longitude, self.grid.latitude)
        x = np.where(m, X, np.nan)
        y = np.where(m, Y, np.nan)
        
        xmin, xmax = np.nanmin(x)-gres/2, np.nanmax(x)+gres/2
        ymin, ymax = np.nanmin(y)-gres/2, np.nanmax(y)+gres/2

        xspace = np.linspace(xmin, xmax, int(round((xmax - xmin)/gres,0))+1)
        yspace = np.linspace(ymin, ymax, int(round((ymax - ymin)/gres,0))+1)   

        gridlat = [[[xmin, y],[xmax, y]] for y in yspace]
        gridlon = [[[x, ymin],[x, ymax]] for x in xspace]
        gridshp = np.append(pygeos.linestrings(gridlat), 
                            pygeos.linestrings(gridlon), axis=0)
    
        if geo:
            return gpd.GeoSeries(gridshp)
        else:
            return gridshp
    
    @lru_cache(maxsize=None) 
    def interior_centroids_geo(self, region, buffer=0):
        m = self.mask(buffer).sel(region=region)
        return bool_array_to_geoseries(m)

    @lru_cache(maxsize=None) 
    def border_centroids(self, buffers=None, output='mask'):
        if buffers is None:
            buffers = (-self.grid.resolution, self.grid.resolution)
        m1 = self.mask(buffer = buffers[0])
        m2 = self.mask(buffer = buffers[1])
        
        mborder = np.logical_and(np.logical_not(m1), m2)
        if output=='geo':
            return bool_array_to_geoseries(mborder)
        elif output=='points':
            return bool_array_to_centroid_array(mborder)
        elif output=='mask':
            return mborder
        else:
            raise NotImplementedError
    
    def centroids_to_cell(self, msk, chunksize=100, datatype='xarray', 
                          compute=True, return_boxes=False): 
        pol = (self.georegions.poly_array(buffer=0)
                      .rechunk((self.rchunk, -1, -1)))
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
        mlon = np.moveaxis(np.array(cmask.lon.where(cmask)), -1, 0)
        mlat = np.moveaxis(np.array(cmask.lat.where(cmask)), -1, 0)
        lonpoints = [dask.array.from_array(mlon + x, chunks=(self.rchunk, -1, -1)) 
                         for x in [-self.grid.resolution/2, self.grid.resolution/2]]
        latpoints = [dask.array.from_array(mlat + x, chunks=(self.rchunk, -1, -1))
                         for x in [-self.grid.resolution/2, self.grid.resolution/2]]
        boxes = lonpoints[0].map_blocks(pygeos.box, latpoints[0], 
                                        lonpoints[1], latpoints[1], dtype=float)
        if return_boxes:
            tr=boxes.compute()
            tr = np.moveaxis(result.compute(), 0, -1)
            return xr.DataArray(data=tr, dims=msk.dims, coords=msk.coords)
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
            else:
                raise NotImplementedError
        else:
            return result
   
    @lru_cache(maxsize=None)
    def border_centroids_to_cell(self, compute=True):
        return self.centroids_to_cell(self.border_centroids())
    
    @lru_cache(maxsize=None)
    def calculate_area_weights(self):
        borders = self.border_centroids_to_cell(compute=False)
        b_area = ( borders.map_blocks(pygeos.area)
                  .compute()
                  .fillna(0) ) / self.grid.cell_area    
        cells = self.mask(buffer=-self.grid.resolution)*self.grid.cell_area
        weights = xr.DataArray(data=b_area, 
                               dims=cells.dims, 
                               coords=cells.coords) + cells
        return weights
        
    @lru_cache(maxsize=None)
    def calculate_weighted_area_weights(self):

        aw = self.calculate_area_weights()

        dsw = self.raster_weights.raster
        wts = np.multiply(aw, dsw).data
        da = xr.DataArray(
            data=wts,
            dims = ['region', 'latitude', 'longitude'],
            coords = dict(
                lat = (['latitude'], aw.lat.values),
                lon = (['longitude'], aw.lon.values),
                region = (['region'], aw.region.values)
            )   
        )
        return da    
    
    @lru_cache(maxsize=None)
    def weights(self, chunk=None):
        if chunk is None:
            c = max([int(len(self.georegions.regions)/55),1])
            chunk = (c, -1, -1)
        if self.raster_weights is None:
            return self.calculate_area_weights().chunk(chunk)
        else:
            return self.calculate_weighted_area_weights().chunk(chunk)
    
    def plot_weights(self, region, buffer=0, **kwargs):
        mask = self.mask(buffer).sel(region=region)
        w = self.raster_weights.raster.where(mask, drop=True)
        return xr.DataArray(w, 
            dims=['latitude','longitude'],
            coords=dict(
                longitude=('longitude', w.lon.values),
                latitude=('latitude', w.lat.values))).plot(**kwargs)
    
    def plot_grid(self, region, buffer=0, geo=True, **kwargs):
        gridshp = self.grid_geo_from_centroids(region, buffer, geo)
        return gridshp.plot(**kwargs)
    
    def plot_interior_centroids(self, region, buffer=0, **kwargs):
        gridshp = self.interior_centroids_geo(region, buffer)
        return gridshp.plot(**kwargs)
    
    def plot_border_areas(self, region, **kwargs):
        border_centroids = self.border_centroids_to_cell()
        return gpd.GeoSeries(areas).plot(**kwargs)

def from_objects(clim, georegions, crop='corn'):
    
    if crop is not None:
        raster_weights = crop_weights.from_name(
            name='cropland', crop=crop, grid=clim.grid)
    else:
        raster_weights = None
    
    return GridWeights(clim.grid, georegions, raster_weights)
    
    
    