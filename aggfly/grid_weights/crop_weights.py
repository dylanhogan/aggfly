import os
import warnings
from functools import lru_cache
from hashlib import sha256
import json

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

from ..dataset import reformat_grid

class CropWeights:
    
    def __init__(
            self,
            raster,
            crop="corn"
        ):
        
        self.crop = crop
        self.raster = raster

    @lru_cache(maxsize=None)    
    def rescale_weights_to_grid(self, grid):
        
        print(f'Rescaling {self.crop} weights to grid.')
        print('This might take a few minutes and use a lot of memory...')
        
        lon = grid.longitude.values
        lat = grid.latitude.values
        template = xr.DataArray(
            data = np.zeros((len(lat),len(lon))),
            dims = ['latitude', 'longitude'],
            coords = dict(
                lon=(["longitude"], lon),
                lat=(["latitude"], lat)
            )
        )
        
        g = (xr.DataArray(
            data = grid.centroids().squeeze(),
            dims = ['y', 'x'],
            coords = dict(
                x = (['x'], lon),
                y = (['y'], lat)
            )
        ).rio.write_crs('WGS84'))
        
        weights = xr.apply_ufunc(np.single, 
                self.raster, dask='parallelized')
        
        dsw = weights.rio.reproject_match(g, nodata=0, resampling=Resampling.average)
        
        self.raster = xr.DataArray(data=dsw.values.squeeze(),
                            dims=template.dims, 
                            coords=template.coords)

def from_path(path, crop='corn', grid=None, write=False, name=None, feed=None):
    
    # Separate file path from file extension
    file, ex = os.path.splitext(path)
    
    # If grid supplied, check to see if a file exists with rescaled raster
    if grid is not None:
        
        gdict = {'grid':grid.resolution, 
                 'longitude':grid.longitude, 
                 'latitude':grid.latitude, 
                 'area': grid.cell_area, 
                 'crop':crop}
        
        if name is not None:
            gdict['name'] = name
            
        if feed is not None:
            gdict['feed'] = feed
            
        dump = json.dumps(str(gdict),sort_keys=True).encode('utf8')
        code = '_dscale-' + sha256(dump).hexdigest()[:15]
        if os.path.exists(file + code + '.zarr') and not write:
            file = file + code
            ex = '.zarr'
            grid = None
    
    da = open_raster(file+ex, crop)
    
    weights = CropWeights(da, crop)
    
    if grid is not None:
        print(f"Code: {code}")
        weights.rescale_weights_to_grid(grid)
        write=True
        write_da = (weights.raster
            .expand_dims('crop')
            .assign_coords(crop=(
                'crop',
                np.array(crop).reshape(1)))
            .to_dataset(name='layer'))
        file = file + code
    
    if write:
        p = file + '.zarr'
        print(f'Rescaled raster saved to {p}')
        write_da.to_zarr(p, mode='w')
        
    return weights     

def from_name(name='cropland', crop='corn', grid=None, feed='total', write=False):
    if name == 'cropland':
        path = "/home3/dth2133/data/cropland/2021_crop_mask.zarr"
        # preprocess = 
    elif name == 'GAEZ':
        path = f"/home3/dth2133/data/GAEZ/GAEZ_2015_all-crops_{feed}.nc"
    else:
        raise NotImplementedError
    return from_path(path, 
                     crop=crop, 
                     grid=grid, 
                     write=write, 
                     name=name, 
                     feed=feed)

def open_raster(path, crop, preprocess=None, **kwargs):
    
    # Separate file path from file extension
    file, ex = os.path.splitext(path)
    
    if ex == '.tif':
        da = rioxarray.open_rasterio(
            path,
            chunks=True,
            lock=False,  **kwargs)
        
        if preprocess is not None:
            da = preprocess(da, crop)
        else:
            da = format_cropland_tif_da(da, crop)
            
    elif ex =='.zarr':
        da = xr.open_zarr(path,  **kwargs)
        da = da.layer.sel(crop=crop)
    elif ex == '.nc':
        da = xr.open_dataset(path,  **kwargs)
        da = da.layer.sel(crop=crop)
    else:
        raise NotImplementedError
    
    return da

def format_cropland_tif_da(da, crop):
    return (da.isin([cropland_id(crop)])
            .drop('band')
            .squeeze()
            .expand_dims('crop')
            .assign_coords(crop=(
                'crop',
                np.array(self.crop_dict[num]).reshape(1)))
            .to_dataset(name='layer'))

def cropland_id(crop):
    crop_dict = {
        'corn':1,
        'cotton':2,
        'rice':3,
        'sorghum':4,
        'soybeans':5,
        'spring wheat':23
    }
    return crop_dict[crop]
