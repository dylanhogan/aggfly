import numpy as np
import pandas as pd
import xarray as xr
import zarr
import dask
import dask.array

from .grid import Grid
from .grid_utils import *

class Dataset:
    
    def __init__(self, da, name=None):
        self.da = da
        self.name = name
        self.coords = self.da.coords
        self.longitude = self.da.longitude
        self.latitude = self.da.latitude
        assert np.all([x in list(self.coords) for x in 
            ['latitude', 'longitude']])
        self.grid = Grid(self.longitude,
                                self.latitude)       
            
    def rechunk(self, chunks='auto'):
        # Rechunk data
        self.da = self.da.chunk(chunks)
            
    def clip_data_to_georegions_extent(self, georegions, split=False):
        self.grid.clip_grid_to_georegions_extent(georegions)
        with dask.config.set(**{'array.slicing.split_large_chunks': split}):
            self.da = self.da.sel(latitude = self.grid.latitude, 
                                  longitude = self.grid.longitude)
        self.coords = self.da.coords
        self.longitude = self.da.longitude
        self.latitude = self.da.latitude
        
    def update(self, array, drop_dims=None, new_dims=None, pos=0, dask_array=True, chunks=None):
        
        if type(array) == xr.core.dataarray.DataArray:
            # Coerce data into dask array if necessary
            if dask_array:
                if type(array.data) != dask.array.core.Array:
                    self.da = xr.DataArray(
                        data = dask.array.from_array(array.data),
                        dims = array.dims,
                        coords = array.coords)
                else:
                    self.da = array
            else:
                if type(array.data) != dask.array.core.Array:
                    self.da = array.compute()
                else:
                    self.da = array
                
        else:
            
            if drop_dims is not None:
                dargs = dict()
                for dd in drop_dims:
                    dargs[dd] = 0
                self.da = self.da.isel(**dargs)
                for dd in drop_dims:
                    self.da = self.da.drop(dd)
            
            if type(array) != dask.array.core.Array and dask_array:
                array = dask.array.from_array(array)
                
            if new_dims is None:
                self.da.data = array
            else:
                cdict = dict()
                for k in new_dims.keys():
                    cdict[k] = ([k], new_dims[k])
                for i in self.da.coords:
                    cdict[i] = ([i], self.da.coords[i].values)
            
                ndims = tuple()
                i=0
                for d in self.da.dims:
                    if i == pos:
                        ndims = ndims + tuple(new_dims.keys())
                    ndims = ndims + (d,)
                    i += 1
            
                self.da = xr.DataArray(
                    data=array,
                    dims=ndims,
                    coords=cdict)
                
            if chunks is not None:
                self.rechunk(chunks)
    
    def sel(self, **kwargs):
        da = self.da
        for k in kwargs.keys():
            d = {k:kwargs[k]}
            da = da.sel(d).expand_dims(k).transpose(*self.da.dims)
        self.update(da)
            
                
                
def from_path(path, var, engine, preprocess=None, name=None, chunks='auto', **kwargs):
    if "*" in path:
        # array = xr.open_mfdataset(path, engine=engine, chunks=chunks,
        #                           preprocess=preprocess, **kwargs)[var]
        with dask.config.set(**{'array.slicing.split_large_chunks': False}): 
            array = xr.open_mfdataset(path,
                           engine=engine, 
                           preprocess=preprocess,
                           parallel=True)[var]
    else:
        if engine == 'zarr':
            array = xr.open_zarr(path, chunks=chunks, **kwargs)[var]
        else:
            array = xr.open_dataset(path, engine=engine, **kwargs)[var]
        
        if preprocess is not None:
            array = preprocess(array)
    return Dataset(array, name)
    
def from_name(name, var, chunks='auto', **kwargs):
    path, engine, preprocess = get_path(name)
    clim = from_path(path, var, engine, preprocess, name, chunks, **kwargs)
    return clim

def get_path(name):
    if name == "era5l":
        return ("/home3/dth2133/data/annual/*.zarr", 'zarr', preprocess_era5l)
    elif name == "era5l_diag":
        return ("/home3/dth2133/data/ERA5", 'zarr', None)
    else:
        raise NotImplementedError
    
#  NEED TO DO THIS (I THINK)
# self.dda = dask.array.from_zarr(self.zarr_path, component=climvar, chunks=chunks)
# if self.climdata == "era5l":
#     self.array.coords['longitude'] = (self.array.coords['longitude'] + 180) % 360 - 180
#     self.array.transpose("latitude", "longitude", "time")
              
# lon_ind = np.where(np.in1d(self.xda.longitude.values, grid.longitude))[0]
# lat_ind = np.where(np.in1d(self.xda.latitude.values, grid.latitude))[0]
# self.dda = self.dda[lat_ind,:,:,:,:][:,lon_ind,:,:,:]
#         bounds = self.grid_weights.georegions.shp.total_bounds
#         lon = climate_data.data.longitude
#         lat = climate_data.data.latitude
        
#         inlon = np.logical_and(
#             lon >= bounds[0],
#             lon <= bounds[2])
#         inlon_b = [lon[inlon].min(), lon[inlon].max()]

#         inlat = np.logical_and(
#             lat >= bounds[1],
#             lat <= bounds[3])
#         inlat_b = [lat[inlat].min(), lat[inlat].max()]
        
#         self.longitude, self.latitude = grid_centroids(inlon_b, inlat_b, self.resolution)
#         self.centroids = reformat_grid(self.longitude, self.latitude, datatype='dask')[0]
   