import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import numba
import zarr
import dask
import dask.array
from dataclasses import dataclass

from .grid import Grid
from .grid_utils import *
from ..regions import GeoRegions

class Dataset:
    
    def __init__(self,
                 da, 
                 xycoords=('longitude', 'latitude'),
                 time_sel=None,
                 preprocess=None,
                 clip_geom=None,
                 time_fix=False,
                 name=None):
        
        da = clean_dims(da, xycoords)
        if time_sel is not None:
            da = da.sortby('time').sel(time=time_sel)
            time_fix=True
        if preprocess is not None:
            da = preprocess(da)
            
        self.update(da, init=True)
        self.name = name
        self.coords = self.da.coords
        self.longitude = self.da.longitude
        self.latitude = self.da.latitude
        assert np.all([x in list(self.coords) for x in 
            ['latitude', 'longitude']])
        self.grid = Grid(self.longitude,
                                self.latitude)       
        self.history = []
        
        if clip_geom is not None:
            self.clip_data_to_georegions_extent(clip_geom)
        if time_fix:
            self.update(timefix(self.da), init=True)
            
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
        
    def update(self, array, drop_dims=None, new_dims=None, pos=0, dask_array=True, chunks=None, init=False):
        
        if not init:
            old_coords = self.da.coords
        
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
                    print('here')
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
        
        if not init:
            # Update history: Spatial collapse
            spatial_old = 'longitude' in old_coords and 'latitude' in old_coords
            spatial_new = 'longitude' in self.da.coords and 'latitude' in self.da.coords
            if spatial_old and not spatial_new:
                self.history.append('spatial')

            # Update history: Temporal collapse
            temp_dims = [x for x in old_coords if x in ['year','month','day','hour']]
            new_temp_dims = [x for x in self.da.coords if x in ['year','month','day','hour']]
            temp_dims_changed = not np.array_equiv(temp_dims, new_temp_dims)
            if temp_dims_changed:
                self.history.append('temporal')
        
    @lru_cache(maxsize=None)
    def interior_cells(self, georegions, buffer=None, dtype='georegions', maxsize=None):
        
        if buffer is None:
            buffer = self.grid.resolution
        
        # mask = self.grid.mask(georegions, buffer=buffer)
        cells = self.grid.centroids_to_cell(georegions, buffer=buffer)
        # if dtype == 'gpd':
        
        cells = xr.DataArray(
            data = cells,
            dims = ['region', 'latitude', 'longitude'],
            coords = dict(
                latitude = ('latitude', self.latitude.values),
                longitude = ('longitude', self.longitude.values),
                region = ('region', georegions.regions)
            )
        )
        
        if dtype == 'xarray':
            return cells
        elif dtype == 'gdf' or dtype == 'georegions':
            cells.name = 'geometry'
            out = cells.to_dataframe()
            df = out.loc[np.logical_not(out.geometry.isnull())]
            df = df.reset_index()
            if dtype == 'gdf':
                return gpd.GeoDataFrame(df)
            elif dtype == 'georegions':
                if maxsize is None:
                    count = df.groupby(['region']).cumcount()+1
                    df['cellid'] = [f'{df.region[i]}.{count[i]}' for i in range(len(df.region))]
                else:
                    subregion = df.groupby(['region']).cumcount()+1
                    subregion = np.int64(np.floor(subregion/maxsize)) + 1
                    df['subregion'] = [f'{df.region[i]}.{subregion[i]}' for i in range(len(df.region))]
                    count = df.groupby(['region', 'subregion']).cumcount()+1
                    df['cellid'] = [f'{df.region[i]}.{subregion[i]}.{count[i]}' for i in range(len(df.region))]
                    
                gdf = GeoRegions(gpd.GeoDataFrame(df), 'cellid')
                return GeoRegions(gpd.GeoDataFrame(df), 'cellid')
        else:
            return NotImplementedError
        
    def sel(self, **kwargs):
        da = self.da
        for k in kwargs.keys():
            d = {k:kwargs[k]}
            da = da.sel(d).expand_dims(k).transpose(*self.da.dims)
        self.update(da)
            
    def power(self, exp):
        arr = self.da.data.map_blocks(_power, exp) 
        self.update(arr)
        self.history.append(f'power{exp}')
        
    def interact(self, inter):
        
        if type(inter) == Dataset:
            inter = inter.da.data
        
        assert self.da.data.shape == inter.shape
        arr = self.da.data.map_blocks(_interact, inter) 
        self.update(arr)
        self.history.append('interacted')
        

            
@numba.njit(fastmath=True, parallel=True)               
def _power(array, exp):
    return np.power(array, exp)

@numba.njit(fastmath=True, parallel=True)               
def _interact(array, inter):
    return np.multiply(array, inter)



def from_path(path, var, engine, xycoords=('longitude', 'latitude'), time_sel=None, clip_geom=None,
              time_fix=False, preprocess=None, name=None, chunks='auto', **kwargs):
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
        
        # if time_sel is not None:
        #     array = array.sortby('time').sel(time=time_sel)
        # if preprocess is not None:
        #     array = preprocess(array)
    return Dataset(
        array,
        xycoords=xycoords,
        time_sel=time_sel,
        preprocess=preprocess,
        clip_geom=clip_geom,
        time_fix=time_fix,
        name=name)
    
def from_name(name, var, chunks='auto', **kwargs):
    # if name == 'prism':
    #         
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
   