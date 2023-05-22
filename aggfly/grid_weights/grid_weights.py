import os
import warnings
from functools import lru_cache

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
# import pygeos
import dask
import dask.array
import dask_geopandas
import rasterio
from rasterio.enums import Resampling
import rioxarray
from pprint import pformat, pprint
from copy import deepcopy

from . import crop_weights
from . import pop_weights
from ..dataset import *
from ..utils import *
from ..cache import *

class GridWeights:
    
    def __init__(
            self, 
            grid,
            georegions,
            raster_weights,
            chunks=30,
            project_dir=None,
            simplify=None,
            default_to_area_weights=True,
            verbose=True
        ):
        
        self.grid = grid  
        self.georegions = georegions      
        self.grid.clip_grid_to_georegions_extent(georegions)
        self.raster_weights = raster_weights
        self.chunks = chunks
        self.project_dir = project_dir
        self.simplify = simplify
        self.default_to_area_weights = default_to_area_weights
        self.verbose=True
        
        self.cache = initialize_cache(self)
    
    @lru_cache(maxsize=None)
    def simplify_poly_array(self):
        georegions = deepcopy(self.georegions)
        simplified = dask_geopandas.from_geopandas(
            georegions.shp, 
            npartitions=30).simplify(self.simplify).compute()
        georegions.shp['geometry'] = simplified
        return georegions
    
    @lru_cache(maxsize=None)
    def mask(self, buffer=0):
            
        centroids = self.grid.centroids()
        fc = gpd.GeoDataFrame(geometry=centroids.flatten()).set_crs('EPSG:4326')
        
        # fc = fc.rechunk(chunks)[...,None]
        fc = dask_geopandas.from_geopandas(fc, npartitions=self.chunks)
        
        if self.simplify is not None:
            georegions = self.simplify_poly_array()
        else:
            georegions = self.georegions
        
        poly_array = georegions.poly_array(buffer, chunks=self.chunks).squeeze()
            
        poly_shp = poly_array.shape
        poly_array = dask_geopandas.from_geopandas(
            gpd.GeoDataFrame(geometry=poly_array),
            npartitions=1)
        
        # mask = fc.map_blocks(pygeos.within, poly_array, dtype=bool)
        mask = fc.sjoin(poly_array, predicate='within').compute()
        
        return mask
    
    
    def get_border_cells(self, buffers=None):
        
        print('Searching for border cells...')
        if buffers is None:
            buffers = (-self.grid.resolution, self.grid.resolution)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            print('Negative buffer')
            m1 = self.mask(buffer = buffers[0])
            print('Positive buffer')
            m2 = self.mask(buffer = buffers[1])
        
        m2['cell_id'] = m2.index.values
        m = m2.merge(m1, how='outer', indicator=True)
        border = m.loc[m._merge=='left_only'].drop(columns='_merge')
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            print('Generating cells...')
            cells = border.buffer(self.grid.resolution/2, cap_style=3)
        border['geometry'] = cells
        
        return border

    def intersect_border_cells(self):
        
        border = self.get_border_cells()
        
        reg = border[['index_right']].merge(
            self.georegions.shp,
            how='left',
            left_on='index_right',
            right_index=True
        )
        reg = dask_geopandas.from_geopandas(gpd.GeoDataFrame(reg, geometry='geometry'), npartitions=1)
        dgb = dask_geopandas.from_geopandas(gpd.GeoDataFrame(border), npartitions=self.chunks)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            print('Intersecting...')
            inter = dgb.geometry.intersection(reg.geometry).compute()
            print('Calculating area weight')
            inter = inter.area / self.grid.cell_area
            
        border['area_weight'] = inter
        
        return border
    
    def get_cell_id_dataframe(self):
        return pd.DataFrame({
            'cell_id':self.grid.index.flatten(),
            'longitude':self.grid.lon_array.flatten(),
            'latitude':self.grid.lat_array.flatten()
        })
    
    def get_area_weights(self):
        
        border_cells = self.intersect_border_cells()
        
        interior_cells = self.mask(buffer = -self.grid.resolution)
        interior_cells = interior_cells.reset_index().rename(columns={'index':'cell_id'})
        interior_cells['area_weight'] = 1
        
        area_weights = pd.concat(
            [interior_cells.drop(columns='geometry'),
             border_cells.drop(columns='geometry')],
            axis=0)
        
        area_weights = area_weights.loc[area_weights.area_weight > 0]
        
        cell_df = self.get_cell_id_dataframe()
        area_weights = area_weights.merge(cell_df, how='left', on='cell_id')
        
        return area_weights
    
    def get_weighted_area_weights(self):
        
        area_weights = self.get_area_weights()
            
        raster_weights = self.raster_weights.raster.to_dataframe(name='raster_weight')
        w = area_weights.merge(
            raster_weights,
            how='left',
            left_on=['latitude', 'longitude'],
            right_index=True
        )
        
        # Parallelize
        dw = dask.dataframe.from_pandas(w, npartitions=self.chunks)
        
        # Check for weight totals
        raster_total = (dw[['index_right', 'raster_weight']]
            .groupby('index_right')
            .sum()
            .rename(columns={'raster_weight':'total_weight'})
        )
        raster_total['zero_weight'] = raster_total.total_weight == 0
        
        # Merge raster totals
        tw = dw.merge(raster_total, how='left', left_on='index_right', right_index=True).compute()
        
        # Rescale raster weights
        tw.loc[np.logical_not(tw.zero_weight), ['weight']] = tw.area_weight * (tw.raster_weight / tw.total_weight)
        
        # Default to area weights for places with zero raster weight if indicated
        if self.default_to_area_weights:
            tw.loc[tw.zero_weight, ['weight']] = tw.area_weight
        else:
            tw = tw.loc[np.logical_not(tw.zero_weight)]
            
        return tw

    # @lru_cache(maxsize=None)
    def weights(self):
        
        gdict = {'func':'weights'} 
        
        # Load raster weights if needed
        if self.raster_weights is not None:
            self.raster_weights.rescale_raster_to_grid(self.grid, verbose=self.verbose)
            gdict['raster_weights'] = self.raster_weights.cdict()
        else:
            gdict['raster_weights'] = None
        
        # Check to see if file is cached
        if self.cache is not None:
            cache = self.cache.uncache(gdict, extension='.feather')
        else:
            cache = None
            
        if cache is not None:
            print(f'Loading rescaled weights from cache')
            if self.verbose:
                print('Cache dictionary:')
                pprint(gdict)
            return cache
        else:
            if self.raster_weights is None:
                w = self.get_area_weights()
                w['weight'] = w['area_weight']
            else:
                w = self.get_weighted_area_weights()    
            if self.cache is not None:
                    self.cache.cache(w, gdict, extension='.feather')
            return w
        
    def cdict(self):
        gdict = {
            'grid':clean_object(self.grid),
            # 'georegions': clean_object(self.georegions),
            'georegions':{
                'regions':str(self.georegions.regions),
                'geometry':str(self.georegions.shp.geometry)
            }
        }

        if self.raster_weights is not None:
            gdict['raster_weights'] = clean_object(self.raster_weights)
        else:
            gdict['raster_weights'] = None

        return gdict   
        
#     # @lru_cache(maxsize=None)    
#     def grid_geo_from_centroids(self, region, buffer=0, geo=True):
#         gres = self.grid.resolution
#         m = self.mask(buffer).sel(region=region)
#         X, Y = np.meshgrid(self.grid.longitude, self.grid.latitude)
#         x = np.where(m, X, np.nan)
#         y = np.where(m, Y, np.nan)
        
#         xmin, xmax = np.nanmin(x)-gres/2, np.nanmax(x)+gres/2
#         ymin, ymax = np.nanmin(y)-gres/2, np.nanmax(y)+gres/2

#         xspace = np.linspace(xmin, xmax, int(round((xmax - xmin)/gres,0))+1)
#         yspace = np.linspace(ymin, ymax, int(round((ymax - ymin)/gres,0))+1)   

#         gridlat = [[[xmin, y],[xmax, y]] for y in yspace]
#         gridlon = [[[x, ymin],[x, ymax]] for x in xspace]
#         gridshp = np.append(pygeos.linestrings(gridlat), 
#                             pygeos.linestrings(gridlon), axis=0)
    
#         if geo:
#             return gpd.GeoSeries(gridshp)
#         else:
#             return gridshp
    
#     # @lru_cache(maxsize=None) 
#     def interior_centroids_geo(self, region, buffer=0):
#         m = self.mask(buffer).sel(region=region)
#         return bool_array_to_geoseries(m)

#     # @lru_cache(maxsize=None) 
#     def border_centroids(self, buffers=None, output='mask'):
#         if buffers is None:
#             buffers = (-self.grid.resolution, self.grid.resolution)
        
#         m1 = self.mask(buffer = buffers[0])
#         m2 = self.mask(buffer = buffers[1])
#         mborder = np.logical_and(np.logical_not(m1), m2)
#         if output=='geo':
#             return bool_array_to_geoseries(mborder)
#         elif output=='points':
#             return bool_array_to_centroid_array(mborder)
#         elif output=='mask':
#             return mborder
#         else:
#             raise NotImplementedError
    
#     def centroids_to_cell(self, msk, chunksize=100, datatype='xarray', 
#                           compute=True, return_boxes=False):
#         pol = (self.georegions.poly_array(buffer=0)
#                       .rechunk((self.rchunk, -1, -1)))
#         X, Y = np.meshgrid(msk.lon, msk.lat)
#         cmask = xr.DataArray(
#             data = msk.data,
#             dims = ['region', 'latitude', 'longitude'],
#             coords = dict(
#                 lat = (['latitude', 'longitude'], Y),
#                 lon = (['latitude', 'longitude'], X),
#                 region = ('region', msk.region.values)
#             )
#         )
#         # cmask = cmask.compute()
#         mlon = np.moveaxis(np.array(cmask.lon.where(cmask)), -1, 0)
#         mlat = np.moveaxis(np.array(cmask.lat.where(cmask)), -1, 0)
#         lonpoints = [dask.array.from_array(mlon + x, chunks=(self.rchunk, -1, -1)) 
#                          for x in [-self.grid.resolution/2, self.grid.resolution/2]]
#         latpoints = [dask.array.from_array(mlat + x, chunks=(self.rchunk, -1, -1))
#                          for x in [-self.grid.resolution/2, self.grid.resolution/2]]
        
#         boxes = lonpoints[0].map_blocks(pygeos.box, latpoints[0], 
#                                         lonpoints[1], latpoints[1], dtype=float)
#         if return_boxes:
#             tr=boxes.compute()
#             tr = np.moveaxis(result.compute(), 0, -1)
#             return xr.DataArray(data=tr, dims=msk.dims, coords=msk.coords)
#         result = pol.map_blocks(pygeos.intersection, boxes, dtype=float)
#         if compute:
#             with warnings.catch_warnings():
#                 warnings.simplefilter("ignore")
#                 tr = result.compute()
#                 # tr = np.moveaxis(result.compute(), 0, -1)
#             if datatype == 'xarray':
#                 return xr.DataArray(data=tr, dims=msk.dims, coords=msk.coords)
#             elif datatype == 'dask':
#                 return dask.array.from_array(tr, chunks=chunksize)
#             elif datatype == 'array':
#                 return tr               
#             else:
#                 raise NotImplementedError
#         else:
#             return result
   
#     # @lru_cache(maxsize=None)
#     def border_centroids_to_cell(self, data={}, compute=True):
#         if 'border_centroids' not in data:
#             bc = self.border_centroids()
#         else:
#             bc = data['border_centroids']
#         return self.centroids_to_cell(bc)
    
#     # @lru_cache(maxsize=None)
#     def calculate_area_weights(self, data={}):
        
#         if 'border_cells' not in data:
#             borders = self.border_centroids_to_cell(data=data, compute=True)
#         else:
#             borders = data['border_cells']
        
#         b_area = ( borders.map_blocks(pygeos.area)
#                   .compute()
#                   .fillna(0) ) / self.grid.cell_area    
#         cells = (self.mask(buffer=-self.grid.resolution) * self.grid.cell_area) / self.grid.cell_area  
#         weights = xr.DataArray(data=b_area, 
#                                dims=cells.dims, 
#                                coords=cells.coords) + cells
#         return xr.DataArray(
#             data=weights.data,
#             dims = ['region', 'latitude', 'longitude'],
#             coords = dict(
#                 latitude = (['latitude'], weights.lat.values),
#                 longitude = (['longitude'], weights.lon.values),
#                 region = (['region'], weights.region.values)
#             )   
#         )
        
#     # @lru_cache(maxsize=None)
#     def calculate_weighted_area_weights(self, data={}, default_to_area_weights=True):
        
#         if 'area_weights' not in data:
#             aw = self.calculate_area_weights(data)
#         else:
#             aw = data['area_weights']
        
#         dsw = self.raster_weights.raster
        
#         wts = np.multiply(aw, dsw).data
        
#         da = xr.DataArray(
#             data=wts,
#             dims = ['region', 'latitude', 'longitude'],
#             coords = dict(
#                 latitude = (['latitude'], aw.latitude.values),
#                 longitude = (['longitude'], aw.longitude.values),
#                 region = (['region'], aw.region.values)
#             )   
#         )
        
#         # Something weird going on here
#         if default_to_area_weights:
#             w_sum = np.nansum(da.data.compute(), axis=(1,2))
#             # w_dims = (len(w_sum[w_sum == 0]),) + da.shape[1:3]
#             da = da.compute()
#             aw = aw.compute()
#             da[w_sum == 0,:,:] = aw[w_sum == 0,:,:]
            
#         return da    
    
#     # @lru_cache(maxsize=None)
#     def weights(self, data={}, chunk=-1, verbose=False):
        
#         # Set chunks
#         if chunk is None:
#             c = max([int(len(self.georegions.regions)/55),1])
#             chunk = (c, -1, -1)
        
#         gdict = {'func':'weights'} 
        
#         # Load raster weights if needed
#         if self.raster_weights is not None:
#             self.raster_weights.rescale_raster_to_grid(self.grid, verbose=verbose)
#             # gdict['raster_weights'] = dict(
#             #     path = self.raster_weights.path,
#             #     name = self.raster_weights.name,
#             #     crop = self.raster_weights.crop,
#             #     feed = self.raster_weights.feed
#             # )
#             gdict['raster_weights'] = self.raster_weights.cdict()
#         else:
#             gdict['raster_weights'] = None
        
#         # Check to see if file is cached
#         if self.cache is not None:
#             cache = self.cache.uncache(gdict)
#         else:
#             cache = None
            
#         if cache is not None:
#             print(f'Loading rescaled weights from cache')
#             if verbose:
#                 print('Cache dictionary:')
#                 pprint(gdict)
#             return cache.chunk(chunk)
#         else:
#             if self.raster_weights is None:
#                 w = self.calculate_area_weights(data).chunk(chunk)
#             else:
#                 w = self.calculate_weighted_area_weights(data).chunk(chunk)     
#             if self.cache is not None:
#                     self.cache.cache(w, gdict)
#             return w
    
#     def plot_weights(self, region, buffer=0, **kwargs):
#         mask = self.mask(buffer).sel(region=region)
#         w = self.raster_weights.raster.where(mask, drop=True)
#         return xr.DataArray(w, 
#             dims=['latitude','longitude'],
#             coords=dict(
#                 longitude=('longitude', w.lon.values),
#                 latitude=('latitude', w.lat.values))).plot(**kwargs)
    
#     def plot_grid(self, region, buffer=0, geo=True, **kwargs):
#         gridshp = self.grid_geo_from_centroids(region, buffer, geo)
#         return gridshp.plot(**kwargs)
    
#     def plot_interior_centroids(self, region, buffer=0, **kwargs):
#         gridshp = self.interior_centroids_geo(region, buffer)
#         return gridshp.plot(**kwargs)
    
#     def plot_border_areas(self, region, **kwargs):
#         border_centroids = self.border_centroids_to_cell()
#         return gpd.GeoSeries(areas).plot(**kwargs)


def from_objects(clim, georegions, wtype='crop', name='cropland', crop='corn', feed=None, write=False, project_dir=None, **kwargs):
    
    if wtype == 'crop':
        if crop is not None:
            raster_weights = crop_weights.from_name(
                name=name, crop=crop, feed=feed, write=write, project_dir=project_dir)
        else:
            raise NotImplementedError
    elif wtype == 'pop':
        raster_weights = pop_weights.from_name(
                name=name, write=write, project_dir=project_dir)
    else:
        raster_weights = None
    
    return GridWeights(
        clim.grid, georegions, raster_weights, project_dir=project_dir, **kwargs)
    
    
    