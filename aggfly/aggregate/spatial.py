from functools import lru_cache, partial
from copy import deepcopy
import warnings

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

class SpatialAggregator:
    
    def __init__(self, clim, weights, grid, names='climate'):
        if type(clim) != list:
            self.clim = [self.clim]
        else:
            self.clim = clim
        self.grid = grid
        # self.agg_from = agg_from
        self.weights = weights
        self.names=names
        # self.func = self.assign_func()
        self.weighted_mean = dask.dataframe.Aggregation(
            name='weighted_mean',
            chunk=process_chunk,
            agg=agg,
            finalize=finalize)
    
    def assign_func(self):
        if self.calc == 'avg':
            f = _avg
            self.args = (self.poly,)
        return f
    
    def execute(self, arr, weight, **kwargs):
        # return **kwargs
        return self.func(arr, weight, *self.args, **kwargs)
    
    def compute(self, npartitions=30):
        
        print('COMPUTING')
        clim_ds = dask.compute([x.da for x in self.clim])[0]
        clim_ds = xr.combine_by_coords(
            [x.to_dataset(name=self.names[i]) for i, x in enumerate(clim_ds)]
        )
        
        clim_df = (clim_ds
            .stack({'cell_id':['latitude', 'longitude']})
            .drop_vars(['cell_id', 'latitude', 'longitude'])
            .assign_coords(coords={'cell_id':('cell_id', self.grid.index.flatten())})
            .to_dataframe()
            .reset_index('time')
            .dropna(subset=self.names)
        )
        
        self.weights['region_id'] = self.weights.index_right
        merged_df = clim_df.merge(self.weights, how='inner', on='cell_id')
        merged_df = merged_df.dropna(subset=self.names)
        
        group_key = ( merged_df[['region_id', 'time']]
            .drop_duplicates()
            .reset_index(drop=True)
            .reset_index()
            .rename(columns={'index':'group_ID'})
        )        

        merged_df = ( merged_df
             .merge(group_key, on=['region_id', 'time'])
             .set_index('group_ID')[['weight', *self.names]]
        )
        
        ddf = dask.dataframe.from_pandas(merged_df, npartitions=50)
        
        out = ddf[self.names].mul(ddf['weight'], axis=0)
        out['weight'] = ddf['weight']
        out = out.groupby(out.index).sum()
        out = out[self.names].div(out['weight'], axis=0)
        
        aggregated = (out
            .merge(group_key, how='right', left_index=True, right_on='group_ID')
            .compute()
            .reset_index(drop=True)
            .drop(columns='group_ID')[['region_id', 'time'] + self.names]
        )
            
        return aggregated
    

@numba.njit(fastmath=True, parallel=True)
def _avg(frame, weight, poly):

    frame_shp = frame.shape
    res = np.zeros((weight.shape[0],) + frame.shape[2:], dtype=np.float64)
    wes = np.zeros((weight.shape[0],) + frame.shape[2:], dtype=np.float64)
    for r in prange(weight.shape[0]):
        # print(r)
        yl, xl = np.nonzero(weight[r,:,:])
        for t in prange(frame.shape[2]):
            if len(yl) == 0:
                res[r,a,m,d,h] = np.nan
                wes[r,a,m,d,h] = np.nan
            else:
                for l in prange(len(yl)):
                    y = yl[l]
                    x = xl[l]
                    # I can't believe this was actually the solution
                    # https://github.com/numba/numba/issues/2919
                    if int(frame[y,x,a,m,d,h]) != -9223372036854775808:
                        f = frame[y,x,a,m,d,h]
                        w = weight[r,y,x]
                        res[r,a,m,d,h] += f * w
                        wes[r,a,m,d,h] += w
    out = res/wes
    return out.reshape((weight.shape[0],) + frame_shp[2:])

def process_chunk(chunk):
    def weighted_func(df):
        # print(df)
        return (df["weight"] * df["climate"]).sum()
    return (chunk.apply(weighted_func), chunk.sum()["weight"])
        
def agg(total, weights):
    return (total.sum(), weights.sum())

def finalize(total, weights):
    return total / weights

def weighted(x, cols, w="weight"):
    return pd.Series(np.average(x[cols], weights=x[w], axis=0), cols)

def from_name(name='era5l',
              calc=('avg', 1)):
    if name == 'era5l':
        return SpatialAggregator(calc=calc[0], poly=calc[1])
    else:
        raise NotImplementedError
