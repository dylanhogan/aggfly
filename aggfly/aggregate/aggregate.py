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

from aggregate.ClimateDataset import ClimateDataset
from aggregate.GridWeights import GridWeights
from aggregate.SpatialAggregator import SpatialAggregator
from aggregate.TemporalAggregator import TemporalAggregator

from aggregate.utils import *

class ClimateAggregator:
    
    def __init__(self,
                 climate_data=ClimateDataset(),
                 grid_weights=GridWeights(),
                 temporal_aggregator=TemporalAggregator(),
                 spatial_aggregator=SpatialAggregator()):
    
        self.climate_data = climate_data
        self.grid_weights = grid_weights
        self.temporal_aggregator = temporal_aggregator
        self.spatial_aggregator = spatial_aggregator
        
    def execute(self):
        
        daily = self.temporal_aggregator.daily.execute(self.climate_data.data.values)
        
        return (
            self.temporal_aggregator.yearly.execute(
            self.spatial_aggregator.execute(
            self.temporal_aggregator.daily.execute(self.climate_data.data.values), self.grid_weights)))
            
#     def clip_data_to_georegions_extent(self, climate_data):
        
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
        
    
    

    
    
    