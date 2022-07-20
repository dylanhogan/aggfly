import numpy as np
import pandas as pd
import xarray as xr
import pygeos
import geopandas as gpd
import os
import dask
import dask.array
from functools import lru_cache

from aggregate.gridfuncs import *

class ClimateGrid:
    
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
        return np.diff(self.longitude).mean()

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