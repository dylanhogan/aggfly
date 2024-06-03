import numpy as np
import pandas as pd
import xarray as xr
import shapely
import geopandas as gpd
import os
import dask
import dask.array
import dask_geopandas
from functools import lru_cache
import warnings

from .grid_utils import *


class Grid:
    """
    A class representing a spatial grid with longitude and latitude coordinates.

    Attributes:
    -----------
    longitude: np.ndarray
        Array of longitude coordinates.
    latitude: np.ndarray
        Array of latitude coordinates.
    lon_array: np.ndarray
        Meshgrid array of longitude coordinates.
    lat_array: np.ndarray
        Meshgrid array of latitude coordinates.
    name: str
        Name of the grid.
    lon_is_360: bool
        Indicates if the longitude coordinates are in the range [0, 360].
    index: np.ndarray
        Index array for the grid.
    cell_id: np.ndarray
        Flattened index array representing cell IDs.
    resolution: float
        Resolution of the grid.
    cell_area: np.ndarray
        Array representing the area of each cell in the grid.

    Methods:
    --------
    get_index():
        Generates an index array for the grid.
    get_resolution():
        Calculates the resolution of the grid.
    get_cell_area():
        Calculates the area of each cell in the grid.
    """
    
    def __init__(self, longitude, latitude, name, lon_is_360):
        """
        Initializes the Grid object with longitude and latitude coordinates.

        Parameters:
        -----------
        longitude: np.ndarray
            Array of longitude coordinates.
        latitude: np.ndarray
            Array of latitude coordinates.
        name: str
            Name of the grid.
        lon_is_360: bool
            Indicates if the longitude coordinates are in the range [0, 360].
        """
        self.longitude = longitude
        self.latitude = latitude
        # Create meshgrid arrays for longitude and latitude
        self.lon_array, self.lat_array = np.meshgrid(self.longitude, self.latitude)
        self.name = name
        self.lon_is_360 = lon_is_360
        # Generate index array for the grid
        self.index = self.get_index()
        # Flatten index array to get cell IDs
        self.cell_id = self.index.flatten()
        # Calculate resolution of the grid
        self.resolution = self.get_resolution()
        # Calculate area of each cell in the grid
        self.cell_area = self.get_cell_area()

    @lru_cache(maxsize=None)
    # Function that gnerates the centroids for the grid.
    def centroids(self, datatype="points", chunks=30):
        # Reformat the grid to the specified data structure
        centroids = reformat_grid(self.lon_array, self.lat_array, datatype, chunks)
        # Return the centroids
        return centroids

    def get_resolution(self):
        return abs(np.diff(self.latitude).mean())

    def get_cell_area(self):
        cell = shapely.box(0, 0, self.resolution, self.resolution)
        return shapely.area(cell)
    
    def get_index(self):
        if self.lon_is_360:
            longitude = lon_to_180(self.longitude)
        else:
            longitude = self.longitude
        
        lon_array, lat_array = np.meshgrid(longitude, self.latitude)
        return(np.indices(lon_array.flatten().shape).reshape(lon_array.shape))

    @lru_cache(maxsize=None)
    def clip_grid_to_georegions_extent(self, georegions):
        bounds = georegions.shp.total_bounds
        if self.lon_is_360:
            all_bounds = lon_to_360(np.array(georegions.shp.bounds)[:, [0, 2]])
            xmin = all_bounds[:, 0].min()
            xmax = all_bounds[:, 1].max()
            bounds = georegions.shp.total_bounds
            bounds[[0, 2]] = [xmin, xmax]

        self.clip_grid_to_bbox(bounds)

    def clip_grid_to_bbox(self, bounds):
        inlon = np.logical_and(
            self.longitude >= bounds[0] - self.resolution / 2,
            self.longitude <= bounds[2] + self.resolution / 2,
        )
        inlon_b = [self.longitude[inlon].min(), self.longitude[inlon].max()]

        inlat = np.logical_and(
            self.latitude >= bounds[1] - self.resolution / 2,
            self.latitude <= bounds[3] + self.resolution / 2,
        )
        inlat_b = [self.latitude[inlat].min(), self.latitude[inlat].max()]

        longitude, latitude = grid_centroids(inlon_b, inlat_b, self.resolution)

        self.longitude = self.longitude[inlon]
        self.latitude = self.latitude[inlat]
        self.lon_array, self.lat_array = np.meshgrid(self.longitude, self.latitude)
        self.index = np.indices(self.lon_array.flatten().shape).reshape(
            self.lon_array.shape
        )

    @lru_cache(maxsize=None)
    def mask(self, georegions, buffer=0, chunksize=100, compute=True):
        mask = (
            georegions.poly_array(buffer, "dask")
            .rechunk(chunksize)
            .map_blocks(shapely.contains, self.centroids(), dtype=float)
        )
        if compute:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # m = np.moveaxis(mask.compute().squeeze(), -1, 0)
                m = mask.compute().squeeze()
                da = xr.DataArray(
                    data=m,
                    dims=["region", "latitude", "longitude"],
                    coords=dict(
                        lon=(["longitude"], self.longitude.values),
                        lat=(["latitude"], self.latitude.values),
                        region=(["region"], georegions.regions.values),
                    ),
                )
                return da
        else:
            return mask

    @lru_cache(maxsize=None)
    def mask(self, buffer=0):
        centroids = self.grid.centroids()
        fc = gpd.GeoDataFrame(geometry=centroids.flatten()).set_crs("EPSG:4326")

        # fc = fc.rechunk(chunks)[...,None]
        fc = dask_geopandas.from_geopandas(fc, npartitions=self.chunks)

        if self.simplify is not None:
            georegions = self.simplify_poly_array()
        else:
            georegions = self.georegions

        poly_array = georegions.poly_array(buffer, chunks=self.chunks).squeeze()

        poly_shp = poly_array.shape
        poly_array = dask_geopandas.from_geopandas(
            gpd.GeoDataFrame(geometry=poly_array), npartitions=1
        )

        # mask = fc.map_blocks(shapely.within, poly_array, dtype=bool)
        mask = fc.sjoin(poly_array, predicate="within").compute()

        return mask

    def centroids_to_cell(
        self,
        georegions,
        buffer=0,
        chunksize=100,
        datatype="xarray",
        compute=True,
        intersect_cells=False,
    ):
        # Generate dask array of geometries
        pol = georegions.poly_array(buffer=buffer)
        print(pol)

        # Generate mask array contining T/F for centroids within (buffered) geometry
        msk = self.mask(georegions, buffer=buffer)
        X, Y = np.meshgrid(msk.lon, msk.lat)
        cmask = xr.DataArray(
            data=msk.data,
            dims=["region", "latitude", "longitude"],
            coords=dict(
                lat=(["latitude", "longitude"], Y),
                lon=(["latitude", "longitude"], X),
                region=("region", msk.region.values),
            ),
        )

        #
        mlon = np.moveaxis(np.array(cmask.lon.where(cmask)), -1, 0)
        mlat = np.moveaxis(np.array(cmask.lat.where(cmask)), -1, 0)
        lonpoints = [
            dask.array.from_array(mlon + x, chunks=(chunksize, -1, -1))
            for x in [-self.resolution / 2, self.resolution / 2]
        ]
        latpoints = [
            dask.array.from_array(mlat + x, chunks=(chunksize, -1, -1))
            for x in [-self.resolution / 2, self.resolution / 2]
        ]
        boxes = lonpoints[0].map_blocks(
            shapely.box, latpoints[0], lonpoints[1], latpoints[1], dtype=float
        )

        # Return full cells or intersection of cell and polygon (e.g. for area weights)
        if not intersect_cells:
            tr = boxes.compute()
            # tr = np.moveaxis(tr, 0, -1)
            return tr

        result = pol.map_blocks(shapely.intersection, boxes, dtype=float)
        if compute:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tr = result.compute()
                # tr = np.moveaxis(result.compute(), 0, -1)
            if datatype == "xarray":
                return xr.DataArray(data=tr, dims=msk.dims, coords=msk.coords)
            elif datatype == "dask":
                return dask.array.from_array(tr, chunks=chunksize)
            elif datatype == "array":
                return tr
            # elif datatype == 'dataframe':
            else:
                raise NotImplementedError
        else:
            return result
