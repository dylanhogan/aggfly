# This script provides utility functions converting longitudes between different ranges,
# generating grid centroids, and preprocessing ERA5-Land datasets by adjusting longitudes, sorting, and extracting time components.

import numpy as np
import pandas as pd
import xarray as xr
import shapely
import os
import dask
import dask.array
from functools import lru_cache

from ..utils import autochunk

# Function that converts longitude values to the range [-180, 180].
@np.vectorize 
def lon_to_180(longitude):
    """
    Converts longitude values to the range [-180, 180].

    Parameters:
    -----------
    longitude: float
        The longitude value to convert.

    Returns:
    --------
    float:
        The converted longitude value.
    """
    return (longitude + 180) % 360 - 180
    
# Function that converts longitude values to the range [0, 360].
@np.vectorize
def lon_to_360(longitude):
    """
    Converts longitude values to the range [0, 360].

    Parameters:
    -----------
    longitude: float
        The longitude value to convert.

    Returns:
    --------
    float:
        The converted longitude value.
    """
    return (longitude < 0) * (longitude + 360) + (longitude >= 0) * longitude

# Function that converts the longitude coordinate of an xarray DataArray to the range [-180, 180]
def array_lon_to_180(array):
    """
    Converts the longitude coordinate of an xarray DataArray to the range [-180, 180].

    Parameters:
    -----------
    array: xarray.DataArray
        The DataArray with longitude coordinates to convert.

    Returns:
    --------
    xarray.DataArray:
        The DataArray with converted longitude coordinates.
    """
    # to_180 = (array.coords['longitude'] + 180) % 360 - 180
    # Convert the longitude coordinates using lon_to_180
    to_180 = lon_to_180(array.coords["longitude"])
    # Assign the converted longitude coordinates back to the array and sort by longitude
    array = array.assign_coords({"longitude": ("longitude", to_180.data)}).sortby(
        "longitude"
    )
    return array


def array_lon_to_360(array):
    """
    Converts the longitude coordinate of an xarray DataArray to the range [0, 360].

    Parameters:
    -----------
    array: xarray.DataArray
        The DataArray with longitude coordinates to convert.

    Returns:
    --------
    xarray.DataArray:
        The DataArray with converted longitude coordinates.
    """
    # Convert the longitude coordinates using lon_to_360
    to_360 = lon_to_360(array.coords["longitude"])
    # Assign the converted longitude coordinates back to the array and sort by longitude
    array = array.assign_coords({"longitude": ("longitude", to_360.data)}).sortby(
        "longitude"
    )
    return array


def grid_centroids(lon_bound, lat_bound, res):
    """
    Generates a grid of centroids based on longitude and latitude bounds and resolution.

    Parameters:
    -----------
    lon_bound: list
        The longitude bounds [min, max].
    lat_bound: list
        The latitude bounds [min, max].
    res: float
        The resolution of the grid.

    Returns:
    --------
    tuple:
        Two numpy arrays representing the longitude and latitude centroids.
    """
    # Create a mesh grid of longitude and latitude centroids
    longitude, latitude = np.meshgrid(
        np.arange(lon_bound[0], lon_bound[1], res),
        np.arange(lat_bound[0], lat_bound[1], res),
    )
    return longitude, latitude


@lru_cache(maxsize=None)
def prism_grid_centroids(datatype="array", chunks=30):
    """
    Generates the centroids for the PRISM grid with specific bounds and resolution.

    Parameters:
    -----------
    datatype: str, optional
        The type of data structure to return ("array" or "dataframe"). Default is "array".
    chunks: int, optional
        The size of chunks for Dask arrays. Default is 30.

    Returns:
    --------
    tuple:
        A tuple containing the centroids, unique longitudes, and unique latitudes.
    """
    res = 0.041666666666 # Set the resolution
    lon_bound = [-125, -66.5] # Define the longitude bounds
    lat_bound = [24 + 1 / 12, 49.9375 - 1 / 48] # Define the latitude bounds
    longitude, latitude = grid_centroids(lon_bound, lat_bound, res) # Generate grid centroids
    centroids = reformat_grid(longitude, latitude, datatype, chunks) # Reformat the grid centroids
    return centroids, np.unique(longitude), np.unique(latitude) # Return the centroids and unique longitudes and latitudes


@lru_cache(maxsize=None)
def era5l_grid_centroids(datatype="array", chunks=30, usa=False):
    """
    Generates the centroids for the ERA5-Land grid with specific bounds and resolution.

    Parameters:
    -----------
    datatype: str, optional
        The type of data structure to return ("array", "coords", "points", "dask", "xarray", or "empty"). Default is "array".
    chunks: int, optional
        The size of chunks for Dask arrays. Default is 30.
    usa: bool, optional
        Whether to select data only for the USA. Default is False.

    Returns:
    --------
    centroids:
        The centroids in the specified data structure.
    """
    # Open the ERA5 data
    example = "/home3/dth2133/data/ERA5"
    clim = xr.open_zarr(example).t2m[:, :, 0, 0, 0]
    # clim.coords['longitude'] = (clim.coords['longitude'] + 180) % 360 - 180
    # with dask.config.set(**{'array.slicing.split_large_chunks': True}):
    #     clim = clim.sortby(clim.longitude)
    # Optionally filter for USA region
    if usa:
        clim = clim.sel(longitude=slice(-126, -67), latitude=slice(50, 24))
    # Return coordinates or reformatted grid based on datatype
    if datatype == "coords":
        return clim.longitude.values, clim.latitude.values
    else:
        longitude, latitude = np.meshgrid(clim.longitude.values, clim.latitude.values)
        centroids = reformat_grid(longitude, latitude, datatype, chunks)
        return centroids


def reformat_grid(longitude, latitude, datatype="array", chunks=30):
    """
    Reformats grid coordinates into different data structures.

    Parameters:
    -----------
    longitude: np.ndarray or dask.array.Array
        The longitude values.
    latitude: np.ndarray or dask.array.Array
        The latitude values.
    datatype: str, optional
        The type of data structure to return ("array", "coords", "points", "dask", "xarray", or "empty"). Default is "array".
    chunks: int, optional
        The size of chunks for Dask arrays. Default is 30.

    Returns:
    --------
    centroids:
        The centroids in the specified data structure.
    """
    if datatype == "points":
        # Create Shapely points
        centroids = shapely.points(longitude, y=latitude)
    elif datatype == "dask":
        # Convert to Dask arrays if necessary and create Shapely points
        if type(longitude) is not dask.array.core.Array:
            longitude, latitude = [
                dask.array.from_array(x, chunks="auto") for x in [longitude, latitude]
            ]
        centroids = longitude.map_blocks(shapely.points, y=latitude, dtype=float)
        centroids = centroids.rechunk(autochunk(centroids))
    elif datatype == "xarray":
        # Convert to Dask arrays if necessary, create Shapely points, and return as xarray.DataArray
        if type(longitude) is not dask.array.core.Array:
            longitude, latitude = [
                dask.array.from_array(x, chunks="auto") for x in [longitude, latitude]
            ]
        centroids = longitude.map_blocks(shapely.points, y=latitude, dtype=float)
        centroids = centroids.rechunk(autochunk(centroids))
        return xr.DataArray(
            data=centroids,
            dims=["latitude", "longitude"],
            coords={"longitude": longitude, "latitude": latitude},
        )
    elif datatype == "array":
        # Return as numpy arrays
        centroids = (longitude, latitude)
    elif datatype == "empty":
        # Return None
        return None
    else:
        # Raise an error for unsupported data types
        raise NotImplementedError
    return centroids


def preprocess_era5l(array):
    """
    Preprocesses an ERA5-Land array by adjusting longitudes, sorting, extracting time components, and converting temperature units.

    Parameters:
    -----------
    array: xarray.DataArray
        The ERA5-Land data array to preprocess.

    Returns:
    --------
    xarray.DataArray:
        The preprocessed ERA5-Land data array.
    """
    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        # Adjust longitudes to the range [-180, 180]
        array.coords["longitude"] = (array.coords["longitude"] + 180) % 360 - 180
        # Sort the array by longitude
        array = array.sortby(array.longitude)
        # Extract year, month, day, and hour from the time coordinate
        array["year"] = array.time.dt.year
        array["month"] = array.time.dt.month
        array["day"] = array.time.dt.day
        array["hour"] = array.time.dt.hour
        
        # Set the time coordinate to a multi-index of year, month, day, and hour and unstack the time dimension
        array = array.set_index(time=("year", "month", "day", "hour")).unstack("time")
        
        # Convert temperature from Kelvin to Celsius
        array = array - 273.15
    return array


def timefix_era5l(array):
    """
    Fixes the time coordinate of an ERA5-Land array by extracting time components and setting a multi-index.

    Parameters:
    -----------
    array: xarray.DataArray
        The ERA5-Land data array to fix the time coordinate.

    Returns:
    --------
    xarray.DataArray:
        The ERA5-Land data array with fixed time coordinates.
    """
    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        # Extract year, month, day, and hour from the time coordinate
        array["year"] = array.time.dt.year
        array["month"] = array.time.dt.month
        array["day"] = array.time.dt.day
        array["hour"] = array.time.dt.hour
        
        # Set the time coordinate to a multi-index of year, month, day, and hour and unstack the time dimension
        array = array.set_index(time=("year", "month", "day", "hour")).unstack("time")
    return array


def timefix(array, split_chunks=False):
    """
    Fixes the time coordinate of an array by extracting time components and setting a multi-index.

    Parameters:
    -----------
    array: xarray.DataArray
        The data array to fix the time coordinate.
    split_chunks: bool, optional
        Whether to split large chunks during array slicing. Default is False.

    Returns:
    --------
    xarray.DataArray:
        The data array with fixed time coordinates.
    """
    # Configure Dask to handle large chunks based on the split_chunks parameter
    with dask.config.set(**{"array.slicing.split_large_chunks": split_chunks}):
        # Extract year, month, day, and hour from the time coordinate
        array["year"] = array.time.dt.year
        array["month"] = array.time.dt.month
        array["day"] = array.time.dt.day
        array["hour"] = array.time.dt.hour
        
        # Set the time coordinate to a multi-index of year, month, day, and hour and unstack the time dimension
        array = array.set_index(time=("year", "month", "day", "hour")).unstack("time")
        return array


def clean_dims(da, xycoords):
    """
    Renames the dimensions of a DataArray to "longitude" and "latitude" if they are different, and transposes the DataArray.

    Parameters:
    -----------
    da: xarray.DataArray
        The data array to clean dimensions.
    xycoords: tuple
        The current names of the spatial coordinates (longitude, latitude).

    Returns:
    --------
    xarray.DataArray:
        The data array with cleaned dimensions.
    """
    # Rename the dimensions if they are different from "longitude" and "latitude"
    if xycoords != ("longitude", "latitude"):
        da = da.rename({xycoords[0]: "longitude", xycoords[1]: "latitude"})
    # Transpose the DataArray to ensure "latitude" and "longitude" are the first dimensions
    return da.transpose("latitude", "longitude", ...)
