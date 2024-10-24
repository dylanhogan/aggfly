import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import shapely
import dask.array
from hashlib import sha256
import json


def bool_array_to_geoseries(m):
    """
    Convert a boolean array to a GeoSeries of points.

    Parameters:
    -----------
    m : xarray.DataArray
        The boolean array with dimensions lat and lon.

    Returns:
    --------
    gpd.GeoSeries
        The GeoSeries of points where the boolean array is True.
    """
    # Create meshgrid of lon and lat values
    X, Y = np.meshgrid(m.lon.values, m.lat.values)
    # Flatten array with lon where m is True
    inlon = np.where(m, X, np.nan).flatten()
    # Flatten array with lat where m is True
    inlat = np.where(m, Y, np.nan).flatten()

    cent = shapely.points(
        inlon[np.logical_not(np.isnan(inlon))], inlat[np.logical_not(np.isnan(inlat))]
    ) # Create points from valid lon and lat values
    return gpd.GeoSeries(cent) # Return GeoSeries of points


def bool_array_to_centroid_array(m, compute=True, chunksize=100):
    """
    Convert a boolean array to an array of centroids using Dask.

    Parameters:
    -----------
    m : xarray.DataArray
        The boolean array with dimensions lat and lon.
    compute : bool, optional
        Whether to compute the result immediately (default is True).
    chunksize : int, optional
        The chunk size for Dask arrays (default is 100).

    Returns:
    --------
    dask.array.Array or np.ndarray
        The array of centroids.
    """
    # Create meshgrid of lon and lat values
    X, Y = np.meshgrid(m.lon.values, m.lat.values) 
    # Create Dask array for lon
    inlon = dask.array.from_array(np.where(m, X, np.nan), chunks=chunksize) 
    # Create Dask array for lat
    inlat = dask.array.from_array(np.where(m, Y, np.nan), chunks=chunksize)
    # Map points to Dask arrays
    output = inlon.map_blocks(shapely.points, inlat, dtype=float) 
    if compute:
        # Compute and return result
        return output.compute()
    else:
        # Return Dask array
        return output


def geom_array_to_geoseries(ar, region):
    """
    Convert an array of geometries to a GeoSeries for a specific region.

    Parameters:
    -----------
    ar : xarray.DataArray
        The array of geometries.
    region : str
        The region to select.

    Returns:
    --------
    gpd.GeoSeries
        The GeoSeries of geometries for the specified region.
    """
    # Select and flatten geometries for region
    series = ar.sel(region=region).values.flatten()
    # Remove None values
    series = series[series != None]
    # Return GeoSeries
    return gpd.GeoSeries(series)


def autochunk(arr, ncpu=55):
    """
    Determine the chunk size for an array based on the number of CPUs.

    Parameters:
    -----------
    arr : np.ndarray or dask.array.core.Array
        The array to chunk.
    ncpu : int, optional
        The number of CPUs to use (default is 55).

    Returns:
    --------
    tuple
        The chunk size for the array.
    """
    if type(arr) == dask.array.core.Array:
        # Create template array
        template = np.empty_like(arr.compute(), dtype=int)
    else:
        # Create template array
        template = np.empty_like(arr, dtype=int)
    # Determine chunk size
    chunks = np.array_split(template, ncpu)[0].shape
    # Return chunk size
    return chunks


def hash_obj(obj, n=10):
    """
    Compute a hash for an object.

    Parameters:
    -----------
    obj : object
        The object to hash.
    n : int, optional
        The length of the hash (default is 10).

    Returns:
    --------
    str
        The hash of the object.
    """
    # Get grid dictionary from object
    gdict = obj.grid.__dict__
    # Serialize and encode dictionary
    dump = json.dumps(str(gdict), sort_keys=True).encode("utf8")
    # Compute and return hash
    return sha256(dump).hexdigest()[:n]
