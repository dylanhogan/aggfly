import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import shapely
import dask.array
from hashlib import sha256
import json


def bool_array_to_geoseries(m):
    X, Y = np.meshgrid(m.lon.values, m.lat.values)
    inlon = np.where(m, X, np.nan).flatten()
    inlat = np.where(m, Y, np.nan).flatten()

    cent = shapely.points(
        inlon[np.logical_not(np.isnan(inlon))], inlat[np.logical_not(np.isnan(inlat))]
    )
    return gpd.GeoSeries(cent)


def bool_array_to_centroid_array(m, compute=True, chunksize=100):
    X, Y = np.meshgrid(m.lon.values, m.lat.values)
    inlon = dask.array.from_array(np.where(m, X, np.nan), chunks=chunksize)
    inlat = dask.array.from_array(np.where(m, Y, np.nan), chunks=chunksize)
    output = inlon.map_blocks(shapely.points, inlat, dtype=float)
    if compute:
        return output.compute()
    else:
        return output


def geom_array_to_geoseries(ar, region):
    series = ar.sel(region=region).values.flatten()
    series = series[series != None]
    return gpd.GeoSeries(series)


def autochunk(arr, ncpu=55):
    if type(arr) == dask.array.core.Array:
        template = np.empty_like(arr.compute(), dtype=int)
    else:
        template = np.empty_like(arr, dtype=int)
    chunks = np.array_split(template, ncpu)[0].shape
    return chunks


def hash_obj(obj, n=10):
    gdict = obj.grid.__dict__
    dump = json.dumps(str(gdict), sort_keys=True).encode("utf8")
    return sha256(dump).hexdigest()[:n]
