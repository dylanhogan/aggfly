# This script defines the GeoRegions class for representing and manipulating geographical regions using shapefiles.
# It includes methods for initializing the GeoRegions object, selecting and dropping regions, generating polygon arrays,
# and plotting region boundaries. Additionally, it provides utility functions for loading GeoRegions from paths or names.

import os
from typing import List, Optional, Union
import numpy as np
import matplotlib.pyplot as mpl

import geopandas as gpd
import geopandas as gpd
import dask_geopandas
import dask
import dask.array
from copy import deepcopy
import warnings

from .shp_utils import *

# Weird bug in pyproj or geopandas that results in inf values the first time
# a shapefile is loaded.. only for certain installations of PROJ
# https://github.com/arup-group/genet/issues/213
import pyproj
pyproj.network.set_network_enabled(False)


class GeoRegions:
    """
    A class used to represent geographical regions.

    Attributes
    ----------
    shp : geopandas.GeoDataFrame
        The shapefile of the geographical regions.
    regionid : str
        The identifier of the regions.
    regions : geopandas.GeoSeries
        The series of regions.
    name : str
        The name of the geographical regions.
    path : str
        The path to the shapefile.
    """

    def __init__(
        self,
        shp: gpd.GeoDataFrame = None,
        regionid: str = None,
        region_list: list = None,
        name: str = None,
        path: str = None,
        crs: str = "WGS84"
    ):
        """
        Constructs all the necessary attributes for the GeoRegions object.

        Parameters
        ----------
        shp : geopandas.GeoDataFrame, optional
            The shapefile of the geographical regions (default is None).
        regionid : str, optional
            The identifier of the regions (default is None).
        region_list : list, optional
            The list of regions to select (default is None).
        name : str, optional
            The name of the geographical regions (default is None).
        path : str, optional
            The path to the shapefile (default is None).
        crs : str, optional
        The coordinate reference system for the shapefile (default is "WGS84").
        """
        try: 
            shp.crs
            # Check if the shapefile has a coordinate reference system (CRS)
            if crs != shp.crs:
                print(f"Converting shapefile CRS to {crs}")
                shp = shp.to_crs(crs)   
        except:
            # Raise an error if the shapefile does not have a CRS
            raise ValueError('Shapefile does not have a CRS assigned to it')

        # Reset the index of the shapefile GeoDataFrame
        self.shp = shp.reset_index(drop=True)
        # Set the region identifier
        self.regionid = regionid
        # Extract the regions from the shapefile using the region identifier
        self.regions = self.shp[self.regionid]
        # If a list of regions is provided, select these regions
        if region_list is not None:
            self.sel(region_list, update=True)
        # Set the name and path attributes
        self.name = name
        self.path = path

    def poly_array(
        self, buffer: int = 0, datatype: str = "array", chunks: int = 20
    ) -> Union[np.ndarray, dask.array.Array]:
        """
        Returns a polygon array of the geographical regions.

        Parameters
        ----------
        buffer : int, optional
            The buffer size (default is 0).
        datatype : str, optional
            The type of the data (default is "array").
        chunks : int, optional
            The number of chunks (default is 20).

        Returns
        -------
        Union[np.ndarray, dask.array.Array]
            The polygon array.
        """
        # If a buffer size is specified, create a buffered polygon array
        if buffer != 0:
            # Convert GeoDataFrame to Dask GeoDataFrame with specified number of partitions
            ddf = dask_geopandas.from_geopandas(self.shp, npartitions=chunks)
            # Suppress warnings related to buffering operation
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Suppress warnings related to buffering operation
                bufferPoly = ddf.buffer(buffer).compute()
        else:
            # Suppress warnings related to buffering operation
            bufferPoly = self.shp.geometry

        # Suppress warnings related to buffering operation
        if datatype == "dask":
            # Create a Dask array from the buffered polygons with specified chunk size
            ar = dask.array.from_array(
                bufferPoly, chunks=int(len(bufferPoly) / chunks)
            ).reshape(len(bufferPoly), 1, 1)
            return ar
        elif datatype == "array":
            # Return the buffered polygons as a NumPy array
            return bufferPoly
        else:
            # Raise an error if the datatype is not supported
            raise NotImplementedError

    def plot_region(self, region: str, **kwargs):
        """
        Plots the boundary of a region.

        Parameters
        ----------
        region : str
            The region to plot.

        Returns
        -------
        mpl.pyplot
            The plot of the region boundary.
        """
        # Get the geometry of the specified region
        geo = self.shp.loc[self.regions == region].geometry
        # Plot the boundary of the region using the specified keyword arguments
        return geo.boundary.plot(**kwargs)

    def sel(self, region_list: Union[str, list], update: bool = False):
        """
        Selects regions.

        Parameters
        ----------
        region_list : Union[str, list]
            The list of regions to select.
        update : bool, optional
            A flag indicating if the regions should be updated (default is False).
        Returns
        -------
        GeoRegions
            The GeoRegions object with the selected regions.
        """
        # Ensure region_list is a list
        region_list = (
            [region_list] if not isinstance(region_list, list) else region_list
        )
        # Determine whether to update in place or create a deepcopy
        if update:
            shp = self
        else:
            shp = deepcopy(self)
            
        # Create a mask to select the specified regions
        m = np.in1d(shp.regions, region_list)
        # Apply the mask to the shapefile and regions
        shp.shp = shp.shp[m].reset_index(drop=True)
        shp.regions = shp.regions[m].reset_index(drop=True)
        # Return the GeoRegions object with the selected regions
        return shp

    def drop(self, region_list: Union[str, list], update: bool = False):
        """
        Drops regions.

        Parameters
        ----------
        region_list : Union[str, list]
            The list of regions to select.
        update : bool, optional
            A flag indicating if the regions should be updated (default is False).
        Returns
        -------
        GeoRegions
            The GeoRegions object with the specified regions dropped.
        """
        # Ensure region_list is a list
        region_list = (
            [region_list] if not isinstance(region_list, list) else region_list
        )
        # Determine whether to update in place or create a deepcopy
        if update:
            shp = self
        else:
            shp = deepcopy(self)
            
        # Create a mask to drop the specified regions
        m = np.in1d(shp.regions, region_list)
        # Apply the mask to the shapefile and regions, dropping the specified regions
        shp.shp = shp.shp[~m].reset_index(drop=True)
        shp.regions = shp.regions[~m].reset_index(drop=True)
        # Return the GeoRegions object with the specified regions dropped
        return shp


def georegions_from_path(
    path: str, regionid: str, region_list: Optional[List[str]] = None
) -> "GeoRegions":
    """
    Loads a GeoRegions object from a shapefile.

    Parameters
    ----------
    path : str
        The path to the shapefile.
    regionid : str
        The identifier for the region.
    region_list : list of str, optional
        A list of regions to include (default is None, which means all regions are included).

    Returns
    -------
    GeoRegions
        The loaded GeoRegions object.
    """
    # Read the shapefile from the specified path
    shp = gpd.read_file(path)
    # Create and return a GeoRegions object using the shapefile, region identifier, and optional region list
    return GeoRegions(shp, regionid, region_list)


def georegions_from_name(name="usa", region_list=None):
    """
    Returns a GeoRegions object based on the given name and region list.

    Parameters
    ----------
    name : str
        The name of the GeoRegions object to create. Valid values are "usa", "counties", and "global".
    region_list : list of str, optional
        A list of region names to include in the GeoRegions object. If None, all regions are included.

    Returns
    -------
    GeoRegions
        A GeoRegions object based on the given name and region list.

    Raises
    -------
    NotImplementedError
        If an invalid name is provided.
    """
    # Determine the GeoRegions object to create based on the given name
    if name == "usa":
        return GeoRegions(open_usa_shp(), "state", region_list, name=name)
    elif name == "counties":
        return GeoRegions(open_counties_shp(), "fips", region_list, name=name)
    elif name == "global":
        return GeoRegions(open_global_shp(), "OBJECTID", region_list, name=name)
    else:
        # Raise an error if the name is not supported
        raise NotImplementedError
