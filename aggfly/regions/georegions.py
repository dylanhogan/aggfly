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
        """
        try: 
            shp.crs
            if crs != shp.crs:
                print(f"Converting shapefile CRS to {crs}")
                shp = shp.to_crs(crs)   
        except:
            raise ValueError('Shapefile does not have a CRS assigned to it')
        self.shp = shp.reset_index(drop=True)
        self.regionid = regionid
        self.regions = self.shp[self.regionid]
        if region_list is not None:
            self.sel(region_list, update=True)
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
        if buffer != 0:
            ddf = dask_geopandas.from_geopandas(self.shp, npartitions=chunks)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                bufferPoly = ddf.buffer(buffer).compute()
        else:
            bufferPoly = self.shp.geometry

        if datatype == "dask":
            ar = dask.array.from_array(
                bufferPoly, chunks=int(len(bufferPoly) / chunks)
            ).reshape(len(bufferPoly), 1, 1)
            return ar
        elif datatype == "array":
            return bufferPoly
        else:
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
        geo = self.shp.loc[self.regions == region].geometry
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
        """
        region_list = (
            [region_list] if not isinstance(region_list, list) else region_list
        )
        if update:
            shp = self
        else:
            shp = deepcopy(self)

        m = np.in1d(shp.regions, region_list)
        shp.shp = shp.shp[m].reset_index(drop=True)
        shp.regions = shp.regions[m].reset_index(drop=True)
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
        """
        region_list = (
            [region_list] if not isinstance(region_list, list) else region_list
        )
        if update:
            shp = self
        else:
            shp = deepcopy(self)
        
        m = np.in1d(shp.regions, region_list)
        shp.shp = shp.shp[~m].reset_index(drop=True)
        shp.regions = shp.regions[~m].reset_index(drop=True)
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
    
    shp = gpd.read_file(path)
    return GeoRegions(shp, regionid, region_list)


def georegions_from_name(name="usa", region_list=None):
    """
    Returns a GeoRegions object based on the given name and region list.

    Args:
        name (str): The name of the GeoRegions object to create. Valid values are "usa", "counties", and "global".
        region_list (list): A list of region names to include in the GeoRegions object. If None, all regions are included.

    Returns:
        GeoRegions: A GeoRegions object based on the given name and region list.

    Raises:
        NotImplementedError: If an invalid name is provided.
    """
    if name == "usa":
        return GeoRegions(open_usa_shp(), "state", region_list, name=name)
    elif name == "counties":
        return GeoRegions(open_counties_shp(), "fips", region_list, name=name)
    elif name == "global":
        return GeoRegions(open_global_shp(), "OBJECTID", region_list, name=name)
    else:
        raise NotImplementedError
