# This script defines the GeoRegions class for representing and manipulating geographical regions using shapefiles.
# It includes methods for initializing the GeoRegions object, selecting and dropping regions, generating polygon arrays,
# and plotting region boundaries. Additionally, it provides utility functions for loading GeoRegions from a path or an
# in-memory GeoDataFrame, and for inspecting a vector file before loading it.

import os
from typing import Dict, List, Optional, Union
import numpy as np
import geopandas as gpd
import dask.array  # used by poly_array(datatype="dask")
from copy import deepcopy
import warnings


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
            # Region polygons are a small set; buffer with plain geopandas (no dask).
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                bufferPoly = self.shp.buffer(buffer)
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
        m = np.isin(shp.regions, region_list)
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
        m = np.isin(shp.regions, region_list)
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


def georegions_from_gdf(
    gdf: gpd.GeoDataFrame,
    regionid: str,
    region_list: Optional[List[str]] = None,
    name: Optional[str] = None,
    crs: str = "WGS84",
) -> "GeoRegions":
    """
    Build a GeoRegions object from an in-memory GeoDataFrame.

    The counterpart to :func:`georegions_from_path`, for regions you have
    already loaded, filtered, dissolved or constructed in code — no round trip
    through a file on disk.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        The regions. Must carry a CRS and contain ``regionid`` as a column.
    regionid : str
        Name of the column holding the region identifier. It becomes the key of
        the output panel.
    region_list : list of str, optional
        Restrict to this subset of region ids.
    name : str, optional
        A label for this region set.
    crs : str, optional
        Reproject to this CRS if the frame is in another one (default "WGS84").

    Returns
    -------
    GeoRegions

    Raises
    ------
    TypeError
        If `gdf` is not a GeoDataFrame.
    ValueError
        If it is empty, has no CRS, or `regionid` is not one of its columns.
    """
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError(
            f"georegions_from_gdf expects a GeoDataFrame, got {type(gdf).__name__}. "
            "A plain DataFrame needs a geometry column first: "
            "gpd.GeoDataFrame(df, geometry=..., crs=...)."
        )
    if len(gdf) == 0:
        raise ValueError("georegions_from_gdf: the GeoDataFrame is empty")
    if regionid not in gdf.columns:
        raise ValueError(
            f"regionid {regionid!r} is not a column of the GeoDataFrame. "
            f"Available columns: {sorted(c for c in gdf.columns if c != gdf.geometry.name)}"
        )
    if gdf.crs is None:
        raise ValueError(
            "The GeoDataFrame has no CRS. Set one before building GeoRegions, "
            'e.g. gdf.set_crs("WGS84") if the coordinates are already lon/lat.'
        )

    ids = gdf[regionid]
    if ids.isna().any():
        warnings.warn(
            f"{int(ids.isna().sum())} row(s) have a missing {regionid!r}; "
            "those regions cannot be matched in the output panel.",
            stacklevel=2,
        )
    if ids.duplicated().any():
        dupes = ids[ids.duplicated()].unique()[:5]
        warnings.warn(
            f"{regionid!r} is not unique ({int(ids.duplicated().sum())} repeated "
            f"values, e.g. {list(dupes)}). Rows sharing an id are treated as "
            "separate regions and will produce duplicate panel rows; dissolve "
            "them first if they should be one region.",
            stacklevel=2,
        )

    # Copy so that later in-place operations (e.g. sel(update=True)) never reach
    # back into the caller's frame.
    return GeoRegions(gdf.copy(), regionid, region_list, name=name, crs=crs)


def shapefile_info(path: str, n: int = 5, uniqueness: bool = False) -> Dict:
    """
    Print a summary of a vector file without loading it.

    Intended for working out what to pass as ``regionid`` before committing to a
    full read. Metadata (fields, CRS, feature count, bounds) comes from the file
    header at no I/O cost; only ``n`` rows are actually read.

    Parameters
    ----------
    path : str
        Path to a shapefile, GeoPackage, GeoJSON — anything GDAL can open.
    n : int, optional
        Number of rows to show (default 5). Pass 0 to skip the preview.
    uniqueness : bool, optional
        Also report which columns are unique across **every** feature, which is
        what qualifies a column as a region id. This reads all attributes but no
        geometry, so it is far cheaper than a full read — though not free on a
        large file. Default False.

    Returns
    -------
    dict
        The same information, for programmatic use: ``fields``, ``dtypes``,
        ``crs``, ``features``, ``total_bounds``, ``geometry_type``, ``head``
        and (when requested) ``unique_columns``.
    """
    import pyogrio

    info = pyogrio.read_info(path)
    # These come back as numpy arrays, so `x or []` would raise on the ambiguous
    # truth value of a multi-element array.
    _fields = info.get("fields")
    _dtypes = info.get("dtypes")
    fields = [] if _fields is None else list(_fields)
    dtypes = [] if _dtypes is None else list(_dtypes)
    bounds = info.get("total_bounds")
    crs = info.get("crs")

    out = {
        "path": path,
        "driver": info.get("driver"),
        "layer": info.get("layer_name"),
        "geometry_type": info.get("geometry_type"),
        "features": info.get("features"),
        "crs": crs,
        "total_bounds": bounds,
        "fields": fields,
        "dtypes": dtypes,
        "head": None,
        "unique_columns": None,
    }

    print(f"{path}")
    print(f"  driver     : {info.get('driver')}  layer={info.get('layer_name')}")
    print(f"  geometry   : {info.get('geometry_type')}  features={info.get('features')}")

    if crs:
        print(f"  crs        : {crs}")
    else:
        print("  crs        : NONE — GeoRegions requires a CRS; set one after loading")

    if bounds is not None:
        xmin, ymin, xmax, ymax = bounds
        print(f"  bounds     : lon {xmin:.4f} .. {xmax:.4f} | lat {ymin:.4f} .. {ymax:.4f}")
        # A 0-360 frame needs BOTH a non-negative minimum and a maximum past
        # 180. Testing xmax alone misfires on an ordinary -180..180 file, whose
        # xmax lands a hair above 180 through floating point.
        if xmin >= 0 and xmax > 180:
            print("               longitudes run 0–360, not -180–180")

    if not fields:
        print("  fields     : none — this file has no attribute table, so there is")
        print("               no column to use as regionid")
    else:
        print(f"  fields     : {len(fields)}")
        for f, d in zip(fields, dtypes):
            print(f"      {f:<24} {d}")

    if n and fields:
        head = gpd.read_file(path, rows=n)
        geom_col = head.geometry.name if head.geometry is not None else None
        preview = head.drop(columns=[geom_col]) if geom_col in head.columns else head
        out["head"] = preview
        print(f"  first {min(n, len(head))} row(s) (geometry omitted):")
        for line in preview.to_string().splitlines():
            print(f"      {line}")

    if uniqueness and fields:
        attrs = pyogrio.read_dataframe(path, read_geometry=False)
        unique = [
            c for c in attrs.columns
            if attrs[c].notna().all() and not attrs[c].duplicated().any()
        ]
        out["unique_columns"] = unique
        if unique:
            print(f"  unique across all {len(attrs)} features (regionid candidates):")
            print(f"      {', '.join(unique)}")
        else:
            print("  no column is unique across all features — none can serve as a")
            print("  regionid on its own")

    return out
