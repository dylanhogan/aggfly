# This script defines the GridWeights class for calculating spatial weights over a geographical grid. 
# It includes methods to calculate weights, simplify polygons, mask grids, and handle area-based weights. 
# Additionally, utility functions are provided for creating GridWeights objects from various data sources.

import os
import warnings
from functools import lru_cache
from typing import Optional, Union, List, Dict

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

# import pygeos
import dask
import dask.array
import dask_geopandas
from pprint import pprint
from copy import deepcopy

from ..regions import GeoRegions
from ..dataset import Dataset, Grid, array_lon_to_360
from . import CropWeights, PopWeights
from . import crop_weights, pop_weights
from ..utils import *
from ..cache import *
from ..aggregate import is_distributed, shutdown_dask_client, start_dask_client


class GridWeights:
    def __init__(
        self,
        grid: Grid,
        georegions: GeoRegions,
        raster_weights: Optional[Union[CropWeights, PopWeights]] = None,
        chunks: int = 30,
        project_dir: Optional[str] = None,
        simplify: Optional[Union[float, int]] = None,
        default_to_area_weights: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize a GridWeights object.

        Parameters
        ----------
        grid : Grid
            The grid to use.
        georegions : GeoRegions
            The georegions to use.
        raster_weights : RasterWeights, optional
            The raster weights to use (default is None).
        chunks : int, optional
            The number of chunks to use (default is 30).
        project_dir : str, optional
            The project directory (default is None).
        simplify : float or int, optional
            The simplification factor to use (default is None).
        default_to_area_weights : bool, optional
            Whether to default to area weights (default is True).
        verbose : bool, optional
            Whether to print verbose output (default is True).
        weights : GeoDataFrame
            The calculated weights.
        """
        self.grid = grid
        assert not self.grid.lon_is_360  # Ensure longitude is not in 360-degree format
        self.georegions = georegions
        self.raster_weights = raster_weights
        self.chunks = chunks
        self.project_dir = project_dir
        self.simplify = simplify
        self.default_to_area_weights = default_to_area_weights
        self.verbose = True
        self.weights = None
        self.nonzero_weight_coords = None
        self.nonzero_weight_mask = None
        
        # Initialize the cache
        self.cache = initialize_cache(self)

    def calculate_weights(self):
        """
        Calculate the weights for the grid.
        """
        # Create a dictionary to store function and parameters
        gdict = {"func": "weights"}

        # Check if a Dask client is running and shut it down for weight calculation
        if is_distributed():
            print("Dask client detected, which is not compatible with weight calculation.")
            print("Stopping Dask client for weight calculation")
            dask_args = shutdown_dask_client()
            restart = True
        else:
            restart = False
            
        # Simplify the polygon array if a simplification factor is provided
        if self.simplify is not None:
            self.simplify_poly_array()

        # Load raster weights if provided, ensuring CRS matches between raster and georegions
        if self.raster_weights is not None:
            assert self.raster_weights.raster.rio.crs == self.georegions.shp.crs, "CRS mismatch"
            self.raster_weights.rescale_raster_to_grid(self.grid, verbose=self.verbose)
            gdict["raster_weights"] = self.raster_weights.cdict()
        else:
            gdict["raster_weights"] = None

        # Check if weights are cached and load them if available
        if self.cache is not None:
            cache = self.cache.uncache(gdict, extension=".feather")
        else:
            cache = None
            
        # Load weights from cache if available, otherwise calculate them
        if cache is not None:
            print(f"Loading rescaled weights from cache")
            if self.verbose:
                print("Cache dictionary:")
                pprint(gdict)
            self.weights = cache
        else:
            # Calculate weights based on area or raster weights
            if self.raster_weights is None:
                w = self.get_area_weights()
                w["weight"] = w["area_weight"]
            else:
                w = self.get_weighted_area_weights()
            # Cache the calculated weights if caching is enabled
            if self.cache is not None:
                self.cache.cache(w, gdict, extension=".feather")
            self.weights = w

        # Merge weights with georegions shapefile
        self.weights = self.georegions.shp[[self.georegions.regionid]].merge(
            self.weights, right_on='index_right', left_index=True
        )
        
        # Identify non-zero weights and create a mask
        nonzero_weights = np.isin(self.grid.index, self.weights.cell_id)
        self.nonzero_weight_coords = nonzero_weights.nonzero()
        self.nonzero_weight_mask = xr.DataArray(
            data=nonzero_weights,
            dims=["latitude", "longitude"],
            coords={
                "latitude": ("latitude", self.grid.latitude.values),
                "longitude": ("longitude", self.grid.longitude.values),
            },
        )
        
        # Restart Dask client if it was previously shut down
        if restart:
            print('Restarting distributed Dask client')
            client = start_dask_client(**dask_args)


    @lru_cache(maxsize=None)
    def simplify_poly_array(self):
        """
        Simplify the polygon array.
    
        This method simplifies the polygon geometries of the geographical regions
        for faster processing.
    
        Steps:
        - Deepcopy the georegions to preserve the original data.
        - Use Dask to simplify the geometries in parallel.
        - Update the georegions with the simplified geometries.
        """
        # Create a deepcopy of the georegions to preserve the original data
        georegions = deepcopy(self.georegions)
        # Simplify the geometries in parallel using Dask
        simplified = (
            dask_geopandas.from_geopandas(georegions.shp, npartitions=30)
            .simplify(self.simplify)
            .compute()
        )
        # Update the geometries of the georegions with the simplified geometries
        georegions.shp["geometry"] = simplified
        self.georegions = georegions

    @lru_cache(maxsize=None)
    def mask(self, buffer: int = 0) -> gpd.GeoDataFrame:
        """
        Mask the grid based on the geographical regions.
    
        Parameters
        ----------
        buffer : int, optional
            The buffer size around the regions (default is 0).
    
        Returns
        -------
        geopandas.GeoDataFrame
            The masked grid containing only the regions of interest.
        """
        # Get the centroids of the grid cells
        centroids = self.grid.centroids()
        # Create a GeoDataFrame of the centroids with the specified CRS
        fc = gpd.GeoDataFrame(geometry=centroids.flatten()).set_crs("EPSG:4326")
        # Convert the GeoDataFrame to a Dask GeoDataFrame for parallel processing
        fc = dask_geopandas.from_geopandas(fc, npartitions=self.chunks)
        # Get the polygon array of the geographical regions with the specified buffer
        poly_array = np.array(self.georegions.poly_array(buffer, chunks=self.chunks))
        # Convert the polygon array to a Dask GeoDataFrame
        poly_array = dask_geopandas.from_geopandas(
            gpd.GeoDataFrame(geometry=poly_array), npartitions=10
        )
        # Perform a spatial join to find the centroids within the polygons and compute the result
        mask = fc.sjoin(poly_array, predicate="within").compute()
        
        return mask

    def get_border_cells(
        self, buffers: Optional[List[float]] = None
    ) -> gpd.GeoDataFrame:
        """
        Get the border cells of the grid.
    
        Parameters
        ----------
        buffers : list of float, optional
            The buffer sizes to define the border (default is None).
    
        Returns
        -------
        geopandas.GeoDataFrame
            The border cells of the grid.
        """
        print("Searching for border cells...")
        if buffers is None:
            # Default buffer sizes to define the border
            buffers = (-self.grid.resolution, self.grid.resolution)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Mask the grid with a negative buffer to find the inner cells
            print("Negative buffer")
            m1 = self.mask(buffer=buffers[0])
            # Mask the grid with a positive buffer to find the outer cells
            print("Positive buffer")
            m2 = self.mask(buffer=buffers[1])

        # Add cell_id to the positive buffer mask
        m2["cell_id"] = m2.index.values
        # Merge the two masks to find the border cells (cells in the positive buffer but not in the negative buffer)
        m = m2.merge(m1, how="outer", indicator=True)
        border = m.loc[m._merge == "left_only"].drop(columns="_merge")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Generate the cells geometry by buffering the border points
            print("Generating cells...")
            cells = border.buffer(self.grid.resolution / 2, cap_style=3)
        border["geometry"] = cells
        return border

    def intersect_border_cells(self) -> gpd.GeoDataFrame:
        """
        Intersect the border cells of the grid.

        Returns
        -------
        geopandas.GeoDataFrame
            The intersected border cells.
        """
        # Get the border cells
        border = self.get_border_cells()

        # Merge the border cells with the geographical regions shapefile
        reg = border[["index_right"]].merge(
            self.georegions.shp, how="left", left_on="index_right", right_index=True
        )
        # Convert the merged GeoDataFrame to a Dask GeoDataFrame
        reg = dask_geopandas.from_geopandas(
            gpd.GeoDataFrame(reg, geometry="geometry"), npartitions=1
        )
        dgb = dask_geopandas.from_geopandas(
            gpd.GeoDataFrame(border), npartitions=self.chunks
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Intersect the border cells with the geographical regions
            print("Intersecting...")
            inter = dgb.geometry.intersection(reg.geometry).compute()
            # Calculate the area weight for the intersected cells
            print("Calculating area weight")
            inter = inter.area / self.grid.cell_area

        # Add the area weights to the border cells GeoDataFrame
        border["area_weight"] = inter

        return border

    def get_cell_id_dataframe(self) -> pd.DataFrame:
        """
        Get a DataFrame with the cell IDs.

        Returns
        -------
        pandas.DataFrame
            The DataFrame with the cell IDs.
        """
        # Create a DataFrame with cell IDs, longitudes, and latitudes from the grid
        return pd.DataFrame(
            {
                "cell_id": self.grid.index.flatten(), # Flattened array of cell IDs
                "longitude": self.grid.lon_array.flatten(), # Flattened array of longitudes
                "latitude": self.grid.lat_array.flatten(), # Flattened array of latitudes
            }
        )

    def get_area_weights(self) -> pd.DataFrame:
        """
        Get the area weights of the grid.

        Returns
        -------
        pandas.DataFrame
            The DataFrame with the area weights.
        """
        # Intersect border cells to get area weights for border cells
        border_cells = self.intersect_border_cells()

        # Mask the grid with a negative buffer to get interior cells
        interior_cells = self.mask(buffer=-self.grid.resolution)
        interior_cells = interior_cells.reset_index().rename(
            columns={"index": "cell_id"}
        )
        interior_cells["area_weight"] = 1 # Set area weight to 1 for interior cells

        # Concatenate interior and border cells
        area_weights = pd.concat(
            [
                interior_cells.drop(columns="geometry"),
                border_cells.drop(columns="geometry"),
            ],
            axis=0,
        )

        # Keep only cells with positive area weights
        area_weights = area_weights.loc[area_weights.area_weight > 0]

        # Merge area weights with cell ID DataFrame
        cell_df = self.get_cell_id_dataframe()
        area_weights = area_weights.merge(cell_df, how="left", on="cell_id")

        return area_weights

    def get_weighted_area_weights(self) -> pd.DataFrame:
        """
        Get the weighted area weights of the grid.
    
        Returns
        -------
        pandas.DataFrame
            The DataFrame with the weighted area weights.
        """
        # Get the area weights
        area_weights = self.get_area_weights()

        # Convert raster weights to a DataFrame
        raster_weights = self.raster_weights.raster.to_dataframe(name="raster_weight")

        # Merge area weights with raster weights
        weights = area_weights.merge(
            raster_weights,
            how="left",
            on=["latitude", "longitude"]
        )

        # Parallelize the weights DataFrame using Dask
        dw = dask.dataframe.from_pandas(weights, npartitions=self.chunks)

        # Check total raster weight per region
        raster_total = (
            dw[["index_right", "raster_weight"]]
            .groupby("index_right")
            .sum()
            .rename(columns={"raster_weight": "total_weight"})
        )
        raster_total["zero_weight"] = raster_total.total_weight == 0 # Identify regions with zero total weight

        # Merge total raster weights with the weights DataFrame
        tw = dw.merge(
            raster_total, how="left", left_on="index_right", right_index=True
        ).compute()

        # Rescale raster weights for non-zero regions
        tw.loc[np.logical_not(tw.zero_weight), ["weight"]] = tw.area_weight * (
            tw.raster_weight / tw.total_weight
        )

        # Default to area weights for regions with zero raster weight if indicated
        if self.default_to_area_weights:
            tw.loc[tw.zero_weight, ["weight"]] = tw.area_weight
        else:
            tw = tw.loc[np.logical_not(tw.zero_weight)]
        
        return tw

    def cdict(self) -> Dict:
        """
        Get a dictionary representation of the GridWeights object.

        Returns
        -------
        dict
            The dictionary representation of the GridWeights object.
        """
        # Create a dictionary representation of the GridWeights object
        gdict = {
            "grid": clean_object(self.grid),
            "georegions": {
                "regions": str(self.georegions.regions),
                "geometry": str(self.georegions.shp.geometry),
            },
            "simplify": self.simplify,
            "default_to_area_weights": self.default_to_area_weights,
        }

        # Add raster weights to the dictionary if available
        if self.raster_weights is not None:
            gdict["raster_weights"] = clean_object(self.raster_weights)
        else:
            gdict["raster_weights"] = None

        return gdict
    
    def plot_weights(self, region, type='total', **kwargs):
        """
        Plot the weights for a specific region.
    
        Parameters
        ----------
        region : str
            The region to plot.
        type : str, optional
            The type of weights to plot ('total', 'secondary', or 'area').
        """
        # Determine the weight variable to plot based on the type
        if type == 'total':
            wvar = 'weight'
        elif type == 'secondary':
            wvar = 'raster_weight'
        elif type == 'area':
            wvar = 'area_weight'
        else:
            raise NotImplementedError
        
        import matplotlib.pyplot as plt
        # Filter the weights DataFrame for the specified region
        plot_df = self.weights.loc[
                self.weights[self.georegions.regionid].isin([region])
            ]
        # Filter the geographical regions shapefile for the specified region
        plot_shp = self.georegions.shp.loc[
            self.georegions.shp[self.georegions.regionid].isin([region])
        ]
        # plot_ds = (plot_df[['latitude', 'longitude', wvar]]
        #     .set_index(['latitude', 'longitude'])
        #     .to_xarray()
        # )
        # Create a GeoDataFrame for plotting
        plot_df = gpd.GeoDataFrame(
            plot_df[['latitude', 'longitude', wvar]],
            geometry=gpd.points_from_xy(plot_df.longitude, plot_df.latitude)
        )
        # Buffer the points to create cell geometries
        plot_df.geometry = plot_df.buffer(self.grid.resolution / 2, cap_style=3)
        # print(plot_df)

        # Plot the weights
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        # plot_ds[wvar].plot(ax=ax)

        plot_shp.plot(ax=ax, edgecolor='black', linewidth=2, color='none')
        plot_df.plot(ax=ax, column=wvar, alpha=0.75, legend=True)
        



def weights_from_objects(
    clim: Dataset,
    georegions: GeoRegions,
    secondary_weights: Optional[Union[CropWeights, PopWeights]] = None,
    wtype: str = None,
    name: str = None,
    crop: Optional[str] = "corn",
    feed: Optional[str] = None,
    write: bool = False,
    project_dir: Optional[str] = None,
    **kwargs,
) -> GridWeights:
    """
    Create a GridWeights object from given objects.

    Parameters
    ----------
    clim : Climate
        The climate data.
    georegions : GeoRegions
        The georegions data.
    secondary_weights : Union[CropWeights, PopWeights], optional
        The secondary weights data (default is None).
    wtype : str, optional
        The weight type (default is "crop").
    name : str, optional
        The name (default is "cropland").
    crop : str, optional
        The crop type (default is "corn").
    feed : str, optional
        The feed type, rainfed, irrigated, or total (default is None).
    write : bool, optional
        Whether to write the data (default is False).
    project_dir : str, optional
        The project directory (default is None).

    Returns
    -------
    GridWeights
        The created GridWeights object.
    """
    if clim.lon_is_360:
        clim = deepcopy(clim)
        clim.rescale_longitude()

    if secondary_weights is None:
        if wtype == "crop":
            if crop is not None:
                secondary_weights = crop_weights.from_name(
                    name=name,
                    crop=crop,
                    feed=feed,
                    write=write,
                    project_dir=project_dir,
                )
            else:
                raise NotImplementedError
        elif wtype == "pop":
            secondary_weights = pop_weights.from_name(
                name=name, write=write, project_dir=project_dir
            )
        else:
            secondary_weights = None

    return GridWeights(
        clim.grid, georegions, secondary_weights, project_dir=project_dir, **kwargs
    )
