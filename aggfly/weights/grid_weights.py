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
import shapely

# import pygeos
from pprint import pprint
from copy import deepcopy

from ..regions import GeoRegions
from ..dataset import Dataset, Grid, array_lon_to_360
from . import CropWeights, PopWeights
from ..utils import *
from ..cache import *
from ..aggregate import is_distributed, shutdown_dask_client, start_dask_client


ZERO_WEIGHT_POLICIES = {"nan", "area", "drop"}


class GridWeights:
    def __init__(
        self,
        grid: Grid,
        georegions: GeoRegions,
        raster_weights: Optional[Union[CropWeights, PopWeights]] = None,
        chunks: int = 30,
        project_dir: Optional[str] = None,
        simplify: Optional[Union[float, int]] = None,
        zero_weight: str = "nan",
        default_to_area_weights: Optional[bool] = None,
        cosine_area: Optional[bool] = None,
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
        zero_weight : {"nan", "area", "drop"}, optional
            What to do with a region whose secondary weights sum to zero — a
            county with no population, or no hectares of the crop in question.

            - ``"nan"`` (default) keeps the region and reports NaN for it. The
              quantity really is undefined there ("the temperature experienced
              by the average person" needs a person), and NaN says so in the
              output instead of hiding it.
            - ``"area"`` falls back to area weights for that region. Note this
              silently mixes two estimands in one column: those rows answer a
              different question from the rest.
            - ``"drop"`` omits the region entirely, so the panel has fewer
              regions than the shapefile with nothing to say which are missing.

            Both non-default policies warn and name the regions affected.
        default_to_area_weights : bool, optional
            Deprecated alias for ``zero_weight``: True maps to ``"area"``,
            False to ``"drop"``.
        cosine_area : bool, optional
            Whether to correct area weights for cell-area distortion by latitude
            (multiply by cos(latitude)).

            Defaults to None, meaning "choose automatically": True for area-only
            weights, False when ``raster_weights`` is given.

            The correction converts a cell's extent in degrees into physical
            area, which is what area weighting wants. A secondary raster such as
            LandScan or WorldPop already reports how many people (or hectares)
            are *in* each cell, so a poleward cell being physically smaller is
            already reflected in its value; applying cos(latitude) on top of it
            counts the same distortion twice and biases regions that span a wide
            range of latitudes.

            Set it explicitly to override: pass True if your secondary raster is
            a density per unit *physical* area (e.g. people per km squared),
            where the conversion is still required.
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

        # Resolve the deprecated boolean onto the three-valued policy.
        if default_to_area_weights is not None:
            warnings.warn(
                "default_to_area_weights is deprecated; use "
                'zero_weight="area" (True) or zero_weight="drop" (False).',
                DeprecationWarning,
                stacklevel=2,
            )
            zero_weight = "area" if default_to_area_weights else "drop"
        if zero_weight not in ZERO_WEIGHT_POLICIES:
            raise ValueError(
                f"zero_weight must be one of {sorted(ZERO_WEIGHT_POLICIES)}, "
                f"got {zero_weight!r}"
            )
        self.zero_weight = zero_weight
        self.verbose = True
        self.weights = None
        self.nonzero_weight_coords = None
        self.nonzero_weight_mask = None
        # Resolve the automatic default. Store the resolved boolean so it lands
        # in cdict() and therefore in the cache key: the two modes must not
        # share a cache entry.
        if cosine_area is None:
            cosine_area = raster_weights is None
        self.cosine_area = cosine_area

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
        # Simplify the geometries (region polygons are a small set; plain geopandas)
        simplified = georegions.shp.geometry.simplify(self.simplify)
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
        # Get the polygon array of the geographical regions with the specified buffer
        poly_array = np.array(self.georegions.poly_array(buffer, chunks=self.chunks))
        poly = gpd.GeoDataFrame(geometry=poly_array)
        if poly.crs is None:
            poly = poly.set_crs(fc.crs)
        # Spatial join: which cell centroids fall within which regions. Plain geopandas
        # (one STRtree over the regions) is faster and lighter than dask-geopandas for
        # this in-memory join at every grid size tested (see benchmarks/bench_sjoin.py).
        mask = gpd.sjoin(fc, poly, predicate="within", how="inner")

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
            # Generate the cells geometry from the border points. A square
            # buffer would be wrong on a non-square grid (it leaves gaps along
            # the wider axis and mis-sizes the cell that area_weight divides
            # by), so build explicit rectangles instead.
            print("Generating cells...")
            pts = border.geometry
            dx = self.grid.resolution_lon / 2
            dy = self.grid.resolution_lat / 2
            cells = gpd.GeoSeries(
                shapely.box(pts.x - dx, pts.y - dy, pts.x + dx, pts.y + dy),
                index=border.index,
                crs=border.crs,
            )
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

        # Line up each border cell with its region polygon (one region per cell); the
        # left merge preserves border's row order, so the two geometry columns are
        # positionally aligned.
        reg = border[["index_right"]].merge(
            self.georegions.shp, how="left", left_on="index_right", right_index=True
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Intersect the border cells with their regions. Border cells are a small
            # set, so this is plain geopandas (no dask). We compare positionally
            # (align=False, values) to reproduce the old partition-wise behaviour and
            # avoid geopandas>=1.0's align=True index alignment, which would form a
            # cartesian product across duplicate region labels.
            print("Intersecting...")
            border_geom = gpd.GeoSeries(border.geometry.values, crs=border.crs)
            reg_geom = gpd.GeoSeries(reg.geometry.values, crs=border.crs)
            inter = border_geom.intersection(reg_geom, align=False)
            # Calculate the area weight for the intersected cells
            print("Calculating area weight")
            area_weight = inter.area / self.grid.cell_area

        # Add the area weights to the border cells GeoDataFrame (positional).
        border["area_weight"] = np.asarray(area_weight)

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

        if self.cosine_area:
            # Correct for cell-area distortion by latitude: a grid cell of fixed
            # degree extent covers less surface area toward the poles (~cos(lat)).
            # No per-region renormalization is needed here — the spatial step
            # divides by each region's summed weight.
            area_weights["area_weight"] *= np.cos(np.radians(area_weights.latitude))

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

        # A cell can end up without a raster value two ways: it falls outside the
        # secondary raster's extent (so the left merge above yields NaN), or its
        # whole footprint is nodata (so the rescaling average has nothing to
        # average). Either way the cell carries none of the weighted quantity, so
        # it contributes zero -- exactly like a cell that really is empty.
        #
        # Coercing here rather than letting the NaN through matters: `sum()` below
        # skips NaN, so `total_weight` would come out finite and `zero_weight`
        # False, while the cell's own `weight` stayed NaN. That NaN then poisoned
        # the spatial sum and silently dropped the entire region from the panel.
        n_missing = int((~np.isfinite(weights["raster_weight"])).sum())
        if n_missing:
            weights = weights.copy()
            weights["raster_weight"] = (
                weights["raster_weight"]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0)
            )
            warnings.warn(
                f"{n_missing} of {len(weights)} cell-region pairs had no secondary "
                "raster value (outside its extent, or entirely nodata) and were "
                "given zero weight. A region with no valid cells at all falls back "
                "to whatever the zero_weight policy specifies.",
                stacklevel=2,
            )

        # Total raster weight per region (a small per-region groupby; plain pandas)
        raster_total = (
            weights[["index_right", "raster_weight"]]
            .groupby("index_right")
            .sum()
            .rename(columns={"raster_weight": "total_weight"})
        )
        # Non-finite totals count as zero so the fallback below still applies.
        raster_total["zero_weight"] = ~(raster_total.total_weight > 0)

        # Merge total raster weights with the weights DataFrame
        tw = weights.merge(
            raster_total, how="left", left_on="index_right", right_index=True
        )

        # Rescale raster weights for non-zero regions
        tw.loc[np.logical_not(tw.zero_weight), ["weight"]] = tw.area_weight * (
            tw.raster_weight / tw.total_weight
        )

        # Regions whose secondary weights sum to zero: no population, or none of
        # the crop in question. What to do about them is a modelling choice, so
        # it is explicit and both non-default policies say what they did.
        zero_regions = sorted(tw.loc[tw.zero_weight, "index_right"].unique())
        if zero_regions:
            shown = zero_regions[:5]
            more = f" (+{len(zero_regions) - 5} more)" if len(zero_regions) > 5 else ""
            if self.zero_weight == "area":
                warnings.warn(
                    f"{len(zero_regions)} region(s) have zero secondary weight and "
                    f"fall back to AREA weights: {shown}{more}. Those rows answer a "
                    "different question from the rest of the panel.",
                    stacklevel=2,
                )
                tw.loc[tw.zero_weight, ["weight"]] = tw.area_weight
            elif self.zero_weight == "drop":
                warnings.warn(
                    f"{len(zero_regions)} region(s) have zero secondary weight and "
                    f"are DROPPED from the output: {shown}{more}.",
                    stacklevel=2,
                )
                tw = tw.loc[np.logical_not(tw.zero_weight)]
            else:  # "nan"
                # Keep the rows at zero weight. The region then has a zero
                # denominator in the spatial step, which already yields NaN --
                # SpatialAggregator just has to not discard that row.
                tw.loc[tw.zero_weight, ["weight"]] = 0.0
        elif self.zero_weight == "drop":
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
            "zero_weight": self.zero_weight,
            "cosine_area": self.cosine_area,
        }

        # Add raster weights to the dictionary if available
        if self.raster_weights is not None:
            gdict["raster_weights"] = clean_object(self.raster_weights)
        else:
            gdict["raster_weights"] = None

        return gdict
    
    def plot_weights(self, region, type='total', log=False, ax=None, legend=False, **kwargs):
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
        # Build cell geometries as rectangles. A square buffer would leave
        # visible gaps between cells whenever the grid is non-square.
        dx = self.grid.resolution_lon / 2
        dy = self.grid.resolution_lat / 2
        pts = plot_df.geometry
        plot_df.geometry = shapely.box(pts.x - dx, pts.y - dy, pts.x + dx, pts.y + dy)
        # print(plot_df)

        # Plot the weights
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        # plot_ds[wvar].plot(ax=ax)

        # if log==True:
        # plot_df[wvar] = np.log(plot_df[wvar]+1)
        plot_df['normalized_weights'] = (plot_df[wvar] / plot_df[wvar].max())
        
        plot_df.plot(ax=ax, column='normalized_weights', alpha=1, legend=legend, **kwargs)
        plot_shp.plot(ax=ax, edgecolor='red', linewidth=2, color='none')
        



def weights_from_objects(
    clim: Dataset,
    georegions: GeoRegions,
    secondary_weights: Optional[Union[CropWeights, PopWeights]] = None,
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
        The secondary weights data (default is None). Build one with
        ``pop_weights_from_path`` / ``crop_weights_from_path`` /
        ``secondary_weights_from_path``.
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

    return GridWeights(
        clim.grid, georegions, secondary_weights, project_dir=project_dir, **kwargs
    )
