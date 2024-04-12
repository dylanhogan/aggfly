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
        """
        self.grid = grid
        assert not self.grid.lon_is_360
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

        self.cache = initialize_cache(self)

    def calculate_weights(self):
        """
        Calculate the weights for the grid.
        """
        gdict = {"func": "weights"}

        if is_distributed():
            print("Dask client detected, which is not compatible with weight calculation.")
            print("Stopping Dask client for weight calculation")
            dask_args = shutdown_dask_client()
            restart = True
        else:
            restart = False
        
        if self.simplify is not None:
            self.simplify_poly_array()

        # Load raster weights if needed
        if self.raster_weights is not None:
            assert self.raster_weights.raster.rio.crs == self.georegions.shp.crs, "CRS mismatch"
            self.raster_weights.rescale_raster_to_grid(self.grid, verbose=self.verbose)
            gdict["raster_weights"] = self.raster_weights.cdict()
        else:
            gdict["raster_weights"] = None

        # Check to see if file is cached
        if self.cache is not None:
            cache = self.cache.uncache(gdict, extension=".feather")
        else:
            cache = None

        if cache is not None:
            print(f"Loading rescaled weights from cache")
            if self.verbose:
                print("Cache dictionary:")
                pprint(gdict)
            self.weights = cache
        else:
                
            if self.raster_weights is None:
                w = self.get_area_weights()
                w["weight"] = w["area_weight"]
            else:
                w = self.get_weighted_area_weights()
            if self.cache is not None:
                self.cache.cache(w, gdict, extension=".feather")
            self.weights = w

        self.weights = self.georegions.shp[[self.georegions.regionid]].merge(
            self.weights, right_on='index_right', left_index=True
        )
        
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
        
        if restart:
            print('Restarting distributed Dask client')
            client = start_dask_client(**dask_args)


    @lru_cache(maxsize=None)
    def simplify_poly_array(self):
        """
        Simplify the polygon array.
        """
        georegions = deepcopy(self.georegions)
        simplified = (
            dask_geopandas.from_geopandas(georegions.shp, npartitions=30)
            .simplify(self.simplify)
            .compute()
        )
        georegions.shp["geometry"] = simplified
        self.georegions = georegions

    @lru_cache(maxsize=None)
    def mask(self, buffer: int = 0) -> gpd.GeoDataFrame:
        """
        Mask the grid.

        Parameters
        ----------
        buffer : int, optional
            The buffer size (default is 0).

        Returns
        -------
        geopandas.GeoDataFrame
            The masked grid.
        """

        centroids = self.grid.centroids()
        fc = gpd.GeoDataFrame(geometry=centroids.flatten()).set_crs("EPSG:4326")
        fc = dask_geopandas.from_geopandas(fc, npartitions=self.chunks)
        poly_array = np.array(self.georegions.poly_array(buffer, chunks=self.chunks))
        poly_array = dask_geopandas.from_geopandas(
            gpd.GeoDataFrame(geometry=poly_array), npartitions=10
        )
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
            The buffer sizes (default is None).

        Returns
        -------
        geopandas.GeoDataFrame
            The border cells.
        """
        print("Searching for border cells...")
        if buffers is None:
            buffers = (-self.grid.resolution, self.grid.resolution)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            print("Negative buffer")
            m1 = self.mask(buffer=buffers[0])
            print("Positive buffer")
            m2 = self.mask(buffer=buffers[1])

        m2["cell_id"] = m2.index.values
        m = m2.merge(m1, how="outer", indicator=True)
        border = m.loc[m._merge == "left_only"].drop(columns="_merge")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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
        border = self.get_border_cells()

        reg = border[["index_right"]].merge(
            self.georegions.shp, how="left", left_on="index_right", right_index=True
        )
        reg = dask_geopandas.from_geopandas(
            gpd.GeoDataFrame(reg, geometry="geometry"), npartitions=1
        )
        dgb = dask_geopandas.from_geopandas(
            gpd.GeoDataFrame(border), npartitions=self.chunks
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            print("Intersecting...")
            inter = dgb.geometry.intersection(reg.geometry).compute()
            print("Calculating area weight")
            inter = inter.area / self.grid.cell_area

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
        return pd.DataFrame(
            {
                "cell_id": self.grid.index.flatten(),
                "longitude": self.grid.lon_array.flatten(),
                "latitude": self.grid.lat_array.flatten(),
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
        border_cells = self.intersect_border_cells()

        interior_cells = self.mask(buffer=-self.grid.resolution)
        interior_cells = interior_cells.reset_index().rename(
            columns={"index": "cell_id"}
        )
        interior_cells["area_weight"] = 1

        area_weights = pd.concat(
            [
                interior_cells.drop(columns="geometry"),
                border_cells.drop(columns="geometry"),
            ],
            axis=0,
        )

        area_weights = area_weights.loc[area_weights.area_weight > 0]

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
        area_weights = self.get_area_weights()

        raster_weights = self.raster_weights.raster.to_dataframe(name="raster_weight")

        weights = area_weights.merge(
            raster_weights,
            how="left",
            on=["latitude", "longitude"]
        )

        # Parallelize
        dw = dask.dataframe.from_pandas(weights, npartitions=self.chunks)

        # Check for weight totals
        raster_total = (
            dw[["index_right", "raster_weight"]]
            .groupby("index_right")
            .sum()
            .rename(columns={"raster_weight": "total_weight"})
        )
        raster_total["zero_weight"] = raster_total.total_weight == 0

        # Merge raster totals
        tw = dw.merge(
            raster_total, how="left", left_on="index_right", right_index=True
        ).compute()

        # Rescale raster weights
        tw.loc[np.logical_not(tw.zero_weight), ["weight"]] = tw.area_weight * (
            tw.raster_weight / tw.total_weight
        )

        # Default to area weights for places with zero raster weight if indicated
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
        gdict = {
            "grid": clean_object(self.grid),
            "georegions": {
                "regions": str(self.georegions.regions),
                "geometry": str(self.georegions.shp.geometry),
            },
            "simplify": self.simplify,
            "default_to_area_weights": self.default_to_area_weights,
        }

        if self.raster_weights is not None:
            gdict["raster_weights"] = clean_object(self.raster_weights)
        else:
            gdict["raster_weights"] = None

        return gdict
    
    def plot_weights(self, region, type='total', **kwargs):
        """
        Plot the weights.
        """
        
        if type == 'total':
            wvar = 'weight'
        elif type == 'secondary':
            wvar = 'raster_weight'
        elif type == 'area':
            wvar = 'area_weight'
        else:
            raise NotImplementedError
        
        import matplotlib.pyplot as plt
        plot_df = self.weights.loc[
                self.weights[self.georegions.regionid].isin([region])
            ]
        plot_shp = self.georegions.shp.loc[
            self.georegions.shp[self.georegions.regionid].isin([region])
        ]
        # plot_ds = (plot_df[['latitude', 'longitude', wvar]]
        #     .set_index(['latitude', 'longitude'])
        #     .to_xarray()
        # )
        plot_df = gpd.GeoDataFrame(
            plot_df[['latitude', 'longitude', wvar]],
            geometry=gpd.points_from_xy(plot_df.longitude, plot_df.latitude)
        )
        plot_df.geometry = plot_df.buffer(self.grid.resolution / 2, cap_style=3)
        # print(plot_df)
        
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
