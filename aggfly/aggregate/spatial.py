"""
SpatialAggregator class that aggregates climate data over regions using weights.

Attributes:
    clim (list): List of xarray DataArrays containing climate data.
    weights (xarray Dataset): Dataset containing region weights.
    names (list): List of names for the climate data variables.

Methods:
    compute(npartitions=30): Aggregates climate data over regions using weights.
    weighted_average(ddf, names): Computes weighted average of climate data.
"""
import warnings

import xarray as xr
import dask
from dask.distributed import progress

from .aggregate_utils import distributed_client, is_distributed
from ..dataset import Dataset
from ..weights import GridWeights

import logging
from typing import List, Union
import dask.dataframe
import pandas as pd
import xarray as xr

class SpatialAggregator:
    """
    A class for spatially aggregating climate data using weights.

    Parameters:
    -----------
    clim : list or Dataset
        A list of Dataset objects containing climate data to be aggregated.
    weights : GridWeights 
        A GridWeights object containing the weights to be used for aggregation.
    names : str or list of str, optional
        The name(s) of the climate variable(s) to be aggregated. Default is "climate".

    Methods:
    --------
    compute(npartitions: int = 30) -> pd.DataFrame:
        Compute the spatial aggregation.

    weighted_average(ddf: dask.dataframe.DataFrame, names: List[str]) -> pd.DataFrame:
        Compute the weighted average of the climate data.

    """
    def __init__(self, dataset: Union[list, Dataset], weights: GridWeights, names: Union[str, List[str]] = "climate") -> None:
        """
        Initialize a SpatialAggregator object.

        Parameters
        ----------
        dataset : list or Dataset
            A list of Dataset objects, each containing the climate data
            for a different temporal aggregation. Alternatively, a single Dataset
            object can be passed.
        weights : GridWeights
            A GridWeights object containing the spatial weights used to
            aggregate the climate data.
        names : str or list of str, optional
            The name(s) of the climate variable(s) being aggregated. If a single
            variable is being aggregated, a string can be passed. If multiple
            variables are being aggregated, a list of strings can be passed.

        Returns
        -------
        None

        """
        if type(dataset) != list:
            self.dataset = [dataset]
        else:
            self.dataset = dataset
        _ = [x.rescale_longitude() for x in self.dataset if x.lon_is_360]
        self.grid = weights.grid
        self.weights = weights.weights
        self.names = [names] if isinstance(names, str) else names

    def compute(self, npartitions: int = 30) -> pd.DataFrame:
        """
        Compute the weighted average of the climate data over the regions defined by the weights.

        Parameters:
        -----------
        npartitions : int, optional
            The number of partitions to use for the Dask DataFrame. Default is 30.

        Returns:
        --------
        aggregated : pandas.DataFrame
            A DataFrame containing the weighted average of the climate data over the regions and time periods.
        """
        # with dask.config.set({"multiprocessing.context": "forkserver"}):
        print("Computing...")
        clim_ds = dask.compute([x.da for x in self.dataset])[0] #, scheduler='processes'
        
        print("Combining datasets...")
        clim_ds = xr.combine_by_coords(
            [x.to_dataset(name=self.names[i]) for i, x in enumerate(clim_ds)]
        )

        print("Stacking...")
        clim_df = (
            clim_ds.stack({"cell_id": ["latitude", "longitude"]})
            .drop_vars(["cell_id", "latitude", "longitude"])
            .assign_coords(coords={"cell_id": ("cell_id", self.dataset[0].grid.cell_id)})
            .to_dataframe()
            .reset_index("time")
            .dropna(subset=self.names)
        )

        print("Merging...")
        self.weights["region_id"] = self.weights.index_right
        merged_df = clim_df.merge(self.weights, how="inner", on="cell_id")
        merged_df = merged_df.dropna(subset=self.names)

        print("Grouping...")
        group_key = (
            merged_df[["region_id", "time"]]
            .drop_duplicates()
            .reset_index(drop=True)
            .reset_index()
            .rename(columns={"index": "group_ID"})
        )
        
        print("Merging again...")
        merged_df = merged_df.merge(group_key, on=["region_id", "time"]).set_index(
            "group_ID"
        )[["weight", *self.names]]

        print("Creating Dask DataFrame...")
        ddf = dask.dataframe.from_pandas(merged_df, npartitions=50)
        
        print("Aggregating...")
        out = self.weighted_average(ddf, self.names).compute()
        aggregated = (
            out.merge(group_key, how="right", left_index=True, right_on="group_ID")
            .drop(columns="group_ID")[["region_id", "time"] + self.names]
            .reset_index(drop=True)
        )

        return aggregated
    
    @staticmethod
    def weighted_average(ddf: dask.dataframe.DataFrame, names: List[str]) -> pd.DataFrame:
        out = ddf[names].mul(ddf["weight"], axis=0)
        out["weight"] = ddf["weight"]
        out = out.groupby(out.index).sum()
        out = out[names].div(out["weight"], axis=0)
        return out
