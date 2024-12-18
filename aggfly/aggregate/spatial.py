# This script defines the SpatialAggregator class, which is used to aggregate climate data over regions using weights.
# The class includes methods to compute the spatial aggregation and to compute weighted averages of the data.


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
        # Ensure dataset is a list of Dataset objects
        if type(dataset) != list:
            self.dataset = [dataset]
        else:
            self.dataset = dataset
        
        # Rescale longitude for datasets if necessary
        _ = [x.rescale_longitude() for x in self.dataset if x.lon_is_360]
        
        # Assign grid and weights attributes
        self.grid = weights.grid
        self.weights = weights.weights
        
        # Ensure names is a list
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
        # Begin computation process
        print("Computing...")
        # Compute Dask arrays for the dataset
        clim_ds = dask.compute([x.da for x in self.dataset])[0] #, scheduler='processes'
        
        # Combine datasets by coordinates
        print("Combining datasets...")
        clim_ds = xr.combine_by_coords(
            [x.to_dataset(name=self.names[i]) for i, x in enumerate(clim_ds)]
        )

        # Stack the dataset to form a DataFrame with cell_id as the index
        print("Stacking...")
        clim_df = (
            clim_ds.stack({"cell_id": ["latitude", "longitude"]})
            .drop_vars(["cell_id", "latitude", "longitude"])
            .assign_coords(coords={"cell_id": ("cell_id", self.dataset[0].grid.cell_id)})
            .to_dataframe()
            .reset_index("time")
            .dropna(subset=self.names)
        )

        # Merge the climate data with the weights data
        print("Merging...")
        self.weights["region_id"] = self.weights.index_right
        merged_df = clim_df.merge(self.weights, how="inner", on="cell_id")
        merged_df = merged_df.dropna(subset=self.names)

        # Group data by region_id and time, creating a unique group_ID for each group
        print("Grouping...")
        group_key = (
            merged_df[["region_id", "time"]]
            .drop_duplicates()
            .reset_index(drop=True)
            .reset_index()
            .rename(columns={"index": "group_ID"})
        )
        
        # Merge the grouped data back into the main DataFrame
        print("Merging again...")
        merged_df = merged_df.merge(group_key, on=["region_id", "time"]).set_index(
            "group_ID"
        )[["weight", *self.names]]

        # Convert the merged DataFrame to a Dask DataFrame
        print("Creating Dask DataFrame...")
        ddf = dask.dataframe.from_pandas(merged_df, npartitions=50)
        
        # Compute the weighted average of the climate data
        print("Aggregating...")
        out = self.weighted_average(ddf, self.names).compute()
        
        # Merge the aggregated data with the group keys and format the final DataFrame
        aggregated = (
            out.merge(group_key, how="right", left_index=True, right_on="group_ID")
            .drop(columns="group_ID")[["region_id", "time"] + self.names]
            .reset_index(drop=True)
        )

        return aggregated
    
    @staticmethod
    def weighted_average(ddf: dask.dataframe.DataFrame, names: List[str]) -> pd.DataFrame:
        """
        Compute the weighted average of the specified columns in the Dask DataFrame.
    
        Parameters:
        -----------
        ddf : dask.dataframe.DataFrame
            The Dask DataFrame containing the data to be aggregated.
        names : list of str
            The names of the columns to compute the weighted average for.
    
        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the weighted averages of the specified columns.
        """

        # Multiply the specified columns by the weight column
        out = ddf[names].mul(ddf["weight"], axis=0)
        # Add the weight column to the result
        out["weight"] = ddf["weight"]
        # Group by the index and sum the groups
        out = out.groupby(out.index).sum()
        # Divide the summed columns by the total weight to get the weighted average
        out = out[names].div(out["weight"], axis=0)
        return out
