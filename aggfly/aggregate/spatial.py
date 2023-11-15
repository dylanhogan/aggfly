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
from ..dataset.dataset import Dataset


class SpatialAggregator:
    """
    A class for spatially aggregating climate data using weights.

    Parameters:
    -----------
    clim : list or xarray.DataArray
        A list of xarray.DataArray objects containing climate data to be aggregated.
    weights : xarray.Dataset
        A xarray.Dataset object containing the weights to be used for aggregation.
    names : str or list of str, optional
        The name(s) of the climate variable(s) to be aggregated. Default is "climate".

    Methods:
    --------
    compute(npartitions=30)
        Compute the spatial aggregation.

    weighted_average(ddf, names)
        Compute the weighted average of the climate data.

    """
    def __init__(self, clim, weights, names="climate"):
        """
        Initialize a SpatialAggregator object.

        Parameters
        ----------
        clim : list or Dataset
            A list of Dataset objects, each containing the climate data
            for a different temporal aggregation. Alternatively, a single Dataset
            object can be passed.
        weights : Weights
            A Weights object containing the spatial weights used to
            aggregate the climate data.
        names : str or list of str, optional
            The name(s) of the climate variable(s) being aggregated. If a single
            variable is being aggregated, a string can be passed. If multiple
            variables are being aggregated, a list of strings can be passed.

        Returns
        -------
        None

        """
        if type(clim) != list:
            self.clim = [clim]
        else:
            self.clim = clim
        _ = [x.rescale_longitude() for x in self.clim if x.lon_is_360]
        self.grid = weights.grid
        self.weights = weights.weights
        self.names = [names] if isinstance(names, str) else names

    def compute(self, npartitions=30):
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
        clim_ds = dask.compute([x.da for x in self.clim])[0]
        clim_ds = xr.combine_by_coords(
            [x.to_dataset(name=self.names[i]) for i, x in enumerate(clim_ds)]
        )

        clim_df = (
            clim_ds.stack({"cell_id": ["latitude", "longitude"]})
            .drop_vars(["cell_id", "latitude", "longitude"])
            # .assign_coords(coords={"cell_id": ("cell_id", self.grid.index.flatten())})
            .to_dataframe()
            .reset_index("time")
            .dropna(subset=self.names)
        )

        self.weights["region_id"] = self.weights.index_right
        merged_df = clim_df.merge(self.weights, how="inner", on="cell_id")
        merged_df = merged_df.dropna(subset=self.names)

        group_key = (
            merged_df[["region_id", "time"]]
            .drop_duplicates()
            .reset_index(drop=True)
            .reset_index()
            .rename(columns={"index": "group_ID"})
        )

        merged_df = merged_df.merge(group_key, on=["region_id", "time"]).set_index(
            "group_ID"
        )[["weight", *self.names]]

        ddf = dask.dataframe.from_pandas(merged_df, npartitions=50)
        
        client = distributed_client()
        if client is not None:
            future = client.scatter(ddf)
            # disable user warnings for this call
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = client.submit(self.weighted_average, future, self.names)
                progress(out)
                out = out.result().compute()
        else:
            out = self.weighted_average(ddf, self.names)

        aggregated = (
            out.merge(group_key, how="right", left_index=True, right_on="group_ID")
            .drop(columns="group_ID")[["region_id", "time"] + self.names]
            .reset_index(drop=True)
        )

        return aggregated
    
    @staticmethod
    def weighted_average(ddf, names):
        out = ddf[names].mul(ddf["weight"], axis=0)
        out["weight"] = ddf["weight"]
        out = out.groupby(out.index).sum()
        out = out[names].div(out["weight"], axis=0)
        return out
