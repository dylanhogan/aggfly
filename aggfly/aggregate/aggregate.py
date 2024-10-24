"""
This module contains functions for aggregating datasets over time and space.

Functions:
- transform_dataset(dataset: Dataset, key: str, **kwargs: Union[int, List[int]]) -> Dict[str, Dataset]:
    Transform a given dataset by raising it to a power or interacting with another dataset.

- aggregate_time(dataset: Dataset, aggregator_dict: Dict[str, Union[List[Tuple], TemporalAggregator]] = None, weights: GridWeights = None, **kwargs) -> Dict[str, Dataset]:
    Aggregate a dataset over time using the specified temporal aggregators.

- aggregate_space(dataset_dict: Dict[str, Dataset], weights: GridWeights) -> pd.DataFrame:
    Aggregate a dictionary of datasets over space.

- aggregate_dataset(dataset: Dataset, weights: GridWeights, aggregator_dict: Dict[str, Union[List[Tuple], TemporalAggregator]] = None, **kwargs) -> pd.DataFrame:
    Aggregate a dataset over time and space.
"""

from typing import List, Dict, Union, Tuple
import pandas as pd
import dask
from dask.distributed import LocalCluster, Client, progress

from .temporal import TemporalAggregator
from .spatial import SpatialAggregator
from ..dataset import Dataset
from ..weights import GridWeights
from .aggregate_utils import distributed_client, is_distributed, start_dask_client, shutdown_dask_client


from dask.diagnostics import ProgressBar
ProgressBar().register()



def transform_dataset(
    dataset: Dataset, key: str, **kwargs: Union[int, List[int]]
) -> Dict[str, Dataset]:
    """
    Transform a given dataset by raising it to a power or interacting with another dataset.

    Args:
        dataset (Dataset): The dataset to transform.
        key (str): The name of the dataset.
        **kwargs:   Keyword arguments for the transformation.
                    If 'exp' is provided, raise the dataset to the power of the provided exponent(s).
                    If 'inter' is provided, interact the dataset with another dataset.

    Returns:
        dict: A dictionary containing the transformed dataset(s).
    """

    # Check if 'exp' keyword argument is provided
    if "exp" in kwargs:
        # Ensure 'exp' is a list
        if not isinstance(kwargs["exp"], list):
            kwargs["exp"] = [kwargs["exp"]]
        # Raise the dataset to the power of each exponent in the list
        dataset = [dataset.power(exp) for exp in kwargs["exp"][0]]
        # Generate new keys for each transformed dataset
        new_keys = [f"{key}_{exp}" for exp in kwargs["exp"][0]]
        # Create a dictionary with new keys and transformed datasets
        output_dict = dict(zip(new_keys, dataset))
    # Check if 'inter' keyword argument is provided
    elif "inter" in kwargs:
        # Interact the dataset with another dataset
        dataset = dataset.interact(kwargs["inter"])
        # Create a dictionary with the original key and the interacted dataset
        output_dict = {key: dataset}
    else:
        # Raise an error if neither 'exp' nor 'inter' is provided
        raise ValueError("No valid transform argument provided.")
    # Return the values and keys of the transformed datasets dictionary
    return output_dict.values(), output_dict.keys()


# def aggregate_time(
#     dataset: Dataset,
#     weights: GridWeights = None,
#     aggregator_dict: Dict[str, Union[List[Tuple], TemporalAggregator]] = None,
#     **kwargs,
# ) -> Dict[str, Dataset]:
#     """
#     Aggregate a dataset over time using the specified temporal aggregators.

#     Args:
#         dataset (Dataset): The dataset to aggregate.
#         weights (GridWeights, optional): The weights to use for aggregation. Defaults to None.
#         aggregator_dict (Dict[str, Union[List[Tuple], TemporalAggregator]], optional): A dictionary of temporal aggregators to apply to the dataset. Defaults to None.
#         **kwargs: Additional keyword arguments to use if `aggregator_dict` is not provided.

#     Returns:
#         Dict[str, Dataset]: A dictionary of aggregated datasets, with keys corresponding to the keys in `aggregator_dict`.
#     """


def aggregate_time(
    dataset: Dataset,
    weights: GridWeights = None,
    aggregator_dict: Dict[str, Union[List[Tuple], TemporalAggregator]] = None,
    **kwargs,
) -> Dict[str, Dataset]:
    """
    Aggregate a dataset over time using the specified temporal aggregators.

    Args:
        dataset (Dataset): The dataset to aggregate.
        weights (GridWeights, optional): The weights to use for aggregation. Defaults to None.
        aggregator_dict (Dict[str, Union[List[Tuple], TemporalAggregator]], optional): A dictionary of temporal aggregators to apply to the dataset. Defaults to None.
        **kwargs: Additional keyword arguments to use if `aggregator_dict` is not provided.

    Returns:
        Dict[str, Dataset]: A dictionary of aggregated datasets, with keys corresponding to the keys in `aggregator_dict`.
    """
    # Check if aggregator_dict is provided, if not, use kwargs
    if aggregator_dict is None:
        if kwargs is None:
            raise ValueError("No arguments provided.")
        else:
            aggregator_dict = kwargs
    out_dict = {}
    # Iterate over each key-value pair in aggregator_dict
    for key, value in aggregator_dict.items():
        keys = [key]
        data = [dataset.deepcopy()]
        # Process each tuple in the value list
        for key2, value2 in value:
            if key2 == "aggregate":
                # If 'aggregate' key is found, execute the temporal aggregator
                if not isinstance(value2, TemporalAggregator):
                    value2 = TemporalAggregator(**value2)
                data = [value2.execute(x, weights) for x in data]

                # Handle multi_dd case for multiple data dimensions
                if value2.multi_dd:
                    if len(data) > 1:
                        raise ValueError(
                            "Cannot aggregate multiple datasets with multiple ddargs, e.g., multiple polynomials for multiple bins"
                        )
                    data, keys = multi_dd_to_dict(data[0], key, value2.ddargs)

            elif key2 == "transform":
                # If 'transform' key is found, apply transformation to the dataset
                transformed_data, transformed_keys = [], []
                for d, k in zip(data, keys):
                    d2, k2 = transform_dataset(d, k, **value2)
                    transformed_data.extend(d2)
                    transformed_keys.extend(k2)
                data, keys = transformed_data, transformed_keys

        # Update the output dictionary with the processed data
        data_dict = dict(zip(keys, data))
        out_dict = out_dict | data_dict
    return out_dict


def aggregate_space(
    dataset_dict: Dict[str, Dataset], weights: GridWeights,
    npartitions=None, **kwargs
) -> pd.DataFrame:
    """
    Aggregate a dictionary of datasets over space.

    Args:
        dataset_dict (dict): A dictionary containing the datasets to aggregate, where
                            the keys are the names of the datasets and the values are the datasets themselves.
        weights (GridWeights): The weights to use for aggregation.
        npartitions (int, optional): The number of partitions for parallel computation. Defaults to None.

    Returns:
        df: A dataframe containing the aggregated data.
    """
    
    # Convert the values of dataset_dict to a list
    dataset_list = list(dataset_dict.values())
    
    # client = distributed_client()
    # if client is None and npartitions is None:
    #     npartitions=1
    # else:
        # npartitions=len(client.scheduler_info()["workers"])
        # da_list = dask.persist([x.da for x in dataset_list])[0]
        # progress(da_list)
        # for i, dataset in enumerate(dataset_list):
        #     dataset.da = da_list[i]
    # Perform spatial aggregation on the dataset list with the specified weights
    df = SpatialAggregator(
        dataset_list, weights, names=list(dataset_dict.keys()),
    ).compute(npartitions=npartitions)
    return df


def aggregate_dataset(
    weights: GridWeights,
    dataset: Dataset = None,
    aggregator_dict: Dict[str, Union[List[Tuple], TemporalAggregator]] = None,
    dataset_dict = None,
    n_workers = 50,
    threads_per_worker = 1,
    processes = True,
    memory_limit=None,
    cluster_args = {},
    **kwargs,
) -> pd.DataFrame:
    """
    Aggregate a dataset over time and space.

    Args:
        dataset (Dataset): The dataset to aggregate.
        weights (GridWeights): The weights to use for aggregation.
        agg_dict (dict): A dictionary containing the arguments for creating TemporalAggregator objects.
                        The keys of the dictionary are names, and the values are a list of either tuples or TemporalAggregator objects.
                        If the list contains tuples, use them as arguments to instantiate a temporal aggregator.
        dataset_dict (dict, optional): A dictionary containing pre-aggregated datasets. Defaults to None.
        n_workers (int, optional): Number of workers for parallel processing. Defaults to 50.
        threads_per_worker (int, optional): Number of threads per worker. Defaults to 1.
        processes (bool, optional): Whether to use processes or threads. Defaults to True.
        memory_limit (str, optional): Memory limit per worker. Defaults to None.
        cluster_args (dict, optional): Additional arguments for cluster configuration. Defaults to {}.
        **kwargs: Additional keyword arguments.

    Returns:
        df: A dataframe containing the aggregated data.
    """

    # Check if a dataset is provided
    if dataset is None:
        raise ValueError("No dataset provided.")

    # If aggregator_dict is not provided, use kwargs
    if aggregator_dict is None and kwargs is not None:
        aggregator_dict = kwargs

    # Aggregate over time if aggregator_dict is provided
    if aggregator_dict is not None:
        dataset_dict = aggregate_time(dataset, weights, aggregator_dict)
    # If no dataset_dict is provided, create a default one
    elif dataset_dict is None:
        dataset_dict = {"variable": dataset}
        if dataset_dict is None and dataset is None:
            raise ValueError("No aggregator dict or dataset dict provided.")
    
    # Aggregate over space using the dataset_dict
    df = aggregate_space(dataset_dict, weights)
    # Merge the aggregated data with geographical regions
    df = (
        weights.georegions.shp[[weights.georegions.regionid]].merge(
            df, left_index=True, right_on="region_id"
        )
    ).drop(columns="region_id")
    
    # client.shutdown()

    # _ = shutdown_dask_client()
    
    return df


def multi_dd_to_dict(data, key, ddargs):
    """
    Converts a multi-variable list of datasets to a dictionary with keys
    generated from the given key and ddargs.

    Args:
        data (list): The list of Datasets to convert to a dictionary.
        key (str): The base key to use for generating the dictionary keys.
        ddargs (list): A list of tuples representing the dimensions of the array.

    Returns:
        dict: A dictionary with keys generated from the given key and ddargs, and values from the list of datasets.
    """
    # Generate keys by combining the base key with each pair of dimensions in ddargs
    keys = [f"{key}_{x[0]}_{x[1]}" for x in ddargs]
    # Combine keys and data into a dictionary
    # data_dict = dict(zip(keys, data))
    # Return the list of datasets and the generated keys
    return data, keys
