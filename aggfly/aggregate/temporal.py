from copy import deepcopy
import os
# os.environ['USE_PYGEOS'] = '0'

import numpy as np
import dask.array as da
from .aggregate_utils import *
from ..dataset import Dataset, array_lon_to_360
from ..weights import GridWeights

import numpy as np
from typing import List, Union
from copy import deepcopy

class TemporalAggregator:
    """
    A class for aggregating temporal data.

    Parameters:
    -----------
    calc: str
        The type of calculation to perform. Can be one of "mean", "sum", "dd", "bins", "min", or "max".
    groupby: str
        The time frequency to group the data by.
    ddargs: List[Union[int, float]], optional
        A list of values to use for the "dd" calculation. Only used if calc is "dd" or "bins".

    Attributes:
    -----------
    calc: str
        The type of calculation to perform.
    groupby: str
        The time frequency to group the data by.
    kwargs: dict
        Additional keyword arguments to pass to the reduce function.
    ddargs: List[Union[int, float]], optional
        A list of values to use for the "dd" calculation.
    multi_dd: bool
        Whether or not multiple "dd" values were provided.
    func: function
        The function to use for the calculation.

    Methods:
    --------
    assign_func(self) -> function:
        Assigns the appropriate function based on the value of self.calc.
    get_ddargs(self, ddargs: List[Union[int, float]]) -> List[Union[int, float]]:
        Returns the ddargs list if it is not None, and sets self.multi_dd to True if there are multiple values.
    execute(self, dataset: Dataset, weights: Dataset, update: bool, **kwargs) -> Dataset:
        Executes the aggregation and returns the result.
    """

    def __init__(self, calc: str, groupby: str, ddargs: List[Union[int, float]] = None):
        self.calc = calc
        self.groupby = translate_groupby(groupby)
        self.kwargs = {}
        self.ddargs = self.get_ddargs(ddargs)
        self.func = self.assign_func()

    def assign_func(self):
        """
        Assigns the appropriate function based on the value of self.calc.

        Returns:
        --------
        function:
            The function to use for the calculation.
        """
        if self.calc == "mean":
            f = np.mean
        if self.calc == "sum":
            f = np.sum
        elif self.calc == "dd":
            if self.multi_dd:
                f = _multi_dd
            else:
                f = _dd
            self.kwargs = {"ddargs": self.ddargs}
        elif self.calc == "bins":
            if self.multi_dd:
                f = _multi_bins
            else:
                f = _bins
            self.kwargs = {"ddargs": self.ddargs}
        if self.calc == "min":
            f = np.min
        if self.calc == "max":
            f = np.max
        return f

    def get_ddargs(self, ddargs: List[Union[int, float]]) -> List[Union[int, float]]:
        """
        Returns the ddargs list if it is not None, and sets self.multi_dd to True if there are multiple values.

        Parameters:
        -----------
        ddargs: List[Union[int, float]]
            A list of values to use for the "dd" calculation.

        Returns:
        --------
        List[Union[int, float]]:
            The ddargs list.
        """
        if ddargs is None:
            self.multi_dd = False
            return None
        else:
            ddarr = np.array(ddargs)
            if len(ddarr.shape) > 1:
                self.multi_dd = True
            else:
                self.multi_dd = False
            return ddargs

    def execute(self, dataset: Dataset, weights: GridWeights = None, update: bool = False, **kwargs) -> Dataset:
        """
        Executes the aggregation and returns the result.

        Parameters:
        -----------
        dataset: Dataset
            The data to aggregate.
        weights: GridWeights, optional
            The weights to use for the aggregation.
        update: bool, optional
            Whether or not to update the input data with the result.
        **kwargs:
            Additional keyword arguments to pass to the reduce function.

        Returns:
        --------
        Dataset:
            The aggregated data.
        """
        ds = deepcopy(dataset.da)

        # if weights is not None:
        #     if dataset.grid.lon_is_360:
        #         weights.nonzero_weight_mask = array_lon_to_360(weights.nonzero_weight_mask)
        #     ds = ds.where(weights.nonzero_weight_mask)

        if self.multi_dd:
            ds = ds.expand_dims("dd", axis=-1)
            if not update:
                dataset_list = [deepcopy(dataset) for x in np.arange(len(self.ddargs))]
        else:
            if not update:
                dataset = deepcopy(dataset)
        with dask.config.set(**{'array.slicing.split_large_chunks': False}):
            out = ds.resample(time=self.groupby).reduce(self.func, **self.kwargs)

        if self.multi_dd:
            out = out.to_dataset(dim="dd")
            out = [out[var_name] for var_name in out.variables]

            # Update object and return result
            if type(dataset) == Dataset:
                [x.update(y) for x, y in zip(dataset_list, out)]
                [x.history.append(self.groupby) for x in dataset_list]
                if len(dataset_list) == 1:
                    return dataset_list[0]
                else:
                    return dataset_list
            else:
                return out
        else:
            # Update object and return result
            if type(dataset) == Dataset:
                dataset.update(out)
                dataset.history.append(self.groupby)
                return dataset
            else:
                return out


def _dd(frame, axis, ddargs):
    return (
        (frame > ddargs[0])
        * (frame < ddargs[1])
        * np.absolute(frame - ddargs[ddargs[2]])
    ).sum(axis=axis)


def _multi_dd(frame, axis, ddargs):
    return da.concatenate([_dd(frame, axis, ddarg) for ddarg in ddargs], axis=-1)


def _bins(frame, axis, ddargs):
    return ((frame > ddargs[0]) * (frame < ddargs[1])).sum(axis=axis)


def _multi_bins(frame, axis, ddargs):
    return da.concatenate([_bins(frame, axis, ddarg) for ddarg in ddargs], axis=-1)


def translate_groupby(groupby):
    return {"date": "1D", "month": "ME", "year": "YE"}[groupby]
