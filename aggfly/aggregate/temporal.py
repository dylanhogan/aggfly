from copy import deepcopy
import os
import warnings

import numpy as np
import dask.array as da
from .aggregate_utils import *
from ..dataset import Dataset, array_lon_to_360
from ..weights import GridWeights
from typing import List, Union


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

    def __init__(
        self,
        calc: str,
        groupby: str,
        ddargs: List[Union[int, float]] = None,
        pre_compute: bool = False,
    ):
        self.calc = calc
        self.groupby = translate_groupby(groupby)
        self.kwargs = {}
        self.ddargs = self.get_ddargs(ddargs)
        self.func = self.assign_func()
        self.pre_compute = pre_compute

        if self.calc == "sine_dd":
            warnings.warn(
                """
                Sine-interpolated degree day aggregation requires that the 
                dataset be loaded into memory before computation. This can
                have memory and efficiency implications, especially if applied to high resolution
                datasets or hourly data. If the latter, consider using the standard
                degree day method or aggregating to daily max and min before calculating 
                sine-interpolated degree days
                """
            )
            self.pre_compute = True

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
        elif self.calc == "sine_dd":
            if self.multi_dd:
                f = _multi_sine_dd
            else:
                f = _sine_dd
            self.kwargs = {"ddargs": self.ddargs}
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

    def execute(
        self,
        dataset: Dataset,
        weights: GridWeights = None,
        update: bool = False,
        **kwargs
    ) -> Dataset:
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

        if self.pre_compute:
            # raise warning about pre_compute
            warnings.warn(
                "Pre-computing the aggregation may result in memory errors for large datasets."
            )
            ds = ds.compute()

        with dask.config.set(**{"array.slicing.split_large_chunks": False}):
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


def _sine_dd(frame, axis, ddargs):
    # frame = frame.compute()
    tmax = np.max(frame, axis=axis)
    dd_shape = tmax.shape
    tmax = tmax.flatten()
    tmin = np.min(frame, axis=axis).flatten()
    tavg = (tmax + tmin) / 2
    alpha = (tmax - tmin) / 2
    degree_day_list = []
    if ddargs[2] == 0:
        for threshold in ddargs[0:2]:
            arr = np.full_like(tavg, fill_value=np.nan)
            arr[threshold >= tmax,] = np.zeros_like(tmax)[threshold >= tmax,]
            arr[threshold <= tmin,] = (tavg - threshold)[threshold <= tmin,]
            ii = (threshold < tmax) & (threshold > tmin)
            arr[ii] = (
                (tavg[ii] - threshold)
                * np.arccos(
                    (2 * threshold - tmax[ii] - tmin[ii]) / (tmax[ii] - tmin[ii])
                )
                + (tmax[ii] - tmin[ii])
                * np.sin(
                    np.arccos(
                        (2 * threshold - tmax[ii] - tmin[ii]) / (tmax[ii] - tmin[ii])
                    )
                )
                / 2
            ) / np.pi
            degree_day_list.append(arr)
        degree_days = degree_day_list[0] - degree_day_list[1]
    elif ddargs[2] == 1:
        for threshold in ddargs[0:2]:
            arr = np.full_like(tavg, fill_value=np.nan)
            arr[(threshold >= tmax)] = (threshold - tavg)[(threshold >= tmax)]
            arr[(threshold <= tmin)] = np.zeros_like(arr)[(threshold <= tmin)]
            ii = (threshold < tmax) * (threshold > tmin)
            arr[ii] = (1 / (np.pi)) * (
                (threshold - tavg[ii])
                * (
                    np.arctan(
                        ((threshold - tavg[ii]) / alpha[ii])
                        / np.sqrt(1 - ((threshold - tavg[ii]) / alpha[ii]) ** 2)
                    )
                    + (np.pi / 2)
                )
                + alpha[ii]
                * np.cos(
                    (
                        np.arctan(
                            ((threshold - tavg[ii]) / alpha[ii])
                            / np.sqrt(1 - ((threshold - tavg[ii]) / alpha[ii]) ** 2)
                        )
                    )
                )
            )
            degree_day_list.append(arr)
        degree_days = degree_day_list[1] - degree_day_list[0]
    else:
        raise ValueError("Invalid degree day type")

    return degree_days.reshape(dd_shape)


def _multi_sine_dd(frame, axis, ddargs):
    return da.concatenate([_sine_dd(frame, axis, ddarg) for ddarg in ddargs], axis=-1)


def _bins(frame, axis, ddargs):
    return ((frame > ddargs[0]) * (frame < ddargs[1])).sum(axis=axis)


def _multi_bins(frame, axis, ddargs):
    return da.concatenate([_bins(frame, axis, ddarg) for ddarg in ddargs], axis=-1)


def translate_groupby(groupby):
    return {"date": "1D", "month": "ME", "year": "YE"}[groupby]
