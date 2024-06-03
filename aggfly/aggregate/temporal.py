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
        """
        Initialize a TemporalAggregator object.

        Parameters
        ----------
        calc : str
            The type of calculation to perform.
        groupby : str
            The time frequency to group the data by.
        ddargs : List[Union[int, float]], optional
            A list of values to use for the "dd" calculation. Only used if calc is "dd" or "bins".

        Returns
        -------
        None
        """
        
        self.calc = calc # Type of calculation to perform
        # Translate the groupby string to the appropriate format
        self.groupby = translate_groupby(groupby)
        # Additional keyword arguments for the reduce function
        self.kwargs = {}
        # Retrieve the ddargs list
        self.ddargs = self.get_ddargs(ddargs)
        # Assign the appropriate function based on the calculation type
        self.func = self.assign_func()

    def assign_func(self):
        """
        Assigns the appropriate function based on the value of self.calc.

        Returns:
        --------
        function:
            The function to use for the calculation.
        """
        # Check if the calculation type is "mean" and assign np.mean function
        if self.calc == "mean":
            f = np.mean
        # Check if the calculation type is "sum" and assign np.sum function
        if self.calc == "sum":
            f = np.sum
        # Check if the calculation type is "dd" and assign appropriate dd function
        elif self.calc == "dd":
            if self.multi_dd:
                f = _multi_dd # Assign _multi_dd if multiple dd values are provided
            else:
                f = _dd # Assign _dd for a single dd value
            self.kwargs = {"ddargs": self.ddargs} # Set ddargs in kwargs
        # Check if the calculation type is "bins" and assign appropriate bins function
        elif self.calc == "bins":
            if self.multi_dd:
                f = _multi_bins # Assign _multi_bins if multiple bins values are provided
            else:
                f = _bins # Assign _bins for a single bins value
            self.kwargs = {"ddargs": self.ddargs} # Set ddargs in kwargs
        # Check if the calculation type is "min" and assign np.min function
        if self.calc == "min":
            f = np.min
        # Check if the calculation type is "max" and assign np.max function
        if self.calc == "max":
            f = np.max
        # Return the assigned function
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
        # If ddargs is None, set multi_dd to False and return None
        if ddargs is None:
            self.multi_dd = False
            return None
        else:
            # Convert ddargs to a numpy array
            ddarr = np.array(ddargs)
            # Check if the array has more than one dimension
            if len(ddarr.shape) > 1:
                self.multi_dd = True # Set multi_dd to True if multiple dimensions are found
            else:
                self.multi_dd = False # Set multi_dd to False if only a single dimension is found
            # Return the ddargs list
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
        # Create a deep copy of the dataset's data array
        ds = deepcopy(dataset.da)

        # if weights is not None:
        #     if dataset.grid.lon_is_360:
        #         weights.nonzero_weight_mask = array_lon_to_360(weights.nonzero_weight_mask)
        #     ds = ds.where(weights.nonzero_weight_mask)

        # Handle multi_dd case by expanding dimensions if necessary
        if self.multi_dd:
            # Expand dimensions for multi_dd
            ds = ds.expand_dims("dd", axis=-1)
            if not update:
                # Create a list of deep copies of the dataset for each ddarg
                dataset_list = [deepcopy(dataset) for x in np.arange(len(self.ddargs))]
        else:
            # Create a deep copy of the dataset if not updating
            if not update:
                dataset = deepcopy(dataset)
        
        # Set Dask configuration to avoid splitting large chunks during slicing and perform resampling and reduction
        with dask.config.set(**{'array.slicing.split_large_chunks': False}):
            out = ds.resample(time=self.groupby).reduce(self.func, **self.kwargs)

        # Handle multi_dd output by converting to dataset and splitting by variables
        if self.multi_dd:
            out = out.to_dataset(dim="dd")
            out = [out[var_name] for var_name in out.variables]

            # Update the dataset objects and return the result
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
            # Update the dataset objects and return the result
            if type(dataset) == Dataset:
                dataset.update(out)
                dataset.history.append(self.groupby)
                return dataset
            else:
                return out


def _dd(frame, axis, ddargs):
    """
    Custom function for degree days ('dd') calculation.

    Parameters:
    -----------
    frame: np.ndarray
        The data frame to apply the calculation to.
    axis: int
        The axis to sum over.
    ddargs: list
        A list of arguments for the 'dd' calculation.

    Returns:
    --------
    np.ndarray:
        The result of the 'dd' calculation.
    """
    # Calculate the 'dd' value by checking the conditions and applying the absolute difference
    return (
        (frame > ddargs[0]) # Check if frame values are greater than the first ddarg
        * (frame < ddargs[1]) # Check if frame values are less than the second ddarg
        * np.absolute(frame - ddargs[ddargs[2]]) # Compute the absolute difference from the specified ddarg
    ).sum(axis=axis) # Sum the result along the specified axis


def _multi_dd(frame, axis, ddargs):
    """
    Custom function for 'multi_dd' calculation.

    Parameters:
    -----------
    frame: np.ndarray
        The data frame to apply the calculation to.
    axis: int
        The axis to concatenate over.
    ddargs: list
        A list of ddargs lists for the 'dd' calculation.

    Returns:
    --------
    da.Array:
        The result of concatenating multiple 'dd' calculations along the specified axis.
    """
    # Apply the '_dd' function for each ddarg in ddargs and concatenate the results along the specified axis
    return da.concatenate([_dd(frame, axis, ddarg) for ddarg in ddargs], axis=-1)


def _bins(frame, axis, ddargs):
    """
    Custom function for 'bins' calculation.

    Parameters:
    -----------
    frame: np.ndarray
        The data frame to apply the calculation to.
    axis: int
        The axis to sum over.
    ddargs: list
        A list of arguments for the 'bins' calculation.

    Returns:
    --------
    np.ndarray:
        The result of the 'bins' calculation.
    """
    # Calculate the 'bins' value by checking the conditions and summing the result
    return ((frame > ddargs[0]) # Check if frame values are greater than the first ddarg
            * (frame < ddargs[1])) # Check if frame values are less than the second ddarg 
            .sum(axis=axis) # Sum the result along the specified axis


def _multi_bins(frame, axis, ddargs):
    """
    Custom function for 'multi_bins' calculation.

    Parameters:
    -----------
    frame: np.ndarray
        The data frame to apply the calculation to.
    axis: int
        The axis to concatenate over.
    ddargs: list
        A list of ddargs lists for the 'bins' calculation.

    Returns:
    --------
    da.Array:
        The result of concatenating multiple 'bins' calculations along the specified axis.
    """
    # Apply the '_bins' function for each ddarg in ddargs and concatenate the results along the specified axis
    return da.concatenate([_bins(frame, axis, ddarg) for ddarg in ddargs], axis=-1)


def translate_groupby(groupby):
    """
    Translates a groupby string to a corresponding frequency string.

    Parameters:
    -----------
    groupby: str
        The string indicating the grouping frequency ("date", "month", "year").

    Returns:
    --------
    str:
        The corresponding frequency string for resampling.
    """
    # Translate the groupby string to a frequency string using a dictionary lookup
    return {"date": "1D", "month": "ME", "year": "YE"}[groupby]
