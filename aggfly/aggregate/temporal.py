from functools import lru_cache, partial
from copy import deepcopy
import datetime
import os
os.environ['USE_PYGEOS'] = '0'

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import pygeos
import dask
import dask.array as da
import rasterio
import rioxarray
from rasterio.enums import Resampling
import numba
from numba import prange
import zarr

from .aggregate_utils import *
from ..dataset import Dataset, array_lon_to_360


class TemporalAggregator:
    def __init__(self, calc, groupby, ddargs=None):
        self.calc = calc
        self.groupby = translate_groupby(groupby)
        self.kwargs = {}
        self.ddargs = self.get_ddargs(ddargs)
        self.func = self.assign_func()

    def assign_func(self):
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

    def get_ddargs(self, ddargs):
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

    def execute(self, clim, weights=None, update=False, **kwargs):
        ds = deepcopy(clim.da)

        if weights is not None:
            if clim.grid.lon_is_360:
                weights.nonzero_weight_mask = array_lon_to_360(weights.nonzero_weight_mask)
            ds = ds.where(weights.nonzero_weight_mask)

        if self.multi_dd:
            ds = ds.expand_dims("dd", axis=-1)
            if not update:
                clim_list = [deepcopy(clim) for x in np.arange(len(self.ddargs))]
        else:
            if not update:
                clim = deepcopy(clim)

        out = ds.resample(time=self.groupby).reduce(self.func, **self.kwargs)

        if self.multi_dd:
            out = out.to_dataset(dim="dd")
            out = [out[var_name] for var_name in out.variables]

            # Update object and return result
            if type(clim) == Dataset:
                [x.update(y) for x, y in zip(clim_list, out)]
                [x.history.append(self.groupby) for x in clim_list]
                if len(clim_list) == 1:
                    return clim_list[0]
                else:
                    return clim_list
            else:
                return out
        else:
            # Update object and return result
            if type(clim) == Dataset:
                clim.update(out)
                clim.history.append(self.groupby)
                return clim
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
    return {"date": "1D", "month": "M", "year": "Y"}[groupby]
