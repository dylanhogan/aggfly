from copy import deepcopy
from typing import Callable, Tuple, Union, List, Dict, Any, Optional
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import numba
import dask
import dask.array

from .grid import Grid
from .grid_utils import *
from ..regions import GeoRegions


class Dataset:
    """
    A class used to represent a Dataset.

    Attributes
    ----------
    da : xarray.DataArray
        The data array to be processed.
    name : str
        The name of the dataset.
    lon_is_360 : bool
        A flag indicating if the longitude is 360.
    coords : xarray.DataArray.coords
        The coordinates of the data array.
    longitude : xarray.DataArray.longitude
        The longitude of the data array.
    latitude : xarray.DataArray.latitude
        The latitude of the data array.
    grid : Grid
        The grid of the dataset.
    history : list
        The history of the dataset operations.
    georegions : GeoRegions
        The geographical regions associated with the dataset.
    """

    def __init__(
        self,
        da: xr.DataArray,
        xycoords: tuple = ("longitude", "latitude"),
        time_sel: str = None,
        lon_is_360: bool = True,
        preprocess: callable = None,
        georegions: GeoRegions = None,
        time_fix: bool = False,
        name: str = None,
    ):
        """
        Constructs all the necessary attributes for the Dataset object.

        Parameters
        ----------
        da : xarray.DataArray
            The data array to be processed.
        xycoords : tuple, optional
            The x and y coordinates (default is ("longitude", "latitude")).
        time_sel : str, optional
            The time selection (default is None).
        lon_is_360 : bool, optional
            A flag indicating if the longitude is 360 (default is True).
        preprocess : callable, optional
            The preprocessing function (default is None).
        georegions : GeoRegions, optional
            The geographical regions associated with the dataset (default is None).
        time_fix : bool, optional
            A flag indicating if the time needs to be fixed (default is False).
        name : str, optional
            The name of the dataset (default is None).
        """
        
        with dask.config.set(**{"array.slicing.split_large_chunks": False}):  
            da = clean_dims(da, xycoords)
            da = da.sortby("time")
            if time_sel is not None:
                da = da.sortby("time").sel(time=time_sel)
                # time_fix=True
            if preprocess is not None:
                da = preprocess(da)

            self.update(da, init=True)
            self.name = name
            self.lon_is_360 = lon_is_360
            self.coords = self.da.coords

            self.longitude = self.da.longitude
            self.latitude = self.da.latitude
            assert np.all([x in list(self.coords) for x in ["latitude", "longitude"]])
            self.grid = Grid(self.longitude, self.latitude, self.name, self.lon_is_360)
            self.history = []
            self.georegions = georegions
            if self.georegions is not None:
                self.clip_data_to_georegions_extent(self.georegions)
            if time_fix:
                self.update(timefix(self.da), init=True)

    def rechunk(self, chunks: str = "auto"):
        """
        Rechunks the data array.

        Parameters
        ----------
        chunks : str, optional
            The chunk size (default is "auto").
        """
        # Rechunk data
        self.da = self.da.chunk(chunks)

    def clip_data_to_grid(self, split: bool = False):
        """
        Clips the data array to the grid.

        Parameters
        ----------
        split : bool, optional
            A flag indicating if the large chunks should be split (default is False).
        """

        self.da = self.da.sel(
            latitude=self.grid.latitude, longitude=self.grid.longitude
        )
        self.coords = self.da.coords
        self.longitude = self.da.longitude
        self.latitude = self.da.latitude

    def clip_data_to_georegions_extent(
        self, georegions: GeoRegions, split: bool = False, update: bool = True
    ):
        """
        Clips the data array to the extent of the georegions.

        Parameters
        ----------
        georegions : GeoRegions
            The georegions to clip the data to.
        split : bool, optional
            A flag indicating if the large chunks should be split (default is False).
        update : bool, optional
            A flag indicating if the data array should be updated (default is True).
        """

        if update:
            self.grid.clip_grid_to_georegions_extent(georegions)
            self.clip_data_to_grid(split)
        else:
            slf = self.deepcopy()
            slf.grid.clip_grid_to_georegions_extent(georegions)
            slf.clip_data_to_grid(split)
            return slf

    def clip_data_to_bbox(self, bounds: tuple, split: bool = False) -> None:
        """
        Clips the data array to the bounding box.

        Parameters
        ----------
        bounds : tuple
            The bounding box to clip the data to.
        split : bool, optional
            A flag indicating if the large chunks should be split (default is False).
        """
        self.grid.clip_grid_to_bbox(bounds)
        self.clip_data_to_grid(split)

    def compute(
        self, dask_array: bool = True, chunks: Union[str, tuple] = None
    ) -> None:
        """
        Computes the data array.

        Parameters
        ----------
        dask_array : bool, optional
            A flag indicating if the data array should be a dask array (default is True).
        chunks : str or tuple, optional
            The chunk sizes for the dask array (default is None).
        """
        # ...
        self.update(self.da.compute(), dask_array=dask_array, chunks=chunks)

    def deepcopy(self) -> "Dataset":
        """
        Creates a deep copy of the Dataset object.

        Returns
        -------
        Dataset
            A deep copy of the Dataset object.
        """
        return deepcopy(self)

    def update(
        self,
        array: xr.DataArray,
        drop_dims: List[str] = None,
        new_dims: Dict[str, Any] = None,
        pos: int = 0,
        dask_array: bool = True,
        chunks: Union[str, tuple] = "auto",
        init: bool = False,
    ) -> None:
        """
        Updates the data array.

        Parameters
        ----------
        array : xarray.DataArray
            The new data array.
        drop_dims : list of str, optional
            The dimensions to drop (default is None).
        new_dims : dict, optional
            The new dimensions (default is None).
        pos : int, optional
            The position to insert the new dimensions at (default is 0).
        dask_array : bool, optional
            A flag indicating if the data array should be a dask array (default is True).
        chunks : str or tuple, optional
            The chunk sizes for the dask array (default is "auto").
        init : bool, optional
            A flag indicating if this is the initial update (default is False).
        """
        if not init:
            old_coords = self.da.coords

        # print(type(array))

        if type(array) == xr.core.dataarray.DataArray:
            # Coerce data into dask array if necessary
            if dask_array:
                if type(array.data) != dask.array.core.Array:
                    self.da = xr.DataArray(
                        data=dask.array.from_array(array.data, chunks=chunks),
                        dims=array.dims,
                        coords=array.coords,
                    )
                else:
                    self.da = array
            else:
                if type(array.data) != dask.array.core.Array:
                    self.da = array.compute()
                else:
                    self.da = array
        else:
            if drop_dims is not None:
                dargs = dict()
                for dd in drop_dims:
                    dargs[dd] = 0
                self.da = self.da.isel(**dargs)
                for dd in drop_dims:
                    self.da = self.da.drop(dd)

            if type(array) != dask.array.core.Array and dask_array:
                array = dask.array.from_array(array)

            if new_dims is None:
                self.da.data = array
            else:
                cdict = dict()
                for k in new_dims.keys():
                    cdict[k] = ([k], new_dims[k])
                for i in self.da.coords:
                    cdict[i] = ([i], self.da.coords[i].values)

                ndims = tuple()
                i = 0
                for d in self.da.dims:
                    if i == pos:
                        ndims = ndims + tuple(new_dims.keys())
                    ndims = ndims + (d,)
                    i += 1

                self.da = xr.DataArray(data=array, dims=ndims, coords=cdict)

    @lru_cache(maxsize=None)
    def interior_cells(
        self,
        georegions: GeoRegions,
        buffer: Optional[float] = None,
        dtype: str = "georegions",
        maxsize: Optional[int] = None,
    ) -> Union[xr.DataArray, gpd.GeoDataFrame, GeoRegions]:
        """
        Returns the interior cells of the dataset.

        Parameters
        ----------
        georegions : GeoRegions
            The geographical regions to consider.
        buffer : float, optional
            The buffer size (default is the grid resolution).
        dtype : str, optional
            The output data type (default is "georegions").
        maxsize : int, optional
            The maximum size of the output (default is None).

        Returns
        -------
        xarray.DataArray or geopandas.GeoDataFrame or GeoRegions
            The interior cells.
        """
        if buffer is None:
            buffer = self.grid.resolution

        # mask = self.grid.mask(georegions, buffer=buffer)
        cells = self.grid.centroids_to_cell(georegions, buffer=buffer)
        # if dtype == 'gpd':

        cells = xr.DataArray(
            data=cells,
            dims=["region", "latitude", "longitude"],
            coords=dict(
                latitude=("latitude", self.latitude.values),
                longitude=("longitude", self.longitude.values),
                region=("region", georegions.regions),
            ),
        )

        if dtype == "xarray":
            return cells
        elif dtype == "gdf" or dtype == "georegions":
            cells.name = "geometry"
            out = cells.to_dataframe()
            df = out.loc[np.logical_not(out.geometry.isnull())]
            df = df.reset_index()
            if dtype == "gdf":
                return gpd.GeoDataFrame(df)
            elif dtype == "georegions":
                if maxsize is None:
                    count = df.groupby(["region"]).cumcount() + 1
                    df["cellid"] = [
                        f"{df.region[i]}.{count[i]}" for i in range(len(df.region))
                    ]
                else:
                    subregion = df.groupby(["region"]).cumcount() + 1
                    subregion = np.int64(np.floor(subregion / maxsize)) + 1
                    df["subregion"] = [
                        f"{df.region[i]}.{subregion[i]}" for i in range(len(df.region))
                    ]
                    count = df.groupby(["region", "subregion"]).cumcount() + 1
                    df["cellid"] = [
                        f"{df.region[i]}.{subregion[i]}.{count[i]}"
                        for i in range(len(df.region))
                    ]

                gdf = GeoRegions(gpd.GeoDataFrame(df), "cellid")
                return GeoRegions(gpd.GeoDataFrame(df), "cellid")
        else:
            return NotImplementedError

    def sel(self, **kwargs) -> None:
        """
        Selects data by label along the specified dimensions.

        Parameters
        ----------
        **kwargs : dict
            The dimensions and labels to select.
        """
        da = self.da
        for k in kwargs.keys():
            d = {k: kwargs[k]}
            da = da.sel(d).expand_dims(k).transpose(*self.da.dims)
        self.update(da)

    def rescale_longitude(self) -> None:
        """
        Rescales the longitude of the dataset.
        """
        # Update Longitude coordinates
        if self.lon_is_360:
            self.update(array_lon_to_180(self.da))
            self.lon_is_360 = False
        else:
            self.update(array_lon_to_360(self.da))
            self.lon_is_360 = True
        # Regenerate attributes
        self.longitude = self.da.longitude
        self.latitude = self.da.latitude
        self.grid = Grid(self.longitude, self.latitude, self.name, self.lon_is_360)

    def power(self, exp: int, update: bool = False) -> Optional["Dataset"]:
        """
        Raises the data array to the specified power.

        Parameters
        ----------
        exp : int
            The power to raise the data array to.
        update : bool, optional
            A flag indicating if the data array should be updated (default is False).

        Returns
        -------
        Dataset, optional
            The updated Dataset object (only if update is False).
        """
        arr = self.da.data.map_blocks(_power, exp)
        if update:
            self.update(arr)
            self.history.append(f"power{exp}")
        else:
            slf = self.deepcopy()
            slf.update(arr)
            slf.history.append(f"power{exp}")
            return slf

    def interact(
        self, inter: Union["Dataset", xr.DataArray], update: bool = False
    ) -> Optional["Dataset"]:
        """
        Interacts the data array with another data array.

        Parameters
        ----------
        inter : Dataset or xarray.DataArray
            The data array to interact with.
        update : bool, optional
            A flag indicating if the data array should be updated (default is False).
        """
        if type(inter) == Dataset:
            inter = inter.da.data

        assert self.da.data.shape == inter.shape
        arr = self.da.data.map_blocks(_interact, inter)
        if update:
            self.update(arr)
            self.history.append("interacted")
        else:
            slf = self.deepcopy()
            slf.update(arr)
            slf.history.append("interacted")
            return slf


@numba.njit(fastmath=True, parallel=True)
def _power(array, exp):
    return np.power(array, exp)


@numba.njit(fastmath=True, parallel=True)
def _interact(array, inter):
    return np.multiply(array, inter)


def dataset_from_path(
    path: Union[str, List[str]],
    var: str,
    xycoords: Tuple[str, str] = ("longitude", "latitude"),
    time_sel: Optional[str] = None,
    georegions: Optional[GeoRegions] = None,
    lon_is_360: bool = True,
    time_fix: bool = False,
    preprocess: Optional[Callable[[xr.Dataset], xr.Dataset]] = None,
    name: Optional[str] = None,
    chunks: Dict[str, Union[str, int]] = {
        "time": 24,
        "latitude": -1,
        "longitude": -1,
    },
    preprocess_at_load = False,
    **kwargs,
) -> Dataset:
    """
    Loads a Dataset from a file or a set of files.

    Parameters
    ----------
    path : str or list of str
        The file path or paths.
    var : str
        The variable to load.
    xycoords : tuple of str, optional
        The names of the x and y coordinates (default is ("longitude", "latitude")).
    time_sel : str, optional
        The time selection (default is None).
    georegions : GeoRegions, optional
        The geographical regions associated with the dataset, used to clip
        dataset to regional bounds (default is None).
    lon_is_360 : bool, optional
        A flag indicating if the longitude is scaled from 0 to 360 (default is True).
    time_fix : bool, optional
        A flag indicating if the time should be fixed (default is False). [deprecated]
    preprocess : callable, optional
        A function to preprocess the data when loaded (default is None).
    name : str, optional
        The name of the Dataset (default is None).
    chunks : dict, optional
        The chunk sizes (default is {"time": "auto", "latitude": -1, "longitude": -1}).

    Returns
    -------
    Dataset
        The loaded Dataset.
    """
    
    with dask.config.set(**{"array.slicing.split_large_chunks": False}):  
        if "*" in path or type(path) is list:
            
                array = xr.open_mfdataset(
                    path, preprocess=preprocess, parallel=True, chunks=chunks, **kwargs
                )

                preprocess = None

        else:
            if ".zarr" in path:
                array = xr.open_dataset(path, engine='zarr', chunks=chunks, **kwargs)
            else:
                array = xr.open_dataset(path, chunks=chunks, **kwargs)
        
        if preprocess_at_load:
            array = preprocess(array)
            preprocess=None
        
        array = array[var]

    return Dataset(
        array,
        xycoords=xycoords,
        time_sel=time_sel,
        lon_is_360=lon_is_360,
        preprocess=preprocess,
        georegions=georegions,
        time_fix=time_fix,
        name=name,
    )


def from_name(name, var, chunks="auto", **kwargs):
    # if name == 'prism':
    #
    path, engine, preprocess = get_path(name)
    clim = dataset_from_path(path, var, engine, preprocess, name, chunks, **kwargs)
    return clim


def get_path(name):
    if name == "era5l":
        return ("/home3/dth2133/data/annual/*.zarr", "zarr", preprocess_era5l)
    elif name == "era5l_diag":
        return ("/home3/dth2133/data/ERA5", "zarr", None)
    else:
        raise NotImplementedError
