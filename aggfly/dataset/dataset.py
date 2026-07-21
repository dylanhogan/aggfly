# This script defines the Dataset class used to represent and manipulate gridded climate data.
# It includes methods for initializing the dataset, performing preprocessing, clipping data to specific regions,
# computing the data array, and applying operations like power and interaction with another dataset.

import importlib.util
from copy import deepcopy
from typing import Callable, Tuple, Union, List, Dict, Any, Optional
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
# import numba
import dask
import dask.array

from .grid import Grid
from .grid_utils import *
from ..regions import GeoRegions


class Dataset():
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
        timecoord: str = "time",
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
        timecoord : str, optional
            The time coordinate (default is "time").
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
        # Set Dask configuration to not split large chunks during array slicing
        with dask.config.set(**{"array.slicing.split_large_chunks": False}):  
            # Clean dimensions and ensure proper coordinate names
            da = clean_dims(da, xycoords, timecoord)
            # Sort the data array by time
            da = da.sortby("time")
            # If a time selection is provided, select the data for that time
            if time_sel is not None:
                da = da.sortby("time").sel(time=time_sel)
                # time_fix=True
            # If a preprocessing function is provided, apply it
            if preprocess is not None:
                da = preprocess(da)
                
            # Update the Dataset with the processed data array
            self.update(da, init=True)
            self.name = name
            self.lon_is_360 = lon_is_360
            self.coords = self.da.coords

            self.longitude = self.da.longitude
            self.latitude = self.da.latitude
            
            # Ensure latitude and longitude are in the coordinates
            assert np.all([x in list(self.coords) for x in ["latitude", "longitude"]])
            
            # Initialize the Grid object
            self.grid = Grid(self.longitude, self.latitude, self.name, self.lon_is_360)
            self.history = []
            self.georegions = georegions
            # If georegions are provided, clip data to their extent
            if self.georegions is not None:
                self.clip_data_to_georegions_extent(self.georegions)
            # If time needs to be fixed, update the Dataset with fixed time
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
        # Select data within the latitude and longitude bounds of the grid
        self.da = self.da.sel(
            latitude=self.grid.latitude, longitude=self.grid.longitude
        )
        # Update coordinates, longitude, and latitude attributes
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
            # Clip the grid to the extent of the georegions
            self.grid.clip_grid_to_georegions_extent(georegions)
            # Clip the data array to the grid
            self.clip_data_to_grid(split)
        else:
            # Create a deepcopy of the dataset
            slf = self.deepcopy()
            # Clip the grid to the extent of the georegions in the copy
            slf.grid.clip_grid_to_georegions_extent(georegions)
            # Clip the data array to the grid in the copy
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
        # Clip the grid to the bounding box
        self.grid.clip_grid_to_bbox(bounds)
        # Clip the data array to the grid
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
        # Compute the data array and update the dataset
        self.update(self.da.compute(), dask_array=dask_array, chunks=chunks)

    def deepcopy(self) -> "Dataset":
        """
        Creates a deep copy of the Dataset object.

        Returns
        -------
        Dataset
            A deep copy of the Dataset object.
        """
        # Return a deep copy of the dataset
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
        # Store the old coordinates if this is not the initial update
        if not init:
            old_coords = self.da.coords

        # print(type(array))

        # Check if the input array is an xarray DataArray
        if type(array) == xr.core.dataarray.DataArray:
            # Coerce data into dask array if necessary
            if dask_array:
                # If the array data is not already a Dask array, convert it
                if type(array.data) != dask.array.core.Array:
                    self.da = xr.DataArray(
                        data=dask.array.from_array(array.data, chunks=chunks),
                        dims=array.dims,
                        coords=array.coords,
                    )
                else:
                    self.da = array
            else:
                # If the array data is not a Dask array and dask_array is False, compute it
                if type(array.data) != dask.array.core.Array:
                    self.da = array.compute()
                else:
                    self.da = array
        else:
            # Drop specified dimensions if drop_dims is provided
            if drop_dims is not None:
                dargs = dict()
                for dd in drop_dims:
                    dargs[dd] = 0
                self.da = self.da.isel(**dargs)
                for dd in drop_dims:
                    self.da = self.da.drop(dd)
                    
            # Convert the array to a Dask array if it is not already and dask_array is True
            if type(array) != dask.array.core.Array and dask_array:
                array = dask.array.from_array(array)

            # Update the data array with new dimensions if provided, else directly assign the array
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
        # Use the grid resolution as the default buffer size if none is provided
        if buffer is None:
            buffer = self.grid.resolution

        # Convert centroids to cells within the given georegions and buffer
        # mask = self.grid.mask(georegions, buffer=buffer)
        cells = self.grid.centroids_to_cell(georegions, buffer=buffer)
        # if dtype == 'gpd':

        # Create an xarray DataArray from the cells
        cells = xr.DataArray(
            data=cells,
            dims=["region", "latitude", "longitude"],
            coords=dict(
                latitude=("latitude", self.latitude.values),
                longitude=("longitude", self.longitude.values),
                region=("region", georegions.regions),
            ),
        )

        # Return the cells as an xarray DataArray if dtype is "xarray"
        if dtype == "xarray":
            return cells
        # Convert to a GeoDataFrame or GeoRegions if dtype is "gdf" or "georegions"
        elif dtype == "gdf" or dtype == "georegions":
            cells.name = "geometry"
            out = cells.to_dataframe()
            df = out.loc[np.logical_not(out.geometry.isnull())]
            df = df.reset_index()
            # Return as a GeoDataFrame if dtype is "gdf"
            if dtype == "gdf":
                return gpd.GeoDataFrame(df)
            # Return as GeoRegions if dtype is "georegions"
            elif dtype == "georegions":
                # If maxsize is None, create a unique cell ID for each region
                if maxsize is None:
                    count = df.groupby(["region"]).cumcount() + 1
                    df["cellid"] = [
                        f"{df.region[i]}.{count[i]}" for i in range(len(df.region))
                    ]
                else:
                    # Create subregions and unique cell IDs within each subregion
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

                # Create and return a GeoRegions object
                gdf = GeoRegions(gpd.GeoDataFrame(df), "cellid")
                return GeoRegions(gpd.GeoDataFrame(df), "cellid")
        else:
            # Raise a NotImplementedError if the dtype is not supported
            return NotImplementedError

    def sel(self, **kwargs) -> None:
        """
        Selects data by label along the specified dimensions.

        Parameters
        ----------
        **kwargs : dict
            The dimensions and labels to select.
        """
        # Create a reference to the data array
        da = self.da
        # Iterate through the keyword arguments to select data along specified dimensions
        for k in kwargs.keys():
            d = {k: kwargs[k]}
            da = da.sel(d).expand_dims(k).transpose(*self.da.dims)
        # Update the dataset with the selected data array
        self.update(da)

    def rescale_longitude(self) -> None:
        """
        Rescales the longitude of the dataset.
        """
        # Update Longitude coordinates
        if self.lon_is_360:
            # Convert longitude from 0-360 to -180 to 180
            self.update(array_lon_to_180(self.da))
            # Set the flag to indicate longitude is now in -180 to 180 format
            self.lon_is_360 = False
        else:
            # Convert longitude from -180 to 180 to 0-360
            self.update(array_lon_to_360(self.da))
            # Set the flag to indicate longitude is now in 0-360 format
            self.lon_is_360 = True
        # Regenerate attributes
        # Update the longitude attribute with the new longitude values
        self.longitude = self.da.longitude
        # Update the latitude attribute (remains unchanged but reassign for consistency)
        self.latitude = self.da.latitude
        # Reinitialize the grid with the updated longitude and latitude values
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
        # Apply the power operation to each block of the data array
        arr = self.da.data.map_blocks(_power, exp)
        if update:
            # Update the data array in place
            self.update(arr)
            # Record the operation in the history
            self.history.append(f"power{exp}")
        else:
            # Create a deep copy of the dataset
            slf = self.deepcopy()
            # Update the copied dataset with the new data array
            slf.update(arr)
            # Record the operation in the history of the copied dataset
            slf.history.append(f"power{exp}")
            # Return the updated copy
            return slf

    def spline(self) -> Optional["Dataset"]:
        # Construct design matrix for linear spline
        design_matrix = lambda x: (x > 20) * (x-20)
        arr = self.da.map_blocks(design_matrix)
        slf = self.deepcopy()
        slf.update(arr)
        return (self, slf)
    
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
        # If the input is a Dataset object, extract the data array
        if type(inter) == Dataset:
            inter = inter.da.data

        # Ensure the shapes of the two data arrays match
        assert self.da.data.shape == inter.shape

        # Apply the interaction (multiplication) operation to each block of the data array
        arr = self.da.data.map_blocks(_interact, inter)
        if update:
            # Update the data array in place
            self.update(arr)
            # Record the operation in the history
            self.history.append("interacted")
        else:
            # Create a deep copy of the dataset
            slf = self.deepcopy()
            # Update the copied dataset with the new data array
            slf.update(arr)
            # Record the operation in the history of the copied dataset
            slf.history.append("interacted")
            # Return the updated copy
            return slf
    
    def __repr__(self):
        return self.da.__repr__()

    def _repr_html_(self):
        return self.da._repr_html_()

# @numba.njit(fastmath=True, parallel=True)
def _power(array, exp):
    """
    Raises each element in the array to the specified power.

    Parameters
    ----------
    array : numpy.ndarray
        The input array.
    exp : int
        The exponent to raise each element to.

    Returns
    -------
    numpy.ndarray
        The array with each element raised to the specified power.
    """
    return np.power(array, exp)


# @numba.njit(fastmath=True, parallel=True)
def _interact(array, inter):
    """
    Multiplies each element in the array by the corresponding element in another array.

    Parameters
    ----------
    array : numpy.ndarray
        The input array.
    inter : numpy.ndarray
        The array to multiply with.

    Returns
    -------
    numpy.ndarray
        The array with each element multiplied by the corresponding element in the other array.
    """
    return np.multiply(array, inter)


# Object-store URL schemes and the fsspec backend each one needs. aggfly never
# imports these itself -- xarray -> zarr -> fsspec dispatches on the scheme --
# so a missing backend otherwise surfaces as an opaque error from deep inside
# fsspec. Checking up front lets us name the extra that fixes it.
_REMOTE_BACKENDS = {
    "gs": ("gcsfs", "gcs"),
    "gcs": ("gcsfs", "gcs"),
    "s3": ("s3fs", "s3"),
}


# Suffixes that are definitely not a zarr store. Probing these would cost a
# pointless stat call on every ordinary NetCDF read.
_NON_ZARR_SUFFIXES = (
    ".nc", ".nc4", ".netcdf", ".cdf", ".h5", ".hdf5",
    ".grib", ".grb", ".grib2", ".tif", ".tiff",
)

# Metadata objects that mark the root of a zarr store: v3, v2-consolidated, and
# plain v2 group/array respectively.
_ZARR_MARKERS = ("zarr.json", ".zmetadata", ".zgroup", ".zarray")


def _looks_like_zarr(path: str, storage_options: Optional[Dict] = None) -> bool:
    """
    Best-effort detection of a zarr store.

    The name alone is not enough: Pangeo's CMIP6 collection lives at paths like
    ``gs://cmip6/.../v20180701/`` with no ``.zarr`` anywhere, so a substring
    check sends them to the NetCDF backend. When the name is inconclusive, look
    for zarr's root metadata instead. Any failure here is non-fatal -- we simply
    fall back to xarray's own engine selection.
    """
    if not isinstance(path, str):
        return False
    lowered = path.lower().rstrip("/")
    if ".zarr" in lowered:
        return True
    if lowered.endswith(_NON_ZARR_SUFFIXES):
        return False
    try:
        import fsspec

        fs, _, paths = fsspec.get_fs_token_paths(
            path, storage_options=storage_options or {}
        )
        root = paths[0].rstrip("/")
        return any(fs.exists(f"{root}/{marker}") for marker in _ZARR_MARKERS)
    except Exception:
        return False


def _require_remote_backend(path: Union[str, List[str]]) -> None:
    """Raise a clear error if `path` needs an object-store backend that is missing."""
    paths = path if isinstance(path, (list, tuple)) else [path]
    for p in paths:
        if not isinstance(p, str) or "://" not in p:
            continue
        scheme = p.split("://", 1)[0].lower()
        entry = _REMOTE_BACKENDS.get(scheme)
        if entry is None:
            continue
        module, extra = entry
        if importlib.util.find_spec(module) is None:
            raise ImportError(
                f"Reading {scheme}:// paths requires the '{module}' package, which is "
                f'not installed. Install it with:  pip install "aggfly[{extra}]"'
            )


def dataset_from_path(
    path: Union[str, List[str]],
    var: str,
    xycoords: Tuple[str, str] = ("longitude", "latitude"),
    timecoord: str = "time",
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
    parallel = True,
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
    timecoord : str, optional
        The name of the time coordinate (default is "time").
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
    preprocess_at_load : bool, optional
        Whether to preprocess the data at load time (default is False).

    Returns
    -------
    Dataset
        The loaded Dataset.
    """
    
    # Fail fast with an actionable message if an object-store path needs a
    # backend that isn't installed.
    _require_remote_backend(path)

    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        if "*" in path or type(path) is list:
        
                # Load multiple files as a single dataset
                array = xr.open_mfdataset(
                    path, preprocess=preprocess, chunks=chunks, parallel=parallel, **kwargs
                )
                
                # array = xr.open_mfdataset(
                #     path
                # )               

                preprocess = None

        else:
            # Load a single file. An explicit engine= always wins; otherwise
            # detect a zarr store (by name, else by probing for its metadata).
            engine = kwargs.pop("engine", None)
            if engine is None and _looks_like_zarr(path, kwargs.get("storage_options")):
                engine = "zarr"
            if engine is None:
                array = xr.open_dataset(path, chunks=chunks, **kwargs)
            else:
                array = xr.open_dataset(path, engine=engine, chunks=chunks, **kwargs)
        
        if preprocess_at_load:
            # Apply preprocessing function at load time if specified
            array = preprocess(array)
            preprocess=None
            
        # Select the variable of interest
        array = array[var]

    return Dataset(
        array,
        xycoords=xycoords,
        timecoord=timecoord,
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
