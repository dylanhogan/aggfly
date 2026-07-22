# This script defines the RasterWeights class and its subclass SecondaryWeights, used for variables such as population or crop land covered, for handling raster-based weights.
# It includes methods for initializing these objects, rescaling rasters to match a grid, caching results, and creating weights from file paths.

from pprint import pformat, pprint
import numpy as np
import xarray as xr
from rasterio.enums import Resampling
import os
import rioxarray

from ..dataset import reformat_grid
from ..cache import *


class RasterWeights:
    def __init__(self, raster, name=None, path=None, project_dir=None):
        """
        Initialize a RasterWeights object.

        Parameters
        ----------
        raster : xarray.DataArray
            The raster data array.
        name : str, optional
            The name of the weights (default is None).
        path : str, optional
            The path to the raster file (default is None).
        project_dir : str, optional
            The project directory (default is None).
        """
        # Check if the raster has a CRS assigned
        if raster.rio.crs is None:
            raise ValueError("Raster does not have a CRS assigned to it. Specify a CRS, e.g., `crs='WGS84'`.")
        self.raster = raster # Store the raster data array
        self.wtype = "none" # Default weight type
        self.name = name # Store the name of the weights
        self.path = path # Store the path to the raster file
        self.project_dir = project_dir # Store the project directory

    def rescale_raster_to_grid(
        self,
        grid,
        verbose=False,
        resampling=Resampling.average,
        nodata=0,
        return_raw=False,
    ):
        """
        Rescale the raster to match a specified grid.

        Parameters
        ----------
        grid : Grid
            The grid to rescale the raster to.
        verbose : bool, optional
            Whether to print verbose output (default is False).
        resampling : Resampling method, optional
            The resampling method to use (default is Resampling.average).
        nodata : int, optional
            The value to use for nodata (default is 0).
        return_raw : bool, optional
            Whether to return the raw rescaled values (default is False).

        Returns
        -------
        numpy.ndarray, optional
            The rescaled raster values if return_raw is True.
        """
        gdict = {"func": "rescale_raster_to_grid", "grid": clean_object(grid)}

        # Check if cache is available
        if self.cache is not None:
            cache = self.cache.uncache(gdict)
        else:
            cache = None

        if cache is not None:
            print(f"Loading rescaled {self.wtype} weights from cache")
            # Load the cached raster and drop spatial reference
            self.raster = cache.drop('spatial_ref').to_array()
            if verbose:
                print("Cache dictionary:")
                pprint(gdict)
        else:
            print(f"Rescaling {self.wtype} weights to grid.")
            print("This might take a few minutes and use a lot of memory...")

            # Create a new DataArray with the grid's centroids and write CRS
            g = xr.DataArray(
                    data=np.empty_like(grid.centroids()), # Create an empty array with the same shape as the grid centroids
                    dims=["y", "x"], # Define the dimensions
                    coords=dict(
                        x=(["x"], np.float64(grid.longitude.values)),  # Define the longitude coordinates
                        y=(["y"], np.float64(grid.latitude.values)) # Define the latitude coordinates
                    )
                ).rio.write_crs("WGS84") # Write the CRS as WGS84

            # Resample the raster to match the grid
            dsw = self.raster.rio.reproject_match(g, nodata=nodata, resampling=resampling)
            dsw = dsw.rename({'x': 'longitude', 'y': 'latitude'}).squeeze()

            if return_raw:
                return dsw.values.squeeze() # Return raw values if requested

            self.raster = dsw # Update the raster attribute

            # Cache the rescaled raster
            if self.cache is not None:
                self.cache.cache(self.raster, gdict)
        
    def cdict(self):
        """
        Get a dictionary representation of the RasterWeights object.

        Returns
        -------
        dict
            The dictionary representation of the RasterWeights object.
        """
        gdict = {
            "wtype": self.wtype, # Weight type
            "name": self.name,  # Name of the weights
            "path": self.path, # Path to the raster file
            "raster": pformat(self.raster), # Formatted raster data
        }
        return gdict
    
class SecondaryWeights(RasterWeights):
    def __init__(self, raster, name=None, path=None, project_dir=None,
                 wtype="raster", cache_identifier=None):
        """
        Initialize a SecondaryWeights object.

        Parameters
        ----------
        raster : xarray.DataArray
            The raster data array.
        name : str, optional
            The name of the weights (default is None).
        path : str, optional
            The path to the raster file (default is None).
        project_dir : str, optional
            The project directory (default is None).
        wtype : str, optional
            The weight type (default is "raster").
        cache_identifier : hashable, optional
            An extra discriminator folded into the cache key. Use it when two
            variants of the same weights would otherwise collide — that is, when
            what distinguishes them is *not* already visible in ``path`` or in
            the raster itself (which both already feed the key). A `preprocess`
            applied before construction is the usual case.
        """
        super().__init__(raster, name, path, project_dir) # Initialize the base class
        if self.raster.rio.crs is None:
            raise ValueError("Raster does not have a CRS assigned to it. Specify a CRS, e.g., `crs='WGS84'`.")
        self.wtype = wtype # Set the weight type
        self.cache_identifier = cache_identifier
        self.cache = initialize_cache(self) # Initialize cache for the SecondaryWeights object

    def cdict(self):
        """Cache key: the base fields plus the extra discriminator."""
        gdict = super().cdict()
        gdict["cache_identifier"] = self.cache_identifier
        return gdict


def secondary_weights_from_path(
    path, name=None, project_dir=None, crs=None, wtype="raster",
    var=None, sel=None, cache_identifier=None, preprocess=None, **kwargs
):
    """
    Create SecondaryWeights from a file path.

    Parameters
    ----------
    path : str
        The path to the raster file.
    name : str, optional
        The name of the weights (default is None).
    project_dir : str, optional
        The project directory (default is None).
    crs : str, optional
        The coordinate reference system to use (default is None).
    wtype : str, optional
        The weight type (default is "raster").

    Returns
    -------
    SecondaryWeights
        The created SecondaryWeights object.
    """
    da = open_raster(path, var=var, sel=sel, preprocess=preprocess, **kwargs)
    if crs is not None:
        da = da.rio.write_crs(crs) # Write CRS to the raster if provided
    return SecondaryWeights(
        da, name=name, path=path, project_dir=project_dir,
        wtype=wtype, cache_identifier=cache_identifier,
    )

    
def open_raster(path, var=None, sel=None, preprocess=None, **kwargs):
    """
    Open a raster as an xarray object, optionally selecting part of it.

    Parameters
    ----------
    path : str
        Path to a GeoTIFF, Zarr store, or NetCDF file.
    var : str, optional
        Data variable to take from a multi-variable file, e.g. ``"layer"`` for a
        cropland store. Ignored for a plain single-band GeoTIFF.
    sel : dict, optional
        Coordinate selection applied after ``var``, passed to ``.sel()`` —
        e.g. ``{"crop": "corn"}`` to pick one crop out of a cropland store, or
        ``{"band": 1}`` to pick a band. This replaces the crop-specific loader:
        any coordinate can be selected, not just a crop.
    preprocess : callable, optional
        Applied to the opened object before ``var``/``sel``.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
    """
    ext = os.path.splitext(str(path).rstrip("/"))[1].lower()

    if ext in (".tif", ".tiff"):
        da = rioxarray.open_rasterio(path, chunks=True, lock=False, masked=True, **kwargs)
    elif ext == ".zarr":
        da = xr.open_zarr(path, **kwargs)
    elif ext in (".nc", ".nc4", ".netcdf", ".cdf"):
        da = xr.open_dataset(path, **kwargs)
    else:
        raise NotImplementedError(
            f"Unsupported raster format {ext!r} for {path!r}. "
            "Supported: .tif/.tiff, .zarr, .nc/.nc4."
        )

    if preprocess is not None:
        da = preprocess(da)
    if var is not None:
        da = da[var]
    if sel:
        da = da.sel(**sel)

    return da
