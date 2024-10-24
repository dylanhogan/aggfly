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
    def __init__(self, raster, name=None, path=None, project_dir=None, wtype="raster"):
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
        """
        super().__init__(raster, name, path, project_dir) # Initialize the base class
        if self.raster.rio.crs is None:
            raise ValueError("Raster does not have a CRS assigned to it. Specify a CRS, e.g., `crs='WGS84'`.")
        self.wtype = wtype # Set the weight type
        self.cache = initialize_cache(self) # Initialize cache for the SecondaryWeights object
        
def secondary_weights_from_path(
    path, name=None, project_dir=None, crs=None, wtype="raster"
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
    da = open_raster(path) # Open the raster file
    
    if crs is not None:
        da = da.rio.write_crs(crs) # Write CRS to the raster if provided
    
    weights = SecondaryWeights(da, name, path, project_dir, wtype) # Create SecondaryWeights object

    return weights

    
def open_raster(path, preprocess=None, **kwargs):
    """
    Open a raster file.

    Parameters
    ----------
    path : str
        The path to the raster file.
    preprocess : callable, optional
        A function to preprocess the data when loaded (default is None).

    Returns
    -------
    xarray.DataArray
        The opened raster data array.
    """
    # Separate file path from file extension
    file, ex = os.path.splitext(path)

    if ex == ".tif":
        da = rioxarray.open_rasterio(
            path, chunks=True, lock=False, masked=True, **kwargs
        )

        if preprocess is not None:
            da = preprocess(da)  # Apply preprocessing if provided

    # elif ex =='.zarr':
    #     da = xr.open_zarr(path,  **kwargs)
    #     da = da.layer.sel(crop=crop)
    # elif ex == '.nc':
    #     da = xr.open_dataset(path,  **kwargs)
    #     da = da.layer.sel(crop=crop)
    else:
        raise NotImplementedError

    return da  # Return the opened raster data array
