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
        if raster.rio.crs is None:
            raise ValueError("Raster does not have a CRS assigned to it. Specify a CRS, e.g., `crs='WGS84'`.")
        self.raster = raster
        self.wtype = "none"
        self.name = name
        self.path = path
        self.project_dir = project_dir

    def rescale_raster_to_grid(
        self,
        grid,
        verbose=False,
        resampling=Resampling.average,
        nodata=0,
        return_raw=False,
    ):
        gdict = {"func": "rescale_raster_to_grid", "grid": clean_object(grid)}

        if self.cache is not None:
            cache = self.cache.uncache(gdict)
        else:
            cache = None

        if cache is not None:
            print(f"Loading rescaled {self.wtype} weights from cache")
            self.raster = cache.drop('spatial_ref').to_array()
            if verbose:
                print("Cache dictionary:")
                pprint(gdict)
        else:
            print(f"Rescaling {self.wtype} weights to grid.")
            print("This might take a few minutes and use a lot of memory...")

            g = xr.DataArray(
                    data=np.empty_like(grid.centroids()),
                    dims=["y", "x"],
                    coords=dict(
                        x=(["x"], np.float64(grid.longitude.values)), 
                        y=(["y"], np.float64(grid.latitude.values))
                    )
                ).rio.write_crs("WGS84")

            dsw = self.raster.rio.reproject_match(g, nodata=nodata, resampling=resampling)
            
            dsw = dsw.rename({'x': 'longitude', 'y': 'latitude'}).squeeze()

            if return_raw:
                return dsw.values.squeeze()

            self.raster = dsw

            if self.cache is not None:
                self.cache.cache(self.raster, gdict)
        
    def cdict(self):
        gdict = {
            "wtype": self.wtype,
            "name": self.name,
            "path": self.path,
            "raster": pformat(self.raster),
        }
        return gdict
    
class SecondaryWeights(RasterWeights):
    def __init__(self, raster, name=None, path=None, project_dir=None, wtype="raster"):
        super().__init__(raster, name, path, project_dir)
        if self.raster.rio.crs is None:
            raise ValueError("Raster does not have a CRS assigned to it. Specify a CRS, e.g., `crs='WGS84'`.")
        self.wtype = wtype
        self.cache = initialize_cache(self)
        
def secondary_weights_from_path(
    path, name=None, project_dir=None, crs=None, wtype="raster"
):
    da = open_raster(path)
    
    if crs is not None:
        da = da.rio.write_crs(crs)
    
    weights = SecondaryWeights(da, name, path, project_dir, wtype)

    return weights

    
def open_raster(path, preprocess=None, **kwargs):
    # Separate file path from file extension
    file, ex = os.path.splitext(path)

    if ex == ".tif":
        da = rioxarray.open_rasterio(
            path, chunks=True, lock=False, masked=True, **kwargs
        )

        if preprocess is not None:
            da = preprocess(da)

    # elif ex =='.zarr':
    #     da = xr.open_zarr(path,  **kwargs)
    #     da = da.layer.sel(crop=crop)
    # elif ex == '.nc':
    #     da = xr.open_dataset(path,  **kwargs)
    #     da = da.layer.sel(crop=crop)
    else:
        raise NotImplementedError

    return da