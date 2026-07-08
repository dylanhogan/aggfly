# The following code defines tests for various functions related to the package. 
# These tests are useful for ensuring that the dataset transformation, aggregation, 
# and weighting functions work correctly. 

import os
import pytest

import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely

import aggfly as af

def dataset_360():
    """
    Create a sample dataset with longitude ranging from 0 to 360 degrees.
    
    Returns:
    --------
    af.Dataset
        The created dataset.
    """
    # Set random seed for reproducibility
    np.random.seed(1216)
    # Generate evenly spaced values from 0 to 360
    x = np.linspace(0, 360, 3)
    # Calculate midpoints for longitude
    longitude = (x[1:] + x[:-1]) / 2
    # Generate evenly spaced values from -90 to 90
    y = np.linspace(-90, 90, 3)
    # Calculate midpoints for latitude
    latitude = (y[1:] + y[:-1]) / 2

    # Generate time range
    time = pd.date_range('2000-07-01', periods=4, freq='12h')

    # Create random data
    arr = np.random.normal(20, 15, (len(time), len(latitude), len(longitude)))

    # Return dataset with longitude 0-360
    da = xr.DataArray(
        data = arr,
        dims = ['time', 'latitude', 'longitude'],
        coords = {
            'time': time,
            'latitude': latitude,
            'longitude': longitude,
        }
    )
    return af.Dataset(da, lon_is_360=True)

@pytest.fixture(name='dataset_360')
def dataset_360_fixture():
    """
    Fixture for the dataset_360 function.
    
    Returns:
    --------
    af.Dataset
        The created dataset.
    """
    return dataset_360()

def georegion():
    """
    Create a sample georegion using random points.
    
    Returns:
    --------
    af.GeoRegions
        The created georegion.
    """
    # Set random seed for reproducibility
    np.random.seed(1216)
    # Generate random longitudes
    longitude = np.random.uniform(-180, 180, 20)
    # Generate random latitudes
    latitude = np.random.uniform(-90, 90, 20)
    
    # Create convex hull polygon from points (shapely.union_all is stable across
    # geopandas 0.14/1.x, unlike the deprecated GeoDataFrame.unary_union)
    polygon = shapely.union_all(
        np.asarray(gpd.points_from_xy(longitude, latitude))
    ).convex_hull
    gdf = gpd.GeoDataFrame(
        {
            'geoid' : 'region_1',
            'geometry': [polygon]
        }
    )
    # Set coordinate reference system
    gdf = gdf.set_crs('WGS84')
    # Return georegion
    return af.GeoRegions(gdf, regionid='geoid')

@pytest.fixture(name='georegion')
def georegion_fixture():
    """
    Fixture for the georegion function.
    
    Returns:
    --------
    af.GeoRegions
        The created georegion.
    """
    return georegion()

def secondary_weights():
    """
    Create a sample secondary weights dataset.
    
    Returns:
    --------
    af.SecondaryWeights
        The created secondary weights dataset.
    """
    # Set random seed for reproducibility
    np.random.seed(1216)

    # Generate evenly spaced values for longitude
    x = np.linspace(-180, 180, 5)
    # Calculate midpoints for longitude
    longitude = (x[1:] + x[:-1]) / 2

    # Generate evenly spaced values for latitude
    y = np.linspace(-90, 90, 5)
    # Calculate midpoints for latitude
    latitude = (y[1:] + y[:-1]) / 2

    # Create random data
    arr = np.random.rand(1, len(latitude), len(longitude))
    
    da = xr.DataArray(
        data = arr,
        dims = ['band', 'y', 'x'],
        coords = {
            'band' : [1],
            'y': latitude,
            'x': longitude,
        }
    )
    # Write coordinate reference system
    da = da.rio.write_crs("WGS84") 
    # Return secondary weights dataset
    return af.SecondaryWeights(da)

@pytest.fixture(name='secondary_weights')
def secondary_weights_fixture():
    """
    Fixture for the secondary_weights function.
    
    Returns:
    --------
    af.SecondaryWeights
        The created secondary weights dataset.
    """
    return secondary_weights()

def weights(dataset_360, georegion, secondary_weights):
    """
    Calculate weights for the dataset and georegion.
    
    Returns:
    --------
    af.GridWeights
        The calculated weights.
    """
    # Create weights object
    w = af.weights_from_objects(dataset_360, georegion, secondary_weights)
    # Calculate weights
    w.calculate_weights()
    # Sort weights by cell ID
    w.weights = w.weights.sort_values('cell_id')
    return w

@pytest.fixture(name='weights')
def weights_fixture(dataset_360, georegion, secondary_weights):
    """
    Fixture for the weights function.
    
    Returns:
    --------
    af.GridWeights
        The calculated weights.
    """
    return weights(dataset_360, georegion, secondary_weights)

def test_weights(weights):
    """
    Test the calculated weights.
    
    Parameters:
    -----------
    weights : af.GridWeights
        The calculated weights.
    """    
    # Check type of weights
    assert isinstance(weights, af.GridWeights)
    # Check type of grid
    assert isinstance(weights.grid, af.Grid)
    # Check type of georegions
    assert isinstance(weights.georegions, af.GeoRegions)
    # Check type of raster weights
    assert isinstance(weights.raster_weights, af.SecondaryWeights)
    # Print weights
    print(weights.weights)
    # Check area weights (cosine-of-latitude corrected: cells here are at lat +-45)
    assert np.allclose(
        weights.weights.area_weight,
        np.array([0.62062975, 0.59818877, 0.30089622, 0.66666658])
    )
    # Check raster weights
    assert np.allclose(
        weights.weights.raster_weight,
        np.array([0.67392287, 0.80659155, 0.56727215, 0.38801016])
    )
    # Check final weights
    assert np.allclose(
        weights.weights.weight,
        np.array([0.17171243, 0.19808467, 0.07007565, 0.10619663])
    )

def test_aggregate_time(dataset_360, weights):
    """
    Test the time aggregation function.
    
    Parameters:
    -----------
    dataset_360 : af.Dataset
        The dataset to aggregate.
    weights : af.GridWeights
        The weights to use for aggregation.
    
    This test performs several aggregations on the dataset over time, 
    using different aggregation functions and transformations. It 
    verifies that the aggregated results are correct by comparing 
    them with expected values.
    """
    adict = af.aggregate_time(
        dataset=dataset_360, 
        weights=weights,
            bins= [
                ('aggregate', {'calc':'mean', 'groupby':'date'}), # Mean aggregation by date
                ('aggregate', {'calc':'bins', 'groupby':'month', 'ddargs':[[-99,20,0],[20,99,0]]}) # Bin aggregation by month
            ],
            cooling_dday = [
                ('aggregate', {'calc':'dd', 'groupby':'date', 'ddargs':[20,99,0]}), # Degree day aggregation by date
                ('aggregate', {'calc':'sum', 'groupby':'month'}) # Sum aggregation by month
            ],
            tavg = [
                ('aggregate', {'calc':'mean', 'groupby':'date'}), # Mean aggregation by date
                ('transform', {'transform':'power', 'exp':np.arange(1,3)}), # Polynomial transformation
                ('aggregate', {'calc':'sum', 'groupby':'month'}) # Sum aggregation by month
            ]    
        )
    # Combine results into a DataFrame
    df = xr.combine_by_coords([ adict[x].da.rename(x) for x in adict.keys()]).to_dataframe()
    # Check that the aggregated values match expected values
    assert np.allclose(df.values, 
        np.array([[   0.      ,    2.      ,   44.945648,   62.472824, 1956.361671],
                [   1.      ,    1.      ,   25.910298,   39.60287 ,  801.80304 ],
                [   1.      ,    1.      ,    9.12584 ,   35.789426,  670.521066],
                [   1.      ,    1.      ,   14.932308,   37.648473,  858.069229]])
    )


def test_aggregate(dataset_360, weights):
    """
    Test the dataset aggregation function.

    Parameters:
    -----------
    dataset_360 : af.Dataset
        The dataset to aggregate.
    weights : af.GridWeights
        The weights to use for aggregation.

    This test performs the following operations:
    1. Aggregates the dataset by calculating the mean by date.
    2. Transforms the aggregated data by raising it to the power of 1 and 2.
    3. Aggregates the transformed data by summing it by month.
    It then checks if the aggregated values match the expected results.
    """
    df = af.aggregate_dataset(
        dataset=dataset_360,
        weights=weights,
        tavg = [
                ('aggregate', {'calc':'mean', 'groupby':'date'}), # Aggregate by mean per date
                ('transform', {'transform':'power', 'exp':np.arange(1,3)}), # Transform by raising to power of 1 and 2
                ('aggregate', {'calc':'sum', 'groupby':'month'}), # Aggregate by summing per month
        ]
    )

    # Check if the aggregated values are as expected
    assert np.allclose(df[['tavg_1', 'tavg_2']].values,
        np.array([[  46.906441, 1202.304441]])
    )


def test_aggregate_time_numba(dataset_360, weights):
    # The numba engine must be bit-equivalent to the dask path (and the
    # hardcoded expectations) across mean/sum/dd/bins/multi-dd/power.
    adict = af.aggregate_time(
        dataset=dataset_360,
        weights=weights,
        engine="numba",
            bins= [
                ('aggregate', {'calc':'mean', 'groupby':'date'}),
                ('aggregate', {'calc':'bins', 'groupby':'month', 'ddargs':[[-99,20,0],[20,99,0]]})
            ],
            cooling_dday = [
                ('aggregate', {'calc':'dd', 'groupby':'date', 'ddargs':[20,99,0]}),
                ('aggregate', {'calc':'sum', 'groupby':'month'})
            ],
            tavg = [
                ('aggregate', {'calc':'mean', 'groupby':'date'}),
                ('transform', {'transform':'power', 'exp':np.arange(1,3)}),
                ('aggregate', {'calc':'sum', 'groupby':'month'})
            ]
        )
    df = xr.combine_by_coords([ adict[x].da.rename(x) for x in adict.keys()]).to_dataframe()
    assert np.allclose(df.values,
        np.array([[   0.      ,    2.      ,   44.945648,   62.472824, 1956.361671],
                [   1.      ,    1.      ,   25.910298,   39.60287 ,  801.80304 ],
                [   1.      ,    1.      ,    9.12584 ,   35.789426,  670.521066],
                [   1.      ,    1.      ,   14.932308,   37.648473,  858.069229]])
    )


def test_aggregate_numba(dataset_360, weights):
    df = af.aggregate_dataset(
        dataset=dataset_360,
        weights=weights,
        engine="numba",
        tavg = [
                ('aggregate', {'calc':'mean', 'groupby':'date'}),
                ('transform', {'transform':'power', 'exp':np.arange(1,3)}),
                ('aggregate', {'calc':'sum', 'groupby':'month'}),
        ]
    )

    assert np.allclose(df[['tavg_1', 'tavg_2']].values,
        np.array([[  46.906441, 1202.304441]])
    )


from types import SimpleNamespace
from aggfly.aggregate.spatial import SpatialAggregator


class _DShim:
    """Minimal stand-in for a Dataset that SpatialAggregator can consume."""
    def __init__(self, da):
        self.da = da
        self.lon_is_360 = False
    def rescale_longitude(self):
        return self


def _wavg_oracle(vals, time, grid_cell_ids, wdf, names):
    """Independent weighted-average reference (pure loops) for parity checks.

    vals: dict name -> (n_cells, n_time) in grid_cell_ids order. A cell/time is used
    only if every name is non-NaN there; a region/time with no valid weight is dropped.
    """
    cellpos = {int(c): i for i, c in enumerate(grid_cell_ids)}
    rows = []
    for r in np.sort(wdf["index_right"].unique()):
        sub = wdf[wdf["index_right"] == r]
        cidx = sub["cell_id"].map(cellpos).to_numpy()
        wv = sub["weight"].to_numpy(dtype=float)
        for ti in range(len(time)):
            valid = np.ones(len(cidx), bool)
            for nm in names:
                valid &= ~np.isnan(vals[nm][cidx, ti])
            den = wv[valid].sum()
            if den == 0:
                continue
            row = {"region_id": r, "time": time[ti]}
            for nm in names:
                row[nm] = (wv[valid] * vals[nm][cidx, ti][valid]).sum() / den
            rows.append(row)
    return pd.DataFrame(rows)


def _run_spatial(da, wdf, names, time_chunk=None):
    """Drive the real SpatialAggregator over a 2x2-cell grid via lightweight shims."""
    if time_chunk is not None:
        da = da.chunk({"time": time_chunk})
    grid = SimpleNamespace(cell_id=np.array([0, 1, 2, 3]))
    weights = SimpleNamespace(grid=grid, weights=wdf)
    dlist = [_DShim(da.rename(nm)) for nm in names] if len(names) > 1 else [_DShim(da)]
    return SpatialAggregator(dlist, weights, names=names).compute()


def test_spatial_matmul_multiregion_nan():
    """Two regions sharing cells with fractional weights + per-timestep NaNs:
    the sparse-matmul aggregator must match an independent weighted-average oracle,
    including across multiple lazy time chunks."""
    lat = np.array([0.0, 1.0]); lon = np.array([0.0, 1.0])
    time = pd.date_range("2000-07-01", periods=3, freq="D")
    vals = np.random.default_rng(7).normal(20, 5, (3, 2, 2))
    vals[1, 0, 0] = np.nan   # cell 0 NaN at t1
    vals[2, 1, 1] = np.nan   # cell 3 NaN at t2
    da = xr.DataArray(vals, dims=["time", "latitude", "longitude"],
                      coords={"time": time, "latitude": lat, "longitude": lon})
    wdf = pd.DataFrame({
        "cell_id":     [0, 1, 2, 1, 2, 3],
        "index_right": [0, 0, 0, 1, 1, 1],
        "weight":      [0.5, 0.3, 0.2, 0.4, 0.4, 0.2],
    })
    names = ["v"]
    vals_flat = {"v": vals.reshape(3, 4).T}  # (n_cells, n_time), C-order == cell_id order
    oracle = _wavg_oracle(vals_flat, time.values, [0, 1, 2, 3], wdf, names)

    got = _run_spatial(da, wdf, names, time_chunk=1)  # 3 time chunks -> exercises map_blocks
    got = got.sort_values(["region_id", "time"]).reset_index(drop=True)
    oracle = oracle.sort_values(["region_id", "time"]).reset_index(drop=True)
    assert got.shape == oracle.shape
    assert (got[["region_id"]].values == oracle[["region_id"]].values).all()
    assert np.allclose(got["v"].values, oracle["v"].values, equal_nan=True)


def test_spatial_matmul_dropna_empty_group():
    """A region whose cells are all NaN at a timestep is dropped (den == 0),
    row-for-row like the oracle."""
    lat = np.array([0.0, 1.0]); lon = np.array([0.0, 1.0])
    time = pd.date_range("2000-07-01", periods=2, freq="D")
    vals = np.random.default_rng(3).normal(20, 5, (2, 2, 2))
    vals[0, 0, 0] = vals[0, 0, 1] = vals[0, 1, 0] = np.nan  # region 0 cells all NaN at t0
    da = xr.DataArray(vals, dims=["time", "latitude", "longitude"],
                      coords={"time": time, "latitude": lat, "longitude": lon})
    wdf = pd.DataFrame({
        "cell_id":     [0, 1, 2, 1, 2, 3],
        "index_right": [0, 0, 0, 1, 1, 1],
        "weight":      [0.5, 0.3, 0.2, 0.4, 0.4, 0.2],
    })
    names = ["v"]
    vals_flat = {"v": vals.reshape(2, 4).T}
    oracle = _wavg_oracle(vals_flat, time.values, [0, 1, 2, 3], wdf, names)

    got = _run_spatial(da, wdf, names).sort_values(["region_id", "time"]).reset_index(drop=True)
    oracle = oracle.sort_values(["region_id", "time"]).reset_index(drop=True)
    assert got.shape == oracle.shape                       # dropped group absent in both
    assert (got[["region_id", "time"]].values == oracle[["region_id", "time"]].values).all()
    assert np.allclose(got["v"].values, oracle["v"].values, equal_nan=True)


def test_auto_chunks_policy():
    from aggfly.dataset.zarr_convert import _auto_chunks
    # short series: full-time chunk, square spatial tile under budget
    c = _auto_chunks({"latitude": 721, "longitude": 1440, "time": 8784}, 4, 256)
    assert c["time"] == -1 and c["latitude"] == c["longitude"] and c["latitude"] >= 32
    # very long hourly series: time must be split, spatial tile fixed, chunk under budget
    c = _auto_chunks({"latitude": 721, "longitude": 1440, "time": 350640}, 4, 256)
    assert c["time"] > 0 and c["time"] < 350640
    assert c["time"] * c["latitude"] * c["longitude"] * 4 <= 256 * 1024 * 1024
    # tiny grid: tile capped by extent
    c = _auto_chunks({"latitude": 2, "longitude": 2, "time": 4}, 8, 256)
    assert c["latitude"] <= 2 and c["longitude"] <= 2


def _synthetic_dataset(nt, ny, nx, fill):
    import aggfly as af
    t = pd.date_range("2000-01-01", periods=nt, freq="D")
    da = xr.DataArray(
        np.full((nt, ny, nx), fill, dtype="float32"),
        dims=["time", "latitude", "longitude"],
        coords={"time": t, "latitude": np.arange(ny, dtype=float),
                "longitude": np.arange(nx, dtype=float)},
    )
    return af.Dataset(da, lon_is_360=False)


def test_dataset_to_zarr_roundtrip(dataset_360, tmp_path):
    import aggfly as af
    store = str(tmp_path / "roundtrip.zarr")
    ds = dataset_360.deepcopy()
    out = af.dataset_to_zarr(ds, store, overwrite=True)
    # store exists; round-trip returns a Dataset with time-contiguous chunks and values
    assert os.path.isdir(store)
    assert isinstance(out, af.Dataset)
    tdim = out.da.get_axis_num("time")
    assert max(out.da.chunks[tdim]) == out.da.sizes["time"]  # time is one chunk
    assert np.allclose(
        ds.da.transpose("latitude", "longitude", "time").values,
        out.da.transpose("latitude", "longitude", "time").values,
    )
    # refuses to clobber without overwrite
    with pytest.raises(FileExistsError):
        af.dataset_to_zarr(dataset_360.deepcopy(), store, overwrite=False)


def test_dataset_to_zarr_compresses(tmp_path):
    import aggfly as af
    ds = _synthetic_dataset(100, 50, 50, fill=0.0)  # ~1 MB raw, highly compressible
    store = str(tmp_path / "compress.zarr")
    af.dataset_to_zarr(ds, store, return_dataset=False, overwrite=True)
    raw = int(np.prod(list(ds.da.sizes.values()))) * ds.da.dtype.itemsize
    on_disk = sum(os.path.getsize(os.path.join(r, f))
                  for r, _, fs in os.walk(store) for f in fs)
    assert on_disk < raw  # compression happened (data >> metadata here)


def test_zarr_from_path_agnostic(tmp_path):
    """The convenience path is agnostic to source naming: a file with ERA5-style
    valid_time/lat/lon coords converts correctly via the same normalization params."""
    import aggfly as af
    t = pd.date_range("2016-06-01", periods=48, freq="h")
    src = xr.Dataset(
        {"t2m": (["valid_time", "lat", "lon"],
                 np.arange(48 * 20 * 30, dtype="float32").reshape(48, 20, 30))},
        coords={"valid_time": t, "lat": np.linspace(10, 20, 20),
                "lon": np.linspace(-100, -80, 30)},
    )
    nc = str(tmp_path / "src.nc"); src.to_netcdf(nc)
    store = str(tmp_path / "out.zarr")
    out = af.zarr_from_path(nc, var="t2m", store=store,
                            xycoords=("lon", "lat"), timecoord="valid_time",
                            lon_is_360=False)
    # dims normalized to latitude/longitude/time, time contiguous, values preserved
    assert isinstance(out, af.Dataset)
    assert set(out.da.dims) >= {"latitude", "longitude", "time"}
    tdim = out.da.get_axis_num("time")
    assert max(out.da.chunks[tdim]) == out.da.sizes["time"]
    exp = src["t2m"].transpose("lat", "lon", "valid_time").values
    got = out.da.transpose("latitude", "longitude", "time").values
    assert np.allclose(exp, got)


def _cube(nlat, nlon, latchunk, lonchunk, dask=True):
    """A small time x lat x lon DataArray, optionally dask-chunked in space."""
    t = pd.date_range("2016-01-01", periods=24, freq="h")
    da = xr.DataArray(
        np.ones((24, nlat, nlon)),
        dims=["time", "latitude", "longitude"],
        coords={"time": t, "latitude": np.arange(nlat), "longitude": np.arange(nlon)},
    )
    if dask:
        da = da.chunk({"time": -1, "latitude": latchunk, "longitude": lonchunk})
    return da


def test_max_spatial_block_cells():
    from aggfly.aggregate.nb_kernels import max_spatial_block_cells
    # dask-chunked: largest spatial block = latchunk * lonchunk
    assert max_spatial_block_cells(_cube(500, 500, 50, 50)) == 2500
    assert max_spatial_block_cells(_cube(500, 500, 250, 250)) == 62500
    # single block (-1 chunks) = full spatial extent
    assert max_spatial_block_cells(_cube(361, 361, -1, -1)) == 361 * 361
    # non-dask array = whole spatial extent
    assert max_spatial_block_cells(_cube(60, 60, 0, 0, dask=False)) == 3600


def test_resolve_engine():
    from aggfly.aggregate.nb_kernels import resolve_engine, NUMBA_MAX_CELLS_PER_BLOCK
    small = _cube(500, 500, 50, 50)     # 2500 cells/block -> below threshold
    large = _cube(500, 500, 250, 250)   # 62500 cells/block -> above threshold
    # auto picks by chunk size
    assert resolve_engine("auto", small, "mean") == "numba"
    assert resolve_engine("auto", large, "mean") == "dask"
    # explicit choices are honored for supported calcs
    assert resolve_engine("dask", small, "mean") == "dask"
    assert resolve_engine("numba", large, "mean") == "numba"
    # calcs the numba backend can't do always fall back to dask
    assert resolve_engine("auto", small, "not_a_numba_calc") == "dask"
    assert resolve_engine("numba", small, "not_a_numba_calc") == "dask"
    # invalid engine raises
    import pytest
    with pytest.raises(ValueError):
        resolve_engine("bogus", small, "mean")
