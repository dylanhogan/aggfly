# API reference

Everything below is re-exported from the top-level package:

```python
import aggfly as af
```

Signatures are abbreviated to the arguments you are most likely to set; all
functions accept additional keyword arguments. See the docstrings for full detail.

## Regions

### `georegions_from_path`

```python
af.georegions_from_path(path, regionid, region_list=None) -> GeoRegions
```

Load target regions from a shapefile. `regionid` names the column holding the
region identifier; `region_list` optionally restricts to a subset.

### `georegions_from_name`

```python
af.georegions_from_name(name="usa", region_list=None) -> GeoRegions
```

Load a bundled region set by name.

### `GeoRegions`

Wraps a GeoDataFrame of target regions.

## Datasets

### `dataset_from_path`

```python
af.dataset_from_path(
    path, var,
    xycoords=("longitude", "latitude"),
    timecoord="time",
    time_sel=None,
    georegions=None,
    lon_is_360=True,
    preprocess=None,
    name=None,
    chunks={"time": 24, "latitude": -1, "longitude": -1},
    **kwargs,
) -> Dataset
```

Open a raster (Zarr/netCDF; `path` may be a list) as a dask-backed `Dataset`.
`preprocess` is applied to raw values (e.g. Kelvin→Celsius). Passing `georegions`
clips the raster to the region extent as a read optimization.

Use `aggfly info <path>` to discover the right values for `xycoords`, `timecoord`,
and `lon_is_360`.

### `zarr_from_path` / `dataset_to_zarr`

```python
af.zarr_from_path(path, var, store, ...) -> Dataset
af.dataset_to_zarr(dataset, store, chunking="auto", ...)
```

Read from / write to a Zarr store.

### `Dataset`

Wraps an `xarray.DataArray` (`.da`), normalizing dimension names and the 0–360 vs
−180–180 longitude convention (`lon_is_360`).

### `Grid`

Describes the raster's lon/lat mesh, cell ids, resolution, and cell areas. Derived
from a `Dataset` and used for weight computation.

## Weights

### `weights_from_objects`

```python
af.weights_from_objects(
    clim, georegions,
    secondary_weights=None,
    wtype=None, name=None,
    crop="corn", feed=None,
    write=False,
    project_dir=None,
    **kwargs,
) -> GridWeights
```

Build a `GridWeights` object. Call `.calculate_weights()` to populate `.weights`.
`project_dir` enables the cache. See [Weights](guide/weights.md).

> `calculate_weights()` shuts down an active Dask client — compute weights before
> starting an execution client.

### Secondary-weight loaders

```python
af.pop_weights_from_path(path, grid=None, name=None, feed=None, project_dir=None, crs=None)
af.crop_weights_from_path(path, crop="corn", grid=None, ...)
af.secondary_weights_from_path(path, name=None, project_dir=None, crs=None, wtype="raster")
```

### `GridWeights`, `SecondaryWeights`, `PopWeights`, `CropWeights`

Weight classes. `GridWeights.weights` is a DataFrame keyed by `cell_id`/`region_id`
with `area_weight`, `raster_weight`, and combined `weight` columns.

## Aggregation

### `aggregate_dataset`

```python
af.aggregate_dataset(
    weights, dataset=None,
    aggregator_dict=None,
    dataset_dict=None,
    engine="auto",
    **named_specs,
) -> pd.DataFrame
```

The main entry point: runs temporal then spatial aggregation and returns a panel
merged onto the region ids. Output variables are given either as `**named_specs`
keyword arguments or as an `aggregator_dict`. See [Aggregation](guide/aggregation.md)
for the spec DSL and [Execution](guide/execution.md) for `engine`.

### `aggregate_time` / `aggregate_space`

```python
af.aggregate_time(dataset, weights=None, aggregator_dict=None, engine="auto", ...) -> Dict[str, Dataset]
af.aggregate_space(dataset_dict, weights, npartitions=None, **kwargs) -> pd.DataFrame
```

The two halves of `aggregate_dataset`, usable separately.

### `TemporalAggregator` / `SpatialAggregator`

The objects built from the spec DSL and used internally by the two stages.

## Execution

```python
af.start_dask_client(n_workers=2, threads_per_worker=2, cap_numba_threads=1, ...)
af.shutdown_dask_client()
af.is_distributed() -> bool
af.distributed_client()
```

Helpers for managing a `LocalCluster`. `cap_numba_threads=1` prevents
`n_workers` × numba threads from oversubscribing the machine. See
[Execution & scaling](guide/execution.md).

## Command-line interface

The package installs an `aggfly` executable covering the same pipeline from a YAML
config. See the [CLI reference](cli.md).
