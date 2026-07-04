# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`aggfly` is a Python package for spatial and temporal aggregation of gridded climate data (e.g. ERA5) onto administrative regions defined by a shapefile. The canonical use is turning fine-grained raster climate data into region-by-time-period panel data (weighted by area and optionally by a secondary variable such as population or cropland).

The package is imported as `af` (`import aggfly as af`). The public API is re-exported from `aggfly/__init__.py`.

## Commands

Dependency management is via Poetry (`pyproject.toml`, `poetry.lock`):

```bash
poetry install              # install deps into a virtualenv
poetry run pytest           # run the full test suite
poetry run pytest aggfly/tests/test_aggregate.py::test_weights   # run a single test
```

Tests live in `aggfly/tests/test_aggregate.py`. The fixtures (`dataset_360`, `georegion`, `secondary_weights`, `weights`) build small synthetic in-memory objects, so tests need no external data files. Assertions compare against hardcoded numeric arrays via `np.allclose`, so changes to the aggregation math will require updating the expected values in those tests.

### Python version constraint (important)

The version pin is deliberately narrow: `python = ">=3.11.6, <3.12.3"`. Python 3.12.3+ breaks `dask-geopandas` (dask dataframe internals changed). Do not loosen this or bump `dask`/`dask-geopandas` independently — they must move together once `dask-geopandas` releases a compatible version. See the comments at the top of `pyproject.toml`.

## Architecture

The workflow is a three-stage pipeline. Each stage produces an object consumed by the next.

**1. Load inputs** (`aggfly/regions/`, `aggfly/dataset/`)
- `af.georegions_from_path(shp_path, regionid=...)` → `GeoRegions`, wrapping a GeoDataFrame of target regions. `regionid` names the column holding the region identifier.
- `af.dataset_from_path(path, var=..., georegions=..., preprocess=...)` → `Dataset`, wrapping an `xarray.DataArray` (backed by dask). `preprocess` is a lambda applied to raw values (e.g. Kelvin→Celsius). `Dataset` normalizes dimension names and handles the 0–360 vs -180–180 longitude convention (`lon_is_360`).
- `Grid` (`aggfly/dataset/grid.py`) describes the raster's lon/lat mesh, cell ids, resolution, and cell areas — derived from a `Dataset` and used for weight computation.

**2. Compute weights** (`aggfly/weights/`)
- `af.weights_from_objects(dataset, georegions, secondary_weights=None, project_dir=...)` → `GridWeights`. Call `.calculate_weights()` to populate `.weights` (a dataframe keyed by `cell_id`/`region_id` with `area_weight`, `raster_weight`, and combined `weight` columns).
- **Area weights** account for the share of each grid cell covered by a region and for cell-area distortion by latitude (cosine correction).
- **Secondary weights** (optional) reweight cells by another raster: `SecondaryWeights` (generic), `PopWeights` (`pop_weights_from_path`), `CropWeights` (`crop_weights_from_path`). The final `weight` is the product of area and secondary weights.
- `GridWeights` requires the grid to be in -180–180 convention (`assert not self.grid.lon_is_360`).

**3. Aggregate** (`aggfly/aggregate/`) — the core is `af.aggregate_dataset(dataset, weights, **named_specs)`, which runs temporal aggregation then spatial aggregation and returns a pandas DataFrame merged back onto the region ids.

Temporal-then-spatial split:
- `aggregate_time` (in `aggregate/aggregate.py`) turns a single `Dataset` into a dict of `Dataset`s, one per named output variable, by applying a sequence of steps.
- `aggregate_space` (using `SpatialAggregator` in `aggregate/spatial.py`) computes the weighted average over each region for every dataset in that dict, via dask dataframes, producing the output panel.

**The aggregation spec DSL.** Each keyword arg to `aggregate_dataset`/`aggregate_time` names an output variable and maps to a list of `(step_type, params)` tuples applied in order:
- `('aggregate', {'calc': ..., 'groupby': ..., 'ddargs': ...})` builds a `TemporalAggregator` (`aggregate/temporal.py`). `calc` is one of `mean`, `min`, `max`, `sum`, `dd` (degree days), `bins`, `sine_dd`. `groupby` is a time frequency (`date`, `month`, `year`, ...). `ddargs` gives thresholds: `[low, high, inc]` for `dd`, or a list of such triples for `bins`.
- `('transform', {'transform': 'power', 'exp': np.arange(1,3)})` raises the variable to given powers, producing one output per exponent (keys suffixed `_1`, `_2`, ...). Also supports `inter` (interact with another dataset) and `spline`.

A single `dd`/`bins` step with multiple `ddargs` (`multi_dd`) fans one variable out into several outputs keyed by threshold (see `multi_dd_to_dict`). You cannot combine multi-`ddargs` with multiple upstream datasets (e.g. multiple polynomials × multiple bins).

**Dask.** Aggregation is dask-backed and can run on a `LocalCluster`. `aggregate_dataset` accepts cluster args (`n_workers`, `threads_per_worker`, etc.); helpers `start_dask_client`/`shutdown_dask_client`/`is_distributed` live in `aggregate/aggregate_utils.py`. A `ProgressBar` is registered globally on import.

**Temporal engine (`engine=`).** `aggregate_dataset`/`aggregate_time` take `engine="dask"` (default) or `engine="numba"`. The numba engine (`aggregate/nb_kernels.py`) replaces the per-group `.resample(freq).reduce(func)` call with a single `nogil`, `parallel` compiled pass per spatial chunk (via `map_blocks`), covering `mean/nanmean/sum/min/max/dd/bins/sine_dd`. It is bit-equivalent to the dask path. It is dramatically faster **only on small/native spatial chunks** (the GIL-bound, low-memory regime): ~12× on native ERA5 chunks at lower memory. On large rechunked blocks (e.g. 250×250) it is *slower* — dask's vectorized numpy is already compute-bound there and per-block numba threads oversubscribe the dask pool. Rule of thumb: `engine="numba"` + native chunks; don't pair it with a large `.chunk()`. (The legacy `sine_dd` NaN masking in `temporal.py` is buggy — wrong-axis `frame[:, :, 0]`; the numba kernel uses the correct any-NaN-in-window rule.)

**Caching** (`aggfly/cache/project_cache.py`): `ProjectCache` hashes the parameters of a module (weights, etc.) into a SHA and caches intermediate results under `{project_dir}/tmp/{module}/{sha}/`. This is why many entry points take a `project_dir`.

## Notes

- `notebooks/` contains example workflows (`run_t2m_us-counties_example.ipynb`, `giovanni_example/`) and scratch/experimental scripts — these are not part of the package and are not tested.
- `aggfly/aggregate/z_old/` and `aggfly/scratch.py` are dead/legacy code; don't rely on them.
- The README documents the intended user-facing workflow with fuller examples; several sections there are marked TODO/incomplete.
