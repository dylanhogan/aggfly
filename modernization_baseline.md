# Modernization baseline & inventory (plan.md item 5, step 1)

Captured 2026-07-07 on branch `auto-select-engine`. This is the reference state the
dependency/Python modernization is measured against.

## Baseline test suite (the correctness net)

```
poetry venv (python 3.11): python -m pytest -q
7 passed in 8.82s
```

Tests: `aggfly/tests/test_aggregate.py` — synthetic in-memory fixtures, `np.allclose`
numeric assertions. Any stack bump must keep these 7 green (and numba↔dask
bit-equivalence for the temporal engine).

## Declared constraints (`pyproject.toml`) vs resolved (`poetry.lock`)

| package        | declared (pyproject) | resolved (lock) | notes |
|----------------|----------------------|-----------------|-------|
| python         | `>=3.11.6, <3.12.3`  | 3.11 venv       | ceiling was the dask-geopandas blocker (now stale) |
| dask           | `^2023.11.0`         | 2023.12.1       | pre query-planning; target `>=2025.1.0` |
| distributed    | (via dask)           | 2023.12.1       | moves with dask |
| dask-geopandas | `^0.3.1`             | 0.3.1           | target 0.5.0 (Jun 2025); needs dask>=2025.1.0, py>=3.10 |
| xarray         | `^2024.5.0`          | 2024.5.0        | |
| zarr           | `^2.18.2`            | 2.18.2          | target v3 (breaking) — but see risk note below |
| numpy          | `^1.26.4`            | 1.26.4          | target 2.x; numba>=0.60 required for numpy 2 |
| numba          | `^0.59.1`            | 0.59.1          | hot path (`nb_kernels.py`); target >=0.60 |
| pandas         | `^2.2.2`             | 2.2.2           | |
| geopandas      | `^0.14.4`            | 0.14.4          | target 1.x (dask-geopandas 0.5.0 tracks gpd 1.1) |
| rioxarray      | `^0.15.5`            | 0.15.5          | |
| pyproj         | (via geo stack)      | 3.6.1           | |
| rasterio       | `^1.3.10`            | 1.3.10          | |
| shapely        | `^2.0.4`             | 2.0.4           | |
| netcdf4        | `^1.6.5`             | 1.6.5           | |
| dill           | `^0.3.8`             | 0.3.8           | ProjectCache pickle backend |
| fsspec         | (via stack)          | 2024.5.0        | |
| numcodecs      | (via zarr)           | 0.12.1          | |

The `pyproject.toml` header comments (lines 10–14) cite dask-geopandas issues
[#284](https://github.com/geopandas/dask-geopandas/issues/284) /
[#289](https://github.com/geopandas/dask-geopandas/issues/289) as the reason for the
Python ceiling. Those are resolved upstream (dask-geopandas 0.4.3+ requires
`dask>=2025.1.0`), so the ceiling is stale.

## API touchpoint inventory (refines the plan's risk ranking)

### 🔴 Primary surface — modern dask dataframe + dask-geopandas
This is where the real migration work is (query-planning / dask-expr changed the
dataframe + dask-geopandas API between 2023.12 and 2025.x).
- `aggfly/aggregate/spatial.py:155,159,171` — `dask.dataframe.from_pandas(merged_df, npartitions=50)`
  then a groupby weighted-average `.compute()`. groupby-apply semantics under dask-expr
  are the thing most likely to shift. (Also the known `npartitions=50` hardcode ignoring the param.)
- `aggfly/weights/grid_weights.py:182,210,214,283,286,389` — `dask_geopandas.from_geopandas`,
  `.sjoin(predicate="within")`, `.geometry.intersection(...)`, `dask.dataframe.from_pandas`.
- `aggfly/dataset/grid.py:257,270,276` — `dask_geopandas.from_geopandas`, `.sjoin(...).compute()`.
- `aggfly/regions/georegions.py:118,123` — `dask_geopandas.from_geopandas(...).buffer(...).compute()`.
- `dask.array.map_blocks` usage (`nb_kernels.py:272,281`, `dataset.py:458,477,503`,
  `grid.py`, `utils.py`) — array API, low risk, but re-verify after bump.

### 🟡 numpy 1→2 — trivial
- No removed aliases found (`np.float_`, `np.NaN`, `np.int0`, `np.bool8`, `np.product`,
  `np.round_`, `np.trapz`, ... all absent).
- Only `np.in1d` (deprecated in numpy 2, use `np.isin`) at 3 sites:
  `aggfly/regions/georegions.py:187,220`, `aggfly/regions/shp_utils.py:32`.
- numba must go to >=0.60 to import under numpy 2; re-benchmark `nb_kernels.py` after.

### 🟢 zarr v2→v3 — NOT a risk for existing code
- The package never imports `zarr` directly and has no `to_zarr`/consolidated-metadata usage.
- `ProjectCache` (`aggfly/cache/project_cache.py`) persists via `to_netcdf` (`.nc`) + dill
  pickle — **no zarr store**. Only `xr.open_zarr` reads exist
  (`grid_utils.py:171`, `crop_weights.py:231`), which xarray abstracts across zarr v2/v3.
- zarr v3 only becomes relevant for the *new* to_zarr conversion helper (plan item 4).

## Step 2 audit results — modern stack resolved & suite run (2026-07-07)

**Resolution confirmed via pip on python 3.12** (the publish-path proxy, not conda — a
conda solve would pin differently and wouldn't prove the pip/PyPI resolution). A throwaway
`python3.12 -m venv` cleanly resolved and imported the whole stack; the geo deps all ship
manylinux wheels now, so no conda needed:

| dask | distributed | dask-geopandas | geopandas | xarray | zarr | numpy | numba | pandas | scipy |
|------|-------------|----------------|-----------|--------|------|-------|-------|--------|-------|
| 2026.7.0 | 2026.7.0 | 0.5.0 | 1.1.4 | 2026.4.0 | 3.2.1 | 2.4.6 | 0.66.0 | 3.0.3 | 1.18.0 |

`import aggfly` works under this stack. Note the resolver also pulls **pandas 3.0** (a large
jump beyond what the plan assumed) and **geopandas 1.1** — both matter below.

**Suite under the modern stack: 4 passed / 5 errors.**
- ✅ The 4 passing include BOTH new `test_spatial_matmul_*` tests and the engine-resolver
  tests → the sparse-matmul spatial rewrite and the numba engine selection work unchanged
  under numpy 2 / pandas 3 / dask 2026. The whole non-weights surface is clean.
- ❌ All 5 errors are the **`weights` fixture setup** — the dask-geopandas surface, exactly
  as the plan predicted. Every failing test just depends on that fixture.

**Root cause (walked in the scratch env): `GridWeights.intersect_border_cells`
(`grid_weights.py:294–300`).** The dask-geopandas element-wise `.intersection` across two
*differently-partitioned* GeoSeries breaks in two independent ways — it is NOT a flag flip:
1. **geopandas 1.0 flipped binary-op default to `align=True`.** With duplicate `index_right`
   labels (many border cells per region), index alignment turns the row-wise intersection
   into a cartesian product → `ValueError: Length of values (16) does not match ... (4)`.
2. Adding `align=False` then hits **dask-geopandas 0.5.0 partition-count semantics** (`reg`
   has npartitions=1, `dgb` has npartitions=chunks) → `ValueError: Lengths of inputs do not
   match. Left: 1, Right: 4`.
   → the whole `from_geopandas(...).geometry.intersection(...)` pattern needs a rewrite.

**Scoping for step 4 — split the dask-geopandas surface by data size:**
- **Small data → drop dask-geopandas, use plain geopandas 1.0** (`gs.intersection(other,
  align=False)` on in-memory GeoSeries; positional `.to_numpy()` when assigning computed
  series back). These never needed parallelism:
  - `grid_weights.py` `intersect_border_cells` (border cells only) — the actual blocker.
  - `grid_weights.py` `simplify_poly_array` (region polygons).
  - `georegions.py` buffer (`from_geopandas(...).buffer(...).compute()`, regions).
- **Large data → keep dask-geopandas, update to 0.5.0 API + verify:**
  - `grid.py` `Grid.mask` `fc.sjoin(poly_array, predicate="within")` — `fc` is every grid-cell
    centroid (~1e6 for a full ERA5 grid); genuine parallel sjoin. `.sjoin` API is stable, so
    likely low-effort, but couldn't be exercised (blocked behind the weights fixture).
  This split may let aggfly **drop dask-geopandas entirely** if the one large sjoin is the only
  real user and can be met another way — parallel to how `spatial.py` shed `dask.dataframe`.

**Other modern-stack touch-ups seen:**
- `np.in1d` → `np.isin` (deprecated in numpy 2): `georegions.py:187,220`, `shp_utils.py:32`.
- geopandas 1.0: `GeoSeries.unary_union` → `union_all()` (DeprecationWarning in the test
  fixture at `test_aggregate.py:82`; grep `shp_utils.py` too).
- pandas 3.0 is a bigger bump than planned (copy-on-write default, stricter index alignment —
  the reindex/length errors above are partly this); the weights rewrite must use
  pandas-3-safe idioms (explicit `.to_numpy()` on positional assignments).

Scratch env for step 4 lives at `scratchpad/modernenv` (py3.12, pip). The project poetry venv
and lock were left untouched; old env still 9/9 green.

## Environment notes
- System/shell python is 3.12.3 (unsupported by the pin), so `poetry install` can't
  auto-select the interpreter; all commands run the py3.11 poetry venv directly:
  `~/.cache/pypoetry/virtualenvs/aggfly-Mgh0elN4-py3.11/bin/python`.
- The venv currently also has `kerchunk==0.2.7`, `h5py`, `ujson` (installed for item-4
  profiling); these are NOT in pyproject and are inert for the test suite.
