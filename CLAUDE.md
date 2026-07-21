# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`aggfly` is a Python package for spatial and temporal aggregation of gridded climate data (e.g. ERA5) onto administrative regions defined by a shapefile. The canonical use is turning fine-grained raster climate data into region-by-time-period panel data (weighted by area and optionally by a secondary variable such as population or cropland).

The package is imported as `af` (`import aggfly as af`). The public API is re-exported from `aggfly/__init__.py`.

## Commands

Dependency management is via [uv](https://docs.astral.sh/uv/) (`pyproject.toml` with PEP 621 `[project]` metadata, `uv.lock`):

```bash
uv sync                     # create .venv and install deps (incl. the dev group)
uv run pytest               # run the full test suite
uv run pytest aggfly/tests/test_aggregate.py::test_weights   # run a single test
uv run aggfly --help        # invoke the CLI entry point
```

uv manages its own Python interpreter (pinned in `.python-version`), so runs are isolated from any system/conda Python. Migrated from Poetry 2026-07; the build backend is hatchling and `pytest` lives in the `dev` dependency group.

Tests live in `aggfly/tests/test_aggregate.py`. The fixtures (`dataset_360`, `georegion`, `secondary_weights`, `weights`) build small synthetic in-memory objects, so tests need no external data files. Assertions compare against hardcoded numeric arrays via `np.allclose`, so changes to the aggregation math will require updating the expected values in those tests.

### Releasing to PyPI

Last release: **0.2.0**, 2026-07-21 (previous was 0.1.5, 2024-10-24). A PyPI version can never be re-uploaded, even after deletion — a mistake burns the number. Work through the checks before publishing.

**1. Decide the version.** Check what is actually on PyPI first; the repo has drifted behind before (`pyproject.toml` said 0.1.4 while PyPI had 0.1.5).

```bash
curl -s https://pypi.org/pypi/aggfly/json | python3 -c "import sys,json;print(json.load(sys.stdin)['info']['version'])"
```

Bump `version` in `pyproject.toml`. Anything that changes numerical output (weighting, aggregation math) warrants a minor bump and a prominent note, not a patch.

**2. Sync the lockfile.** `uv.lock` records the project's own version; `uv build`/`uv sync` rewrites it after a bump. Commit it, or the tag points at a tree that goes dirty immediately.

**3. Build and inspect the sdist.** Hatchling's default sdist sweeps the whole project root — this once produced a 5.8 MB archive containing the presentation PDF, scratch notebooks, `internal/`, and `.claude/settings.local.json`. `[tool.hatch.build.targets.sdist]` in `pyproject.toml` scopes it; a healthy sdist is ~80 KB.

```bash
uv build --out-dir /tmp/rel
tar tzf /tmp/rel/aggfly-*.tar.gz | sed 's|^aggfly-[^/]*/||' | cut -d/ -f1 | sort -u
```

**4. Fresh-install test — the one that matters.** Every dev environment has pytest, matplotlib and the rest, so import bugs stay invisible until a user hits them. Install the built wheel into an isolated env and import it. This is how the 0.2.0 `ModuleNotFoundError: No module named 'pytest'` was caught (`aggfly/__init__.py` re-exported test fixtures, which import pytest at module load, after pytest moved to the dev group).

```bash
uv run --isolated --no-project --with /tmp/rel/aggfly-*.whl python -c "import sys, aggfly; print('matplotlib' in sys.modules, 'PIL' in sys.modules)"
uv run --isolated --no-project --with /tmp/rel/aggfly-*.whl aggfly --version
```

Both module checks must print `False` — `import aggfly` must not pull in matplotlib or PIL (see issue #5). Nothing outside the plotting methods may import an optional heavy dependency at module load.

**5. Tag and write release notes.** Annotated tag on the release commit, then a GitHub release. Lead with anything that changes user-visible numbers.

```bash
git tag -a v0.2.0 -m "aggfly 0.2.0"
git push origin main && git push origin v0.2.0
gh release create v0.2.0 --title v0.2.0 --verify-tag --notes-file notes.md
```

Use `--notes-file` with a file written by a plain heredoc. Do not pipe content into a `python3 - <<'PY'` heredoc — stdin is already the script, the pipe silently wins, and `gh release edit` will happily overwrite the release body with an empty file.

**6. Publish.** The token lives in `~/.config/aggfly-pypi-token` (mode 600), project-scoped. Expand it inline so it never lands in a transcript:

```bash
UV_PUBLISH_TOKEN=$(cat ~/.config/aggfly-pypi-token) uv publish /tmp/rel/aggfly-*
```

**7. Verify from PyPI, not from the build.** The JSON API is CDN-cached and can lag by minutes; `https://pypi.org/simple/aggfly/` is authoritative.

```bash
uv run --isolated --no-project --with 'aggfly==X.Y.Z' python -c "import aggfly; print('ok')"
uv run --isolated --no-project --with 'aggfly==X.Y.Z' aggfly --version
```

### Dependency stack (modernized 2026-07)

`python = ">=3.11,<3.14"` on a current stack: numpy 2, pandas 3, zarr 3, geopandas 1, numba ≥0.60, and CalVer dask/xarray (2025+). The old narrow `<3.12.3` pin existed only because `dask-geopandas` lagged dask's dataframe internals and broke at 3.12.3. **`dask-geopandas` has been removed from aggfly entirely** — its only uses were two point-in-polygon sjoins now done with plain `geopandas.sjoin` (faster and lighter; see `benchmarks/bench_sjoin.py`) — so that coupling is gone. `dask` is declared with the `distributed` extra (previously distributed came in transitively via dask-geopandas). CalVer packages use `">="` constraints, not caret (a caret would wrongly cap at the release year). The `np.allclose` test fixtures are the correctness net across the bump.

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

**Temporal engine (`engine=`).** `aggregate_dataset`/`aggregate_time` take `engine="auto"` (default), `"dask"`, or `"numba"`. The numba engine (`aggregate/nb_kernels.py`) replaces the per-group `.resample(freq).reduce(func)` call with a single `nogil`, `parallel` compiled pass per spatial chunk (via `map_blocks`), covering `mean/nanmean/sum/min/max/dd/bins/sine_dd`. It is bit-equivalent to the dask path. It is dramatically faster **only on small/native spatial chunks** (the GIL-bound, low-memory regime): ~12× on native ERA5 chunks at lower memory. On large rechunked blocks (e.g. 250×250) it is *slower* — dask's vectorized numpy is already compute-bound there and per-block numba threads oversubscribe the dask pool. `"auto"` resolves this per step via `resolve_engine()`: numba when the largest spatial block is ≤ `NUMBA_MAX_CELLS_PER_BLOCK` (default `150*150`) cells, dask otherwise (and always dask for calcs outside `NUMBA_CALCS`). Force a backend with `"dask"`/`"numba"`; pair explicit `"numba"` with native chunks, not a large `.chunk()`.

**Caching** (`aggfly/cache/project_cache.py`): `ProjectCache` hashes the parameters of a module (weights, etc.) into a SHA and caches intermediate results under `{project_dir}/tmp/{module}/{sha}/`. This is why many entry points take a `project_dir`.

## Notes

- **Documentation layout** (reorganized 2026-07): `README.md` is a slim landing page (overview, install, quickstart, links). All prose docs live under `docs/` — `index.md` (TOC), `installation.md`, `concepts.md`, `guide/{quickstart,weights,aggregation,execution,calendars}.md`, `cli.md`, `api.md`. Update the relevant `docs/` page, not the README, when behavior changes.
- `internal/` holds historical planning docs (`cli-plan.md`, `backend-plan.md`, `modernization-baseline.md`) — design records, not user docs and not necessarily current.
- `examples/` holds runnable CLI configs and, under `examples/notebooks/`, `quickstart_public_data.ipynb` — a standalone notebook that downloads all its own public data (Natural Earth regions, CMIP6 climate from GCS, WorldPop population) and runs the full pipeline plus an equivalent CLI run. Not part of the package and not tested in CI, but it was verified to execute end-to-end. Everything it writes lands in `examples/notebooks/aggfly_example_data/`, which is gitignored except for the generated `config.yaml`, kept as a worked CLI example.
- `benchmarks/bench_{sjoin,weights}.py` read a US-counties shapefile from `benchmarks/data/`.
- `notebooks/` is scratch/experimental only.
- `aggfly/aggregate/z_old/` and `aggfly/scratch.py` are dead/legacy code; don't rely on them.
