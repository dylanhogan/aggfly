# aggfly command-line interface

`aggfly` ships a command-line interface for running a full spatial/temporal
aggregation — regions → dataset → weights → aggregate → panel — from a YAML
config file, without writing a Python script.

Installing the package puts an `aggfly` executable on your `PATH`:

```bash
uv sync
uv run aggfly --help
```

## The four commands

| Command | Purpose |
|---|---|
| `aggfly info PATH` | Inspect a dataset (dims, calendar, longitude convention, time span) to help you write a config. |
| `aggfly validate CONFIG` | Statically check a config — no data is read. Prints a normalized plan. |
| `aggfly weights CONFIG` | Build and cache the spatial weights only. |
| `aggfly run CONFIG` | Run the full pipeline and write the output panel. |

The natural authoring loop is **`info` → `validate` → `run`**:

```bash
aggfly info era5_2000.zarr --var t2m     # discover coord names, units, calendar
aggfly validate config.yaml              # catch config mistakes with no data read
aggfly run config.yaml                   # aggregate → panel
```

## `aggfly info`

```bash
aggfly info PATH [--var NAME] [--storage-options JSON]
```

Opens the dataset lazily (reads only coordinate arrays) and reports the values
you need to fill into a config: coordinate names (`xycoords`, `timecoord`), the
longitude convention (`lon_is_360`), the calendar (flagging non-standard
`cftime` calendars like `360_day`), the time span, and chunking. `PATH` may be a
local file/Zarr or a remote URL; pass credentials for object stores via
`--storage-options`, e.g. `--storage-options '{"token": "anon"}'` for public GCS.

## `aggfly validate`

```bash
aggfly validate CONFIG [--strict]
```

Verifies the config's shape, the aggregation spec (calcs, groupby, ddargs),
`preprocess`, output format, and year expansion — **all in one pass**, so every
problem is reported at once. It then prints a normalized plan. Local input paths
are checked for existence and reported as **warnings**; `--strict` promotes those
to errors (nonzero exit). Remote URLs are never fetched.

## `aggfly weights`

```bash
aggfly weights CONFIG [--project-dir DIR] [-v/--verbose]
```

Weights depend only on the grid and the regions — not the time series — so this
precomputes them once. With `weights.project_dir` set, the result is cached; a
later `run` with the same parameters reuses it automatically. Handy when you will
aggregate many years or many variables against the same grid.

## `aggfly run`

```bash
aggfly run CONFIG [-o/--output PATH] [--engine auto|dask|numba]
                  [--years START:END] [--project-dir DIR]
                  [--backend threads|processes|none] [--n-workers N]
                  [-v/--verbose]
```

Loads regions, builds (and caches) weights, aggregates every resolved dataset
path over time and space, and writes the region-by-period panel. Flags override
the corresponding config fields. On a user error the CLI prints a one-line
message and exits nonzero; pass `-v` for a full traceback.

## Config schema

A config is YAML with four stages plus job control. Minimal example:

```yaml
regions:
  path: usa_counties.shp
  regionid: fips
dataset:
  path: era5_t2m_2000.zarr
  var: t2m
  preprocess: kelvin_to_celsius
  lon_is_360: true
aggregate:
  variables:
    tavg:
      - [aggregate, {calc: mean, groupby: date}]
      - [aggregate, {calc: mean, groupby: month}]
output:
  path: out/panel.parquet
```

Fully annotated:

```yaml
# --- 1. target regions -------------------------------------------------
regions:
  path: usa_counties.shp        # shapefile of target regions
  regionid: fips                # column holding the region id
  region_list: null             # optional list to subset regions

# --- 2. climate dataset ------------------------------------------------
dataset:
  path: era5_t2m_{year}.zarr    # {year} is looped (see `years`); may be a glob or list
  var: t2m
  preprocess: kelvin_to_celsius # builtin name OR a safe expression "x - 273.15"
  # preprocess_from: prep.py:clean_era5   # (alternative: a .py function; runs your code)
  lon_is_360: true              # true if longitudes run 0..360 (ERA5); false for -180..180
  timecoord: time
  xycoords: [longitude, latitude]
  time_sel: null                # optional time slice
  chunks: {time: 24, latitude: -1, longitude: -1}
  clip_to_regions: true         # clip raster to region extent (read optimization; see note)
  # engine: zarr                # force the reader backend (usually auto-detected)
  # storage_options: {token: anon}   # credentials for gs:// / s3:// paths

# --- 3. weights --------------------------------------------------------
weights:
  project_dir: ./proj           # enables the weight cache
  secondary:                    # optional; omit for area-only weights
    type: pop                   # pop | crop | generic
    path: landscan-global-2016.tif
    # crop: corn                # crop-only knobs
    # feed: rainfed

# --- 4. aggregation ----------------------------------------------------
aggregate:
  engine: auto                  # auto | dask | numba
  variables:                    # each name → a pipeline of steps
    tavg:
      - [aggregate, {calc: mean, groupby: date}]
      - [transform, {transform: power, exp: [1, 2]}]
      - [aggregate, {calc: sum,  groupby: year}]
    gdd:
      - [aggregate, {calc: dd,  groupby: date, ddargs: [10, 30, 0]}]
      - [aggregate, {calc: sum, groupby: year}]

# --- job control (also overridable by flags) ---------------------------
years: 1980:1990                # "start:end" inclusive, or a list; drives {year}
execution:
  backend: threads              # threads | processes | none (bring-your-own client)
  n_workers: 16
  threads_per_worker: 1
output:
  path: out/panel.parquet
  format: parquet               # parquet | feather | csv (inferred from extension if omitted)
```

### The aggregation spec

Each entry under `aggregate.variables` names an output variable and maps to a
list of steps applied in order. A step is `[step_type, params]`:

- **`aggregate`** — `calc` is one of `mean`, `min`, `max`, `sum`, `dd` (degree
  days), `bins`, `sine_dd`; `groupby` is `date`, `month`, `year`, or `week`.
  `dd`/`bins`/`sine_dd` require `ddargs` (`[low, high, inc]`, or a list of such
  triples for `bins`).
- **`transform`** — `{transform: power, exp: [1, 2]}` raises the variable to each
  power (outputs suffixed `_1`, `_2`, …). Also supports `inter` and `spline`.

A single `bins`/multi-`dd` step (a list of triples) fans one variable into
several outputs. You cannot combine that with a multi-output `transform` (e.g.
multiple exponents) — `validate` catches this.

> **Calendars.** Non-standard CF calendars (CMIP6 `noleap`/`360_day`, …) are
> supported and preserved. `groupby: week` is **not** available on those
> calendars — use `date`/`month`/`year`. See the README's "Calendars" section.

### `preprocess`

Three ways to transform raw values before aggregation:

1. **Named builtin**: `kelvin_to_celsius`, `celsius_to_kelvin`, `identity`,
   `pa_to_kpa`, `m_to_mm`.
2. **Safe arithmetic expression** in `x` — `"x - 273.15"`, `"(x - 32) * 5 / 9"`,
   `"x * 0.1"`. Only arithmetic on `x` and numbers is allowed (no function calls,
   attribute access, or other names).
3. **File escape hatch** — `preprocess_from: path/to/file.py:func`, pointing at a
   function `func(x)`. This imports and runs your code; use it only for files you
   trust. Mutually exclusive with `preprocess`.

### `clip_to_regions`

When true (default), the raster is clipped to the regions' bounding extent as it
loads — a read-reduction that **never changes results** (weights select the
relevant cells regardless). Disable it for regions that wrap the antimeridian in
the 0–360 convention, where the extent clip is ill-defined.

### Reading from object storage

`dataset.path` may point at a cloud store. `storage_options` is forwarded
verbatim to xarray's fsspec backend, so anything that backend accepts works —
`{token: anon}` for public buckets, or credentials for private ones. Install the
matching extra (`pip install "aggfly[gcs]"` or `"aggfly[s3]"`).

```yaml
dataset:
  path: gs://cmip6/CMIP6/CMIP/NOAA-GFDL/GFDL-CM4/historical/r1i1p1f1/day/tas/gr1/v20180701/
  var: tas
  xycoords: [lon, lat]
  lon_is_360: true
  engine: zarr                  # needed when the path has no .zarr in it
  storage_options: {token: anon}
```

`aggfly validate` prints which `storage_options` keys are set but **never their
values**, so a config carrying a token can be validated without leaking it into
logs. Note that the config file itself holds the credential in plain text —
prefer `{token: anon}` for public data, or a backend that reads ambient
credentials (e.g. `GOOGLE_APPLICATION_CREDENTIALS`), over pasting secrets here.

`dataset.engine` is a separate knob from `aggregate.engine`: the former picks
xarray's *reader* backend, the latter the *temporal* kernel.

## Execution & scaling

Two independent knobs, mirroring the library:

- **`aggregate.engine`** picks the temporal kernel (`auto`/`dask`/`numba`).
- **`execution.backend`** picks how tasks execute:
  - `threads` (default) — Dask's threaded scheduler; correct for a laptop or a
    single disk, zero setup.
  - `processes` — a local process cluster (`--n-workers`, `threads_per_worker`);
    parallelizes GIL-bound warm reads on a fat node.
  - `none` — start no client; use whatever scheduler is already active.

Results are identical across backends — only speed changes. The CLI always
builds weights **before** starting the execution client (weight computation is
incompatible with an active distributed client), then shuts the client down when
the run finishes.

For HPC (multi-node), start your own `dask-jobqueue` cluster in a wrapper and set
`backend: none`; aggfly will use the active client. See the README's
"Execution & scaling" section for hardware-specific recipes.

## Examples

Runnable configs live in [`examples/`](../examples):

- [`era5_counties_area.yaml`](../examples/era5_counties_area.yaml) — area-weighted
  county panel over multiple years.
- [`era5_counties_pop.yaml`](../examples/era5_counties_pop.yaml) — the same, but
  population-weighted via a secondary raster.
