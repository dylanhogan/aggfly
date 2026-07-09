# aggfly CLI — implementation plan

A command-line interface for aggfly so a full spatial/temporal aggregation run
(regions → dataset → weights → aggregate → panel) can be driven from a
config file, without writing a Python script.

## Design decisions (locked)

- **Config-file first, flags for job control.** A YAML config expresses the whole
  pipeline; flags override job-control knobs (`--years`, `--engine`, `--backend`,
  `-o/--output`, `--project-dir`). The aggregation spec is a nested list-of-tuples
  DSL that does not fit flags, so the config file is the primary surface.
- **`preprocess`: named builtins + safe expression, plus a `.py` escape hatch.**
  Covers the common unit conversions with no code exec; the escape hatch handles
  anything exotic (local research tool, trusted files).
- **v1 scope: `run`, `weights`, `info`, `validate`.**
- **Deps:** `click` (CLI framework) + `pyyaml` (config). Both pure-Python,
  universal, pip-friendly — consistent with the "validate deps with pip/venv"
  policy. No pydantic in v1 (hand-rolled validation with clear messages).

## Why the DSL maps cleanly to YAML

`aggregate_time` unpacks each step with `for key2, value2 in value:`, so a YAML
list `[aggregate, {calc: dd, groupby: date, ddargs: [10,30,0]}]` unpacks exactly
like the Python tuple `('aggregate', {...})`. Numpy expressions like
`np.arange(1,3)` become explicit lists (`exp: [1, 2]`). **No translation layer is
needed** between parsed YAML and `aggregate_dataset` — the config's `variables`
block is passed through as the `aggregator_dict`.

## Architecture

Keep the library as the single source of truth; the CLI is a **thin orchestrator**
that (a) parses/validates config, (b) resolves `preprocess`, (c) loops years, and
(d) calls the existing `af.*` functions. The valuable, fast-to-test core is the
pure `config → validated params` layer — no dask, no I/O.

New package `aggfly/cli/`:

| Module | Responsibility |
|---|---|
| `main.py` | `click` group + the four subcommands; flag parsing; friendly errors |
| `config.py` | Load YAML, schema-validate, normalize into a typed `RunConfig` dataclass, expand `{year}` templates |
| `preprocess.py` | Named-builtin registry + safe arithmetic-expression evaluator (AST allowlist) + `path.py:func` loader |
| `pipeline.py` | Orchestrate the 4 stages + multi-year loop + output writing (used by `run`/`weights`) |
| `info.py` | Lazy-open a dataset and report dims/coords/vars/calendar/lon-convention/time-span/chunks |

Entry point in `pyproject.toml`:

```toml
[tool.poetry.scripts]
aggfly = "aggfly.cli.main:cli"
```

## Config schema

```yaml
# --- 1. target regions -------------------------------------------------
regions:
  path: usa_counties.shp
  regionid: fips                 # column holding the region id
  region_list: null              # optional subset

# --- 2. climate dataset ------------------------------------------------
dataset:
  path: era5_{year}.zarr         # {year} → looped; may also be a glob or list
  var: t2m
  preprocess: kelvin_to_celsius  # builtin name | safe expr "x - 273.15"
  # preprocess_from: prep.py:clean_era5   # (alternative escape hatch)
  lon_is_360: true
  timecoord: time
  xycoords: [longitude, latitude]
  time_sel: null                 # optional time slice
  chunks: {time: 24, latitude: -1, longitude: -1}
  clip_to_regions: true          # clip raster to region extent (read opt; off for antimeridian-wrapping regions)

# --- 3. weights --------------------------------------------------------
weights:
  project_dir: ./proj            # enables the weight cache (.feather)
  secondary:                     # optional; omit for area-only weights
    type: pop                    # pop | crop | generic
    path: landscan-2016.tif
    # crop: corn                 # crop-only knobs
    # feed: rainfed

# --- 4. aggregation ----------------------------------------------------
aggregate:
  engine: auto                   # auto | dask | numba
  variables:                     # <- the DSL, passed through verbatim
    tavg:
      - [aggregate, {calc: mean, groupby: date}]
      - [transform, {transform: power, exp: [1, 2]}]
      - [aggregate, {calc: sum,  groupby: year}]
    gdd:
      - [aggregate, {calc: dd,  groupby: date, ddargs: [10, 30, 0]}]
      - [aggregate, {calc: sum, groupby: year}]

# --- job control (also overridable by flags) ---------------------------
years: 1980:1990                 # "start:end" inclusive, or a list; drives {year}
execution:
  backend: threads               # threads | processes | none(bring-your-own)
  n_workers: 16
  threads_per_worker: 1
output:
  path: out/panel.parquet
  format: parquet                # parquet | feather | csv (inferred from ext if omitted)
```

## Subcommands

### `aggfly run <config.yaml> [flags]`
The workhorse. Orchestration (note the **ordering constraint**):

1. Load `GeoRegions` (`georegions_from_path`).
2. Load a **sample layer** to define the grid (first year / first file).
3. Build weights (`weights_from_objects` + `calculate_weights`) — **before** any
   execution client, because `calculate_weights()` shuts down an active Dask
   client. Cached under `project_dir`, so this is a no-op on reruns.
4. **Now** start the execution client per `execution.backend`
   (`processes` → `start_dask_client(n_workers, threads_per_worker)`;
   `threads`/`none` → nothing).
5. For each year: format the `{year}` path, `dataset_from_path(...)`,
   `aggregate_dataset(dataset, weights, **variables, engine=...)`; collect frames.
6. `pd.concat`, write to `output.path` in the requested format.
7. `finally:` shut the client down.

Flags override config: `--years 1980:1990`, `--engine`, `--backend`,
`--n-workers`, `-o/--output`, `--project-dir`, `-v/--verbose`, `--dry-run`
(≡ `validate` then stop).

### `aggfly weights <config.yaml>`
Runs stages 1–3 only: build + cache weights, then exit. Lets users precompute the
(dataset-independent, reusable) weights once; `run` then hits the cache. Prints
the cache location and the weights summary.

### `aggfly info <path> [--var t2m]`
No pipeline. Lazily `xr.open_dataset` and print: dims & sizes, coord names, data
variables, detected **calendar** (cftime vs datetime64), **lon convention**
(0–360 vs −180–180 heuristic from the lon range), time span, and chunking. This
is the "help me fill in the config" command — it surfaces exactly the fields a
user must set (`xycoords`, `timecoord`, `lon_is_360`, units for `preprocess`).

### `aggfly validate <config.yaml>`
Static checks, no dask, no data read: schema shape; `regions.path` /
`dataset.path` glob/template resolvable; `preprocess` resolves; every `calc` in
the allowed set (`mean/min/max/sum/dd/bins/sine_dd`); `groupby` recognized;
`ddargs` present when required; output format supported. Prints a normalized plan
(vars, steps, resolved year list, output target) and exits nonzero on any error.
Also surfaces known library limits early (e.g. `groupby: week` on a non-standard
calendar).

## `preprocess` resolution

`preprocess.py` exposes `resolve(spec) -> Callable | None`:

- **Builtin name** (from a small registry): `kelvin_to_celsius`,
  `celsius_to_kelvin`, `identity`, … → returns the registered lambda.
- **Safe arithmetic expression** (a string containing `x`): parse with `ast`,
  walk with a node allowlist — `Expression, BinOp, UnaryOp, Constant, Name(id='x')`,
  and operators `Add/Sub/Mult/Div/Pow`. Reject `Call`, `Attribute`, `Subscript`,
  names other than `x`. Covers `x - 273.15`, `x * 0.1`, `(x - 32) * 5 / 9`.
  No `eval` of arbitrary code.
- **`preprocess_from: path.py:func`** (mutually exclusive with `preprocess`):
  import the file via `importlib.util.spec_from_file_location`, fetch `func`.
  Documented as arbitrary-code / trusted-files-only.

`config.py` enforces that at most one of `preprocess` / `preprocess_from` is set.

## Multi-year templating

`years` accepts `"START:END"` (inclusive) or an explicit list; `--years` overrides.
If `dataset.path` contains `{year}`, loop and `pd.concat`. If it has no `{year}`
(single multi-year file or glob), run once and ignore `years`. Weights are built
once from the sample layer and reused across all years (the grid is identical), so
per-year cost is just read + aggregate. Extensible later to arbitrary template
keys; `{year}` only in v1.

## Error handling

User errors (bad config, missing file, unknown calc) raise
`click.ClickException` with a one-line message and nonzero exit — **no traceback**.
Tracebacks are reserved for actual bugs. `--verbose` re-enables full tracebacks.

## Testing

- **Unit (pure, fast, no dask):**
  - config parse/validate — table-driven good/bad configs; year expansion; flag override precedence.
  - `preprocess.resolve` — builtins; safe expr correctness; **rejects** `__import__`, attribute access, other names; file loader happy path + missing-func error.
  - output-format dispatch by extension/explicit format.
- **Integration (`click.testing.CliRunner`, `tmp_path`):**
  - Build a tiny synthetic zarr + shapefile (reuse the `dataset_360`/`georegion`
    fixture-construction approach already in `tests/`), write a config, run
    `aggfly run`, and assert the output panel **matches the direct-API result**
    (`np.allclose`) — same correctness net as the existing suite.
  - `aggfly info` prints the expected calendar/lon-convention for a known cube.
  - `aggfly validate` exits nonzero on a malformed spec and zero on a good one.

## Docs

- New README "## Command-line interface" section: the four commands, an annotated
  config, and the `info → validate → run` authoring loop.
- Ship `examples/` with a runnable config (area-only) and a pop-weighted one.

## Milestones (incremental, each independently mergeable)

1. **Skeleton + `info`.** ✅ DONE. `aggfly/cli/` package (`main.py` click group +
   `info.py`); `[tool.poetry.scripts] aggfly = "aggfly.cli.main:cli"`; `click`
   dep added (already present transitively via distributed). `info` opens a
   dataset lazily and reports dims/chunks/units + config hints (xycoords,
   `lon_is_360` from the lon range, timecoord, calendar w/ cftime flag, time
   span). `validate`/`weights`/`run` are clean "not implemented yet" stubs so
   the command surface is stable. 7 CLI tests (CliRunner + synthetic zarr);
   full suite 28 passed. (deps: click, pyyaml)
2. **Config layer + `validate`.** ✅ DONE. `config.py` — pure (no dask/I/O):
   `RunConfig`/`SecondaryWeightsConfig` dataclasses, `load_config`/`parse_config`
   (accumulates *all* errors, not just the first), year expansion
   (`"start:end"`/list/int), `resolved_paths()` for `{year}` templating,
   `to_aggregator_dict()` (normalizes `exp`→`np.array` so the library's `[0]`
   indexing works), `check_paths()` (local-only, skips remote URLs), and
   `describe()` normalized-plan printout. Validated against the real accepted
   values from `temporal.py` (calcs, groupby) and catches the documented
   multi-`ddargs`×multi-exponent runtime conflict statically. `validate` command
   prints the plan + path warnings (`--strict` promotes them to errors).
   13 new tests; full suite 41 passed.
3. **`preprocess.py`.** ✅ DONE. `resolve(preprocess, preprocess_from)` →
   callable|None. Named builtins (`kelvin_to_celsius`, `celsius_to_kelvin`,
   `identity`, `pa_to_kpa`, `m_to_mm`); safe arithmetic-in-`x` expressions via an
   `ast` node allowlist + recursive evaluator (`operator`-based, no `eval`) —
   rejects `Call`/`Attribute`/`Subscript`/foreign names/non-numeric constants and
   requires `x` to appear; `path.py:func` file escape hatch via `importlib`
   (clear errors for missing file/func/non-callable). Wired into `validate` (also
   confirms the escape-hatch function exists). 18 new tests; full suite 59 passed.
4. **`run` single-dataset.** ✅ DONE. `pipeline.py` orchestrates
   regions→weights→aggregate over `resolved_paths()` and returns the panel;
   `write_output` handles parquet/feather/csv. `run` command with `-o/--output`,
   `--engine`, `--years`, `--project-dir`, `-v/--verbose` overrides; user errors
   are `ClickException`s (full traceback only under `-v`). **Parity test**:
   CLI output == the equivalent hand-written `af.*` script (`np.allclose`).
   Two findings handled: (a) added **`pyarrow`** dep — pandas 3 here ships
   without it, so parquet/feather output needed it; (b) `dataset_from_path`'s
   clip-to-extent breaks for regions that wrap the antimeridian in 0–360, so
   clipping is now an opt-out flag **`dataset.clip_to_regions`** (default True) —
   it's a read optimization that never changes results (test asserts
   clip-on == clip-off). Weights are built once and reused across paths; the
   first path's dataset is reused for aggregation (weights deep-copy internally).
   6 new tests; full suite 63 passed.
5. **Multi-year loop + `weights` command + backend wiring.** ✅ DONE. Execution
   client wired in `run_pipeline` with the mandatory ordering — `compute_weights`
   first (no client; `calculate_weights` tears one down), then
   `_start_execution_client` (`processes`→`start_dask_client`; `threads`/`none`→
   nothing), aggregate, then `shutdown_dask_client` in a `finally`. `run` gains
   `--backend`/`--n-workers` overrides; standalone `weights` command precomputes
   + caches (reused by `run` — verified: real process-cluster run loaded weights
   from the cache the `weights` command wrote, and its result matched the
   threaded run exactly). Multi-year loop already landed in M4 (`resolved_paths`
   + concat). Ordering guaranteed by a deterministic monkeypatch test
   (`events == ["weights","start","shutdown"]`); backend selection + overrides +
   `weights` command tested. 5 new tests; full suite 67 passed.
6. **Docs + examples.** ✅ DONE. CLI documentation lives in its own file
   **`docs/cli.md`** (kept separate ahead of a planned docs overhaul), covering
   the four commands, the annotated config schema, the spec DSL, `preprocess`,
   `clip_to_regions`, and execution/backends. Runnable configs in **`examples/`**
   (`era5_counties_area.yaml`, `era5_counties_pop.yaml`) — both pass
   `aggfly validate`. README gains a short pointer section to `docs/cli.md`
   rather than inlining the docs.

## Open items to confirm during implementation

- Secondary-weights config surface for `crop` (crop/feed knobs) vs `pop` vs
  `generic` — map cleanly onto `pop_weights_from_path` / `crop_weights_from_path`
  / `secondary_weights_from_path`.
- Whether to expose `time_sel` semantics 1:1 or offer a friendlier `--start/--end`.
- Confirm `weights_from_objects` picks up the cache purely from `project_dir` (the
  code path uses `self.cache`; verify how `cache`/`ProjectCache` is attached from
  `weights_from_objects`).
