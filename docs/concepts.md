# Concepts

This page explains *what* `aggfly` does and *why* the pipeline is shaped the way
it is. For runnable code, see the [Quickstart](guide/quickstart.md).

## The problem

Gridded climate data (e.g. ERA5) is published on a fine raster — hourly values on
a 0.25° × 0.25° global grid. Most research questions, however, are asked at the
level of administrative regions over coarser time periods: *what was the average
temperature in each US county in each year, and how many growing degree days did
it accumulate?*

Turning the former into the latter is a spatial **and** temporal aggregation, and
doing it correctly is fiddly: grid cells vary in area with latitude, region borders
cut through cells, and the quantity you want is often nonlinear in temperature
(degree days, bins, polynomials) so you cannot simply average first and transform
later. `aggfly` automates this.

## The three inputs

1. **Shapefile** — boundaries of the target administrative regions (e.g. world
   countries, US counties).
2. **Climate dataset** — the fine-grained raster you want to aggregate (e.g. ERA5
   hourly 2m temperature).
3. **Secondary weights dataset** *(optional)* — a raster describing local exposure
   (population, cropland) used to weight cells within a region.

## The three-stage pipeline

Each stage produces an object consumed by the next:

```
    shapefile ──► GeoRegions ─┐
                              ├─► GridWeights ──► aggregate_dataset ──► panel DataFrame
    raster ─────► Dataset ────┘                        ▲
                                                       │
    secondary raster ──► SecondaryWeights ─────────────┘
```

1. **Load inputs** — `georegions_from_path()` and `dataset_from_path()` produce
   `GeoRegions` and `Dataset`. The `Dataset` normalizes dimension names and handles
   the 0–360 vs −180–180 longitude convention.
2. **Compute weights** — `weights_from_objects(...).calculate_weights()` produces a
   `GridWeights` table keyed by `cell_id`/`region_id`. See [Weights](guide/weights.md).
3. **Aggregate** — `aggregate_dataset()` runs **temporal aggregation first, then
   spatial**, returning a region-by-period panel. See [Aggregation](guide/aggregation.md).

### Why temporal before spatial

The order is not arbitrary. Many quantities of interest are **nonlinear** in the
underlying variable — degree days, temperature bins, polynomials. For these,
aggregating over space first and then transforming gives a different (and wrong)
answer than transforming each grid cell's time series and then averaging over
space. `aggfly` therefore applies your temporal specification cell-by-cell, and
only then takes the weighted spatial average.

## Why weights matter

There are two categories of weights:

**Area weights** are always needed, for two reasons:

- **Cell area varies with latitude.** Lines of longitude converge toward the poles,
  so a 0.25° × 0.25° cell covers far less ground at 60°N than at the equator. A
  naive unweighted mean would over-count high-latitude cells. `aggfly` applies a
  cosine-of-latitude correction.
- **Region borders cut through cells.** A cell only partly inside a region should
  contribute in proportion to the overlapping area.

**Secondary weights** capture *who or what experiences* the climate. If you study
human health, the population-weighted average temperature is more meaningful than
the land-area average — most of a large county's area may be empty. If you study
agriculture, weight by cropland (or a specific crop). The final weight is the
**product** of the area weight and the secondary weight.

## Execution model

Aggregation is Dask-backed and **execution-backend-agnostic**: it runs on whatever
Dask scheduler or distributed client is active, falling back to the threaded
scheduler when none is. Results are identical across backends — only speed changes.
Two independent knobs (`engine=` for the temporal kernel, and the active Dask
client for task execution) are covered in [Execution & scaling](guide/execution.md).

## Caching

Several entry points take a `project_dir`. `ProjectCache` hashes a module's
parameters into a SHA and caches intermediate results under
`{project_dir}/tmp/{module}/{sha}/`. Weight computation is the main beneficiary:
weights depend only on the grid and the regions, so they are computed once and
reused across every year of data.

## Calendars

Climate-model output (CMIP6/CMIP5) often uses non-standard CF calendars such as
`noleap` or `360_day`. `aggfly` supports these out of the box and preserves the
model calendar. See [Calendars](guide/calendars.md).
