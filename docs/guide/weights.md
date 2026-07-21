# Weights

Spatial aggregation is a **weighted** average of grid cells over each region. This
page covers how to compute those weights. For *why* weighting is necessary, see
[Concepts](../concepts.md#why-weights-matter).

## Area weights (the default)

Area weights account for two things: the share of a grid cell's area that falls
inside the region, and the latitude distortion of cell area (cosine correction).
They are always applied.

```python
weights = af.weights_from_objects(
    dataset,
    georegions,
    project_dir=project_dir
)
weights.calculate_weights()
```

`weights.weights` is then a DataFrame keyed by `cell_id`/`region_id` with
`area_weight`, `raster_weight`, and a combined `weight` column.

## Secondary weights

Secondary weights reweight cells by another raster — population, cropland, or any
gridded exposure measure. The final `weight` is the **product** of the area weight
and the secondary weight.

Load the secondary raster first, then pass it to `weights_from_objects`:

```python
secondary_weights = af.pop_weights_from_path(
    "~/data/population/landscan-global-2016.tif"
)

weights = af.weights_from_objects(
    dataset,
    georegions,
    secondary_weights=secondary_weights,
    project_dir=project_dir
)
weights.calculate_weights()
```

### Available loaders

| Function | Use for |
|---|---|
| `af.pop_weights_from_path(path)` | Population rasters (e.g. LandScan). |
| `af.crop_weights_from_path(path, crop=..., feed=...)` | Cropland rasters; `crop` selects the crop, `feed` the regime (e.g. rainfed). |
| `af.secondary_weights_from_path(path)` | Any other generic raster. |

## Arguments

| Argument | Meaning |
|---|---|
| `dataset` | A sample layer of the climate raster — supplies the grid structure. |
| `georegions` | The `GeoRegions` object for the target regions. |
| `project_dir` | Project directory; enables the weight cache. |
| `secondary_weights` | Optional `SecondaryWeights` object, as above. |

## Caching

Weights depend only on the grid and the regions — **not** on the data values — so
they are computed once and reused across every year of data. With `project_dir`
set, `calculate_weights()` caches its result under
`{project_dir}/tmp/{module}/{sha}/`, keyed by a hash of the parameters. Reruns are
a cache hit.

From the CLI, `aggfly weights config.yaml` precomputes and caches weights without
running an aggregation; a later `aggfly run` picks them up from the cache.

## Notes and constraints

- `GridWeights` requires the grid to be in the −180–180 longitude convention
  (it asserts `not grid.lon_is_360`). `Dataset` handles the conversion when
  loading, so this is normally transparent.
- `calculate_weights()` **shuts down an active Dask client.** If you are managing
  your own client, compute weights *before* starting it. The CLI enforces this
  ordering automatically.
