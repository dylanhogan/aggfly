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

### The latitude correction and secondary weights

Area weighting multiplies by `cos(latitude)`, because a cell of fixed extent in
degrees covers less physical ground toward the poles.

That correction is **not** applied on top of a secondary raster. A population or
cropland raster already reports how much of the quantity sits in each cell — a
poleward cell being physically smaller is already reflected in its lower count —
so applying `cos(latitude)` as well would count the same distortion twice and
bias any region spanning a wide range of latitudes.

`cosine_area` therefore defaults to **True for area-only weights** and **False
when a secondary raster is supplied**. The defining property holds either way: if
population is spread uniformly over the surface, population weighting reproduces
plain area weighting.

Override it explicitly if your secondary raster is a density per unit *physical*
area (e.g. people per km²), where the conversion is still needed:

```python
weights = af.weights_from_objects(
    dataset, georegions,
    secondary_weights=secondary_weights,
    cosine_area=True,          # raster is people per km², not people per cell
)
```

### Available loaders

| Function | Use for |
|---|---|
| `af.secondary_weights_from_path(path, ...)` | Any raster. The general loader. |
| `af.pop_weights_from_path(path)` | Population rasters (e.g. LandScan). |
| `af.crop_weights_from_path(path, crop=..., feed=...)` | Cropland rasters; `crop` selects the crop, `feed` the regime (e.g. rainfed). |

All three read **`.tif`, `.zarr` and `.nc`**, and the latter two are thin
wrappers over the first.

### Selecting part of a raster

A secondary raster often holds more than one layer. `var` picks a data variable
and `sel` picks along a coordinate:

```python
# one crop out of a multi-crop cropland store
af.secondary_weights_from_path("cropland.zarr", var="layer", sel={"crop": "corn"})

# one band of a multi-band GeoTIFF
af.secondary_weights_from_path("landcover.tif", sel={"band": 3})
```

This is what makes cropland weights work, and it is not crop-specific — any
selectable coordinate (a scenario, a year, a band) can be used.

### Distinguishing variants in the cache

`cache_identifier` is an extra discriminator folded into the cache key. `path`
and the raster itself already feed that key, so you only need it when what
separates two variants is *not* visible in either — for instance the same file
read with two different `preprocess` functions. `crop_weights_from_path` uses it
to keep feed regimes apart.

> **CRS note.** A CRS does not survive a Zarr round trip, so a `.zarr` secondary
> raster often needs an explicit `crs="WGS84"`. Weight construction raises a
> clear error rather than guessing.

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

## Missing data in the secondary raster

Population rasters are usually clipped to a country, leaving nodata outside it
(the WorldPop Kenya file is ~42% nodata). Those pixels are **excluded** from the
rescaling average rather than treated as zero, which makes `raster_weight` a
density per *valid* pixel. Combined with the overlap fraction that is the right
behaviour when the nodata mask follows the region boundary.

It does mean that nodata *inside* a region — inland water, or genuine gaps in the
raster — inflates that cell's weight, since the empty part is ignored rather than
counted as unpopulated. Use a raster with explicit zeros if you want those areas
to count as empty.

## Notes and constraints

- `GridWeights` requires the grid to be in the −180–180 longitude convention
  (it asserts `not grid.lon_is_360`). `Dataset` handles the conversion when
  loading, so this is normally transparent.
- `calculate_weights()` **shuts down an active Dask client.** If you are managing
  your own client, compute weights *before* starting it. The CLI enforces this
  ordering automatically.
