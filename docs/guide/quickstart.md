# Quickstart

A complete end-to-end run with the Python API. If you'd rather drive the pipeline
from a YAML config file with no Python at all, see the [CLI reference](../cli.md).

For the conceptual background — why weights, why temporal-before-spatial — see
[Concepts](../concepts.md).

## Setup

```python
import numpy as np
import aggfly as af

project_dir = '/user/name/aggfly_repository'
```

Setting `project_dir` once enables the weight cache, so you don't pass it to every
call.

## 1. Load the regions and a sample raster layer

Load the shapefile of target regions. `regionid` names the column holding the
region identifier:

```python
georegions = af.georegions_from_path(
    "~/data/shapefiles/county/cb_2018_us_county_500k.shp",
    regionid='GEOID'
)
```

Then load a **sample layer** of the climate raster. This is only used to describe
the grid so that weights can be computed — it does not need to be the full dataset:

```python
dataset = af.dataset_from_path(
    "~/data/annual/tempPrecLand2017.zarr",
    var='t2m',
    name='era5',
    georegions=georegions,
    preprocess=lambda x: (x - 273.15),   # Kelvin → Celsius
)
dataset.da
```

Key arguments:

| Argument | Meaning |
|---|---|
| `var` | The variable to transform and aggregate. |
| `preprocess` | A function applied to raw values before aggregation — unit conversion, time shifts, etc. |
| `georegions` | The `GeoRegions` object created above. |
| `name` | A label for this dataset. |

Not sure what to pass for coordinate names, longitude convention, or calendar?
Run `aggfly info <path> --var t2m` — it reports exactly those fields.

## 2. Compute weights

For plain area weights:

```python
weights = af.weights_from_objects(
    dataset,
    georegions,
    project_dir=project_dir
)
weights.calculate_weights()
```

To weight by population, cropland, or another raster, see [Weights](weights.md).

Weights depend only on the grid and the regions — not on the data values — so they
are computed once and reused across every year.

## 3. Transform and aggregate

Load the full dataset (same call as step 1, pointed at the data you actually want
to aggregate), then aggregate:

```python
dataset = af.dataset_from_path(
    f"~/data/annual/tempPrecLand{year}.zarr",
    var='t2m',
    name='era5',
    georegions=georegions,
    preprocess=lambda x: (x - 273.15)
)

output_df = af.aggregate_dataset(
    dataset=dataset,
    weights=weights,
    tavg=[
        ('aggregate', {'calc': 'mean', 'groupby': 'date'}),
        ('transform', {'transform': 'power', 'exp': np.arange(1, 3)}),
        ('aggregate', {'calc': 'sum', 'groupby': 'year'})
    ],
    growing_dday=[
        ('aggregate', {'calc': 'dd', 'groupby': 'date', 'ddargs': [10, 30, 0]}),
        ('aggregate', {'calc': 'sum', 'groupby': 'year'}),
    ],
)
```

The result is a pandas DataFrame: one row per region per period, one column per
named output variable.

Each keyword argument (`tavg`, `growing_dday`, …) names an output variable and maps
to a list of steps applied in order. That mini-language is documented in full in
[Aggregation](aggregation.md).

## Next steps

- [Weights](weights.md) — area and secondary (population/crop) weighting
- [Aggregation](aggregation.md) — the full spec DSL: calcs, transforms, bins
- [Execution & scaling](execution.md) — pick a Dask backend to match your hardware
- [Calendars](calendars.md) — CMIP6 and non-standard calendars
- [`quickstart_public_data.ipynb`](../../examples/notebooks/quickstart_public_data.ipynb) — a
  runnable version of this page that downloads its own public data (no local files needed)
- [`examples/`](../../examples/) — runnable CLI configs
