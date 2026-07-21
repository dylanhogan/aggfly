# aggfly documentation

`aggfly` turns fine-grained gridded climate data (e.g. ERA5) into region-by-period
panel data — weighted by area and, optionally, by a secondary variable such as
population or cropland.

> `aggfly` is still in development and may not be stable for new users. Please
> proceed with caution.

## Start here

| | |
|---|---|
| [Installation](installation.md) | Install with pip; develop with uv. |
| [Concepts](concepts.md) | What the pipeline does and why — weights, ordering, caching. |
| [Quickstart](guide/quickstart.md) | A complete end-to-end run with the Python API. |

## User guide

| | |
|---|---|
| [Weights](guide/weights.md) | Area weights and secondary (population/crop) weights. |
| [Aggregation](guide/aggregation.md) | The spec DSL: calcs, transforms, degree days, bins. |
| [Execution & scaling](guide/execution.md) | Dask backends, the `engine=` knob, recipes by hardware. |
| [Calendars](guide/calendars.md) | CMIP6 and non-standard CF calendars. |

## Reference

| | |
|---|---|
| [CLI](cli.md) | Run the whole pipeline from a YAML config — `info`, `validate`, `weights`, `run`. |
| [API](api.md) | The public Python API. |

## Examples

- [`examples/`](../examples/) — runnable CLI configs (area-weighted and
  population-weighted).
- [**`quickstart_public_data.ipynb`**](../examples/notebooks/quickstart_public_data.ipynb) — a
  standalone notebook that downloads everything it needs (~25 MB of public data, no accounts or
  local files) and runs the full pipeline, comparing area- vs population-weighted results.
- [`aggfly_example_data/config.yaml`](../examples/notebooks/aggfly_example_data/config.yaml) —
  the CLI config that notebook generates, kept as a worked example.

## Two ways to use aggfly

**Python API** — full control, composable into a larger script:

```python
import aggfly as af

georegions = af.georegions_from_path("counties.shp", regionid="GEOID")
dataset    = af.dataset_from_path("era5_2017.zarr", var="t2m",
                                  georegions=georegions,
                                  preprocess=lambda x: x - 273.15)
weights    = af.weights_from_objects(dataset, georegions, project_dir=".")
weights.calculate_weights()

df = af.aggregate_dataset(dataset=dataset, weights=weights,
        tavg=[('aggregate', {'calc': 'mean', 'groupby': 'date'})])
```

**Command line** — the same pipeline from a config file, no Python required:

```bash
aggfly info era5_2017.zarr --var t2m   # discover coords, calendar, lon convention
aggfly validate config.yaml            # check the config, no data read
aggfly run config.yaml                 # aggregate → panel
```
