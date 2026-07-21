# aggfly: Efficient climate data aggregation

[![PyPI version](https://badge.fury.io/py/aggfly.svg)](https://badge.fury.io/py/aggfly)

> **NOTE:** aggfly is still in development and may not be stable for new users.
> Please proceed with caution.

`aggfly` is a Python package developed to facilitate and harmonize the temporal and
spatial aggregation of gridded climate data. It performs linear and nonlinear
aggregations of weather data across different time periods — degree days, bins,
daily polynomials — and supports spatial aggregation by weighting gridded data
according to administrative boundaries and local exposures such as human
populations and crop distributions.

The package automates complex and memory-intensive geospatial operations, which are
common challenges for researchers dealing with large climate datasets. `aggfly` is
useful for researchers studying the impacts of weather and climate on other
variables, such as human health, agriculture, and economic growth.

## Installation

```bash
pip install aggfly
```

Requires Python 3.11–3.13. Full instructions — including the uv-based development
setup and Jupyter kernels — are in **[docs/installation.md](docs/installation.md)**.

## Quickstart

Three inputs: a **shapefile** of target regions, a **climate raster** to aggregate,
and optionally a **secondary weights raster** (population, cropland).

```python
import numpy as np
import aggfly as af

# 1. Load regions and a sample raster layer
georegions = af.georegions_from_path("counties.shp", regionid="GEOID")
dataset = af.dataset_from_path(
    "era5_2017.zarr", var="t2m",
    georegions=georegions,
    preprocess=lambda x: x - 273.15,     # Kelvin → Celsius
)

# 2. Compute weights (cached; reused across years)
weights = af.weights_from_objects(dataset, georegions, project_dir="./proj")
weights.calculate_weights()

# 3. Transform and aggregate → a region-by-period panel
df = af.aggregate_dataset(
    dataset=dataset,
    weights=weights,
    tavg=[
        ('aggregate', {'calc': 'mean', 'groupby': 'date'}),
        ('transform', {'transform': 'power', 'exp': np.arange(1, 3)}),
        ('aggregate', {'calc': 'sum', 'groupby': 'year'}),
    ],
    growing_dday=[
        ('aggregate', {'calc': 'dd', 'groupby': 'date', 'ddargs': [10, 30, 0]}),
        ('aggregate', {'calc': 'sum', 'groupby': 'year'}),
    ],
)
```

See the [Quickstart guide](docs/guide/quickstart.md) for a walkthrough of each step.

## Command-line interface

The whole workflow can also be driven from a YAML config file — no Python script
required:

```bash
aggfly info era5_2017.zarr --var t2m   # discover coords, calendar, lon convention
aggfly validate config.yaml            # check the config, no data read
aggfly run config.yaml                 # aggregate → panel
```

See the **[CLI reference](docs/cli.md)** and runnable configs in
[`examples/`](examples/).

## Documentation

| | |
|---|---|
| [Documentation home](docs/index.md) | Full table of contents. |
| [Installation](docs/installation.md) | pip for users, uv for development. |
| [Concepts](docs/concepts.md) | How the pipeline works and why weights matter. |
| [Quickstart](docs/guide/quickstart.md) | End-to-end Python API walkthrough. |
| [Weights](docs/guide/weights.md) | Area and secondary (population/crop) weighting. |
| [Aggregation](docs/guide/aggregation.md) | The spec DSL: calcs, transforms, degree days, bins. |
| [Execution & scaling](docs/guide/execution.md) | Dask backends and recipes by hardware. |
| [Calendars](docs/guide/calendars.md) | CMIP6 and non-standard CF calendars. |
| [CLI](docs/cli.md) | `info`, `validate`, `weights`, `run`. |
| [API reference](docs/api.md) | The public Python API. |

Worked examples live in [`examples/`](examples/) (CLI configs) and
[`examples/notebooks/`](examples/notebooks/).

**New to aggfly?** [`quickstart_public_data.ipynb`](examples/notebooks/quickstart_public_data.ipynb)
is a standalone notebook that downloads all of its own data (~25 MB, public, no accounts) and runs
the full pipeline end to end — no local files required.

## Acknowledgements

I gratefully acknowledge the support and funding provided by the Climate &
Environment Program at Private Enterprise Development in Low Income Countries
(PEDL, CEPR), and stellar research assistance from Giovanni Brocca and Nick Silvis.
Research assistance was funded by Professors Colmer, Porzio, and Rossi through their
project "The Human (Capital) Side of Climate Change," which was financed by the
International Growth Center (XXX-23020) and Columbia Business School. The funding
and research assistance provided were instrumental in the development of this
package. I sincerely thank them for their commitment to advancing research and
innovation.
