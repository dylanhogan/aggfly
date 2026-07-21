# Installation

> `aggfly` is still in development and may not be stable for new users. Please
> proceed with caution.

## Requirements

- **Python 3.11–3.13** (`>=3.11,<3.14`)

`aggfly` depends on a geospatial stack (geopandas, rasterio, rioxarray, shapely)
and a Dask/xarray stack (dask, xarray, zarr, netCDF4). All of these ship binary
wheels on PyPI, so no system GDAL/PROJ installation is required.

## Using the package (pip)

```bash
pip install aggfly
```

We recommend installing into a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install aggfly
```

Installing the package also puts an `aggfly` executable on your `PATH` — see the
[CLI reference](cli.md).

### Reading from cloud object storage

aggfly can open Zarr stores directly from object storage, but the backend that
talks to each service is an optional extra — `gcsfs` alone pulls in around 38
packages that a user reading local files never needs:

```bash
pip install "aggfly[gcs]"     # Google Cloud Storage  (gs:// , gcs://)
pip install "aggfly[s3]"      # Amazon S3             (s3://)
pip install "aggfly[cloud]"   # both
```

```python
ds = af.dataset_from_path(
    "gs://cmip6/CMIP6/CMIP/NOAA-GFDL/GFDL-CM4/historical/r1i1p1f1/day/tas/gr1/v20180701/",
    var="tas", storage_options={"token": "anon"},
    xycoords=("lon", "lat"), lon_is_360=True,
)
```

Passing a `gs://` or `s3://` path without the matching extra raises an error
naming the one to install.

Zarr stores are detected even when the path does not end in `.zarr` — as with
Pangeo's CMIP6 collection above — by looking for zarr's root metadata. Pass
`engine="zarr"` explicitly to skip that check.

## Developing aggfly (uv)

The repository uses [uv](https://docs.astral.sh/uv/) for dependency management
(`pyproject.toml` with PEP 621 metadata + `uv.lock`):

```bash
git clone https://github.com/dylanhogan/aggfly.git
cd aggfly
uv sync                  # creates .venv and installs deps, incl. the dev group
uv run pytest            # run the test suite
uv run aggfly --help     # invoke the CLI
```

uv downloads and manages its own Python interpreter (pinned via `.python-version`),
so runs are fully isolated from any system or conda Python. This matters in
practice: a broken PROJ database in an ambient conda environment will otherwise
surface as `pyproj` `CRSError: Invalid projection: WGS84` during weight
computation.

To run a single test:

```bash
uv run pytest aggfly/tests/test_aggregate.py::test_weights
```

## A note on conda

Earlier versions of this document recommended conda. That is no longer necessary —
the dependency stack installs cleanly from PyPI wheels. If you do use conda,
create the environment with a supported Python and install `aggfly` with `pip`
inside it, and be aware that a misconfigured conda PROJ installation is a common
source of CRS errors.

## Jupyter

To use `aggfly` from a Jupyter session, register your environment as a kernel:

```bash
source .venv/bin/activate
pip install ipykernel
python -m ipykernel install --user --name aggfly
```

See the [IPython documentation](https://ipython.readthedocs.io/en/latest/install/kernel_install.html#kernels-for-different-environments)
if the environment does not appear in Jupyter.
