"""``aggfly info`` — inspect a raster dataset to help author a config.

Opens the dataset lazily (no full read) and reports the handful of facts a user
needs to fill into a config file: coordinate names (``xycoords``/``timecoord``),
the longitude convention (``lon_is_360``), the calendar, the time span, and the
chunking. It reads only coordinate arrays (small, 1-D), never the data variable.
"""

import json

import click
import numpy as np
import xarray as xr

# Common coordinate aliases, in preference order.
_LON_NAMES = ("longitude", "lon", "x", "nav_lon")
_LAT_NAMES = ("latitude", "lat", "y", "nav_lat")
_TIME_NAMES = ("time", "valid_time", "t")


def _open(path, storage_options):
    """Lazily open ``path`` as an xarray Dataset, choosing the zarr engine by name."""
    kwargs = {}
    if storage_options is not None:
        kwargs["storage_options"] = storage_options
    if ".zarr" in str(path):
        kwargs["engine"] = "zarr"
    # chunks={} keeps everything lazy/dask-backed so nothing is read eagerly.
    return xr.open_dataset(path, chunks={}, **kwargs)


def _first_present(names, candidates):
    """Return the first candidate found in ``names`` (case-insensitive), or None."""
    lowered = {str(n).lower(): n for n in names}
    for c in candidates:
        if c in lowered:
            return lowered[c]
    return None


def _lon_report(ds):
    """Detect the longitude coordinate and infer the 0–360 vs −180–180 convention."""
    lon_name = _first_present(ds.coords, _LON_NAMES)
    if lon_name is None:
        return None, None, None
    vals = np.asarray(ds[lon_name].values)
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    # ERA5-style stores use 0..360; the -180..180 convention never exceeds 180.
    is_360 = hi > 180.0
    return lon_name, (lo, hi), is_360


def _time_report(ds):
    """Detect the time coordinate and report its calendar, span, and length."""
    time_name = _first_present(ds.coords, _TIME_NAMES)
    if time_name is None:
        return None, None, None, None, None
    tcoord = ds[time_name]
    try:
        calendar = tcoord.dt.calendar
    except (AttributeError, TypeError):
        calendar = "unknown"
    is_cftime = isinstance(ds.indexes.get(time_name), xr.CFTimeIndex)
    n = int(tcoord.sizes.get(time_name, tcoord.size))
    try:
        span = (str(tcoord.values.min()), str(tcoord.values.max()))
    except Exception:
        span = None
    return time_name, calendar, is_cftime, n, span


def run(path, var=None, storage_options=None):
    """Print a human-readable report for the dataset at ``path``."""
    if storage_options is not None and isinstance(storage_options, str):
        try:
            storage_options = json.loads(storage_options)
        except json.JSONDecodeError as e:
            raise click.ClickException(f"--storage-options is not valid JSON: {e}")

    try:
        ds = _open(path, storage_options)
    except Exception as e:  # surface any backend error as a clean CLI message
        raise click.ClickException(f"Could not open {path!r}: {e}")

    data_vars = list(ds.data_vars)
    if var is not None and var not in data_vars:
        raise click.ClickException(
            f"Variable {var!r} not found. Available: {', '.join(data_vars) or '(none)'}"
        )

    click.echo(f"Dataset: {path}")
    click.echo(f"  data variables : {', '.join(data_vars) or '(none)'}")

    # Per-variable dims/shape/chunks (all vars, or just the requested one).
    for name in ([var] if var else data_vars):
        da = ds[name]
        dims = ", ".join(f"{d}={s}" for d, s in da.sizes.items())
        click.echo(f"  {name}:")
        click.echo(f"    dims   : {dims}")
        if da.chunks is not None:
            chunks = ", ".join(
                f"{d}={c[0]}" for d, c in zip(da.dims, da.chunks)
            )
            click.echo(f"    chunks : {chunks}")
        units = da.attrs.get("units")
        if units:
            click.echo(f"    units  : {units}")

    # Coordinate / config hints.
    lon_name, lon_range, is_360 = _lon_report(ds)
    lat_name = _first_present(ds.coords, _LAT_NAMES)
    time_name, calendar, is_cftime, n_time, span = _time_report(ds)

    click.echo("  config hints:")
    if lon_name and lat_name:
        click.echo(f"    xycoords   : [{lon_name}, {lat_name}]")
    if lon_range is not None:
        click.echo(
            f"    lon range  : {lon_range[0]:.4g} .. {lon_range[1]:.4g}"
            f"  → lon_is_360: {str(bool(is_360)).lower()}"
        )
    if time_name:
        click.echo(f"    timecoord  : {time_name}")
        cft = "  (cftime / non-standard)" if is_cftime else ""
        click.echo(f"    calendar   : {calendar}{cft}")
        if n_time is not None:
            click.echo(f"    time steps : {n_time}")
        if span is not None:
            click.echo(f"    time span  : {span[0]} .. {span[1]}")

    ds.close()
