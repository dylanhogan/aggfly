"""
Convert a normalized aggfly ``Dataset`` into a rechunked, time-contiguous Zarr store.

**Dataset-agnostic by construction.** This operates on an ``af.Dataset``, whose dims are
already normalized to ``(latitude, longitude, time)`` by ``dataset_from_path`` via its
``xycoords`` / ``timecoord`` / ``lon_is_360`` / ``preprocess`` parameters. Every
per-source difference (dim names, time-coord name, units, longitude convention, variable
name) is absorbed there, so the conversion itself needs no per-dataset special-casing —
any gridded source aggfly can load can be converted.

**Why.** Raw NetCDF is typically bricked in time (and HDF5-lock-bound on read); a
time-contiguous Zarr layout makes temporal aggregation dramatically faster and the store
smaller (see ``benchmarks/profile_netcdf_zarr.py``). Chunk sizes are chosen size-aware:
full-time chunks when they fit a memory budget, otherwise time split into budget-sized
blocks with a reasonable spatial tile.
"""
import os
import shutil
from typing import Dict, Optional, Union

from .dataset import Dataset, dataset_from_path

# Below this square-tile size, keeping full-time chunks would fragment space too much,
# so we split time instead. Above it, we keep time contiguous.
_MIN_TILE = 32
# Spatial tile used when time must be split; capped so tiles stay cache-friendly.
_SPLIT_TILE = 128
_MAX_TILE = 256


def _auto_chunks(sizes: Dict[str, int], itemsize: int, target_mb: float) -> Dict[str, int]:
    """Pick time-contiguous chunks under a per-chunk byte budget.

    Prefer a single full-time chunk when a square spatial tile of at least ``_MIN_TILE``
    keeps the chunk within ``target_mb``; otherwise split time into the largest blocks
    that fit alongside a fixed ``_SPLIT_TILE`` spatial tile.
    """
    Y, X, T = sizes["latitude"], sizes["longitude"], sizes["time"]
    budget = max(1, int(target_mb * 1024 * 1024 / itemsize))  # elements per chunk
    s_full = int((budget / T) ** 0.5)  # square spatial tile if time stays contiguous
    if s_full >= _MIN_TILE:
        s = int(min(s_full, _MAX_TILE, Y, X))
        return {"time": -1, "latitude": s, "longitude": s}
    # Time too long to keep contiguous within budget -> split time, fix a decent tile.
    s = int(min(_SPLIT_TILE, Y, X))
    t = max(1, budget // (s * s))
    return {"time": min(t, T), "latitude": s, "longitude": s}


def dataset_to_zarr(
    dataset: Dataset,
    store: str,
    chunking: Union[str, Dict[str, int]] = "auto",
    target_mb: float = 256,
    overwrite: bool = False,
    return_dataset: bool = True,
) -> Optional[Dataset]:
    """
    Write a ``Dataset`` to a time-contiguous Zarr store.

    Parameters
    ----------
    dataset : Dataset
        A normalized aggfly Dataset (dims ``latitude``, ``longitude``, ``time``).
    store : str
        Path of the Zarr store to create.
    chunking : "auto" or dict, optional
        ``"auto"`` (default) chooses time-contiguous chunks under ``target_mb`` per chunk.
        A dict of ``{dim: size}`` sets chunk sizes explicitly (use ``-1`` for whole-dim).
    target_mb : float, optional
        Per-chunk size budget for ``chunking="auto"`` (default 256 MB).
    overwrite : bool, optional
        Replace ``store`` if it already exists (default False -> raise).
    return_dataset : bool, optional
        If True (default), reopen the store with its native chunks and return a Dataset
        that plugs straight into ``weights_from_objects`` / ``aggregate_dataset``.

    Returns
    -------
    Dataset or None
        The reopened Dataset (or None if ``return_dataset`` is False).
    """
    da = dataset.da
    name = getattr(dataset, "name", None) or da.name or "variable"

    if chunking == "auto":
        chunks = _auto_chunks(dict(da.sizes), da.dtype.itemsize, target_mb)
    elif isinstance(chunking, dict):
        chunks = dict(chunking)
    else:
        raise ValueError("chunking must be 'auto' or a dict of chunk sizes")
    # Chunk any extra dims (e.g. band/level) whole.
    for d in da.dims:
        chunks.setdefault(d, -1)

    # Strip source encoding so a source's scale_factor/add_offset can't silently
    # re-quantize the data and stale chunk-encoding (e.g. NetCDF bricks) can't conflict
    # with the new Zarr chunks.
    da = da.copy()
    da.encoding = {}
    for c in da.coords:
        da[c].encoding = {}

    if os.path.exists(store):
        if overwrite:
            shutil.rmtree(store)
        else:
            raise FileExistsError(f"{store} exists; pass overwrite=True to replace it")

    # Default (consolidated) metadata for fast read-back of chunky climate stores. On
    # zarr v3 this emits an informational "not yet in the v3 spec" notice; it is benign
    # and xarray reads it back fine.
    da.to_dataset(name=name).chunk(chunks).to_zarr(store, mode="w")

    if not return_dataset:
        return None
    # Reopen with the store's native (time-contiguous) chunks so downstream aggregation
    # reads the fast layout; the data is already normalized, so defaults line up.
    return dataset_from_path(
        store, var=name, lon_is_360=dataset.lon_is_360, chunks={}
    )


def zarr_from_path(
    path,
    var: str,
    store: str,
    *,
    xycoords=("longitude", "latitude"),
    timecoord: str = "time",
    lon_is_360: bool = True,
    preprocess=None,
    chunking: Union[str, Dict[str, int]] = "auto",
    target_mb: float = 256,
    overwrite: bool = False,
    **kwargs,
) -> Optional[Dataset]:
    """
    Load any gridded source with ``dataset_from_path`` and convert it to Zarr in one call.

    Accepts the same normalization parameters as ``dataset_from_path``
    (``var``/``xycoords``/``timecoord``/``lon_is_360``/``preprocess``), so it is agnostic
    to the source dataset. Returns a Dataset pointing at the new time-contiguous store.
    """
    ds = dataset_from_path(
        path,
        var=var,
        xycoords=xycoords,
        timecoord=timecoord,
        lon_is_360=lon_is_360,
        preprocess=preprocess,
        **kwargs,
    )
    return dataset_to_zarr(
        ds, store, chunking=chunking, target_mb=target_mb, overwrite=overwrite
    )
