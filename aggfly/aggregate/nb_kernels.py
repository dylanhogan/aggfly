"""
Numba nogil grouped-reduction kernels for the temporal aggregation path.

These are an optional, faster backend for `TemporalAggregator.execute`, selected
with ``engine="numba"``. They fuse the per-group reduction into a single compiled,
GIL-releasing pass per spatial chunk (via ``dask.array.map_blocks``), which lets
many small chunks run in parallel at low memory — where the default
``.resample(freq).reduce(func)`` path is serialized by the GIL on per-task Python
overhead.

`numba_resample` returns an ``xarray.DataArray`` shaped exactly like the current
``.resample(freq).reduce(func)`` output, including the trailing ``dd`` axis for
multi-``ddargs`` reductions, so it is a drop-in replacement inside ``execute``.

NaN convention (matches the current numpy funcs, except where noted):
  - mean/sum/min/max/dd/sine_dd : any NaN in a group window -> NaN for that group.
  - bins                        : a NaN *value* is treated as out-of-range (a
                                  count, never NaN), matching the bool-mask sum.
  - empty bins (time gaps)      : every reducer -> NaN, matching xarray's reindex
                                  fill for empty resample bins.
  - sine_dd                     : uses the any-NaN-in-window rule above. NOTE the
                                  legacy `_sine_cdd`/`_sine_hdd` mask NaNs along the
                                  wrong axis (`frame[:, :, 0]` indexes longitude,
                                  not time); this kernel fixes that. On non-NaN data
                                  the two agree to ~1e-15.
"""
import numpy as np
import pandas as pd
import xarray as xr
import dask.array as darr
from numba import njit, prange

_STAT_CODE = {"mean": 0, "sum": 1, "min": 2, "max": 3, "nanmean": 4}
NUMBA_CALCS = frozenset({"mean", "nanmean", "sum", "min", "max", "dd", "bins", "sine_dd"})

# Cells per spatial block at/below which the numba engine wins under engine="auto".
# The numba kernel dominates on many small spatial blocks (the GIL-bound, low-memory
# regime); the dask path's vectorized numpy wins on few large blocks (already
# compute-bound, and per-block numba threads oversubscribe the dask pool). On real
# ERA5, numba was ~12x faster at 50x50 (2.5k cells) and ~0.7x at 250x250 (62.5k);
# 150x150 (22.5k) sits near the crossover. Tunable.
NUMBA_MAX_CELLS_PER_BLOCK = 150 * 150


def max_spatial_block_cells(da: xr.DataArray) -> int:
    """Largest number of cells in a single spatial (non-time) dask block.

    For a non-dask-backed array (no chunks), the whole spatial extent is one block.
    """
    spatial = [d for d in da.dims if d != "time"]
    if da.chunks is None:
        return int(np.prod([da.sizes[d] for d in spatial])) if spatial else 1
    biggest = 1
    for d in spatial:
        biggest *= max(da.chunks[da.get_axis_num(d)])
    return int(biggest)


def resolve_engine(engine: str, da: xr.DataArray, calc: str) -> str:
    """Resolve engine=("dask"|"numba"|"auto") to a concrete "dask" or "numba".

    - Calcs the numba backend can't do always resolve to "dask".
    - "auto" picks "numba" only for small spatial blocks (see NUMBA_MAX_CELLS_PER_BLOCK).
    Pure function (no compute) so it is cheap and unit-testable.
    """
    if calc not in NUMBA_CALCS:
        return "dask"
    if engine == "numba":
        return "numba"
    if engine == "dask":
        return "dask"
    if engine == "auto":
        return "numba" if max_spatial_block_cells(da) <= NUMBA_MAX_CELLS_PER_BLOCK else "dask"
    raise ValueError(f"engine must be 'dask', 'numba', or 'auto', got {engine!r}")


# --------------------------------------------------------------------------- #
# group construction — mirror xarray's resample bins
# --------------------------------------------------------------------------- #
def resample_groups(tindex, freq: str):
    """Return (contiguous group bounds, output time labels) matching xarray resample.

    Bins mirror xarray/pandas exactly: contiguous, time-sorted, and INCLUDING
    empty interior bins as zero-width ranges, so the output time axis aligns with
    the dask path even when the series has gaps. Group ``g`` spans array positions
    ``[bounds[g], bounds[g + 1])`` — this contiguity requires a monotonic time
    index, which xarray's resample also enforces.

    Works for both a standard ``DatetimeIndex`` and a ``CFTimeIndex`` (non-standard
    CF calendars — ``noleap``/``360_day`` etc. from CMIP6). pandas ``.resample``
    rejects a ``CFTimeIndex``, so for cftime we group with *xarray's* resample —
    the same grouping the dask path uses — over a position array, which keeps the
    numba bounds aligned with the dask reduce by construction.
    """
    if not tindex.is_monotonic_increasing:
        raise ValueError(
            "numba engine requires a monotonic-increasing time index "
            "(xarray's resample path enforces the same)."
        )
    if isinstance(tindex, xr.CFTimeIndex):
        # cftime: xarray resample of a position array yields one count per output
        # bin, in bin order — the same bins the dask path reduces over. Unlike pandas
        # (which fills empty bins with 0), xarray's cftime `.count()` fills an empty
        # interior bin with NaN, so zero-fill before the int64 cumsum keeps the bin as
        # a zero-width group (-> NaN output), matching the dask reduce's reindex.
        pos = xr.DataArray(np.arange(len(tindex)), coords={"time": tindex}, dims="time")
        counted = pos.resample(time=freq).count()
        counts = np.nan_to_num(counted.values, nan=0.0)
        bounds = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        return bounds, counted.get_index("time")
    # `.resample(freq).count()` uses the same pandas offset/label convention as
    # xarray and yields a count (0 for empty bins) per output bin, in bin order.
    counts = pd.Series(1, index=tindex).resample(freq).count()
    bounds = np.concatenate([[0], np.cumsum(counts.values)]).astype(np.int64)
    return bounds, pd.DatetimeIndex(counts.index)


# --------------------------------------------------------------------------- #
# kernels
# --------------------------------------------------------------------------- #
@njit(nogil=True, parallel=True, fastmath=False)
def _block_stat(cube, bounds, code, out):
    G = bounds.shape[0] - 1
    NY = cube.shape[1]; NX = cube.shape[2]
    for iy in prange(NY):
        for ix in range(NX):
            for g in range(G):
                lo = bounds[g]; hi = bounds[g + 1]
                n = 0; s = 0.0; mn = np.inf; mx = -np.inf; hasnan = False
                for k in range(lo, hi):
                    v = cube[k, iy, ix]
                    if np.isnan(v):
                        hasnan = True
                    else:
                        s += v; n += 1
                        if v < mn: mn = v
                        if v > mx: mx = v
                if hi == lo:
                    # empty bin: xarray fills empty resample bins with NaN for
                    # every reducer (it's a reindex, not a call to the reducer).
                    out[g, iy, ix] = np.nan
                elif code == 4:
                    # nanmean skips NaNs; all-NaN window -> NaN
                    out[g, iy, ix] = (s / n) if n > 0 else np.nan
                elif hasnan:
                    # mean/sum/min/max propagate NaNs (like numpy)
                    out[g, iy, ix] = np.nan
                elif code == 0:
                    out[g, iy, ix] = s / n
                elif code == 1:
                    out[g, iy, ix] = s
                elif code == 2:
                    out[g, iy, ix] = mn
                else:
                    out[g, iy, ix] = mx


@njit(nogil=True, parallel=True, fastmath=False)
def _block_dd(cube, bounds, ddargs, out):
    G = bounds.shape[0] - 1
    NY = cube.shape[1]; NX = cube.shape[2]; D = ddargs.shape[0]
    for iy in prange(NY):
        for ix in range(NX):
            for g in range(G):
                lo = bounds[g]; hi = bounds[g + 1]
                for d in range(D):
                    t0 = ddargs[d, 0]; t1 = ddargs[d, 1]
                    base = t0 if ddargs[d, 2] == 0 else t1
                    acc = 0.0; hasnan = False
                    for k in range(lo, hi):
                        v = cube[k, iy, ix]
                        if np.isnan(v):
                            hasnan = True
                        elif v > t0 and v < t1:
                            av = v - base
                            if av < 0.0: av = -av
                            acc += av
                    # empty bin (hi==lo) -> NaN to match xarray's reindex fill
                    out[g, iy, ix, d] = np.nan if (hasnan or hi == lo) else acc


@njit(nogil=True, parallel=True, fastmath=False)
def _block_bins(cube, bounds, ddargs, out):
    G = bounds.shape[0] - 1
    NY = cube.shape[1]; NX = cube.shape[2]; D = ddargs.shape[0]
    for iy in prange(NY):
        for ix in range(NX):
            for g in range(G):
                lo = bounds[g]; hi = bounds[g + 1]
                for d in range(D):
                    t0 = ddargs[d, 0]; t1 = ddargs[d, 1]
                    c = 0.0
                    for k in range(lo, hi):
                        v = cube[k, iy, ix]
                        if v > t0 and v < t1:
                            c += 1.0
                    # empty bin (hi==lo) -> NaN to match xarray's reindex fill;
                    # NaN *values* in a non-empty bin count as out-of-range.
                    out[g, iy, ix, d] = np.nan if hi == lo else c


@njit(nogil=True, parallel=True, fastmath=False)
def _block_sine_dd(cube, bounds, ddargs, out):
    G = bounds.shape[0] - 1
    NY = cube.shape[1]; NX = cube.shape[2]; D = ddargs.shape[0]
    PI = np.pi
    for iy in prange(NY):
        for ix in range(NX):
            for g in range(G):
                lo = bounds[g]; hi = bounds[g + 1]
                n = 0; s = 0.0; tmax = -np.inf; tmin = np.inf; hasnan = False
                for k in range(lo, hi):
                    v = cube[k, iy, ix]
                    if np.isnan(v):
                        hasnan = True
                    else:
                        s += v; n += 1
                        if v > tmax: tmax = v
                        if v < tmin: tmin = v
                for d in range(D):
                    if hasnan or n == 0:
                        out[g, iy, ix, d] = np.nan
                        continue
                    tavg = s / n
                    kind = ddargs[d, 2]
                    val = 0.0
                    for j in range(2):
                        thr = ddargs[d, j]
                        if kind == 0:  # cooling degree days
                            if thr <= tmin:
                                part = tavg - thr
                            elif thr < tmax and tmin < thr:
                                rng = tmax - tmin
                                a = np.arccos((2.0 * thr - tmax - tmin) / rng)
                                part = ((tavg - thr) * a + rng * np.sin(a) / 2.0) / PI
                            else:
                                part = 0.0
                            val += part if j == 0 else -part
                        else:          # heating degree days
                            if thr >= tmax:
                                part = thr - tavg
                            elif thr < tmax and tmin < thr:
                                alpha = (tmax - tmin) / 2.0
                                r = (thr - tavg) / alpha
                                at = np.arctan(r / np.sqrt(1.0 - r * r))
                                part = (1.0 / PI) * ((thr - tavg) * (at + PI / 2.0)
                                                     + alpha * np.cos(at))
                            else:
                                part = 0.0
                            val += -part if j == 0 else part
                    out[g, iy, ix, d] = val


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def _stat_wrap(blk, bounds=None, code=None):
    # Preserve the input dtype (accumulation is float64 inside the kernel, the
    # stored result matches the dask path, e.g. float32 stays float32).
    out = np.empty((bounds.shape[0] - 1, blk.shape[1], blk.shape[2]), blk.dtype)
    _block_stat(np.ascontiguousarray(blk), bounds, code, out)
    return out


def _dd_wrap(blk, bounds=None, dda=None, fn=None):
    out = np.empty((bounds.shape[0] - 1, blk.shape[1], blk.shape[2], dda.shape[0]), blk.dtype)
    fn(np.ascontiguousarray(blk), bounds, dda, out)
    return out


def numba_resample(da: xr.DataArray, freq: str, calc: str, ddargs=None, multi_dd=False):
    """Grouped temporal reduction matching ``da.resample(time=freq).reduce(func)``.

    Parameters mirror ``TemporalAggregator``: ``freq`` is the already-translated
    pandas offset (e.g. ``"1D"``, ``"ME"``), ``calc`` one of ``NUMBA_CALCS``.
    """
    spatial = [d for d in da.dims if d != "time"]
    if len(spatial) != 2:
        raise ValueError(f"numba engine expects 2 spatial dims, got {spatial}")
    da_t = da.transpose("time", *spatial).chunk({"time": -1})
    tindex = da_t.get_index("time")
    bounds, out_time = resample_groups(tindex, freq)
    G = len(out_time)
    ych, xch = da_t.chunks[1], da_t.chunks[2]
    coords = {"time": out_time, spatial[0]: da_t[spatial[0]], spatial[1]: da_t[spatial[1]]}

    if calc in _STAT_CODE:
        code = _STAT_CODE[calc]
        out = darr.map_blocks(
            _stat_wrap, da_t.data, bounds=bounds, code=code,
            dtype=da_t.dtype, chunks=((G,), ych, xch),
        )
        return xr.DataArray(out, dims=["time", *spatial], coords=coords)

    dda = np.atleast_2d(np.asarray(ddargs, dtype=np.float64))
    D = dda.shape[0]
    fn = {"dd": _block_dd, "bins": _block_bins, "sine_dd": _block_sine_dd}[calc]
    out = darr.map_blocks(
        _dd_wrap, da_t.data, bounds=bounds, dda=dda, fn=fn,
        dtype=da_t.dtype, chunks=((G,), ych, xch, (D,)), new_axis=[3],
    )
    arr = xr.DataArray(out, dims=["time", *spatial, "dd"], coords=coords)
    if not multi_dd:
        arr = arr.isel(dd=0, drop=True)
    return arr