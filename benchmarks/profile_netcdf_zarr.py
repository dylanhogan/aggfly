"""
Measure the "make NetCDF read like Zarr" ceiling for a temporal reduction on the
raw ERA5 bricks. Extends profile_netcdf.py (configs A/C) with two conversion paths:

  Z1) convert the window once to a Zarr store (time-contiguous chunks), then read
      -> the "convert once, process many times" ceiling. One-time write cost reported.
  K1) kerchunk reference over the ORIGINAL NetCDF (no data copy), native brick chunks
      -> isolates the HDF5-lock effect: same bricks as A, but decompressed by numcodecs
         in-process (no HDF5 global lock), so threads can run in parallel.
  K2) kerchunk reference + rechunk time-contiguous
      -> kerchunk (no copy) combined with the C-style layout.

Also re-runs A (native brick + threads) and C (rechunk + threads) IN THIS PROCESS so
all numbers share the same page-cache warmth. Low mean CPU on a long wall = lock/IO
bound; CPU climbing toward n*100% = threads actually parallelizing.

Caveat: no way to drop the page cache without root, so absolute MB/s is warm-cache-
optimistic and equal across configs; read the CPU% and the A-vs-K1 gap, not raw MB/s.

Run: <venv>/bin/python benchmarks/profile_netcdf_zarr.py --year 2016
"""
import argparse, os, time, shutil
import numpy as np
import xarray as xr
import dask
from dask.diagnostics import ResourceProfiler

NC = "/shared/vol1/ERA5/raw/ERA5_{year}.nc"
SCRATCH = "/tmp/claude-1000/-home-dhogan-repositories-aggfly/b13fde27-dc41-4b9c-bcdd-c2abf192a0b1/scratchpad"


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"


def dirsize(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total


def timed(label, fn, nbytes, dt=0.25):
    t0 = time.perf_counter()
    with ResourceProfiler(dt=dt) as rp:
        fn()
    el = time.perf_counter() - t0
    cpu = np.array([r.cpu for r in rp.results]) if rp.results else np.array([0.0])
    mem = np.array([r.mem for r in rp.results]) if rp.results else np.array([0.0])
    thru = nbytes / el if el else 0
    print(f"{label:46s} {el:7.2f}s  CPU mean={cpu.mean():5.0f}% max={cpu.max():5.0f}%  "
          f"peakRAM={human(mem.max()*1e6)}  ~{human(thru)}/s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2016)
    ap.add_argument("--var", default="t2m")
    ap.add_argument("--nlat", type=int, default=120)
    ap.add_argument("--nlon", type=int, default=120)
    ap.add_argument("--lat0", type=int, default=200)
    ap.add_argument("--lon0", type=int, default=400)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    path = NC.format(year=args.year)
    la, lo = slice(args.lat0, args.lat0 + args.nlat), slice(args.lon0, args.lon0 + args.nlon)

    def load_nc(chunks):
        return xr.open_dataset(path, chunks=chunks)[args.var].isel(latitude=la, longitude=lo)

    da0 = load_nc({})
    ntime = da0.sizes["valid_time"]
    nbytes = int(np.prod(list(da0.sizes.values()))) * 4
    print(f"file={path}")
    print(f"var={args.var}  window={args.nlat}x{args.nlon} x {ntime}t = {human(nbytes)} decompressed"
          f"  on-disk chunks={da0.encoding.get('chunksizes')}\n")

    def daily_mean(da):
        return da.resample(valid_time="1D").mean().compute()

    # ---- baselines in-process (share page-cache warmth) --------------------
    with dask.config.set(scheduler="threads", num_workers=args.workers):
        timed("A) netcdf native brick, threads",
              lambda: daily_mean(load_nc({})), nbytes)
        timed("C) netcdf rechunk time=-1, threads",
              lambda: daily_mean(load_nc({}).chunk({"valid_time": -1, "latitude": 60, "longitude": 60})),
              nbytes)

    # ---- Z1) convert window -> zarr once, then read ------------------------
    zstore = os.path.join(SCRATCH, f"era5_{args.year}_{args.var}_win.zarr")
    if os.path.exists(zstore):
        shutil.rmtree(zstore)
    t0 = time.perf_counter()
    (load_nc({}).chunk({"valid_time": -1, "latitude": 60, "longitude": 60})
        .to_dataset(name=args.var).to_zarr(zstore, mode="w"))
    conv = time.perf_counter() - t0
    zsz = dirsize(zstore)
    print(f"\n[Z1 one-time] wrote {zstore.split('/')[-1]}  {human(zsz)} on disk  "
          f"in {conv:.1f}s  (ratio to decompressed {zsz/nbytes:.2f}x)")
    with dask.config.set(scheduler="threads", num_workers=args.workers):
        timed("Z1) zarr time-contiguous, threads",
              lambda: daily_mean(xr.open_zarr(zstore)[args.var]), nbytes)

    # ---- kerchunk reference over the original netcdf (no copy) -------------
    try:
        from kerchunk.hdf import SingleHdf5ToZarr
        import fsspec
    except Exception as e:
        print(f"\n[kerchunk unavailable: {e}]")
        return

    t0 = time.perf_counter()
    refs = SingleHdf5ToZarr(path, inline_threshold=0).translate()
    kt = time.perf_counter() - t0
    nrefs = len(refs.get("refs", refs))
    print(f"\n[kerchunk one-time] translated {path.split('/')[-1]} -> {nrefs} chunk refs "
          f"in {kt:.1f}s  (metadata scan of the whole file, no data copy)")

    def load_kerchunk():
        fs = fsspec.filesystem("reference", fo=refs, remote_protocol="file")
        ds = xr.open_dataset(fs.get_mapper(""), engine="zarr",
                             backend_kwargs={"consolidated": False}, chunks={})
        da = ds[args.var].isel(latitude=la, longitude=lo)
        # coords loaded via the reference fs carry an un-copyable _json.Scanner in
        # .encoding; xarray deepcopies encoding during resample -> strip it.
        da.encoding = {}
        for c in da.coords:
            da[c].encoding = {}
        return da

    with dask.config.set(scheduler="threads", num_workers=args.workers):
        timed("K1) kerchunk native brick, threads",
              lambda: daily_mean(load_kerchunk()), nbytes)
        timed("K2) kerchunk rechunk time=-1, threads",
              lambda: daily_mean(load_kerchunk().chunk({"valid_time": -1, "latitude": 60, "longitude": 60})),
              nbytes)

    shutil.rmtree(zstore, ignore_errors=True)


if __name__ == "__main__":
    main()
