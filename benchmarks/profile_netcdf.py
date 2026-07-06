"""
Profile reading the raw ERA5 NetCDFs (brick-chunked, zlib+shuffle) for a temporal
reduction, to see whether the cost is the HDF5 read lock, decompression, or disk.

Bounded to a spatial window (full time) so it runs in seconds-to-minutes but still
forces the full time-reduction read pattern. Compares:
  A) native brick chunks, threaded scheduler
  B) native brick chunks, processes scheduler (own HDF5 lock per worker)
  C) time-contiguous rechunk, threaded
and reports CPU utilization (low mean CPU during a long wall = lock/IO bound).

Run: <venv>/bin/python benchmarks/profile_netcdf.py --year 2016
"""
import argparse, time
import numpy as np
import xarray as xr
import dask
from dask.diagnostics import ResourceProfiler

NC = "/shared/vol1/ERA5/raw/ERA5_{year}.nc"


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"


def timed(label, fn, dt=0.25):
    t0 = time.perf_counter()
    with ResourceProfiler(dt=dt) as rp:
        nbytes = fn()
    dt_ = time.perf_counter() - t0
    cpu = np.array([r.cpu for r in rp.results]) if rp.results else np.array([0.0])
    mem = np.array([r.mem for r in rp.results]) if rp.results else np.array([0.0])
    thru = nbytes / dt_ if dt_ else 0
    print(f"{label:42s} {dt_:7.2f}s  CPU mean={cpu.mean():5.0f}% max={cpu.max():5.0f}%  "
          f"peakRAM={human(mem.max()*1e6)}  ~{human(thru)}/s decompressed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2016)
    ap.add_argument("--var", default="t2m")
    ap.add_argument("--nlat", type=int, default=120, help="lat window size (cells)")
    ap.add_argument("--nlon", type=int, default=120, help="lon window size (cells)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    path = NC.format(year=args.year)

    def load(chunks):
        ds = xr.open_dataset(path, chunks=chunks)
        da = ds[args.var].isel(
            latitude=slice(200, 200 + args.nlat),
            longitude=slice(400, 400 + args.nlon),
        )
        return da

    da0 = load({})
    print(f"file={path}")
    print(f"var={args.var} full shape={dict(load({}).sizes)}  on-disk chunks="
          f"{da0.encoding.get('chunksizes')}  compression=zlib{da0.encoding.get('complevel')}")
    nbytes = int(np.prod(list(load({}).sizes.values()))) * 4  # float32 window, full time
    ntime = da0.sizes["valid_time"]
    print(f"window: {args.nlat}x{args.nlon} cells x {ntime} timesteps = "
          f"{human(nbytes)} decompressed\n")

    # daily mean forces reading the full time axis for the window
    def daily_mean(da):
        return da.resample(valid_time="1D").mean().compute()

    # A) native brick chunks, threads
    with dask.config.set(scheduler="threads", num_workers=args.workers):
        timed("A) native brick chunks, threads",
              lambda: (daily_mean(load({})), nbytes)[1])

    # B) native brick chunks, processes (own HDF5 lock per worker)
    with dask.config.set(scheduler="processes", num_workers=args.workers):
        timed("B) native brick chunks, processes",
              lambda: (daily_mean(load({})), nbytes)[1])

    # C) time-contiguous rechunk (ideal for a time reduction), threads
    with dask.config.set(scheduler="threads", num_workers=args.workers):
        timed("C) rechunk time=-1 (time-contiguous), threads",
              lambda: (daily_mean(load({}).chunk({"valid_time": -1,
                        "latitude": 60, "longitude": 60})), nbytes)[1])


if __name__ == "__main__":
    main()
