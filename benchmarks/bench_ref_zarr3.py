"""
Re-benchmark the NetCDF read-path options on the MODERN stack (zarr 3), cold + warm:
  A  = raw NetCDF native bricks (threads)
  K1 = kerchunk reference, native bricks
  K2 = kerchunk reference + rechunk time-contiguous
  Z1 = to_zarr (rechunk time-contiguous) convert-once, then read

Each config runs on its OWN untouched ERA5 year so rep0 is a genuine cold read (no
cross-config page-cache pollution; can't drop caches without root). Window = 120x120 x
full year (~482 MB decompressed), matching the earlier old-stack table.
"""
import argparse, os, time, shutil
import numpy as np, xarray as xr, dask, fsspec
from dask.diagnostics import ResourceProfiler
from kerchunk.hdf import SingleHdf5ToZarr

NC = "/shared/vol1/ERA5/raw/ERA5_{year}.nc"
SCRATCH = "/tmp/claude-1000/-home-dhogan-repositories-aggfly/b13fde27-dc41-4b9c-bcdd-c2abf192a0b1/scratchpad"
LA, LO = slice(200, 320), slice(400, 520)


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"


def daily_mean(da):
    return da.resample(valid_time="1D").mean().compute()


def timed(fn):
    t0 = time.perf_counter()
    with ResourceProfiler(dt=0.25) as rp:
        fn()
    dt = time.perf_counter() - t0
    cpu = np.array([r.cpu for r in rp.results]) if rp.results else np.array([0.0])
    return dt, cpu.mean()


def load_A(year):
    return xr.open_dataset(NC.format(year=year), chunks={})["t2m"].isel(latitude=LA, longitude=LO)


def load_K(year, rechunk):
    refs = SingleHdf5ToZarr(NC.format(year=year), inline_threshold=0).translate()
    fs = fsspec.filesystem("reference", fo=refs, remote_protocol="file")
    ds = xr.open_dataset(fs.get_mapper(""), engine="zarr", backend_kwargs={"consolidated": False}, chunks={})
    da = ds["t2m"].isel(latitude=LA, longitude=LO)
    da.encoding = {}
    for c in da.coords:
        da[c].encoding = {}
    if rechunk:
        da = da.chunk({"valid_time": -1, "latitude": 60, "longitude": 60})
    return da


def run_config(cfg, year, nbytes):
    with dask.config.set(scheduler="threads", num_workers=8):
        if cfg == "A":
            reads = [lambda: daily_mean(load_A(year))] * 3
        elif cfg == "K1":
            reads = [lambda: daily_mean(load_K(year, False))] * 3
        elif cfg == "K2":
            reads = [lambda: daily_mean(load_K(year, True))] * 3
        elif cfg == "Z1":
            store = os.path.join(SCRATCH, f"z1_{year}.zarr")
            shutil.rmtree(store, ignore_errors=True)
            t0 = time.perf_counter()
            (load_A(year).chunk({"valid_time": -1, "latitude": 60, "longitude": 60})
                .to_dataset(name="t2m").to_zarr(store, mode="w"))
            conv = time.perf_counter() - t0
            sz = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(store) for f in fs)
            print(f"  [Z1 one-time convert: {conv:.1f}s, store {human(sz)} ({sz/nbytes:.2f}x)]")
            reads = [lambda: daily_mean(xr.open_zarr(store)["t2m"])] * 3

        cold, cold_cpu = timed(reads[0])
        warm = min(timed(reads[i])[0] for i in (1, 2))
    print(f"{cfg:3s} y{year}: COLD {cold:6.2f}s ({cold_cpu:3.0f}% CPU, {human(nbytes/cold)}/s)   "
          f"WARM {warm:6.2f}s ({human(nbytes/warm)}/s)")
    if cfg == "Z1":
        shutil.rmtree(os.path.join(SCRATCH, f"z1_{year}.zarr"), ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="A:2010,K1:2011,K2:2012,Z1:2013")
    args = ap.parse_args()
    da0 = load_A(int(args.pairs.split(",")[0].split(":")[1]))
    ntime = da0.sizes["valid_time"]
    nbytes = 120 * 120 * ntime * 4
    print(f"window 120x120 x {ntime}t = {human(nbytes)} decompressed (each config on its own untouched year)\n")
    for pair in args.pairs.split(","):
        cfg, year = pair.split(":")
        run_config(cfg, int(year), nbytes)


if __name__ == "__main__":
    main()
