"""
Hard before/after benchmark: dask reduce path vs the numba nogil engine on the
real GRACE + ERA5 temporal workload, at NATIVE (small) chunks — the low-memory
regime where the GIL-free kernel is expected to win.

Run:
    <venv>/bin/python benchmarks/bench_engine.py --year 2016
"""
import argparse, time
import numpy as np
import dask
from dask.diagnostics import ResourceProfiler

import aggfly as af

PROJECT_DIR = "/home/dhogan/data/grace"
SHP = f"{PROJECT_DIR}/grace_grid/grace_gsfc_cells.shp"
ERA5 = "/shared/vol1/ERA5/autoproc/ERA5_{year}.zarr"
SPEC = {"tavg": [
    ("aggregate", {"calc": "mean", "groupby": "date"}),
    ("transform", {"transform": "power", "exp": np.arange(1, 5)}),
    ("aggregate", {"calc": "sum", "groupby": "month"}),
]}


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"


def load(year, rechunk):
    ds = af.dataset_from_path(ERA5.format(year=year), var="t2m", name="era5",
                             preprocess=lambda x: x - 273.15, chunks={})
    if rechunk:
        ds.da = ds.da.chunk({"time": -1, "latitude": 250, "longitude": 250})
    return ds


def timed(label, fn):
    fn()  # warm caches / JIT compile (not timed)
    t0 = time.perf_counter()
    with ResourceProfiler(dt=0.5) as rp:
        fn()
    dt = time.perf_counter() - t0
    cpu = np.array([r.cpu for r in rp.results]) if rp.results else np.array([0])
    mem = np.array([r.mem for r in rp.results]) if rp.results else np.array([0])
    print(f"{label:34s} {dt:7.2f}s  CPU mean={cpu.mean():5.0f}% max={cpu.max():5.0f}%  "
          f"peakRAM={human(mem.max()*1e6)}")
    return dt


def temporal(ds, engine):
    out = af.aggregate_time(dataset=ds, weights=None, engine=engine, **SPEC)
    dask.compute(*[d.da for d in out.values()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2016)
    args = ap.parse_args()
    dask.config.set(scheduler="threads")

    print("=== TEMPORAL, native small chunks (low-memory regime) ===")
    dsd = load(args.year, rechunk=False); dnat = timed("dask   (native chunks)", lambda: temporal(dsd, "dask"))
    dsn = load(args.year, rechunk=False); nnat = timed("numba  (native chunks)", lambda: temporal(dsn, "numba"))
    print(f"  --> numba speedup (native): {dnat/nnat:.1f}x\n")

    print("=== TEMPORAL, 250x250 chunks (the original script's rechunk) ===")
    dsd2 = load(args.year, rechunk=True); dre = timed("dask   (250x250)", lambda: temporal(dsd2, "dask"))
    dsn2 = load(args.year, rechunk=True); nre = timed("numba  (250x250)", lambda: temporal(dsn2, "numba"))
    print(f"  --> numba speedup (250x250): {dre/nre:.1f}x")


if __name__ == "__main__":
    main()
