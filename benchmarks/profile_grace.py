"""
Instrumented version of notebooks/aggregate_pre_temp.py.

Profiles the real GRACE + ERA5 workload one stage at a time to locate the
bottleneck and to A/B-test the 250x250 rechunk on line 50 of the original
against the native on-disk Zarr chunks.

Run:
    <venv>/bin/python benchmarks/profile_grace.py --year 2016
"""
import argparse
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
import xarray as xr
import dask
from dask.diagnostics import ResourceProfiler

import aggfly as af

PROJECT_DIR = "/home/dhogan/data/grace"
SHP = f"{PROJECT_DIR}/grace_grid/grace_gsfc_cells.shp"
ERA5 = "/shared/vol1/ERA5/autoproc/ERA5_{year}.zarr"

SPEC = [
    ("aggregate", {"calc": "mean", "groupby": "date"}),
    ("transform", {"transform": "power", "exp": np.arange(1, 5)}),
    ("aggregate", {"calc": "sum", "groupby": "month"}),
]


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"


@contextmanager
def resources(label):
    print(f"\n=== {label} ===")
    t0 = time.perf_counter()
    with ResourceProfiler(dt=0.5) as rp:
        yield
    dt = time.perf_counter() - t0
    print(f"--> {label}: {dt:.1f}s wall")
    if rp.results:
        cpu = np.array([r.cpu for r in rp.results])
        mem = np.array([r.mem for r in rp.results])
        print(f"    CPU: mean={cpu.mean():.0f}%  max={cpu.max():.0f}%  "
              f"(32 cores => 3200% = full)")
        print(f"    RSS: mean={human(mem.mean()*1e6)}  max={human(mem.max()*1e6)}")
        verdict = ("COMPUTE-bound (kernel is the cost -> Numba/Rust helps)"
                   if cpu.mean() > 60 * 8 else
                   "NOT compute-bound (rechunk/IO/merge dominates -> language won't help)")
        print(f"    >>> {verdict}")
    return dt


def chunk_report(da, tag):
    enc = da.encoding.get("chunks") or da.encoding.get("preferred_chunks")
    first = tuple(c[0] for c in da.chunks) if da.chunks else None
    nblocks = int(np.prod([len(c) for c in da.chunks])) if da.chunks else 1
    blk = int(np.prod(first)) * da.dtype.itemsize if first else da.nbytes
    print(f"[{tag}] shape={da.shape} dtype={da.dtype} total={human(da.nbytes)}")
    print(f"       on-disk chunks={enc}  dask first-block={first}  "
          f"n_blocks={nblocks}  chunk~{human(blk)}")


def load(year, rechunk):
    ds = af.dataset_from_path(
        ERA5.format(year=year), var="t2m", name="era5",
        preprocess=lambda x: x - 273.15, chunks={},
    )
    chunk_report(ds.da, f"year {year} native (chunks=%s)" % "{}")
    if rechunk:
        ds.da = ds.da.chunk({"time": -1, "latitude": 250, "longitude": 250})
        chunk_report(ds.da, f"year {year} after .chunk(250x250)")
    return ds


def temporal_only(ds, weights):
    out = af.aggregate_time(dataset=ds, weights=weights, tavg=SPEC)
    dask.compute(*[d.da for d in out.values()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2016)
    ap.add_argument("--scheduler", default="threads")
    args = ap.parse_args()
    dask.config.set(scheduler=args.scheduler)
    print(f"scheduler={args.scheduler}")

    # ---- weights (cached) -------------------------------------------------
    with resources("build weights"):
        gr = af.georegions_from_path(SHP, regionid="labels")
        ds0 = af.dataset_from_path(ERA5.format(year=args.year), var="t2m",
                                   name="era5", georegions=gr, chunks={})
        weights = af.weights_from_objects(ds0, gr, project_dir=PROJECT_DIR)
        weights.calculate_weights()

    # ---- A: temporal on NATIVE chunks ------------------------------------
    ds_native = load(args.year, rechunk=False)
    ds_native.georegions = gr
    with resources("TEMPORAL only — native 50x50 chunks"):
        temporal_only(ds_native, weights)

    # ---- B: temporal on 250x250 rechunk (as in the script) ---------------
    ds_re = load(args.year, rechunk=True)
    ds_re.georegions = gr
    with resources("TEMPORAL only — 250x250 rechunk (original script)"):
        temporal_only(ds_re, weights)

    # ---- C: full pipeline (temporal + spatial merges) --------------------
    ds_full = load(args.year, rechunk=True)
    ds_full.georegions = gr
    with resources("FULL aggregate_dataset (temporal + spatial)"):
        df = af.aggregate_dataset(dataset=ds_full, weights=weights, tavg=SPEC)
    print(f"\nresult: {df.shape[0]} rows x {df.shape[1]} cols; "
          f"columns={list(df.columns)}")


if __name__ == "__main__":
    main()