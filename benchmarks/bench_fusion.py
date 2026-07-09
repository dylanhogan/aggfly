"""
Fused vs per-step numba on the canonical poly SPEC (mean/date -> power[1..4] ->
sum/month) at NATIVE (small) ERA5 chunks — the regime where engine="auto" picks
numba and where fusion pays off (fewer map_blocks tasks, no intermediate daily/
power arrays materialized).

Both configs use engine="numba"; the "per-step" baseline monkeypatches
`fusible_poly_chain` to return None so the chain runs as three separate numba
map_blocks passes (the pre-fusion behavior). Fusion collapses those to one pass.

Run:
    <venv>/bin/python benchmarks/bench_fusion.py --year 2016
"""
import argparse, time
import numpy as np
import dask
from dask.diagnostics import ResourceProfiler

import aggfly as af
import aggfly.aggregate.aggregate as agg

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


def load(year):
    return af.dataset_from_path(ERA5.format(year=year), var="t2m", name="era5",
                                preprocess=lambda x: x - 273.15, chunks={})


def timed(label, fn):
    fn()  # warm caches / JIT compile (not timed)
    t0 = time.perf_counter()
    with ResourceProfiler(dt=0.5) as rp:
        result = fn()
    dt = time.perf_counter() - t0
    cpu = np.array([r.cpu for r in rp.results]) if rp.results else np.array([0])
    mem = np.array([r.mem for r in rp.results]) if rp.results else np.array([0])
    print(f"{label:30s} {dt:7.2f}s  CPU mean={cpu.mean():5.0f}%  peakRAM={human(mem.max()*1e6)}")
    return dt, result


def temporal(ds, fuse):
    orig = agg.fusible_poly_chain
    if not fuse:
        agg.fusible_poly_chain = lambda steps: None   # force per-step path
    try:
        out = af.aggregate_time(dataset=ds, weights=None, engine="numba", **SPEC)
        return dask.compute(*[d.da for d in out.values()])
    finally:
        agg.fusible_poly_chain = orig


def maxrel(r_ps, r_fu):
    """Max relative diff between two result tuples (NaN-aware)."""
    worst = 0.0
    for a, b in zip(r_ps, r_fu):
        a, b = a.values, b.values
        m = ~(np.isnan(a) | np.isnan(b))
        worst = max(worst, float(np.max(np.abs(a[m] - b[m]) / (np.abs(b[m]) + 1e-30))))
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2016)
    args = ap.parse_args()
    dask.config.set(scheduler="threads")

    # End-to-end (includes reading the zarr) at native chunks.
    print("=== poly SPEC (mean/date -> power[1..4] -> sum/month), native chunks, end-to-end ===")
    ps, r_ps = timed("numba per-step (3 passes)", lambda: temporal(load(args.year), fuse=False))
    fu, r_fu = timed("numba fused    (1 pass)", lambda: temporal(load(args.year), fuse=True))
    print(f"  --> fused speedup: {ps/fu:.2f}x   |   max rel diff fused-vs-per-step: {maxrel(r_ps, r_fu):.1e}\n")

    # Compute-only: pre-load the cube into RAM so the timed section excludes disk I/O
    # (isolates whether fusion has any *compute* advantage from fewer passes/tasks).
    print("=== compute-only (cube persisted in RAM) ===")
    ds = load(args.year)
    ds.da = ds.da.persist()
    dask.compute(ds.da.sum())  # force the persist to finish before timing
    psc, _ = timed("numba per-step (3 passes)", lambda: temporal(ds, fuse=False))
    fuc, _ = timed("numba fused    (1 pass)", lambda: temporal(ds, fuse=True))
    print(f"  --> fused speedup (compute-only): {psc/fuc:.2f}x")


if __name__ == "__main__":
    main()
