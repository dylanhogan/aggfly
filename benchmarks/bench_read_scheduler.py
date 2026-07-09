"""
Process-based vs threaded scheduler for the READ-bound temporal path.

The warm zarr read is GIL-serialized to ~2 cores under dask's threaded scheduler
(blosc releases the GIL but the zarr/xarray/dask per-chunk orchestration holds it);
separate worker processes each get their own GIL and read in parallel. Since the
read is ~80% of end-to-end for the poly SPEC, this should translate to a large
end-to-end win on warm/cached data.

This drives the poly SPEC (mean/date -> power[1..4] -> sum/month) end-to-end
(read + numba temporal reduction) under:
  - THREADS   : dask threaded scheduler (aggfly's current default)
  - PROCESSES : LocalCluster(processes=True), numba capped to 1 thread/worker so
                n_workers processes give the parallelism (no thread oversubscription)

Both are warmed (page cache + numba JIT) before timing; results are checked equal.

Run:
    <venv>/bin/python benchmarks/bench_read_scheduler.py --year 2016 --workers 16
"""
import argparse, time, threading
import numpy as np
import dask
import psutil
from dask.distributed import Client, LocalCluster

import aggfly as af

ERA5 = "/shared/vol1/ERA5/autoproc/ERA5_{year}.zarr"
SPEC = {"tavg": [
    ("aggregate", {"calc": "mean", "groupby": "date"}),
    ("transform", {"transform": "power", "exp": np.arange(1, 5)}),
    ("aggregate", {"calc": "sum", "groupby": "month"}),
]}


def load(year):
    return af.dataset_from_path(ERA5.format(year=year), var="t2m", name="era5",
                                preprocess=lambda x: x - 273.15, chunks={})


def temporal(ds):
    out = af.aggregate_time(dataset=ds, weights=None, engine="numba", **SPEC)
    return dask.compute(*[d.da for d in out.values()])


class Monitor:
    """Sample system-wide CPU% and process-tree RSS while the block runs."""
    def __init__(self):
        self.cpu = []; self.rss = []; self._stop = threading.Event()
        self.proc = psutil.Process()

    def _tree_rss(self):
        procs = [self.proc] + self.proc.children(recursive=True)
        tot = 0
        for p in procs:
            try:
                tot += p.memory_info().rss
            except psutil.Error:
                pass
        return tot

    def _run(self):
        psutil.cpu_percent(None)
        while not self._stop.wait(0.25):
            self.cpu.append(psutil.cpu_percent(None))
            self.rss.append(self._tree_rss())

    def __enter__(self):
        self.t = threading.Thread(target=self._run, daemon=True); self.t.start(); return self

    def __exit__(self, *a):
        self._stop.set(); self.t.join()


def timed(label, fn):
    fn()  # warm page cache + numba JIT (in every worker) — not timed
    with Monitor() as m:
        t0 = time.perf_counter(); r = fn(); dt = time.perf_counter() - t0
    ncpu = psutil.cpu_count()
    cpu = np.mean(m.cpu) if m.cpu else 0
    ram = (max(m.rss) / 1e9) if m.rss else 0
    print(f"  {label:30s} {dt:6.2f}s  sysCPU~{cpu:4.0f}% ({cpu/100:4.1f}/{ncpu})  peakRSS~{ram:4.1f}GB", flush=True)
    return dt, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2016)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    print(f"poly SPEC, ERA5 {args.year}, native chunks (read + numba reduce), warm:\n")

    with dask.config.set(scheduler="threads"):
        tt, rt = timed("THREADS (default)", lambda: temporal(load(args.year)))

    cl = LocalCluster(n_workers=args.workers, threads_per_worker=1,
                      processes=True, dashboard_address=None)
    c = Client(cl)
    try:
        import numba
        c.run(numba.set_num_threads, 1)   # 1 numba thread/worker -> no oversubscription
        tp, rp = timed(f"PROCESSES ({args.workers}w x1)", lambda: temporal(load(args.year)))
    finally:
        c.close(); cl.close()

    ok = all(np.allclose(a.values, b.values, rtol=1e-5, atol=1e-4, equal_nan=True)
             for a, b in zip(rt, rp))
    print(f"\n  --> process speedup: {tt/tp:.2f}x   |   results match: {ok}")


if __name__ == "__main__":
    main()
