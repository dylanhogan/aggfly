"""
Profile aggfly's temporal (and optionally spatial) aggregation to separate
*compute* cost from *chunking / IO / shuffle* cost.

Motivating question: is the temporal transform slow because the numeric kernels
are expensive (→ a compiled/Numba backend helps), or because the dask arrays are
chunked badly for a resample-over-time and dask spends its time rechunking
(→ no backend language helps; fix the chunking / Zarr encoding instead)?

The harness answers that by:
  1. Reporting on-disk (Zarr encoding) chunks vs. the dask chunks aggfly loads.
  2. Checking how many chunks each resample group spans (alignment diagnostic).
  3. Timing the temporal transform under dask.diagnostics, which records per-task
     time AND CPU/memory over the run — low CPU utilization during a long wall
     time is the fingerprint of a shuffle/IO bottleneck, not a compute one.
  4. A/B-testing the same transform after rechunking time-contiguous, which
     directly confirms or refutes the chunking hypothesis.
  5. Optionally timing the spatial aggregation (needs a shapefile).

Usage
-----
Real data:
    python benchmarks/profile_temporal.py \
        --zarr /path/to/tempPrecLand2017.zarr --var t2m \
        --groupby year --calc dd \
        --shp /path/to/counties.shp --regionid GEOID --project-dir /tmp/aggfly

Synthetic fallback (no data needed, sanity-checks the harness itself):
    python benchmarks/profile_temporal.py --synthetic

Outputs a text report to stdout and, if --html is passed, a dask diagnostics
plot you can open in a browser.
"""

import argparse
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
import xarray as xr

import dask
from dask.diagnostics import Profiler, ResourceProfiler, CacheProfiler, ProgressBar

import aggfly as af


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
@contextmanager
def timed(label):
    print(f"\n=== {label} ===")
    t0 = time.perf_counter()
    yield
    print(f"--> {label}: {time.perf_counter() - t0:.2f}s wall")


def human_bytes(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def report_chunking(da: xr.DataArray, groupby_freq: str):
    """Print the diagnostics that reveal a chunking/alignment problem."""
    print("\n" + "#" * 70)
    print("# CHUNKING DIAGNOSTIC")
    print("#" * 70)
    print(f"shape={dict(zip(da.dims, da.shape))}  dtype={da.dtype}")
    print(f"total size (uncompressed): {human_bytes(da.nbytes)}")

    # On-disk encoding chunks (what Zarr actually stored) vs dask chunks
    # (how aggfly asked to read it). A mismatch => rechunk on every load.
    enc_chunks = da.encoding.get("chunks") or da.encoding.get("preferred_chunks")
    print(f"\non-disk (Zarr) chunks : {enc_chunks}")
    if da.chunks is None:
        print("dask chunks           : <not dask-backed>")
        return
    dask_chunks = {d: c for d, c in zip(da.dims, da.chunks)}
    print(f"dask chunks (per dim) : { {d: c[0] for d, c in dask_chunks.items()} }"
          f"  (first block per dim)")

    n_blocks = int(np.prod([len(c) for c in da.chunks]))
    # size of one block
    block_elems = int(np.prod([c[0] for c in da.chunks]))
    print(f"n chunks total        : {n_blocks}")
    print(f"one chunk             : ~{human_bytes(block_elems * da.dtype.itemsize)}")

    if enc_chunks is not None and tuple(c[0] for c in da.chunks) != tuple(enc_chunks):
        print("\n  ⚠  dask chunks DIFFER from on-disk chunks: every load rechunks.")
        print("     This alone can dominate wall time. Consider matching chunks= to")
        print("     the Zarr encoding, or re-encoding the Zarr for your access pattern.")

    # Alignment: how many time-chunks does one resample group span?
    if "time" in da.dims:
        time_chunk = da.chunks[da.get_axis_num("time")][0]
        try:
            per_group = (
                pd.Series(1, index=da["time"].to_index())
                .resample(groupby_freq).count()
            )
            typical_group_len = int(per_group.median())
        except Exception:
            typical_group_len = None
        print(f"\ntime chunk length     : {time_chunk} steps")
        if typical_group_len:
            spanned = max(1, round(typical_group_len / time_chunk))
            print(f"typical '{groupby_freq}' group : {typical_group_len} steps "
                  f"→ spans ~{spanned} time-chunk(s)")
            if spanned > 4:
                print("  ⚠  each group spans many chunks → grouped reduction must")
                print("     combine across chunks (flox helps; a coarser time chunk")
                print("     aligned to the group would help more).")
            if time_chunk > typical_group_len * 2:
                print("  ⚠  time chunk is much larger than a group → over-reads per group.")


def build_spec(calc, groupby):
    """A representative single-variable temporal spec for the given calc."""
    if calc == "mean":
        return [("aggregate", {"calc": "mean", "groupby": groupby})]
    if calc == "dd":
        return [
            ("aggregate", {"calc": "dd", "groupby": "date", "ddargs": [10, 30, 0]}),
            ("aggregate", {"calc": "sum", "groupby": groupby}),
        ]
    if calc == "bins":
        return [
            ("aggregate", {"calc": "mean", "groupby": "date"}),
            ("aggregate", {"calc": "bins", "groupby": groupby,
                           "ddargs": [[-99, 20, 0], [20, 99, 0]]}),
        ]
    if calc == "poly":
        return [
            ("aggregate", {"calc": "mean", "groupby": "date"}),
            ("transform", {"transform": "power", "exp": np.arange(1, 4)}),
            ("aggregate", {"calc": "sum", "groupby": groupby}),
        ]
    raise ValueError(f"unknown calc {calc}")


def run_temporal(dataset, spec, label):
    """Materialize a temporal aggregation under dask diagnostics; report where time goes."""
    with Profiler() as prof, ResourceProfiler(dt=0.25) as rprof, CacheProfiler() as cprof:
        with timed(label):
            out = af.aggregate_time(dataset=dataset, weights=None, variable=spec)
            # force the whole graph, mimicking a real run
            _ = dask.compute(*[d.da for d in out.values()])

    # summarize the resource trace: compute-bound vs waiting
    if rprof.results:
        cpu = np.array([r.cpu for r in rprof.results])
        mem = np.array([r.mem for r in rprof.results])
        print(f"    CPU util over run : mean={cpu.mean():.0f}%  max={cpu.max():.0f}%")
        print(f"    RSS over run      : mean={human_bytes(mem.mean()*1e6)}  "
              f"max={human_bytes(mem.max()*1e6)}")
        if cpu.mean() < 60:
            print("    ⚠  low mean CPU → wall time is NOT compute-bound "
                  "(likely rechunk/IO/shuffle). A faster kernel language won't help much.")
        else:
            print("    ✔  high CPU → genuinely compute-bound; a Numba/Rust kernel is the lever.")
    return prof, rprof, cprof


# --------------------------------------------------------------------------- #
# data loaders
# --------------------------------------------------------------------------- #
def load_real(args):
    ds = af.dataset_from_path(
        args.zarr,
        var=args.var,
        name="profile",
        lon_is_360=args.lon_is_360,
        preprocess=(lambda x: x - 273.15) if args.kelvin else None,
    )
    return ds


def load_synthetic(days=365, ny=180, nx=360):
    """A dask-backed cube big enough to exercise the pipeline without real data."""
    time_ix = pd.date_range("2017-01-01", periods=days * 24, freq="h")
    lat = np.linspace(-89.5, 89.5, ny)
    lon = np.linspace(-179.5, 179.5, nx)
    data = dask.array.random.normal(
        20, 15, size=(len(time_ix), ny, nx), chunks=(24, -1, -1)
    )
    da = xr.DataArray(
        data, dims=["time", "latitude", "longitude"],
        coords={"time": time_ix, "latitude": lat, "longitude": lon},
    )
    return af.Dataset(da, lon_is_360=False)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zarr", help="path to a .zarr (or netcdf) climate file")
    p.add_argument("--var", default="t2m")
    p.add_argument("--synthetic", action="store_true",
                   help="use a synthetic in-process cube instead of --zarr")
    p.add_argument("--syn-days", type=int, default=365)
    p.add_argument("--kelvin", action="store_true", help="apply x-273.15 preprocess")
    p.add_argument("--lon-is-360", dest="lon_is_360", action="store_true")
    p.add_argument("--groupby", default="year", choices=["date", "month", "year", "week"])
    p.add_argument("--calc", default="dd", choices=["mean", "dd", "bins", "poly"])
    p.add_argument("--scheduler", default="threads",
                   choices=["threads", "processes", "synchronous"])
    p.add_argument("--shp", help="shapefile for the spatial step (optional)")
    p.add_argument("--regionid", default="GEOID")
    p.add_argument("--project-dir", default="/tmp/aggfly-profile")
    p.add_argument("--html", help="write dask diagnostics plot to this .html path")
    args = p.parse_args()

    dask.config.set(scheduler=args.scheduler)
    print(f"dask scheduler: {args.scheduler}")

    if args.synthetic or not args.zarr:
        print("Loading SYNTHETIC dataset "
              f"({args.syn_days} days hourly, global) ...")
        dataset = load_synthetic(days=args.syn_days)
    else:
        print(f"Loading {args.zarr} (var={args.var}) ...")
        dataset = load_real(args)

    report_chunking(dataset.da, args.groupby)

    spec = build_spec(args.calc, args.groupby)

    # 1) as-loaded
    run_temporal(dataset, spec, f"TEMPORAL [{args.calc}/{args.groupby}] as-loaded")

    # 2) A/B: rechunk time-contiguous (whole time axis in one chunk per space block)
    #    If this is dramatically faster, the bottleneck was chunking, not compute.
    rechunked = af.Dataset(
        dataset.da.chunk({"time": -1, "latitude": "auto", "longitude": "auto"}),
        lon_is_360=dataset.lon_is_360,
    )
    run_temporal(rechunked, spec,
                 f"TEMPORAL [{args.calc}/{args.groupby}] rechunked time-contiguous")

    # 3) optional spatial step
    if args.shp:
        with timed("build weights"):
            gr = af.georegions_from_path(args.shp, regionid=args.regionid)
            dataset.georegions = gr
            weights = af.weights_from_objects(dataset, gr, project_dir=args.project_dir)
            weights.calculate_weights()
        with timed(f"FULL aggregate_dataset [{args.calc}/{args.groupby}]"):
            df = af.aggregate_dataset(dataset=dataset, weights=weights, variable=spec)
            print(df.head())

    if args.html:
        from dask.diagnostics import visualize
        # re-run quickly capturing for the plot
        with Profiler() as prof, ResourceProfiler(dt=0.25) as rprof:
            out = af.aggregate_time(dataset=dataset, weights=None, variable=spec)
            dask.compute(*[d.da for d in out.values()])
        visualize([prof, rprof], filename=args.html, show=False)
        print(f"\nwrote diagnostics plot → {args.html}")


if __name__ == "__main__":
    main()
