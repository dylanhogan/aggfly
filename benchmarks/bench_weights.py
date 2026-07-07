"""
Profile GridWeights.calculate_weights() (area-weights path) old-code vs new-code on
the SAME stack, to check whether dropping dask parallelism in the weights module hurts.

Realistic case: 50 US states x a 0.25-degree CONUS grid (~24k cells). The area path
exercises the rewritten intersect_border_cells (dropped dask) + the kept mask() sjoin
(still dask). A fresh GridWeights is built per rep because mask()/simplify are lru_cached.

Reports per-rep wall time and process peak RSS (ru_maxrss).
Run: <venv>/bin/python bench_weights.py --reps 3 [--simplify 0.05]
"""
import argparse, os, time, resource, gc
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import aggfly as af

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHP = os.path.join(_REPO, "notebooks/giovanni_example/usa_simple_noHI.shp")


def maxrss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    c = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return (r + c) / 1024.0  # ru_maxrss is KB on Linux


def build_inputs(res, simplify):
    gdf = gpd.read_file(SHP).set_crs("EPSG:4326")
    gdf["rid"] = np.arange(len(gdf))
    georegions = af.GeoRegions(gdf, regionid="rid")

    # 0.25-deg CONUS grid; synthetic climate values (weights only use the geometry)
    lon = np.arange(-125.0, -66.0, res)
    lat = np.arange(24.0, 50.0, res)
    time_ = pd.date_range("2000-01-01", periods=1, freq="D")
    da = xr.DataArray(
        np.zeros((1, len(lat), len(lon)), dtype="float32"),
        dims=["time", "latitude", "longitude"],
        coords={"time": time_, "latitude": lat, "longitude": lon},
    )
    clim = af.Dataset(da, lon_is_360=False)
    return clim, georegions, len(lat) * len(lon)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--res", type=float, default=0.25)
    ap.add_argument("--simplify", type=float, default=None)
    ap.add_argument("--label", default="?")
    args = ap.parse_args()

    clim, georegions, ncells = build_inputs(args.res, args.simplify)
    print(f"[{args.label}] grid={ncells} cells ({args.res} deg), {len(georegions.shp)} regions, "
          f"simplify={args.simplify}")

    walls = []
    nweights = None
    for i in range(args.reps):
        gc.collect()
        w = af.weights_from_objects(clim, georegions, project_dir=None, simplify=args.simplify)
        t0 = time.perf_counter()
        w.calculate_weights()
        dt = time.perf_counter() - t0
        walls.append(dt)
        nweights = len(w.weights)
        print(f"  rep {i}: {dt:7.2f}s   weights_rows={nweights}")

    walls = np.array(walls)
    print(f"[{args.label}] wall: min={walls.min():.2f}s median={np.median(walls):.2f}s "
          f"max={walls.max():.2f}s  |  peakRSS={maxrss_mb():.0f} MB  |  weight_rows={nweights}")


if __name__ == "__main__":
    main()
