"""
Profile the mask() spatial join (grid-cell centroids -> region polygons, predicate
'within') two ways, to decide whether to keep dask-geopandas or drop it:
  --impl dask  : dask_geopandas.from_geopandas(fc, npartitions).sjoin(poly).compute()
  --impl plain : geopandas.sjoin(fc, poly, predicate='within')

fc is a GLOBAL lon/lat grid at --res (the real fc: mask() joins ALL grid centroids,
most of which match nothing) against the 50-state shapefile. One impl per process so
ru_maxrss is isolated. Prints wall (min of reps), peak RSS, and matched-pair count.

Run: <py> bench_sjoin.py --impl {dask,plain} --res 0.25 [--fcparts 30] [--reps 2]
"""
import argparse, os, time, resource, gc
import numpy as np, pandas as pd, geopandas as gpd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHP = os.path.join(_REPO, "benchmarks/data/usa_simple_noHI.shp")


def maxrss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def build(res):
    lon = np.arange(-180.0, 180.0, res)
    lat = np.arange(-90.0, 90.0, res)
    xx, yy = np.meshgrid(lon, lat)
    fc = gpd.GeoDataFrame(
        {"cell_id": np.arange(xx.size)},
        geometry=gpd.points_from_xy(xx.ravel(), yy.ravel()),
        crs="EPSG:4326",
    )
    poly = gpd.read_file(SHP).set_crs("EPSG:4326")
    poly = poly.reset_index()[["index", "geometry"]]
    return fc, poly


def run_plain(fc, poly):
    return gpd.sjoin(fc, poly, predicate="within", how="inner")


def run_dask(fc, poly, fcparts):
    import dask, dask_geopandas
    with dask.config.set(scheduler="threads"):
        dfc = dask_geopandas.from_geopandas(fc, npartitions=fcparts)
        dpoly = dask_geopandas.from_geopandas(poly, npartitions=1)
        return dfc.sjoin(dpoly, predicate="within").compute()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["dask", "plain"], required=True)
    ap.add_argument("--res", type=float, default=0.25)
    ap.add_argument("--fcparts", type=int, default=30)
    ap.add_argument("--reps", type=int, default=2)
    args = ap.parse_args()

    fc, poly = build(args.res)
    fn = (lambda: run_dask(fc, poly, args.fcparts)) if args.impl == "dask" else (lambda: run_plain(fc, poly))

    walls, n = [], None
    for _ in range(args.reps):
        gc.collect()
        t0 = time.perf_counter()
        r = fn()
        walls.append(time.perf_counter() - t0)
        n = len(r)
    walls = np.array(walls)
    print(f"impl={args.impl:5s} res={args.res} fc={len(fc)} parts={args.fcparts} | "
          f"wall min={walls.min():6.2f}s med={np.median(walls):6.2f}s | "
          f"peakRSS={maxrss_mb():5.0f}MB | matched={n}")


if __name__ == "__main__":
    main()
