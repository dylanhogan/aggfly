"""Micro-bench the get_weighted_area_weights per-region total: OLD dask.dataframe vs
NEW plain pandas, on a synthetic weights frame (n_regions groups x n_rows)."""
import time, resource, gc, sys
import numpy as np, pandas as pd, dask, dask.dataframe

def synth(n_rows, n_regions, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "index_right": rng.integers(0, n_regions, n_rows),
        "raster_weight": rng.random(n_rows),
        "area_weight": rng.random(n_rows),
    })

def new_path(weights):  # plain pandas (current code)
    rt = (weights[["index_right","raster_weight"]].groupby("index_right").sum()
          .rename(columns={"raster_weight":"total_weight"}))
    rt["zero_weight"] = rt.total_weight == 0
    return weights.merge(rt, how="left", left_on="index_right", right_index=True)

def old_path(weights, chunks=30):  # dask.dataframe (previous code)
    dw = dask.dataframe.from_pandas(weights, npartitions=chunks)
    rt = (dw[["index_right","raster_weight"]].groupby("index_right").sum()
          .rename(columns={"raster_weight":"total_weight"}))
    rt["zero_weight"] = rt.total_weight == 0
    return dw.merge(rt, how="left", left_on="index_right", right_index=True).compute()

def timeit(fn, arg, reps=3):
    ts=[]
    for _ in range(reps):
        gc.collect(); t0=time.perf_counter(); r=fn(arg); ts.append(time.perf_counter()-t0)
    return min(ts), len(r)

def mb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024

with dask.config.set(scheduler="threads"):
    print(f"{'n_rows':>10} {'regions':>8} {'OLD dask':>10} {'NEW pandas':>11} {'speedup':>8}")
    for n_rows, n_reg in [(100_000,3000),(500_000,3000),(2_000_000,3000),(5_000_000,3000)]:
        w = synth(n_rows, n_reg)
        t_old,_ = timeit(lambda x: old_path(x), w)
        t_new,n = timeit(lambda x: new_path(x), w)
        print(f"{n_rows:>10} {n_reg:>8} {t_old:>9.3f}s {t_new:>10.3f}s {t_old/t_new:>7.1f}x")
    print(f"peakRSS={mb():.0f} MB")
