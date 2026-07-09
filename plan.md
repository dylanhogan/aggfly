# aggfly backend acceleration — follow-up plan

Context: the `engine="numba"` temporal backend (`aggfly/aggregate/nb_kernels.py`) is
landed and bit-equivalent to the dask path. On real ERA5 it is ~11.8× faster than
dask on native/small spatial chunks (15.2s @ 9.3 GB vs 179s @ 12.4 GB) but ~0.7×
(slower) on large 250×250 chunks, because dask's vectorized numpy is already
compute-bound on big blocks and per-block numba threads oversubscribe the dask
thread pool. These three items build on that result.

---

## 1. Auto-select the temporal engine from chunk size — ✅ DONE

Implemented as `resolve_engine()` / `max_spatial_block_cells()` in `nb_kernels.py`
with the tunable `NUMBA_MAX_CELLS_PER_BLOCK` (default 150*150). `engine` now defaults
to `"auto"` across `TemporalAggregator`/`aggregate_time`/`aggregate_dataset` and is
resolved per `execute()` against that step's chunking; `"dask"`/`"numba"` still force
a backend. Unit tests: `test_resolve_engine`, `test_max_spatial_block_cells`.

**Goal:** users get the fast path without having to know the "numba + small chunks,
dask + big chunks" rule. Pick the engine per spatial-chunk size.

**Why:** the crossover is real and measured — numba wins when spatial blocks are
small (many GIL-bound dask tasks) and loses when they are large (few compute-bound
vectorized blocks + numba/dask thread oversubscription).

**Approach:**
- Add `engine="auto"` and make it the default resolution inside
  `TemporalAggregator.execute` (or when `TemporalAggregator` is constructed in
  `aggregate_time`). Keep explicit `"dask"` / `"numba"` as overrides.
- Decision rule from the spatial chunk size of `dataset.da` (cells per block =
  product of the non-time chunk lengths). Threshold near the measured crossover:
  small blocks (≈ ≤150×150, i.e. tens of blocks over the grid) → `numba`; large
  blocks → `dask`. Start with a cells-per-block threshold (e.g. ~150*150 ≈ 22_500)
  and expose it as a module constant so it is tunable.
- Also gate on GIL pressure proxy: `n_spatial_blocks` (numba's advantage grows with
  more blocks). Prefer numba when `n_spatial_blocks` is large regardless of absolute
  block size.
- If `calc` is not in `NUMBA_CALCS`, `"auto"` must fall back to `"dask"`.

**Validation:** extend `benchmarks/bench_engine.py` to sweep chunk sizes
(50/100/150/250) for both engines and confirm `auto` picks the faster one at each
size. Add a unit test asserting the engine resolver returns the expected backend for
representative chunk shapes (pure function, no compute).

**Risk/notes:** keep the resolver a small pure function
(`_resolve_engine(engine, da, calc) -> "dask"|"numba"`) so it is unit-testable and
the threshold is documented in one place. Do not change results, only scheduling.

---

## 2. Full multi-step fusion (daily-mean → poly → monthly-sum in one kernel pass)

**Goal:** for common multi-step specs, run the whole temporal chain in a single
compiled pass per spatial chunk with no intermediate dask arrays materialized —
the biggest remaining win for polynomial/degree-day panels.

**Why:** the current `engine="numba"` still runs each spec step as its own
`map_blocks` (daily mean, then power, then monthly sum), materializing intermediate
daily arrays and rebuilding the graph per step. The standalone prototype in the
scratchpad (`numba_fused.py`) did hourly→daily-mean→powers→monthly-sum in ONE pass
and hit ~10× on a single chunk-year with zero intermediate materialization.

**Approach:**
- Recognize a fusible spec chain in `aggregate_time` before dispatching per step.
  Target the dominant pattern first:
  `[('aggregate', mean/date), ('transform', power[...]), ('aggregate', sum|mean/<freq>)]`
  and the degree-day variant
  `[('aggregate', dd/date or sine_dd/date), ('aggregate', sum/<freq>)]`.
- Add a `fused_*` kernel in `nb_kernels.py` parameterized by: inner group (date),
  outer group (month/year), inner reduction (mean), transform (powers array) or
  degree-day thresholds, outer reduction (sum/mean). Emit output shape
  `(n_outer, P, Y, X)` (or per-threshold) directly.
- Keep the two-level group bookkeeping (`day_of`, `month_of_day`, bounds) that the
  prototype already validated.
- Gate behind the same `engine`/`auto` selection and the small-chunk regime.
- Non-fusible specs (splines, interactions, `inter`, multiple upstream datasets,
  multi-`ddargs` crossed with powers) fall back to the current per-step numba path.

**Validation:** assert bit-equivalence to the per-step numba path (and thus dask)
on the test fixtures and on a synthetic NaN-containing cube; then benchmark the
fused vs per-step numba on real ERA5 for the canonical poly SPEC.

**Risk/notes:** fusion multiplies the combinatorics of supported specs — scope it to
the two named chains above and keep a clean "is this chain fusible?" predicate;
everything else uses the existing path. Do not try to fuse arbitrary DSL chains.

---

## 3. Fix the `sine_dd` NaN masking bug (dask path) — **DONE**

**Goal:** correct a latent correctness bug in the existing dask degree-day code,
independent of the numba work.

> **Status (2026-07):** the code fix landed with the numba work (commit `a36fb81`);
> this branch adds the dedicated regression test the plan called for
> (`test_sine_dd_partial_nan_masking`). The bug description below was **corrected**
> from an earlier draft that assumed the wrong dimension order — see the strikethrough
> notes. Net effect and fix are unchanged.

**Bug:** in `aggfly/aggregate/temporal.py`, `_sine_cdd` and `_sine_hdd` computed
`nan_cells = da.where(np.isnan(frame[:, :, 0]), np.nan, 1)`. Within
`resample(...).reduce(func, axis=<time>)`, after `clean_dims` `frame` is
`(lat, lon, time)` and time is the last (reduction) axis. So `frame[:, :, 0]` is a
valid `(lat, lon)` slice — **the window's first timestep** — not longitude.
~~indexes longitude 0~~. The mask therefore only inspected the *first* timestep of
each resample window: a cell that was **valid at the window's first step but NaN at a
later step was left unmasked**, and since the `da.where` case conditions evaluate
False against the NaN-propagated `tmax/tmin`, that cell silently got `0` instead of
`NaN`. (~~this mis-set ~27k cells~~ — the "both directions / ~27k cells" figure and
the ~~"can outright CRASH" / broadcast `ValueError`~~ were artifacts of a scratch
harness that did not transpose to `(lat, lon, time)`; in the real pipeline layout the
shapes always broadcast and there is no crash.)

**Correct behavior:** a group's degree-day output is NaN iff the group window
contains any NaN (the convention the numba kernel `_block_sine_dd` already uses; on
valid data the two paths agree to ~1e-15).

**Fix (landed):**
- `nan_cells = da.where(np.isnan(frame).any(axis=axis), np.nan, 1)` in both
  `_sine_cdd` and `_sine_hdd`. `_sine_dd` / `_multi_sine_dd` compose these and inherit
  it. `np.zeros_like(frame[:, :, 0])` remains as a correct `(lat, lon)` shape template.
- `tmax`/`tmin`/`tavg` already propagate NaN via `np.max/min/mean` over `axis`, so the
  explicit mask only needs to force the same "any NaN in window" set to NaN.

**Validation (this branch):** `test_sine_dd_partial_nan_masking` builds a 2×2 grid ×
2-day (12h) dataset with a NaN at a window's *second* step (the exact missed case) and
at another window's *first* step, and asserts dask `sine_dd` == numba `sine_dd`
(`equal_nan`), that both NaN windows → NaN, and that valid windows stay finite/positive.
Reverting the mask to `frame[:, :, 0]` makes this test fail (dask yields `0.` where
numba yields `NaN`), confirming it guards the regression.

**Risk/notes:** the fix changed existing outputs on NaN-containing (ocean/masked) cells
for `sine_dd` only — flag as a bugfix in the changelog since it alters numbers for
affected cells. Pure-land datasets are unaffected.

---

## 4. NetCDF read path — investigation ✅ MEASURED, decision pinned

**Context:** with the numba engine making temporal compute cheap, the read is now the
dominant cost for raw-NetCDF workloads. Raw ERA5 lives at
`/shared/vol1/ERA5/raw/ERA5_{year}.nc` (85 files, ~21 GB each, 1.7 TB). On-disk HDF5
layout is pessimal for a time reduction: chunks `(138, 23, 45)` — bricked in time AND
both spatial dims — with zlib1+shuffle, dims `valid_time`(8784)×`latitude`(721)×
`longitude`(1440), float32, vars `t2m`/`tp`.

**Benchmarks (`benchmarks/profile_netcdf.py` = configs A/B/C cold; `profile_netcdf_zarr.py`
= A/C/Z1/K1/K2 warm), 120×120 × full-year window = 482 MB decompressed:**
- Cold cache exposes the HDF5 global read lock: A) native brick + threads = 12.0s / **35%
  CPU** / 40 MB/s (cores idle on the serialized lock, NOT disk-bound); B) native brick +
  **processes** = 4.0s / 119 MB/s (3×); C) rechunk time=-1 + threads = 3.0s / 158 MB/s (4×).
- Warm cache (can't drop caches w/o root → lock penalty understated): A = 2.78s / 140% /
  173 MB/s; **Z1) `to_zarr` rechunk time=-1, blosc = 1.07s / 253% / 450 MB/s, store 0.53×
  the decompressed size** (blosc beats zlib1+shuffle) — 2.6× A, one-time window write 2.7s;
  K1) kerchunk ref, native bricks = 2.57s / 187 MB/s (≈A warm; lock removal shows more
  cold); K2) kerchunk ref + rechunk = 1.85s / 261 MB/s, **zero data copy**.
- kerchunk `SingleHdf5ToZarr` scans the whole 21 GB file in **0.5s → 131 k chunk refs**
  (metadata only). **Correctness:** kerchunk daily-mean is **bit-identical** to netcdf;
  zarr differs by 6e-5 K = float32 reduction-order noise from rechunking only (raw stored
  values bit-identical, lossless — ERA5 t2m is true float32, no scale/offset packing).

**Decision (pinned until deps are modernized — see item 5):** for aggfly's repeated-
processing pattern, **convert each year to Zarr once** (rechunk time-contiguous, uniform
spatial tiles, blosc) — the clear ceiling. Keep **kerchunk** as the zero-duplication
fallback when a second copy of 1.7 TB is unaffordable.

**Helper — ✅ DONE (`aggfly/dataset/zarr_convert.py`).** `af.dataset_to_zarr(dataset, store,
chunking="auto", target_mb=256, overwrite=...)` + `af.zarr_from_path(path, var, store, ...)`.
Dataset-agnostic by construction: it operates on a normalized `af.Dataset` (dims already
`latitude/longitude/time` via `dataset_from_path`'s `xycoords`/`timecoord`/`lon_is_360`/
`preprocess`), so any gridded source aggfly can load converts with no special-casing —
validated by a test that converts an ERA5-style `valid_time`/`lat`/`lon` file. Size-aware
chunking keeps time contiguous under a per-chunk budget (splitting time only for very long
hourly series). Encodes both original gotchas: (1) uniform `.chunk()` before `to_zarr`
(rejects ragged chunks); (2) strips source `.encoding` on var+coords (prevents
scale_factor/add_offset re-quantization and stale brick-chunk conflicts). Returns a Dataset
on the new store that plugs into `weights_from_objects`/`aggregate_dataset`. Works on zarr
2 and 3 (via xarray). kerchunk remains the deferred zero-copy alternative.

**Why pinned:** kerchunk needs `h5py` and a version compatible with aggfly's `zarr<3` pin
(`kerchunk==0.2.7`; ≥0.2.10 demands zarr≥3). Rather than special-case old kerchunk, we
first modernize the dependency stack (item 5), then re-evaluate to_zarr vs kerchunk vs
`virtualizarr` (the actively-maintained successor to kerchunk's reference approach) on
current zarr-v3 / xarray, and only then build the conversion helper.

---

## 5. Dependency + Python version modernization — ✅ LANDED (2026-07)

**STATUS:** `pyproject.toml` bumped to a current stack and `poetry.lock` regenerated;
`python = ">=3.11,<3.14"`. Resolved + installed by poetry into a py3.12 venv and the full
suite passes **9/9** on the actual locked stack: dask/distributed 2026.7.0, xarray 2026.4.0,
zarr 3.1.6, numpy 2.4.6, numba 0.66.0, pandas 3.0.3, geopandas 1.1.4, shapely 2.1.2,
rioxarray 0.19.0, rasterio 1.4.4 — and **no dask-geopandas**. CLAUDE.md + the pyproject
comment block rewritten. Remaining (optional follow-up, not blocking): re-run the perf
benchmarks on the new stack — `benchmarks/bench_engine.py` (numba vs dask under numpy 2 /
numba 0.66, re-check `NUMBA_MAX_CELLS_PER_BLOCK`) and item-4's `profile_netcdf_zarr.py`
against zarr 3 / current kerchunk-or-virtualizarr — then build the ERA5→Zarr helper (item 4).

**Goal:** move aggfly off the narrow `python >=3.11.6,<3.12.3` pin and the frozen
dask/dask-geopandas/zarr/numpy versions onto current stable releases, so we can (a) use
zarr v3, modern xarray, and `virtualizarr`/current kerchunk for the read-path work, and
(b) stop being blocked by the 3.12.3 `dask-geopandas` breakage.

**Why now:** the read-path decision (item 4) is gated on the zarr version; the whole stack
is >1 year stale and pinned around a since-fixed `dask-geopandas` bug. Doing this first
unblocks item 4's helper and de-risks future work.

**Binding constraint — RESOLVED UPSTREAM (checked 2026-07-07).** The Python pin was narrow
because Python 3.12.3+ broke `dask-geopandas` (dask query-planning / dask-expr dataframe
internals). That is now fixed upstream: `dask-geopandas` is actively maintained (NOT
deprecated), latest **0.5.0 (Jun 2025)**, and **v0.4.3 (Jan 2025) requires `dask>=2025.1.0`
and `python>=3.10`** — i.e. it has been ported to modern dask and imposes **no Python
ceiling**. So the jointly-compatible pair is effectively already chosen: `dask-geopandas
0.5.0` + `dask>=2025.1.0` + `python>=3.10` (target 3.12/3.13). This de-risks the whole
effort — the migration cost moves from "unsolvable version conflict" to "audit aggfly's own
`spatial.py` dask-dataframe usage against modern dask's API" (a code fix, not a blocker).
Still verify the exact pair poetry resolves and that `dask`/`dask-geopandas` move **together**.

**Other constraints to respect:**
- numpy is pinned `<2.0.0` today; numba `<0.60` needs numpy `<1.27`. numba ≥0.60 supports
  numpy 2.x — the numba engine (`nb_kernels.py`) must be re-benchmarked after any numpy-2
  bump (kernels are the hot path).
- zarr v2→v3 is a breaking API change (store/consolidated-metadata/codecs) — audit every
  `zarr`/`to_zarr`/`open_zarr` call and the `ProjectCache` store usage.

**Approach (staged, each stage green before the next):**
1. **Baseline & inventory.** ✅ DONE — see `modernization_baseline.md`. Baseline is
   **7 passed in 8.82s** on the locked py3.11 stack. Inventory refined the risk ranking:
   the primary migration surface is **modern dask dataframe + dask-geopandas**
   (`spatial.py` groupby weighted-average, `grid_weights.py`/`grid.py`/`georegions.py`
   `from_geopandas`/`.sjoin`/`.buffer`/`.intersection`); **numpy 1→2 is trivial** (no
   removed aliases; only `np.in1d`→`np.isin` at 3 sites + numba≥0.60); **zarr v2→v3 is
   NOT a risk for existing code** (package never imports zarr; `ProjectCache` uses
   `to_netcdf`+dill, not a zarr store — zarr v3 only matters for item-4's new helper).
2. **Confirm the joint solution (already resolved upstream — just verify).** The
   `dask-geopandas 0.5.0` + `dask>=2025.1.0` + `python>=3.10` combo removes the old blocker;
   confirm the exact versions poetry resolves and that the pair moves together. The real work
   this exposes is the `spatial.py` audit against modern dask's dataframe API (see step 4),
   not a version hunt.
3. **Widen Python + bump core stack in a throwaway env.** In a scratch venv, bump
   python, dask(+distributed), dask-geopandas, xarray, zarr(→v3), numpy(→2.x), numba(≥0.60),
   geopandas/rioxarray/pyproj. Resolve with poetry; capture the new lock.
4. **Fix breakage by subsystem** (scoped by the step-2 audit — see
   `modernization_baseline.md`; scratch env at `scratchpad/modernenv`, py3.12 pip). Under the
   modern stack the suite is **4 passed / 5 errors**, and every error is the `weights` fixture —
   the non-weights surface (spatial matmul, engine) is already clean. Concrete work:
   - **Weights module — ✅ DONE.** Rewrote the small-data dask ops in `grid_weights.py` as
     plain geopandas/pandas (stack-agnostic — green on BOTH the old locked stack and the
     modern gpd1.1/pandas3/dask2026 stack, numeric assertions unchanged):
     `intersect_border_cells` now does a positional `GeoSeries.intersection(other,
     align=False)` (fixes the geopandas-1.0 align + dask-geopandas partition breakage);
     `get_weighted_area_weights` uses a plain pandas groupby (dropped `dask.dataframe`);
     `simplify_poly_array` uses plain `geometry.simplify`. Kept the one large sjoin over all
     cell centroids (`mask()`) on dask-geopandas — it already works on the modern stack.
     Removed now-dead `import dask`/`import dask.array`. **With this fix the whole test suite
     passes on the modern stack (9/9)** — the temporal/numba paths were only blocked behind
     the weights fixture. (The `mask()` sjoin was later also moved off dask-geopandas — see the
     dask-geopandas removal note below.) **Perf validated** (`benchmarks/bench_weights.py`,
     `benchmarks/bench_groupby.py`, 50 states × 0.25°/0.1° CONUS grid, same stack): dropping
     dask is a net win at realistic scale — end-to-end `calculate_weights` ~10-13% faster,
     memory equal (±2%), identical output, less variance. The groupby has a crossover (dask
     wins beyond ~2-5M weight rows) but realistic grid-resolution weight frames are ~10k-500k
     rows where pandas is 3-10× faster; a size-thresholded switch is the mitigation if ever
     needed.
   - **Regions module — ✅ DONE.** `georegions.py` `poly_array` buffer now uses plain
     `self.shp.buffer()` (dropped `dask_geopandas`; removed the `dask_geopandas`/redundant
     `dask` imports — only `dask.array` remains, for the optional `datatype="dask"` branch).
     `np.in1d`→`np.isin` in `georegions.py` (`sel`/`drop`) and `shp_utils.py`. Test fixture
     `unary_union`→`shapely.union_all(np.asarray(pts))` (stack-agnostic — shapely 2 is on both;
     `geopandas.union_all()` doesn't exist in the locked gpd 0.14). All stack-agnostic: 9/9 on
     BOTH envs, and the modern env now runs **warning-free**.
   - **dask-geopandas DROPPED ENTIRELY — ✅ DONE.** Profiled the two remaining `mask()` sjoins
     (`benchmarks/bench_sjoin.py`, global grid × 50 states, modern stack): plain `gpd.sjoin`
     beats dask-geopandas at **every** size — 8.5× @ 259k pts, 4.2× @ 1M (global ERA5 0.25°),
     2.3× @ 6.5M, 1.3× @ 26M — with ~30% less memory and byte-identical matched counts; dask
     never crosses over (everything fits in RAM, so its partition/graph overhead is pure cost).
     Replaced both `mask()` sjoins with `gpd.sjoin(..., predicate="within")`, removed the
     `dask_geopandas` imports, and **removed `dask-geopandas` from `pyproject.toml`**. Proven
     independent: uninstalled dask-geopandas from the modern env → suite still 9/9. **This
     eliminates the dependency whose churn forced the narrow `python <3.12.3` pin — the Python
     range can now widen freely.** (poetry.lock regeneration deferred to step 6.)
   - **numpy 2:** `np.in1d`→`np.isin` (`georegions.py:187,220`, `shp_utils.py:32`).
   - **geopandas 1.0:** `unary_union`→`union_all()`.
   - **pandas 3.0** (bigger bump than assumed — CoW default, stricter index alignment): use
     positional `.to_numpy()` on computed-series assignments in the weights rewrite.
   - zarr v3 API in `ProjectCache` + any `to_zarr`/`open_zarr`; xarray resample/`.reduce`
     drift in `temporal.py` — re-verify (not yet exercised, blocked behind the weights fixture).
   NOTE: `spatial.py` no longer uses `dask.dataframe` at all (rewritten as a sparse
   weight-operator matmul, item 6). Re-run tests per subsystem.
5. **Re-benchmark the hot paths** on the new stack: `benchmarks/bench_engine.py` (numba vs
   dask, numpy-2) and the NUMBA_MAX_CELLS threshold; then re-run item 4's
   `profile_netcdf_zarr.py` with current kerchunk/virtualizarr on zarr-v3.
6. **Update pins & docs.** Rewrite the `pyproject.toml` version comments (they document the
   old 3.12.3 rationale) and the CLAUDE.md "Python version constraint" section to the new reality.

**Validation:** full `pytest` green on the new stack (the fixtures + `np.allclose` numeric
assertions are the correctness net); numba engine still bit-equivalent to dask; benchmarks
re-run and recorded. Land as its own branch/PR separate from feature work.

**Risk/notes:** the step-1 inventory (`modernization_baseline.md`) re-ranked the risks: the
dominant surface is **modern dask dataframe + dask-geopandas** (`spatial.py` groupby
weighted-average under dask-expr, plus `from_geopandas`/`.sjoin`/`.buffer`/`.intersection`
in the weights/regions modules). **numpy 1→2 is trivial** (only `np.in1d`→`np.isin`, +
numba≥0.60). **zarr v2→v3 does not touch existing code** — only item-4's new to_zarr helper.
Keep the old `poetry.lock` recoverable in case a transitive dep still forces a compromise.

---

## 6. Spatial aggregation rewritten as a sparse weight-operator matmul — ✅ DONE

**What:** `SpatialAggregator.compute()` (`aggfly/aggregate/spatial.py`) previously melted
the space×time cube to a pandas frame, merged the weights, encoded `(region_id, time)` as a
synthetic `group_ID`, wrapped the already-in-RAM frame in `dask.dataframe.from_pandas(...,
npartitions=50)`, and did a groupby-sum. It is now the weighted average expressed directly:
`result = (W @ C_masked) / (W @ valid)`, where `W` is a sparse (region × cell) operator built
once from the weights table (COO triplets) and applied lazily via `dask.array.map_blocks`
over the climate array's time chunks.

**Why (both a modernization win and a real improvement):**
- Removes the `dask.dataframe` groupby/`from_pandas` idiom — exactly the surface that churns
  under modern dask's query planner — so this step stops depending on the fragile API before
  the bump. It's now `dask.array` (stable) only.
- No shuffle: each time-chunk block is independent (`W @ block`). No full-cube
  materialization (the old path `dask.compute`'d everything then `to_dataframe`'d it).
- Drops the `group_ID` hack, the hardcoded `npartitions=50` (which ignored the param), and the
  in-place `self.weights["region_id"] = ...` mutation of the shared weights frame.
- Zero new dependencies: the scatter is numpy `gather + np.add.at` (scipy.sparse is NOT in the
  lock, so it was deliberately avoided; a scipy CSR fast-path can be added later if desired).

**Parity / tests:** bit-for-bit behaviour preserved. Existing `test_aggregate` /
`test_aggregate_numba` still green; added `test_spatial_matmul_multiregion_nan` (many-to-many
fractional overlap + per-timestep NaN renormalization, across multiple lazy time chunks) and
`test_spatial_matmul_dropna_empty_group` (all-NaN region/time dropped when denominator is 0),
each checked against an independent pure-loop weighted-average oracle. Full suite: 9 passed.

**Preserved quirk (flagged for later):** a cell/time is used only if *every* output name is
non-NaN there (the old `dropna(subset=names)` coupled names through a shared denominator).
Kept for exact parity; revisit if per-variable NaN masks are wanted.

---

## 7. Execution backend / hardware scaling — **✅ DONE (idiomatic option, 2026-07)**

**Goal:** one API that runs well from a modest single-disk box up to a multi-node HPC
cluster with a parallel filesystem, without aggfly reasoning about hardware.

**Why now:** the read path is the bottleneck (see [[zarr-read-gil-bound]] / item 4), and the
*optimal* I/O strategy is hardware-dependent — measured on this box:
- warm read is GIL-serialized to ~2 cores under the threaded scheduler; a process cluster
  gives ~1.86× end-to-end (16 workers × 1 numba thread; 32 regresses + worker-comm errors).
- `/dev/sda` is a **7200rpm HDD** (ST12000NM0008): cold reads are seek-bound, so concurrency
  (processes / `open_mfdataset` / kerchunk fan-out) *hurts* — sequential single-stream is best.
- On SSD/NVMe (queue depth), multiple disks / parallel FS (independent channels), or object
  storage (latency-bound), concurrency instead *helps*. So no single default is right.

**Design principle:** mirror `engine="auto"` for *scheduling* — aggfly stays
**execution-backend-agnostic**: it uses whatever dask scheduler/client is active and defaults
to threads when none. `engine=` (which kernel) and the scheduler (how tasks execute) are
orthogonal, both auto-defaulted, both overridable. **Correctness invariant:** numba results are
identical across threaded vs process execution (verified) — "pick the backend for your
hardware; the numbers never change, only the speed."

**Per-hardware recipes (docs):**
- single disk (HDD/SSD): default threaded `engine="auto"` — nothing to configure; sequential.
- fat single node: `client = af.start_dask_client(n_workers=~16, threads_per_worker=1)` → process
  parallelism for the GIL-bound warm read.
- HPC multi-node + parallel FS: user brings a `dask_jobqueue.SLURMCluster`/`PBSCluster` +
  `Client(cluster)`; aggfly uses it with **no HPC-specific code and no dask-jobqueue dependency**.
- cloud/object store: object-store-backed zarr + distributed/async client (ties to item 4/cloud).

**Decision:** idiomatic option chosen (ambient-client contract + docs + the two fixes below).
Not the batteries-included preset factory — no new execution abstraction.

**What landed:**
1. **Fixed the trap:** removed the dead `n_workers/threads_per_worker/processes/memory_limit/
   cluster_args` params from `aggregate_dataset` (documented but never acted on — the
   `shutdown_dask_client()` calls were commented out). To stay non-breaking, those kwargs are
   now absorbed from `**kwargs` and raise a `DeprecationWarning` pointing at `start_dask_client`,
   rather than crashing or being silently misread as aggregation variables. Docstring now states
   the ambient-client contract.
2. **Hardened `start_dask_client`:** added `cap_numba_threads=1`, which runs
   `client.run(numba.set_num_threads, n)` on every worker (best-effort) so N workers × per-core
   numba threads don't oversubscribe. The safe "easy button" for local process parallelism.
3. Confirmed no `dask.compute(..., scheduler=...)` hard-codes a scheduler; `aggregate_space`/
   `aggregate_time` flow through the ambient scheduler. `is_distributed()`/`distributed_client()`
   seams already existed.
4. **Docs:** README "Execution & scaling" section — per-hardware recipes (single disk → default;
   fat node → `start_dask_client`; HPC → `dask_jobqueue`; cloud → object-store client) + the
   correctness invariant. `benchmarks/bench_read_scheduler.py` measures threads-vs-processes.

**Non-goals (held):** no hardware detection heuristic (HDD/SSD/Lustre/S3 or SLURM allocation
can't be reliably sniffed, and the user's client already encodes that knowledge); no new
execution abstraction beyond the ambient-client contract + the `start_dask_client` helper.

---

## 8. cftime-aware temporal bounds builder (non-standard calendars / CMIP6) — **✅ DONE (2026-07)**

> **Core.** `resample_groups` detects a `CFTimeIndex` and builds bounds from an xarray resample of
> a position array (datetime64 path untouched). Bug found & fixed: unlike pandas, xarray's cftime
> `.count()` fills empty bins with **NaN**, so the cftime branch zero-fills before the int64 cumsum.
>
> **Audit (this pass).** An end-to-end smoke test (synthetic 360_day cube + real georegions/weights
> → `aggregate_dataset`) revealed the surrounding pipeline is already calendar-agnostic: weights are
> grid-derived; `clean_dims`/`sortby`/`time_sel` string-indexing work on cftime; the load path
> (`dataset_from_path` → `open_dataset`) preserves the calendar (xarray auto-uses cftime); and the
> output panel carries the model-calendar cftime stamps (e.g. `2000-02-30`) — numba == dask
> throughout. **One real gap:** `groupby='week'` — cftime has NO weekly offset (`W`/`W-SUN`/`1W`
> all rejected by xarray, in *both* engines), so `execute` now raises a clear `NotImplementedError`
> instead of a cryptic "Invalid frequency string". `time_fix` (opt-in, ERA5-Land-specific) runs on
> cftime without crashing.
>
> **Policy = preserve by default** (works out of the box); conversion to a standard calendar is a
> user-side, opt-in `DataArray.convert_calendar(...)` (lossy, so never silent) — documented in the
> README "Calendars" section along with the week caveat.
>
> **Tests:** `test_cftime_resample_groups_bounds`, `test_cftime_numba_dask_parity`
> (mean/sum/max/dd/bins/sine_dd × 360_day/noleap × ±NaN), `test_cftime_empty_bin_parity`,
> `test_cftime_end_to_end_aggregate_dataset`, `test_cftime_week_groupby_raises`,
> `test_cftime_roundtrip_dataset_from_path`. Full suite green.

**Problem.** Climate-model output (CMIP6/CMIP5) frequently uses non-standard CF calendars —
`noleap`/`365_day` (never a Feb 29), `360_day` (every month 30 days; a valid "Feb 30"),
`all_leap`, `julian`. NumPy `datetime64` / pandas `DatetimeIndex` can only represent the real
proleptic-Gregorian calendar, so xarray loads these as **`cftime`** objects and
`da.get_index("time")` returns a **`CFTimeIndex`**. aggfly's numba engine builds its group
boundaries in `resample_groups` (`aggfly/aggregate/nb_kernels.py`) with
`pd.Series(1, index=tindex).resample(freq).count()`, and **pandas resample rejects a
CFTimeIndex** (`TypeError: Only valid with DatetimeIndex, TimedeltaIndex or PeriodIndex`).
So `engine="numba"` is dead on any non-standard-calendar dataset. (The dask path uses
`ds.resample(time=freq).reduce(...)`, which xarray *does* implement for cftime — so it likely
already works; this item makes the numba path match.)

**Verified design foundation (2026-07 scratch check).** On a `360_day` daily index,
pandas `.resample("ME")` raises as above; an **xarray resample of a position array works** and
yields clean contiguous bounds — `[30,30,30] -> bounds [0,30,60,90]`, labels a `CFTimeIndex`
including `2000-02-30`; `noleap` Jan/Feb gives `[31,28]`. Because this reuses the *same* xarray
resample the dask path uses, numba bounds align with the dask reduce **by construction**.

**Goal & scope.** Make the numba temporal path calendar-correct and bit-parity with dask on
cftime indices, across all reductions (mean/nanmean/sum/min/max/dd/bins/sine_dd) and groupbys
(date/month/year). Preserve the model calendar (don't silently convert). **In scope:** the
bounds builder + numba driver + a surrounding-pipeline audit for datetime64 assumptions +
synthetic-cftime tests. **Out of scope (explicit):** the scientific semantics of comparing a
360-day/noleap panel to a Gregorian one (that is the user's modeling decision, offered as an
opt-in `convert_calendar`, not forced); the full CMIP6 source adapter (separate item).

**Design.**
- `resample_groups(tindex, freq)`: **detect** cftime (`isinstance(tindex, xr.CFTimeIndex)`).
  - datetime64 branch → the existing pandas path, **unchanged** (keeps the hardcoded
    `test_aggregate_time_numba` values bit-for-bit; zero risk to the common case).
  - cftime branch → build bounds from `xr.DataArray(np.arange(n), coords={"time": tindex})
    .resample(time=freq).count()`: cumsum the per-bin counts (empty interior bins included as
    zero-width, matching the dask reindex), and return the resample bin labels (a CFTimeIndex)
    as `out_time`.
- `numba_resample`: unaffected in structure — it already assigns `out_time` as the output
  `time` coord; a CFTimeIndex coord is fine. Confirm `da_t.get_index("time")` yields the
  CFTimeIndex and the transpose/chunk steps are calendar-agnostic (they are — they touch axes,
  not values). The reduction kernels operate on *values*, never on time, so dd/bins/sine_dd are
  calendar-agnostic already.
- Keep the monotonic-increasing guard (works for CFTimeIndex too).

**Surrounding-pipeline audit (investigate, fix only what breaks).**
- `TemporalAggregator.execute` dask path — confirm `ds.resample(time=freq).reduce(func)` works
  on cftime (expected yes; establishes the parity oracle).
- `translate_groupby` freqs `1D`/`ME`/`YE`/`W` under cftime — verify `ME`/`YE` (month/year end)
  on 360_day/noleap; `W` (7-day weeks) is an edge case on 30-day months — flag, likely rare.
- `dataset_from_path` / `clean_dims` / `time_sel` / `time_fix` — audit for datetime64
  assumptions. `time_sel` string indexing works on CFTimeIndex; `time_fix` and any
  day-of-year/`.normalize()`-style math need checking. Ensure loading preserves cftime
  (`use_cftime`) for non-standard calendars.
- Output path — `aggregate_space` -> `to_dataframe()` yields a cftime "time" column (object
  dtype); the region merge is on `region_id`, so this is fine, but note downstream consumers
  get cftime stamps, not `Timestamp`s.

**Calendar detection & policy (adjacent, include).** Expose the detected calendar
(`ds.time.dt.calendar`) and two documented policies: **preserve** (default — route cftime
through the new path; faithful) and **convert** (opt-in `convert_calendar("standard",
align_on=...)`; lossy — drops Feb 29 / spreads 360->365; user picks `align_on`). The bounds
builder is what makes "preserve" viable on *both* engines.

**Testing (synthetic cftime cubes — no external CMIP6 files needed).**
- Fixtures via `xr.date_range(..., calendar=..., use_cftime=True)` for `360_day` and `noleap`.
- Parity: numba == dask (`equal_nan`) for mean/sum/dd/bins/sine_dd × groupby date/month/year,
  with and without NaN, on each calendar.
- Bounds hand-checks: 360_day daily→monthly = 30-day bins (`bounds` at multiples of 30);
  noleap Feb = 28; annual on 360_day = 360 steps.
- Empty-bin parity: a gapped cftime series → empty interior month → NaN group == dask.
- Regression: all existing datetime64 tests stay green (that path is untouched).
- Edge: a standard-calendar series far outside the datetime64 range (xarray loads it as cftime)
  routes through the cftime branch and still matches.

**Risks / notes.** Parity relies on xarray including empty interior bins in cftime resample the
same way it does for datetime64 — mitigated by building the numba bounds from that very
resample. `W`/week on 360_day is the one groupby to eyeball. No change to the datetime64 path,
so the common case and its hardcoded tests are unaffected.

**Deliverables.** Branch `feat/cftime-bounds`; `resample_groups` cftime branch (+ any audit
fixes) in `nb_kernels.py`/`temporal.py`; cftime parity tests in `test_aggregate.py`; this plan
item marked done; a short README note on calendar handling.
