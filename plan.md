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

## 3. Fix the `sine_dd` wrong-axis NaN masking bug (dask path)

**Goal:** correct a latent correctness bug in the existing dask degree-day code,
independent of the numba work.

**Bug:** in `aggfly/aggregate/temporal.py`, `_sine_cdd` and `_sine_hdd` compute
`nan_cells = da.where(np.isnan(frame[:, :, 0]), np.nan, 1)`. Within
`resample(...).reduce(func, axis=<time>)`, `frame` is `(time, lat, lon)` and the
time axis is the reduction axis — so `frame[:, :, 0]` indexes **longitude 0**, not
the time slice. The NaN mask is therefore built along the wrong axis and broadcast
incorrectly. In testing this mis-set ~27k cells (both directions: valid cells marked
NaN and NaN cells given numbers).

**Severity is worse than "wrong numbers": the dask path can outright CRASH.** When
the spatial chunk length differs from a resample group's timestep count, the
mis-shaped `nan_cells` fails to broadcast against the reduced `(lat, lon)` output —
observed as `ValueError: operands could not be broadcast together with shapes
(24, 16) (16, 16)` from `output = (output + case_2 + case_3) * nan_cells` in
`_sine_cdd`, for hourly data (24-step daily groups) on a 16-wide spatial chunk. So
`engine="dask"` + `sine_dd` is broken on realistic chunkings, while `engine="numba"`
+ `sine_dd` works. Fixing the mask both corrects the ~27k mis-set cells AND removes
the crash, making the two engines agree.

**Correct behavior:** a group's degree-day output should be NaN iff the group window
contains any NaN (the convention the numba kernel `_block_sine_dd` already uses; on
valid data the two paths agree to ~1e-15).

**Approach:**
- Replace the `frame[:, :, 0]` mask with a proper "any NaN along the time/reduction
  axis" test, e.g. `np.isnan(frame).any(axis=axis)`, applied to the reduced output.
  Mirror the fix in both `_sine_cdd` and `_sine_hdd` (and confirm `_sine_dd` /
  `_multi_sine_dd` inherit it).
- Confirm `tmax`/`tmin`/`tavg` already propagate NaN via `np.max/min/mean` over
  `axis`, so the explicit mask only needs to cover the same "any NaN in window" set.

**Validation:** add a test with a partial-NaN window (first timestep valid, a later
one NaN, and vice versa) asserting dask `sine_dd` == numba `sine_dd` and that both
yield NaN. This is the case the current code gets wrong.

**Risk/notes:** this changes existing outputs on NaN-containing (ocean/masked) cells
for `sine_dd` only — flag it as a bugfix in the changelog since it alters numbers for
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
fallback when a second copy of 1.7 TB is unaffordable. Deferred deliverable: a first-class
`af` helper "convert ERA5 year → Zarr". Two gotchas the helper must handle: (1) `to_zarr`
rejects the uneven dask chunks an `isel` window yields — `.chunk()` to uniform sizes first;
(2) kerchunk-loaded coords carry an un-copyable `_json.Scanner` in `.encoding` that crashes
`resample` — strip `da.encoding`/coord encodings after opening.

**Why pinned:** kerchunk needs `h5py` and a version compatible with aggfly's `zarr<3` pin
(`kerchunk==0.2.7`; ≥0.2.10 demands zarr≥3). Rather than special-case old kerchunk, we
first modernize the dependency stack (item 5), then re-evaluate to_zarr vs kerchunk vs
`virtualizarr` (the actively-maintained successor to kerchunk's reference approach) on
current zarr-v3 / xarray, and only then build the conversion helper.

---

## 5. Dependency + Python version modernization (NEXT)

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
1. **Baseline & inventory.** Record current resolved versions (`poetry show`), run the full
   `pytest` suite green as the reference, and grep the codebase for direct API touchpoints:
   `xarray`, `dask`, `dask_geopandas`, `zarr`, `numpy`, `geopandas`, `rioxarray`, `numba`.
2. **Confirm the joint solution (already resolved upstream — just verify).** The
   `dask-geopandas 0.5.0` + `dask>=2025.1.0` + `python>=3.10` combo removes the old blocker;
   confirm the exact versions poetry resolves and that the pair moves together. The real work
   this exposes is the `spatial.py` audit against modern dask's dataframe API (see step 4),
   not a version hunt.
3. **Widen Python + bump core stack in a throwaway env.** In a scratch venv, bump
   python, dask(+distributed), dask-geopandas, xarray, zarr(→v3), numpy(→2.x), numba(≥0.60),
   geopandas/rioxarray/pyproj. Resolve with poetry; capture the new lock.
4. **Fix breakage by subsystem.** zarr v3 API in `ProjectCache` + any `to_zarr`/`open_zarr`;
   numpy-2 dtype/`np.float_`-style removals; xarray resample/`.reduce` signature drift in
   `temporal.py`; dask-geopandas dataframe API in `spatial.py`. Re-run tests per subsystem.
5. **Re-benchmark the hot paths** on the new stack: `benchmarks/bench_engine.py` (numba vs
   dask, numpy-2) and the NUMBA_MAX_CELLS threshold; then re-run item 4's
   `profile_netcdf_zarr.py` with current kerchunk/virtualizarr on zarr-v3.
6. **Update pins & docs.** Rewrite the `pyproject.toml` version comments (they document the
   old 3.12.3 rationale) and the CLAUDE.md "Python version constraint" section to the new reality.

**Validation:** full `pytest` green on the new stack (the fixtures + `np.allclose` numeric
assertions are the correctness net); numba engine still bit-equivalent to dask; benchmarks
re-run and recorded. Land as its own branch/PR separate from feature work.

**Risk/notes:** with the dask-geopandas/Python blocker resolved upstream, the highest-risk
bumps are now **zarr v2→v3** and **numpy 1→2**, plus adapting `spatial.py` to modern dask's
dataframe API (the surface `dask-geopandas` 0.5.0 rides on). Keep the old `poetry.lock`
recoverable in case a transitive dep still forces a compromise.
