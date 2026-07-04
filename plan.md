# aggfly backend acceleration — follow-up plan

Context: the `engine="numba"` temporal backend (`aggfly/aggregate/nb_kernels.py`) is
landed and bit-equivalent to the dask path. On real ERA5 it is ~11.8× faster than
dask on native/small spatial chunks (15.2s @ 9.3 GB vs 179s @ 12.4 GB) but ~0.7×
(slower) on large 250×250 chunks, because dask's vectorized numpy is already
compute-bound on big blocks and per-block numba threads oversubscribe the dask
thread pool. These three items build on that result.

---

## 1. Auto-select the temporal engine from chunk size

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
