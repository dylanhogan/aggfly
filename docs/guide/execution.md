# Execution & scaling

aggfly is **execution-backend-agnostic**: aggregation runs on whatever
[Dask](https://www.dask.org/) scheduler or distributed client is active in your
session, and falls back to Dask's threaded scheduler when none is. You choose the
backend to fit your hardware; **the results are identical across backends — only
the speed changes.**

This matters because the temporal pipeline is usually **read-bound**, and the
fastest way to read depends entirely on your storage. There is no universally best
default, so aggfly does not guess — it simply uses the client you provide.

## Two independent knobs

- **`engine=`** picks the *temporal kernel*: `"auto"` (default), `"numba"`, or
  `"dask"`.
- **The active Dask client** picks *how tasks execute* (threads, processes, or a
  cluster). Set it once before you aggregate.

## The temporal engine

The numba engine (`aggregate/nb_kernels.py`) replaces the per-group
`.resample(freq).reduce(func)` call with a single `nogil`, `parallel` compiled pass
per spatial chunk, covering `mean`/`nanmean`/`sum`/`min`/`max`/`dd`/`bins`/`sine_dd`.
It is **bit-equivalent** to the dask path.

It is dramatically faster **only on small/native spatial chunks** — the GIL-bound,
low-memory regime (~12× on native ERA5 chunks). On large rechunked blocks (e.g.
250×250) it is *slower*: dask's vectorized numpy is already compute-bound there, and
per-block numba threads oversubscribe the dask pool.

`"auto"` resolves this per step: numba when the largest spatial block is
≤ `NUMBA_MAX_CELLS_PER_BLOCK` (default `150*150`) cells, dask otherwise — and always
dask for calcs outside the numba-supported set. If you force `"numba"`, pair it with
native chunks, not a large `.chunk()`.

## Recipes by hardware

### Laptop / single disk (HDD or SSD) — do nothing

The default threaded scheduler is correct and needs no setup. On a single spinning
disk, sequential reads are actually optimal, so leaving concurrency low is the right
choice.

```python
df = af.aggregate_dataset(dataset=dataset, weights=weights, tavg=[...])   # threaded, zero config
```

### Fat single node (many cores, lots of RAM) — start a process cluster

Warm/cached reads are serialized by the GIL under the threaded scheduler; separate
worker processes read in parallel. Start a client first and aggfly will use it
automatically:

```python
client = af.start_dask_client(n_workers=16, threads_per_worker=1)
df = af.aggregate_dataset(dataset=dataset, weights=weights, tavg=[...])
af.shutdown_dask_client()
```

`start_dask_client` caps numba to one thread per worker by default
(`cap_numba_threads=1`) so `n_workers` × per-core numba threads don't oversubscribe
the machine — the numba kernels get their parallelism from Dask fanning spatial
blocks across workers.

> **Ordering constraint:** `weights.calculate_weights()` shuts down an active Dask
> client. Compute weights **before** starting your execution client.

### HPC (multi-node + parallel filesystem) — bring your own cluster

Use the standard [`dask-jobqueue`](https://jobqueue.dask.org/) tooling; aggfly needs
no HPC-specific configuration and does not depend on `dask-jobqueue`:

```python
from dask_jobqueue import SLURMCluster
from dask.distributed import Client

cluster = SLURMCluster(cores=16, memory="64GB", ...)
cluster.scale(jobs=8)
client = Client(cluster)
df = af.aggregate_dataset(dataset=dataset, weights=weights, tavg=[...])
```

### Cloud / object storage

Point `dataset_from_path` at an object-store-backed Zarr and use a distributed
client; the same pattern applies.

> **Note:** whether opening multiple files at once or using more worker processes
> *helps* depends on your storage serving parallel reads (SSD/NVMe, striped/parallel
> filesystems, and object stores benefit; a single spinning disk does not). Match the
> client to the hardware.

## From the CLI

The same choices are available in a config file via the `execution` block
(`backend: threads | processes | none`, `n_workers`, `threads_per_worker`) and
`aggregate.engine`, or as flags (`--backend`, `--n-workers`, `--engine`). See the
[CLI reference](../cli.md).
