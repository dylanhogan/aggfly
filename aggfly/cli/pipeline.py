"""Orchestrate the aggfly pipeline from a validated RunConfig.

This is the layer that finally calls the real ``af.*`` functions. It maps config
fields onto ``georegions_from_path`` / ``dataset_from_path`` /
``weights_from_objects`` / ``aggregate_dataset`` — i.e. it reproduces exactly the
script a user would hand-write, which is what the parity test pins down.

Milestone 4 runs on whatever Dask scheduler is active (threads by default). The
execution-backend client wiring (and the weights-before-client ordering it
requires) lands in milestone 5.
"""

import os

import pandas as pd

import aggfly as af

from . import preprocess as preprocess_mod


def build_regions(config):
    return af.georegions_from_path(
        config.regions_path, config.regionid, config.region_list
    )


def load_dataset(config, path, georegions):
    """Load one dataset file, applying the config's coord names and preprocess.

    ``georegions`` is passed to ``dataset_from_path`` only when
    ``clip_to_regions`` is set — it clips the raster to the regions' extent
    (a read-reduction optimization). Weights still use the full georegions, so
    clipping never changes results; disable it for regions that wrap the
    antimeridian in the 0–360 convention.
    """
    kwargs = {}
    if config.chunks is not None:
        kwargs["chunks"] = config.chunks
    return af.dataset_from_path(
        path,
        var=config.var,
        xycoords=config.xycoords,
        timecoord=config.timecoord,
        time_sel=config.time_sel,
        georegions=georegions if config.clip_to_regions else None,
        lon_is_360=config.lon_is_360,
        preprocess=preprocess_mod.resolve_from_config(config),
        name=config.var,
        **kwargs,
    )


def build_secondary(config):
    """Construct the optional secondary-weights object from the config."""
    s = config.secondary
    if s is None:
        return None
    if s.type == "pop":
        return af.pop_weights_from_path(
            s.path, feed=s.feed, project_dir=config.project_dir
        )
    if s.type == "crop":
        return af.crop_weights_from_path(
            s.path, crop=s.crop or "corn", feed=s.feed, project_dir=config.project_dir
        )
    return af.secondary_weights_from_path(s.path, project_dir=config.project_dir)


def build_weights(config, dataset, georegions):
    """Build and populate GridWeights (area + optional secondary), cached by project_dir."""
    secondary = build_secondary(config)
    w = af.weights_from_objects(
        dataset,
        georegions,
        secondary_weights=secondary,
        project_dir=config.project_dir,
    )
    w.calculate_weights()
    return w


def compute_weights(config, log=lambda m: None):
    """Load regions + a sample layer and build the (cached) weights.

    Returns ``(weights, georegions, sample)`` so callers can reuse the sample.
    Must run with NO active Dask client — ``calculate_weights`` tears one down.
    """
    log(f"Loading regions: {config.regions_path}")
    georegions = build_regions(config)
    path0 = config.resolved_paths()[0]
    log(f"Building weights from sample layer: {path0}")
    sample = load_dataset(config, path0, georegions)
    weights = build_weights(config, sample, georegions)
    return weights, georegions, sample


def _start_execution_client(config, log):
    """Start the execution client for the config's backend, or None for threads.

    Called *after* weights are built: ``calculate_weights`` shuts down any active
    client, so starting one earlier would have it torn down mid-run.
    """
    if config.backend == "processes":
        log(
            f"Starting Dask process cluster "
            f"(n_workers={config.n_workers}, threads_per_worker={config.threads_per_worker})"
        )
        return af.start_dask_client(
            n_workers=config.n_workers,
            threads_per_worker=config.threads_per_worker,
        )
    # "threads" (default scheduler) and "none" (bring-your-own) start nothing.
    return None


def run_pipeline(config, log=lambda m: None):
    """Execute regions → weights → aggregate for every resolved path; return the panel.

    ``log`` is an optional progress callback (the CLI passes ``click.echo``).
    Ordering is deliberate: weights first (no client), then start the execution
    client, then aggregate, then shut the client down.
    """
    weights, georegions, sample = compute_weights(config, log)
    paths = config.resolved_paths()

    client = _start_execution_client(config, log)
    try:
        aggregator_dict = config.to_aggregator_dict()
        frames = []
        for i, path in enumerate(paths):
            log(f"Aggregating [{i + 1}/{len(paths)}]: {path}")
            # Reuse the already-loaded sample for the first path
            # (weights_from_objects deep-copies internally, so it is unmodified).
            ds = sample if i == 0 else load_dataset(config, path, georegions)
            df = af.aggregate_dataset(
                dataset=ds,
                weights=weights,
                aggregator_dict=aggregator_dict,
                engine=config.engine,
            )
            frames.append(df)
        result = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    finally:
        if client is not None:
            log("Shutting down Dask client")
            af.shutdown_dask_client()

    return result


def write_output(df, path, fmt):
    """Write the panel to ``path`` in the requested format (parquet/feather/csv)."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(path, index=False)
    elif fmt == "feather":
        # feather requires a default RangeIndex
        df.reset_index(drop=True).to_feather(path)
    elif fmt == "csv":
        df.to_csv(path, index=False)
    else:  # pragma: no cover - guarded by config validation
        raise ValueError(f"unsupported output format: {fmt}")
