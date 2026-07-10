"""Top-level ``aggfly`` command group.

Milestone 1 wires the entry point and the ``info`` command. The remaining
commands (``validate``, ``weights``, ``run``) are added in later milestones;
until then they are declared as clear "not yet implemented" stubs so the command
surface and ``--help`` output are stable.
"""

import click

from . import config as config_mod
from . import info as info_cmd
from . import preprocess as preprocess_mod


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="aggfly", prog_name="aggfly")
def cli():
    """aggfly — spatial & temporal aggregation of gridded climate data.

    Typical authoring loop:

    \b
      aggfly info    DATA.zarr --var t2m   # inspect a dataset to write the config
      aggfly validate config.yaml          # dry-run the config (no data read)
      aggfly run     config.yaml           # aggregate → panel

    See CLI_PLAN.md for the config schema and the full command set.
    """


@cli.command()
@click.argument("path", type=str)
@click.option("--var", default=None, help="Restrict the report to this data variable.")
@click.option(
    "--storage-options",
    default=None,
    help="JSON dict passed to the storage backend (e.g. '{\"token\": \"anon\"}' for GCS).",
)
def info(path, var, storage_options):
    """Inspect a raster dataset (dims, calendar, lon convention, time span).

    Use this to discover the values you need to fill into a config file:
    ``xycoords``, ``timecoord``, ``lon_is_360``, and the units that determine
    ``preprocess``. PATH may be a local file/zarr or a remote URL.
    """
    info_cmd.run(path, var=var, storage_options=storage_options)


@cli.command()
@click.argument("config", type=click.Path())
@click.option(
    "--strict",
    is_flag=True,
    help="Treat unresolved input paths as errors (exit nonzero), not warnings.",
)
def validate(config, strict):
    """Statically check a config file without reading any data.

    Verifies the config's shape, the aggregation spec (calcs, groupby, ddargs),
    output format, and year expansion, then prints a normalized plan. Local input
    paths are checked for existence and reported as warnings (or errors with
    ``--strict``). Remote URLs are not fetched.
    """
    try:
        cfg = config_mod.load_config(config)
    except config_mod.ConfigError as e:
        click.echo("Config is invalid:", err=True)
        for msg in e.errors:
            click.echo(f"  - {msg}", err=True)
        raise SystemExit(1)

    # Resolve preprocess (safe expression / builtin / file escape hatch). The
    # file hatch imports the user's module, so this also confirms the function
    # actually exists — the same resolution `run` will do later.
    try:
        preprocess_mod.resolve_from_config(cfg)
    except preprocess_mod.PreprocessError as e:
        click.echo("Config is invalid:", err=True)
        click.echo(f"  - preprocess: {e}", err=True)
        raise SystemExit(1)

    warnings = config_mod.check_paths(cfg)
    click.echo(config_mod.describe(cfg))
    if warnings:
        click.echo("")
        stream_label = "Errors" if strict else "Warnings"
        click.echo(f"{stream_label}:", err=strict)
        for w in warnings:
            click.echo(f"  - {w}", err=strict)
        if strict:
            raise SystemExit(1)
    click.echo("\nConfig OK.")


@cli.command()
@click.argument("config", type=click.Path())
@click.option(
    "--project-dir", default=None, help="Override weights.project_dir (weight cache)."
)
@click.option("-v", "--verbose", is_flag=True, help="Print per-step progress.")
def weights(config, project_dir, verbose):
    """Build and cache spatial weights only, then exit.

    Weights depend only on the grid + regions (not the time series), so this
    precomputes them once; a later ``run`` with the same parameters reuses the
    cache under ``weights.project_dir``.
    """
    from . import pipeline as pipeline_mod

    try:
        cfg = config_mod.load_config(config)
    except config_mod.ConfigError as e:
        click.echo("Config is invalid:", err=True)
        for msg in e.errors:
            click.echo(f"  - {msg}", err=True)
        raise SystemExit(1)

    if project_dir is not None:
        cfg.project_dir = project_dir

    try:
        preprocess_mod.resolve_from_config(cfg)
    except preprocess_mod.PreprocessError as e:
        raise click.ClickException(f"preprocess: {e}")

    log = (lambda m: click.echo(m)) if verbose else (lambda m: None)
    try:
        w, _, _ = pipeline_mod.compute_weights(cfg, log=log)
    except Exception as e:
        if verbose:
            raise
        raise click.ClickException(f"{type(e).__name__}: {e}")

    n = len(w.weights)
    click.echo(f"Computed weights: {n} cell-region rows.")
    if cfg.project_dir:
        click.echo(f"Cached under: {cfg.project_dir}")
    else:
        click.echo(
            "No weights.project_dir set — weights were computed but not cached. "
            "Set weights.project_dir to persist and reuse them."
        )


@cli.command()
@click.argument("config", type=click.Path())
@click.option("-o", "--output", default=None, help="Override output.path from the config.")
@click.option(
    "--engine",
    type=click.Choice(sorted(config_mod.ALLOWED_ENGINE)),
    default=None,
    help="Override the temporal engine (auto/dask/numba).",
)
@click.option(
    "--years",
    default=None,
    help="Override years for a {year}-templated dataset path (e.g. 1980:1990).",
)
@click.option(
    "--project-dir", default=None, help="Override weights.project_dir (weight cache)."
)
@click.option(
    "--backend",
    type=click.Choice(sorted(config_mod.ALLOWED_BACKEND)),
    default=None,
    help="Override execution.backend (threads/processes/none).",
)
@click.option(
    "--n-workers", type=int, default=None, help="Override execution.n_workers."
)
@click.option("-v", "--verbose", is_flag=True, help="Print per-step progress.")
def run(config, output, engine, years, project_dir, backend, n_workers, verbose):
    """Run the full aggregation pipeline from a config file.

    Loads regions, builds (and caches) weights, aggregates every resolved
    dataset path over time and space, and writes the region-by-period panel to
    ``output.path``. Flags override the corresponding config fields.
    """
    from . import pipeline as pipeline_mod

    try:
        cfg = config_mod.load_config(config)
    except config_mod.ConfigError as e:
        click.echo("Config is invalid:", err=True)
        for msg in e.errors:
            click.echo(f"  - {msg}", err=True)
        raise SystemExit(1)

    # Apply flag overrides before resolving/running.
    if output is not None:
        cfg.output_path = output
        ext = output.rsplit(".", 1)[-1].lower() if "." in output else ""
        cfg.output_format = {"pq": "parquet"}.get(ext, ext) or cfg.output_format
    if engine is not None:
        cfg.engine = engine
    if project_dir is not None:
        cfg.project_dir = project_dir
    if backend is not None:
        cfg.backend = backend
    if n_workers is not None:
        cfg.n_workers = n_workers
    if years is not None:
        errs = []
        cfg.years = config_mod._parse_years(years, errs)
        if errs:
            raise click.ClickException("; ".join(errs))

    try:
        preprocess_mod.resolve_from_config(cfg)
    except preprocess_mod.PreprocessError as e:
        raise click.ClickException(f"preprocess: {e}")

    log = (lambda m: click.echo(m)) if verbose else (lambda m: None)
    try:
        df = pipeline_mod.run_pipeline(cfg, log=log)
    except Exception as e:
        if verbose:
            raise
        raise click.ClickException(f"{type(e).__name__}: {e}")

    pipeline_mod.write_output(df, cfg.output_path, cfg.output_format)
    click.echo(f"Wrote {len(df)} rows to {cfg.output_path} ({cfg.output_format}).")


if __name__ == "__main__":
    cli()
