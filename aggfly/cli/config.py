"""Config loading, validation, and normalization for the aggfly CLI.

A config file is YAML that mirrors the four pipeline stages plus job control (see
``internal/cli-plan.md``). This module is deliberately **pure** — it parses and validates
the config and normalizes it into typed dataclasses, but does no I/O against the
climate data and imports nothing from dask. That keeps ``aggfly validate`` fast
and the whole layer unit-testable without fixtures.

The one non-obvious normalization: a ``transform`` step's ``exp`` must be handed
to the library as a NumPy array, not a plain list — ``transform_dataset`` wraps
non-list values and then indexes ``[0]``, so a bare list ``[1, 2]`` would be
mis-parsed as the scalar ``1``. ``to_aggregator_dict`` converts it.
"""

import glob
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

# ---- accepted values (kept in sync with aggregate/temporal.py) ---------------
ALLOWED_CALCS = {"mean", "nanmean", "sum", "min", "max", "dd", "bins", "sine_dd"}
CALCS_NEEDING_DDARGS = {"dd", "bins", "sine_dd"}
ALLOWED_GROUPBY = {"date", "month", "year", "week"}
ALLOWED_ENGINE = {"auto", "dask", "numba"}
ALLOWED_BACKEND = {"threads", "processes", "none"}
ALLOWED_FORMAT = {"parquet", "feather", "csv"}
ALLOWED_SECONDARY = {"pop", "crop", "generic"}
ALLOWED_STEP_TYPES = {"aggregate", "transform"}


class ConfigError(Exception):
    """Raised when a config fails validation. Carries a list of messages."""

    def __init__(self, errors):
        self.errors = list(errors)
        super().__init__("\n".join(f"- {e}" for e in self.errors))


# ---- typed config structures -------------------------------------------------
@dataclass
class SecondaryWeightsConfig:
    type: str
    path: str
    crop: Optional[str] = None
    feed: Optional[str] = None


@dataclass
class RunConfig:
    # regions
    regions_path: str
    regionid: str
    region_list: Optional[List[str]]
    # dataset
    dataset_path: str
    var: str
    preprocess: Optional[str]
    preprocess_from: Optional[str]
    lon_is_360: bool
    timecoord: str
    xycoords: Tuple[str, str]
    time_sel: Optional[str]
    chunks: Optional[Dict[str, object]]
    clip_to_regions: bool
    # weights
    project_dir: Optional[str]
    secondary: Optional[SecondaryWeightsConfig]
    # aggregate
    engine: str
    variables: Dict[str, List]
    # job control
    years: Optional[List[int]]
    backend: str
    n_workers: int
    threads_per_worker: int
    output_path: str
    output_format: str

    @property
    def templated(self) -> bool:
        """True when the dataset path contains a ``{year}`` placeholder."""
        return "{year}" in self.dataset_path

    def resolved_paths(self) -> List[str]:
        """The concrete dataset path(s) after substituting years (if templated)."""
        if not self.templated:
            return [self.dataset_path]
        years = self.years or []
        return [self.dataset_path.format(year=y) for y in years]

    def to_aggregator_dict(self) -> Dict[str, List]:
        """Normalize ``variables`` into the ``aggregator_dict`` aggregate_dataset wants.

        Converts each ``transform`` step's ``exp`` to a NumPy array so the
        library's ``[0]`` indexing behaves (see module docstring).
        """
        out = {}
        for name, steps in self.variables.items():
            norm_steps = []
            for step_type, params in steps:
                params = dict(params)
                if step_type == "transform" and "exp" in params:
                    params["exp"] = np.array(params["exp"])
                norm_steps.append((step_type, params))
            out[name] = norm_steps
        return out


# ---- parsing helpers ---------------------------------------------------------
def _parse_years(spec, errors):
    """Expand a years spec (``"1980:1990"`` inclusive | list | int | None)."""
    if spec is None:
        return None
    if isinstance(spec, bool):  # guard: YAML `true` is an int subclass
        errors.append("years: must be a range 'start:end', a list, or an int")
        return None
    if isinstance(spec, int):
        return [spec]
    if isinstance(spec, list):
        try:
            return [int(y) for y in spec]
        except (TypeError, ValueError):
            errors.append(f"years: list must contain integers, got {spec!r}")
            return None
    if isinstance(spec, str):
        try:
            if ":" in spec:
                a, b = spec.split(":")
                return list(range(int(a), int(b) + 1))
            return [int(spec)]
        except ValueError:
            errors.append(f"years: could not parse {spec!r} (use 'start:end' or an int)")
            return None
    errors.append(f"years: unsupported type {type(spec).__name__}")
    return None


def _validate_steps(name, steps, errors):
    """Validate one variable's list of pipeline steps; record any problems."""
    if not isinstance(steps, list) or not steps:
        errors.append(f"aggregate.variables.{name}: must be a non-empty list of steps")
        return
    for i, step in enumerate(steps):
        loc = f"aggregate.variables.{name}[{i}]"
        if not (isinstance(step, (list, tuple)) and len(step) == 2):
            errors.append(f"{loc}: each step must be [step_type, params]")
            continue
        step_type, params = step
        if step_type not in ALLOWED_STEP_TYPES:
            errors.append(
                f"{loc}: unknown step type {step_type!r} "
                f"(expected one of {sorted(ALLOWED_STEP_TYPES)})"
            )
            continue
        if not isinstance(params, dict):
            errors.append(f"{loc}: params must be a mapping")
            continue
        if step_type == "aggregate":
            _validate_aggregate_step(loc, params, errors)
        else:
            _validate_transform_step(loc, params, errors)


def _validate_aggregate_step(loc, params, errors):
    calc = params.get("calc")
    groupby = params.get("groupby")
    if calc not in ALLOWED_CALCS:
        errors.append(f"{loc}: calc {calc!r} not in {sorted(ALLOWED_CALCS)}")
    if groupby not in ALLOWED_GROUPBY:
        errors.append(f"{loc}: groupby {groupby!r} not in {sorted(ALLOWED_GROUPBY)}")
    if calc in CALCS_NEEDING_DDARGS:
        dd = params.get("ddargs")
        if not isinstance(dd, list) or not dd:
            errors.append(f"{loc}: calc {calc!r} requires a non-empty 'ddargs' list")


def _validate_transform_step(loc, params, errors):
    kind = params.get("transform")
    has_exp = "exp" in params
    has_inter = "inter" in params
    is_spline = kind == "spline" or "spline" in params
    if not (has_exp or has_inter or is_spline):
        errors.append(
            f"{loc}: transform step needs one of 'exp' (power), 'inter', or "
            "transform: spline"
        )
    if has_exp and not isinstance(params["exp"], (list, int)):
        errors.append(f"{loc}: 'exp' must be an int or a list of ints")


def _multiplicity(steps):
    """Rough count of output datasets a variable's steps produce, for the multi-dd guard."""
    n = 1
    for step_type, params in steps:
        if step_type == "transform" and "exp" in params:
            exp = params["exp"]
            n = len(exp) if isinstance(exp, list) else 1
        if step_type == "aggregate" and params.get("calc") in CALCS_NEEDING_DDARGS:
            dd = params.get("ddargs")
            # a list of triples (2-D) is a multi-dd fan-out
            is_multi = isinstance(dd, list) and dd and isinstance(dd[0], list)
            if is_multi and n > 1:
                return "conflict"
    return n


def parse_config(raw) -> RunConfig:
    """Validate ``raw`` (a parsed YAML mapping) and build a RunConfig.

    Raises ConfigError with every problem found (not just the first).
    """
    errors: List[str] = []
    if raw is None or not isinstance(raw, dict):
        raise ConfigError(["config must be a non-empty YAML mapping"])

    def section(key):
        val = raw.get(key)
        if val is None:
            return {}
        if not isinstance(val, dict):
            errors.append(f"{key}: must be a mapping")
            return {}
        return val

    regions = section("regions")
    dataset = section("dataset")
    weights = section("weights")
    aggregate = section("aggregate")
    execution = section("execution")
    output = section("output")

    # regions
    regions_path = regions.get("path")
    regionid = regions.get("regionid")
    if not regions_path:
        errors.append("regions.path is required")
    if not regionid:
        errors.append("regions.regionid is required")

    # dataset
    dataset_path = dataset.get("path")
    var = dataset.get("var")
    if not dataset_path:
        errors.append("dataset.path is required")
    if not var:
        errors.append("dataset.var is required")
    preprocess = dataset.get("preprocess")
    preprocess_from = dataset.get("preprocess_from")
    if preprocess is not None and preprocess_from is not None:
        errors.append(
            "dataset: set at most one of 'preprocess' and 'preprocess_from'"
        )
    if preprocess_from is not None and ":" not in str(preprocess_from):
        errors.append("dataset.preprocess_from must be 'path/to/file.py:function'")
    xycoords = dataset.get("xycoords", ["longitude", "latitude"])
    if not (isinstance(xycoords, list) and len(xycoords) == 2):
        errors.append("dataset.xycoords must be a 2-item list [lon_name, lat_name]")
        xycoords = ["longitude", "latitude"]

    # weights
    project_dir = weights.get("project_dir")
    secondary_raw = weights.get("secondary")
    secondary = None
    if secondary_raw is not None:
        if not isinstance(secondary_raw, dict):
            errors.append("weights.secondary must be a mapping")
        else:
            stype = secondary_raw.get("type")
            spath = secondary_raw.get("path")
            if stype not in ALLOWED_SECONDARY:
                errors.append(
                    f"weights.secondary.type {stype!r} not in {sorted(ALLOWED_SECONDARY)}"
                )
            if not spath:
                errors.append("weights.secondary.path is required")
            secondary = SecondaryWeightsConfig(
                type=stype,
                path=spath,
                crop=secondary_raw.get("crop"),
                feed=secondary_raw.get("feed"),
            )

    # aggregate
    engine = aggregate.get("engine", "auto")
    if engine not in ALLOWED_ENGINE:
        errors.append(f"aggregate.engine {engine!r} not in {sorted(ALLOWED_ENGINE)}")
    variables = aggregate.get("variables")
    if not isinstance(variables, dict) or not variables:
        errors.append("aggregate.variables must be a non-empty mapping of name -> steps")
        variables = {}
    else:
        for name, steps in variables.items():
            _validate_steps(name, steps, errors)
            if _multiplicity(steps) == "conflict":
                errors.append(
                    f"aggregate.variables.{name}: cannot combine a multi-'ddargs' "
                    "(bins) step with a multi-output transform (e.g. multiple "
                    "exponents) — the library rejects this at runtime"
                )

    # job control
    years = _parse_years(raw.get("years"), errors)
    backend = execution.get("backend", "threads")
    if backend not in ALLOWED_BACKEND:
        errors.append(f"execution.backend {backend!r} not in {sorted(ALLOWED_BACKEND)}")
    n_workers = execution.get("n_workers", 1)
    threads_per_worker = execution.get("threads_per_worker", 1)

    # output
    output_path = output.get("path")
    if not output_path:
        errors.append("output.path is required")
    output_format = output.get("format")
    if output_format is None and output_path:
        ext = os.path.splitext(str(output_path))[1].lstrip(".").lower()
        output_format = {"pq": "parquet"}.get(ext, ext)
    if output_format not in ALLOWED_FORMAT:
        errors.append(
            f"output.format {output_format!r} not in {sorted(ALLOWED_FORMAT)} "
            "(set output.format or use a .parquet/.feather/.csv extension)"
        )

    # cross-cutting: templated path needs years
    if dataset_path and "{year}" in str(dataset_path) and not years:
        errors.append(
            "dataset.path contains '{year}' but no 'years' were given "
            "(add years: 'start:end')"
        )

    if errors:
        raise ConfigError(errors)

    return RunConfig(
        regions_path=regions_path,
        regionid=regionid,
        region_list=regions.get("region_list"),
        dataset_path=dataset_path,
        var=var,
        preprocess=preprocess,
        preprocess_from=preprocess_from,
        lon_is_360=bool(dataset.get("lon_is_360", True)),
        timecoord=dataset.get("timecoord", "time"),
        xycoords=(xycoords[0], xycoords[1]),
        time_sel=dataset.get("time_sel"),
        chunks=dataset.get("chunks"),
        clip_to_regions=bool(dataset.get("clip_to_regions", True)),
        project_dir=project_dir,
        secondary=secondary,
        engine=engine,
        variables=variables,
        years=years,
        backend=backend,
        n_workers=int(n_workers),
        threads_per_worker=int(threads_per_worker),
        output_path=output_path,
        output_format=output_format,
    )


def load_config(path) -> RunConfig:
    """Read and validate a YAML config file into a RunConfig."""
    try:
        with open(path) as f:
            raw = yaml.safe_load(f)
    except FileNotFoundError:
        raise ConfigError([f"config file not found: {path}"])
    except yaml.YAMLError as e:
        raise ConfigError([f"could not parse YAML: {e}"])
    return parse_config(raw)


# ---- reporting ---------------------------------------------------------------
def _is_remote(path) -> bool:
    return "://" in str(path)


def check_paths(config: RunConfig) -> List[str]:
    """Return warnings for local input paths that don't resolve (skips remote URLs)."""
    warnings = []
    if not _is_remote(config.regions_path) and not os.path.exists(config.regions_path):
        warnings.append(f"regions.path does not exist: {config.regions_path}")
    for p in config.resolved_paths():
        if _is_remote(p):
            continue
        if not glob.glob(p) and not os.path.exists(p):
            warnings.append(f"dataset.path does not resolve: {p}")
    if config.secondary is not None and not _is_remote(config.secondary.path):
        if not os.path.exists(config.secondary.path):
            warnings.append(
                f"weights.secondary.path does not exist: {config.secondary.path}"
            )
    return warnings


def describe(config: RunConfig) -> str:
    """A human-readable normalized plan for the config (used by `validate`)."""
    lines = []
    lines.append("Normalized plan")
    lines.append(f"  regions   : {config.regions_path}  (id column: {config.regionid})")
    lines.append(f"  dataset   : {config.dataset_path}  var={config.var}")
    lines.append(
        f"              lon_is_360={config.lon_is_360} "
        f"timecoord={config.timecoord} xycoords={list(config.xycoords)}"
    )
    if config.preprocess:
        lines.append(f"              preprocess: {config.preprocess}")
    elif config.preprocess_from:
        lines.append(f"              preprocess_from: {config.preprocess_from}")
    if config.templated:
        yrs = config.years or []
        span = f"{yrs[0]}..{yrs[-1]} ({len(yrs)} files)" if yrs else "(none)"
        lines.append(f"  years     : {span}")
    if config.secondary is not None:
        lines.append(
            f"  weights   : {config.secondary.type} secondary ({config.secondary.path})"
        )
    else:
        lines.append("  weights   : area-only")
    lines.append(f"  engine    : {config.engine}   backend: {config.backend}")
    lines.append(f"  output    : {config.output_path}  ({config.output_format})")
    lines.append(f"  variables : {len(config.variables)}")
    for name, steps in config.variables.items():
        summary = " -> ".join(
            f"{st}:{params.get('calc') or params.get('transform') or '?'}"
            + (f"@{params['groupby']}" if params.get("groupby") else "")
            for st, params in steps
        )
        lines.append(f"    - {name}: {summary}")
    return "\n".join(lines)
