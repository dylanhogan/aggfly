"""Tests for the aggfly command-line interface (milestone 1: skeleton + info).

These use click's CliRunner and small synthetic zarr stores written to tmp_path,
so they need no external data files (consistent with the rest of the suite).
"""

import os

import numpy as np
import pytest
import xarray as xr
from click.testing import CliRunner

from aggfly.cli import config as cfgmod
from aggfly.cli.main import cli

# ---------------------------------------------------------------------------
# a minimal valid config used across the config-layer tests
# ---------------------------------------------------------------------------
GOOD_CONFIG = {
    "regions": {"path": "counties.shp", "regionid": "fips"},
    "dataset": {"path": "era5_{year}.zarr", "var": "t2m", "preprocess": "x - 273.15"},
    "weights": {"project_dir": "./proj"},
    "aggregate": {
        "engine": "auto",
        "variables": {
            "tavg": [
                ["aggregate", {"calc": "mean", "groupby": "date"}],
                ["transform", {"transform": "power", "exp": [1, 2]}],
                ["aggregate", {"calc": "sum", "groupby": "year"}],
            ],
            "gdd": [
                ["aggregate", {"calc": "dd", "groupby": "date", "ddargs": [10, 30, 0]}],
                ["aggregate", {"calc": "sum", "groupby": "year"}],
            ],
        },
    },
    "years": "1980:1982",
    "execution": {"backend": "processes", "n_workers": 8},
    "output": {"path": "out/panel.parquet"},
}


def _write_zarr(path, calendar="standard", lon_360=True, var="t2m", units="K"):
    """Write a tiny 3-D (time, lat, lon) cube to ``path`` for info to inspect."""
    use_cftime = calendar not in ("standard", "proleptic_gregorian")
    time = xr.date_range(
        "2000-01-01", periods=6, freq="D", calendar=calendar, use_cftime=use_cftime
    )
    lon = np.array([0.0, 90.0, 270.0]) if lon_360 else np.array([-180.0, -90.0, 90.0])
    lat = np.array([-45.0, 0.0, 45.0])
    da = xr.DataArray(
        np.random.rand(len(time), len(lat), len(lon)).astype("float32"),
        coords={"time": time, "latitude": lat, "longitude": lon},
        dims=("time", "latitude", "longitude"),
        name=var,
    )
    if units:
        da.attrs["units"] = units
    da.to_dataset().chunk({"time": 3}).to_zarr(path)
    return str(path)


def test_cli_help_lists_commands():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    for cmd in ("info", "validate", "weights", "run"):
        assert cmd in result.output


def test_cli_version():
    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "aggfly" in result.output


def test_info_datetime64_era5_like(tmp_path):
    path = _write_zarr(tmp_path / "era5.zarr", calendar="standard", lon_360=True)
    result = CliRunner().invoke(cli, ["info", path])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "xycoords   : [longitude, latitude]" in out
    assert "lon_is_360: true" in out
    assert "timecoord  : time" in out
    assert "units  : K" in out
    # standard calendar must NOT be flagged as cftime
    assert "cftime" not in out


def test_info_cftime_360day_cmip6_like(tmp_path):
    path = _write_zarr(
        tmp_path / "cmip6.zarr", calendar="360_day", lon_360=False, var="tas", units=""
    )
    result = CliRunner().invoke(cli, ["info", path, "--var", "tas"])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "calendar   : 360_day" in out
    assert "cftime / non-standard" in out
    assert "lon_is_360: false" in out


def test_info_missing_var_errors(tmp_path):
    path = _write_zarr(tmp_path / "era5.zarr")
    result = CliRunner().invoke(cli, ["info", path, "--var", "nope"])
    assert result.exit_code != 0
    assert "not found" in result.output


def test_info_bad_storage_options_json(tmp_path):
    path = _write_zarr(tmp_path / "era5.zarr")
    result = CliRunner().invoke(
        cli, ["info", path, "--storage-options", "{not json}"]
    )
    assert result.exit_code != 0
    assert "not valid JSON" in result.output


# ===========================================================================
# Milestone 2: config layer + validate
# ===========================================================================
import copy

import yaml


def _cfg(**overrides):
    c = copy.deepcopy(GOOD_CONFIG)
    for k, v in overrides.items():
        c[k] = v
    return c


def test_parse_good_config():
    cfg = cfgmod.parse_config(GOOD_CONFIG)
    assert cfg.regionid == "fips"
    assert cfg.var == "t2m"
    assert cfg.templated is True
    assert cfg.years == [1980, 1981, 1982]
    assert cfg.resolved_paths() == [
        "era5_1980.zarr",
        "era5_1981.zarr",
        "era5_1982.zarr",
    ]
    assert cfg.backend == "processes"
    assert cfg.output_format == "parquet"  # inferred from extension
    assert cfg.secondary is None


def test_to_aggregator_dict_converts_exp_to_array():
    cfg = cfgmod.parse_config(GOOD_CONFIG)
    agg = cfg.to_aggregator_dict()
    # find the transform step in tavg
    transform_step = next(s for s in agg["tavg"] if s[0] == "transform")
    exp = transform_step[1]["exp"]
    assert isinstance(exp, np.ndarray)
    assert list(exp) == [1, 2]


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("1980:1983", [1980, 1981, 1982, 1983]),
        ([1990, 1995], [1990, 1995]),
        (2001, [2001]),
        (None, None),
    ],
)
def test_year_expansion(spec, expected):
    errors = []
    assert cfgmod._parse_years(spec, errors) == expected
    assert errors == []


def test_bad_config_reports_all_errors():
    bad = {
        "regions": {"path": "c.shp"},  # missing regionid
        "dataset": {
            "path": "d_{year}.zarr",
            "var": "t2m",
            "preprocess": "x-1",
            "preprocess_from": "p.py:f",  # mutually exclusive
        },
        "aggregate": {
            "engine": "turbo",  # bad engine
            "variables": {
                "v": [["aggregate", {"calc": "avg", "groupby": "fortnight"}]]
            },
        },
        "output": {"path": "o.xlsx"},  # bad format
        # no years despite {year}
    }
    with pytest.raises(cfgmod.ConfigError) as exc:
        cfgmod.parse_config(bad)
    joined = "\n".join(exc.value.errors)
    assert "regionid is required" in joined
    assert "at most one of 'preprocess'" in joined
    assert "engine 'turbo'" in joined
    assert "calc 'avg'" in joined
    assert "groupby 'fortnight'" in joined
    assert "no 'years'" in joined
    assert "xlsx" in joined


def test_multi_dd_times_multi_exp_conflict():
    c = _cfg(
        aggregate={
            "engine": "auto",
            "variables": {
                "x": [
                    ["transform", {"transform": "power", "exp": [1, 2]}],
                    [
                        "aggregate",
                        {"calc": "bins", "groupby": "year",
                         "ddargs": [[25, 99, 0], [30, 99, 0]]},
                    ],
                ]
            },
        }
    )
    with pytest.raises(cfgmod.ConfigError) as exc:
        cfgmod.parse_config(c)
    assert any("cannot combine" in e for e in exc.value.errors)


def test_secondary_weights_parsed():
    c = _cfg(weights={"project_dir": "./p",
                      "secondary": {"type": "pop", "path": "landscan.tif"}})
    cfg = cfgmod.parse_config(c)
    assert cfg.secondary.type == "pop"
    assert cfg.secondary.path == "landscan.tif"


def test_dd_step_missing_ddargs_errors():
    c = _cfg(
        aggregate={
            "engine": "auto",
            "variables": {"x": [["aggregate", {"calc": "dd", "groupby": "date"}]]},
        }
    )
    with pytest.raises(cfgmod.ConfigError) as exc:
        cfgmod.parse_config(c)
    assert any("requires a non-empty 'ddargs'" in e for e in exc.value.errors)


def test_validate_command_good(tmp_path):
    p = tmp_path / "c.yaml"
    # non-templated path so there is no missing-file noise
    c = _cfg(dataset={"path": "d.zarr", "var": "t2m"}, years=None)
    p.write_text(yaml.safe_dump(c))
    result = CliRunner().invoke(cli, ["validate", str(p)])
    assert result.exit_code == 0, result.output
    assert "Normalized plan" in result.output
    assert "Config OK." in result.output


def test_validate_command_bad_exits_nonzero(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"regions": {"path": "x"}}))
    result = CliRunner().invoke(cli, ["validate", str(p)])
    assert result.exit_code != 0
    assert "invalid" in result.output.lower()


def test_validate_strict_turns_missing_paths_into_errors(tmp_path):
    p = tmp_path / "c.yaml"
    c = _cfg(dataset={"path": "nope.zarr", "var": "t2m"}, years=None)
    p.write_text(yaml.safe_dump(c))
    ok = CliRunner().invoke(cli, ["validate", str(p)])
    assert ok.exit_code == 0  # missing path is only a warning by default
    strict = CliRunner().invoke(cli, ["validate", str(p), "--strict"])
    assert strict.exit_code != 0  # ... but an error under --strict


# ===========================================================================
# Milestone 3: preprocess resolver
# ===========================================================================
from aggfly.cli import preprocess as ppmod


def test_preprocess_builtins():
    k2c = ppmod.resolve("kelvin_to_celsius")
    assert np.allclose(k2c(np.array([273.15, 283.15])), [0.0, 10.0])
    assert ppmod.resolve("identity")(np.array([1.0, 2.0])).tolist() == [1.0, 2.0]


@pytest.mark.parametrize(
    "expr,inp,expected",
    [
        ("x - 273.15", [273.15, 283.15], [0.0, 10.0]),
        ("(x - 32) * 5 / 9", [32.0, 212.0], [0.0, 100.0]),
        ("x ** 2", [2.0, 3.0], [4.0, 9.0]),
        ("-x", [1.0, -2.0], [-1.0, 2.0]),
        ("x * 0.1 + 5", [10.0, 20.0], [6.0, 7.0]),
    ],
)
def test_preprocess_safe_expressions(expr, inp, expected):
    f = ppmod.resolve(expr)
    assert np.allclose(f(np.array(inp)), expected)


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('echo hi')",  # Call
        "x.values",                              # Attribute
        "x[0]",                                  # Subscript
        "y + 1",                                 # foreign name
        "os",                                    # bare identifier / unknown builtin
        "1 + 2",                                 # does not reference x
    ],
)
def test_preprocess_rejects_unsafe(expr):
    with pytest.raises(ppmod.PreprocessError):
        ppmod.resolve(expr)


def test_preprocess_none_and_mutual_exclusion():
    assert ppmod.resolve(None, None) is None
    with pytest.raises(ppmod.PreprocessError):
        ppmod.resolve("x - 1", "prep.py:f")


def test_preprocess_from_file(tmp_path):
    mod = tmp_path / "prep.py"
    mod.write_text("def clean(x):\n    return x - 273.15\n")
    f = ppmod.resolve(None, f"{mod}:clean")
    assert np.allclose(f(np.array([273.15, 300.15])), [0.0, 27.0])


def test_preprocess_from_missing_function(tmp_path):
    mod = tmp_path / "prep.py"
    mod.write_text("def other(x):\n    return x\n")
    with pytest.raises(ppmod.PreprocessError) as exc:
        ppmod.resolve(None, f"{mod}:clean")
    assert "not found" in str(exc.value)


def test_preprocess_from_missing_file():
    with pytest.raises(ppmod.PreprocessError) as exc:
        ppmod.resolve(None, "/no/such/file.py:clean")
    assert "not found" in str(exc.value)


def test_validate_catches_bad_expression(tmp_path):
    p = tmp_path / "c.yaml"
    c = _cfg(
        dataset={"path": "d.zarr", "var": "t2m", "preprocess": "x.values"},
        years=None,
    )
    p.write_text(yaml.safe_dump(c))
    result = CliRunner().invoke(cli, ["validate", str(p)])
    assert result.exit_code != 0
    assert "preprocess" in result.output.lower()


def test_validate_accepts_builtin_preprocess(tmp_path):
    p = tmp_path / "c.yaml"
    c = _cfg(
        dataset={"path": "d.zarr", "var": "t2m", "preprocess": "kelvin_to_celsius"},
        years=None,
    )
    p.write_text(yaml.safe_dump(c))
    result = CliRunner().invoke(cli, ["validate", str(p)])
    assert result.exit_code == 0, result.output


# ===========================================================================
# Milestone 4: run end-to-end + output writers + parity
# ===========================================================================
import pandas as pd

import aggfly as af


def _write_run_inputs(tmp_path):
    """Write a small −180..180 raster to a zarr and a compact region to a shapefile.

    The region is a plain box well inside the grid so the default clip-to-extent
    path works (a globe-spanning region would wrap the antimeridian and break the
    0–360 clip — that path is covered by the clip_to_regions=False test).
    """
    import geopandas as gpd
    import shapely

    lon = np.arange(-100.0, -60.0, 5.0)  # 8 cells, -180..180 convention
    lat = np.arange(20.0, 60.0, 5.0)     # 8 cells
    time = pd.date_range("2000-07-01", periods=4, freq="12h")
    np.random.seed(7)
    arr = np.random.normal(20, 15, (len(time), len(lat), len(lon)))
    da = xr.DataArray(
        arr,
        dims=["time", "latitude", "longitude"],
        coords={"time": time, "latitude": lat, "longitude": lon},
        name="t2m",
    )
    zpath = tmp_path / "ds.zarr"
    da.to_dataset().to_zarr(zpath)

    box = shapely.geometry.box(-92, 28, -68, 52)
    gdf = gpd.GeoDataFrame({"geoid": ["r1"], "geometry": [box]}).set_crs("WGS84")
    spath = tmp_path / "regions.shp"
    gdf.to_file(spath)
    return str(zpath), str(spath)


def _run_config(zpath, spath, out, **dataset_extra):
    dataset = {"path": zpath, "var": "t2m", "lon_is_360": False}
    dataset.update(dataset_extra)
    return {
        "regions": {"path": spath, "regionid": "geoid"},
        "dataset": dataset,
        "aggregate": {
            "engine": "auto",
            "variables": {
                "tavg": [
                    ["aggregate", {"calc": "mean", "groupby": "date"}],
                    ["transform", {"transform": "power", "exp": [1, 2]}],
                    ["aggregate", {"calc": "sum", "groupby": "month"}],
                ]
            },
        },
        "output": {"path": out},
    }


def test_run_matches_direct_api(tmp_path):
    zpath, spath = _write_run_inputs(tmp_path)
    out = str(tmp_path / "panel.parquet")
    cfg = _run_config(zpath, spath, out)
    cpath = tmp_path / "config.yaml"
    cpath.write_text(yaml.safe_dump(cfg))

    # --- actual: via the CLI ---
    result = CliRunner().invoke(cli, ["run", str(cpath)])
    assert result.exit_code == 0, result.output
    actual = pd.read_parquet(out)

    # --- expected: the equivalent hand-written af.* script ---
    gr = af.georegions_from_path(spath, "geoid")
    ds = af.dataset_from_path(
        zpath, var="t2m", lon_is_360=False, georegions=gr, name="t2m"
    )
    w = af.weights_from_objects(ds, gr)
    w.calculate_weights()
    expected = af.aggregate_dataset(
        dataset=ds,
        weights=w,
        tavg=[
            ("aggregate", {"calc": "mean", "groupby": "date"}),
            ("transform", {"transform": "power", "exp": np.arange(1, 3)}),
            ("aggregate", {"calc": "sum", "groupby": "month"}),
        ],
    )

    assert list(actual.columns) == list(expected.columns)
    assert np.allclose(
        actual[["tavg_1", "tavg_2"]].values, expected[["tavg_1", "tavg_2"]].values
    )


def test_run_clip_disabled_matches_clip_enabled(tmp_path):
    """clip_to_regions is a read optimization: results must be identical either way."""
    zpath, spath = _write_run_inputs(tmp_path)
    outs = {}
    for clip in (True, False):
        out = str(tmp_path / f"panel_{clip}.parquet")
        cfg = _run_config(zpath, spath, out, clip_to_regions=clip)
        cpath = tmp_path / f"config_{clip}.yaml"
        cpath.write_text(yaml.safe_dump(cfg))
        result = CliRunner().invoke(cli, ["run", str(cpath)])
        assert result.exit_code == 0, result.output
        outs[clip] = pd.read_parquet(out)
    assert np.allclose(
        outs[True][["tavg_1", "tavg_2"]].values,
        outs[False][["tavg_1", "tavg_2"]].values,
    )


def test_run_output_formats(tmp_path):
    zpath, spath = _write_run_inputs(tmp_path)
    for fmt, ext in [("parquet", "parquet"), ("feather", "feather"), ("csv", "csv")]:
        out = str(tmp_path / f"panel.{ext}")
        cfg = _run_config(zpath, spath, out)
        cpath = tmp_path / f"config_{fmt}.yaml"
        cpath.write_text(yaml.safe_dump(cfg))
        result = CliRunner().invoke(cli, ["run", str(cpath)])
        assert result.exit_code == 0, result.output
        assert os.path.exists(out)
        reader = {"parquet": pd.read_parquet, "feather": pd.read_feather,
                  "csv": pd.read_csv}[fmt]
        df = reader(out)
        assert len(df) >= 1
        assert "tavg_1" in df.columns


def test_run_output_override(tmp_path):
    zpath, spath = _write_run_inputs(tmp_path)
    cfg = _run_config(zpath, spath, str(tmp_path / "ignored.parquet"))
    cpath = tmp_path / "config.yaml"
    cpath.write_text(yaml.safe_dump(cfg))
    out = str(tmp_path / "override.csv")
    result = CliRunner().invoke(cli, ["run", str(cpath), "-o", out])
    assert result.exit_code == 0, result.output
    assert os.path.exists(out)
    assert not os.path.exists(str(tmp_path / "ignored.parquet"))


# ===========================================================================
# Milestone 5: execution-backend wiring + weights command
# ===========================================================================
import aggfly.cli.pipeline as pipeline_mod


def test_backend_processes_starts_client_after_weights(tmp_path, monkeypatch):
    """The client must be started AFTER weights (calculate_weights tears one down)."""
    zpath, spath = _write_run_inputs(tmp_path)
    out = str(tmp_path / "p.parquet")
    cfg = _run_config(zpath, spath, out)
    cfg["execution"] = {"backend": "processes", "n_workers": 2}
    cpath = tmp_path / "c.yaml"
    cpath.write_text(yaml.safe_dump(cfg))

    events = []
    real_build = pipeline_mod.build_weights

    def spy_build(config, dataset, gr):
        events.append("weights")
        return real_build(config, dataset, gr)

    class FakeClient:
        pass

    monkeypatch.setattr(pipeline_mod, "build_weights", spy_build)
    monkeypatch.setattr(
        pipeline_mod.af, "start_dask_client",
        lambda **k: (events.append("start"), FakeClient())[1],
    )
    monkeypatch.setattr(
        pipeline_mod.af, "shutdown_dask_client", lambda: events.append("shutdown")
    )

    result = CliRunner().invoke(cli, ["run", str(cpath)])
    assert result.exit_code == 0, result.output
    assert events == ["weights", "start", "shutdown"]


def test_backend_threads_starts_no_client(tmp_path, monkeypatch):
    zpath, spath = _write_run_inputs(tmp_path)
    cfg = _run_config(zpath, spath, str(tmp_path / "p.parquet"))
    # default execution backend is "threads"
    cpath = tmp_path / "c.yaml"
    cpath.write_text(yaml.safe_dump(cfg))

    started = []
    monkeypatch.setattr(
        pipeline_mod.af, "start_dask_client", lambda **k: started.append(1)
    )
    result = CliRunner().invoke(cli, ["run", str(cpath)])
    assert result.exit_code == 0, result.output
    assert started == []


def test_backend_flag_override(tmp_path, monkeypatch):
    zpath, spath = _write_run_inputs(tmp_path)
    cfg = _run_config(zpath, spath, str(tmp_path / "p.parquet"))  # threads in config
    cpath = tmp_path / "c.yaml"
    cpath.write_text(yaml.safe_dump(cfg))

    kwargs_seen = {}

    class FakeClient:
        pass

    monkeypatch.setattr(
        pipeline_mod.af, "start_dask_client",
        lambda **k: (kwargs_seen.update(k), FakeClient())[1],
    )
    monkeypatch.setattr(pipeline_mod.af, "shutdown_dask_client", lambda: None)

    result = CliRunner().invoke(
        cli, ["run", str(cpath), "--backend", "processes", "--n-workers", "4"]
    )
    assert result.exit_code == 0, result.output
    assert kwargs_seen.get("n_workers") == 4


def test_weights_command_no_project_dir(tmp_path):
    zpath, spath = _write_run_inputs(tmp_path)
    cfg = _run_config(zpath, spath, str(tmp_path / "unused.parquet"))
    cpath = tmp_path / "c.yaml"
    cpath.write_text(yaml.safe_dump(cfg))
    result = CliRunner().invoke(cli, ["weights", str(cpath)])
    assert result.exit_code == 0, result.output
    assert "Computed weights" in result.output
    assert "not cached" in result.output


def test_weights_command_with_project_dir(tmp_path):
    zpath, spath = _write_run_inputs(tmp_path)
    proj = tmp_path / "proj"
    cfg = _run_config(zpath, spath, str(tmp_path / "unused.parquet"))
    cfg["weights"] = {"project_dir": str(proj)}
    cpath = tmp_path / "c.yaml"
    cpath.write_text(yaml.safe_dump(cfg))
    result = CliRunner().invoke(cli, ["weights", str(cpath)])
    assert result.exit_code == 0, result.output
    assert "Cached under" in result.output
    assert proj.exists()  # the project dir/cache was created


# ---------------------------------------------------------------------------
# dataset.storage_options / dataset.engine
#
# These are forwarded verbatim to dataset_from_path -> xarray so a config can
# point at object storage (gs://, s3://). Before this, the CLI could only read
# local paths: there was nowhere to put credentials or a reader backend.
# ---------------------------------------------------------------------------

def _cfg_with_dataset(**dataset_extra):
    import copy
    cfg = copy.deepcopy(GOOD_CONFIG)
    cfg["dataset"].update(dataset_extra)
    return cfg


def test_storage_options_and_engine_parse():
    cfg = _cfg_with_dataset(
        storage_options={"token": "anon"}, engine="zarr"
    )
    parsed = cfgmod.parse_config(cfg)
    assert parsed.storage_options == {"token": "anon"}
    assert parsed.reader_engine == "zarr"


def test_storage_options_default_to_none():
    parsed = cfgmod.parse_config(GOOD_CONFIG)
    assert parsed.storage_options is None
    assert parsed.reader_engine is None


def test_storage_options_must_be_a_mapping():
    with pytest.raises(Exception) as exc:
        cfgmod.parse_config(_cfg_with_dataset(storage_options="token=anon"))
    assert "storage_options must be a mapping" in str(exc.value)


def test_reader_engine_must_be_a_string():
    with pytest.raises(Exception) as exc:
        cfgmod.parse_config(_cfg_with_dataset(engine=["zarr"]))
    assert "dataset.engine must be a string" in str(exc.value)


def test_describe_hides_storage_option_values():
    parsed = cfgmod.parse_config(
        _cfg_with_dataset(storage_options={"token": "super-secret", "anon": True})
    )
    out = cfgmod.describe(parsed)
    assert "super-secret" not in out          # never print credentials
    assert "token" in out and "anon" in out   # but do show which keys were set
    assert "values hidden" in out


def test_pipeline_forwards_storage_options_and_engine(monkeypatch):
    """load_dataset must hand both straight to dataset_from_path."""
    from aggfly.cli import pipeline as pl

    seen = {}

    def fake_from_path(path, **kwargs):
        seen.update(kwargs)
        seen["path"] = path
        return "DATASET"

    monkeypatch.setattr(pl.af, "dataset_from_path", fake_from_path)

    parsed = cfgmod.parse_config(
        _cfg_with_dataset(storage_options={"token": "anon"}, engine="zarr")
    )
    out = pl.load_dataset(parsed, "gs://bucket/store", georegions=None)

    assert out == "DATASET"
    assert seen["storage_options"] == {"token": "anon"}
    assert seen["engine"] == "zarr"
    assert seen["path"] == "gs://bucket/store"


def test_pipeline_omits_them_when_unset(monkeypatch):
    """Absent config keys must not appear as kwargs at all."""
    from aggfly.cli import pipeline as pl

    seen = {}

    def fake_from_path(path, **kwargs):
        seen.update(kwargs)
        return "DATASET"

    monkeypatch.setattr(pl.af, "dataset_from_path", fake_from_path)
    pl.load_dataset(cfgmod.parse_config(GOOD_CONFIG), "local.zarr", georegions=None)

    assert "storage_options" not in seen
    assert "engine" not in seen


# ---------------------------------------------------------------------------
# `aggfly regions` — the shapefile counterpart to `aggfly info`
# ---------------------------------------------------------------------------

def _write_shp(tmp_path, unique_second=True):
    import geopandas as gpd
    import shapely

    names = ["a", "b", "c"] if unique_second else ["a", "b", "a"]
    gdf = gpd.GeoDataFrame(
        {"fips": ["01", "02", "03"], "name": names,
         "geometry": [shapely.box(i, 0, i + 1, 1) for i in range(3)]},
        crs="WGS84",
    )
    path = tmp_path / "r.shp"
    gdf.to_file(path)
    return path


def test_regions_command_reports_fields_and_bounds(tmp_path):
    path = _write_shp(tmp_path)
    res = CliRunner().invoke(cli, ["regions", str(path)])
    assert res.exit_code == 0, res.output
    assert "fips" in res.output and "name" in res.output
    assert "features=3" in res.output
    assert "bounds" in res.output
    assert "EPSG:4326" in res.output


def test_regions_command_uniqueness_flag(tmp_path):
    path = _write_shp(tmp_path, unique_second=False)   # 'name' repeats
    res = CliRunner().invoke(cli, ["regions", str(path), "--uniqueness"])
    assert res.exit_code == 0, res.output
    # only fips qualifies as a region id
    assert "regionid candidates" in res.output
    line = [l for l in res.output.splitlines() if "fips" in l and "name" not in l]
    assert line, res.output


def test_regions_command_rows_zero_skips_preview(tmp_path):
    path = _write_shp(tmp_path)
    res = CliRunner().invoke(cli, ["regions", str(path), "-n", "0"])
    assert res.exit_code == 0
    assert "row(s)" not in res.output


def test_regions_command_missing_file_is_a_clean_error(tmp_path):
    res = CliRunner().invoke(cli, ["regions", str(tmp_path / "nope.shp")])
    assert res.exit_code != 0
    assert "Error:" in res.output
    assert "Traceback" not in res.output       # user error, not a bug
