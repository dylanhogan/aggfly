"""Spline mathematics, lazy execution, and pipeline integration."""
from types import SimpleNamespace

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import aggfly as af
from aggfly.dataset.splines import spline_arrays, validate_spline


def matrix(values, **params):
    x = xr.DataArray(np.asarray(values, dtype=float), dims='sample').chunk(sample=17)
    _, arrays = spline_arrays(x, **params)
    return np.column_stack([a.compute().values for a in arrays])


def stagg_reference(t, knots):
    h = lambda k: np.maximum(t - k, 0) ** 3
    a, b = knots[-2:]
    return np.column_stack([t] + [
        h(k) - h(a) * (b-k)/(b-a) + h(b) * (a-k)/(b-a)
        for k in knots[:-2]
    ])


def test_stagg_formula_and_nan():
    t = np.array([-20, 0, 5, 7.5, 12.5, 20, 30, 100, np.nan])
    knots = [0, 7.5, 12.5, 20]
    actual = matrix(t, degree=3, knots=knots, restricted=True)
    np.testing.assert_allclose(actual, stagg_reference(t, knots), equal_nan=True)


@pytest.mark.parametrize('degree', [1, 2, 3, 4])
@pytest.mark.parametrize('restricted', [False, True])
def test_basis_spaces_and_tails(degree, restricted):
    knots = [-10, 0, 10, 20, 30]
    t = np.linspace(-50, 70, 241)
    params = dict(degree=degree, knots=knots, restricted=restricted)
    power = matrix(t, **params)
    bs = matrix(t, **params, basis='bspline')
    expected = len(knots) - degree + 2 if restricted else len(knots) + degree
    assert power.shape == bs.shape == (len(t), expected)
    # Include the constant: the two bases omit different intercept directions.
    a = np.column_stack([np.ones(len(t)), power])
    b = np.column_stack([np.ones(len(t)), bs])
    a /= np.linalg.norm(a, axis=0)
    b /= np.linalg.norm(b, axis=0)
    assert np.linalg.matrix_rank(a) == expected + 1
    assert np.linalg.matrix_rank(b) == expected + 1
    np.testing.assert_allclose(b @ np.linalg.lstsq(b, a, rcond=None)[0], a, atol=1e-10)
    if restricted:
        for basis in ('truncated_power', 'bspline'):
            for tail in ([-40, -30, -20], [40, 50, 60]):
                v = matrix(tail, **params, basis=basis)
                np.testing.assert_allclose(np.diff(v, n=2, axis=0), 0, atol=1e-7)


@pytest.mark.parametrize('basis', ['truncated_power', 'bspline'])
def test_lazy_metadata_and_no_mutation(basis):
    ds = make_dataset()
    before = ds.da.copy(deep=True)
    outputs = ds.spline(degree=4, knots=[-10, 0, 10, 20, 30], restricted=True, basis=basis)
    for output in outputs:
        assert isinstance(output.da.data, da.Array)
        assert output.da.chunks == ds.da.chunks
        assert output.da.dims == ds.da.dims
        assert output.history and not ds.history
    xr.testing.assert_equal(ds.da, before)
    for values in ([-30, 0, np.nan, 40],):
        assert np.isnan(matrix(values, degree=4, knots=[-10, 0, 10, 20, 30], restricted=True, basis=basis)[2]).all()


def make_dataset():
    values = np.array([0, 20, 20, 40], dtype=float)[:, None, None] + np.array([[0, 2], [4, 6]])
    array = xr.DataArray(values, dims=['time', 'latitude', 'longitude'], coords={
        'time': pd.date_range('2001-01-01', periods=4, freq='12h'),
        'latitude': [0., 1.], 'longitude': [0., 1.],
    }).chunk({'time': 2, 'latitude': 1, 'longitude': 2})
    return af.Dataset(array, lon_is_360=False)


@pytest.mark.parametrize('engine', ['dask', 'numba'])
def test_daily_annual_spatial_pipeline(engine):
    ds = make_dataset()
    knots = [0, 7.5, 12.5, 20]
    w = np.array([1., 2., 3., 4.])
    weights = SimpleNamespace(grid=ds.grid, weights=pd.DataFrame({
        'cell_id': ds.grid.cell_id, 'index_right': [0]*4, 'weight': w,
    }), zero_weight='nan')
    specs = {'temperature': [
        ('aggregate', {'calc': 'mean', 'groupby': 'date'}),
        ('transform', {'transform': 'spline', 'degree': 3, 'restricted': True, 'knots': knots}),
        ('aggregate', {'calc': 'sum', 'groupby': 'year'}),
    ]}
    temporal = af.aggregate_time(ds, aggregator_dict=specs, engine=engine)
    result = af.aggregate_space(temporal, weights)
    daily = ds.da.transpose('time', 'latitude', 'longitude').values.reshape(2, 2, 4).mean(axis=1)
    expected = sum((stagg_reference(day, knots) * w[:, None]).sum(axis=0) / w.sum() for day in daily)
    np.testing.assert_allclose(result[['temperature_power_1', 'temperature_term_1', 'temperature_term_2']].iloc[0], expected)


def test_legacy_default():
    ds = make_dataset()
    result = af.aggregate_time(ds, temperature=[('transform', {'transform': 'spline'})])
    assert list(result) == ['temperature_spline1', 'temperature_spline2']
    xr.testing.assert_equal(result['temperature_spline1'].da, ds.da)
    np.testing.assert_allclose(result['temperature_spline2'].da, np.maximum(ds.da - 20, 0))


@pytest.mark.parametrize('params', [
    {'degree': 0}, {'degree': 5}, {'degree': True}, {'degree': 2.5},
    {'restricted': 'yes'}, {'basis': 'unknown'}, {'knots': []},
    {'knots': [1, 1]}, {'knots': [2, 1]}, {'knots': [0, np.nan]},
    {'knots': [[1, 2]]}, {'knots': None},
    {'restricted': True, 'degree': 4, 'knots': [0, 10, 20]},
])
def test_invalid_specs(params):
    with pytest.raises(ValueError):
        validate_spline(**params)
