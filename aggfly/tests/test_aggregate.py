import pytest

import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd

import aggfly as af

def dataset_360():
    np.random.seed(1216)
    
    x = np.linspace(0, 360, 3)
    longitude = (x[1:] + x[:-1]) / 2
    
    y = np.linspace(-90, 90, 3)
    latitude = (y[1:] + y[:-1]) / 2
    
    time = pd.date_range('2000-07-01', periods=4, freq='12h')
    
    arr = np.random.normal(20, 15, (len(time), len(latitude), len(longitude)))
    
    da = xr.DataArray(
        data = arr,
        dims = ['time', 'latitude', 'longitude'],
        coords = {
            'time': time,
            'latitude': latitude,
            'longitude': longitude,
        }
    )
    return af.Dataset(da, lon_is_360=True)

@pytest.fixture(name='dataset_360')
def dataset_360_fixture():
    return dataset_360()

def georegion():
    np.random.seed(1216)
    longitude = np.random.uniform(-180, 180, 20)
    latitude = np.random.uniform(-90, 90, 20)
    
    polygon = gpd.GeoDataFrame(
            {'geometry': gpd.points_from_xy(longitude, latitude)}
        ).unary_union.convex_hull
    gdf = gpd.GeoDataFrame(
        {
            'geoid' : 'region_1',
            'geometry': [polygon]
        }
    )
    gdf = gdf.set_crs('WGS84')
    return af.GeoRegions(gdf, regionid='geoid')

@pytest.fixture(name='georegion')
def georegion_fixture():
    return georegion()

def secondary_weights():
    np.random.seed(1216)
    
    x = np.linspace(-180, 180, 5)
    longitude = (x[1:] + x[:-1]) / 2
    
    y = np.linspace(-90, 90, 5)
    latitude = (y[1:] + y[:-1]) / 2

    arr = np.random.rand(1, len(latitude), len(longitude))
    
    da = xr.DataArray(
        data = arr,
        dims = ['band', 'y', 'x'],
        coords = {
            'band' : [1],
            'y': latitude,
            'x': longitude,
        }
    )
    da = da.rio.write_crs("WGS84")
    return af.SecondaryWeights(da)

@pytest.fixture(name='secondary_weights')
def secondary_weights_fixture():
    return secondary_weights()

def weights(dataset_360, georegion, secondary_weights):
    w = af.weights_from_objects(dataset_360, georegion, secondary_weights)
    w.calculate_weights()
    w.weights = w.weights.sort_values('cell_id')
    return w

@pytest.fixture(name='weights')
def weights_fixture(dataset_360, georegion, secondary_weights):
    return weights(dataset_360, georegion, secondary_weights)

def test_weights(weights):
    assert isinstance(weights, af.GridWeights)
    assert isinstance(weights.grid, af.Grid)
    assert isinstance(weights.georegions, af.GeoRegions)
    assert isinstance(weights.raster_weights, af.SecondaryWeights)
    print(weights.weights)
    assert np.allclose(
        weights.weights.area_weight,
        np.array([0.87770301, 0.84596667, 0.42553152, 0.94280892])
    )
    assert np.allclose(
        weights.weights.raster_weight,
        np.array([0.67392287, 0.80659155, 0.56727215, 0.38801016])
    )
    assert np.allclose(
        weights.weights.weight,
        np.array([0.24283805, 0.28013403, 0.09910194, 0.15018472])
    )

def test_aggregate_time(dataset_360, weights):
    adict = af.aggregate_time(
        dataset=dataset_360, 
        weights=weights,
            bins= [
                ('aggregate', {'calc':'mean', 'groupby':'date'}),
                ('aggregate', {'calc':'bins', 'groupby':'month', 'ddargs':[[-99,20,0],[20,99,0]]})
            ],
            cooling_dday = [
                ('aggregate', {'calc':'dd', 'groupby':'date', 'ddargs':[20,99,0]}),
                ('aggregate', {'calc':'sum', 'groupby':'month'})
            ],
            tavg = [
                ('aggregate', {'calc':'mean', 'groupby':'date'}),
                ('transform', {'transform':'power', 'exp':np.arange(1,3)}),
                ('aggregate', {'calc':'sum', 'groupby':'month'})
            ]    
        )
    df = xr.combine_by_coords([ adict[x].da.rename(x) for x in adict.keys()]).to_dataframe()
    assert np.allclose(df.values, 
        np.array([[   0.      ,    2.      ,   44.945648,   62.472824, 1956.361671],
                [   1.      ,    1.      ,   25.910298,   39.60287 ,  801.80304 ],
                [   1.      ,    1.      ,    9.12584 ,   35.789426,  670.521066],
                [   1.      ,    1.      ,   14.932308,   37.648473,  858.069229]])
    )


def test_aggregate(dataset_360, weights):
    df = af.aggregate_dataset(
        dataset=dataset_360, 
        weights=weights,
        tavg = [
                ('aggregate', {'calc':'mean', 'groupby':'date'}),
                ('transform', {'transform':'power', 'exp':np.arange(1,3)}),
                ('aggregate', {'calc':'sum', 'groupby':'month'}),
        ]    
    )
    
    assert np.allclose(df[['tavg_1', 'tavg_2']].values, 
        np.array([[  46.906441, 1202.304441]])
    )