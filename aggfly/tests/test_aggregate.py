# The following code defines tests for various functions related to the package. 
# These tests are useful for ensuring that the dataset transformation, aggregation, 
# and weighting functions work correctly. 

import pytest

import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd

import aggfly as af

def dataset_360():
    """
    Create a sample dataset with longitude ranging from 0 to 360 degrees.
    
    Returns:
    --------
    af.Dataset
        The created dataset.
    """
    # Set random seed for reproducibility
    np.random.seed(1216)
    # Generate evenly spaced values from 0 to 360
    x = np.linspace(0, 360, 3)
    # Calculate midpoints for longitude
    longitude = (x[1:] + x[:-1]) / 2
    # Generate evenly spaced values from -90 to 90
    y = np.linspace(-90, 90, 3)
    # Calculate midpoints for latitude
    latitude = (y[1:] + y[:-1]) / 2

    # Generate time range
    time = pd.date_range('2000-07-01', periods=4, freq='12h')

    # Create random data
    arr = np.random.normal(20, 15, (len(time), len(latitude), len(longitude)))

    # Return dataset with longitude 0-360
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
    """
    Fixture for the dataset_360 function.
    
    Returns:
    --------
    af.Dataset
        The created dataset.
    """
    return dataset_360()

def georegion():
    """
    Create a sample georegion using random points.
    
    Returns:
    --------
    af.GeoRegions
        The created georegion.
    """
    # Set random seed for reproducibility
    np.random.seed(1216)
    # Generate random longitudes
    longitude = np.random.uniform(-180, 180, 20)
    # Generate random latitudes
    latitude = np.random.uniform(-90, 90, 20)
    
    polygon = gpd.GeoDataFrame(
            {'geometry': gpd.points_from_xy(longitude, latitude)}
        ).unary_union.convex_hull # Create convex hull polygon from points
    gdf = gpd.GeoDataFrame(
        {
            'geoid' : 'region_1',
            'geometry': [polygon]
        }
    )
    # Set coordinate reference system
    gdf = gdf.set_crs('WGS84')
    # Return georegion
    return af.GeoRegions(gdf, regionid='geoid')

@pytest.fixture(name='georegion')
def georegion_fixture():
     """
    Fixture for the georegion function.
    
    Returns:
    --------
    af.GeoRegions
        The created georegion.
    """
    return georegion()

def secondary_weights():
    """
    Create a sample secondary weights dataset.
    
    Returns:
    --------
    af.SecondaryWeights
        The created secondary weights dataset.
    """
    # Set random seed for reproducibility
    np.random.seed(1216)

    # Generate evenly spaced values for longitude
    x = np.linspace(-180, 180, 5)
    # Calculate midpoints for longitude
    longitude = (x[1:] + x[:-1]) / 2

    # Generate evenly spaced values for latitude
    y = np.linspace(-90, 90, 5)
    # Calculate midpoints for latitude
    latitude = (y[1:] + y[:-1]) / 2

    # Create random data
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
    # Write coordinate reference system
    da = da.rio.write_crs("WGS84") 
    # Return secondary weights dataset
    return af.SecondaryWeights(da)

@pytest.fixture(name='secondary_weights')
def secondary_weights_fixture():
    """
    Fixture for the secondary_weights function.
    
    Returns:
    --------
    af.SecondaryWeights
        The created secondary weights dataset.
    """
    return secondary_weights()

def weights(dataset_360, georegion, secondary_weights):
    """
    Calculate weights for the dataset and georegion.
    
    Returns:
    --------
    af.GridWeights
        The calculated weights.
    """
    # Create weights object
    w = af.weights_from_objects(dataset_360, georegion, secondary_weights)
    # Calculate weights
    w.calculate_weights()
    # Sort weights by cell ID
    w.weights = w.weights.sort_values('cell_id')
    return w

@pytest.fixture(name='weights')
def weights_fixture(dataset_360, georegion, secondary_weights):
    """
    Fixture for the weights function.
    
    Returns:
    --------
    af.GridWeights
        The calculated weights.
    """
    return weights(dataset_360, georegion, secondary_weights)

def test_weights(weights):
    """
    Test the calculated weights.
    
    Parameters:
    -----------
    weights : af.GridWeights
        The calculated weights.
    """    
    # Check type of weights
    assert isinstance(weights, af.GridWeights)
    # Check type of grid
    assert isinstance(weights.grid, af.Grid)
    # Check type of georegions
    assert isinstance(weights.georegions, af.GeoRegions)
    # Check type of raster weights
    assert isinstance(weights.raster_weights, af.SecondaryWeights)
    # Print weights
    print(weights.weights)
    # Check area weights
    assert np.allclose(
        weights.weights.area_weight,
        np.array([0.87770301, 0.84596667, 0.42553152, 0.94280892])
    )
    # Check raster weights
    assert np.allclose(
        weights.weights.raster_weight,
        np.array([0.67392287, 0.80659155, 0.56727215, 0.38801016])
    )
    # Check final weights
    assert np.allclose(
        weights.weights.weight,
        np.array([0.24283805, 0.28013403, 0.09910194, 0.15018472])
    )

def test_aggregate_time(dataset_360, weights):
    """
    Test the time aggregation function.
    
    Parameters:
    -----------
    dataset_360 : af.Dataset
        The dataset to aggregate.
    weights : af.GridWeights
        The weights to use for aggregation.
    
    This test performs several aggregations on the dataset over time, 
    using different aggregation functions and transformations. It 
    verifies that the aggregated results are correct by comparing 
    them with expected values.
    """
    adict = af.aggregate_time(
        dataset=dataset_360, 
        weights=weights,
            bins= [
                ('aggregate', {'calc':'mean', 'groupby':'date'}), # Mean aggregation by date
                ('aggregate', {'calc':'bins', 'groupby':'month', 'ddargs':[[-99,20,0],[20,99,0]]}) # Bin aggregation by month
            ],
            cooling_dday = [
                ('aggregate', {'calc':'dd', 'groupby':'date', 'ddargs':[20,99,0]}), # Degree day aggregation by date
                ('aggregate', {'calc':'sum', 'groupby':'month'}) # Sum aggregation by month
            ],
            tavg = [
                ('aggregate', {'calc':'mean', 'groupby':'date'}), # Mean aggregation by date
                ('transform', {'transform':'power', 'exp':np.arange(1,3)}), # Polynomial transformation
                ('aggregate', {'calc':'sum', 'groupby':'month'}) # Sum aggregation by month
            ]    
        )
    # Combine results into a DataFrame
    df = xr.combine_by_coords([ adict[x].da.rename(x) for x in adict.keys()]).to_dataframe()
    # Check that the aggregated values match expected values
    assert np.allclose(df.values, 
        np.array([[   0.      ,    2.      ,   44.945648,   62.472824, 1956.361671],
                [   1.      ,    1.      ,   25.910298,   39.60287 ,  801.80304 ],
                [   1.      ,    1.      ,    9.12584 ,   35.789426,  670.521066],
                [   1.      ,    1.      ,   14.932308,   37.648473,  858.069229]])
    )


def test_aggregate(dataset_360, weights):
    """
    Test the dataset aggregation function.

    Parameters:
    -----------
    dataset_360 : af.Dataset
        The dataset to aggregate.
    weights : af.GridWeights
        The weights to use for aggregation.

    This test performs the following operations:
    1. Aggregates the dataset by calculating the mean by date.
    2. Transforms the aggregated data by raising it to the power of 1 and 2.
    3. Aggregates the transformed data by summing it by month.
    It then checks if the aggregated values match the expected results.
    """
    df = af.aggregate_dataset(
        dataset=dataset_360, 
        weights=weights,
        tavg = [
                ('aggregate', {'calc':'mean', 'groupby':'date'}), # Aggregate by mean per date
                ('transform', {'transform':'power', 'exp':np.arange(1,3)}), # Transform by raising to power of 1 and 2
                ('aggregate', {'calc':'sum', 'groupby':'month'}), # Aggregate by summing per month
        ]    
    )
    
    # Check if the aggregated values are as expected
    assert np.allclose(df[['tavg_1', 'tavg_2']].values, 
        np.array([[  46.906441, 1202.304441]])
    )
