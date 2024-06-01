
# aggfly: Efficient climate data aggregation
[![PyPI version](https://badge.fury.io/py/aggfly.svg)](https://badge.fury.io/py/aggfly)

NOTE: aggfly is still in development and may not be stable for new users. Please proceed with caution.


## Overview: Why aggfly?

TODO: Brief introduction on the purpose of aggfly.

## Installation

### Required dependencies
- Python 3.11.6-3.12.2

### Instructions
Since aggfly relies on several packages with version restrictions, we recommend installing the package inside a virtual environment, such as `conda` (see [instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)).

1. Use `pip` to install package from PyPI:
```
pip install aggfly
```

### Jupyter

You may want to use aggfly to run batch jobs or in Jupyter sessions. In the case in which you experience issues when accessing the environment in Jupyter, try [this](https://ipython.readthedocs.io/en/latest/install/kernel_install.html#kernels-for-different-environments) to make the environment you have created available in Jupyter. In some of the cases, it will be sufficient to run the following commands:

```
conda activate <environment_name>
conda install ipykernel
python -m ipykernel install --user --name <environment_name>
conda deactivate
``` 

to then be able start the Jupyter session from your terminal.  


## Input datasets

The three raw inputs to be used to obtain an aggregated dataset containing climatic information at a relatively coarser spatial and temporal level are:

1. **Shapefile**: The shapefile containing the information on the boundaries of the target administrative regions. For example, a shapefile with the boundaries of world countries.
2. **Climatic dataset**: The raster dataset with the information at the relatively fine level that you want to aggregate at a spatially and/or temprally coarser level. For example, an ERA5 raster data with the hourly average temperature for each 0.25x0.25 degrees grid cell for the whole world. 
3. **(Optional) Secondary weights dataset**: The raster dataset containing the the information on the variable that you want to use to compute the weights that will be used to compute the weighted average of the climatic data over each administrative region.

## Workflow

We will here present the workflow with the main functionalities of the package. For a specific example application refer to the example notebook.

To correctly aggregate the raster data containing climatic information at the grid cell level, you will need to follow three steps:

1. Loading the shapefile containing the target administrative regions and the raster dataset to compute the area weights
2. Computing the weights to be used in the aggregation
3. Transforming and aggregating the climatic data spatially and temporally 

Remember to set the ```project_dir``` at the start of your code, to avoid having to specify it in the inputs of every command:

```
project_dir = '/user/name/aggfly_repository'
```

### 1. Loading the shapefile and the raster dataset

The first step towards aggregating the climatic dataset is to load the shapefile containing the target administrative regions at the level of which you want to aggregate the climatic data with the ```georegions_from_path()``` function:

``` 
georegions = af.georegions_from_path(
    "~/data/shapefiles/county/cb_2018_us_county_500k.shp",
    regionid='GEOID'
)
```

You now load a sample layer of the climatic raster dataset that you want to aggregate with the ```dataset_from_path()``` function. This will be used to compute the area weights - see the next paragraph for more details on weights:

```
# Open example dataset to construct weights
dataset = af.dataset_from_path(
    f"/home3/dth2133/data/annual/tempPrecLand2017.zarr", 
    var = 't2m',
    name = 'era5',
    georegions=georegions,
    preprocess = lambda x: (x - 273.15),
)
dataset.da
```

Main arguments:
- **var**: The selected variable to transform and aggregate.
- **preprocess**: It is used to specify a function for processing the raw values of your data before they are aggregated. For instance, it can be used to convert degrees Kelvin to degrees Celsius or to shift every osbervation back by one hour.
- **georegions**: The georegions object you have previously created.
- **name**: The name you want to assign to this dataset.


### 2. Computing the weights 

We first start with a brief explanation of why weights are an important component of this aggregation procedure to then show how to compute them and the main options you can choose.

#### Why are weights important?

There are two categories of weights that you may use to spatially aggregate the climatic data:

1. **Area weights** are the standard weights that we need to use, which consider the share of the area of an administrative that falls in a grid cell as the weight assigned to that cell. It is important to compute the weighted average of the climatic data over each administrative region, rather than the unweighted one, for two main reasons. First, the global grid cells have different dimension, since the longitude lines converge at the equator and, hence, the linear distance of longitude is larger at the equator and it converges to zero at the poles. Second, the border of some of our administrative regions may intersect some cells. In the latter case, we want the weight of the intersected cell to be proportional to the area covered by the administrative region.
2. **Secondary weights** are useful when we are interested in the average climate experienced by a particular subject. For example, if we are studying the effect of climate on human health, it may be appropriate to weight the climatic data by the number of humans that live in a grid cell. Alternatively, if we are interested in the responses of agricultural productivity to climate change, we may want to use the share of land covered by crops - or a specific crop - to compute the weight of each grid cell weight.

#### Implementation without secondary weights

This is the standard case, in which area weights are computed from the ```weights_from_objects``` without specifying any ```secondary weights``` in the options.

```
# Calculate area weights.
weights = af.weights_from_objects(
    dataset,
    georegions,
    project_dir=project_dir
)
weights.calculate_weights()
```

#### Implementation with secondary weights

To calculate weights based on a secondary variable, we first load the secondary variable dataset with one among ```secondary_weights_from_path```, ```pop_weights_from_path``` and ```crop_weights_from_path```. Then, we compute the weights through the ```weights_from_objects``` specifying ```secondary weights``` in the options.

```
secondary_weights = af.pop_weights_from_path("~/data/population/landscan-global-2016.tif")

# Calculate weights.
weights = af.weights_from_objects(
    dataset,
    georegions,
    secondary_weights=secondary_weights,
    project_dir=project_dir
)
weights.calculate_weights()
```

```weights``` will now contain the array of weights to be used for the aggregation.

Main arguments:
- **georegions**: The georegions object you have previously created.
- **dataset**: The layer of the dataset that is used to obtain the informations on the structure of the grid in order to compute the weights.
- **project_dir**: The project directory.
- **secondary_weights**: the ```secondary_weights``` object you have previously created.

### 3. Transforming and aggregating 

You first load the full dataset that you want to aggregate using the same procedure as in step 1 - when you however loaded just a sample layer of the dataset - and you then finally aggregate it with the ```aggregate_dataset()``` function.

```
dataset = af.dataset_from_path(
    f"~//data/annual/tempPrecLand{year}.zarr", 
    var = 't2m',
    name = 'era5',
    georegions=georegions,
    preprocess = lambda x: (x - 273.15)
)

output_df = af.aggregate_dataset(
    dataset=dataset, 
    weights=weights,
    tavg = [
        ('aggregate', {'calc':'mean', 'groupby':'date'}),
        ('transform', {'transform':'power', 'exp':np.arange(1,2)}),
        ('aggregate', {'calc':'sum', 'groupby':'year'})
    ],
    bins= [
        ('aggregate', {'calc':'mean', 'groupby':'date'}),
        ('aggregate', {'calc':'bins', 'groupby':'year', 'ddargs':[[25,99,0],[30,99,0]]})
    ],
    growing_dday = [
        ('aggregate', {'calc':'dd', 'groupby':'date', 'ddargs':[10,30,0]}),
        ('aggregate', {'calc':'sum', 'groupby':'year'}),
    ],
    heating_dday = [
        ('aggregate', {'calc':'dd', 'groupby':'date', 'ddargs':[-99,20,1]}),
        ('aggregate', {'calc':'sum', 'groupby':'year'}),
    ]
)
```

Notice that the function will first compute the aggregation across time in the way described by the lists of 

Main arguments:
- **dataset**: The complete raster that you have just loaded, which contains the climatic data you want to aggregate.
- **georegions**: The georegions object you have previously created.
- **weights**: 

**TO COMPLETE**
agg_dict (dict): A dictionary containing the arguments for creating TemporalAggregator objects.
                        The keys of the dictionary are names, and the values are a list of either tuples or TemporalAggregator objects.
                        If the list contains tuples, use them as arguments to instantiate a temporal aggregator.



Available transformations include:

- **mean** computes the average value of the within the time period specified by ```groupby```.
- **min** computes the minimum value within the time period.
- **max**: computes the maximum value within the time period.
- **sum**: computes the sum over the time period.
- **dd**: 
- **bin**:
- **exp**: computes the polynomials of the specified degrees.


For a more detailed application of the aggregation, refer to the example notebook.
