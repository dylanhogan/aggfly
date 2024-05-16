# aggfly

## Introduction

Brief introduction on the purporse of aggfly.

## Installation

We will start by assuming that you have already created a local copy of the aggfly repository by cloning it on your computer, but you can find the instructions to do it in the [git documentation](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository).

To install the package, you first need to create a conda environment where it will be installed. Assuming you already have conda installed in your system - otherwise follow the [instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) -, download the `aggfly-dev-environment.yml` file located in aggfly's main repository. Use this file, which provides the information on the core packages you need to run the code, to create your conda environment by running the following command from your terminal:

```
conda env create --file /path/to/aggfly-dev-environment.yml --name aggfly-dev 
```

with ```/path/to/``` specifying the location of the YAML file and ```aggfly-dev``` being the name of the environment you will create. For other details on how to use YAML files and how to manage environments, check out the [conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).   

After having created the environment, you can install the package by activating the environment and by using the ``` pip install -e``` command, only after having installed ```pip```:

```
conda activate aggfly-dev
conda install pip
pip install -e
```

from the root directory of your local aggfly git repository.

You may want to use aggfly to run batch jobs or in Jupyter sessions. In the case in which you want to use an interactive Jupyter session, make the environment you have created available in JupyterLab by following the [instructions](https://ipython.readthedocs.io/en/latest/install/kernel_install.html#kernels-for-different-environments). In some of the cases, it will be sufficient to run the following commands:

```
conda activate aggfly-dev
conda install ipykernel
python -m ipykernel install --user --aggfly-dev
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
    "/home3/dth2133/data/shapefiles/county/cb_2018_us_county_500k.shp",
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

Main ptions:
- **var**:
- **preprocess**:


### 2. Computing the weights 

We first start with a brief explanation of why weights are an important component of this aggregation procedure to then show how to compute them and the main options you can choose.

#### Why are weights important?


#### Implementation without secondary weights

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

```
secondary_weights = af.pop_weights_from_path("/home3/dth2133/data/population/landscan-global-2016.tif")

# Calculate weights.
weights = af.weights_from_objects(
    dataset,
    georegions,
    secondary_weights=secondary_weights,
    project_dir=project_dir
)
weights.calculate_weights()
```




### 3. Transforming and aggregating 

You first load the full dataset that you want to aggregate using the same procedure as in step 1 - when you however loaded just a sample layer of the dataset - and you then finally aggregate it with the ```aggregate_dataset()``` function.

```
dataset = af.dataset_from_path(
    f"/home3/dth2133/data/annual/tempPrecLand{year}.zarr", 
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


Available transformations include:

- **avg**:
- **min**:
- **max**:
- **sum**:
- **polynomial**:
- **dd**:
- **bin**:

For a more detailed application of the aggregation, refer to the example notebook.pip install -e .
```

Of course, please feel free to suggest changes/improvements/extensions in [Issues](https://github.com/dylanhogan/aggfly/issues) :)
