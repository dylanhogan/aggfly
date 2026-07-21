from .aggregate import (
    TemporalAggregator,
    SpatialAggregator,
    aggregate_dataset,
    aggregate_time,
    aggregate_space,
    distributed_client,
    is_distributed,
    start_dask_client,
    shutdown_dask_client
)
from .dataset import Dataset, Grid, dataset_from_path, dataset_to_zarr, zarr_from_path
from .weights import (
    CropWeights, 
    PopWeights, 
    GridWeights, 
    SecondaryWeights,
    weights_from_objects, 
    pop_weights_from_path, 
    crop_weights_from_path,
    secondary_weights_from_path
)
from .regions import (
    GeoRegions,
    georegions_from_path,
    georegions_from_gdf,
    shapefile_info,
)

# The test fixtures are deliberately NOT re-exported here. `aggfly.tests`
# imports pytest at module load, so importing them made pytest a hard runtime
# requirement of the library — `pip install aggfly; import aggfly` failed with
# ModuleNotFoundError once pytest moved to the dev dependency group. Test
# fixtures are not public API; import them directly from
# `aggfly.tests.test_aggregate` if you need them.