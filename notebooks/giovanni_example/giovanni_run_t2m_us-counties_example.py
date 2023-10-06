#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import os
import os.path
os.environ['USE_PYGEOS'] = '0'
import xarray as xr
import numpy as np
import pandas as pd
import fiona
import glob
import dask_geopandas
import geopandas as gpd
import time
from copy import deepcopy
from functools import reduce

import aggfly
import netCDF4

import dask
from dask.distributed import Client
from dask.diagnostics import ProgressBar

from aggfly import regions
from aggfly import dataset, regions, grid_weights
from aggfly.aggregate import TemporalAggregator, SpatialAggregator
# from aggfly.aggregate import TemporalAggregator, SpatialAggregator

ProgressBar().register()
# client = Client(n_workers=2)

project_dir = '/Users/gb2884/Desktop/aggfly_loc'


# In[2]:


gdf = gpd.read_file('/Users/gb2884/Desktop/aggfly_loc/usa_simple_noHI.shp')
print(gdf.head())
gdf.plot(color='blue', legend=True)


# In[3]:


import os
import os.path

# Years to aggregate
years = np.arange(1951,2020)
years = years[years != 1959]

georegions = regions.from_path('/Users/gb2884/Desktop/aggfly_loc/usa_simple_noHI.shp',regionid='geometry')


# In[13]:


dataset1 = xr.open_dataset("/Users/gb2884/The Lab Dropbox/Giovanni Brocca/Macro_Enviro_Dev/1_Data/world_example/era_june_2000_2tm.nc")


# In[ ]:





# In[5]:


def preprocess_era5l(array):
    # Function for cleaning the era5l data.
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):   
        # Reformat longitude
        array.coords['longitude'] = (array.coords['longitude'] + 180) % 360 - 180
        array = array.sortby(array.longitude)
        # Kelvin -> Celsius
        array = array - 273.15
    return array

# Open example dataset to construct weights.
clim = dataset.from_path(
    #f"/Users\gb2884/The Lab Dropbox/Giovanni Brocca/Macro_Enviro_Dev/1_Data/world_example/era_june_2000_2tm.nc", 
    f"/Users/gb2884/The Lab Dropbox/Giovanni Brocca/Macro_Enviro_Dev/1_Data/IPUMS_1level_example/era_1st_June_2000_tm.nc",
    var = 't2m',
    engine = 'netcdf4',
    name='era5',
    clip_geom=georegions,
    preprocess = preprocess_era5l)

# Calculate area and pop layer weights.
weights = grid_weights.from_objects(
    clim,
    georegions,
    wtype=None,
    simplify=0.001,
    project_dir=project_dir)


# clim.clip_data_to_georegions_extent(georegions)

print("Loading weights")
w = weights.weights()
nzw_ind = np.isin(clim.grid.index, w.cell_id).nonzero()
print(w) 
#print(nzw_ind)
#print(clim.grid.index)
#print(w.cell_id)


# In[6]:


print("Scheduling aggregation")
dailies = [
    TemporalAggregator('avg', 'time.date')
]
monthly = TemporalAggregator('sum', 'time.month')


# In[11]:


out = pd.DataFrame()
#for year in years:
#    print(year)
#    start = time.time()

print('Loading climate data')
clim = dataset.from_path(
    #f"/Users\gb2884/The Lab Dropbox/Giovanni Brocca/Macro_Enviro_Dev/1_Data/world_example/era_june_2000_2tm.nc", 
    f"/Users/gb2884/The Lab Dropbox/Giovanni Brocca/Macro_Enviro_Dev/1_Data/IPUMS_1level_example/era_1st_June_2000_tm.nc",
    var = 't2m',
    engine = 'netcdf4',
    name='era5',
    clip_geom=georegions,
    preprocess = preprocess_era5l)
clim.update(clim.da.persist())

print('Aggregating')
day_ds = [x.map_execute(clim, nzw_ind) for x in dailies]
month_ds = [monthly.map_execute(x, nzw_ind) for x in day_ds]


# In[12]:


names = ['avg']
df = SpatialAggregator(month_ds, w, weights.grid, names=names).compute()
    
# Clean output dataset and append
df = df.reset_index().rename(columns={'time':'month'})
df['year'] = year
df = df[['region_id', 'year', 'month'] + names]
df = georegions.shp[['fips']].merge(df, left_index=True, right_on='region_id')

out = pd.concat([out, df])


# In[ ]:


out.reset_index().to_feather("/home3/dth2133/data/clim_data_requests/maya/temp_data_maya.feather")


# In[ ]:


slc = out.loc[out.year==2019].groupby('fips').mean()
georegions.shp.merge(slc, right_index=True, left_on='fips').plot(column='CDD_20C')


# In[ ]:




