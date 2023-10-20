import numpy as np
import pandas as pd
import xarray as xr

# import pygeos
import geopandas as gpd
from functools import lru_cache


@lru_cache(maxsize=None)
def open_usa_shp(region_list=None):
    usa = gpd.read_file(
        "/home3/dth2133/repositories/aggfly/data/shapefiles/usa_simple_noHI.shp"
    )
    usa["state"] = usa.HASC_1.str.slice(start=3)
    usa = usa.loc[usa.state != "AK"]
    if region_list:
        usa = usa[usa.state.isin(region_list)]
    return usa


@lru_cache(maxsize=None)
def open_counties_shp(region_list=None):
    f = "/home3/dth2133/data/shapefiles/county/cb_2018_us_county_500k.shp"
    counties = gpd.read_file(f)
    counties["fips"] = counties.GEOID
    counties["STATEFP"] = counties.STATEFP.astype("int")
    counties = counties[
        (counties.STATEFP < 60) & (np.logical_not(np.in1d(counties.STATEFP, [2, 15])))
    ]
    if region_list:
        counties = counties[counties.fips.isin(region_list)]
    return counties


@lru_cache(maxsize=None)
def open_global_shp(region_list=None):
    f = "/home3/dth2133/data/shapefiles/country/world_countries_2020.shp"
    countries = gpd.read_file(f)

    # Drop Antarctica and disputed areas...
    countries = countries.loc[countries.OBJECTID != 232]
    countries = countries.loc[countries.CNTRY_CODE != "9999"]

    if region_list:
        countries = countries[countries.OBJECTID.isin(region_list)]
    return countries
