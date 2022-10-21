import numpy as np
import pandas as pd
import xarray as xr
import pygeos
import geopandas as gpd
from functools import lru_cache

@lru_cache(maxsize=None)
def open_usa_shp(region_list=None, 
        usa_shp_path = "/home3/dth2133/repositories/aggfly/data/shapefiles/usa_simple_noHI.shp"):
    usa = gpd.read_file(usa_shp_path)
    usa['state'] = usa.HASC_1.str.slice(start=3)
    usa = usa.loc[usa.state!='AK']
    if region_list:
        usa = usa[usa.state.isin(region_list)]
    return usa

@lru_cache(maxsize=None)
def open_counties_shp(region_list=None, 
    usa_county_path =  "/home3/dth2133/data/shapefiles/county/cb_2018_us_county_500k.shp"):
    counties = gpd.read_file(usa_county_path)
    counties['fips'] = counties.GEOID
    counties['STATEFP'] = counties.STATEFP.astype('int')
    counties = counties[(counties.STATEFP < 60) 
                        & (np.logical_not(np.in1d(counties.STATEFP, [2, 15])))]
    if region_list:
        counties = counties[counties.fips.isin(region_list)]
    return counties

def open_uk_shp(region_list=None, 
        uk_shp_path = '/projects/OPPENHEIMER/tb/shp/uk/data/International_Territorial_Level_3_(January_2021)_UK_BFE_V3/ITL3_JAN_2021_UK_BFE_V3.shp'):

    # Load shapefile
    uk_shp_path = '/projects/OPPENHEIMER/tb/shp/uk/data/International_Territorial_Level_3_(January_2021)_UK_BFE_V3/ITL3_JAN_2021_UK_BFE_V3.shp'
    uk = gpd.read_file(uk_shp_path)
    
    # Change the names
    uk['id'] = uk.ITL321CD
    uk = uk.rename(columns ={'LONG':'longitude', 'LAT':'latitude'})
    
    # Change the crs
    code = "/home/bearpark/Documents/aggfly/"
    dylans_shp = gpd.read_file(f"{code}data/shapefiles/usa_simple_noHI.shp")
    crs = dylans_shp.crs
    uk = uk.to_crs(crs)
    
    if region_list:
        uk = uk[uk.id.isin(region_list)]
    return uk