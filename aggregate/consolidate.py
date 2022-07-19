import zarr
from os import listdir

for d in listdir('/home3/dth2133/data/annual'):
    zarr.consolidate_metadata('/home3/dth2133/data/annual/' + d)