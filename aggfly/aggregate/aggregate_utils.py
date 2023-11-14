from functools import lru_cache

import numba
import numpy as np


@lru_cache(maxsize=None)
def get_time_dim(time):
    dim_dict = dict(hour=5, day=4, month=3, year=2)
    return dim_dict[time]
