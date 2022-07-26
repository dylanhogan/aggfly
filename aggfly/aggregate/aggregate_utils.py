from functools import lru_cache

import numba
import numpy as np

@lru_cache(maxsize=None)
def get_time_dim(time):
    dim_dict = dict(
        hour=5,
        day=4,
        month=3,
        year=2)
    return dim_dict[time]
                                  
@numba.generated_jit(nopython=True)
def nb_expander(x):
    if x.ndim == 3:
        return lambda x: np.expand_dims(
                            np.expand_dims(
                                np.expand_dims(x, axis=3), axis=4), axis=5)
    if x.ndim == 4:
        return lambda x: np.expand_dims(
                                np.expand_dims(x, axis=4), axis=5)
    if x.ndim == 5:
        return lambda x: np.expand_dims(x, axis=5)
    else:
        return lambda x: x
    
@numba.generated_jit(nopython=True)
def nb_weight_expander(x):
    if x.ndim == 3:
        return lambda x: np.expand_dims(
                            np.expand_dims(
                                np.expand_dims(
                                    np.expand_dims(x, axis=3), 
                                                      axis=4), 
                                                      axis=5), 
                                                      axis=6)
    if x.ndim == 4:
        return lambda x: np.expand_dims(
                            np.expand_dims(
                                np.expand_dims(x, axis=3), axis=4), axis=5)
    if x.ndim == 5:
        return lambda x: np.expand_dims(
                                np.expand_dims(x, axis=4), axis=5)
    if x.ndim == 6:
        return lambda x: np.expand_dims(x, axis=5)
    else:
        return lambda x: x

@numba.generated_jit(nopython=True)
def nb_contractor(x,res):
    if res.ndim==5:
            return lambda x, res: x.shape[0:-1]
    if res.ndim==4:
        if x.ndim==6:
            return lambda x, res: x.shape[0:-2]
        if x.ndim==5:
            return lambda x, res: x.shape[0:-1]
    if res.ndim==3:
        if x.ndim==6:
            return lambda x, res: x.shape[0:-3]
        if x.ndim==5:
            return lambda x, res: x.shape[0:-2]
        if x.ndim==4:
            return lambda x, res: x.shape[0:-1]
    if res.ndim==2:
        if x.ndim==6:
            return lambda x, res: x.shape[0:-4]
        if x.ndim==5:
            return lambda x, res: x.shape[0:-3]
        if x.ndim==4:
            return lambda x, res: x.shape[0:-2]
        if x.ndim==3:
            return lambda x, res: x.shape[0:-1]
    else:
        return lambda x, res: x