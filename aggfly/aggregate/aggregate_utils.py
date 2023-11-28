import dask.distributed
from dask.distributed import Client
import logging

def is_distributed():
    """
    Returns True if the code is running in a distributed environment, False otherwise.
    
    This function checks if the code is running in a distributed environment by attempting to get the global Dask client.
    If the client is not None, it means that the code is running in a distributed environment.
    
    Returns:
        bool: True if the code is running in a distributed environment, False otherwise.
    """
    client = dask.distributed.client._get_global_client()
    if client is not None:
        return True
    else:
        return False
    
def distributed_client():
    """
    Returns the global Dask distributed client object.

    Returns:
    --------
    client : dask.distributed.Client
        The global Dask distributed client object.
    """
    client = dask.distributed.client._get_global_client()
    return client 

        
def start_dask_client(n_workers: int = 2, threads_per_worker: int = 2, **kwargs):
    """
    Start a dask distributed cluster.

    Args:
        n_workers (int, optional): The number of workers to use. Defaults to 2.
        threads_per_worker (int, optional): The number of threads per worker. Defaults to 2.
        **kwargs: Additional keyword arguments to pass to the dask Client constructor.

    Returns:
        client: A dask distributed client.
    """
    client = Client(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        **kwargs
    )
        
    arg_dict = {
        "n_workers": n_workers, 
        "threads_per_worker": threads_per_worker,
    }
    all_dict = {**arg_dict, **kwargs}
    for k in all_dict.keys():
        client.set_metadata(['args', k], all_dict[k])
    
    return client


def shutdown_dask_client():
    """
    Shutdown the global Dask distributed client object.
    """
    client = dask.distributed.client._get_global_client()
    if client is not None:
        try:
            args = client.get_metadata('args')
        except:
            raise ValueError("Please start Dask client with af.start_dask_client() or run weight calculation without Dask distributed client.")
        client.shutdown()
        return args
    else:
        return None