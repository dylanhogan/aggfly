# This script sets up and manages the Dask distributed computing environment.
# It includes functions to check if the environment is distributed, retrieve the global Dask client,
# start a Dask client with specified configurations, and shut down the client when needed.

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

        
def start_dask_client(
    n_workers: int = 2,
    threads_per_worker: int = 2,
    cap_numba_threads: int = 1,
    **kwargs,
):
    """
    Start a dask distributed cluster for aggfly.

    Args:
        n_workers (int, optional): The number of workers to use. Defaults to 2.
        threads_per_worker (int, optional): The number of threads per worker. Defaults to 2.
        cap_numba_threads (int, optional): Numba threads to allow per worker process.
            Each worker otherwise defaults to one numba thread per core, so with a
            process cluster ``n_workers`` x cores_per_worker numba threads would
            oversubscribe the machine. The numba temporal kernels get their
            parallelism from dask fanning spatial blocks across workers, so 1
            thread/worker is the safe default. Pass ``None`` to leave numba's
            default untouched. Defaults to 1.
        **kwargs: Additional keyword arguments to pass to the dask Client constructor.

    Returns:
        client: A dask distributed client.
    """
    client = Client(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        **kwargs
    )

    # Cap numba threads on every worker to avoid process-cluster oversubscription.
    # Best-effort: a worker without numba, or an older numba, must not break startup.
    if cap_numba_threads is not None:
        try:
            import numba
            client.run(numba.set_num_threads, cap_numba_threads)
        except Exception as exc:  # noqa: BLE001 - startup must not fail on this
            logging.warning("Could not cap numba threads on workers: %s", exc)

    arg_dict = {
        "n_workers": n_workers,
        "threads_per_worker": threads_per_worker,
        "cap_numba_threads": cap_numba_threads,
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
