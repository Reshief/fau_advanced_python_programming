#!/usr/bin/env python3

import argparse
import sys
import os
from typing import List, Callable, Sequence, TypeVar, Optional
from .map_reduce import map, reduce
from concurrent.futures import ProcessPoolExecutor, Future
import numpy as np

R = TypeVar("R")

# Because the type system does not support tuples with unspecified length or type, we need to set ... in the Callable


def _map_helper_func(func, data):
    return [func(*params) for params in data]


def map_parallel(func: Callable[..., R], *iterable: Sequence, min_executor_data_count=5) -> List[R]:
    """Function to apply the provided function `func` to all entries of tuples in `iterable` and combine the results into a list

    The function `func` must accept one argument for each `iterable` provided.
    Evaluation will be eager compared to the built-in map function.

    Parameters
    ----------
    func : Callable[..., R]
        A function, accepting exactly as many arguments as there are positional iterables in `iterable`. The result will be an entry in the resulting list
    *iterable
        an arbitrary number of sequences. `func` will be applied to all tuples created from entries at the same positions in `iterable`. 
        All sequences provided for `iterable` must have the same length.
        The types of values provided by the sequences must match the parameters of `func` in the order that the sequences are provided.
    min_executor_data_count: int, optional
        The minimum chunk size of data assigned to each executor process. 

    Returns
    --------
    List of R
        The list of the results obtained through parallel application of func to the entries in *iterable

    See also
    --------
    map: For similar functionality with lazy evaluation

    """
    in_data = [params for params in zip(*iterable)]

    total_length = len(in_data)

    number_available_threads = os.cpu_count()

    # We need at least one but at most as many executors as we have logical CPU cores
    num_executors: int = max(1, min(number_available_threads, int(
        np.floor(total_length/min_executor_data_count))))

    with ProcessPoolExecutor(max_workers=num_executors) as executor:
        result_futures: list[Future] = []
        for index_executor in range(num_executors):
            # This calculates the begin index using integer arithmetics for rounding
            # Will start at 0 and go up about total_length/num_executors per entry
            begin_index = int((total_length*index_executor)/num_executors)
            # This calculates the end index. As it uses the index +1 it will eventually exactly reach the end index
            # A neat trick to split the data into individual slices of approximately equal size
            end_index = int((total_length*(index_executor+1))/num_executors)

            # Get the slice of data
            data = in_data[begin_index:end_index]

            # Apply the map function in parallel
            result_futures.append(executor.submit(
                _map_helper_func, func, data))

        # Collect the results and return the combined list
        result = []
        for future in result_futures:
            result.extend(future.result())

        return result


# For the cumulative type
C = TypeVar("C")

# For the next entry
N = TypeVar("N")


def reduce_parallel(func: Callable[[C, N], C], combine: Callable[[C, C], C], iterable: Sequence[N], initial: Optional[C] = None, min_executor_data_count=5) -> C:
    """Function to mimic reduce() functionality.

    Takes an iterable to repeatedly call `func`(cumulative, next_entry) on. Will start with the first entry as a cumulative starting value, unless another initial value is provided as `initial`

    Parameters
    ----------
    func: Callable[[C, N], C]
        Function accepting a cumulative value and the next entry and returning the next cumulative value
    combine: Callable[[C, C], C]
        Function combining the results of calls to func() into one aggregate result. Allows for the parallel execution of reduce and then eventual recombination before returning.
    iterable: Sequence of N
        The values to be reduced into one cumulative value
    initial: C, optional
        An optional initial cumulative value. If not provided, the first entry of `iterable` will be used as an initial cumulative value. Then the types N and C must match.
    min_executor_data_count: int, optional
        The minimum chunk size of data assigned to each executor process. 

    Returns
    --------
    C
        The cumulative value after repeated evaluations of `func` on the entire list

    See also
    --------
    functools.reduce: For similar functionality

    """

    in_data = [params for params in iterable]

    total_length = len(in_data)

    number_available_threads = os.cpu_count()

    # We need at least one but at most as many executors as we have logical CPU cores
    num_executors: int = max(1, min(number_available_threads, int(
        np.floor(total_length/min_executor_data_count))))

    with ProcessPoolExecutor(max_workers=num_executors) as executor:
        result_futures: list[Future] = []
        for index_executor in range(num_executors):
            # This calculates the begin index using integer arithmetics for rounding
            # Will start at 0 and go up about total_length/num_executors per entry
            begin_index = int((total_length*index_executor)/num_executors)
            # This calculates the end index. As it uses the index +1 it will eventually exactly reach the end index
            # A neat trick to split the data into individual slices of approximately equal size
            end_index = int((total_length*(index_executor+1))/num_executors)

            # Get the slice of data
            data = in_data[begin_index:end_index]

            # Apply the reduce function in parallel to the individual slices
            result_futures.append(executor.submit(
                reduce, func, data, initial))

        # Collect the results and return the combined list
        result = None
        for future in result_futures:
            # If we have no prior entry, we use the current result as a basis.
            # For all subsequent futures, we use the combine function to merge
            # with an existing resul
            result = future.result() if result is None else combine(result, future.result())

        return result


# For map result type
R = TypeVar("R")
# For the cumulative type
C = TypeVar("C")
# For the next entry
N = TypeVar("N")


def map_reduce_parallel(func_map: Callable[..., R], reduce_func: Callable[[N, R], N], combine_func: Callable[[N, N], N], *iterable: Sequence, reduce_initial: N = None, min_executor_data_count=5) -> List[N]:
    """Function to apply first apply a map with func_map to the input iterables and then apply reduce with reduce_func and combine_func to the result.

    The function `func_map` must accept one argument for each `iterable` provided.
    Evaluation will be eager.
    The reduce() function will be called with reduce_func, combine_func and reduce_initial as arguments. 
    Overall, this function is equivalent to 
    reduce_parallel(reduce_func, combine_func, map_parallel(func_map, iterable, min_executor_data_count), min_executor_data_count, initial=reduce_initial)

    Parameters
    ----------
    func_map : Callable[..., R]
        A function, accepting exactly as many arguments as there are positional iterables in `iterable`. The result will be an entry in the resulting list
    *iterable
        an arbitrary number of sequences. `func` will be applied to all tuples created from entries at the same positions in `iterable`. 
        All sequences provided for `iterable` must have the same length.
        The types of values provided by the sequences must match the parameters of `func` in the order that the sequences are provided.
    min_executor_data_count: int, optional
        The minimum chunk size of data assigned to each executor process. 
    reduce_func: Callable[[N, R], N]
        Function accepting a cumulative value and the next entry and returning the next cumulative value
    combine_func: Callable[[N, N], N]
        Function combining the results of calls to reduce_func() into one aggregate result. Allows for the parallel execution of reduce and then eventual recombination before returning.
    reduce_initial: N, optional
        An optional initial cumulative value. If not provided, the first entry of the result of the map call will be used as an initial cumulative value. Then the types N and R must match.

    Returns
    --------
    List of N
        The list of the results obtained through first calling map and then reduce

    See also
    --------
    map_parallel, reduce_parallel

    """

    return reduce_parallel(reduce_func, combine_func, map_parallel(func_map, *iterable, min_executor_data_count=min_executor_data_count), initial=reduce_initial, min_executor_data_count=min_executor_data_count)
