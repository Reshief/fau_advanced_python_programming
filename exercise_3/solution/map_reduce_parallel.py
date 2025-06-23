#!/usr/bin/env python3

import argparse
import sys
import os
from typing import List, Callable, Sequence, TypeVar, Optional
from .map_reduce import map, reduce
from concurrent.futures import ProcessPoolExecutor,Future
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

    Returns
    --------
    List of R
        The list of the results obtained through 

    See also
    --------
    map: For similar functionality with lazy evaluation

    """
    in_data = [params for params in zip(*iterable)]

    total_length = len(in_data)

    number_available_threads = os.process_cpu_count()

    num_executors: int = min(number_available_threads, int(
        np.floor(total_length/min_executor_data_count)))

    with ProcessPoolExecutor(max_workers=num_executors) as executor:
        result_futures: list[Future] = []
        for index_executor in range(num_executors):
            # This calculates the begin index using integer arithmetics for rounding
            # Will start at 0 and go up about total_length/num_executors per entry
            begin_index = (total_length*index_executor)/num_executors
            # This calculates the end index. As it uses the index +1 it will eventually exactly reach the end index
            # A neat trick to split the data into individual slices of approximately equal size
            end_index = (total_length*(index_executor+1))/num_executors

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


def reduce_parallel(func: Callable[[C, N], C], combine: Callable[[C, C], C], iterable: Sequence[N], initial: Optional[C] = None) -> C:
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

    Returns
    --------
    C
        The cumulative value after repeated evaluations of `func` on the entire list

    See also
    --------
    functools.reduce: For similar functionality

    """
    # Get an iterator for the sequence
    iterator = iter(iterable)
    # Initialize the cumulative value either with the provided initial value or with the first entry of the sequence
    cumulative = initial if initial is not None else next(iterator)

    # We do not know how long the iteration will take, so assume it will go on forever
    while True:
        try:
            # See if there is another entry
            next_entry = next(iterator)
            # Reduce to a new cumulative value
            cumulative = func(cumulative, next_entry)
        except StopIteration:
            # If there is no entry, break the loop
            break

    return cumulative


# 2.1a is just basic setup
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        sys.argv[0], description="Script to run some basic map and reduce operations. Here, we use list comprehensions to replace map and filter")

    # We keep the log setup just in case
    parser.add_argument(
        "-log",
        choices=["info", "debug", "warn", "error",],
        default="warn",
        help="The log level to be set on the logger"
    )

    # Allow for configuring the length of our lists
    parser.add_argument(
        "-n",
        type=int,
        default=2,
        help="The length of the generated lists for this example"
    )
    args = parser.parse_args(sys.argv[1:])

    loglevel = args.log
    list_length = args.n
