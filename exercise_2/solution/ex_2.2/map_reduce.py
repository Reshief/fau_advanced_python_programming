#!/usr/bin/env python3

import argparse
import sys
import random
from typing import List, Callable, Sequence, TypeVar, Optional

R = TypeVar("R")

# Because the type system does not support tuples with unspecified length or type, we need to set ... in the Callable
def map(func: Callable[..., R], *iterable: Sequence) -> List[R]:
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
    # The * in front of parameters for a function means that the variable will be interpreted as a set of parameters to the function
    return [func(*params) for params in zip(*iterable)]


# For the cumulative type
C = TypeVar("C")

# For the next entry
N = TypeVar("N")


def reduce(func: Callable[[C, N], C], iterable: Sequence[N], initial: Optional[C] = None) -> C:
    """Function to mimic reduce() functionality.

    Takes an iterable to repeatedly call `func`(cumulative, next_entry) on. Will start with the first entry as a cumulative starting value, unless another initial value is provided as `initial`

    Parameters
    ----------
    func: Callable[[C, N], C]
        Function accepting a cumulative value and the next entry and returning the next cumulative value
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
