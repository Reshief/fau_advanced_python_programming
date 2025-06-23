from numpy.testing import assert_almost_equal
import unittest
import numpy as np
from typing import List, Callable, Sequence, TypeVar, Optional, Union, Tuple
import matplotlib.pyplot as plt
import logging
from .map_reduce_parallel import map_parallel, reduce_parallel, map_reduce_parallel, map

logger = logging.getLogger(__name__)


def _center_derivative_helper(lower_neighbor_x, lower_neighbor_y, upper_neighbor_x, upper_neighbor_y):
    return (upper_neighbor_y-lower_neighbor_y)/(upper_neighbor_x-lower_neighbor_x)


def get_center_derivative_parallel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Function to calculate the central numerical derivative to the positions x and the function values y.

    Parameters
    ----------
    x, y : np.ndarray of np.float32
        The x and y positions of the points respectively. Both of same length.

    Returns
    --------
    np.ndarray of np.float32
        The resulting ndarray with the center derivatives. Has length of two less than x and y

    """
    # Sort by x positions
    indices = np.argsort(x)
    x = x[indices]
    y = y[indices]

    # Get indices offset by -1
    lower_neighbors_x = x[:-2]
    lower_neighbors_y = y[:-2]

    # Get indices offset by +1
    upper_neighbors_x = x[2:]
    upper_neighbors_y = y[2:]

    # return map_parallel(_center_derivative_helper, lower_neighbors_x, lower_neighbors_y, upper_neighbors_x, upper_neighbors_y, min_executor_data_count=20)
    return map(_center_derivative_helper, lower_neighbors_x, lower_neighbors_y, upper_neighbors_x, upper_neighbors_y)


def _integral_helper(lower_neighbors_x, lower_neighbors_y, upper_neighbors_x, upper_neighbors_y):
    return (upper_neighbors_y+lower_neighbors_y)*(upper_neighbors_x-lower_neighbors_x)


def _integral_sum_helper(x, y):
    return x+y


def get_trapezoid_integral_parallel(x: np.ndarray, y: np.ndarray) -> np.float32:
    """Function to calculate the numerical integral of the function denoted by the (x,y) positions using the trapezoid rule

    Parameters
    ----------
    x, y : np.ndarray of np.float32
        The x and y positions of the function graph respectively. Both of same length.

    Returns
    --------
    np.float32
        The area integral of the discrete function

    """
    # Sort by x positions
    indices = np.argsort(x)
    x = x[indices]
    y = y[indices]

    # Get indices offset by 0
    lower_neighbors_x = x[:-1]
    lower_neighbors_y = y[:-1]

    # Get indices offset by +1
    upper_neighbors_x = x[1:]
    upper_neighbors_y = y[1:]

    return map_reduce_parallel(_integral_helper, _integral_sum_helper, _integral_sum_helper, lower_neighbors_x, lower_neighbors_y, upper_neighbors_x, upper_neighbors_y, reduce_initial=0.0, min_executor_data_count=20)/2.0


if __name__ == '__main__':
    unittest.main()
