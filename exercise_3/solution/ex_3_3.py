from numpy.testing import assert_almost_equal
import unittest
import numpy as np
from typing import List, Callable, Sequence, TypeVar, Optional, Union, Tuple
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


def get_center_derivative(x: np.ndarray, y: np.ndarray) -> np.ndarray:
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

    return (upper_neighbors_y-lower_neighbors_y)/(upper_neighbors_x-lower_neighbors_x)


def get_trapezoid_integral(x: np.ndarray, y: np.ndarray) -> np.float32:
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

    return np.sum((upper_neighbors_y+lower_neighbors_y)*(upper_neighbors_x-lower_neighbors_x)/2.0)


def get_biscection_root(f: Callable[[np.float32], np.float32], x_min: np.float32, x_max: np.float32, max_steps: int = 20) -> np.float32:
    """Function to find a numerical root of the function f via bisection of the interval [x_min, x_max]

    Parameters
    ----------
    f: Callable[[np.float32], np.float32]
        A function mapping floating point numbers to floating point numbers
    x_min, x_max: np.float32
        The lower and upper interval boundary in which to look for the root of f. f must have different signs on them
    max_steps: int, optional
        The maximum number of bisection steps to perform. Default: 20

    Raises
    --------
    ValueError
        If the function f does not have different signs on the boundaries, then the function will log an error and throw a ValueError

    Returns
    --------
    np.float32
        The closest approximation of the root of f in max_steps steps

    """
    f_x_min = f(x_min)
    f_x_max = f(x_max)
    if f_x_min * f_x_max > 0:
        logging.error(
            "Function f must have different signs on x_min and x_max. Had same sign on both boundaries.")
        raise ValueError("Interval not valid for bisection")

    for _ in range(max_steps):
        x_mid = (x_min+x_max) / 2.0
        f_x_mid = f(x_mid)

        if f_x_mid == 0.0:
            return x_mid

        # Keep boundaries with different signs of f(x)
        if f_x_mid * f_x_max > 0:
            x_max = x_mid
            f_x_max = f_x_mid
        else:
            x_min = x_mid
            f_x_min = f_x_mid

    return (x_min+x_max)/2.0


def lin_f(x, m, t):
    return x*m+t


class TestCenterDerivative(unittest.TestCase):
    def test_random_linear(self):
        data_x = np.random.exponential(10, size=(100,))
        data_y = lin_f(data_x, 14.2, 4)

        result = get_center_derivative(data_x, data_y)
        expected_res = np.zeros((98,))+14.2

        assert_almost_equal(result, expected_res)


class TestTrapezoidIntegral(unittest.TestCase):
    def test_random_linear(self):
        data_x = np.random.exponential(10, size=(100,))
        data_y = lin_f(data_x, 14.2, 4)

        x_min = np.min(data_x)
        x_max = np.max(data_x)

        result = get_trapezoid_integral(data_x, data_y)
        expected_res = 14.2 * (x_max**2-x_min**2)*0.5 + 4*(x_max-x_min)

        assert_almost_equal(result, expected_res)

class TestBisectionRoot(unittest.TestCase):
    def test_random(self):
        data_x = np.random.exponential(10, size=(100,))
        data_y = lin_f(data_x, 14.2, 4)

        f = lambda x: x*4.5 -18
        

        result = get_biscection_root(f, x_min=-3, x_max=20, max_steps=150)
        expected_res = 4

        assert_almost_equal(result, expected_res)


if __name__ == '__main__':
    unittest.main()
