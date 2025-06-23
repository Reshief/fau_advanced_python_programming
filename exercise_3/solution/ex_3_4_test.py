from numpy.testing import assert_almost_equal
import unittest
import numpy as np
import logging
from .ex_3_4 import get_center_derivative_parallel, get_trapezoid_integral_parallel

logger = logging.getLogger(__name__)

def lin_f(x, m, t):
    return x*m+t


class TestCenterDerivativeParallel(unittest.TestCase):
    def test_random_linear(self):
        n = 501
        data_x = np.linspace(0, n-1, n)
        data_y = lin_f(data_x, 123.4, -234.33)

        expected_res = np.gradient(data_y, data_x,)

        result = get_center_derivative_parallel(data_x, data_y)
        # Need to cut off padding at beginning and end of reference
        assert_almost_equal(result, expected_res[1:-1])

    def test_random_linear_2(self):
        n = 501
        data_x = np.random.exponential(10, (n,))
        m = 3472.5553
        data_y = lin_f(data_x, m, -234.33)

        result = get_center_derivative_parallel(data_x, data_y)
        expected_res = np.zeros_like(result)+m
        # Need to cut off padding at beginning and end of reference
        assert_almost_equal(result, expected_res)

    def test_random_constant(self):
        n = 501
        data_x = np.random.exponential(10, (n,))
        m = 0.0
        data_y = lin_f(data_x, m, -234.33)

        result = get_center_derivative_parallel(data_x, data_y)
        expected_res = np.zeros_like(result)+m
        # Need to cut off padding at beginning and end of reference
        assert_almost_equal(result, expected_res)


class TestTrapezoidIntegralParallel(unittest.TestCase):
    def test_random(self):
        data_x = np.linspace(0, 100, 100)
        data_y = np.random.exponential(10, size=(100,))

        from scipy.integrate import trapezoid
        result = get_trapezoid_integral_parallel(data_x, data_y)
        expected_res = trapezoid(data_y, data_x)

        assert_almost_equal(result, expected_res)

    def test_linear(self):
        data_x = np.random.exponential(10, size=(100,))
        data_y = lin_f(data_x, 14.2, 4)

        x_min = np.min(data_x)
        x_max = np.max(data_x)

        result = get_trapezoid_integral_parallel(data_x, data_y)
        expected_res = 14.2 * (x_max**2-x_min**2)*0.5 + 4*(x_max-x_min)

        assert_almost_equal(result, expected_res)

    def test_constant(self):
        data_x = np.random.exponential(10, size=(100,))
        data_y = lin_f(data_x, 0.0, 4)

        x_min = np.min(data_x)
        x_max = np.max(data_x)

        result = get_trapezoid_integral_parallel(data_x, data_y)
        expected_res = 4*(x_max-x_min)

        assert_almost_equal(result, expected_res)


if __name__ == '__main__':
    unittest.main()
