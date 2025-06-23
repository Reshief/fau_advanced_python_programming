import unittest
import numpy as np
from .map_reduce_parallel import map_parallel, reduce_parallel, map_reduce_parallel
from numpy.testing import assert_almost_equal


def calc_square(x: np.float32) -> np.float32:
    return x*x


def calc_sum(x: np.float32, y: np.float32) -> np.float32:
    return x+y


class TestMapMethod(unittest.TestCase):
    def test_single_iterator(self):
        data_list = np.random.exponential(10, size=(100,))
        data_list_result = data_list**2
        result_list = np.array(map_parallel(calc_square, data_list))
        assert_almost_equal(result_list, data_list_result)

    def test_multiple_iterator(self):
        data_list_x = np.random.exponential(10, size=(100,))
        data_list_y = np.random.exponential(20, size=(100,))

        data_list_result = data_list_x+data_list_y
        result_list = np.array(map_parallel(
            calc_sum, data_list_x, data_list_y))
        assert_almost_equal(result_list, data_list_result)

    def test_empty_list(self):
        data_list = []
        result_list = map_parallel(calc_square, data_list)
        self.assertEqual(len(result_list), 0)

    def test_wrong_number_of_iterators(self):
        data_list_x = np.random.exponential(10, size=(100,))
        data_list_y = np.random.exponential(20, size=(100,))

        # Make sure of the error being triggered by the wrong number of arguments
        with self.assertRaises(TypeError):
            map_parallel(calc_square, data_list_x, data_list_y)

        # Make sure of the error being triggered by the wrong number of arguments
        with self.assertRaises(TypeError):
            map_parallel(calc_sum, data_list_x)


class TestReduceMethod(unittest.TestCase):
    def test_empty_list_initial(self):
        data_list = []
        result_1 = reduce_parallel(calc_sum, calc_sum, data_list, initial=1)
        self.assertEqual(result_1, 1)

    def test_simple_addition_multiple_workers(self):
        data_list = np.random.exponential(10, size=(100,))
        expected_result = np.sum(data_list)
        result_sum = np.array(reduce_parallel(calc_sum, calc_sum, data_list))
        assert_almost_equal(expected_result, result_sum)

    def test_less_than_one_executor(self):
        data_list = np.random.exponential(10, size=(10,))
        expected_result = np.sum(data_list)
        result_sum = np.array(reduce_parallel(
            calc_sum, calc_sum, data_list, min_executor_data_count=100))
        assert_almost_equal(expected_result, result_sum)

    def test_no_input_without_initial(self):
        data_list = []
        with self.assertRaises(StopIteration):
            reduce_parallel(calc_sum, calc_sum, data_list)


def power2(x: np.float32) -> np.float32:
    return x**2


def power3(x: np.float32) -> np.float32:
    return x**3


def power7(x: np.float32) -> np.float32:
    return x**7


class TestMapReduceMethod(unittest.TestCase):
    def test_power_2(self):
        data_list = np.random.exponential(10, size=(100,))
        expected_res = np.sum(data_list**2)
        result_1 = map_reduce_parallel(
            power2, calc_sum, calc_sum, data_list, reduce_initial=0)
        assert_almost_equal(result_1, expected_res)
        
    def test_power_3(self):
        data_list = np.random.exponential(10, size=(50,))
        expected_res = np.sum(data_list**3)
        result_1 = map_reduce_parallel(
            power3, calc_sum, calc_sum, data_list, reduce_initial=0)
        assert_almost_equal(result_1, expected_res)
        
    def test_power_7(self):
        data_list = np.random.exponential(1, size=(4,))
        expected_res = np.sum(data_list**7)
        result_1 = map_reduce_parallel(
            power7, calc_sum, calc_sum, data_list, reduce_initial=0, min_executor_data_count=20)
        assert_almost_equal(result_1, expected_res)

    def test_no_input_without_initial(self):
        data_list = []
        with self.assertRaises(StopIteration):
            map_reduce_parallel(
            power7, calc_sum, calc_sum, data_list, min_executor_data_count=20)


if __name__ == '__main__':
    unittest.main()
