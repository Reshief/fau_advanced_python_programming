import numpy as np
from typing import List, Callable, Sequence, TypeVar, Optional, Union, Tuple
import matplotlib.pyplot as plt


def get_random_array(n: Union[int, Tuple[int]], mean: np.float32 = 0.0, sigma: np.float32 = 1.0, ) -> np.ndarray:
    """Function to obtain a numpy array of length/shape n filled with normal distributed random numbers with the denoted mean and sigma

    Parameters
    ----------
    n : int or Tuple[int]
        The length (if int) or shape (if tuple) of the target array of random numbers
    mean: np.float32, optional
        The desired mean of the numbers
    sigma: np.float32, optional
        The desired standard deviation of numbers

    Returns
    --------
    np.ndarray of np.float32
        The resulting list of randomly generated values

    See also
    --------
    np.ndarray, np.random.randn

    """
    if isinstance(n, int):
        n = (n,)
    return np.random.randn(*n)*sigma + mean


if __name__ == "__main__":
    random_number_2d_array = get_random_array((2, 100))
    random_number_1d_array = get_random_array(100)

    # This starts from the second entry
    every_other_odd = random_number_1d_array[1::2]
    # This starts from the first entry
    every_other_even = random_number_1d_array[0::2]

    index_half = int(len(random_number_1d_array)/2)
    first_half = random_number_1d_array[:index_half]
    second_half = random_number_1d_array[index_half:]

    last_third_index = int(2*len(random_number_2d_array[0]/3.))
    # Make sure, we are at an odd index
    last_third_index = last_third_index + (1 - (last_third_index % 2))
    # Set every other entry in the last third of each row to zero
    random_number_2d_array[:, last_third_index::2] = 0

    # With a slightly different interpretation, we could also do the following instead
    last_third_index = int(2*len(random_number_2d_array[0]/3.))
    random_number_2d_array[last_third_index:, 1::2] = 0

    loaded_data = np.loadtxt(
        "./ex_3.2.txt",
        ndmin=2,  # To make result at least 2d
        # only use all columns but the third. We might need to build this tuple dynamically for an arbitrary file
        usecols=(0, 1, 3, 4),
    )

    random_number_2d_array = get_random_array((2, 1000000))
    random_number_1d_array = get_random_array(1000000)

    hist_1d, bin_edges_1d = np.histogram(random_number_1d_array, density=True)
    hist_2d, bin_edges_2d_x, bin_edges_2d_y = np.histogram2d(
        random_number_2d_array[0], random_number_2d_array[1], density=True)

    bin_centers_1d = (bin_edges_1d[:-1]+bin_edges_1d[1:])/2.
    bin_centers_2d_x = (bin_edges_2d_x[:-1]+bin_edges_2d_x[1:])/2.
    bin_centers_2d_y = (bin_edges_2d_y[:-1]+bin_edges_2d_y[1:])/2.

    plt.clf()
    plt.scatter(bin_centers_1d, hist_1d, s=4)
    plt.show()
    plt.savefig("1d_hist.pdf", dpi=300)

    plt.clf()
    y, x = np.meshgrid(bin_centers_2d_x, bin_centers_2d_y)
    plt.pcolormesh(x, y, hist_2d, cmap="inferno")
    plt.show()
    plt.savefig("2d_hist.pdf", dpi=300)

    # Fit data with a normal distribution and print the resulting parameters
    from scipy.stats import norm as normal_distr
    from scipy.optimize import curve_fit

    params, cov = curve_fit(
        normal_distr.pdf, bin_centers_1d, hist_1d, p0=(2, 2))

    errors = np.sqrt(np.diag(cov))

    print(params, "+/-", errors)
