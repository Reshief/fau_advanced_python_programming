import numpy as np
from typing import List, Callable, Sequence, TypeVar, Optional, Union, Tuple

def get_random_array(n:Union[int, Tuple[int]], mean: np.float32 = 0.0, sigma: np.float32 = 1.0, ) -> np.ndarray:
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

