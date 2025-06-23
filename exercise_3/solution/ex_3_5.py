import matplotlib.pyplot as plt
import time
import numpy as np
from .ex_3_4 import get_center_derivative_parallel, get_trapezoid_integral_parallel
from scipy.integrate import trapezoid


def run_sized_derivative_integral(n, rep=3):

    time_own_deriv_parallel = 0
    time_numpy_deriv_parallel = 0
    time_own_integral_parallel = 0
    time_scipy_integral_parallel = 0

    print("Running sample size: ", n)

    x_data = np.random.exponential(10, (n,))
    y_data = np.random.randn(n)

    for _ in range(rep):
        start = time.perf_counter_ns()
        get_center_derivative_parallel(x_data, y_data)
        end = time.perf_counter_ns()
        time_own_deriv_parallel += end-start

        start = time.perf_counter_ns()
        np.gradient(x_data, y_data)
        end = time.perf_counter_ns()
        time_numpy_deriv_parallel += end-start

        start = time.perf_counter_ns()
        get_trapezoid_integral_parallel(x_data, y_data)
        end = time.perf_counter_ns()
        time_own_integral_parallel += end-start

        start = time.perf_counter_ns()
        trapezoid(y_data, x_data)
        end = time.perf_counter_ns()
        time_scipy_integral_parallel += end-start

    return (time_own_deriv_parallel/rep,
            time_numpy_deriv_parallel/rep,
            time_own_integral_parallel/rep,
            time_scipy_integral_parallel/rep)


if __name__ == "__main__":
    n_opts = [int(x)
              for x in np.exp(np.linspace(np.log(4), np.log(1e7), num=30))]

    time_results = np.array([run_sized_derivative_integral(i) for i in n_opts])

    own_deriv = time_results[:, 0]
    np_deriv = time_results[:, 1]
    own_int = time_results[:, 2]
    sp_int = time_results[:, 3]

    plt.clf()

    plt.plot(n_opts, own_deriv, "own deriv parallel")
    plt.plot(n_opts, np_deriv, "np deriv")
    plt.plot(n_opts, own_int, "own trapezoid parallel")
    plt.plot(n_opts, sp_int, "scipy trapezoid")

    plt.xlabel("Set size n")
    plt.ylabel("Runtime in [ns]")
    plt.legend()

    plt.tight_layout()

    plt.savefig("runtime_results.pdf", dpi=300)
