#!/usr/bin/env python3

import argparse
import sys
import random
from typing import List


def create_int_list(length: int, min_num: int = -1000, max_num: int = 1000) -> List[int]:
    """A helper function to create a list of random integers

    Needs the number of entries and optionally a maximum and minimum number of the range from which the numbers will be drawn.

    Parameters
    ----------
    length: int
        The desired length of the returned list
    min_num: int, optional
        The minimum value of generated integers. Default: -1000
    max_num: int, optional
        The maximum value of generated integers. Default: 1000

    Returns
    --------
    List of int
        The list of randomly generated integers

    """
    return [random.randint(min_num, max_num) for _ in range(length)]


def create_float_list(length: int, min_num: float = -10.0, max_num: float = 10.0) -> List[float]:
    """A helper function to create a list of random floating point numbers

    Needs the number of entries and optionally a maximum and minimum number of the range from which the numbers will be drawn.

    Parameters
    ----------
    length: int
        The desired length of the returned list
    min_num: int, optional
        The minimum value of generated floats. Default: -10
    max_num: float, optional
        The maximum value of generated floats. Default: 10

    Returns
    --------
    List of float
        The list of randomly generated floats

    """
    return [random.uniform(min_num, max_num) for _ in range(length)]


# 2.1a is just basic setup
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        sys.argv[0], description="Script to run some basic map and reduce operations")

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

    #
    # 2.1b Map function
    #
    # 2.1b 1.

    intlist = create_int_list(list_length)

    def triple_numbers(x: int) -> int:
        return x*3

    tmp = map(triple_numbers, intlist)
    # This uses a lambda function instead:
    tmp_lambda = map(lambda x: x*3, intlist)

    # Print the reference numbers for a test
    print(intlist)

    # Prints a map-object instead of a list
    print(tmp)

    # Prints the list of tripled integers
    print(list(tmp))

    # Prints an empty list, because the iterator has reached the end of the list in the previous line
    print(list(tmp))

    # Create the second temporary value
    tmp2 = map(triple_numbers, intlist)
    # Append additional number to initial list
    intlist.append(1)
    # This will print one more entry, namely the additional 1 also being tripled, despite it being
    # appended after the call to map. This demonstrates the lazy execution and its pitfalls.
    # The result of map is only calculated here, when we try and turn it into a list
    print(list(tmp2))

    # 2.1b 2.

    stringlist = ["We want you to get familiar with the map ",
                  "and reduce function to process data sets first"
                  "There are certain disadvantages to using them compared to list ",
                  "comprehensions and generators which we will illustrate further in this exercise.",
                  "Eventually, you will build an improved version of both the map and the filter functions.",
                  "In this context, we will also illustrate the use ",
                  "of so-called lambda functions to simplify the code."]

    def split_string(x: str) -> List[str]:
        return x.split()

    result = list(map(split_string, stringlist))
    # This uses a lambda function instead:
    result_lambda = list(map(lambda x: x.split(), stringlist))

    # Print the initial string list
    print(stringlist)
    # Print the list with strings being split into words
    print(result)

    # 2.1b 3.

    floatlist_a = create_float_list(list_length)
    floatlist_b = create_float_list(list_length)

    print("Initial:", floatlist_a, floatlist_b)

    def add(a, b):
        return a+b

    def sub(a, b):
        return a-b

    result_add = list(map(add, floatlist_a, floatlist_b))
    # This uses a lambda function instead
    result_add_lambda = list(map(lambda x, y: x+y, floatlist_a, floatlist_b))
    result_sub = list(map(sub, floatlist_a, floatlist_b))
    # This uses a lambda function instead
    result_sub_lambda = list(map(lambda x, y: x-y, floatlist_a, floatlist_b))

    print("Addition:", result_add)
    print("Subtraction:", result_sub)

    # 2.1b 4.

    intlist_a = create_int_list(list_length)
    intlist_b = create_int_list(list_length)
    intlist_c = create_int_list(list_length)
    print("Initial:", intlist_a, intlist_b, intlist_c)

    def add_three(a, b, c):
        return a+b+c

    result_add = list(map(add_three, intlist_a, intlist_b, intlist_c))
    # Same operation but with a lambda function
    result_add_lambda = list(
        map(lambda x, y, z: x+y+z, intlist_a, intlist_b, intlist_c))

    print("Result without lambda:", result_add)
    print("Result with lambda:", result_add_lambda)

    #
    # 2.1c Reduce functionality
    #
    from functools import reduce

    # 2.1c 1.
    intlist = create_int_list(list_length)

    # Calculate the total sum of the previous list
    # The start value 0 is optional for this
    total_sum = reduce(add, intlist, 0)

    print("Sum(", intlist, ") =", total_sum)

    # 2.1c 2.
    intlist = create_int_list(list_length)

    # Calculate the first entry minus all others
    first_minus_rest = reduce(sub, intlist)

    print("First minus rest (", intlist, ") =", first_minus_rest)

    # 2.1c 2.
    intlist = create_int_list(list_length)

    def alt_sub(a, b):
        return -a + b

    # Calculate the first entry minus all others
    alternating_sum = reduce(alt_sub, intlist)

    # if the list has an even length, we need to negate the result:
    if len(intlist) % 2 == 0:
        alternating_sum = -alternating_sum

    print("Alternating Sum (", intlist, ") =", alternating_sum)

    #
    # 2.1d Map, filter, and reduce functionality
    #

    # 2.1d 1.
    intlist = create_int_list(list_length, -10, 10)

    def square(x):
        return x*x

    # Calculate the total sum of the squares of the previous list
    # The start value 0 is optional for this
    total_square_sum = reduce(add, map(square, intlist), 0)

    print("Square sum(", intlist, ") =", total_square_sum)

    # 2.1d 2.

    def is_odd(x):
        return x % 2 == 1

    # Here, we additionally filter out all functions that are not odd before applying the previous calculation
    total_square_sum = reduce(add, map(square, filter(is_odd, intlist)), 0)

    print("Square sum of odds(", intlist, ") =", total_square_sum)

    # 2.1d 3.

    def has_odd_num_digits(x):
        # Remember, there may be negative numbers
        if x[0] == "-":
            # Account for the minus sign
            return len(x)-1 % 2 == 1
        else:
            return len(x) % 2 == 1

    # We first convert to strings with map
    # Then we filter whether strings have odd numbers of digits, while accounting for the possible leading minus
    # Then we combine the numbers into one comma-separated list
    # Here, we need the initial value "0" as asked in the problem statement for the prefix
    stringified = reduce(
        lambda x, y: x + ", " + y, filter(has_odd_num_digits, map(str, intlist)), "0")

    print("Only odd digit count stringified(", intlist, ") =", stringified)
