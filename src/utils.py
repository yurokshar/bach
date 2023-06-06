import typing
import itertools

import numpy.typing


Matrix: typing.TypeAlias = numpy.typing.NDArray[typing.Any]


def is_power_of_two(n: int) -> bool:
    return (n & (n-1) == 0) and n != 0


def binary_to_gray(n: int) -> int:
    # if n == 0:
    #     return 0
    # m = 1
    # for a, b in zip(bin(n)[2:], bin(n)[3:]):
    #     m <<= 1
    #     m += int(a) ^ int(b)
    # return m

    return n ^ (n >> 1)
