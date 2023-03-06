from __future__ import annotations
from typing import Any
import enum

import numpy
import numpy.typing
from strenum import StrEnum

from src.utils import is_power_of_two



class HadamardMatrix:

    class GenStrategy(StrEnum):
        default = enum.auto()
        sylvester = enum.auto()

    def __init__(
        self: HadamardMatrix,
        order: int = 1,
        data: numpy.typing.NDArray[Any] | None = None,
        assert_is_valid: bool = True,
    ):
        self.order: int = order
        self.data = data or HadamardMatrix.generate(
            order=self.order,
            strategy=HadamardMatrix.GenStrategy.sylvester,
        )

        if assert_is_valid:
            if not HadamardMatrix.is_ok(self.data):
                raise ValueError('Provided data does not contain a valid Hadamard Matrix')

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.data=})'

    def __str__(self) -> str:
        return self.__repr__()

    @staticmethod
    def is_ok(data: numpy.typing.NDArray[Any]) -> bool:
        if not isinstance(data, numpy.ndarray):
            raise TypeError('Value is not of type "numpy.ndarray"')
        if (
            len(data.shape) != 2 or
            data.shape[0] != data.shape[1]
        ):
            return False
        is_linear_independent = (
            numpy.linalg.matrix_rank(data) == data.shape[0]
        )
        return is_linear_independent

    @staticmethod
    def generate(
        order: int,
        strategy: GenStrategy = GenStrategy.default,
    ) -> numpy.typing.NDArray[Any]:
        builder = getattr(HadamardMatrixBuilder, strategy.value, None)
        if builder is None:
            raise ValueError(f'Strategy "{strategy}" does not exist')
        return builder(n=order)


class HadamardMatrixBuilder:
    @staticmethod
    def default(n: int) -> numpy.typing.NDArray[Any]:
        matrix = numpy.zeros((n, n), dtype=int)
        matrix[0, :] = 1
        matrix[:, 0] = 1
        for i in range(1, n):
            for j in range(1, n):
                matrix[i, j] = -matrix[i - 1, j - 1]
        return matrix

    @staticmethod
    def sylvester(n: int) -> numpy.typing.NDArray[Any]:
        if not is_power_of_two(n):
            raise ValueError('Order is not a power of two')
        if n == 1:
            return numpy.array([[1]])
        h = HadamardMatrixBuilder.sylvester(n // 2)
        h_up = numpy.concatenate((h, h), axis=1)
        h_down = numpy.concatenate((h, -h), axis=1)
        return numpy.concatenate((h_up, h_down), axis=0)
