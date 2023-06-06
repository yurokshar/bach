from __future__ import annotations
from typing import Any
import enum

import numpy
import numpy.typing
from strenum import StrEnum

from src.utils import (
    is_power_of_two,
    binary_to_gray,
    Matrix,
)


class HadamardMatrix:

    class GenStrategy(StrEnum):
        default = enum.auto()
        sylvester = enum.auto()
        matrix16_1 = enum.auto()
        matrix32_1 = enum.auto()

    class Ordering(StrEnum):
        walsh = enum.auto()
        hadamard = enum.auto()
        paley = enum.auto()

    def __init__(
        self: HadamardMatrix,
        order: int = 1,
        data: numpy.typing.NDArray[Any] | None = None,
        assert_is_valid: bool = True,
        strategy: HadamardMatrix.GenStrategy = GenStrategy.sylvester,
        ordering: HadamardMatrix.Ordering = Ordering.walsh,
    ):
        self.order: int = order
        self.data = data or HadamardMatrix.generate(
            order=self.order,
            strategy=strategy,
        )
        if ordering == HadamardMatrix.Ordering.hadamard:
            pass
        elif ordering == HadamardMatrix.Ordering.walsh:
            self.convert_to_walsh()
        elif ordering == HadamardMatrix.Ordering.paley:
            self.convert_to_paley()

        if assert_is_valid:
            if not HadamardMatrix.is_ok(self.data):
                raise ValueError(
                    'Provided data does not contain a valid Hadamard Matrix')


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.data=})'

    def __str__(self) -> str:
        return self.__repr__()

    def convert_to_walsh(self):
        def sign_changes(row) -> int:
            return numpy.sum(numpy.abs(numpy.diff(numpy.sign(row)))) // 2

        counts = numpy.apply_along_axis(sign_changes, 1, self.data)
        self.data = self.data[numpy.argsort(counts)]

    def convert_to_paley(self):
        data = numpy.zeros_like(self.data)
        for i in range(self.data.shape[0]):
            j = binary_to_gray(i)
            data[i] = self.data[j]
        self.data = data

    def forward(self, signal: Matrix) -> Matrix:
        if self.data.shape != signal.shape:
            raise ValueError(
                'Input signal size must match the size of Hadamard Matrix: %s, %s', self.data.shape, signal.shape)

        return (self.data @ signal @ self.data) / (self.data.shape[0] ** 1)

    def inverse(self, spectrum: Matrix) -> Matrix:
        if self.data.shape != spectrum.shape:
            raise ValueError(
                'Input spectrum size must match the size of Hadamard Matrix: %s, %s', self.data.shape, spectrum.shape)

        return (self.data @ spectrum @ self.data) / (self.data.shape[0] ** 1)

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
    def default(n: int) -> Matrix:
        matrix = numpy.zeros((n, n), dtype=float)
        matrix[0, :] = 1
        matrix[:, 0] = 1
        for i in range(1, n):
            for j in range(1, n):
                matrix[i, j] = -matrix[i - 1, j - 1]
        return matrix

    @staticmethod
    def sylvester(n: int, _base=(1, [[1]])) -> Matrix:
        if not is_power_of_two(n):
            raise ValueError('Order is not a power of two')
        if n == _base[0]:
            return numpy.array(_base[1], dtype=float)
        h = HadamardMatrixBuilder.sylvester(n // 2, _base=_base)
        h_up = numpy.concatenate((h, h), axis=1)
        h_down = numpy.concatenate((h, -h), axis=1)
        return numpy.concatenate((h_up, h_down), axis=0)

    MAT_16_1 = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, -1, 1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1, -1],
        [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
        [1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1],
        [1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1],
        [1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1],
        [1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1],
        [1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, -1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1],
        [1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1],
        [1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1],
        [1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1],
        [1, -1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1],
        [1, 1, -1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1, -1, -1, 1],
        [1, -1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1],
    ]

    MAT_32_1 = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1,
            1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
        [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1,
            1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
        [1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1,
            1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1],
        [1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1,
            1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1],
        [1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1,
            1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1],
        [1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1,
            1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1],
        [1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1,
            1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1],
        [1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
            1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1,
            1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1],
        [1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1,
            1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1],
        [1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1,
            1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1],
        [1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1,
            1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1],
        [1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1,
            1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1],
        [1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1,
            1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1],
        [1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1,
            1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -
            1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, -
            1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1],
        [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, -
            1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1],
        [1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -
            1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1],
        [1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -
            1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1],
        [1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -
            1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1],
        [1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -
            1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1],
        [1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -
            1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -
            1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, -
            1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1],
        [1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, -
            1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1],
        [1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -
            1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1],
        [1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -
            1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1],
        [1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, -
            1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1],
        [1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, -
            1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1],
        [1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -
            1, 1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1],
    ]

    @staticmethod
    def matrix16_1(n: int) -> Matrix:
        if not is_power_of_two(n):
            raise ValueError('Order is not a power of two')
        if n < len(HadamardMatrixBuilder.MAT_16_1):
            raise ValueError('Order is too small for this builder')
        return HadamardMatrixBuilder.sylvester(n=n, _base=(16, HadamardMatrixBuilder.MAT_16_1))

    @staticmethod
    def matrix32_1(n: int) -> Matrix:
        if not is_power_of_two(n):
            raise ValueError('Order is not a power of two')
        if n < len(HadamardMatrixBuilder.MAT_32_1):
            raise ValueError('Order is too small for this builder')
        return HadamardMatrixBuilder.sylvester(n=n, _base=(32, HadamardMatrixBuilder.MAT_32_1))