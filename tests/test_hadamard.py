import pytest
import numpy
import numpy.typing

from src.hadamard import HadamardMatrix


@pytest.mark.parametrize('order,data', (
    (1, [[1]]),
    (2, [
        [1, 1],
        [1, -1],
    ]),
    (4, [
        [1,  1,  1,  1],
        [1, -1,  1, -1],
        [1, 1, -1, -1],
        [1, -1, -1,  1],
    ]),
    (8, [
        [1,  1,  1,  1,  1,  1,  1,  1],
        [1, -1,  1, -1,  1, -1,  1, -1],
        [1,  1, -1, -1,  1,  1, -1, -1],
        [1, -1, -1,  1,  1, -1, -1,  1],
        [1,  1,  1,  1, -1, -1, -1, -1],
        [1, -1,  1, -1, -1,  1, -1,  1],
        [1,  1, -1, -1, -1, -1,  1,  1],
        [1, -1, -1,  1, -1,  1,  1, -1],
    ]),
))
def test_gen_sylvester_until8(order, data):
    result = HadamardMatrix(order=order)
    expected = numpy.array(data)
    assert numpy.equal(result.data, expected).all()


def test_gen():
    m1 = HadamardMatrix(order=64)
    m2 = HadamardMatrix(order=64, strategy=HadamardMatrix.GenStrategy.matrix16_1)

    assert numpy.not_equal(m1.data, m2.data).any()
