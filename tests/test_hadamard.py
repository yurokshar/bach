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
def test_gen_sylvester_8(order, data):
    result = HadamardMatrix(order=order)
    expected = numpy.array(data)
    assert numpy.equal(result.data, expected).all()
