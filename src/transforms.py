import typing

from loguru import logger
from PIL import Image, ImageColor
import numpy
import numpy.typing

from src.utils import Matrix, is_power_of_two
from src.hadamard import HadamardMatrix


numpy.set_printoptions(formatter={'float_kind': "{:.2f}".format})


def walsh_hadamard_image_transform(
    target: Image.Image,
    transformer: HadamardMatrix,
    filters: list[typing.Callable[[Matrix, HadamardMatrix], Matrix]] | None = None,
) -> tuple[Matrix, Matrix, Image.Image]:

    if target.size[0] != target.size[1]:
        size = max(target.size)
        img = Image.new(
            mode=target.mode,
            size=(size, size),
            color=ImageColor.getrgb('white'),
        )
        img.paste(
            im=target,
            box=(0, 0),
        )
        target = img

    # if transformer is None:
    #     transformer = HadamardMatrix(order=target.size[0])
    #     transformer.hadamard_to_walsh()
    
    image_data = numpy.array(target, dtype=numpy.float64)

    spectrum = transformer.forward(signal=image_data)

    if filters is None:
        filters = []

    orig_spectrum = spectrum.copy()

    # logger.info(f'Spectrum:\n{orig_spectrum}')
    for filter in filters:
        # logger.info(f'Applying: {filter}')
        spectrum = filter(spectrum, transformer)
        if spectrum.shape != transformer.data.shape:
            raise RuntimeError('Spectrum shape changed after applying filter')
        # logger.info(f'Spectrum:\n{spectrum}')

    transformed_image_data = transformer.inverse(spectrum=spectrum)
    # logger.info(f'Result:\n{transformed_image_data}')

    transformed_image = Image.fromarray(
        numpy.uint8(transformed_image_data),
        mode=target.mode,
    )

    return orig_spectrum, spectrum, transformed_image
