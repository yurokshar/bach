from pathlib import Path
import struct

from PIL import Image
import numpy as np
from skimage import color

from src.hadamard import HadamardMatrix


def padding_size(S, a):
    return ((S - 1) // a + 1) * a - S


def slice(S, bsize):
    w, h = S.shape

    h_padding = padding_size(h, bsize)
    S = np.concatenate((S, np.zeros((h_padding, S.shape[1]))), 0)

    w_padding = padding_size(w, bsize)
    S = np.concatenate((S, np.zeros((S.shape[0], w_padding))), 1)

    blocks = []
    for row in np.vsplit(S, S.shape[0] / bsize):
        blocks.extend(np.hsplit(row, row.shape[1] / bsize))


    return blocks



def compress_rgb_image(path):
    image = Image.open(path, formats=['png', 'tiff', 'bmp'])
    w, h = image.size

    ycbcr = image.convert('YCbCr')
    data = np.array(ycbcr)
    print(data.shape)

    y_bsize = 8
    cb_bsize = 16
    cr_bsize = 16

    y = slice(data[:, :, 0], y_bsize)
    cb = slice(data[:, :, 1], cb_bsize)
    cr = slice(data[:, :, 2], cr_bsize)

    H_y = HadamardMatrix(order=y_bsize)
    S_y = list(map(H_y.forward, y))

    H_cb = HadamardMatrix(order=cb_bsize)
    S_cb = list(map(H_cb.forward, cb))

    H_cr = HadamardMatrix(order=cr_bsize)
    S_cr = list(map(H_cr.forward, cr))

    y_packrate = 4
    cb_packrate = 4
    cr_packrate = 4
    output = path + '.pak'
    header = struct.pack(
        '<IIHHHHHHHHH',
        w, h,
        y_bsize, y_packrate, len(S_y),
        cb_bsize, cb_packrate, len(S_cb),
        cr_bsize, cr_packrate, len(S_cr),
    )
    blocks = [
        [
            block[:packrate, :packrate]
            for block in data
        ]
        for data, packrate in zip(
            (S_y, S_cb, S_cr),
            (y_packrate, cb_packrate, cr_packrate),
        )
    ]
    with open(output, 'wb') as file:
        file.write(header)
        for blockset in blocks:
            for block in blockset:
                data = block.reshape(-1).tolist()
                raw = struct.pack('<' + 'e' * len(data), *data)
                file.write(raw)


def read_blocks(file, bsize, packrate, blocks_num):
    blocks = []
    block_length = packrate * packrate
    pattern = '<' + 'e' * block_length
    for i in range(0, blocks_num):
        data = struct.unpack(pattern, file.read(struct.calcsize(pattern)))
        block = np.zeros((bsize, bsize))
        block[0:packrate, 0:packrate] = np.array(data).reshape(packrate, packrate)
        blocks.append(block)
    return blocks


def merge_blocks(blocks, w, h):
    _, block_width = blocks[0].shape
    blocks_per_row = (w - 1) // block_width + 1
    blocks_per_column = len(blocks) // blocks_per_row
    rows = [
        np.hstack(blocks[i * blocks_per_row: (i + 1) * blocks_per_row])
        for i in range(0, blocks_per_column)
    ]
    return np.vstack(rows)[0:h, 0:w]


def restore_rgb_image(path):
    if not path.endswith('.pak'):
        return
    with open(path, 'rb') as file:

        data = file.read(struct.calcsize('<IIHHHHHHHHH'))
        (
            w, h,
            y_bsize, y_packrate, len_y,
            cb_bsize, cb_packrate, len_cb,
            cr_bsize, cr_packrate, len_cr,
        ) = struct.unpack('<IIHHHHHHHHH', data)

        y = read_blocks(file, y_bsize, y_packrate, len_y)
        cb = read_blocks(file, cb_bsize, cb_packrate, len_cb)
        cr = read_blocks(file, cr_bsize, cr_packrate, len_cr)

    H_y = HadamardMatrix(order=y_bsize)
    y = list(map(H_y.inverse, y))
    y = merge_blocks(y, w, h)

    H_cb = HadamardMatrix(order=cb_bsize)
    cb = list(map(H_cb.inverse, cb))
    cb = merge_blocks(cb, w, h)

    H_cr = HadamardMatrix(order=cr_bsize)
    cr = list(map(H_cr.inverse, cr))
    cr = merge_blocks(cr, w, h)

    data = np.dstack((y, cb, cr))
    rgb_data = color.ycbcr2rgb(data)
    return rgb_data
