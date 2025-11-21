from rime.allele import Allele
from rime.circular import CircularBand, chainable_method
import numpy as np


class PYRAMID:
    def __init__(self, max_layers: int = 9):
        self.max_layers: int = max_layers
        self.bands: list[CircularBand] = []
        self.total = 0

    @chainable_method
    def build(self, gen_iter):
        self.bands, self.total = CircularBand.build_bands(gen_iter, max_batches=self.max_layers, start_batch=8)
        assert self.max_layers == len(self.bands)
        grid_size = 2 * self.max_layers + 1
        assert self.total + 1 == grid_size * grid_size
        return self

    def to_matrix(self, fill_center_with=('O', 'O')):
        return CircularBand.to_square_projection(self.bands, start_batch=8, center_value=fill_center_with)

    def to_3d_matrix(self, fill_center_with=('O', 'O')):
        m = self.to_matrix(fill_center_with=fill_center_with)
        blocks = CircularBand.split_matrix_rotational(m)
        return np.stack(blocks, axis=0)


if __name__ == "__main__":
    genotypes_iter = Allele.genotype_iter_by_freq(1000, 360)
    pyramid: PYRAMID = PYRAMID()
    # m = pyramid.build(genotypes_iter).to_matrix()
    m = pyramid.build(range(1, 361)).to_matrix(fill_center_with=0)
    print(m)
    ml = []
    for i, b in enumerate(m):
        ml.extend(b)
        print(i, b)

    band = CircularBand(ml)
    print(len(band))  # 361
    m2 = band.to_matrix(block_size=19, transpose=True)
    print(m2)
    assert m == m2
    m3 = band.transpose(block_size=19).to_matrix(block_size=19)
    print(m3)
    assert m == m3
    print(f'split_matrix:{len(m)}')

    x, c = band.split_matrix_blocks(m, rotate=True)
    for i, (block, coord) in enumerate(zip(x, c), 1):
        print(
            f"\nR{i}, shape: {len(block)}x{len(block[0])},{block[0][0]}->{block[-1][-1]}, rows:{coord[0]}, cols:{coord[1]}")
        print(block)

    y = band.split_matrix_rotational(m)
    print('rotational')
    z = [b.tolist() for b in y]
    for i, b in enumerate(z):
        print(f"\nR{i}, shape: {len(b)}x{len(b[0])},{b[0][0]}->{b[-1][-1]}")
        print(b)
    assert x == z

    m4 = band.merge_rotated_blocks(blocks=y, center_value=0)
    m5 = m4.tolist()
    assert m == m5
    print(m5)

    d1 = band.blocks_to_diagonal(x)
    d2 = band.blocks_to_diagonal(y)
    print('diagonal', d2.tolist())
    assert d1.tolist() == d2.tolist()
    ax = band.blocks_to_axle(y)
    print(ax.tolist())
    _3d = pyramid.to_3d_matrix(fill_center_with=0)
    print(_3d)
    m6 = band.merge_rotated_blocks(blocks=_3d, center_value=0).tolist()
    assert m == m6

    # for b in band:
    #     print(b)
