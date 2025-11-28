from rime.allele import Allele
from rime.circular import CircularBand, chainable_method
import numpy as np


class PYRAMID:
    def __init__(self, max_layers: int = 9):
        self.max_layers: int = max_layers
        self.bands: list[CircularBand] = []
        self.total = 0
        # activation=None activation_rule

    @chainable_method
    def build(self, gen_iter):
        self.bands, self.total = CircularBand.build_bands(gen_iter, max_batches=self.max_layers, start_batch=8)
        assert self.max_layers == len(self.bands)
        grid_size = 2 * self.max_layers + 1
        assert self.total + 1 == grid_size * grid_size
        return self

    def to_matrix(self, fill_center_with=('O', 'O')):
        # 完整 19×19 展开
        return CircularBand.to_square_projection(self.bands, start_batch=8, center_value=fill_center_with)

    def to_3d_matrix(self, fill_center_with=('O', 'O')):
        # 旋转对称映射,可逆变换,四个 face Face→RotateFace
        m = self.to_matrix(fill_center_with=fill_center_with)
        blocks = CircularBand.split_matrix_rotational(m)
        return np.stack(blocks, axis=0)


class PyramidNN:
    """
    Transformer + GNN 混合结构,递增环 + 旋转内对称结构.
    从 face[i] 预测 face[i+1]（旋转前）或从 band[k] 预测 band[k+1]=> 下一层由当前层“生长出来”
    """

    def __init__(self, dim=24, hidden=48, activation=None):
        self.dim = dim
        # MLP 1: per-element local transform
        self.W1 = np.random.randn(dim, hidden) * 0.1  # out_dim, in_dim
        self.b1 = np.zeros(hidden)

        # MLP 2: aggregation transform
        self.W2 = np.random.randn(hidden * 3, dim) * 0.1
        self.b2 = np.zeros(dim)

        self.W_inter = np.random.randn(dim, dim) * 0.1  # 层间传播矩阵

        self.activation = activation

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    def forward_layer(self, band: np.ndarray) -> np.ndarray:
        """
        层内环状传播 band: (L, dim)
        """
        L, dim = band.shape

        # 1) per-element transform
        h = self.relu(band @ self.W1 + self.b1)  # (L, hidden)

        # 2) neighborhood aggregation (circular)
        agg = []
        for i in range(L):
            left = h[(i - 1) % L]
            mid = h[i]
            right = h[(i + 1) % L]
            merged = np.concatenate([left, mid, right])  # (3*hidden,)
            out = self.relu(merged @ self.W2 + self.b2)  # (dim,)
            agg.append(out)

        return np.array(agg)  # (L, dim) MLP → 层内特征流动

    def cross_attention(self, prev_h, cur_h):
        """
        层间传播
        prev_h shape: (L1, d)
        cur_h  shape: (L2, d)
        """
        scores = cur_h @ prev_h.T / np.sqrt(self.dim)
        weights = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)  # softmax(scores, axis=1)
        return weights @ prev_h  # (L2, d)

    def forward_pyramid(self, pyramid: PYRAMID, mapping_embed: dict) -> list[np.ndarray]:
        prev_h = None
        outputs = []
        for band in pyramid.bands:
            h = np.array(band.encode(mapping_embed))  # (L, dim)
            h = self.forward_layer(h)
            if prev_h is not None:
                inter = self.cross_attention(prev_h, h)
                h = h + inter @ self.W_inter  # 融合上一层
            prev_h = h
            outputs.append(h)  # out

        return outputs


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
