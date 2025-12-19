from rime.allele import Allele
from rime.circular import CircularBand, MatrixFace, chainable_method
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

    @chainable_method
    def from_matrix(self, matrix: list[list]):
        self.bands, self.total = CircularBand.projection_to_bands(matrix)
        self.max_layers = len(self.bands)
        grid_size = 2 * self.max_layers + 1
        assert self.total + 1 == grid_size * grid_size
        return self

    def to_matrix(self, fill_center_with=('O', 'O')) -> list[list]:
        # 完整 19×19 展开
        return CircularBand.to_square_projection(self.bands, start_batch=8, center_value=fill_center_with)

    def to_blocks(self, fill_center_with=('O', 'O')) -> list[list[list]]:
        # 旋转对称映射,可逆变换,四个 face: bands <--> 4个face矩阵 Face → RotateFace
        m = self.to_matrix(fill_center_with=fill_center_with)
        blocks, _ = MatrixFace.split_to_blocks(m, rotate=True)
        return blocks

    def to_3d_matrix(self, fill_center_with=('O', 'O')) -> np.ndarray:
        # 旋转对称映射,可逆变换,四个 face: bands <--> 4个face矩阵 Face → RotateFace
        m = self.to_matrix(fill_center_with=fill_center_with)
        blocks = MatrixFace.split_matrix_rotational(m, dtype=object)
        return np.stack(blocks, axis=0)

    def encode(self, mapping_embed: dict, default=None, dtype=int) -> list[np.ndarray]:
        '''bands->encoded_bands'''
        return [np.asarray(band.encode(mapping_embed, default, False), dtype=dtype) for band in self.bands]


class Pyraminx:
    def __init__(self, max_layers: int = 3):
        self.max_layers: int = max_layers


class PyramidNN:
    """
    Transformer + GNN 混合结构,递增环 + 旋转内对称结构.
    从 face[i] 预测 face[i+1]（旋转前）或从 band[k] 预测 band[k+1]=> 下一层由当前层“生长出来”
    (X @ W + b)
    X shape = (batch, input_dim)
    W shape = (input_dim, output_dim)
    b shape = (output_dim,)
    """

    def __init__(self, dim=2, hidden=48, activation=None):
        self.dim = dim  # d_model
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
        层内环状传播 band: (L, dim) (L, D) @ (D, H) → (L, H)
        """
        L, dim = band.shape

        # 1) per-element transform hidden
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
        scores = cur_h @ prev_h.T / np.sqrt(self.dim)  # Q @ K.T / np.sqrt(dim)
        weights = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)  # softmax(scores, axis=1)
        return weights @ prev_h  # (L2, d)

    def forward_pyramid(self, encoded_bands: list[np.ndarray]) -> list[np.ndarray]:
        prev_h = None
        outputs = []
        for band in encoded_bands:  # (L, dim)
            h = self.forward_layer(band)
            if prev_h is not None:
                inter = self.cross_attention(prev_h, h)
                h = h + inter @ self.W_inter  # 融合上一层
            prev_h = h
            outputs.append(h)  # out

        return outputs

    def backward(self, loss):
        pass


class NextBandHead:
    def __init__(self, dim):
        self.Wq = np.random.randn(dim, dim) * 0.1
        self.Wk = np.random.randn(dim, dim) * 0.1
        self.Wv = np.random.randn(dim, dim) * 0.1
        self.Wo = np.random.randn(dim, dim) * 0.1

    def predict(self, prev_h, L_next: int):
        """
        根据上一层表示 prev_h 预测下一层 (L_next, dim), CrossAttentionHead
        """
        L_prev, dim = prev_h.shape

        # Query: 一组新位置随机初始化
        # 这些 query 就是下一层的“空壳”
        Q = np.random.randn(L_next, dim) @ self.Wq
        K = prev_h @ self.Wk
        V = prev_h @ self.Wv

        scores = Q @ K.T / np.sqrt(dim)
        weights = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)

        out = weights @ V
        return out @ self.Wo


def train_forward(nn: 'PyramidNN', head: NextBandHead, pyramid: PYRAMID, mapping_embed: dict):
    losses = []

    prev_h = None
    encoded_bands = pyramid.encode(mapping_embed)

    # 1) 先跑 PyramidNN 得到所有层的特征
    h_list = nn.forward_pyramid(encoded_bands)

    # 2) 遍历每一层，预测下一层
    for k in range(len(h_list) - 1):
        h_k = h_list[k]

        true_next = h_list[k + 1]
        L_next = true_next.shape[0]
        pred_next = head.predict(h_k, L_next)
        L_gen = ((pred_next - true_next) ** 2).mean()  # mse
        losses.append(L_gen)

    return sum(losses)


if __name__ == "__main__":
    genotypes_iter = Allele.genotype_iter_by_freq(1000, 360)
    pyramid: PYRAMID = PYRAMID()
    center_value = ('O', 'O')
    m = pyramid.build(genotypes_iter).to_matrix(fill_center_with=center_value)
    mapping = Allele.genotype_to_vector_mapping()

    encoded_bands = pyramid.encode(mapping)
    for x in encoded_bands:
        print(x.shape)

    # m = pyramid.build(range(1, 361)).to_matrix(fill_center_with=0)
    # center_value = 0
    print(m)
    ml = []
    for i, b in enumerate(m):
        ml.extend(b)
        print(i, b)

    band = CircularBand(ml)
    print(len(band))  # 361
    m2 = band.to_matrix(block_size=19, transpose=True)
    print(m2)
    print(MatrixFace.flatten(m2))
    assert m == m2
    m3 = band.transpose(block_size=19).to_matrix(block_size=19)
    print(m3)
    assert m == m3
    print(f'split_matrix:{len(m)}')

    x, c = MatrixFace.split_to_blocks(m, rotate=True)
    for i, (block, coord) in enumerate(zip(x, c), 1):
        print(
            f"\nR{i}, shape: {len(block)}x{len(block[0])},{block[0][0]}->{block[-1][-1]}, rows:{coord[0]}, cols:{coord[1]}")
        print(block)

    y = MatrixFace.split_matrix_rotational(m)
    z = [b.tolist() for b in y]
    for i, b in enumerate(z):
        print(f"\nR{i}, shape: {len(b)}x{len(b[0])},{b[0][0]}->{b[-1][-1]}")
        print(b)
    assert x == z, f'{x},\n{z}'
    for a in y:
        print(a.shape, a)
    # y2 = np.stack(y, axis=0)
    m4 = MatrixFace.merge_rotated_blocks(blocks=y, center_value=center_value)
    m5 = m4.tolist()
    assert m == m5
    print(m5)

    d2 = MatrixFace.blocks_to_diagonal(y)
    print('diagonal', d2.tolist())
    ax = MatrixFace.blocks_to_axle(y)
    print('axle', ax.tolist())
    _3d = pyramid.to_3d_matrix(fill_center_with=center_value)
    print(_3d)
    m6 = MatrixFace.merge_rotated_blocks(blocks=_3d, center_value=center_value).tolist()
    assert m == m6

    pyramid2 = PYRAMID()
    m7 = pyramid2.from_matrix(m).to_matrix(fill_center_with=center_value)
    assert m == m7
    print(pyramid2.max_layers, pyramid2.total)
    # for b in band:
    #     print(b)

    PNN = PyramidNN()
    outputs = PNN.forward_pyramid(encoded_bands)
    for b in outputs:
        print(b.shape)
