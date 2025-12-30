from collections import deque
from rime.base import chainable_method
import itertools
import numpy as np
import math


# import numba
# @numba.jit(nopython=True, cache=True)

class MatrixFace:
    @staticmethod
    def to_round(data: list, n_slices: int = 4, pad_value=None, clockwise: bool = True) -> list[tuple]:
        """
        build_slice，数组均匀切成 n 份：(i * size, (i + 1) * size)，size = L // n_slices
        将 band分成n_slices段（默认4）。每层切分的段数（如 4=方形、6=蜂窝、8=八边）
        比如 n_slices=4 表示 top/right/bottom/left 四边；
             n_slices=6 表示六个方向的环形切分。

        Args:
            n_slices: 切分段数
            pad_value: 若数据长度不足，填充该值
            clockwise: 是否按顺时针方向切
        """
        n = len(data)
        per_slice = -(-n // n_slices)  # ceil(n / n_slices)
        total_len = per_slice * n_slices

        if n < total_len:  # 如果不够整除，就补齐到能整除
            data = data + [pad_value] * (total_len - n)

        # 重新分块 top,right,bottom,left
        chunks = [tuple(data[i * per_slice:(i + 1) * per_slice]) for i in range(n_slices)]

        if not clockwise:  # 逆时针 [left, bottom, right, top]
            chunks = [tuple(reversed(chunk)) for chunk in reversed(chunks)]  # chunks.reverse()

        return chunks

    @staticmethod
    def to_matrix(data: list, block_size: int = 4, transpose: bool = False, pad_value=None) -> list[list]:
        """
        将当前 data 视作按列填充的矩阵并返回（不修改 data）。
        把 data 按列填充到 4 行（block_size 行），即 column-major 填充，
        最后返回按行的矩阵（rows x cols）。
        transpose: 是否返回转置后的矩阵
        要求 len(data) % block_size == 0（否则最后一列会被补 None）。
        """
        n = len(data)
        rows = block_size
        cols = -(-n // block_size)

        # 填充扁平数据到 column-major 矩阵
        matrix = [[pad_value] * cols for _ in range(rows)]
        for idx, val in enumerate(data):
            r = idx % rows
            c = idx // rows
            matrix[r][c] = val
        if transpose:
            matrix = [list(row) for row in zip(*matrix)]
        return matrix

    @staticmethod
    def split_to_blocks(matrix: list[list], rotate: bool = True) -> tuple[list[list[list]], list[tuple]]:
        """
        将 n x n 矩阵按中心分成 4 块（奇数去掉中心点）。
        每块元素数 = c*(c+1)，形状 = (c+1) x c, n_layers=c
        每块元素数相同，形状为 (c+1) x c（c = n//2）。rotate 控制是否把
        两块需要旋转的子块转置成相同形状。
        返回 (blocks, coords)：
          blocks: [R1, R2, R3, R4]  2D lists
          coords: [(rows_list, cols_list), ...] 对应每块在原矩阵中的索引范围
        块顺序: R1=左上, R2=右上, R3=右下, R4=左下
        """
        n = len(matrix)
        assert all(len(row) == n for row in matrix), "必须是方阵"
        c = n // 2

        if n % 2 == 0:  # 偶数， total:4*c*c+=n*n
            # 切片范围
            r0 = range(0, c)
            r1 = range(c, n)
            c0 = range(0, c)
            c1 = range(c, n)

            S1 = [row[c0.start:c0.stop] for row in matrix[r0.start:r0.stop]]  # 左上
            S2 = [row[c1.start:c1.stop] for row in matrix[r0.start:r0.stop]]  # 右上
            S3 = [row[c1.start:c1.stop] for row in matrix[r1.start:r1.stop]]  # 右下
            S4 = [row[c0.start:c0.stop] for row in matrix[r1.start:r1.stop]]  # 左下

            coords = [
                (list(r0), list(c0)),
                (list(r0), list(c1)),
                (list(r1), list(c1)),
                (list(r1), list(c0)),
            ]
            if not rotate:
                return [S1, S2, S3, S4], coords

        else:  # 奇数（覆盖除中心外所有点） total:4*c*(c+1)=n*n-1
            # 定义四个区块的行列 range，用 range 合理构建
            r1, c1 = range(0, c), range(0, c + 1)  # S1
            r2, c2 = range(0, c + 1), range(c + 1, n)  # S2
            r3, c3 = range(c + 1, n), range(c, n)  # S3
            r4, c4 = range(c, n), range(0, c)  # S4

            # 切片上界是排他的
            S1 = [row[c1.start:c1.stop] for row in matrix[r1.start:r1.stop]]  # c x (c+1)
            S2 = [row[c2.start:c2.stop] for row in matrix[r2.start:r2.stop]]  # (c+1) x c
            S3 = [row[c3.start:c3.stop] for row in matrix[r3.start:r3.stop]]  # c x (c+1)
            S4 = [row[c4.start:c4.stop] for row in matrix[r4.start:r4.stop]]  # (c+1) x c

            coords = [
                (list(r1), list(c1)),  # S1 rows,cols
                (list(r2), list(c2)),  # S2 rows,cols (原始)
                (list(r3), list(c3)),  # S3
                (list(r4), list(c4))  # S4
            ]
            if not rotate:
                return [S1, S2, S3, S4], coords

        # 选择 S1 作为基准方向，旋转其他三个以与 S1 朝向一致,把 S1..S4 变形/旋转成相同形状并使方向一致
        R1 = S1  # 下面旋转方向选择保证“中心对称旋转关系”
        R2 = [list(row) for row in zip(*S2)]  # transpose
        R2 = R2[::-1]  # reverse row order
        R3 = [row[::-1] for row in S3[::-1]]  # 180°
        R4 = [list(row) for row in zip(*S4)]
        R4 = [row[::-1] for row in R4]

        blocks = [R1, R2, R3, R4]
        return blocks, coords

    @staticmethod
    def rotate(matrix: list[list], direction: int = 1) -> list[list]:
        """
        | 操作               | 代码思想                       |
        | ---------------- | ----------------------------- |
        | 左右翻转 (mirror LR) | `face[r][n-1-c]`           |[row[::-1] for row in A]
        | 上下翻转 (mirror UD) | `face[n-1-r][c]`           |A[::-1]
        | 转置 (mirror diag) | `face[c][r]`                |zip(*A)
        | 顺时针 90°          | `face[n-1-r][c]` + `转置`   |list(zip(*A))[::-1]
        | 逆时针 90°          | `face[r][n-1-c]` + `转置`   |[list(row)[::-1] for row in zip(*A)]
        """
        d = direction % 4
        if d == 0:  # copy
            return [row[:] for row in matrix]
        if d == 1:  # clockwise[[matrix[N - 1 - j][i] for j in range(N)] for i in range(N)] 顺时针：先上下反转，再转置
            return [list(row) for row in zip(*matrix[::-1])]
        if d == 2:  # 180° == reverse rows + reverse each row
            return [row[::-1] for row in matrix[::-1]]
        if d == 3:  # [[matrix[j][N - 1 - i] for j in range(N)] for i in range(N)] # 逆时针，先转置，再上下反转
            return [list(row) for row in zip(*matrix)][::-1]
        return matrix

    @staticmethod
    def rotate_inplace(mat: list[list], direction: int = 1) -> None:
        """
        swap
        矩阵的 90° 旋转可以看成对矩阵的若干“环”（ring）做环内元素循环（四个位置互换）
        对每个层 layer（从外向内），对该层的每个位置做 4-way 交换。
        In-place rotate square matrix mat by direction*90 degrees clockwise.
        direction: integer (positive/negative allowed). direction % 4 gives action:
          0 -> no-op
          1 -> 90 deg CW
          2 -> 180 deg
          3 -> 270 deg CW (or 90 CCW)
        The function mutates mat and returns None.
        """
        d = direction % 4
        if d == 0:
            return

        n = len(mat)
        if n == 0:
            return
        # helper: rotate outer layers in-place
        # layer 0 .. n//2 - 1
        layers = n // 2
        if d == 2:
            # 180°: swap pairs -> for each layer and position swap two pairs
            for layer in range(layers):
                first = layer
                last = n - 1 - layer
                for i in range(first, last):
                    offset = i - first
                    # positions: (first, i), (i, last), (last, last-offset), (last-offset, first)
                    r1, c1 = first, i
                    r2, c2 = i, last
                    r3, c3 = last, last - offset
                    r4, c4 = last - offset, first
                    # swap (r1,c1) <-> (r3,c3); (r2,c2) <-> (r4,c4)
                    mat[r1][c1], mat[r3][c3] = mat[r3][c3], mat[r1][c1]
                    mat[r2][c2], mat[r4][c4] = mat[r4][c4], mat[r2][c2]
            return

        # for d == 1 or d == 3: do 4-way swaps; if d==3 we can do 3 times d==1,
        # but better to perform appropriate direction directly (we implement CW 90 and CCW 90)
        clockwise = (d == 1)
        for layer in range(layers):
            first = layer
            last = n - 1 - layer
            for i in range(first, last):
                offset = i - first
                top = (first, i)
                right = (i, last)
                bottom = (last, last - offset)
                left = (last - offset, first)

                if clockwise:
                    # top <- left, left <- bottom, bottom <- right, right <- top
                    # carry values using temp
                    tmp = mat[left[0]][left[1]]
                    mat[left[0]][left[1]] = mat[bottom[0]][bottom[1]]
                    mat[bottom[0]][bottom[1]] = mat[right[0]][right[1]]
                    mat[right[0]][right[1]] = mat[top[0]][top[1]]
                    mat[top[0]][top[1]] = tmp
                else:
                    # counter-clockwise: top <- right, right <- bottom, bottom <- left, left <- top
                    tmp = mat[top[0]][top[1]]
                    mat[top[0]][top[1]] = mat[right[0]][right[1]]
                    mat[right[0]][right[1]] = mat[bottom[0]][bottom[1]]
                    mat[bottom[0]][bottom[1]] = mat[left[0]][left[1]]
                    mat[left[0]][left[1]] = tmp

    @staticmethod
    def flatten(matrix: list[list] | np.ndarray, dim=2) -> list:
        """拉平,丢位置信息,从二维转成一维序列，保持特征维度"""
        A = np.array(matrix, dtype=object)
        assert A.ndim == 3 and A.shape[2] == dim, f"Expected shape (N,N,{dim}), got {A.shape}"
        return A.reshape(-1, dim).tolist()

    @staticmethod
    def to_ndarray(matrix: list[list], dtype=object) -> np.ndarray:
        n, m = len(matrix), len(matrix[0])
        A = np.empty((n, m), dtype=dtype)
        for i in range(n):
            for j in range(m):
                A[i, j] = matrix[i][j]
        return A  # np.array([item for row in matrix for item in row], dtype=object).reshape(len(matrix), len(matrix[0]))

    @staticmethod
    def split_matrix_rotational(matrix: list[list] | np.ndarray, dtype=object):
        """
        按中心分成四块，并统一方向（中心旋转对称）。
        奇数：去掉中心点，每块 (c+1)×c
        偶数：每块 c×c
        块顺序：(R1,R2,R3,R4) = (左上, 右上, 右下, 左下)
        """
        A = np.asarray(matrix, dtype=dtype)
        if A.ndim > 2:  # 如果元组被展开
            print(f'[split_matrix_rotational],shape:{A.shape}')
            A = MatrixFace.to_ndarray(matrix, dtype=dtype)

        assert A.ndim == 2, f'必须是二维矩阵，shape:{A.shape}'
        assert A.shape[0] == A.shape[1], "必须是方阵"
        n = A.shape[0]
        c = n // 2
        if n % 2 == 0:
            # 偶数：直接 4 象限
            S1 = A[:c, :c]  # 左上
            S2 = A[:c, c:]  # 右上
            S3 = A[c:, c:]  # 右下
            S4 = A[c:, :c]  # 左下
        else:
            # 奇数：去掉中心点
            S1 = A[0:c, 0:c + 1]  # 左上: rows[0:c], cols[0:c+1]
            S2 = A[0:c + 1, c + 1:n]  # 右上: rows[0:c+1], cols[c+1:n]
            S3 = A[c + 1:n, c:n]  # 右下: rows[c+1:n], cols[c:n]
            S4 = A[c:n, 0:c]  # 左下: rows[c:n], cols[0:c]

        R1 = S1.copy()
        R2 = np.flipud(S2.T)  # R2: transpose + flipud, np.rot90(S2, k=-1)
        R3 = np.fliplr(np.flipud(S3))  # R3: 180 degrees = flipud + fliplr,np.rot90(S3, k=2)
        R4 = np.fliplr(S4.T)  # R4: transpose + fliplr, np.rot90(S4, k=1)

        return R1, R2, R3, R4  # [R1.tolist(), R2.tolist(), R3.tolist(), R4.tolist()]

    @staticmethod
    def merge_rotated_blocks(blocks: tuple | list | np.ndarray, center_value=None) -> np.ndarray:
        """
        R1~R4: 四块 numpy array，已按统一朝向旋转
        center_value: 奇数矩阵中心点填充值
        返回原矩阵 numpy array
        """
        if isinstance(blocks, np.ndarray) and blocks.ndim == 3:
            if blocks.shape[0] != 4:
                raise ValueError("3D array 输入必须是 (4, h, w)")
            R1, R2, R3, R4 = blocks[0], blocks[1], blocks[2], blocks[3]
        else:
            R1, R2, R3, R4 = blocks

        c = R1.shape[0]
        if R1.shape[1] == c + 1:  # 奇数矩阵
            n = 2 * c + 1
            mat = np.empty((n, n), dtype=R1.dtype)

            mat[0:c, 0:c + 1] = R1  # 左上: R1 不动
            mat[0:c + 1, c + 1:n] = np.flipud(R2).T  # 右上: R2 逆旋转 -> flipud + transpose
            mat[c + 1:n, c:n] = np.flipud(np.fliplr(R3))  # 右下: R3 逆旋转 -> flipud + fliplr (180°)
            mat[c:n, 0:c] = np.fliplr(R4).T  # 左下: R4 逆旋转 -> fliplr + transpose

            mat[c, c] = center_value  # 中心点

        else:  # 偶数矩阵，c = R1.rows = R1.cols
            n = 2 * c
            mat = np.empty((n, n), dtype=R1.dtype)

            # 同样旋转逆操作
            mat[0:c, 0:c] = R1
            mat[0:c, c:n] = np.flipud(R2).T
            mat[c:n, c:n] = np.flipud(np.fliplr(R3))
            mat[c:n, 0:c] = np.fliplr(R4).T

        return mat

    @staticmethod
    def blocks_to_diagonal(blocks: tuple | list):
        # 提取对角元素
        # [block[i][i] for i in range(min(len(block), len(block[0])))]
        return np.vstack([np.diag(block) for block in blocks])

    @staticmethod
    def blocks_to_axle(blocks: tuple | list):
        # 提取轴列元素(最后一列),左上(右侧)，右上（下），右下（左），左下（上）
        return np.vstack([b[:, -1] for b in blocks])  # top:第一行b[0, 1:]

    @staticmethod
    def build_rotate_map(n: int, clockwise: bool = True):
        """
        返回 ROTATE_MAP[n]，支持任意 n。
        rotate_map[k][i] = 旋转 k 后，第 i 个 slice 去哪里
        """
        rotate_map = {}

        for k in range(n):  # 旋转 k 次
            mapping = {}
            for i in range(n):  # 第 i 个 slice
                if clockwise:
                    mapping[i] = (i - k) % n
                else:
                    mapping[i] = (i + k) % n
            rotate_map[k] = mapping

        return rotate_map


class CircularBand:
    def __init__(self, initial_data=None, capacity=None):
        """
        初始化环形数据结构

        :param initial_data: 初始数据（可迭代对象）
        :param capacity: 最大容量限制（None表示无限制）
        """
        self.data = list(initial_data) if initial_data else []  # container
        self.cursor: int = 0  # 当前指针位置
        self.capacity = capacity

        # 如果设置了最大容量，裁剪超出部分
        if capacity is not None and len(self.data) > capacity:
            self.data = self.data[-capacity:]

    @chainable_method
    def fill(self, new_data, reset_cursor: bool = True, truncate: bool = True):
        """
           用 new_data 替换 CircularBand 的所有内容。
           reset_cursor: 是否将 cursor 置为 0（默认 True）。
           truncate: 如果 new_data 长度超过 capacity，是否截断保留尾部（最近的部分）。仅在 capacity 不为 None 时生效。
        """
        new_list = list(new_data) or []
        # 处理 capacity
        if self.capacity is not None and len(new_list) > self.capacity:
            if truncate:
                new_list = new_list[-self.capacity:]
            else:
                raise ValueError(f"new_data length ({len(new_list)}) exceeds capacity ({self.capacity})")

        self.data = new_list
        self.cursor = 0 if reset_cursor else min(self.cursor, len(self.data) - 1 if self.data else 0)

    @chainable_method
    def append(self, item):
        """在指针后插入元素"""
        if self.capacity is not None and len(self.data) >= self.capacity:
            if not self.data:
                return
            # 容量已满时覆盖,覆盖策略：替换下一个位置元素
            overwrite_pos = (self.cursor + 1) % len(self.data)
            self.data[overwrite_pos] = item
            self.cursor = overwrite_pos
        else:
            insert_pos = (self.cursor + 1) % (len(self.data) + 1)
            self.data.insert(insert_pos, item)
            self.cursor = insert_pos

    @chainable_method
    def remove(self, pos: int = None):
        """删除指针位置元素（自动连接相邻元素）"""
        if not self.data:
            return

        n = len(self.data)
        if pos is None:
            pos = self.cursor
        else:
            pos %= n  # 支持负索引

        del self.data[pos]

        # 处理删除后的光标位置
        if not self.data:
            self.cursor = 0
        elif pos >= len(self.data):
            # 删的是最后一个 → 回到 0（环形首）
            self.cursor = 0
        else:
            # 否则光标仍指向原删除位置（删除后的下一个元素）
            self.cursor = pos

    @chainable_method
    def expand(self, items):
        """扩展多个元素"""
        if not items:
            return
        n = len(self.data)
        m = len(items)

        # 计算需要保留的新元素数量
        if self.capacity is not None:
            available = max(0, self.capacity - n)
            items = items[-available:]  # 只保留能插入的部分
            m = len(items)

        insert_pos = (self.cursor + 1) % (n + 1)  # self.cursor + 1
        # 插入元素,在指针后插入
        self.data[insert_pos:insert_pos] = items
        # 容量限制处理,移除多余元素（从左侧开始移除）
        if self.capacity is not None and len(self.data) > self.capacity:
            excess = len(self.data) - self.capacity
            del self.data[:excess]
            insert_pos -= excess
        # 更新指针到最后一个新元素,self.cursor += num_items
        self.cursor = min(max(0, insert_pos + m - 1), len(self.data) - 1)

    @chainable_method
    def contract(self, k: int):
        """从指针处收缩 k 个元素"""
        if k <= 0 or not self.data:
            return

        start = self.cursor
        end = min(self.cursor + k, len(self.data))
        del self.data[start:end]

        if not self.data:  # 指针调整
            self.cursor = 0
        else:
            self.cursor = min(self.cursor, len(self.data) - 1)

    @chainable_method
    def rotate(self, steps: int = 1):
        """旋转结构 direction（正数右移,顺时针旋转，负数左移,逆时针旋转）"""
        if not self.data:
            return
        self.cursor = (self.cursor + steps) % len(self.data)

    @chainable_method
    def transpose(self, block_size: int = 4):
        """
        按块大小重组数据（类似矩阵转置）,
        order=col 当作(rows=block_size, cols=n/block_size) 的矩阵（按列填充）
        """
        n = len(self.data)
        if n == 0:
            return
        assert n % block_size == 0, f"数据长度 {n} 必须能被块大小 {block_size} 整除"

        cols = block_size
        rows = n // block_size
        # 将数据分成块(每列是 block_size 长）
        blocks = [self.data[i * cols:(i + 1) * cols] for i in range(rows)]
        self.data = [item for col in zip(*blocks) for item in col]  # 按行展平转置后的矩阵
        # 调整指针位置
        r, c = divmod(self.cursor, cols)
        self.cursor = c * rows + r

    @chainable_method
    def mirror(self):
        """将数据结构首尾镜像反转"""
        if not self.data:
            return
        n = len(self.data)
        self.data.reverse()
        # 对称更新光标位置
        self.cursor = n - 1 - self.cursor  # self.data.index(current_item)

    @chainable_method
    def swap(self, offset: int = None):
        """交换当前元素与下一个元素，并将指针移到下一个元素"""
        n = len(self.data)
        if n < 2:
            return
        offset = offset or 1
        next_pos = (self.cursor + offset) % n
        if next_pos == self.cursor:
            return
        self.data[self.cursor], self.data[next_pos] = self.data[next_pos], self.data[self.cursor]
        self.cursor = next_pos

    def current(self):
        """获取当前元素"""
        return self.data[self.cursor] if self.data else None

    def __len__(self):
        """返回数据长度"""
        return len(self.data)

    def __iter__(self):
        """从当前指针开始循环遍历,返回有限一轮"""
        n = len(self.data)
        for i in range(n):
            yield self.data[(self.cursor + i) % n]

    def __next__(self):
        """永远循环,cursor 向前移动"""
        if not self.data:
            raise StopIteration
        value = self.data[self.cursor]
        self.cursor = (self.cursor + 1) % len(self.data)
        return value

    def __getitem__(self, index):
        """
        获取元素（支持环形索引和切片）

        索引规则：
        - 正数索引：从当前指针开始的环形索引
        - 负数索引：从末尾开始的环形索引
        """
        if isinstance(index, slice):
            # 处理切片操作
            start, stop, step = index.indices(len(self))
            return [self[i] for i in range(start, stop, step)]

        if not self.data:
            raise IndexError("CircularBand is empty")
        return self.data[(self.cursor + index) % len(self.data)]

    def __setitem__(self, index, value):
        """设置元素值（支持环形索引）"""
        n = len(self.data)
        if not n:
            raise IndexError("CircularBand is empty")

        pos = (self.cursor + index) % n
        self.data[pos] = value

    def __str__(self):
        """可视化环形结构"""
        if not self.data:
            return "Empty"

        elements = [f"[{x}]" if i == self.cursor else str(x)
                    for i, x in enumerate(self.data)]

        return " → ".join(elements) + f" → [{self.data[0]}]..." + (
            f" (Max: {self.capacity})" if self.capacity is not None else "")

    def __repr__(self):
        return f"CircularBand(data={self.data}, cursor={self.cursor}, capacity={self.capacity})"

    def to_list(self, from_current: bool = True) -> list:
        """
        将环形数据转换为列表

        :param from_current: 是否从当前元素开始,list(self) 用__iter__ 来遍历
        :return: 数据列表
        """
        if not self.data:
            return []
        return list(self) if from_current else list(self.data)

    def encode(self, mapping: dict, default=None, from_current=False) -> list:
        return [mapping.get(x, default) for x in self.to_list(from_current)]

    def forward(self, message, exit_at: int = None, **context):
        """
        环形消息流转，本地 cursor
        """
        if exit_at is None:
            exit_at = len(self) - 1
        cursor = self.cursor
        msg = message
        for _ in range(exit_at + 1):
            f = self.data[cursor]  # next(self) 获取下一个元素
            cursor = (cursor + 1) % len(self.data)
            msg = f(msg, **context)  # 传递消息
            yield msg

    def to_matrix(self, block_size: int = 4, transpose: bool = False, from_current: bool = False,
                  pad_value=None) -> list[list]:
        """
        将当前 data 视作按列填充的矩阵并返回（不修改 data）。
        把 data 按列填充到 4 行（block_size 行），即 column-major 填充，
        最后返回按行的矩阵（rows x cols）。
        transpose: 是否返回转置后的矩阵
        要求 len(data) % block_size == 0（否则最后一列会被补 None）。
        to_matrix(block_size=19, transpose=True)==transpose(block_size=19).to_matrix(block_size=19)
        """
        n = len(self.data)
        if n == 0:
            return []
        data = self.to_list(from_current=from_current)
        return MatrixFace.to_matrix(data, block_size=block_size, transpose=transpose, pad_value=pad_value)

    def to_round(self, n_slices: int = 4, from_current: bool = False,
                 pad_value=None, clockwise: bool = True) -> list[tuple]:
        """
        build_slice，数组均匀切成 n 份：(i * size, (i + 1) * size)，size = L // n_slices
        将 band分成n_slices段（默认4）。每层切分的段数（如 4=方形、6=蜂窝、8=八边）
        比如 n_slices=4 表示 top/right/bottom/left 四边；
            n_slices=6 表示六个方向的环形切分。

        Args:
            n_slices: 切分段数
            from_current: 是否从cursor开始线性展开
            pad_value: 若数据长度不足，填充该值
            clockwise: 是否按顺时针方向切
        """
        if n_slices <= 0:
            raise ValueError("n_slices 必须为正整数")

        data = self.to_list(from_current=from_current)
        return MatrixFace.to_round(data, n_slices=n_slices, pad_value=pad_value, clockwise=clockwise)

    @staticmethod
    def to_square_projection(bands: list['CircularBand'], start_batch: int = 8,
                             center_value=None) -> list[list]:
        """
        使用 bands[i].to_round(per_side) 将 bands 投影到方阵。
        - base: 第一层的 batch_size 基数（如 8），第 i 层的 batch_size = start_batch*(i+1)
        """
        n_layers = len(bands)
        grid_size = 2 * n_layers + 1
        center = n_layers  # 中心坐标 (center, center)
        grid = [[None for _ in range(grid_size)] for _ in range(grid_size)]

        for i, band in enumerate(bands):
            layer = i + 1  # radius 第几层（半径）
            batch_size = start_batch * layer  # e.g. 8,16,...,8*n = 4*per_edge
            assert len(band) == batch_size, f"第 {i} 层数据长度应为 {batch_size}，实际为 {len(band)}"

            slices = band.to_round(n_slices=4, from_current=False, pad_value=center_value, clockwise=True)
            top, right, bottom, left = slices

            top_row = center - layer
            left_col = center - layer
            bottom_row = center + layer
            right_col = center + layer

            # top: (top_row, left_col .. right_col-1)
            for j, val in enumerate(top):
                grid[top_row][left_col + j] = val

            # right: (top_row .. bottom_row-1, right_col)
            for j, val in enumerate(right):
                grid[top_row + j][right_col] = val

            # bottom: (bottom_row, right_col .. left_col+1)  (注意顺序为从右到左以确保连贯)
            for j, val in enumerate(bottom):
                grid[bottom_row][right_col - j] = val

            # left: (bottom_row .. top_row+1, left_col) (从下往上)
            for j, val in enumerate(left):
                grid[bottom_row - j][left_col] = val

        # 处理中心点
        if center_value is not None:
            grid[center][center] = center_value

        return grid

    @classmethod
    def projection_to_bands(cls, matrix: list[list]):
        '''从投影恢复'''
        grid_size = len(matrix)
        layers = (grid_size - 1) // 2
        center = layers
        bands = []
        total = 0
        for i in range(layers):
            layer = i + 1
            top_row = center - layer
            bottom_row = center + layer
            left_col = center - layer
            right_col = center + layer

            # 提取四条边，顺序 top->right->bottom->left
            top = [matrix[top_row][left_col + j] for j in range(right_col - left_col + 1)]
            right = [matrix[top_row + j][right_col] for j in range(1, bottom_row - top_row)]
            bottom = [matrix[bottom_row][right_col - j] for j in range(right_col - left_col + 1)]
            left = [matrix[bottom_row - j][left_col] for j in range(1, bottom_row - top_row)]

            band_data = top + right + bottom + left
            bands.append(cls(band_data))
            total += len(band_data)

        return bands, total

    @classmethod
    def build_bands(cls, gen_iter, max_batches: int = 9, start_batch: int = 8):
        batch_sizes = [start_batch * (i + 1) for i in range(max_batches)]
        bands = [cls(initial_data=[], capacity=size) for size in batch_sizes]

        total = 0
        it = iter(gen_iter)
        for i, size in enumerate(batch_sizes):
            chunk = list(itertools.islice(it, size))
            if not chunk:
                break

            bands[i].fill(chunk)
            total += len(chunk)
            if len(chunk) < size:
                break

        return bands, total

    def save(self, filename):
        """保存数据到文件（包括指针位置）"""
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump({'data': self.data, 'cursor': self.cursor, 'capacity': self.capacity}, f)

    @classmethod
    def load(cls, filename):
        """从文件加载数据"""
        import pickle
        import os
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found")

        with open(filename, 'rb') as f:
            state = pickle.load(f)

        band = cls(state['data'], capacity=state['capacity'])  # type(self)(data)
        band.cursor = state['cursor'] % len(band) if band else 0
        return band


def build_batch_bands(gen_iter, max_batches: int = 9, start_batch: int = 8):
    """
    根据生成器按增量批次填充多层 CircularBand。
    每层容量： start_batch * (i+1) ， i 从 0 开始，共 max_batches 层。
    当缓冲区累计到某层批次大小时，将该批次弹出并写入对应层（替换该层内容）。
    返回： bands 列表（len == max_batches），以及一个 stats 字典记录每层写入次数。
    """
    batch_sizes = [start_batch * (i + 1) for i in range(max_batches)]
    thresholds = []
    cum = 0
    for sz in batch_sizes:
        cum += sz
        thresholds.append(cum)
    window = deque(maxlen=thresholds[-1])
    # bands：每一层一个 CircularBand
    bands = [CircularBand(initial_data=[], capacity=size) for size in batch_sizes]

    stats = {"filled_counts": [0] * max_batches}
    total_processed = 0
    next_threshold_idx = 0
    for item in gen_iter:
        window.append(item)
        total_processed += 1

        # 尝试按每一层的 batch_size 把数据弹出并写入层
        # 注意：从低到高层依次尝试，确保较小层优先消费
        # 如果达到或超过当前阈值，就触发对应层
        while next_threshold_idx < len(thresholds) and total_processed >= thresholds[next_threshold_idx]:
            k = next_threshold_idx  # 对应第 k 层（0-based）
            batch_size = batch_sizes[k]
            # 取最近 batch_size 个元素作为该层内容
            chunk = list(window)[-batch_size:] if len(window) >= batch_size else list(window)
            # 写入（替换）第 k 层
            bands[k].data = chunk[:]  # 直接替换底层数据
            bands[k].cursor = 0
            stats["filled_counts"][k] += 1
            next_threshold_idx += 1

    stats["total_processed"] = total_processed
    # 返回 bands 与统计
    return bands, stats


if __name__ == "__main__":
    # 环形结构初始化
    band = CircularBand(["A", "B", "C"])
    print(band)  # [A] → B → C → [A]...

    band.append("D")
    print(band)  # A → [B] → C → D → [A]...

    band.rotate(2)
    print(band.current())  # D

    print("动态缩放:")
    band.expand(["X", "Y"])
    print(band)  # A → B → C → [D] → X → Y → [A]...

    band.contract(2)
    print(band)  # A → B → C → [D] → [A]...

    # 循环遍历
    print("Loop from current:")
    for item in band:
        print(item, end=" → ")  # D → A → B → C →
    print()

    band = CircularBand(["A", "B", "C", "D", "E"])
    band.rotate(2)
    print("环形索引访问:")
    print(f"索引 0: {band[0]}")  # 当前元素 (C)
    print(f"索引 1: {band[1]}")  # 下一个元素 (D)
    print(f"索引 -1: {band[-1]}")  # 前一个元素 (B)

    print("\n切片操作:")
    print("band[:3]:", band[:3])  # [C, D, E]
    print("band[1:4]:", band[1:4])  # [D, E, A]

    print("\n数据持久化:")
    band.save("circular_data.pkl")
    loaded_band = CircularBand.load("circular_data.pkl")
    print("加载后的数据:", loaded_band)  # A → B → [C] → D → E → [A]...

    print("\n容量限制:")
    limited_band = CircularBand(["X", "Y", "Z"], capacity=3)
    print("初始状态:", limited_band)
    limited_band.append("A")
    print("添加'A'后:", limited_band)  # X → [A] → Z → [X]... (Max: 3)
    limited_band.expand(["B", "C"])
    print("扩展['B','C']后:", limited_band)  # B → C → [Z] → [B]... (Max: 3)

    print("\n完整功能演示:")
    band = CircularBand(["Red", "Green", "Blue"], capacity=5)
    print("初始:", band)  # [Red] → Green → Blue → [Red]... (Max: 5)

    band.append("Yellow")
    print("添加Yellow:", band)

    band.rotate(-1)
    print("左旋:", band)  # [Red] → Yellow → Green → Blue → [Red]... (Max: 5)

    band.expand(["Cyan", "Magenta"])
    print("扩展Cyan,Magenta:", band)

    print("转换为列表:", band.to_list())  # ['Magenta', 'Yellow', 'Green', 'Blue', 'Red']
    print("线性索引[2]:", band[2])
    print("环形索引[-1]:", band[-1])

    band.contract(2)
    print("收缩2个元素:", band)  # Red → [Green] → Blue → [Red]... (Max: 5)

    print("当前元素:", band.current())

    data = CircularBand([1, 2, 3, 4, 5, 6, 7, 8, 9])

    # 块转置 (3x3 矩阵)
    print("原始数据:", data)  # [1] → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9
    data.transpose(3)
    print("块转置后:", data)  # [1] → 4 → 7 → 2 → 5 → 8 → 3 → 6 → 9

    # 镜像反转
    data.mirror()
    print("镜像反转:", data)  # 9 → 6 → 3 → 8 → 5 → 2 → 7 → 4 → [1] → [9]...

    # 相邻交换
    data.swap()
    print("相邻交换:", data)  # [1] → 6 → 3 → 8 → 5 → 2 → 7 → 4 → 9 → [1]...

    print("=== 特殊字符测试 ===")
    band = CircularBand(["正常文本", "特殊\xff字符", "emoji😊"])
    print(band)

    print("\n=== 边界条件测试 ===")
    empty = CircularBand()
    empty.remove()
    empty.contract(5)

    print("\n=== 指针稳定性测试 ===")
    band = CircularBand(["X", "Y", "Z"])
    band.rotate(1)
    band.remove()
    print("当前元素:", band.current())  # 指向Z

    # 实时数据流处理
    history = CircularBand(capacity=50)
    history.append("homepage")

    # 用户导航
    history.append("about_page")
    history.append("contact_page")

    # 回退功能
    history.rotate(-1)
    print("返回上一页:", history.current())

    # 前进功能 history.rotate(1)
    print("前进到下一页:", history.rotate(1).current())
    from rime.allele import Allele

    genotypes_iter = Allele.genotype_iter_by_freq(1000, 360)
    # bands, stats = build_batch_bands(genotypes_iter, max_batches=9, start_batch=8)
    bands, total_processed = CircularBand.build_bands(genotypes_iter, max_batches=9, start_batch=8)

    # 打印每层概况
    for idx, band in enumerate(bands):
        size = (idx + 1) * 8
        print(f"Layer {idx + 1}: capacity={size}, filled_times={bands[idx].capacity}, current_len={len(band)}")
        # 查看该层当前数据（从 cursor 开始）
        print(band.to_list(from_current=True)[:min(8, len(band))])  # 只示例打印前 8 个

        matrix = band.to_matrix(block_size=4)
        for i, r in enumerate(matrix):
            cells = [str(x) if x is not None else '' for x in r]
            print(i, "  ".join([c for c in cells if c != '']))

        print("-" * 40)

    print("total processed:", total_processed)

    g = CircularBand.to_square_projection(bands, start_batch=8)
    print(len(g))
    for i, b in enumerate(g):
        print(i, b)  # 9,9：None

    g[9][9] = ('O', 'O')
    m = []
    for i, b in enumerate(g):
        m.extend(b)

    mapping = {g: i for i, g in enumerate(Allele.genotypes())}
    print(len(m), mapping)
    byte_data = Allele.states_encode(m, mapping)
    print(byte_data)
    print(byte_data.__sizeof__(), f"编码后大小: {len(byte_data)} 字节")
    m2 = Allele.states_decode(byte_data, len(m), mapping)
    print(m2)
    c_id = 9 * 19 + 9
    print(c_id, m2[c_id])
