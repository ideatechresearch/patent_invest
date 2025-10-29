from collections import deque
from functools import wraps
import itertools


def chainable_method(func):
    """装饰器，使方法支持链式调用"""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        func(self, *args, **kwargs)
        return self

    return wrapper


class CircularBand:
    def __init__(self, initial_data=None, maxlen=None):
        """
        初始化环形数据结构

        :param initial_data: 初始数据（可迭代对象）
        :param maxlen: 最大容量限制（None表示无限制）
        """
        self.data = list(initial_data) if initial_data else []  # container
        self.cursor: int = 0  # 当前指针位置
        self.maxlen = maxlen

        # 如果设置了最大容量，裁剪超出部分
        if maxlen is not None and len(self.data) > maxlen:
            self.data = self.data[-maxlen:]

    @chainable_method
    def fill(self, new_data, reset_cursor: bool = True, truncate: bool = True):
        """
           用 new_data 替换 CircularBand 的所有内容。
           reset_cursor: 是否将 cursor 置为 0（默认 True）。
           truncate: 如果 new_data 长度超过 maxlen，是否截断保留尾部（最近的部分）。仅在 maxlen 不为 None 时生效。
        """
        new_list = list(new_data) or []
        # 处理 maxlen
        if self.maxlen is not None and len(new_list) > self.maxlen:
            if truncate:
                new_list = new_list[-self.maxlen:]
            else:
                raise ValueError(f"new_data length ({len(new_list)}) exceeds maxlen ({self.maxlen})")

        self.data = new_list
        self.cursor = 0 if reset_cursor else min(self.cursor, len(self.data) - 1 if self.data else 0)

    @chainable_method
    def append(self, item):
        """在指针后插入元素"""
        if self.maxlen is not None and len(self.data) >= self.maxlen:
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
    def remove(self):
        """删除指针位置元素（自动连接相邻元素）"""
        if not self.data:
            return

        del self.data[self.cursor]
        if not self.data:
            self.cursor = 0
        elif self.cursor >= len(self.data):
            self.cursor = 0

    @chainable_method
    def expand(self, items):
        """扩展多个元素"""
        if not items:
            return
        n = len(self.data)
        m = len(items)

        # 计算需要保留的新元素数量
        if self.maxlen is not None:
            available = max(0, self.maxlen - n)
            items = items[-available:]  # 只保留能插入的部分
            m = len(items)

        insert_pos = (self.cursor + 1) % (n + 1)  # self.cursor + 1
        # 插入元素,在指针后插入
        self.data[insert_pos:insert_pos] = items
        # 容量限制处理,移除多余元素（从左侧开始移除）
        if self.maxlen is not None and len(self.data) > self.maxlen:
            excess = len(self.data) - self.maxlen
            del self.data[:excess]
            insert_pos -= excess
        # 更新指针到最后一个新元素,self.cursor += num_items
        self.cursor = min(max(0, insert_pos + m - 1), len(self.data) - 1)

    @chainable_method
    def contract(self, k):
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
    def rotate(self, steps=1):
        """旋转结构（正数右移,顺时针旋转，负数左移,逆时针旋转）"""
        if not self.data:
            return
        self.cursor = (self.cursor + steps) % len(self.data)

    @chainable_method
    def transpose(self, block_size: int = 4):
        """按块大小重组数据（类似矩阵转置）,当作(rows=block_size, cols=n/block_size) 的矩阵（按列填充）"""
        n = len(self.data)
        if n == 0:
            return
        if n % block_size != 0:
            raise ValueError(f"数据长度 {n} 必须能被块大小 {block_size} 整除")

        original_row = self.cursor // block_size
        original_col = self.cursor % block_size
        # 将数据分成块后转置(每列是 block_size 长）
        blocks = [self.data[i:i + block_size] for i in range(0, n, block_size)]
        transposed = list(zip(*blocks))
        self.data = [item for block in transposed for item in block]  # 按行展平转置后的矩阵
        # 调整指针位置
        self.cursor = original_col * (n // block_size) + original_row

    @chainable_method
    def mirror(self):
        """将数据结构首尾镜像反转"""
        if not self.data:
            return
        current_item = self.data[self.cursor]
        self.data.reverse()
        # 找到原元素的新位置
        self.cursor = self.data.index(current_item)

    @chainable_method
    def swap(self):
        """交换当前元素与下一个元素，并将指针移到下一个元素"""
        n = len(self.data)
        if n < 2:
            return

        next_pos = (self.cursor + 1) % n
        self.data[self.cursor], self.data[next_pos] = self.data[next_pos], self.data[self.cursor]
        self.cursor = next_pos

    def current(self):
        """获取当前元素"""
        return self.data[self.cursor] if self.data else None

    def __iter__(self):
        """从当前指针开始循环遍历"""
        n = len(self.data)
        for i in range(n):
            yield self.data[(self.cursor + i) % n]

    def __len__(self):
        """返回数据长度"""
        return len(self.data)

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
            f" (Max: {self.maxlen})" if self.maxlen is not None else "")

    def __repr__(self):
        return f"CircularBand(data={self.data}, cursor={self.cursor}, maxlen={self.maxlen})"

    def to_list(self, start_from_current=True):
        """
        将环形数据转换为列表

        :param start_from_current: 是否从当前元素开始
        :return: 数据列表
        """
        if not self.data:
            return []
        return list(self) if start_from_current else self.data.copy()

    def to_matrix(self, block_size: int = 4) -> list[list]:
        """
        将当前 data 视作按列填充的矩阵并返回（不修改 data）。
        语义：把 data 按列填充到 4 行（block_size 行），即 column-major 填充，
        最后返回按行的矩阵（rows x cols）。
        要求 len(data) % block_size == 0（否则最后一列会被补 None）。
        """
        import math
        n = len(self.data)
        if n == 0:
            return []

        cols = math.ceil(n / block_size)
        rows = block_size
        # 填充扁平数据到 column-major 矩阵
        matrix = [[None] * cols for _ in range(rows)]
        it = iter(self.data)
        for c in range(cols):
            for r in range(rows):
                try:
                    matrix[r][c] = next(it)
                except StopIteration:
                    matrix[r][c] = None
        return matrix

    def to_round(self, n_slices: int = 4, start_from_current: bool = False,
                 pad_value=None, clockwise: bool = True) -> list[tuple]:
        """
        将band分成n_slices段（默认4）。每层切分的段数（如 4=方形、6=蜂窝、8=八边）
        比如 n_slices=4 表示 top/right/bottom/left 四边；
             n_slices=6 表示六个方向的环形切分。

        Args:
            n_slices: 切分段数
            start_from_current: 是否从cursor开始线性展开
            pad_value: 若数据长度不足，填充该值
            clockwise: 是否按顺时针方向切
        """
        data = self.to_list(start_from_current=start_from_current)
        n = len(data)
        if n_slices <= 0:
            raise ValueError("n_slices 必须为正整数")

        per_slice = n // n_slices
        remainder = n % n_slices
        expected = per_slice * n_slices + (1 if remainder else 0)

        # 如果不够整除，就补齐到能整除
        if n < expected:
            pad_len = expected - n
            data = data + [pad_value] * pad_len  # 使长度为 expected

        chunks = []  # 重新分块
        per_edge = len(data) // n_slices  # step
        for i in range(n_slices):
            start = i * per_edge
            end = (i + 1) * per_edge
            chunks.append(tuple(data[start:end]))  # top,right,bottom,left

        if not clockwise:
            chunks.reverse()

        return chunks

    @classmethod
    def to_square_projection(cls, bands: list['CircularBand'], base: int = 8, fill_center_with=None) -> list:
        """
        使用 bands[i].to_round(per_side) 将 bands 投影到方阵。
        - base: 第一层的 batch_size 基数（如 8），第 i 层的 batch_size = base*(i+1)
        """
        n_layers = len(bands)
        grid_size = 2 * n_layers + 1
        center = n_layers  # 中心坐标 (center, center)
        grid = [[None for _ in range(grid_size)] for _ in range(grid_size)]

        for i, band in enumerate(bands):
            layer = i + 1  # radius 第几层（半径）
            batch_size = base * layer  # e.g. 8,16,...,8*n = 4*per_edge
            assert len(band) == batch_size, f"第 {i} 层数据长度应为 {batch_size}，实际为 {len(band)}"

            slices = band.to_round(n_slices=4, start_from_current=False, pad_value=fill_center_with)
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
        if fill_center_with is not None:
            grid[center][center] = fill_center_with

        return grid

    def save(self, filename):
        """保存数据到文件（包括指针位置）"""
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump({'data': self.data, 'cursor': self.cursor, 'maxlen': self.maxlen}, f)

    @classmethod
    def load(cls, filename):
        """从文件加载数据"""
        import pickle
        import os
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found")

        with open(filename, 'rb') as f:
            state = pickle.load(f)

        band = cls(initial_data=state['data'], maxlen=state['maxlen'])
        band.cursor = state['cursor'] % len(band) if band else 0
        return band

    @classmethod
    def build_bands(cls, gen_iter, max_batches: int = 9, start_batch: int = 8):
        batch_sizes = [start_batch * (i + 1) for i in range(max_batches)]
        bands = [cls(initial_data=[], maxlen=size) for size in batch_sizes]

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
    bands = [CircularBand(initial_data=[], maxlen=size) for size in batch_sizes]

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


class LRUCache:
    def __init__(self, capacity: int):
        from collections import OrderedDict
        self.stack = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key in self.stack:
            self.stack.move_to_end(key)
            return self.stack[key]
        else:
            return None

    def put(self, key, value) -> None:
        if key in self.stack:
            self.stack[key] = value
            self.stack.move_to_end(key)
        else:
            self.stack[key] = value
        if len(self.stack) > self.capacity:
            self.stack.popitem(last=False)

    def change_capacity(self, capacity):
        self.capacity = capacity
        for i in range(len(self.stack) - capacity):
            self.stack.popitem(last=False)

    def delete(self, key):
        if key in self.stack:
            del self.stack[key]

    def keys(self):
        return self.stack.keys()

    def __len__(self):
        return len(self.stack)

    def __contains__(self, key):
        return key in self.stack


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
    limited_band = CircularBand(["X", "Y", "Z"], maxlen=3)
    print("初始状态:", limited_band)
    limited_band.append("A")
    print("添加'A'后:", limited_band)  # X → [A] → Z → [X]... (Max: 3)
    limited_band.expand(["B", "C"])
    print("扩展['B','C']后:", limited_band)  # B → C → [Z] → [B]... (Max: 3)

    print("\n完整功能演示:")
    band = CircularBand(["Red", "Green", "Blue"], maxlen=5)
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
    history = CircularBand(maxlen=50)
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
        print(f"Layer {idx + 1}: capacity={size}, filled_times={bands[idx].maxlen}, current_len={len(band)}")
        # 查看该层当前数据（从 cursor 开始）
        print(band.to_list(start_from_current=True)[:min(8, len(band))])  # 只示例打印前 8 个

        matrix = band.to_matrix(block_size=4)
        for i, r in enumerate(matrix):
            cells = [str(x) if x is not None else '' for x in r]
            print(i, "  ".join([c for c in cells if c != '']))

        print("-" * 40)

    print("total processed:", total_processed)

    g = CircularBand.to_square_projection(bands)
    print(len(g))
    for i, b in enumerate(g):
        print(i, b)  # 9,9：None
