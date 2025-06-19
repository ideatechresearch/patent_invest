class CircularBand:
    def __init__(self, initial_data=None, maxlen=None):
        """
        初始化环形数据结构

        :param initial_data: 初始数据（可迭代对象）
        :param maxlen: 最大容量限制（None表示无限制）
        """
        self.data = list(initial_data) if initial_data else []
        self.cursor = 0  # 当前指针位置
        self.maxlen = maxlen

        # 如果设置了最大容量，裁剪超出部分
        if maxlen is not None and len(self.data) > maxlen:
            self.data = self.data[-maxlen:]

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

    def remove(self):
        """删除指针位置元素（自动连接相邻元素）"""
        if not self.data:
            return

        del self.data[self.cursor]

        # 指针调整
        if not self.data:
            self.cursor = 0
        elif self.cursor >= len(self.data):
            self.cursor = 0

    def expand(self, items):
        """扩展多个元素"""
        if not items:
            return
        # 计算需要保留的新元素数量
        if self.maxlen is not None:
            available = max(0, self.maxlen - len(self.data))
            items = items[-available:]  # 只保留能插入的部分

        insert_pos = (self.cursor + 1) % (len(self.data) + 1)  # self.cursor + 1

        # 插入元素,在指针后插入
        self.data[insert_pos:insert_pos] = items
        # 容量限制处理,移除多余元素（从左侧开始移除）
        if self.maxlen is not None and len(self.data) > self.maxlen:
            del self.data[:len(self.data) - self.maxlen]
            insert_pos -= len(self.data) - self.maxlen
        # 更新指针到最后一个新元素,self.cursor += num_items
        self.cursor = min(insert_pos + len(items) - 1, len(self.data) - 1)

    def contract(self, k):
        """从指针处收缩 k 个元素"""
        if k <= 0 or not self.data:
            return

        start = self.cursor
        end = min(self.cursor + k, len(self.data))
        del self.data[start:end]
        # 指针调整
        if not self.data:
            self.cursor = 0
        else:
            self.cursor = min(self.cursor, len(self.data) - 1)

    def rotate(self, steps=1):
        """旋转结构（正数右移,顺时针旋转，负数左移,逆时针旋转）"""
        if not self.data:
            return
        self.cursor = (self.cursor + steps) % len(self.data)

    def transpose(self, block_size):
        """按块大小重组数据（类似矩阵转置）"""
        n = len(self.data)
        if n == 0:
            return
        if n % block_size != 0:
            raise ValueError(f"数据长度 {n} 必须能被块大小 {block_size} 整除")

        original_row = self.cursor // block_size
        original_col = self.cursor % block_size
        # 将数据分成块后转置
        blocks = [self.data[i:i + block_size] for i in range(0, n, block_size)]
        transposed = list(zip(*blocks))
        self.data = [item for block in transposed for item in block]
        # 调整指针位置
        self.cursor = original_col * (n // block_size) + original_row

    def mirror(self):
        """将数据结构首尾镜像反转"""
        if not self.data:
            return
        current_item = self.data[self.cursor]
        self.data.reverse()
        # 找到原元素的新位置
        self.cursor = self.data.index(current_item)

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

        if index >= 0:
            pos = (self.cursor + index) % n
        else:
            pos = (self.cursor + n + index) % n

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

        if start_from_current:
            return list(self)
        return self.data.copy()

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

    # 前进功能
    history.rotate(1)
    print("前进到下一页:", history.current())
