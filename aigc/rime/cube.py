from rime.base import class_property, class_cache, chainable_method, class_status
import numpy as np
import random, math
from collections import deque, defaultdict


class CubeBase:
    '''
    逻辑 → 几何 → 群论 → 可视化,状态系统 World / State Space
    运算必须封闭
    表示不能混层
    不变量是结构给的,不是定义的
    贴纸世界是表示的一种投影，投影不可逆
    axis = 0 → x → R/L
    axis = 1 → y → U/D
    axis = 2 → z → F/B
    # AXIS_FACE defines the canonical mapping between Cartesian axes and cube faces.

    Coordinate system:
      Right-handed Cartesian system.
      +X → Right  (R)
      +Y → Up     (U)
      +Z → Front  (F)

    AXIS_FACE[axis] = (POS_FACE, NEG_FACE)
      POS_FACE: face whose outward normal aligns with +axis direction
                (outer layer: layer == mid / n-1)
      NEG_FACE: face whose outward normal aligns with -axis direction
                (outer layer: layer == -mid / 0)

    Rotation direction convention:
      A positive rotation `d` is defined as clockwise when viewed
      along the face outward normal (right-hand rule).

    NOTE:
      Face normal directions are fully defined by AXIS_FACE.
      No additional geometric inference (e.g. dot products) is required.
    '''
    # 哪些面在轴的正/负一侧(POS_FACE, NEG_FACE)_SIGN,几何法向,用于坐标推导
    # 约定：layer == +mid  → POS_FACE (法向 = +axis) 几何意义, 实际可见面 = AXIS_FACE[axis][0]
    #      layer == -mid  → NEG_FACE (法向 = -axis) 几何意义, 实际可见面 = AXIS_FACE[axis][1]
    AXIS_FACE = [
        ('R', 'L'),  # X axis (0), X+ → R, X− → L
        ('U', 'D'),  # Y axis (1), Y+ → U, Y− → D
        ('F', 'B'),  # Z axis (2), Z+ → F, Z− → B
    ]  # YOLO 正轴方向,物理右手坐标一致, VISIBLE_FACE 人眼观察标准,视角切换:法向 = −axis
    AXIS_STRIP = (
        ['U', 'F', 'D', 'B'],  # 'X', 从 +X 看 CCW ,+Y 作为参考
        ['F', 'R', 'B', 'L'],  # 'Y',+Z 作为参考
        ['U', 'L', 'D', 'R'],  # 'Z',+X 作为参考
    )  # CCW 视角,从 +axis 方向看过去,4 元环路,trip 顺序,d 是从“面法向外看”时的顺时针旋转次数（右手规则）
    # AXIS_STRIP = (
    #     ['U', 'B', 'D', 'F'],  # X
    #     ['F', 'L', 'B', 'R'],  # Y
    #     ['U', 'R', 'D', 'L'],  # Z
    # )
    # === 依赖 FACES 生成派生结构 ===
    AXIS_VEC = np.eye(3, dtype=int)
    FACE_UP = {
        'U': -AXIS_VEC[2],  # -Z: [0, 0, -1],AXIS_VEC[0]
        'D': AXIS_VEC[2],  # Z:[0, 0, 1]

        'F': AXIS_VEC[1],  # Y:[0, 1, 0]
        'B': AXIS_VEC[1],

        'R': AXIS_VEC[1],
        'L': AXIS_VEC[1],
    }  # 固定局部坐标系
    # === 固定顺序 ===
    FACES = ['U', 'D', 'F', 'B', 'L', 'R']  # 面标识 上 下 前 后 左 右,通常 0=U, 1=D, 2=F, 3=B, 4=L, 5=R
    # (x, y, z) ∈ {±1, ±1, ±1}
    CORNER_POS_SIGNS = [
        (+1, +1, +1),  # URF
        (-1, +1, +1),  # UFL
        (-1, +1, -1),  # ULB
        (+1, +1, -1),  # UBR
        (+1, -1, +1),  # DFR
        (-1, -1, +1),  # DLF
        (-1, -1, -1),  # DBL
        (+1, -1, -1),  # DRB
    ]  # 排序逻辑是魔方标准：从U层右前开始，顺时针绕一圈；D层类似（但右手调整）
    EDGE_POS_SIGNS = [
        (+1, +1, 0),  # UR
        (0, +1, +1),  # UF
        (-1, +1, 0),  # UL
        (0, +1, -1),  # UB
        (+1, 0, +1),  # RF
        (-1, 0, +1),  # LF
        (-1, 0, -1),  # LB
        (+1, 0, -1),  # RB
        (+1, -1, 0),  # DR
        (0, -1, +1),  # DF
        (-1, -1, 0),  # DL
        (0, -1, -1),  # DB
    ]  # EDGE_COORDS
    # 对 y 轴的几何分层 quotient，几何派生接口
    # corner positions
    U_CORNER_POSITIONS = tuple(i for i, (_, y, _) in enumerate(CORNER_POS_SIGNS) if y == +1)  # (0, 1, 2, 3)
    D_CORNER_POSITIONS = tuple(i for i, (_, y, _) in enumerate(CORNER_POS_SIGNS) if y == -1)  # (4, 5, 6, 7)
    # edge positions index（position on cube）
    SLICE_POSITIONS = tuple(i for i, (_, y, _) in enumerate(EDGE_POS_SIGNS) if y == 0)  # slice {FR, FL, BL, BR},从前向后扫描
    NON_SLICE_POSITIONS = tuple(i for i, (_, y, _) in enumerate(EDGE_POS_SIGNS) if y != 0)

    # [0, 1, 2, 3, 8, 9, 10, 11] 非 slice [i for i in range(12) if i not in SLICE_POSITIONS]

    # # Define corner and edge positions
    # corner_positions = np.array(CORNER_POS_SIGNS, dtype=int)
    # edge_positions = np.array(EDGE_POS_SIGNS, dtype=int)

    def __init__(self, n: int = 3):
        self.n = n
        self.solved_idx = np.arange(6 * n * n, dtype=np.uint32).reshape(6, n, n)
        self.solved = np.zeros((6, n, n), dtype=np.uint8)
        for f in range(6):
            self.solved[f, :, :] = f
        self.SOLVED_CORNERS_MAP = self.solved_corners_map()
        self.SOLVED_EDGES_MAP = self.solved_edges_map()

    @property
    def mid(self) -> int:
        return self.n // 2

    @property
    def center_layers(self) -> list:
        return self.center_layers_list(self.n)

    @staticmethod
    def idx_to_state(state_idx: np.ndarray) -> np.ndarray:
        """颜色视图"""
        n = state_idx.shape[1]
        return (state_idx // (n * n)).astype(np.uint8)

    def is_solved(self, state: np.ndarray) -> bool:
        return bool(np.array_equal(state, self.solved))  # not (state ^ self.solved).any()

    def is_solved_idx(self, state_idx: np.ndarray) -> bool:
        return bool(np.array_equal(self.idx_to_state(state_idx), self.solved))

    def diff_coords(self, state: np.ndarray) -> np.ndarray:
        diff_mask = state != self.solved
        return np.argwhere(diff_mask)  # nonzero

    def heuristic(self, state: np.ndarray):
        """
        估价函数：错误块的数量（简单启发）,对 BFS/IDA*/Beam search 可用,小魔方适用
        heuristic(embedding)
        """
        errors = np.count_nonzero(state != self.solved)
        return errors // max(1, self.n)  # 每个错误影响多个面

    @staticmethod
    def encode(state: np.ndarray) -> bytes:
        return np.ascontiguousarray(state, dtype=np.uint8).tobytes()

    @staticmethod
    def embedding(state_idx: np.ndarray) -> np.ndarray:
        n = state_idx.shape[1]
        # (state_idx // (n * n)).astype(float) + (state_idx % (n * n)).astype(float) / (n * n)
        return state_idx.astype(float) / (n * n)

    @staticmethod
    def get_data(state: np.ndarray, coords_def: list | tuple) -> tuple:
        '''把 piece 看成一个“颜色集合”，忽略朝向，只关心它是哪个 piece'''
        return tuple(sorted(state[f, r, c].tolist() for f, r, c in coords_def))

    def get_corners(self, state: np.ndarray) -> np.ndarray:
        """
        返回 8 个角块的三颜色编号（按顺序）shape = (8, 3),
        角块位置： (face, row, col) 三元组的 3 个集合
        """
        res = np.empty((8, 3), dtype=state.dtype)
        for i, corner in enumerate(self.corner_coords(self.n)):
            # corner 是 [(face,row,col), ...]
            for j, (f, r, c) in enumerate(corner):
                res[i, j] = state[f, r, c]
        return res

    def get_edges(self, state: np.ndarray) -> np.ndarray:
        """返回 12 个角块的2颜色编号（按顺序）shape = (12,2)"""
        res = np.empty((12, 2), dtype=state.dtype)
        for i, edge in enumerate(self.edge_coords(self.n)):
            for j, (f, r, c) in enumerate(edge):
                res[i, j] = state[f, r, c]
        return res

    def encode_state(self, state: np.ndarray) -> np.ndarray:
        """48: 8*3+12*2,物理块不去重"""
        return np.concatenate([self.get_corners(state).ravel(), self.get_edges(state).ravel()]).astype(np.uint8)

    def solved_corners_map(self) -> dict:
        """
        角块的 cubie id
        用字典加速 lookup，建立 solved corner 的颜色集合 → index 映射
        忽略朝向，否则同一个块旋转后会被误认为不同块,角块的 twist 信息在贴纸级处理里丢失
        slot 0 为 reference （基准面）
        """
        solved = self.get_corners(self.solved)  # (8, 3) np.sort(, axis=1)
        return {frozenset(c): (pid, c[0]) for pid, c in enumerate(solved)}

    def corner_ids_ori(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        返回每个 corner 对应的 piece id（0~7）, corners_perm (8)
        perm 表示 piece 的重排，当前位置 i 的角块，原来是 solved 的哪一块？ 群作用矩阵里计算 twist
        corner_ori 不是物理量，是约定量
        """
        corners = self.get_corners(state)
        perm = np.empty(8, dtype=np.int8)
        ori = np.empty(8, dtype=np.int8)
        for i, c in enumerate(corners):
            pid, ref = self.SOLVED_CORNERS_MAP[frozenset(c)]
            perm[i] = pid
            ori_raw = list(c).index(ref)  # 原始索引：0/1/2
            ori[i] = (-ori_raw) % 3  # np.roll(c, -k),list(solved[pid]).index(ref)-ori_raw

        # ori = (ori - ori[0]) % 3  # 全局 orientation gauge fix
        ori[-1] = (-ori[:-1].sum()) % 3  # 把 orientation 投影到合法子空间,修正最后一个角方向
        return perm, ori

    def heuristic_corner_perm(self, state: np.ndarray):
        corners_perm, _ = self.corner_ids_ori(state)
        return np.count_nonzero(corners_perm != np.arange(8))

    def solved_edges_map(self) -> dict:
        """
        标准顺序,依赖 solved_edges 的正确定义（顺序必须匹配参考色先）
        cubie id : ref  0/1
        """
        solved = self.get_edges(self.solved)  # (12, 2)
        edge_map = {}
        for pid in range(12):
            a, b = solved[pid]
            edge_map[(a, b)] = (pid, 0)  # forward 正向映射：标准颜色顺序 → id,ori[i] = 0
            edge_map[(b, a)] = (pid, 1)  # flipped 翻转映射：同一个块,ori[i] = 1
        return edge_map

    def edge_ids_ori(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """ edges_perm  (12) 保留方向,按标准顺序"""
        edges = self.get_edges(state)  # current_edges
        perm = np.empty(12, dtype=np.int8)
        ori = np.empty(12, dtype=np.int8)
        for i, (a, b) in enumerate(edges):
            pid, o = self.SOLVED_EDGES_MAP[(a, b)]
            perm[i] = pid
            ori[i] = o
        ori[-1] = (-ori[:-1].sum()) % 2  # 修正最后一个边方向
        return perm, ori

    def orbit_perm(self, state: np.ndarray) -> list[np.ndarray]:
        """
           从 sticker-level 状态生成 orbit_perm
           state: np.ndarray, shape = (6, n, n), 每个元素是颜色或编号
        """
        centers = CubeBase.get_center_rings(self.n)
        flat = []  # flatten 所有中心贴纸，顺序严格对应 center_orbits
        for fidx, rings in enumerate(centers):
            for ring in rings:
                for r, c, _ in ring:
                    flat.append(state[fidx, r, c])

        center_colors = np.array(flat, dtype=np.int32)

        orbits: list[list[int]] = CubeBase.center_orbits(self.n)
        orbits = [o for o in orbits if len(o) > 1]
        orbit_perm: list[np.ndarray] = []  # 每个 array 长度 = len(orbit)
        for orbit_idx in orbits:
            pieces = center_colors[orbit_idx]  # 映射到 0..k-1（orbit 内相对编号）
            order = np.argsort(pieces)
            rel = np.empty_like(order)
            rel[order] = np.arange(len(order))
            orbit_perm.append(rel)

        return orbit_perm

    def basic_generators(self):
        """基础生成元,逻辑层（axis, layer, direction）与几何层解耦,有限邻域 moves（减枝！！）"""
        for axis in range(3):
            for layer in self.center_layers:
                for direction in (-1, 1, 2):  # direction 只用 ±1，2 步可视为两步重复
                    yield axis, layer, direction

    def scramble(self, moves: int = 20) -> list:
        """生成打乱序列，返回 move list"""
        scramble_moves = []
        for _ in range(moves):
            axis = random.choice(range(3))
            layer = random.choice(self.center_layers)
            direction = random.choice((1, 2, 3))
            scramble_moves.append((axis, layer, direction))
        return scramble_moves  # -> act/apply

    @classmethod
    def act_moves(cls, state: np.ndarray, moves: list | tuple):
        """
        连续作用,replace
        action space, action:群生成元
        (axis, layer, direction)
        s_{t+1} = T(s_t, a_t)
        """
        if not isinstance(moves, list):
            moves = [moves]
        for axis, layer, direction in moves:
            cls.rotate_core(state, axis, layer, direction)

    @staticmethod
    def invert_moves(moves: list[tuple]) -> list[tuple]:
        """move 的逆 将 moves 转成可还原的逆操作序列（反向 + 方向反）"""
        return [(axis, layer, -direction) for (axis, layer, direction) in reversed(moves)]

    @staticmethod
    def is_inverse(path: list[tuple], axis: int, layer: int, direction: int) -> bool:
        """
        禁止与上一个动作在同一面（axis+layer）上连续转动且总效果为 0 mod 4
        两个动作加起来等价于什么都没做
        """
        # forbid immediate reversal
        if not path:
            return False
        pa, pl, pd = path[-1]
        if axis == pa and layer == pl:
            return (pd + direction) % 4 == 0
        return False

    @staticmethod
    def commutator(A: list, B: list) -> list:
        """
        交换子,制造局部扰动,奇偶性不变
        A, B: move list
        return: [A, B] = A B A⁻¹ B⁻¹
        """
        return A + B + CubeBase.invert_moves(A) + CubeBase.invert_moves(B)

    @staticmethod
    def conjugate(A: list, B: list) -> list:
        """
        共轭 A B A⁻¹ 改变作用位置,保持结构不变 or A⁻¹ B A
        """
        return A + B + CubeBase.invert_moves(A)

    def transform(self, move: str) -> list[tuple]:
        """
        解析层 Human Move String → transform → sticker simulation（仅用于构表）
        通用 NxN 解析，支持标准记法：
            U, U', U2
            R, L, F, B, D
            Rw, Rw', Rw2, Uw, Fw ...
            2Rw, 3Uw',2Rw2,3U,3Uw',2Fw2  等 |A| = 18
        返回实际执行的 primitive move 列表：[(axis, layer, direction), ...]
        """
        if not move:
            raise ValueError("动作不能为空")

        turn_times = 1
        direction = 1
        import re
        # --- 解析方向 ---
        if move.endswith("2"):
            turn_times = 2
            move = move[:-1]

        if move.endswith("'"):
            direction = -1
            move = move[:-1]

        # --- 正则解析宽度（前缀数字）
        m = re.match(r"(\d*)([URFDLB])(w?)$", move)
        if not m:
            raise ValueError(f"无法解析动作: {move}")

        width_txt, face, wide_flag = m.groups()
        if face not in self.FACES:
            raise ValueError(f"未知面: {face}")

        # 宽度：无数字 → 默认 1；如果有 'w' 则默认 = 2
        if width_txt:
            width = int(width_txt)
            if width < 1 or width > self.n:
                raise ValueError(f"宽度 {width} 超出魔方阶数 {self.n}")
        else:
            width = 2 if wide_flag else 1

        axis, side = self.face_axis[face]
        if side == 0:  # positive_side 正轴面
            layers = list(range(width))  # 正面方向,U:顶部向下 width 层
        else:
            layers = [self.n - 1 - i for i in range(width)]  # 反: list(range(N - 1, N - width - 1, -1))

        # ---- primitive moves ----
        prim_moves = []
        for _ in range(turn_times):
            for layer_idx in layers:
                final_dir = direction if side == 0 else -direction  # “正轴面的顺时针” 在 cube 标准中方向不同
                layer = layer_idx - self.mid
                if self.n % 2 == 0 and layer_idx >= self.mid:  # 偶数阶需要跳过中心 0
                    layer += 1
                prim_moves.append((axis, layer, final_dir))

        return prim_moves  # 记录步骤

    @staticmethod
    def permutation_parity(perm: np.ndarray | list) -> int:
        """
        Return 0 for even, 1 for odd permutation
        """
        visited = np.zeros(len(perm), dtype=bool)
        parity = 0
        for i in range(len(perm)):
            if visited[i]:
                continue
            cycle_len = 0
            j = i
            while not visited[j]:
                visited[j] = True
                j = perm[j]
                cycle_len += 1
            if cycle_len > 0:
                parity ^= (cycle_len - 1) & 1
        return parity

    def dfs(self, state: np.ndarray, depth: int, bound, visited, path, max_depth: int = 25):
        key = self.encode_state(state)
        if key in visited:
            return math.inf, None
        visited.add(key)  # 对象状态快照

        h = self.heuristic(state)
        f = depth + h
        if f > bound:
            visited.remove(key)
            return f, None

        if h == 0:
            return True, path.copy()

        if depth >= max_depth:
            visited.remove(key)
            return math.inf, None

        best = math.inf
        for move in self.basic_generators():
            if self.is_inverse(path, *move):
                continue

            next_state = self.rotate_state(state, *move)
            path.append(move)

            t, sol = self.dfs(next_state, depth + 1, bound, visited, path, max_depth)

            path.pop()
            if t is True:
                return True, sol
            best = min(best, t)

        visited.remove(key)
        return best, None

    @class_status('已废弃')
    def heuristic_corner_old(self, state: np.ndarray) -> int:
        '''隐含了三个前提：
        corner 编号是隐式的（靠位置顺序）
        orientation 不可信 → 用 set 抹掉方向
        solved 状态是一个固定 sticker 模板
        Singmaster / sticker-based 的世界观
        '''
        wrong = 0  # number_of_wrong_corners
        solved = self.get_corners(self.solved)
        cur = self.get_corners(state)
        for a, b in zip(cur, solved):
            if set(a) != set(b):
                wrong += 1
        return wrong

    def heuristic_center(self, state: np.ndarray, r: int = 1) -> int:
        """
        计算中心错误数量，默认中心启发：统计以 mid 为中心的 (2k+1)x(2k+1) 区域中不等于 center color 的数目。
        r 控制区域大小 (2r+1)x(2r+1) 这里默认取 k= (n//2)//2 令中心区域足够大；可以改成只统计 (mid,mid) 周围 3x3。
        越小越接近目标（可用于 IDA*）
        """
        mid = self.mid
        k = max(1, r)  # 可调整为 1 (3x3) 或更大，跳过十字、边缘、角落，只取 3x3 / 5x5 / ... 中心块
        wrong = 0
        for f in range(6):
            face = state[f]
            target = self.solved[f, mid, mid]  # face[mid, mid]

            region = face[mid - k:mid + k + 1, mid - k:mid + k + 1]
            wrong += np.count_nonzero(region != target)
        return int(wrong)

    @classmethod
    def get_vars(cls):
        """获取类中的变量名"""
        return [name for name, value in vars(cls).items() if
                not (callable(value) or isinstance(value, (classmethod, staticmethod)))]

    @class_property('FACE_NORMAL')
    def face_normal(cls) -> dict[str, np.ndarray]:
        '''FACE_NORMAL = {
        'R': (1, 0, 0),
        'L': (-1, 0, 0),
        'U': (0, 1, 0),
        'D': (0, -1, 0),
        'F': (0, 0, 1),
        'B': (0, 0, -1),
        }'''
        mapping = {}
        for axis, (pos_face, neg_face) in enumerate(cls.AXIS_FACE):
            mapping[pos_face] = cls.AXIS_VEC[axis]
            mapping[neg_face] = -cls.AXIS_VEC[axis]
        return mapping

    @class_property('FACE_DEF')
    def face_def(cls) -> dict[str, tuple]:
        mapping = {}  # 六个面的法向和基向 dx, dy
        for face, normal in cls.face_normal.items():  # face 法向量
            up = cls.FACE_UP[face]
            right = np.cross(normal, up)  # 面内基向量（右手系）,right2 = world_up × forward = -right
            mapping[face] = (normal, right, up)

        return mapping

    @class_property('FACE_AXIS')
    def face_axis(cls) -> dict[str, tuple]:
        # 哪个面属于哪个轴
        return {face: (axis, side)
                for axis, pair in enumerate(cls.AXIS_FACE)
                for side, face in enumerate(pair)}

    @class_property('FACE_DEF_AXIS')
    def face_idx(cls) -> dict:
        return {f: i for i, f in enumerate(cls.FACES)}

    @staticmethod
    def center_layers_list(n: int) -> list:
        mid, c = divmod(n, 2)
        if c == 1:
            return list(range(-mid, mid + 1))  # 奇数阶：中心在 0
        return [i for i in range(-mid, mid + 1) if i != 0]  # 偶数阶：无中心层

    @staticmethod
    def layer_index(layer: int, n: int) -> int:
        """数组索引空间"""
        mid, c = divmod(n, 2)
        layer_idx = layer + mid
        if c == 0 and layer >= mid:
            layer_idx -= 1
        return layer_idx

    @staticmethod
    def layer_to_logic(layer: int, n: int) -> float:
        """ 逻辑 / 拓扑坐标空间,逻辑世界坐标,半连续,abstract ℤ / ℤ +½ """
        c = n % 2
        if c == 1:  # 奇数阶：整数 → 整数
            return float(layer)
        # 偶数阶：整数 layer → 半整数坐标
        return layer - 0.5 if layer > 0 else layer + 0.5

    @staticmethod
    def layer_to_geom(layer: int, n: int) -> float:
        """几何模型空间,ℝ³ model space 动画"""
        return -n / 2 + 0.5 + layer

    @classmethod
    def face_rc_to_xyz(cls, face: str, r: int, c: int, n: int):
        """
        把 某个面上的 row/col 映射到世界坐标系 XYZ
        把离散拓扑嵌入到连续空间中保持方向一致性
        face 上的 row / col 方向（固定定义）
        约定：
        #   row 方向 = Y×normal,Z
        #   col 方向 = normal×row
         n = 5 [-2, -1, 0, 1, 2]
         n = 4 [-1.5, -0.5, 0.5, 1.5]
        """
        # 面中心
        k = (n - 1) / 2
        dr = k - r  # r 向下 → y 减小
        dc = c - k  # c 向右 → x 增大
        normal, right, up = cls.face_def[face]

        # 世界坐标,normal 是面中心，dc*right + dr*up2 扩展到局部坐标
        pos = normal + dc * right + dr * up
        if n % 2 == 1:  # 离散化,保证严格整数
            return np.round(pos).astype(int)
        return np.sign(pos) * np.floor(np.abs(pos) + 0.5).astype(int)  # 原点缺失,parity 成为全局约束

    @classmethod
    def face_basis(cls, face: str):
        """
        带法向的面内正交基,面自身的几何事实
        确定局部行列方向,纯几何面内坐标系
        约定：
        - u_dir：面内向上（对应 row）
        - v_dir：面内向右（对应 col 正方向）
        - normal：面外法向
        """
        normal = cls.face_normal[face].astype(float)

        u_dir = cls.FACE_UP[face].astype(float)
        u_dir /= np.linalg.norm(u_dir)  # 归一化

        v_dir = np.cross(normal, u_dir)
        v_dir /= np.linalg.norm(v_dir)

        return normal, u_dir, v_dir

    @classmethod
    def pos_to_face_rc(cls, pos: np.ndarray, normal, n: int) -> tuple:
        for f, nvec in cls.face_normal.items():
            if np.allclose(normal, nvec):
                face = f
                break
        else:
            raise ValueError(f"invalid normal {normal}")

        normal, u_dir, v_dir = cls.face_basis(face)
        u = np.dot(pos, u_dir)
        v = np.dot(pos, v_dir)
        center = (n - 1) / 2.0
        c = int(round(u + center))
        r = int(round(v + center))
        if not (0 <= r < n and 0 <= c < n):
            raise ValueError(f"out of face: {face}, r={r}, c={c}")

        return face, r, c

    @staticmethod
    def sticker_pos(normal, u_dir, v_dir, r: int, c: int, n: int) -> np.ndarray:
        """
         返回贴纸中心在世界坐标系中的 (x, y, z), world 连续浮点坐标
         与 face_rc_to_xyz 一致的面内基
         - 立方体中心在原点，坐标范围 [-center, center]
         - r, c: 从 0 到 n-1
         """
        center = (n - 1) / 2.0
        face_center = normal * center
        # 局部坐标映射到中心坐标 [-center, center]
        s_v = c - center  # col → right x:j
        s_u = r - center  # row → down y:i
        pos_rel = u_dir * s_u + v_dir * s_v
        return face_center + pos_rel  # float, 保留 ±0.5

    @class_cache(cache_name='_LAYER_CACHE', key=lambda axis, layer, n: (axis, layer, n))
    @classmethod
    def get_layer_stickers(cls, axis: int, layer: int, n: int = 3):
        """
        Returns a list of (fidx, r, c, pos) for stickers in the given layer along the axis.
        Assumes center at origin, layers from -x to x where x = (n-1)/2.
        """
        stickers = defaultdict(list)
        axis_vec = cls.AXIS_VEC[axis].astype(float)
        layer_coord = cls.layer_to_logic(layer, n)
        # R = cls.rot90_matrix(axis, 1)
        for idx, face in enumerate(cls.AXIS_STRIP[axis]):  # 已调整为统一 CCW 顺序
            normal, u_dir, v_dir = cls.face_basis(face)
            if abs(np.dot(normal, axis_vec)) > 1e-6:
                continue  # 整个面或不在该 layer
            for r in range(n):
                for c in range(n):
                    xyz = cls.sticker_pos(normal, u_dir, v_dir, r, c, n)  # 中间态
                    # k = int(round(np.dot(xyz, axis_vec)))
                    # xyz_rot=R @ xyz
                    if np.isclose(xyz[axis], layer_coord, atol=1e-6):  # 中心坐标,abs(xyz[axis] - layer) < 1e-6
                        stickers[face].append((r, c, xyz))

        return stickers

    @class_cache(cache_name='_STRIP_CACHE', key=lambda axis, layer, n: (axis, layer, n))
    @classmethod
    def strip_coords_from_axis(cls, axis: int, layer: int, n: int) -> list:
        """
          返回某 axis, face, layer 对应的条带坐标列表，已按中心原点计算
          返回: [(face, r, c), ...]  按 strip 顺序排列
          layer 在“世界坐标系”里定义
          row / col 在“face 局部坐标系”里定义
          strip 级别旋转 ≠ piece 级别置换
          几何排序 ≠ 拓扑顺序
        """
        strips = []
        axis_vec = cls.AXIS_VEC[axis].astype(float)
        face_stickers = cls.get_layer_stickers(axis, layer, n)
        for face, coords in face_stickers.items():
            if not coords:  # face_stickers.get(face, [])
                continue

            fidx = cls.face_idx[face]
            normal, u_dir, v_dir = cls.face_basis(face)
            # if np.dot(normal, axis_vec) < 0:  # 负侧面
            #     v_dir = -v_dir  # 只翻转向右方向
            strip_dir = np.cross(axis_vec, normal)
            strip_dir /= np.linalg.norm(strip_dir)  # 该面对应的旋转条带方向或法向量
            # 确定沿哪个方向 (v 或 u)，并计算 align
            align_u = np.dot(strip_dir, u_dir)
            align_v = np.dot(strip_dir, v_dir)
            if abs(align_u) > abs(align_v):
                key_dir = u_dir  # key = lambda x: x[0]  # r
                reverse = align_u < 0  # 如果 align <0,reverse 排序，使顺序沿局部正方向
            else:
                key_dir = v_dir  # key = lambda x: x[1]  # c
                reverse = align_v < 0
            # print('axis', axis, 'face', face, 'u', align_u, 'v', align_v, reverse)
            # coords_sorted = sorted(coords, key=lambda x: np.dot(x[2], strip_dir))
            coords_sorted = sorted(coords, key=lambda x: np.dot(x[2], key_dir), reverse=reverse)  # 世界坐标投影排序
            strip = [(fidx, r, c) for r, c, _ in coords_sorted]
            strips.append(strip)

        return strips

    @classmethod
    @class_status('参考方法')
    def strip_coords_from_axis_old(cls, axis: int, layer: int, n: int) -> list:
        face_normal = cls.face_normal()
        axis_vec = cls.AXIS_VEC[axis]
        strips = []
        # prev_strip_dir = None  # 用于检查连续性
        # prev_v_dir=None
        for idx, face in enumerate(cls.AXIS_STRIP[axis]):
            fidx = cls.face_idx[face]  # self.FACES.index(face)
            normal = face_normal[face]
            # 相邻面之间“向右”和“向上”的递进方向不连续,选择一致的参考系
            reverse = (axis == 2 and idx % 2 == 1)  # 第1、3个面（R 和 L 对于 Z 轴）,每隔一个面（奇数位）需要翻转 u_dir，使“向右”方向在环路上连续
            # 收集该 face 上属于 layer 的所有贴纸
            coords = []
            for r in range(n):
                for c in range(n):
                    xyz = cls.face_rc_to_xyz(face, r, c, n)
                    if np.isclose(xyz[axis], layer):
                        if reverse:
                            coords.append((c, r, xyz))  # Z₂ flip 对整个面内平面做一次 180° 旋转
                        else:
                            coords.append((r, c, xyz))

            if not coords:
                continue

            # center = np.mean([p for *_, p in coords], axis=0)
            # proj = center - dot(center, axis_vec) * axis_vec  # 投影到旋转平面（垂直于 axis）
            # 条带内部顺序,判断是水平行还是垂直列
            rs = {s[0] for s in coords}
            cs = {s[1] for s in coords}
            if len(rs) == 1:  # 水平行,按列排
                r = rs.pop()
                strip = [(fidx, r, c) for _, c, _ in sorted(coords, key=lambda x: x[1])]
            elif len(cs) == 1:  # 竖直列,按行排
                c = cs.pop()
                strip = [(fidx, r, c) for r, _, _ in sorted(coords, key=lambda x: x[0])]
            else:
                # 整个 face（最外层），按行排
                for r in sorted(rs):
                    strip = [(fidx, r, c) for rr, c, _ in coords if rr == r]
                    strips.append(strip)
                continue

            strip_dir = np.cross(axis_vec, normal)  # 该面对应的旋转条带方向或法向量
            # 检查与前一面的连续性: 如果 prev_strip_dir 存在, dot(strip_dir, prev_u_or_v) <0 则翻转
            # if prev_strip_dir is not None:
            #     if np.dot(strip_dir, prev_v_dir) < 0:  # 如果与前面的 v_dir 反向, 翻转本面基
            #         u_dir, v_dir = -u_dir, -v_dir  # 180° 旋转, 保持正交
            #         strip_dir = -strip_dir  # 相应翻转 strip_dir 以匹配
            #
            # # 更新 prev for next
            # prev_strip_dir = strip_dir
            # prev_v_dir = v_dir  # 或 u_dir, 取决于环是否沿col 方向绕,由于约定 v_dir=向右 (col), 用 v_dir 作为“前进”代理
            # dir_idx = np.argmax(np.abs(strip_dir))
            inc_dir = coords[-1][2] - coords[0][2]  # last - first
            if np.dot(inc_dir, strip_dir) < 0:  # 世界坐标方向向量，用整个条带判断是否需要反转
                strip.reverse()  # SO(3) 群, 逆向轴方向进场，需要 reverse
            strips.append(strip)
        return strips

    @classmethod
    def rotate_slice(cls, state: np.ndarray, axis: int, layer: int, shift: int, n: int = None):
        """旋转一层, inplace"""
        if shift == 0:
            return
        n = n or state.shape[1]
        # 生成每一条 strip 的坐标列表 (f, r, c)
        strips = cls.strip_coords_from_axis(axis, layer, n)  # 获取每一面条带的坐标序列
        # 读取每条 strip 的值（list of lists）,by strip_coords
        vals = [[state[f, r, c] for (f, r, c) in strip] for strip in strips]
        # 循环环移,正向移位
        vals = vals[-shift:] + vals[:-shift]  # CCW rotation
        # 写回
        for strip, val in zip(strips, vals):
            for (f, r, c), v in zip(strip, val):
                state[f, r, c] = v

    @classmethod
    def rotate_core(cls, state: np.ndarray, axis: int, layer: int, direction: int):
        """
        inplace 版本
        生成元 move(axis, layer, dir), SE(3) 中旋转生成元在立方晶格上的离散表示
          state ∈ Sticker(SO(3))
          axis ∈ {0,1,2} 离散化的旋转轴方向（单位向量）: 'x', 'y', 'z'
          layer ∈ {..., -2, -1, 0, 1, 2, ...} 沿旋转轴法向方向的离散标量坐标
          dir ∈ {+1, -1, 2}  θ ∈ {π/2, -π/2, π} 离散化的旋转角 / 旋量大小
        贴纸态 ≠ CubieState（不可逆）
        """
        d = direction % 4
        if d == 0:
            return  # 旋转 0 次
        n = state.shape[1]
        mid = n // 2

        # 处理最外层面本体旋转，x轴 → R/L, y轴 → U/D ,z轴 → F/B
        if abs(layer) == mid:
            side = 0 if layer > 0 else 1  # layer == +mid → 使用几何正向面,[0]（pos face）
            dd = -d if side == 0 else d  # 方向修正：视角翻转补偿
            face = cls.AXIS_FACE[axis][side]
            fidx = cls.face_idx[face]
            cls.rotate_inplace(state[fidx], dd)  # 使用的是「观察者正对该面」的顺时针定义
        # 中层处理
        cls.rotate_slice(state, axis, layer, shift=d, n=n)

    @classmethod
    def rotate_state(cls, state: np.ndarray, axis: int, layer: int, direction: int) -> np.ndarray:
        """
        纯函数版本：不修改传入 state，返回新状态 next_state 副本（已经应用旋转）。rotate_state 只存在于贴纸层
        用于 BFS/IDA*/并行扩展时的安全调用。区别实例方法“就地旋转”，完全独立
        """
        arr = state.copy()
        cls.rotate_core(arr, axis, layer, direction)
        return arr  # new_state

    @classmethod
    def get_corner_faces(cls, pos_sign: tuple[int, int, int]) -> tuple[str, str, str]:
        '''
        dict(enumerate(cls.AXIS_FACE)) 轴到面映射: {0: ('R','L'), 1: ('U','D'), 2: ('F','B')}
        标准顺序，确保 solved 时 U/D 在位置 0,第1位永远是 U 或 D（由 Y)
        U面：从 +Y 看（向上），D面：从 -Y 看（从下方向上看）
        先Y (U/D)，然后X Z (右手: 对于U, Z→X顺时针)
        '''
        sx, sy, sz = pos_sign  # (sx, sy, sz) ∈ {±1, ±1, ±1}
        # 索引：正方向 -> 0, 负方向 -> n-1,每个轴取正方向的面（+1 取正，-1 取反）
        idx = lambda s: 0 if s > 0 else - 1
        face_x = cls.AXIS_FACE[0][idx(sx)]  # R/L
        face_y = cls.AXIS_FACE[1][idx(sy)]  # U/D
        face_z = cls.AXIS_FACE[2][idx(sz)]  # F/B
        # 三个方向向量（单位向量）
        vec_x = np.array([sx, 0, 0])
        vec_z = np.array([0, 0, sz])
        outward_normal = np.array([0, -sy, 0])  # -vec_y
        # 计算叉乘：outward_normal × vec_x 应该指向 vec_z 的方向（右手规则）
        cross = np.cross(outward_normal, vec_x)
        if np.dot(cross, vec_z) > 0:  # 顺序正确：vec_x  →  vec_z 是顺时针
            return face_y, face_x, face_z  # 102
        return face_y, face_z, face_x  # 120

    @class_property('CORNER_FACE_CYCLE')
    def corner_face_cycle(cls) -> list[tuple]:
        '''
         标准顺序，确保 solved 时 U/D 在位置 0,第1位永远是 U 或 D（由 Y
         生成角块, 对应：UFR, URB, UBL, ULF, DLF, DFR, DRB, DBL
         ('U', 'R', 'F'),0: URF
         ('U', 'F', 'L'),1: UFL
         ('U', 'L', 'B'),2: ULB
         ('U', 'B', 'R'),3: UBR
         ('D', 'F', 'R'),4: DFR
         ('D', 'L', 'F'),5: DLF
         ('D', 'B', 'L'),6: DBL
         ('D', 'R', 'B'),7: DRB
         '''
        return [cls.get_corner_faces(pos_sign) for pos_sign in cls.CORNER_POS_SIGNS]

    @classmethod
    def get_edge_faces(cls, pos_sign: tuple[int, int, int]) -> tuple[str, str]:
        """
        针对边块，只涉及两个面: 非零坐标的两个轴对应的面。
        对于一个边块的位置符号 (sx, sy, sz)，其中一个坐标为0，另两个为 ±1
        顺序：先 Y 轴（U/D 优先），然后按右手规则排序剩余两个。
        """
        sx, sy, sz = pos_sign
        assert sx * sy * sz == 0  # 必须有一个为0
        assert len([i for i, s in enumerate([sx, sy, sz]) if s != 0]) == 2, "Edge must have exactly two non-zero coords"
        idx = lambda s: 0 if s > 0 else - 1
        face_x = cls.AXIS_FACE[0][idx(sx)]  # R if +x else L
        face_y = cls.AXIS_FACE[1][idx(sy)]  # U if +y else D
        face_z = cls.AXIS_FACE[2][idx(sz)]  # F if +z else B
        # 优先 U/D → F/B → R/L（与社区最常见命名顺序一致）
        if sy != 0:  # U/D 边的四个
            primary = face_y  # U or D
            if sz != 0:  # UF / UB / DF / DB
                secondary = face_z
            else:  # UR / UL / DR / DL
                secondary = face_x
        else:  # 中层边：FR, FL, BL, BR
            primary = face_z  # F or B
            secondary = face_x  # R or L

        return primary, secondary

    @class_property('EDGE_FACE_CYCLE')
    def edge_face_cycle(cls) -> list[tuple]:
        '''
         标准顺序，顺序：先 Y 轴（U/D 优先），然后按右手规则排序剩余两个。
         参考色放第一位（e.g., [U, R], [F, R]）
         生成角块, 对应：
        EDGE_INDEX_ORDER = [
        ('U', 'R'), ('U', 'F'), ('U', 'L'), ('U', 'B'),
        ('F', 'R'), ('F', 'L'), ('B', 'L'), ('B', 'R'),
        ('D', 'R'), ('D', 'F'), ('D', 'L'), ('D', 'B'),
        ]  # edge cubie 的“世界坐标顺序”
        '''
        return [cls.get_edge_faces(pos_sign) for pos_sign in cls.EDGE_POS_SIGNS]

    @class_cache(cache_name='_FACE_CACHE', key=lambda n: n)
    @classmethod
    def get_face_stickers(cls, n: int):
        """sticker_pos_from_face,不依赖 axis / layer / strip，返回贴纸中心的 3D 世界坐标"""
        stickers = defaultdict(list)
        for face in cls.FACES:
            normal, u_dir, v_dir = cls.face_basis(face)
            # origin = normal * (n / 2)
            for r in range(n):
                for c in range(n):
                    pos = cls.sticker_pos(normal, u_dir, v_dir, r, c, n)
                    # p = origin + dx * (c - n / 2 + 0.5) + dy * (r - n / 2 + 0.5)
                    stickers[face].append((r, c, pos))  # 抽象网格顺序
        return stickers

    @classmethod
    def get_corner_stickers(cls, n: int) -> list:
        """
        不考虑面环路连续性,几何点本身不应该知道邻接
        """
        stickers = []
        for face in cls.FACES:
            normal, u_dir, v_dir = cls.face_basis(face)
            for r in (0, n - 1):
                for c in (0, n - 1):
                    pos = cls.sticker_pos(normal, u_dir, v_dir, r, c, n)
                    stickers.append((face, r, c, pos))
        return stickers

    @classmethod
    def get_edge_stickers(cls, n: int) -> list:
        """
        返回所有 edge 贴纸:
        [(face, r, c, pos), ...]
        不考虑邻接 / 环路 / strip
        """
        stickers = []
        mid, c = divmod(n, 2)
        center = mid if c == 1 else mid - 1
        # 奇数阶选 layer=0（中心中线）,偶数阶选（偏左/下的中线）
        for face in cls.FACES:
            normal, u_dir, v_dir = cls.face_basis(face)
            # 四条边，去掉角
            for r, c in (
                    (0, center),  # 上边
                    (center, n - 1),  # 右边
                    (n - 1, center),  # 下边
                    (center, 0),  # 左边
            ):
                pos = cls.sticker_pos(normal, u_dir, v_dir, r, c, n)
                stickers.append((face, r, c, pos))
        return stickers

    @class_cache(cache_name='_CENTER_RINGS_CACHE', key=lambda n: n)
    @classmethod
    def get_center_rings(cls, n: int) -> list[list[list[tuple]]]:
        """
        返回每个 face 的所有中心贴纸坐标（带 pos）
        返回结构: list[face_idx] -> list[rings] -> list[(r, c, pos)]
        - face_idx 0~5 对应 cls.FACES 顺序
        - 最内层 ring 通常是中心单块（n奇数时只有一个元素）
        """
        rings = []
        mid = n // 2
        max_dist = mid - 1
        for face in cls.FACES:
            normal, u_dir, v_dir = cls.face_basis(face)
            face_rings = [[] for _ in range(max_dist + 1)]  # 每个距离一个 ring
            # 遍历内层区域
            for r in range(1, n - 1):
                for c in range(1, n - 1):
                    dist = max(abs(r - mid), abs(c - mid))  # 计算到中心的曼哈顿距离
                    pos = cls.sticker_pos(normal, u_dir, v_dir, r, c, n)
                    face_rings[dist].append((r, c, pos))
            # for ring in face_rings:
            #     ring.sort(key=lambda x: (x[0], x[1]))# 按 row, col 排序
            rings.append(face_rings)
        return rings

    @class_cache(cache_name='_CENTER_ORBITS_CACHE', key=lambda n: n)
    @classmethod
    def center_orbits(cls, n: int) -> list[list[int]]:
        """
        返回 center orbits（按 ring / face 划分）
        每个 orbit 是 center index 的列表
        """
        centers = []
        for face in range(6):
            for r in range(1, n - 1):
                for c in range(1, n - 1):
                    centers.append((face, r, c))

        index_map = {p: i for i, p in enumerate(centers)}

        face_rings = cls.get_center_rings(n)
        orbits = []
        for fidx, rings in enumerate(face_rings):
            for ring in rings:
                orbits.append([index_map[(fidx, r, c)] for r, c, _ in ring])
        total_centers = sum(len(o) for o in orbits)
        assert total_centers == (n - 2) ** 2 * 6, f"Center count mismatch: {total_centers}"
        return orbits

    @class_cache(cache_name='EDGES_CACHE', key=lambda n: n)
    @classmethod
    def edge_coords(cls, n: int) -> list[list[tuple[int, int, int]]]:
        """
        返回固定顺序的 12 条边的贴纸坐标，按标准顺序：
        [[(face_idx, r, c), (face_idx, r, c)],
          ...
        ]
        生成魔方所有 central edges 的贴纸坐标,12组，每条 edge 两个贴纸
        奇：基于 3D 世界坐标 + EDGE_POS_SIGNS 的 edge 定义
        偶：顺序严格对应 EDGE_FACE_CYCLE 的标准顺序（UR, UF, ..., DB）
        """
        stickers = cls.get_edge_stickers(n)
        assert len(stickers) == 24, f"Expected 24 edge pieces, got {len(stickers)}"
        result = [[] for _ in range(len(cls.EDGE_POS_SIGNS))]  # list[12][2]
        if n % 2 == 1:
            edges = {k: [] for k in cls.EDGE_POS_SIGNS}  # 12 条 edge
            for face, r, c, pos in stickers:
                sign = np.sign(pos).astype(int)
                sign = tuple(sign.tolist())
                if sign in edges:
                    edges[sign].append((face, r, c, pos))

            for eid, sign in enumerate(cls.EDGE_POS_SIGNS):
                group = edges[sign]
                if len(group) != 2:
                    raise ValueError(f"illegal edge {sign}: {len(group)}")

                center = np.mean([p for *_, p in group], axis=0)  # 几何一致性校验
                assert tuple(np.sign(center)) == sign, f"Sign mismatch for edge {eid}: expected {sign}"
                # 排序：哪个贴纸法向量更“外”，就排前
                group.sort(key=lambda x: np.argmax(np.abs(x[3])))
                result[eid] = [(cls.face_idx[f], r, c) for f, r, c, _ in group]
        else:
            pos_groups = defaultdict(list)  # 按位置坐标（四舍五入）分组
            for face, r, c, pos in stickers:
                key = tuple(np.round(pos, decimals=0).astype(int))  # key 是纯坐标，不带 sign
                pos_groups[key].append((face, r, c, pos))
                # for s in cls.EDGE_POS_SIGNS:
                #     if np.dot(s, np.sign(pos)) == 2:
                #         break

            # 筛选出正好有两个贴纸的位置组（这就是中棱）
            edge_groups = [g for g in pos_groups.values() if len(g) == 2]
            if len(edge_groups) != 12:
                raise ValueError(f"Expected 12 edge groups, got {len(edge_groups)}")

            standard_edges = cls.edge_face_cycle()  # list[tuple[str,str]] 长度12，如 [("U","R"), ("U","F"), ...]
            # 建立 边索引 ← 面对集合 的快速查找,预先生成标准顺序映射
            face_pair_to_id = {frozenset(pair): idx for idx, pair in enumerate(standard_edges)}
            for group in edge_groups:
                if len(group) != 2:  # 是长度为 2 的列表，每项是 (face_name, r, c, pos_vec)
                    raise ValueError(f"Edge piece should have exactly 2 stickers, got {group}")
                faces = frozenset(g[0] for g in group)
                eid = face_pair_to_id.get(faces)
                if eid is None:
                    raise ValueError(f"Unknown edge face pair: {faces}")
                group.sort(key=lambda x: np.argmax(np.abs(x[3])))  # 根据 pos 排序：哪个轴的绝对值最大，就认为它更“外侧”
                std_f1 = standard_edges[eid][0]
                if group[0][0] != std_f1:  # 根据标准顺序决定贴纸存储顺序
                    print(group)
                    group.reverse()

                result[eid] = [(cls.face_idx[f], r, c) for f, r, c, _ in group]

        assert len(result) == 12, f"Expected 12 edges, got {len(result)}:{result}"
        return result  # List[12] of [[(fidx,r,c), (fidx,r,c)]]

    @class_cache(cache_name='CORNERS_CACHE', key=lambda n: n)
    @classmethod
    def corner_coords(cls, n: int) -> list[list]:
        """
        [[(face_idx, r, c), (face_idx, r, c), (face_idx, r, c)],
          ...
        ]
        xyz pos(center)-> corner_id
        8 个角法向量组合: 每个角由三个轴的正负组成,3轴各2方向
        signs = [(sx, sy, sz)
                 for sx in (+1, -1)
                 for sy in (+1, -1)
                 for sz in (+1, -1)]  # list(product([1, -1], repeat=3))
        """
        stickers = cls.get_corner_stickers(n)
        corners = {k: [] for k in cls.CORNER_POS_SIGNS}  # 8 个角

        for face, r, c, pos in stickers:
            sx = 1 if pos[0] > 0 else -1
            sy = 1 if pos[1] > 0 else -1
            sz = 1 if pos[2] > 0 else -1
            corners[(sx, sy, sz)].append((face, r, c, pos))

        result = [[] for _ in range(len(cls.CORNER_POS_SIGNS))]
        for cid, sign in enumerate(cls.CORNER_POS_SIGNS):
            group = corners[sign]
            if len(group) != 3:
                raise ValueError(f"illegal corner {sign}: {len(group)}")
            center = np.mean([p for *_, p in group], axis=0)
            assert tuple(np.sign(center)) == sign, f"Sign mismatch for corner {cid}: expected {sign}"
            # “已正确分组后”再做主轴排序
            group.sort(key=lambda x: np.argmax(np.abs(x[3])))
            result[cid] = [(cls.face_idx[f], r, c) for f, r, c, _ in group]

        assert len(result) == 8, f"Expected 8 corners, got {len(result)}"
        return result

    @staticmethod
    def rotate_inplace(mat: np.ndarray, direction: int = 1) -> None:
        """
        rotate square matrix mat by direction*90 degrees clockwise.
        dir_sign: integer (positive/negative allowed). direction % 4 gives action:
          0 -> no-op
          1 -> 90 deg CW
          2 -> 180 deg
          3 -> 270 deg CW (or 90 CCW)
        The function mutates mat and returns None.
        和 rotate_coord 的旋转方向是完全一致的，都是顺时针（CW）。
        """
        dir_sign = direction % 4
        if dir_sign == 0:
            return
        elif dir_sign == 1:
            mat[:] = np.flip(mat.T, axis=1)  # 90 CW : transpose + flip LR
        elif dir_sign == 2:
            mat[:] = np.flip(np.flip(mat, axis=0), axis=1)  # 180 : flip LR + flip UD
        elif dir_sign == 3:  # i.e. 270 CW = 90 CCW
            mat[:] = np.flip(mat.T, axis=0)  # 90 CCW : transpose + flip UD
        else:
            raise ValueError(f"Invalid direction:{direction}")

    @staticmethod
    def rotate_coord(coord, axis: int, dir_sign: int = 1):
        """
        直接对一个3D坐标点[x, y, z]应用90度旋转，返回新坐标
        支持 cw (1) 和 ccw (-1) 的坐标旋转 right-hand
        dir_sign = 1 表示CW，dir_sign = -1 表示CCW
        """
        x, y, z = coord
        if axis == 0:  # X (R/L)  around x
            if dir_sign == 1:  # cw
                return [x, z, -y]
            else:  # ccw
                return [x, -z, y]
        elif axis == 1:  # Y (U/D)
            if dir_sign == 1:
                return [-z, y, x]
            else:
                return [z, y, -x]
        elif axis == 2:  # Z (F/B)
            if dir_sign == 1:
                return [y, -x, z]
            else:
                return [-y, x, z]
        raise ValueError("Invalid axis")

    @staticmethod
    def rot90_matrix(axis: int, dir: int) -> np.ndarray:
        """
        生成一个3x3的旋转矩阵,dir = +1 表示逆时针（CCW，从轴正方向看），dir = -1 表示顺时针（CW）
        axis: 0=x, 1=y, 2=z
        k: +1 = CCW, -1 = CW （从轴正方向看 +axis）
        """
        assert dir in (+1, -1)
        if axis == 0:  # X
            return np.array([
                [1, 0, 0],
                [0, 0, -dir],
                [0, dir, 0],
            ])
        if axis == 1:  # Y
            return np.array([
                [0, 0, dir],
                [0, 1, 0],
                [-dir, 0, 0],
            ])
        if axis == 2:  # Z
            return np.array([
                [0, -dir, 0],
                [dir, 0, 0],
                [0, 0, 1],
            ])
        raise ValueError(axis)

    @staticmethod
    def rotation_matrix(angle: tuple | np.ndarray) -> np.ndarray:
        # --- 合成基本旋转矩阵,整体旋转或欧拉角旋转 ---
        ax, ay, az = angle
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(ax), -np.sin(ax)],
            [0, np.sin(ax), np.cos(ax)]
        ])
        Ry = np.array([
            [np.cos(ay), 0, np.sin(ay)],
            [0, 1, 0],
            [-np.sin(ay), 0, np.cos(ay)]
        ])
        Rz = np.array([
            [np.cos(az), -np.sin(az), 0],
            [np.sin(az), np.cos(az), 0],
            [0, 0, 1]
        ])
        if np.abs(np.abs(ay) - np.pi / 2) < 1e-6:
            print(f"接近万向节锁! ay={ay} 接近 ±90°")
        return Rz @ Ry @ Rx  # 从右向左执行，实际顺序是 X -> Y -> Z

    @class_cache(key=lambda face, n: (face, n))
    @classmethod
    def face_quads(cls, face: str, n: int) -> list:
        """
        生成某一面上的所有小方块的四边形坐标->get_face_stickers
        返回给定面 U/D/F/B/L/R 上 n×n 个小方块的 3D quad 数组
        """
        normal, dx, dy = cls.face_def[face]
        origin = normal * (n / 2)

        result = []
        for i in range(n):
            for j in range(n):
                # 当前小贴纸左上角中心点
                p = origin + dx * (j - n / 2 + 0.5) + dy * (i - n / 2 + 0.5)
                # 小方块 4 个角
                quad = [
                    p + (-dx - dy) * 0.5,
                    p + (dx - dy) * 0.5,
                    p + (dx + dy) * 0.5,
                    p + (-dx + dy) * 0.5,
                ]
                result.append(quad)
        return result

    @classmethod
    def rotate_around_layer(cls, quad: np.ndarray, axis: int, layer_geom: float, ang: float) -> np.ndarray:
        """
        根据给定的旋转轴和角度生成旋转矩阵,任意层的局部轴
        计算旋转层的中心点（该层相对于立方体中心的位置）
        对该层的每个点进行旋转，保证旋转发生在该层平面上
        R = I + (sinθ) * K + (1 - cosθ) * K^2 罗德里格斯公式推导
        """

        # rotate points around the plane of the layer (centered at layer plane)
        # compute layer plane center
        def axis_rot_matrix(axis_vec: np.ndarray, theta: float):
            # Rodrigues' rotation formula
            k = axis_vec / np.linalg.norm(axis_vec)
            K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            I = np.eye(3)
            return I + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)

        rot = axis_rot_matrix(cls.AXIS_VEC[axis], ang)
        center = np.zeros(3)
        center[axis] = layer_geom  # translation to layer center:layer - n / 2 + 0.5
        return np.array([rot @ (v - center) + center for v in quad])

    @class_cache(key=lambda axis, layer, n: (axis, layer, n))
    @classmethod
    def layer_sticker_set(cls, axis: int, layer: int, n: int):
        stickers = cls.get_layer_stickers(axis, layer, n)
        return {
            (f, r, c)
            for f, lst in stickers.items()
            for r, c, _ in lst
        }

    @classmethod
    def should_rotate_by_sticker(cls, face: str, r: int, c: int, axis: int, layer: int, n: int):
        return (face, r, c) in cls.layer_sticker_set(axis, layer, n)

    @classmethod
    @class_status('待完成')
    def rotated_coord(cls, quad: np.ndarray, axis: int, layer: int, n: int, ang: float):
        """
        返回动画阶段的临时颜色
        quad: 4 个角的世界坐标 4x3 的 3D 点 → 只是渲染,贴纸的可视外壳
        axis/layer/ang: 当前旋转信息
        """
        # 找到该 quad 对应的逻辑位置
        # coords = quad[:, axis]
        # min_c, max_c = coords.min(), coords.max()
        stickers = cls.get_layer_stickers(axis, layer, n)  # dict[face] = [(r,c,pos), ...]
        for face, lst in stickers.items():
            for r, c, pos in lst:
                center = np.mean(quad, axis=0)
                # 如果该 quad 的中心接近任何一个贴纸的 pos，就认为它属于旋转层
                if np.allclose(center, pos, atol=0.5):  # 判断 quad 是否接近 pos
                    #  根据旋转角度计算新逻辑位置
                    layer_coord = pos[axis]
                    if abs(ang - np.pi / 2) < 1e-6:  # 顺时针 90°
                        new_r, new_c = c, n - 1 - r
                    elif abs(ang + np.pi / 2) < 1e-6:  # 逆时针 90°
                        new_r, new_c = n - 1 - c, r
                    elif abs(abs(ang) - np.pi) < 1e-6:  # 180°
                        new_r, new_c = n - 1 - r, n - 1 - c
                    else:  # 0°
                        new_r, new_c = r, c

                    return face, new_r, new_c
        return None


class StickerCube(CubeBase):
    COLORS = ['W', 'Y', 'R', 'O', 'G', 'B']  # Rubiks 0:白色, 1:黄色, 2:红色, 3:橙色, 4:绿色, 5:蓝色

    def __init__(self, state: np.ndarray | dict = None, n: int = 3):
        super().__init__(n)

        if state is None:  # 初始化已解决状态
            self.cube = self.solved.copy()
            # for face, color in zip(self.FACES, self.COLORS):
            #     self.cube[face] = [[color] * n for _ in range(n)]
        elif isinstance(state, np.ndarray):
            # state 应当是 (6,n,n) 的数值
            self.cube = state.astype(np.uint8)
            self.n = self.cube.shape[1]
        elif isinstance(state, dict):
            self.cube = self.from_color(state)
            self.n = self.cube.shape[1]
            # 假定传入的 state 是面->二维列表的映射，复制一份以免外部修改
            # self.cube = {f: [row.copy() for row in state[f]] for f in self.FACES}
            # self.n = len(self.cube)
        else:
            raise ValueError('wrong state')
        if self.n != n:
            super().__init__(n)

        # total_stickers = 6 * n ^ 2
        # per_face_outer = 4 * n - 4
        # per_face_inner = (n - 2) ^ 2
        # edge_wings = 12 * (n - 2)  # 边翼, 角块（corners）恒为 8

        # self.__class__.axis_face_idx = {axis: (self.FACES.index(pos), self.FACES.index(neg))
        #                                 for axis, (pos, neg) in enumerate(self.AXIS_FACE)}
        assert list(range(6)) == [self.FACES.index(f) for f in self.FACES]
        random.seed(47)

    def clone(self):
        """深拷贝当前魔方并返回新的实例"""
        return StickerCube(state=self.cube.copy(), n=self.n)

    def reset(self):
        self.cube = self.solved.copy()

    def get_state(self) -> np.ndarray:
        """返回当前魔方状态（用于序列化）"""
        return self.cube.copy()

    def is_solved(self, state: np.ndarray = None) -> bool:
        if state is None:
            state = self.cube
        return super().is_solved(state)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.n != other.n:
            return False
        return bool(np.array_equal(self.cube, other.cube))  # np.all(self.cube == other.cube)

    def __hash__(self):
        return hash(super().encode(self.cube))

    @property
    def flatten_key(self) -> tuple:
        """把 cube 转成 tuple"""
        return tuple(self.cube.flatten())

    @property
    def color(self) -> dict:
        return self.get_color(self.cube)

    @classmethod
    def get_color(cls, state: np.ndarray) -> dict:
        """返回原来使用的 face->二维字符串颜色矩阵"""
        n = state.shape[1]
        face_stickers = cls.get_face_stickers(n=n)  # 返回 [(r, c, pos), ...] 按渲染顺序
        idx_color = {i: c for i, c in enumerate(cls.COLORS)}
        result = {}
        for fidx, face in enumerate(cls.FACES):
            # col_mat = np.vectorize(idx_color.get)(state[fidx, :, :])
            # result[face] = col_mat.tolist()  # 转为普通列表[[str,str]]
            mat = np.empty((n, n), dtype='<U1')
            coords = face_stickers[face]
            for (r, c, _), color_idx in zip(coords, state[fidx].flatten()):
                mat[r, c] = idx_color[color_idx]
            result[face] = mat.tolist()

        return result

    @classmethod
    def from_color(cls, cube_color: dict) -> np.ndarray:
        # 传入的 state 是面->二维列表的映射
        color_index = {color: i for i, color in enumerate(cls.COLORS)}
        n = len(cube_color)
        arr = np.zeros((6, n, n), dtype=np.uint8)
        for fidx, face in enumerate(cls.FACES):
            face_mat = cube_color[face]
            arr[fidx, :, :] = np.vectorize(color_index.get)(face_mat)
        return arr

    def diff(self, other, max_show: int = 20):
        diffs = []
        for f in range(6):
            for r in range(self.n):
                for c in range(self.n):
                    a = int(self.cube[f][r][c])
                    b = int(other.cube[f][r][c])
                    if a != b:
                        diffs.append((f, r, c, a, b))
                        if len(diffs) >= max_show:
                            return diffs
        return diffs

    def central_edge_coords(self, face: str, base_fase: str = 'D') -> list:
        """
        返回 face 与 D 相邻的 central edge 两个 sticker 的位置坐标 (face1, r1, c1), (face2, r2, c2)
        约定 faces order 与 AXIS_STRIP/face_idx 一致。这里选定 D-face 边为目标边位置：
        返回格式：[(fidx1, r1, c1), (fidx2, r2, c2)]，顺序与 edge_face_cycle 一致
        """
        edge_face = {face, base_fase}  # f'{face}{base_fase}'
        eid = next(i for i, pair in enumerate(self.edge_face_cycle()) if set(pair) == edge_face)
        return self.edge_coords(self.n)[eid]

    @class_status('参考方法')
    def rotate_face(self, face: str, direction: int = 1):
        """旋转一个面，direction=1顺时针，-1逆时针,axis 与 face.normal 必然平行"""
        fidx = self.face_idx[face]
        axis, side = self.face_axis[face]
        axis_vec = self.AXIS_VEC[axis]
        normal = self.face_normal[face]
        sign = np.dot(normal, axis_vec)
        assert (side == 0 and sign > 0) or (side == 1 and sign < 0)
        layer = self.mid if side == 0 else -self.mid

        d = direction % 4
        dd = d if side == 0 else -d
        self.rotate_inplace(self.cube[fidx], dd)  # np.rot90(arr, -direction)
        self.rotate_slice(self.cube, axis, layer, shift=d, n=self.n)

    @chainable_method
    def rotate(self, axis: int, layer: int, direction: int = 1):
        """
        统一旋转入口,AXIS-SLICE 旋转
        axis: 'x' | 'y' | 'z':0,1,2
        layer: 0 ~ n-1
        direction: 1 = 顺时针, -1 = 逆时针
        """
        # print('rotate:', axis, layer, direction)
        assert 0 <= axis <= 2, f"unknown axis: {axis}"
        assert -self.mid <= layer <= self.mid, f"layer out of range: {layer}"
        self.rotate_core(self.cube, axis, layer, direction)

    def propose_move(self, layer_span: int = None) -> tuple[int, int, int]:
        """
        采样一个候选 move：
          - axis: X/Y/Z uniformly
          - layer: 优先在中间 layer 附近采样（layer_span 设置距离 mid 的范围），超出时均匀采样全部层
          - direction: 随机取 1 (CW), -1 (CCW) 或 2 (180) （180 的概率可降低）
        """
        if not layer_span:
            layer_span = self.mid
        axis = random.choice(range(3))
        # sample layer: with 80% prob sample near center, 20% anywhere
        low = max(-self.mid, - layer_span)
        high = min(self.mid, layer_span)
        layer = random.randint(low, high)  # 上界包含
        if layer == 0 and self.n % 2 == 0:
            layer = random.choice([-1, 1])
        # direction probabilities: prefer +/-1; occasional 2
        direction = random.choices([-1, 1, 2], weights=[0.48, 0.48, 0.04], k=1)[0]
        return axis, layer, direction

    @chainable_method
    def apply(self, moves: list | tuple):
        self.act_moves(self.cube, moves)

    def apply_move(self, move: str):
        primitive_moves = self.transform(move)
        self.apply(primitive_moves)
        return primitive_moves

    def heuristic_edge_mismatch(self, face: str, base: int, state: np.ndarray = None) -> int:
        """
        score: 0 if matched, higher if mismatch. 用于贪心最小化。
        判断指定 face 的 central edge 是否已在 D 层且两面颜色对齐：
        - D-side 的 sticker == base_color
        - side-face 的 sticker == side-face center color
        """
        if state is None:
            state = self.cube
        (f1, r1, c1), (f2, r2, c2) = self.central_edge_coords(face, 'D')
        s1 = int(state[f1, r1, c1])
        s2 = int(state[f2, r2, c2])  # D-side
        target_side = int(state[f1, self.mid, self.mid])
        score = 0
        if s2 != int(base):  # D 面 base_color 是否匹配
            score += 1
        if s1 != target_side:  # 侧面是否正确颜色
            score += 1
        if r2 != self.mid or c2 not in (self.mid - 1, self.mid, self.mid + 1):  # 如果边块根本不在 D 层：加罚分
            score += 1  # or 2
        return score

    def detect_oll_parity(self, state: np.ndarray = None):
        """
        检查是否存在 odd-flip 的情况（只可能在 NxN Even）。
        原理：检查边配对之后是否存在单边颜色方向异常。
        """
        if state is None:
            state = self.cube

        # 任意找一条 LL 边即可
        # 譬如 UF 的 central edge
        (f1, r1, c1), (f2, r2, c2) = self.central_edge_coords('F', 'D')  # (face, r, c)
        c1 = int(state[f1, r1, c1])
        c2 = int(state[f2, r2, c2])

        # 如果颜色组合不合法（不该出现），则 flip parity
        # 简化：检查两侧中心色是否与该 edge 组合矛盾
        side_center = int(state[f1, self.mid, self.mid])
        down_center = int(state[self.face_idx['D'], self.mid, self.mid])

        # Very loose but useful detection
        if {c1, c2} != {side_center, down_center}:
            return True

        return False

    def detect_pll_parity(self, state: np.ndarray = None):
        """
        在 PLL 阶段（3×3 模式）判断最后两条边是否单独 swap。
        """
        if state is None:
            state = self.cube
        # 基于 3×3 PLL 顶层边块颜色检查
        # 检查 U face 四条边是否出现奇偶交换
        edges = [
            ('U', self.mid, 0),  # UL
            ('U', self.mid, self.n - 1),  # UR
            ('U', 0, self.mid),  # UB
            ('U', self.n - 1, self.mid),  # UF
        ]

        values = [int(state[self.face_idx[f], r, c]) for f, r, c in edges]
        # 如果颜色排列不可能正解 = swap parity
        # 使用简单判断：出现两条 edge 对调
        if len(set(values)) < 4:  # 简化：冲突
            return True

        return False

    def ida_star(self, max_depth: int = 25):
        """
         小深度 IDDFS 搜索，用于“局部修正”中心,Edge 修正,Center 修正
         节点状态假设可 hash（这里直接用 cube.state()）
         3×3 魔方最大深度 20~24
        """
        visited = set()
        bound = self.heuristic(self.cube)  # heuristic_center
        while True:
            visited.clear()
            res, sol = self.dfs(self.cube, 0, bound, visited, [], max_depth)
            if res is True:
                return sol
            if res == math.inf:
                return None
            bound = res

    def greedy_fix_center(self, r: int = 1):
        """
        对中心区域做贪心修正。直到中心收敛
        贪心：尝试旋转一圈，找错误减少最大的 move
        """
        start_wrong = self.heuristic_center(self.cube, r)
        best = None
        best_delta = 0
        for move in self.basic_generators():
            next_state = self.rotate_state(self.cube, *move)
            h = self.heuristic_center(next_state, r)

            delta = start_wrong - h
            if delta > best_delta:
                best_delta = delta
                best = move

        return best, best_delta

    def solve_cross(self, max_iter: int = 100, max_depth: int = 2, shuffle_moves: bool = True) -> list:
        """
        底层十字,BFS ≤ 5
        1. 找到 D 面中心颜色
        2. 把含该颜色的边块移到底层
        3. 校准侧面颜色与正确面一致（方向正确）
        """
        base_color = int(self.cube[self.face_idx['D'], self.mid, self.mid])
        targets = ['F', 'R', 'B', 'L']

        candidate_single_moves = list(self.basic_generators())
        if shuffle_moves:
            random.shuffle(candidate_single_moves)

        # Node = namedtuple('Node', ['state', 'path'])
        moves = []  # applied
        for face in targets:
            if self.heuristic_edge_mismatch(face, base_color, self.cube) == 0:
                continue  # already matched

            # BFS on small depth (try_depth steps)
            start_state = self.get_state()
            queue = deque([(start_state, [])])
            visited = {self.encode(start_state)}  # start_key
            iter_count = 0
            found_seq = None  # greedy local search bounded by max_iter for this face

            while queue and iter_count < max_iter:
                cur_state, cur_path = queue.popleft()
                iter_count += 1

                cur_score = self.heuristic_edge_mismatch(face, base_color, cur_state)
                if cur_score == 0:
                    found_seq = cur_path
                    break

                # depth control: limit path length to try_depth
                if len(cur_path) >= max_depth:
                    continue

                # try single moves (or small template moves)
                for move in candidate_single_moves:
                    if self.is_inverse(cur_path, *move):
                        continue
                    nxt = self.rotate_state(cur_state, *move)
                    key = self.encode(nxt)
                    if key in visited:
                        continue
                    visited.add(key)
                    new_path = cur_path + [move, ]
                    queue.append((nxt, new_path))

            if found_seq is None:
                # fallback: greedy hill-climb (try single moves reducing score)
                cur_state = self.get_state()
                for _ in range(max_iter):
                    cur_score = self.heuristic_edge_mismatch(face, base_color, cur_state)
                    if cur_score == 0:
                        found_seq = []
                        break
                    best_move = None
                    best_score = cur_score
                    for move in candidate_single_moves:
                        nxt = self.rotate_state(cur_state, *move)
                        s = self.heuristic_edge_mismatch(face, base_color, nxt)
                        if s < best_score:
                            best_score = s
                            best_move = move
                    if best_move is None:
                        break
                    # apply move to cur_state and continue (note: not yet applied to self)
                    cur_state = self.rotate_state(cur_state, *best_move)
                    found_seq = found_seq + [best_move, ] if found_seq else [best_move]

            if found_seq:
                # apply the found_seq to actual cube (mutating self) and record moves
                for mv in found_seq:
                    axis, layer, direction = mv
                    self.rotate(axis, layer, direction)
                    moves.append(mv)
                # verify matched
                if self.heuristic_edge_mismatch(face, base_color, self.cube) > 0:
                    # try a few small local repairs (greedy single-step)
                    for _ in range(8):
                        cur_score = self.heuristic_edge_mismatch(face, base_color, self.cube)
                        if cur_score == 0:
                            break
                        # try a single best move in real cube
                        best_move = None
                        best_score = cur_score
                        for mv in candidate_single_moves:
                            nxt = self.rotate_state(self.cube, *mv)
                            s = self.heuristic_edge_mismatch(face, base_color, nxt)
                            if s < best_score:
                                best_score = s
                                best_move = mv
                        if best_move is None:
                            break
                        self.rotate(*best_move)
                        moves.append(best_move)
            else:
                # cannot find small seq; skip this face (user can increase try_depth / max_iter)
                # optionally log or raise
                # print(f"WARNING: cannot pair central edge for {face}")
                pass

        return moves

    def solve_centers(self, greedy_iter: int = 4, max_depth: int = 16):
        """解决中心块,先贪心,把中心粘起来，再 IDA* 补洞局部修正"""

        def greedy_pass():
            moves = []
            for _ in range(greedy_iter):
                mv, gain = self.greedy_fix_center()
                if mv is None or gain <= 0:
                    break
                self.apply(mv)
                moves.append(mv)
            return moves

        g_moves = greedy_pass()
        ida_moves = self.ida_star(max_depth=max_depth) or []
        self.apply(ida_moves)  # 执行 IDA* 结果
        return g_moves + ida_moves

    def solve_bfs(self, max_depth: int = 6) -> list | None:
        '''用 BFS 解决 局部,BFS 5~7'''
        start = self.get_state()
        queue = deque([(start, [], 0)])  # state, path, depth
        visited = {self.encode(start)}

        while queue:
            state, path, depth = queue.popleft()

            if self.is_solved(state):
                return list(path)

            if depth >= max_depth:
                continue

            for move in self.basic_generators():
                # forbid immediate reversal
                if self.is_inverse(path, *move):
                    continue

                next_state = self.rotate_state(state, *move)
                key = self.encode(next_state)

                if key in visited:
                    continue

                visited.add(key)
                queue.append((next_state, path + [move], depth + 1))

        return None

    def solve_edges(self):
        """ RL 解 cube
        解决边块,
        边块配对阶段：
        1. 扫描全部边位置，scan_edges group_edges_by_color,分类为 inner_edges / outer_edges
        2. 计算每种颜色的匹配关系 find_unpaired
        3. 通过特定算法（slice moves + pairing）完成配对
        4. 异常情况处理 (parity)
        """

    def fix_parity(self):
        """
          NxN Parity 修复：
          - OLL Parity: 单边翻转
          - PLL Parity: 最后二边互换
          每个 parity 修复都增加到 moves[]
        """
        moves = []
        # 1. OLL parity
        if self.detect_oll_parity():
            oll_seq = [
                "Rw", "U2", "Rw", "U2",
                "Rw", "U2", "Rw'", "U2",
                "Lw", "U2", "Rw", "U2",
                "Rw", "U2", "Rw'", "U2",
                "Rw'"
            ]
            for mv in oll_seq:
                moves += self.apply_move(mv)

        # 2. PLL parity
        if self.detect_pll_parity():
            pll_seq = [
                "2Rw2", "U2", "2Rw2", "U2",
                "Uw2", "2Rw2", "Uw2"
            ]
            for mv in pll_seq:
                moves + self.apply_move(mv)

        return moves

    def reduction(self):
        """
        解决中心块, 配对解决边块,三阶解法
        1. solve_centers()
        2. solve_edges()
        3. solve_3x3()  # 调用三阶魔方解法
        4. fix_parity() 处理奇偶错误
        """
        moves = []
        # mv_centers = self.solve_centers(
        #     greedy_iter=4,
        #     max_depth=9  # 4×4 用 12~14；6×6 用 16~20
        # )
        # moves += mv_centers
        # print('solve_centers:', mv_centers)

        mv_parity = self.fix_parity()
        moves += mv_parity
        print('fix_parity:', mv_parity)

        mv_cross = self.solve_cross(max_iter=200)
        moves += mv_cross
        print('solve_cross:', mv_cross)

        mv_bfs = self.solve_bfs(max_depth=6)  # F2L
        if mv_bfs:
            self.apply(mv_bfs)
            moves += mv_bfs
        print('solve_bfs:', mv_bfs)
        mv = self.ida_star(max_depth=14)
        if mv:
            self.apply(mv)
            moves += mv
        print('ida_star:', mv)
        return moves

    def solve(self):
        if self.is_solved(self.cube):
            return []
        return self.reduction()


if __name__ == "__main__":
    class_cache.load(StickerCube)
    class_property.load(StickerCube)

    cube = StickerCube(n=5)
    print(cube.face_def())
    print(cube.corner_face_cycle)
    print(cube.edge_face_cycle)
    print(cube.color)

    xx = cube.corner_coords(5)
    yy = cube.edge_coords(5)
    print('corner_coords', len(xx), xx)
    print('edge_coords', len(cube.edge_coords(5)), yy)
    print('map', cube.SOLVED_CORNERS_MAP, cube.SOLVED_EDGES_MAP)

    for layer in range(-cube.mid, cube.mid + 1):
        strip = cube.strip_coords_from_axis(2, layer, cube.n)
        colors = [[cube.cube[f, r, c] for f, r, c in s] for s in strip]
        print(colors)

    print('heuristic_corner_perm', cube.heuristic_corner_perm(cube.cube))

    print('encode_state', cube.encode_state(cube.cube))
    print('encode_state_idx', cube.encode_state(cube.solved_idx).astype(float) / (cube.n * cube.n))
    print('embedding', cube.embedding(cube.solved_idx))

    xxx = cube.get_layer_stickers(0, 1, 5)
    print(len(xxx), xxx)

    xx = cube.get_center_rings(3)
    print('center_rings', len(xx), [len(x) for x in xx], '\n', xx)
    xx = cube.center_orbits(4)
    print('center_orbits', len(xx), xx)

    backup = cube.get_state()  # copy.deepcopy(cube.cube)

    cube0 = cube.clone()
    for _ in range(4):
        cube.rotate(1, 0, 1)

    for axis in (0, 1, 2):
        for layer in range(-cube.mid, cube.mid + 1):
            cube0 = cube.clone()
            for _ in range(4):
                cube.rotate(axis, layer, direction=1)

            if not cube.is_solved():
                print(f'{axis},{layer} not solved')
            if cube != cube0:
                print("FAIL:", axis, layer, cube.diff(cube0, 10))

    assert np.all(cube.cube == backup)

    cube.reset()

    print(cube.encode_state(cube.cube))
    print('.................')


    def test_rotate():
        print("原始 cube:")
        faces = cube.FACES
        for i, f in enumerate(faces):
            print(f"{f}:\n{cube.cube[i]}\n")

        cube.rotate(axis=2, layer=0, direction=1)  # Z轴, 后面
        print("旋转 Z轴 layer0 后:")
        for i, f in enumerate(faces):
            print(f"{f}:\n{cube.cube[i]}\n")

        cube.rotate(axis=0, layer=4, direction=-1)  # X轴, R面
        print("旋转 X轴 layer4 后:")
        for i, f in enumerate(faces):
            print(f"{f}:\n{cube.cube[i]}\n")


    # test_rotate()

    def test_scramble(moves: int = 10):
        mv = cube.scramble(moves)
        cube.apply(mv)
        # print(cube.cube)
        print(mv)
        print('diff', cube.diff_coords(cube.cube))
        # print(cube.heuristic(cube.cube))
        inv_moves = cube.invert_moves(mv)
        cube.apply(inv_moves)
        # print(backup )
        return cube.is_solved()


    test_scramble()


    # err = 0
    # for i in range(1000):
    #     if not test_scramble(80):
    #         print(f"❌ failed at round {i}")
    #         err += 1
    # print(f"✔️ all good.err {err}")

    def test_scramble2(moves: int = 10):
        mv = cube.scramble(moves)
        s0 = cube.solved_idx.copy()
        cube.act_moves(s0, mv)
        if not np.array_equal(np.sort(s0.reshape(-1)), cube.solved_idx.reshape(-1)):
            print('s0', s0.reshape(-1))
            return False
        inv_moves = cube.invert_moves(mv)
        cube.act_moves(s0, inv_moves)
        if not np.array_equal(cube.idx_to_state(s0), cube.solved):
            return False
        return np.array_equal(s0, cube.solved_idx)


    test_scramble2()
    err = 0
    for i in range(1000):
        if not test_scramble2(80):
            print(f"❌ failed at round {i}")
            err += 1
    print(f"✔️ all good.err {err}")

    cube = StickerCube(n=3)
    print(cube.get_state())
    print('corners', cube.get_corners(cube.cube))
    mv = cube.scramble(20)
    cube.apply(mv)
    print(mv)

    # mvs0 = cube.solve()
    # print(mvs0)

    print(cube.is_solved())
    print(cube.color)

    AXIS_VEC = {
        0: (1, 0, 0),
        1: (0, 1, 0),
        2: (0, 0, 1),
    }


    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


    def cross(a, b):
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )


    # class_cache.save(StickerCube)
    # class_property.save(StickerCube)

    print(cube.get_vars())
    print(cube.strip_coords_from_axis.cache)
    from rime.base import check_class_status

    print(check_class_status(StickerCube))
