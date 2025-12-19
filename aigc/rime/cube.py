from rime.circular import chainable_method
from collections.abc import Mapping
import numpy as np
import random, math
from collections import deque, defaultdict


class ClassProperty:
    """
    带缓存的 class-level property
    第一次访问计算，之后缓存到类
    """

    def __init__(self, attr_name: str):
        self.attr_name = attr_name
        self.func = None

    def __call__(self, func):
        self.func = func
        return self

    def __get__(self, obj, cls):
        if not hasattr(cls, self.attr_name):
            setattr(cls, self.attr_name, self.func(cls))
        return getattr(cls, self.attr_name)


class IndexProxy(Mapping):
    """
    兼容：
      proxy[key]
      proxy.get(key)
      proxy()
    """

    def __init__(self, mapping: dict):
        self._map = mapping

    def __getitem__(self, key):
        if key not in self._map:
            raise KeyError(f"Invalid key: {key}")
        return self._map[key]

    def get(self, key, default=None):
        return self._map.get(key, default)

    def __call__(self):
        return self

    def __contains__(self, key):
        return key in self._map

    def __iter__(self):
        return iter(self._map)

    def __len__(self):
        return len(self._map)

    def __repr__(self):
        return f"{self.__class__.__name__}({self._map})"

    def items(self):
        return self._map.items()

    def keys(self):
        return self._map.keys()

    def values(self):
        return self._map.values()


class CubeBase:
    '''
    axis = 0 → x → R/L
    axis = 1 → y → U/D
    axis = 2 → z → F/B
    '''
    FACES = ['U', 'D', 'F', 'B', 'L', 'R']  # 面标识 上 下 前 后 左 右
    AXIS_NAME = ('X', 'Y', 'Z')
    # 哪些面在轴的正/负一侧_SIGN（约定：layer==0 对应正面）
    AXIS_FACE = [
        ('R', 'L'),  # X axis (0), positive,negative
        ('U', 'D'),  # Y axis (1),正面（layer==0），反面（layer==n-1）
        ('F', 'B'),  # Z
    ]  # YOLO 正轴方向,物理右手坐标一致
    AXIS_STRIP = (
        ['U', 'B', 'D', 'F'],  # 'X', 从 +X 看 CCW
        ['F', 'R', 'B', 'L'],  # 'Y'
        ['U', 'R', 'D', 'L'],  # 'Z'
    )  # CCW 视角,从 +axis 方向看过去,4 元环路,trip 顺序
    AXIS_SPIN = (
        +1,  # X
        +1,  # Y
        -1,  # Z
    )  # SO(3) 群, 逆向轴方向进场，需要 reverse
    # === 依赖 FACES 生成派生结构 ===
    AXIS_VEC = np.eye(3, dtype=int)
    FACE_UP = {
        'U': -AXIS_VEC[2],  # -Z: [0, 0, -1]
        'D': AXIS_VEC[2],  # Z:[0, 0, 1]

        'F': AXIS_VEC[1],  # Y:[0, 1, 0]
        'B': AXIS_VEC[1],

        'R': AXIS_VEC[1],
        'L': AXIS_VEC[1],
    }  # 固定局部坐标系

    @ClassProperty('FACE_NORMAL')
    def face_normal(cls):
        mapping = {}
        for axis, (pos_face, neg_face) in enumerate(cls.AXIS_FACE):
            mapping[pos_face] = cls.AXIS_VEC[axis]
            mapping[neg_face] = -cls.AXIS_VEC[axis]
        return IndexProxy(mapping)

    @ClassProperty('FACE_DEF')
    def face_def(cls):
        mapping = {}  # 六个面的法向和基向
        for face, normal in cls.face_normal.items():
            up = cls.FACE_UP[face]
            right = np.cross(up, normal)
            mapping[face] = (normal, right, up)

        return IndexProxy(mapping)

    @ClassProperty('FACE_AXIS')
    def face_axis(cls):
        # 哪个面属于哪个轴
        return IndexProxy({face: (axis, side)
                           for axis, pair in enumerate(cls.AXIS_FACE)
                           for side, face in enumerate(pair)})

    @ClassProperty('FACE_DEF_AXIS')
    def face_idx(cls):
        return IndexProxy({f: i for i, f in enumerate(cls.FACES)})

    @ClassProperty('CORNER_COORDS')
    def corner_coords(cls) -> list:
        """
        根据 FACE_NORMAL 自动生成 8 个角块坐标
        返回: list of 3 tuples (每个角对应的 3 个面 (face, row, col))
        """
        corners = []
        # 轴到面映射
        axis_to_faces = dict(enumerate(cls.AXIS_FACE))  # {0: ('R','L'), 1: ('U','D'), 2: ('F','B')}

        # 8 个角法向量组合: 每个角由三个轴的正负组成
        # 对应：UFR, URB, UBL, ULF, DLF, DFR, DRB, DBL
        signs = [(sy, sz, sx)
                 for sy in (+1, -1)
                 for sz in (+1, -1)
                 for sx in (+1, -1)]

        # 生成角块
        for sy, sz, sx in signs:
            face_y = axis_to_faces[1][sy > 0]  # U/D
            face_z = axis_to_faces[2][sz > 0]  # F/B
            face_x = axis_to_faces[0][sx > 0]  # R/L

            # 索引：正方向 -> 0 层, 负方向 -> n-1 层
            idx = lambda s: 0 if s > 0 else - 1
            # 每个面的坐标 (row, col)
            corners.append((
                (face_y, idx(sy), idx(sx)),  # U/D 面: row 随 Y, col 随 X
                (face_z, idx(sy), idx(sx)),  # F/B 面: row 随 Y, col 随 X
                (face_x, idx(sy), idx(sz)),  # R/L 面: row 随 Y, col 随 Z
            ))

        return corners

    @staticmethod
    def layer_index(layer: int, n: int) -> int:
        mid = n // 2
        layer_idx = layer + mid
        if n % 2 == 0 and layer >= mid:
            layer_idx -= 1
        return layer_idx

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

        normal = cls.face_normal[face]  # face 法向量
        up = cls.FACE_UP[face]  # np.array([0, 1, 0])

        # 面内基向量（右手系）
        right = np.cross(up, normal)
        up2 = np.cross(normal, right)

        # 世界坐标,normal 是面中心，dc*right + dr*up2 扩展到局部坐标
        pos = normal + dc * right + dr * up2

        return np.round(pos).astype(int)  # 离散化,保证严格整数

    @classmethod
    def face_basis(cls, face: str):
        """
        确定局部行列方向
        返回与 face_rc_to_xyz 完全一致的面内基
        row_dir, col_dir
        """
        normal = cls.face_normal[face].astype(float)
        up = cls.FACE_UP[face].astype(float)

        row_dir = np.cross(up, normal)
        row_dir /= np.linalg.norm(row_dir)
        col_dir = np.cross(normal, row_dir)
        col_dir /= np.linalg.norm(col_dir)

        return normal, row_dir, col_dir

    @staticmethod
    def sticker_pos(normal, row_dir, col_dir, r: int, c: int, n: int) -> np.ndarray:
        """
         返回贴纸中心在世界坐标系中的 (x, y, z), world 连续浮点坐标
         - 立方体中心在原点，坐标范围 [-center, center]
         - r, c: 从 0 到 n-1
         """
        center = (n - 1) / 2.0
        face_center = normal * center
        # 局部坐标映射到中心坐标 [-center, center]
        s_row = center - r  # right
        s_col = c - center  # up2
        pos_rel = col_dir * s_row + row_dir * s_col  # row_dir * s_row + col_dir * s_col
        return face_center + pos_rel

    @classmethod
    def get_layer_stickers(cls, axis: int, layer: int, n: int = 3):
        """
        Returns a list of (fidx, r, c, pos) for stickers in the given layer along the axis.
        Assumes center at origin, layers from -x to x where x = (n-1)/2.
        """
        face_stickers = defaultdict(list)
        axis_vec = cls.AXIS_VEC[axis]
        for face in cls.AXIS_STRIP[axis]:
            normal, row_dir, col_dir = cls.face_basis(face)
            # layer 轴 ⟂ face，才可能出 strip
            if abs(np.dot(normal, axis_vec)) > 1e-6:
                continue  # 整个面或不在该 layer
            for r in range(n):
                for c in range(n):
                    xyz = cls.sticker_pos(normal, row_dir, col_dir, r, c, n)
                    if np.isclose(xyz[axis], layer):  # abs(xyz[axis] - layer) < 1e-6:  # 中心坐标
                        face_stickers[face].append((r, c, xyz))

        return face_stickers

    # --- helper: 把 map entry 转成坐标列表 (face, r, c) ---
    @classmethod
    def coords_from_axis(cls, axis: int, layer: int, n: int) -> list:
        strips = []
        axis_vec = cls.AXIS_VEC[axis]

        for face, coords in cls.get_layer_stickers(axis, layer, n).items():
            if not coords:
                continue

            fidx = cls.face_idx[face]
            normal = cls.face_normal[face]
            strip_dir = np.cross(axis_vec, normal)  # 该面对应的旋转条带方向或法向量
            strip_dir *= cls.AXIS_SPIN[axis]
            inc_dir = coords[-1][2] - coords[0][2]  # last - first
            if np.dot(inc_dir, strip_dir) < 0:  # 世界坐标方向向量，用整个条带判断是否需要反转
                coords.reverse()
            strip = [(fidx, r, c) for r, c, _ in coords]
            strips.append(strip)

        return strips

    @classmethod
    def coords_from_axis_strip(cls, axis: int, layer: int, n: int) -> list:
        """
        返回某 axis, face, layer 对应的条带坐标列表，已按中心原点计算
        返回: [(face, r, c), ...]  按 strip 顺序排列
        layer 在“世界坐标系”里定义
        row / col 在“face 局部坐标系”里定义
        """

        face_normal = cls.face_normal()
        axis_vec = cls.AXIS_VEC[axis]

        strips = []
        for face in cls.AXIS_STRIP[axis]:
            fidx = cls.face_idx[face]  # self.FACES.index(face)
            normal = face_normal[face]
            # 收集该 face 上属于 layer 的所有贴纸
            coords = []
            for r in range(n):
                for c in range(n):
                    xyz = cls.face_rc_to_xyz(face, r, c, n)
                    if np.isclose(xyz[axis], layer):
                        coords.append((r, c, xyz))

            if not coords:
                continue

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
            # dir_idx = np.argmax(np.abs(strip_dir))
            strip_dir *= cls.AXIS_SPIN[axis]
            inc_dir = coords[-1][2] - coords[0][2]  # last - first
            if np.dot(inc_dir, strip_dir) < 0:
                strip.reverse()
            strips.append(strip)
        return strips

    @classmethod
    def central_edge_coords(cls, n: int, edge_face: str = None) -> dict | tuple:
        """
        生成魔方所有 central edges 的贴纸坐标
        返回 dict: edge_name -> ((face1, r1, c1), (face2, r2, c2))
        使用 AXIS_STRIP 和 sticker_pos 自动计算，不写死
        """
        mid = n // 2
        edges = {}

        # 每个轴对应的面条带
        for axis, faces in enumerate(cls.AXIS_STRIP):
            for i in range(len(faces)):
                face1 = faces[i]
                face2 = faces[(i + 1) % len(faces)]  # 相邻面
                # 计算两面 central edge 的 row/col
                # 简单约定：face1 的 central row/col 与 axis 正负方向匹配

                # 这里可以根据 face 方向调整 row/col，保持与 coords_from_axis 一致
                if axis == 0:  # X 轴
                    r1, c1 = mid, n - 1 if i == 0 else 0
                    r2, c2 = mid, mid
                elif axis == 1:  # Y 轴
                    r1, c1 = n - 1, mid
                    r2, c2 = mid, mid
                else:  # Z 轴
                    r1, c1 = n - 1, mid
                    r2, c2 = mid, mid

                edge_name = f"{face1}{face2}"
                if edge_face and edge_name == edge_face:
                    return (cls.face_idx[face1], r1, c1), (cls.face_idx[face2], r2, c2)

                edges[edge_name] = ((cls.face_idx[face1], r1, c1), (cls.face_idx[face2], r2, c2))

        return edges

    @staticmethod
    def rotate_inplace(mat: np.ndarray, direction: int = 1) -> None:
        """
        rotate square matrix mat by direction*90 degrees clockwise.
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
        elif d == 1:
            mat[:] = np.flip(mat.T, axis=1)  # 90 CW : transpose + flip LR
        elif d == 2:
            mat[:] = np.flip(np.flip(mat, axis=0), axis=1)  # 180 : flip LR + flip UD
        else:  # k == 3, i.e. 270 CW = 90 CCW
            mat[:] = np.flip(mat.T, axis=0)  # 90 CCW : transpose + flip UD

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
        return Rz @ Ry @ Rx

    @staticmethod
    def rotate_around_layer(quad: np.ndarray, axis: int, layer: int, ang: float) -> np.ndarray:
        """
        根据给定的旋转轴和角度生成旋转矩阵,任意层的局部轴
        计算旋转层的中心点（该层相对于立方体中心的位置）
        对该层的每个点进行旋转，保证旋转发生在该层平面上
        """

        # rotate points around the plane of the layer (centered at layer plane)
        # compute layer plane center
        def axis_rot_matrix(axis_vec: np.ndarray, theta: float):
            # Rodrigues' rotation formula
            k = axis_vec / np.linalg.norm(axis_vec)
            K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            I = np.eye(3)
            return I + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)

        # translation to layer center
        if axis == 0:
            center = np.array([layer, 0, 0])  # - n / 2 + 0.5
            rot = axis_rot_matrix(np.array([1, 0, 0]), ang)
        elif axis == 1:
            center = np.array([0, layer, 0])
            rot = axis_rot_matrix(np.array([0, 1, 0]), ang)
        else:
            center = np.array([0, 0, layer])
            rot = axis_rot_matrix(np.array([0, 0, 1]), ang)

        return np.array([rot @ (v - center) + center for v in quad])


class RubiksCube(CubeBase):
    COLORS = ['W', 'Y', 'R', 'O', 'G', 'B']  # 0:白色, 1:黄色, 2:红色, 3:橙色, 4:绿色, 5:蓝色

    def __init__(self, state: np.ndarray | dict = None, n: int = 3):
        self.n = n
        self.solved = np.zeros((6, n, n), dtype=np.uint8)
        for f in range(6):
            self.solved[f, :, :] = f

        if state is None:  # 初始化已解决状态
            self.cube = self.solved.copy()
            # for face, color in zip(self.FACES, self.COLORS):
            #     self.cube[face] = [[color] * n for _ in range(n)]
        elif isinstance(state, np.ndarray):
            # state 应当是 (6,n,n) 的数值
            self.cube = state.astype(np.uint8)
            self.n = self.cube.shape[1]
        elif isinstance(state, dict):
            self.cube = self.cube_state(state)
            self.n = self.cube.shape[1]
            # 假定传入的 state 是面->二维列表的映射，复制一份以免外部修改
            # self.cube = {f: [row.copy() for row in state[f]] for f in self.FACES}
            # self.n = len(self.cube)

        self.mid = self.n // 2
        # total_stickers = 6 * n ^ 2
        # per_face_outer = 4 * n - 4
        # per_face_inner = (n - 2) ^ 2
        # edge_wings = 12 * (n - 2)  # 边翼, 角块（corners）恒为 8

        # self.__class__.axis_face_idx = {axis: (self.FACES.index(pos), self.FACES.index(neg))
        #                                 for axis, (pos, neg) in enumerate(self.AXIS_FACE)}
        random.seed(123)

    def clone(self):
        """深拷贝当前魔方并返回新的实例"""
        return RubiksCube(state=self.cube.copy(), n=self.n)

    def reset(self):
        self.cube = self.solved.copy()

    def get_state(self) -> np.ndarray:
        """返回当前魔方状态（用于序列化）"""
        return self.cube.copy()

    def is_solved(self, state: np.ndarray = None) -> bool:
        if state is None:
            state = self.cube
        return bool(np.array_equal(state, self.solved))  # not (state ^ self.solved).any()

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.n != other.n:
            return False
        return bool(np.array_equal(self.cube, other.cube))

    def encode(self, state: np.ndarray = None) -> bytes:
        if state is None:
            state = self.cube
        return np.ascontiguousarray(state, dtype=np.uint8).tobytes()

    def __hash__(self):
        return hash(self.encode())

    @property
    def flatten_key(self) -> tuple:
        """把 cube 转成 tuple"""
        return tuple(self.cube.flatten())

    @property
    def color(self) -> dict:
        """返回原来使用的 face->二维字符串颜色矩阵"""
        result = {}
        idx_color = {i: c for i, c in enumerate(self.COLORS)}
        for face_idx, face in enumerate(self.FACES):
            mat = self.cube[face_idx, :, :]
            col_mat = np.vectorize(idx_color.get)(mat)
            result[face] = col_mat.tolist()  # 转为普通列表[[str,str]]
        return result

    @property
    def center_layers(self) -> list:
        if self.n % 2 == 1:
            return list(range(-self.mid, self.mid + 1))  # 奇数阶：中心在 0
        return [i for i in range(-self.mid, self.mid + 1) if i != 0]  # 偶数阶：无中心层

    @classmethod
    def cube_state(cls, cube_color: dict) -> np.ndarray:
        # 传入的 state 是面->二维列表的映射
        color_index = {color: i for i, color in enumerate(cls.COLORS)}
        n = len(cube_color)
        arr = np.zeros((6, n, n), dtype=np.uint8)
        for face_idx, face in enumerate(cls.FACES):
            face_mat = cube_color[face]
            arr[face_idx, :, :] = np.vectorize(color_index.get)(face_mat)
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

    def _central_edge_coords(self, edge_face: str) -> tuple:
        """
        返回 face 与 D 相邻的 central edge 两个 sticker 的位置坐标 (face1, r1, c1), (face2, r2, c2)
        约定 faces order 与 AXIS_STRIP/face_idx 一致。这里选定 D-face 边为目标边位置：
        例如 for 'F' -> (('F', n-1, mid), ('D', mid, n-1))
        注意：针对不同 face 要映射正确位置。
        """
        return self.central_edge_coords(self.n, edge_face)

    def rotate_slice(self, axis: int, layer: int, direction: int = 1):
        """旋转一层，layer: 层索引（0到n-1）， axis: 'x', 'y', 或 'z'"""
        shift = direction % 4
        if shift == 0:
            return
        strips = self.coords_from_axis(axis, layer, self.n)  # 获取每一面条带的坐标序列
        # 获取每一面条带的坐标序列
        vals = [[self.cube[f, r, c] for (f, r, c) in strip] for strip in strips]
        # 循环环移#* self.AXIS_SPIN[axis])
        vals = vals[-shift:] + vals[:-shift]  # CCW rotation
        for strip, v in zip(strips, vals):
            for (f, r, c), val in zip(strip, v):
                self.cube[f, r, c] = val

    def rotate_face(self, face: str, direction: int = 1):
        """旋转一个面，direction=1顺时针，-1逆时针"""
        fidx = self.face_idx[face]
        normal = self.face_normal[face]
        axis, side = self.face_axis[face]

        layer = self.mid if side == 0 else -self.mid
        axis_vec = self.AXIS_VEC[axis]

        d = direction % 4
        d *= self.AXIS_SPIN[axis]
        if np.dot(normal, axis_vec) < 0:
            d = -d
        self.rotate_inplace(self.cube[fidx], d)  # np.rot90(arr, -direction)
        self.rotate_slice(axis, layer, d)

    @chainable_method
    def rotate(self, axis: int, layer: int, direction: int = 1):
        """
        统一旋转入口,AXIS-SLICE 旋转
        axis: 'x' | 'y' | 'z':0,1,2
        layer: 0 ~ n-1
        direction: 1 = 顺时针, -1 = 逆时针
        """
        d = direction % 4
        if d == 0:
            return  # 旋转 0 次

        print('rotate:', axis, layer, direction)

        assert 0 <= axis <= 2, f"unknown axis: {axis}"
        assert -self.mid <= layer < self.mid + 1, f"layer out of range: {layer}"

        axis_vec = self.AXIS_VEC[axis]
        # 最外层需要旋转面本体   x轴 → R/L, y轴 → U/D ,z轴 → F/B
        if layer == self.mid:
            face = self.AXIS_FACE[axis][0]
            fidx = self.face_idx[face]
            normal = self.face_normal[face]
            d *= self.AXIS_SPIN[axis]
            if np.dot(normal, axis_vec) < 0:
                d = -d
            self.rotate_inplace(self.cube[fidx], d)

        elif layer == -self.mid:  # 方向取反
            face = self.AXIS_FACE[axis][1]
            fidx = self.face_idx[face]
            normal = self.face_normal[face]
            d *= self.AXIS_SPIN[axis]
            if np.dot(normal, axis_vec) < 0:
                d = -d
            self.rotate_inplace(self.cube[fidx], -d)

        # 中层处理
        self.rotate_slice(axis, layer, direction)

    @classmethod
    def rotate_state(cls, state: np.ndarray, axis: int, layer: int, direction: int) -> np.ndarray:
        """
          纯函数版本：不修改传入 state，返回新状态 next_state 副本（已经应用旋转）。
          用于 BFS/IDA*/并行扩展时的安全调用。区别实例方法“就地旋转”，完全独立
        """
        arr = state.copy()
        n = arr.shape[1]
        mid = n // 2
        d = direction % 4
        if d == 0:
            return arr

        axis_vec = cls.AXIS_VEC[axis]
        # --- 处理最外层面本体旋转 ---
        if layer == mid:
            face = cls.AXIS_FACE[axis][0]  # 正向面
            fidx = cls.face_idx[face]  # positive / side-0
            normal = cls.face_normal[face]
            dd = d * cls.AXIS_SPIN[axis]
            if np.dot(normal, axis_vec) < 0:
                dd = -dd

            cls.rotate_inplace(arr[fidx], dd)
        elif layer == -mid:
            face = cls.AXIS_FACE[axis][1]  # 反向面
            fidx = cls.face_idx[face]  # negative / side-1
            normal = cls.face_normal[face]

            dd = d * cls.AXIS_SPIN[axis]
            if np.dot(normal, axis_vec) < 0:
                dd = -dd
            cls.rotate_inplace(arr[fidx], -dd)

        # 生成每一条 strip 的坐标列表 (f, r, c)
        strip_coords = cls.coords_from_axis(axis, layer, n)
        # 读取每条 strip 的值（list of lists）
        strips = [[arr[f, r, c] for (f, r, c) in strip] for strip in strip_coords]
        # 环移
        strips = strips[-d:] + strips[:-d]
        # 写回
        for coords, vals in zip(strip_coords, strips):
            for (f, r, c), v in zip(coords, vals):
                arr[f, r, c] = v
        return arr

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

    def all_moves(self):
        """有限邻域 moves（减枝！！）"""
        for axis in range(3):
            for layer in self.center_layers:
                for direction in (1, -1):  # direction 只用 ±1，2 步可视为两步重复
                    yield axis, layer, direction

    def scramble(self, moves: int = 20):
        """生成打乱序列，返回 move list"""
        scramble_moves = []
        for _ in range(moves):
            axis = random.choice(range(3))
            layer = random.choice(self.center_layers)
            direction = random.choice(range(-3, 4))
            scramble_moves.append((axis, layer, direction))
        return scramble_moves  # -> apply

    @staticmethod
    def invert_moves(moves: list):
        """将 moves 转成可还原的逆操作序列（反向 + 方向反）"""
        return [(axis, layer, -direction) for (axis, layer, direction) in reversed(moves)]

    @chainable_method
    def apply(self, moves: list | tuple):
        if not isinstance(moves, list):
            moves = [moves]
        for axis, layer, direction in moves:
            self.rotate(axis, layer, direction)

    def apply_move(self, move: str):
        """
        通用 NxN 解析：
        支持:
            U, U', U2
            R, L, F, B, D
            Rw, Uw, Fw ...
            2Rw, 3Uw', 2Fw2  等
        """
        import re
        # --- 解析方向 ---
        if move.endswith("2"):
            turn_times = 2
            move = move[:-1]
        else:
            turn_times = 1

        if move.endswith("'"):
            dir = -1
            move = move[:-1]
        else:
            dir = +1

        # --- 正则解析宽度（前缀数字）
        m = re.match(r"(\d*)([URFDLB])(w?)$", move)
        if not m:
            raise ValueError(f"无法解析动作: {move}")

        width_txt, face, wide_flag = m.groups()

        # 宽度：无数字 → 默认 1；如果有 'w' 则默认 = 2
        if width_txt:
            width = int(width_txt)
        else:
            width = 2 if wide_flag else 1

        axis, side = self.face_axis[face]
        if side == 0:
            layers = list(range(width))  # 正面方向,U:顶部向下 width 层
        else:
            layers = [self.n - 1 - i for i in range(width)]  # 反: list(range(N - 1, N - width - 1, -1))

        # ---- 执行 primitive moves ----
        prim_moves = []
        for _ in range(turn_times):
            for layer_idx in layers:
                final_dir = dir if side == 0 else -dir  # “正轴面的顺时针” 在 cube 标准中方向不同
                layer = layer_idx - self.mid
                if self.n % 2 == 0 and layer_idx >= self.mid:  # 偶数阶需要跳过中心 0
                    layer += 1
                self.rotate(axis, layer, final_dir)
                prim_moves.append((axis, layer, final_dir))

        return prim_moves  # 可用于记录步骤

    def heuristic(self):
        """估价函数：错误块的数量（简单启发）,对 BFS/IDA*/Beam search 可用,小魔方适用"""
        errors = np.count_nonzero(self.cube != self.solved)
        return errors // max(1, self.n)  # 每个错误影响多个面

    def heuristic_center(self, r: int = 1, state: np.ndarray = None) -> int:
        """
        计算中心错误数量，默认中心启发：统计以 mid 为中心的 (2k+1)x(2k+1) 区域中不等于 center color 的数目。
        r 控制区域大小 (2r+1)x(2r+1) 这里默认取 k= (n//2)//2 令中心区域足够大；可以改成只统计 (mid,mid) 周围 3x3。
        越小越接近目标（可用于 IDA*）
        """
        if state is None:
            state = self.cube
        mid = self.n // 2
        k = max(1, r)  # 可调整为 1 (3x3) 或更大，跳过十字、边缘、角落，只取 3x3 / 5x5 / ... 中心块
        wrong = 0
        for f in range(6):
            face = state[f]
            target = self.solved[f, mid, mid]  # face[mid, mid]

            region = face[mid - k:mid + k + 1, mid - k:mid + k + 1]
            wrong += np.count_nonzero(region != target)

        return int(wrong)

    def get_corners(self, state: np.ndarray = None) -> np.ndarray:
        """返回 8 个角块的三颜色编号（按顺序）shape = (8, 3)"""
        # 角块位置： (face, row, col) 三元组的 3 个集合
        if state is None:
            state = self.cube

        res = np.empty((8, 3), dtype=state.dtype)
        for i, corner in enumerate(self.corner_coords):
            # corner 是 [(face,row,col), ...]
            for j, (f, r, c) in enumerate(corner):
                res[i, j] = state[self.face_idx[f], r, c]
        return res

    def heuristic_corner(self) -> int:
        wrong = 0
        sol = self.get_corners(self.solved)
        cur = self.get_corners()
        for a, b in zip(cur, sol):
            if set(a) != set(b):
                wrong += 1
        return wrong

    def heuristic_edge_mismatch(self, face: str, base: int, state: np.ndarray = None) -> int:
        """
        score: 0 if matched, higher if mismatch. 用于贪心最小化。
        判断指定 face 的 central edge 是否已在 D 层且两面颜色对齐：
        - D-side 的 sticker == base_color
        - side-face 的 sticker == side-face center color
        """
        if state is None:
            state = self.cube
        mid = self.n // 2
        (f1, r1, c1), (f2, r2, c2) = self._central_edge_coords(face)
        s1 = int(state[self.face_idx[f1], r1, c1])
        s2 = int(state[self.face_idx[f2], r2, c2])  # D-side
        target_side = int(state[self.face_idx[f1], mid, mid])
        score = 0
        if s2 != int(base):  # D 面 base_color 是否匹配
            score += 1
        if s1 != target_side:  # 侧面是否正确颜色
            score += 1
        if r2 != mid or c2 not in (mid - 1, mid, mid + 1):  # 如果边块根本不在 D 层：加罚分
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
        (f1, r1, c1), (f2, r2, c2) = self._central_edge_coords('DF')  # (face, r, c)
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
        start_h = self.heuristic_center()
        visited = set()

        def dfs(depth, bound, path, last_move: tuple = None):
            key = self.encode()  # self.get_state()
            if key in visited:
                return math.inf, None
            visited.add(key)

            h = self.heuristic_center()
            f = depth + h
            if f > bound:
                visited.remove(key)
                return f, None

            if h == 0:
                return True, path.copy()

            if depth == max_depth:
                visited.remove(key)
                return math.inf, None

            best = math.inf
            for move in self.all_moves():
                axis, layer, direction = move
                if last_move and last_move == (axis, layer, -direction):  # is_inverse
                    continue

                self.rotate(axis, layer, direction)  # try move,试着旋转一下
                path.append(move)
                res, sol = dfs(depth + 1, bound, path, move)
                self.rotate(axis, layer, -direction)  # undo move
                path.pop()

                if res is True:
                    return True, sol
                if res < best:
                    best = res

            visited.remove(key)
            return best, None

        bound = start_h
        path = []
        while True:
            visited.clear()
            res, sol = dfs(0, bound, path)
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
        start_wrong = self.heuristic_center(r)
        best = None
        best_delta = 0
        for move in self.all_moves():
            next_state = self.rotate_state(self.cube, *move)
            h = self.heuristic_center(r, next_state)

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
        mid = self.n // 2
        base_color = int(self.cube[self.face_idx['D'], mid, mid])
        targets = ['F', 'R', 'B', 'L']

        candidate_single_moves = list(self.all_moves())
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
                    axis, layer, direction = move
                    if cur_path and cur_path[-1] == (axis, layer, -direction):  # is_inverse
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

            for move in self.all_moves():
                axis, layer, direction = move
                # forbid immediate reversal
                if path and path[-1] == (axis, layer, -direction):  # (p1[2] + p2[2]) % 4 == 0
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
        if self.is_solved():
            return []
        return self.reduction()

    @classmethod
    def build_rotate_map(cls) -> dict:
        """
         面的邻接关系（6 * 4） (face, type, idx, reverse)
        从 SLICE_MAP 推导 ROTATE_MAP（返回 dict）。
        - SLICE_MAP 的项形如 ('U','row', None, maybe_reverse) 或者 ('F','col', None, maybe_reverse)
          这里的 idx 是 None 占位，实际旋转时要填 layer/index。
        """
        # 哪些面在轴的正/负一侧（约定：layer==0 对应正面）
        # 这里用 +1 表示正面（layer==0），-1 表示反面（layer==n-1）
        FACE_SIGN = {'U': 1, 'F': 1, 'R': 1, 'D': -1, 'B': -1, 'L': -1}

        # 明确定义「当旋转某个 face 时」，每个邻接面沿哪个 index 接触它（这是固定的魔方拓扑）
        # 这些值可直接来源于常用魔方约定
        CONTACT_IDX = {
            'U': {'F': 0, 'R': 0, 'B': 0, 'L': 0},
            'D': {'F': -1, 'L': -1, 'B': -1, 'R': -1},
            'F': {'U': -1, 'R': 0, 'D': 0, 'L': -1},
            'B': {'U': 0, 'L': 0, 'D': -1, 'R': -1},
            'L': {'U': 0, 'F': 0, 'D': 0, 'B': -1},
            'R': {'U': -1, 'B': 0, 'D': -1, 'F': -1},
        }

        SLICE_MAP = {
            'X': [
                ('U', 'col', None, False),
                ('B', 'col', None, True),  # B 需要 reverse !!!
                ('D', 'col', None, False),
                ('F', 'col', None, False),
            ],  # R/L转动：绕 x 轴转动（右/左） 切 col,U → B → D → F
            'Y': [
                ('F', 'row', None, False),
                ('R', 'row', None, False),
                ('B', 'row', None, True),
                ('L', 'row', None, False),
            ],  # U/D转动（上 ↔ 下） 切 row, F → R → B → L
            'Z': [
                ('U', 'row', None, False),
                ('R', 'col', None, False),
                ('D', 'row', None, True),  # col → row（方向变）要 reverse
                ('L', 'col', None, False),
            ]  # F/B转动（前 ↔ 后） 切 row/col,U → R → D → L
        }
        AXIS_FACE_WALK = {
            0: {  # X
                'U': lambda i, layer, n: (i, layer),
                'B': lambda i, layer, n: (n - 1 - i, n - 1 - layer),
                'D': lambda i, layer, n: (n - 1 - i, layer),
                'F': lambda i, layer, n: (i, layer),
            },
            1: {  # Y
                'F': lambda i, layer, n: (i, layer),
                'R': lambda i, layer, n: (i, layer),
                'B': lambda i, layer, n: (n - 1 - i, layer),
                'L': lambda i, layer, n: (n - 1 - i, layer),
            },
            2: {  # Z
                'U': lambda i, layer, n: (layer, i),
                'R': lambda i, layer, n: (i, n - 1 - layer),
                'D': lambda i, layer, n: (n - 1 - layer, n - 1 - i),
                'L': lambda i, layer, n: (n - 1 - i, layer),
            }
        }
        FACE_AXIS = {face: cls.AXIS_NAME[axis] for axis, pair in enumerate(cls.AXIS_FACE)
                     for face in pair}  # 哪个面属于哪个轴
        # 对同一轴，SLICE_MAP[axis] 给出邻接面顺序（环）
        # 对于“正侧面”（FACE_SIGN==1）按 SLICE_MAP 顺序生成
        # 对于“反侧面”（FACE_SIGN==-1）按 (0,3,2,1) 的顺序（这是与面朝向相关的常见置换）
        NEG_ORDER = [0, 3, 2, 1]

        rotate_map = {}
        # 为每个 face 构造 rot list
        for face in cls.FACES:
            axis = FACE_AXIS[face]  # X/Y/Z
            base = SLICE_MAP[axis]  # SLICE_MAP['Y'] = [('F','row',None,rev),...]
            sign = FACE_SIGN[face]

            # build a small lookup from neighbor face -> (type, default_rev from SLICE_MAP)
            # neighbor_info = {entry[0]: (entry[1], entry[3] if len(entry) > 3 else False) for entry in base}

            # choose order of neighbors depending on face sign
            if sign == 1:
                order_idx = [0, 1, 2, 3]
            else:
                order_idx = NEG_ORDER

            seq = []
            for idx in order_idx:
                neighbor_face, neigh_type, _, neigh_rev = base[idx]
                # actual index where neighbor touches this face (0 or -1), from CONTACT_IDX
                contact_i = CONTACT_IDX[face][neighbor_face]
                # reverse flag: combine neighbor's base reverse with any face-contact inversion
                # Using base rev is usually correct; CONTACT_IDX encodes geometric orientation (we used it above)
                rev = neigh_rev
                seq.append((neighbor_face, neigh_type, contact_i, rev))

            rotate_map[face] = seq

        return rotate_map


if __name__ == "__main__":
    cube = RubiksCube(n=5)
    print(cube.face_def())
    print('corner_coords', cube.corner_coords)
    xxx = cube.get_layer_stickers(0, 1, 5)
    print(xxx, len(xxx))
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
        print(cube.cube)
        print(mv)
        print(cube.heuristic())
        inv_moves = cube.invert_moves(mv)
        cube.apply(inv_moves)
        # print(backup )
        return cube.is_solved()


    test_scramble()
    print('rotate_map', cube.build_rotate_map())
    # err = 0
    # for i in range(1000):
    #     if not test_scramble(80):
    #         print(f"❌ failed at round {i}")
    #         err += 1
    # print(f"✔️ all good.err {err}")

    cube = RubiksCube(n=3)
    print(cube.get_state())
    print('corners', cube.get_corners())
    mv = cube.scramble(20)
    cube.apply(mv)
    print(mv)

    # mvs0 = cube.solve()
    # print(mvs0)

    print(cube.is_solved())
    print(cube.color)

    FACE_NORMAL = {
        'R': (1, 0, 0),
        'L': (-1, 0, 0),
        'U': (0, 1, 0),
        'D': (0, -1, 0),
        'F': (0, 0, 1),
        'B': (0, 0, -1),
    }

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


    print(cube.central_edge_coords(5))
