from rime.base import class_property, class_cache, class_status
from rime.cube import CubeBase, StickerCube
from dataclasses import dataclass
import numpy as np
from math import comb, factorial
from collections import deque
import random


@dataclass(frozen=True)
class CubieState:
    """
    G = (S₈ × S₁₂) ⋉ (ℤ₃⁷ × ℤ₂¹¹)
    perm: dict[int, np.ndarray]      # orbit_id -> permutation
    ori: dict[int, np.ndarray]       # orbit_id -> orientation (optional)
    """
    corners_perm: np.ndarray  # (8,)  0..7, ∈ S₈
    corners_ori: np.ndarray  # (8,)  0..2, ∈ ℤ₃
    edges_perm: np.ndarray  # (12,) 0..11, ∈ S₁₂
    edges_ori: np.ndarray  # (12,) 0..1, ∈ ℤ₂

    @classmethod
    def solved(cls) -> "CubieState":
        """
        fully_solved
        - corners_perm (8!)
        - corners_ori  (Z3^7)
        - edges_perm   (12!)
        - edges_ori    (Z2^11)
        符号（permutation）
        几何（orientation）
        群结构（closure / inverse）
        """
        return cls(
            corners_perm=np.arange(8, dtype=np.int8),  # [0,1,...,7]
            corners_ori=np.zeros(8, dtype=np.int8),
            edges_perm=np.arange(12, dtype=np.int8),  # [0,1,...,11]
            edges_ori=np.zeros(12, dtype=np.int8),
        )

    def with_(self, **kwargs):
        data = dict(
            corners_perm=self.corners_perm,
            corners_ori=self.corners_ori,
            edges_perm=self.edges_perm,
            edges_ori=self.edges_ori,
        )
        data.update(kwargs)
        return CubieState(**data)

    def clone(self):
        return CubieState(
            corners_perm=self.corners_perm,
            corners_ori=self.corners_ori,
            edges_perm=self.edges_perm,
            edges_ori=self.edges_ori,
        )

    def __eq__(self, other):
        if not isinstance(other, CubieState):
            return NotImplemented
        return (
                np.array_equal(self.corners_perm, other.corners_perm) and
                np.array_equal(self.corners_ori, other.corners_ori) and
                np.array_equal(self.edges_perm, other.edges_perm) and
                np.array_equal(self.edges_ori, other.edges_ori)
        )

    def encode_state(self) -> np.ndarray:
        """40:(12+8)*2"""
        return np.concatenate([self.corners_perm, self.edges_perm, self.corners_ori, self.edges_ori])

    def to_stickers(self, n: int = 3) -> np.ndarray:
        """
        从 CubieState 生成完整的贴纸状态 (6, n, n) 数组
        - 值是颜色索引 0~5 或颜色字符（根据需要）
        - 支持任意 n（中心块固定，边角根据 cubie）
        某一个 gauge 下的具体代表
        Sticker equality is not guaranteed; cubie equality is the invariant
        """
        base = CubeBase(n=n)
        # 输出数组：6面 × n × n，值是颜色索引 0~5
        stickers = np.zeros((6, n, n), dtype=np.int8)  # base.solved.copy()
        # 填充中心块（固定颜色）
        center_colors = np.array([0, 1, 2, 3, 4, 5])  # U D F B R L
        for f in range(6):
            stickers[f, n // 2, n // 2] = center_colors[f]

        # solved 状态下每个角块的颜色顺序
        solved_corners = base.get_corners(base.solved)  # (8, 3)

        # solved 边块颜色（12 个位置）
        solved_edges = base.get_edges(base.solved)  # (12, 2)
        # [[(face_idx, r, c), (face_idx, r, c), (face_idx, r, c)],..
        for i, corners in enumerate(base.corner_coords(n=n)):
            cubie_id = self.corners_perm[i]
            twist = self.corners_ori[i]

            # solved 状态下这个 cubie 的 3 个颜色顺序
            colors = solved_corners[cubie_id]  # (3,)
            # 旋转 twist 次（顺时针）
            actual_colors = np.roll(colors, -twist)  # orientation 负 twist = 逆时针 roll

            # 贴到 3 个面
            for sticker_pos, color in zip(corners, actual_colors):  # [(f, r, c)]:[val]
                stickers[sticker_pos] = color

        for i, edges in enumerate(base.edge_coords(n=n)):  # [[(face_idx, r, c), (face_idx, r, c)],
            cubie_id = self.edges_perm[i]
            flip = self.edges_ori[i]

            colors = solved_edges[cubie_id]  # (12,2)
            actual_colors = np.roll(colors, flip)  # 翻转或不翻 flip 1 swap: (colors[1], colors[0])
            for sticker_pos, color in zip(edges, actual_colors):
                stickers[sticker_pos] = color

        return stickers

    def is_solvable(self) -> bool:
        """
        Σ corner orientation ≡ 0 (mod 3)
        Σ edge_orientation ≡ 0 (mod 2)
        parity(corners_perm) == parity(edges_perm)
        不是任意贴纸顺序都能对应到 CubieState，必须保证这些约束
        两个角互换 parity 翻转
        角朝向改变 parity 不变
        两条 edge 内部翻转 parity 不变
        每个 corner 始终是 3 个不同 face
        每个 edge 始终是 2 个不同 face
        没有 sticker 被“拆散”或“拼错”
        """
        # 1. corner orientation
        if self.corners_ori.sum() % 3 != 0:  # 每行 Z3 求和，判断总约束
            return False

        # 2. edge orientation
        if self.edges_ori.sum() % 2 != 0:
            return False

        # 3. parity
        if CubeBase.permutation_parity(self.corners_perm) != CubeBase.permutation_parity(self.edges_perm):
            return False

        return True

    @class_property('UD_SLICE_EDGES')
    def ud_slice_edges(cls) -> tuple:
        '''piece slice cubies（identity 集合),Cubie ID (4, 5, 6, 7)'''
        solved = cls.solved()
        return tuple(int(solved.edges_perm[pos]) for pos in CubeBase.SLICE_POSITIONS)

    @class_property('NON_SLICE_EDGES')
    def non_slice_edges(cls) -> tuple:
        '''非 slice cubie,Cubie ID [0, 1, 2, 3, 8, 9, 10, 11]'''
        solved = cls.solved()
        return tuple(int(solved.edges_perm[pos]) for pos in CubeBase.NON_SLICE_POSITIONS)

    @class_property('SOLVED_UD')
    def solved_ud(cls) -> int:
        ''' UD-slice membership = solved'''
        return cls.solved().ud_slice_coord()

    def is_phase1_solved(self):
        return (
                np.all(self.corners_ori == 0) and
                np.all(self.edges_ori == 0) and
                self.is_ud_slice_separated()  # self.ud_slice_coord() == self.solved_ud: 69
        )

    def is_ud_slice_separated(self) -> bool:
        """
        所有 UD-slice 边都不在 U/D 层 {4, 5, 6, 7}. slice 边在 中层, 只关心集合，不关心顺序
        """
        return all(
            pos in CubeBase.SLICE_POSITIONS
            for pos, cubie in enumerate(self.edges_perm)
            if cubie in self.ud_slice_edges
        )

    def is_phase2_solved(self) -> bool:
        """
        Phase-2 goal:
        - corners_perm solved
        - edges_perm solved
        （ori 在 Phase-1 已保证为 0）
        """
        return (
                np.array_equal(self.corners_perm, np.arange(8)) and
                np.array_equal(self.edges_perm, np.arange(12))  # self.solved().edges_perm
        )

    def is_corner_solved(self):
        return np.all(self.corners_perm == np.arange(8))

    @property
    def corner_parity(self):
        return CubeBase.permutation_parity(self.corners_perm)

    @property
    def edge_parity(self):
        return CubeBase.permutation_parity(self.edges_perm)

    def corner_ori_coord(self) -> int:
        """8 个角，每个 Z₃ 自由度 = 7, 3^7 - 1 = 2186"""
        coord: int = 0
        for i in range(7):
            coord = coord * 3 + int(self.corners_ori[i])
        return coord

    @class_cache(key=lambda coord: coord)
    @staticmethod
    def decode_corner_ori(coord: int) -> np.ndarray:
        """
        coord ∈ [0, 3^7)
        返回 shape (8,) 的 corners_ori，满足 sum ≡ 0 (mod 3)
        diff = (out_ori - delta) % 3
        """
        ori = np.zeros(8, dtype=np.int8)

        s = 0
        for i in range(6, -1, -1):
            ori[i] = coord % 3
            s += ori[i]
            coord //= 3

        ori[7] = (-s) % 3
        return ori

    def edge_ori_coord(self) -> int:
        """12 个 edge，每个 Z₂ 自由度 = 11, 2^11 - 1 = 2047"""
        coord: int = 0
        for i in range(11):
            coord = (coord << 1) | int(self.edges_ori[i])
        return coord

    @class_cache(key=lambda coord: coord)
    @staticmethod
    def decode_edge_ori(coord: int) -> np.ndarray:
        """
        coord ∈ [0, 2^11)
        返回 shape (12,) 的 edges_ori，满足 sum ≡ 0 (mod 2)
        diff = (out_ori ^ delta)
        """
        ori = np.zeros(12, dtype=np.int8)

        s = 0
        for i in range(10, -1, -1):
            ori[i] = coord & 1
            s ^= ori[i]
            coord >>= 1

        ori[11] = s
        return ori

    @classmethod
    def encode_ud_slice(cls, edges_perm: list[int]) -> int:
        """
         从 12 个位置里选 4 个 C(12, 4) = 495
         UD-slice 组合坐标：只看 perm，不看 ori，且中层边定义为不在 U/D 层的 4 条边
         edges_perm[pos] = cubie at position pos rank
        """
        coord = 0
        k = 4  # remaining
        for pos in range(11, -1, -1):
            if edges_perm[pos] in cls.ud_slice_edges():  # cubie ∈ slice cubies
                coord += comb(pos, k)
                k -= 1
                if k == 0:
                    break
        return coord

    def ud_slice_coord(self) -> int:
        """
        Phase-1: which 4 positions are occupied by slice edges
        哪些 edge cubie 在 slice positions（4,5,6,7）
        """
        return self.encode_ud_slice(self.edges_perm.tolist())

    @class_cache(key=lambda coord: coord)
    @classmethod
    def decode_ud_slice(cls, coord: int) -> np.ndarray:
        """
        根据给定的 UD-slice 坐标 (0~494)，解码为一个 edges_perm
        使得它的 ud_slice_coord() == coord，
        其他部分（角块、边块朝向等）保持与 base_state(solved) 相同。
        诱导出 ud_slice_perm[coord] = m 作用后的新坐标
        参数:
            coord: int, 0 到 494
        """
        positions = []
        k = 4
        c = coord
        for pos in range(11, -1, -1):
            if k == 0:
                break
            comb_val = comb(pos, k)
            if c >= comb_val:
                positions.append(pos)
                c -= comb_val
                k -= 1

        positions.sort()  # 升序

        # canonical fill, identity is irrelevant
        slice_cubies = list(cls.ud_slice_edges())  # sorted slice cubies，按 solved 中的顺序填入(4, 5, 6, 7)
        non_slice = list(cls.non_slice_edges())
        it_slice = iter(slice_cubies)
        it_non = iter(non_slice)

        perm = np.zeros(12, dtype=np.int8)
        for pos in range(12):
            if pos in positions:
                perm[pos] = next(it_slice)
            else:
                perm[pos] = next(it_non)

        return perm

    @staticmethod
    def encode_perm(perm: list[int]) -> int:
        """
        perm: 长度 n 的排列，值域 0..n-1
        返回 [0, n!-1]
        """
        n = len(perm)
        code = 0
        factor = 1
        for i in range(n - 1, -1, -1):
            cnt = 0
            for j in range(i + 1, n):
                if perm[j] < perm[i]:
                    cnt += 1
            code += cnt * factor
            factor *= (n - i)
        return code

    @class_cache(key=lambda code, n: (code, n))
    @staticmethod
    def decode_perm(code: int, n: int) -> list[int]:
        """
        code: 0 .. n!-1
        返回 perm，值域 0..n-1
        """
        elems = list(range(n))
        perm = [0] * n

        for i in range(n):
            fact = factorial(n - 1 - i)
            idx = code // fact
            code %= fact
            perm[i] = elems.pop(idx)

        return perm

    @staticmethod
    def encode_perm_coord(edges_perm: list[int], positions: list[int],
                          cubies: list[int] | tuple[int]) -> int:
        """
        从完整的 edges_perm 中，提取指定位置 + 指定 cubie 子集 的相对置换坐标
        相对排列的坐标 (0 ~ len(cubies)! - 1)
        """
        cubie_to_rel = {cubie: i for i, cubie in enumerate(cubies)}  # 固定 cubie id

        # 提取当前状态下，在指定 positions 上出现的 cubie 的相对索引
        rel_indices = [cubie_to_rel[int(edges_perm[pos])] for pos in positions]
        return CubieState.encode_perm(rel_indices)

    @staticmethod
    def comb_to_index(bits: list[int], n: int, k: int) -> int:
        """
        bits: 长度 n 的 0/1，恰有 k 个 1
        返回 [0, C(n,k))
        """
        idx = 0
        r = k
        for i in range(n):
            if bits[i]:
                idx += comb(n - i - 1, r)
                r -= 1
                if r == 0:
                    break
        return idx

    @staticmethod
    def index_to_comb(idx: int, n: int, k: int) -> list[int]:
        bits = [0] * n
        r = k
        for i in range(n):
            if r == 0:
                break
            c = comb(n - i - 1, r)
            if idx >= c:
                bits[i] = 1
                idx -= c
                r -= 1
        return bits

    @staticmethod
    def encode_corner_coset(corners_perm: list[int]) -> int:
        # corner_coset（U 层是哪 4 个 corner）
        bits = [0] * 8
        for pos in CubeBase.U_CORNER_POSITIONS:  # 例如 [0,1,2,3]
            piece = int(corners_perm[pos])
            bits[piece] = 1

        return CubieState.comb_to_index(bits, 8, 4)

    @class_cache(key=lambda corner_coset: corner_coset)
    @staticmethod
    def canonical_corner_coset(corner_coset: int) -> np.ndarray:
        """
        corner_coset ∈ [0, 70)
        canonical corner coset（只放层，不管层内排列）
        """
        corners_perm = np.zeros(8, dtype=np.int8)
        bits = CubieState.index_to_comb(corner_coset, 8, 4)
        u_pieces = [i for i in range(8) if bits[i]]
        d_pieces = [i for i in range(8) if not bits[i]]
        # corner layer-membership 固定顺序（canonical）
        for pos, piece in zip(CubeBase.U_CORNER_POSITIONS, sorted(u_pieces)):
            corners_perm[pos] = piece

        for pos, piece in zip(CubeBase.D_CORNER_POSITIONS, sorted(d_pieces)):
            corners_perm[pos] = piece
        return corners_perm

    @staticmethod
    def encode_ud_slice_perm(edges_perm: list[int]) -> int:
        """
        UD-slice 内 4 个 edge 的排列
        Phase-1 已保证 membership 正确
        返回 [0, 24)
        """
        slice_edges = [edges_perm[pos] for pos in CubeBase.SLICE_POSITIONS]
        return CubieState.encode_perm(slice_edges)  # 0..23

    @class_cache(key=lambda ud_slice: ud_slice)
    @staticmethod
    def canonical_ud_slice_edges(ud_slice: int) -> np.ndarray:
        """
        canonical ud-slice，仅体现  slice membership
        只保证：哪些 piece 在 slice 层
        不保证 slice 内排列
        不保证 parity
        """
        edges_perm = np.zeros(12, dtype=np.int8)
        bits = CubieState.index_to_comb(ud_slice, 12, 4)

        slice_pieces = [i for i in range(12) if bits[i]]
        non_slice_pieces = [i for i in range(12) if not bits[i]]

        # canonical：按 piece id 排序填入
        for pos, piece in zip(CubeBase.SLICE_POSITIONS, sorted(slice_pieces)):
            edges_perm[pos] = piece

        for pos, piece in zip(CubeBase.NON_SLICE_POSITIONS, sorted(non_slice_pieces)):
            edges_perm[pos] = piece

        return edges_perm


@dataclass(frozen=True)
class CubieMove:
    """
    perm_map: dict[int, np.ndarray]       # orbit_id -> σ
    ori_delta: dict[int, np.ndarray]      # orbit_id -> Δ (mod k)
    """
    # permutation: new_pos = perm[old_pos]
    corners_perm: np.ndarray  # σ_c (8,) / tuple[int, ...]
    edges_perm: np.ndarray  # σ_e (12,)

    # orientation delta (mod)
    corners_ori_delta: np.ndarray  # Δ_c (8,)  int mod 3
    edges_ori_delta: np.ndarray  # Δ_e (12,) int mod 2

    def act(self, s: CubieState) -> CubieState:
        '''
        右作用 (state' = state ∘ move)
        用于pruning/BFS/IDA*/solver/phase判断。所有搜索/优化逻辑必须用此，确保半直积自洽和pruning table匹配。
        Phase-1 / Phase-2  / group logic —— 只允许用 act.用于群论/search逻辑
        act(s, m) = s ∘ m 编码等价, (π, o) ∘ (σ, Δ) ，不做 canonical 修正
        self.corners_perm 已经是“索引搬运表” 完全忽略 pull back
        self.corners_ori_delta 已经在 state 的 reference 下
        连续多次 apply：act(act(act(s, m1), m2), m3) = s ∘ m1 ∘ m2 ∘ m3
        new_ori = (old_ori[perm⁻¹] + Δo) % k
        new_ori = (old_ori ∘ perm + ori_delta) mod 3 （复合顺序：先 old 后 self）
        '''
        # 应用 delta
        cp = s.corners_perm[self.corners_perm]  # new_corners_perm
        ep = s.edges_perm[self.edges_perm]  # new_edges_perm

        co = (s.corners_ori[self.corners_perm] + self.corners_ori_delta) % 3  # new_corners_ori
        eo = (s.edges_ori[self.edges_perm] + self.edges_ori_delta) % 2  # new_edges_ori
        return CubieState(cp, co, ep, eo)

    def act_left(self, s: CubieState) -> CubieState:
        """
        左作用 (state' = move ⋅ state),半直积作用律,
        用于几何构造/贴纸旋转/调试/测试。仅限从solved生成state，或与外部模型对齐
        左作用（几何）  apply(m, s) = m ∘ s  = move ∘ state, 用于几何/贴纸
        Apply this CubieMove to a CubieState using semidirect product law.
        This version is topology-safe and orientation-correct.
        |G| ≈ 4.3e19
        G = (Perm × Ori) ⋊ Move
        群作用,严格等价于：
        (σ, Δ) · (π, o) = (σ∘π, o∘σ⁻¹ + Δ∘σ⁻¹)

        new_perm = σ ∘ old_perm
        new_ori[i] = old_ori[σ⁻¹(i)] + Δ[σ⁻¹(i)]
        new_ori[i] = old_ori[ self.perm⁻¹(i) ] + self.ori_delta[ self.perm⁻¹(i) ]
        """
        # ---------- corners ----------
        σc = self.corners_perm
        Δc = self.corners_ori_delta
        σc_inv = np.argsort(σc)

        new_corners_perm = σc[s.corners_perm]
        new_corners_ori = (s.corners_ori[σc_inv] + Δc[σc_inv]) % 3
        # ---------- edges ----------
        σe = self.edges_perm
        Δe = self.edges_ori_delta
        σe_inv = np.argsort(σe)

        new_edges_perm = σe[s.edges_perm]
        new_edges_ori = (s.edges_ori[σe_inv] + Δe[σe_inv]) % 2

        return CubieState(
            corners_perm=new_corners_perm,
            corners_ori=new_corners_ori,
            edges_perm=new_edges_perm,
            edges_ori=new_edges_ori,
        )

    def convert(self) -> "CubieMove":
        """
        桥梁（双向）act_left ↔ act
        Convert this move (assuming left/right action delta) to right/left action equivalent.
        Δ_left = -Δ_right   (mod k)
        delta = -delta % mod
        """
        return CubieMove(
            corners_perm=self.corners_perm,
            corners_ori_delta=(-self.corners_ori_delta) % 3,  # 翻转符号
            edges_perm=self.edges_perm,
            edges_ori_delta=(-self.edges_ori_delta) % 2,
        )

    def compose(self, other: "CubieMove") -> "CubieMove":
        """
        multiply（半直积乘法）右作用复合：self ∘ other = 先 self 后 other
        (self ∘ other).act(s) == other.act(self.act(s))
        (σ₁, Δ₁) ∘ (σ₂, Δ₂) = (σ₁ ∘ σ₂, Δ₁ + Δ₂ ∘ σ₁⁻¹)
        """

        # ---------- corners ----------
        σ1 = self.corners_perm
        Δ1 = self.corners_ori_delta
        σ2 = other.corners_perm
        Δ2 = other.corners_ori_delta

        corners_perm = σ1[σ2]  # σ1 ∘ σ2
        corners_ori_delta = (Δ1[σ2] + Δ2) % 3

        # ---------- edges ----------
        τ1 = self.edges_perm
        δ1 = self.edges_ori_delta
        τ2 = other.edges_perm
        δ2 = other.edges_ori_delta

        edges_perm = τ1[τ2]
        edges_ori_delta = (δ1[τ2] + δ2) % 2

        return CubieMove(
            corners_perm=corners_perm,
            corners_ori_delta=corners_ori_delta,
            edges_perm=edges_perm,
            edges_ori_delta=edges_ori_delta,
        )

    @staticmethod
    def square(m: "CubieMove") -> "CubieMove":
        # m ∘ m
        return m.compose(m)

    def inverse(self) -> "CubieMove":
        """
        右作用逆元（半直积）：
        (σ, Δ)⁻¹ = (σ⁻¹, -Δ ∘ σ⁻¹)
        """
        # ---------- corners ----------
        σ = self.corners_perm
        Δ = self.corners_ori_delta
        σ_inv = np.argsort(σ)

        corners_perm = σ_inv
        corners_ori_delta = (-Δ[σ_inv]) % 3

        # ---------- edges ----------
        τ = self.edges_perm
        δ = self.edges_ori_delta
        τ_inv = np.argsort(τ)

        edges_perm = τ_inv
        edges_ori_delta = (-δ[τ_inv]) % 2

        return CubieMove(
            corners_perm=corners_perm,
            corners_ori_delta=corners_ori_delta,
            edges_perm=edges_perm,
            edges_ori_delta=edges_ori_delta,
        )

    @classmethod
    def identity(cls) -> "CubieMove":
        # Identity,基坐标系,什么都没发生
        return cls(
            corners_perm=np.arange(8, dtype=np.int8),
            corners_ori_delta=np.zeros(8, dtype=np.int8),
            edges_perm=np.arange(12, dtype=np.int8),
            edges_ori_delta=np.zeros(12, dtype=np.int8),
        )

    def __eq__(self, other):
        if not isinstance(other, CubieMove):
            return NotImplemented
        return (
                np.array_equal(self.corners_perm, other.corners_perm) and
                np.array_equal(self.edges_perm, other.edges_perm) and
                np.array_equal(self.corners_ori_delta, other.corners_ori_delta) and
                np.array_equal(self.edges_ori_delta, other.edges_ori_delta)
        )

    def __hash__(self):
        return hash((
            self.corners_perm.tobytes(),
            self.corners_ori_delta.tobytes(),
            self.edges_perm.tobytes(),
            self.edges_ori_delta.tobytes(),
        ))

    @classmethod
    def from_rotation(cls, axis: int, side: int, direction: int) -> 'CubieMove':
        """
        生成的是「右作用 / apply 语义」的 move，理论 move，在 cubie 参考系下定义
        定义在“绝对 reference 坐标系”上的群元素,几何表示
        独立计算 move 的 perm 和 delta（不依赖贴纸，用坐标模拟旋转）
        Build CubieMove from rotation parameters.
        axis: 0 = X (R/L), 1 = Y (U/D), 2 = Z (F/B)
        side: +1 or -1,layer ∈ {+1,-1} side sign，不是层编号
        direction: +1 (90°) or -1 (-90°)
        orientation delta（Z₂） orientation delta（Z₃）
        corner_ori_delta[i] ∈ {0,1,2} new_ori = (old_ori ∘ perm + ori_delta) mod 3
        局部增量,比较“旋转前后”，每个 cubie 去了谁的位置，朝向变了多少,move 对“被搬到 i 位置的角块”额外施加了多少扭转
        """
        assert axis in (0, 1, 2)
        assert side in (-1, 0, 1)

        turns = abs(direction) % 4  # Compute turns direction % 4
        sign_dir = 1 if direction > 0 else -1
        if turns == 0:
            return cls.identity()  # Identity
        # Define corner and edge positions
        corner_positions = np.array(CubeBase.CORNER_POS_SIGNS, dtype=np.int8)
        edge_positions = np.array(CubeBase.EDGE_POS_SIGNS, dtype=np.int8)
        # Current positions for simulation
        current_corner_pos = corner_positions.copy()
        current_edge_pos = edge_positions.copy()
        # Affected masks,affected 集合在 move 内不是常量,必须在 move 开始前就确定
        affected_corners = (corner_positions[:, axis] == side)
        affected_edges = (edge_positions[:, axis] == side)
        # Initialize deltas
        corners_ori_delta = np.zeros(8, dtype=np.int8)
        edges_ori_delta = np.zeros(12, dtype=np.int8)

        for _ in range(turns):
            # Update corner ori deltas if not U/D axis
            if axis != 1:  # U/D 不变,不 twist
                a = (axis + 1) % 3
                b = (axis + 2) % 3
                for i in range(8):
                    if affected_corners[i]:
                        sign_a = np.sign(current_corner_pos[i, a])
                        sign_b = np.sign(current_corner_pos[i, b])
                        # corner 的朝向变化 = 局部右手系在旋转下的 twist,右手规则 + sign_dir 翻转 ccw 加负号是为了让顺时针90°对应 +2 或 -1
                        twist = (-sign_a * sign_b * sign_dir) % 3
                        corners_ori_delta[i] = (corners_ori_delta[i] + twist) % 3

            # Update edge ori deltas if F/B axis
            if axis == 2:  # F/B 变,翻转
                for i in range(12):
                    if affected_edges[i]:
                        edges_ori_delta[i] ^= 1  # Z2 翻转,翻转不依赖 sign_dir（90° 和 -90° 都翻一次） = (edges_ori_delta[i] + 1) % 2

            # R/L (axis=0): edges 不变
            # U/D (axis=1): 都不变

            # Update positions with rotation,必须是 right-hand
            for i in range(8):
                if affected_corners[i]:
                    current_corner_pos[i] = CubeBase.rotate_coord(current_corner_pos[i], axis, sign_dir)

            for i in range(12):
                if affected_edges[i]:
                    current_edge_pos[i] = CubeBase.rotate_coord(current_edge_pos[i], axis, sign_dir)

        # 计算 perm（从 current_pos 映射回原始 pos）
        # Compute perms: for each original i, find the dst where original_pos[dst] == current_pos[i]
        corners_perm = np.zeros(8, dtype=np.int8)
        for i in range(8):
            dst = np.where(np.all(corner_positions == current_corner_pos[i], axis=1))[0][0]
            corners_perm[i] = dst

        edges_perm = np.zeros(12, dtype=np.int8)
        for i in range(12):
            dst = np.where(np.all(edge_positions == current_edge_pos[i], axis=1))[0][0]
            edges_perm[i] = dst

        return cls(
            corners_perm=corners_perm,
            corners_ori_delta=corners_ori_delta,
            edges_perm=edges_perm,
            edges_ori_delta=edges_ori_delta,
        )

    def to_sticker_perm(self, n: int) -> np.ndarray:
        """
        把 CubieMove 转换为 perm 供 StickerMove。
        new_sticker[i] = old_sticker[perm[i]]   （右作用）
        """
        mv = next((a for a, m in self.prim_moves.items() if m == self), None)
        assert mv is not None, 'not prim move,composed!'
        axis, side, direction = mv
        layer = side * (n // 2)
        state_idx = np.arange(6 * n * n, dtype=np.int32).reshape(6, n, n)
        CubeBase.rotate_core(state_idx, axis, layer, direction)  # rotated，得到 pull-back 映射（new_pos → old_pos）
        return np.argsort(state_idx.reshape(-1))  # 逆置换的逆 = 正向置换

    @classmethod
    def act_moves(cls, state: CubieState, moves: list['CubieMove']) -> tuple['CubieMove', 'CubieState']:
        '''state = M_n ∘ ... ∘ M_2 ∘ M_1 (state)'''
        mv = cls.identity()
        current = state
        for m in moves:
            current = m.act(current)
            mv = mv.compose(m)  # 右复合
        return mv, current  # mv.act(state)

    @classmethod
    def apply(cls, state: CubieState, moves: list[tuple]) -> CubieState:
        '''状态级 API 等价 act_moves'''
        for k in moves:
            # print(k, state)
            state = cls.prim_moves[k].act(state)  # cls.from_rotation(*k).act(state)
        return state

    @staticmethod
    def is_redundant(last, cur) -> bool:
        if last is None:
            return False

        axis1, side1, dir1 = last
        axis2, side2, dir2 = cur

        # 同轴同层,连续转，反向,必冗余
        if axis1 == axis2 and side1 == side2:
            return (dir1 + dir2) % 4 == 0

        return False

    @class_property('PRIM_MOVES')
    def prim_moves(cls) -> dict[tuple, 'CubieMove']:
        """
        CubieMove  ──apply──▶ CubieState
        18 BFS / IDDFS 深度可能 +1 / 所有 18 个基本 move（U D R L F B 的 ±90° 和 180°）
        """
        prim_moves = {}  # 生成 CubieMove delta
        for axis in (0, 1, 2):
            for side in (-1, +1):
                for direction in (-1, +1, +2):
                    prim_moves[(axis, side, direction)] = cls.from_rotation(axis, side, direction)  # .convert()
        return prim_moves

    @class_property('PHASE0_MOVES')
    def phase0_moves(cls) -> dict[tuple, 'Phase0Action']:
        return {k: Phase0Action.phi(m) for k, m in cls.prim_moves.items()}

    @class_property('PHASE1_MOVES')
    def phase1_moves(cls) -> dict[tuple, 'Phase1Action']:
        return {k: Phase1Action.lift(p) for k, p in cls.phase0_moves.items()}

    @class_property('PHASE15_MOVES')
    def phase15_moves(cls) -> dict[tuple, 'Phase15Action']:
        return {k: Phase15Action.phi(m) for k, m in cls.prim_moves.items()}

    @class_property('PHASE2_MOVES')
    def phase2_moves(cls) -> dict[tuple, 'Phase2Action']:
        '''⟨ U, D, L², R², F², B² ⟩'''
        moves: dict[tuple, CubieMove] = {}  # 去重
        for (axis, side, direction), m in cls.prim_moves.items():
            if abs(side) != 1:  # 只允许外层
                continue
            if axis == 1:  # U / D 的 ±90°
                moves[(axis, side, direction)] = m
            else:  # X/Z 轴,R/L/F/B，只取 180°
                if direction == 2:
                    moves[(axis, side, direction)] = m
                # if direction == 1:  # 只生成一次，避免重复
                #     key = (axis, side, 2)
                #     moves[key] =  m.compose(m)  # 合并了对称的 180°

        return {k: Phase2Action.phi(m) for k, m in moves.items()}

    @staticmethod
    def scramble_state(length: int = 50, moves: list['CubieMove'] | None = None) -> CubieState:
        """
        生成一个随机打乱的 CubieState,使用 act_left 模拟物理转动顺序，更符合直观和 sticker 模型
        """
        if moves is None:
            moves = list(CubieMove.prim_moves.values())

        state = CubieState.solved()
        # 连续应用随机 move,random.sample(moves, length)
        for _ in range(length):
            m = random.choice(moves)
            state = m.act_left(state)  # act

        assert state.is_solvable()
        return state

    def generate_commutator_moves(cls, gens: dict[tuple, 'CubieMove'], max_len: int = 20) -> dict:
        """
        返回可用的 commutator 序列,构造局部操作序列
        commutator(A, B) = A B A⁻¹ B⁻¹
        防止非法 orientation
        防止 parity 错误
        防止 edge flip / corner twist
        保证物理可达，筛掉非法状态

        贴纸世界	冗余，迁移
        cubie 世界	搜索空间裁剪
        IDA* / Kociemba	必需
        """
        commutators = {}
        solved = CubieState.solved()

        for A, A_move in gens.items():
            for B, B_move in gens.items():
                if A == B:
                    continue

                seq = CubeBase.commutator([A], [B])
                if len(seq) > max_len:
                    continue
                m = A_move.compose(B_move)
                s = m.act(solved)
                if s.is_phase1_solved():  # check_state/is_solvable
                    commutators[tuple(seq)] = m

        return commutators

    @property
    def edge_parity_delta(self) -> int:
        # edge permutation 奇偶 0 or 1
        return CubeBase.permutation_parity(self.edges_perm)

    @staticmethod
    def build_move_part(perm0, ori0, perm1, ori1, mod: int) -> tuple[np.ndarray, np.ndarray]:
        """
        右作用下求 move delta：s1 = s0 ∘ m, s1 = s0 ∘ m → m = s0⁻¹ ∘ s1
        move_ori[new_pos] = ori1[new_pos] - ori0[old_pos]   (mod)
        """
        n = len(perm0)
        move_perm = np.zeros(n, dtype=np.int8)
        move_ori = np.zeros(n, dtype=np.int8)
        # 逆置换
        inv_perm0 = np.argsort(perm0)  # cubie → pos in s0
        for pos in range(n):
            cubie = perm1[pos]  # pos 在 s1 的 cubie
            old_pos = inv_perm0[cubie]  # 这个 cubie 在 s0 的位置
            move_perm[pos] = old_pos  # m 把 old_pos 的内容搬到 pos
            move_ori[pos] = (ori1[pos] - ori0[old_pos]) % mod  # ori delta = ori1[pos] - ori0[old_pos]
            assert (ori0[old_pos] + move_ori[pos]) % mod == ori1[pos]

        return move_perm, move_ori

    @classmethod
    def build(cls, s0: 'CubieState', s1: 'CubieState') -> "CubieMove":
        """
         相对于 s 的局部 delta move
         s0 原始 CubieState
         s1 旋转后状态
         s1 = s0 ∘ m   （右作用语义）
         m = s0⁻¹ ∘ s1
         构建 CubieMove：不依赖贴纸索引顺序来算 delta，直接从 CubieState 计算。
         s0 = CubieState.solved()
         CubieMove.build(s0, move.act(s0)) == move
        """
        assert s0.is_solvable() and s1.is_solvable(), f"States must be solvable:{s0}\n{s1}"
        σc, Δc = cls.build_move_part(s0.corners_perm, s0.corners_ori, s1.corners_perm, s1.corners_ori, 3)
        σe, Δe = cls.build_move_part(s0.edges_perm, s0.edges_ori, s1.edges_perm, s1.edges_ori, 2)
        return cls(
            corners_perm=σc,
            corners_ori_delta=Δc,
            edges_perm=σe,
            edges_ori_delta=Δe,
        )

    @staticmethod
    def build_pruning_table(
            moves: list["QuotientMove"],  # Phase0Action|Phase1Action|Phase2Action
            apply_move: callable,  # (move, coord) -> new_coord
            start_coord: tuple | int = 0,  # solved 状态在该坐标下的编码,
            table_shape: tuple | int = 495,  # ud:495/40320, (3**7, 2**11)
    ) -> np.ndarray:
        """
        Phase 剪枝表构建函数
        |Q| = 3^7 × 2^11 × C(12,4) = 2,217,093,120

        参数:
            moves: List[Phase2Action]，所有允许的移动
            apply_move: callable，函数，签名: (move, current_coord) -> new_coord
                        Phase2 示例: 提取对应 perm_map 的函数
                                    lambda m, idx: m.corner_perm_map[idx]
                                    lambda m, idx: m.edge_perm_map[idx]
                        Phase1 示例:
                                lambda m, (co,eo): (m.corner_ori_map[co], m.edge_ori_map[eo])
                                lambda m, ud: m.ud_slice_map[ud]

            start_coord: int，solved 状态在该坐标下的索引
                (0, 0)
                CubieState.encode_perm(list(range(8)))  # corner 0~7 顺序编码为 0
                CubieState.encode_perm([NON_SLICE_EDGES.index(p) for p in NON_SLICE_EDGES]  # edge 相对排列 0~7
            table_shape:
                    表形状，用于 np.ndarray 分配
                     Phase2: 40320 或 495
                     Phase1: (2187, 2048) CO_EO_PRUNE 或 495
        返回:
            dist: np.ndarray(shape=(40320,), dtype=np.int8)，距离表，-1 表示不可达
        """
        dist = np.full(table_shape, -1, dtype=np.int8)
        dist[start_coord] = 0
        queue = deque([start_coord])

        while queue:
            cur = queue.popleft()
            d = dist[cur]

            for m in moves:
                nxt = apply_move(m, cur)  # 关键：使用传入的 map
                if dist[nxt] == -1:
                    dist[nxt] = d + 1
                    queue.append(nxt)

        return dist


@dataclass(frozen=True)
class QuotientMove:
    """一个群作用在 quotient 空间上的“元素”"""
    cubie_move: CubieMove  # 仅用于 replay/debug，一个合法代表（保留） replay 得到真实 CubieState

    def act(self, coord):
        raise NotImplementedError

    def replay(self, s: CubieState) -> CubieState:
        """使用底层 cubie_move 重放路径，得到完整 CubieState  从 Phase 路径 replay 到真实状态"""
        return self.cubie_move.act(s)


@dataclass(frozen=True)
class Phase0Coord:
    """G/N ≅ (Z3^7 × Z2^11) quotient 空间 物理层（不可违背）一个“只含守恒律”的物理世界 """
    corner_ori: int  # 0 .. 3^7 - 1
    edge_ori: int  # 0 .. 2^11 - 1

    @property
    def key(self) -> tuple:
        return self.corner_ori, self.edge_ori

    @classmethod
    def project(cls, s: CubieState) -> 'Phase0Coord':
        """from_cubie,投影到 Phase-1 坐标空间"""
        return cls(
            corner_ori=s.corner_ori_coord(),
            edge_ori=s.edge_ori_coord(),
        )

    def __eq__(self, other):
        if not isinstance(other, Phase0Coord):
            return NotImplemented
        return self.corner_ori == other.corner_ori and self.edge_ori == other.edge_ori

    def is_solved(self) -> bool:
        return self.corner_ori == 0 and self.edge_ori == 0

    def heuristic(self) -> int:
        '''
        weak heuristic pruning table：CO × EO
        '''
        co_bits = CubieState.decode_corner_ori(self.corner_ori)  # list of 7 ints 0~2
        eo_bits = CubieState.decode_edge_ori(self.edge_ori)  # list of 11 ints 0~1

        corner_h = np.count_nonzero(co_bits)
        edge_h = np.count_nonzero(eo_bits)

        return max(corner_h, edge_h)


@dataclass(frozen=True)
class Phase0Action(QuotientMove):
    """Phase0Action ⊂ End(Phase0Coord) 群作用的投影"""
    corner_ori_map: np.ndarray  # shape (3^7,), permutation of 0..6
    edge_ori_map: np.ndarray  # shape (2^11,), permutation of 0..10
    cubie_move: CubieMove  # 一个合法代表（保留）

    def act(self, c: Phase0Coord) -> Phase0Coord:
        return Phase0Coord(
            corner_ori=int(self.corner_ori_map[c.corner_ori]),
            edge_ori=int(self.edge_ori_map[c.edge_ori])
        )

    @classmethod
    def phi(cls, m: CubieMove) -> "Phase0Action":
        solved = CubieState.solved()
        corner_map = np.zeros(3 ** 7, np.int32)
        # -------- corner ori map --------
        for i in range(3 ** 7):
            s = solved.with_(corners_ori=CubieState.decode_corner_ori(i))
            corner_map[i] = m.act(s).corner_ori_coord()  # 必须保证完全遵循群规则
        # -------- edge ori map --------
        edge_map = np.zeros(2 ** 11, np.int32)
        for i in range(2 ** 11):
            s = solved.with_(edges_ori=CubieState.decode_edge_ori(i))
            edge_map[i] = m.act(s).edge_ori_coord()

        return cls(
            corner_ori_map=corner_map,
            edge_ori_map=edge_map,
            cubie_move=m
        )

    def __eq__(self, other):
        if not isinstance(other, Phase0Action):
            return NotImplemented
        return (
                np.array_equal(self.corner_ori_map, other.corner_ori_map)
                and np.array_equal(self.edge_ori_map, other.edge_ori_map)
        )


@dataclass(frozen=True)
class Phase1Coord:
    """保证方向正确 + slice 边在中层"""
    corner_ori: int  # 0 .. 3^7 - 1
    edge_ori: int  # 0 .. 2^11 - 1

    ud_slice: int  # 0..494,来自 m.act(SOLVED) 在组合空间的像,区分工程 Kociemba/群论 quotient

    @classmethod
    def project(cls, s: CubieState) -> 'Phase1Coord':
        """from_cubie,投影到 Phase-1 坐标空间"""
        return cls(
            corner_ori=s.corner_ori_coord(),
            edge_ori=s.edge_ori_coord(),
            ud_slice=s.ud_slice_coord(),
        )

    @classmethod
    def solved(cls) -> "Phase1Coord":
        return cls(0, 0, CubieState.solved_ud)  # 69

    def is_solved(self) -> bool:
        '''
        is_phase1_solved: coord.co == 0 and coord.eo == 0
        UD-slice membership = 0  注意：不是 quotient，只是 goal 条件
        Phase-1 只解决“进入 G₁”，不解决“进入 solved coset”
        coord 版不检查 slice separation,slice separation 已经体现在 allowed move set + heuristic 中
        '''
        return self.corner_ori == 0 and self.edge_ori == 0 and self.ud_slice == CubieState.solved_ud

    @property
    def key(self) -> tuple:
        return self.corner_ori, self.edge_ori, self.ud_slice

    @property
    def label(self) -> str:
        co, eo, uds = self.key
        return f"CO{co}|EO{eo}|UD{uds}"

    def apply(self, m: "Phase1Action") -> "Phase1Coord":
        return m.act(self)


@dataclass(frozen=True)
class Phase1Action(Phase0Action):
    """
    Phase-1 坐标映射：把 CubieMove 投影到 Phase-1 子空间（CO × EO × UD-slice membership）
    满足：π(s ∘ m) = π(s) ∘ φ(m)   （右作用语义）
    φ : CubieMove → Phase1Action 是同态映射
    用于 pruning table 生成和 IDA*/BFS 中的坐标转移
    """
    # UD-slice membership 的组合置换
    # 作用在 C(12,4) 编码空间上,并不存在一个与 EP 无关的、真正的函数
    ud_slice_map: np.ndarray  # shape (495,), permutation of 0..494

    def act(self, c: Phase1Coord) -> Phase1Coord:
        """
        Phase1Action.act 只作为“坐标映射”，不作为群作用,整数置换,只计算 CO, EO, UD 的变化
        此方法仅用于坐标投影，不是 cubie 层面的群作用
        """
        return Phase1Coord(
            corner_ori=int(self.corner_ori_map[c.corner_ori]),  # apply_corner_ori
            edge_ori=int(self.edge_ori_map[c.edge_ori]),  # apply_edge_ori
            ud_slice=int(self.ud_slice_map[c.ud_slice]),  # state.ud_slice_coord()
        )

    def compose(self, other: "Phase1Action") -> "Phase1Action":
        """
        坐标映射右作用复合：self ∘ other  先应用 other（右边），再应用 self（左边）
        (self ∘ other).act(c) = self.act(other.act(c))
        """
        return Phase1Action(
            corner_ori_map=self.corner_ori_map[other.corner_ori_map],
            edge_ori_map=self.edge_ori_map[other.edge_ori_map],
            ud_slice_map=self.ud_slice_map[other.ud_slice_map],
            cubie_move=self.cubie_move.compose(other.cubie_move)  # 为了 replay / debug
        )

    @classmethod
    def lift(cls, p0: Phase0Action) -> "Phase1Action":
        m = p0.cubie_move
        solved = CubieState.solved()
        # -------- ud slice map --------
        # 构造一个“只有 slice membership 不同”的状态,只要求“slice 成员正确”， 不要求 slice 内部的排列是对的,依赖 edge permutation 的操作
        slice_map = np.zeros(495, dtype=np.int16)
        for coord in range(495):
            s = solved.with_(edges_perm=CubieState.decode_ud_slice(coord))
            # assert s.ud_slice_coord() == coord < 495
            out = m.act(s)
            slice_map[coord] = out.ud_slice_coord()

        return cls(
            corner_ori_map=p0.corner_ori_map,
            edge_ori_map=p0.edge_ori_map,
            ud_slice_map=slice_map,
            cubie_move=m
        )

    @classmethod
    def phi(cls, m: CubieMove) -> "Phase1Action":
        """
        从底层 CubieMove 生成 Phase-1 坐标映射
        保证：对于任意 s，φ(m).act(π(s)) == π(m.act(s))
        """
        return cls.lift(Phase0Action.phi(m))


@dataclass(frozen=True)
class Phase2Coord:
    """处理角块、非 slice 边、slice 内部排列
    Invariants:
    - corner_ori = 0
    - edge_ori = 0
    - UD-slice membership fixed
    - corner parity = edge parity (guaranteed by Phase-1.75)

    Components:
    - corner_perm ∈ A8 (8! / 2)
    - edge_perm ∈ S8 (non-slice edges)
    - ud_slice_perm ∈ S4
    """
    corner_perm: int  # 0 .. 40319 (8! / 2, 去掉整体 parity)
    edge_perm: int  # 0 .. 40319 (8! 只取非-slice edges 8 条)
    ud_slice_perm: int  # 0 .. 23  (4!) 局部自由度

    # edge_parity: int  # Z2

    @classmethod
    def project(cls, s: CubieState) -> "Phase2Coord":
        """
        必须假设 s ∈ G₁ s.is_phase1_solved,Phase-2 只在 G₁ 内搜索
        使用的是 降维后的坐标群
        """
        assert s.is_phase1_solved(), f'phase1 not solved:{s}'

        corner_idx = CubieState.encode_perm(s.corners_perm.tolist())

        # 非 slice 边：相对排列
        edge_idx = CubieState.encode_perm_coord(s.edges_perm.tolist(),
                                                positions=CubeBase.NON_SLICE_POSITIONS,
                                                cubies=CubieState.non_slice_edges())

        ud_idx = CubieState.encode_perm_coord(s.edges_perm.tolist(),
                                              positions=CubeBase.SLICE_POSITIONS,
                                              cubies=CubieState.ud_slice_edges())

        return cls(corner_idx, edge_idx, ud_idx)

    def is_solved(self) -> bool:
        """角块排列复原.非 slice 边排列复原,slice 边排列复原"""
        return self.corner_perm == 0 and self.edge_perm == 0 and self.ud_slice_perm == 0

    @classmethod
    def solved(cls) -> "Phase2Coord":
        return cls(0, 0, 0)

    @property
    def key(self) -> tuple:
        return self.corner_perm, self.edge_perm, self.ud_slice_perm

    def apply(self, m: "Phase2Action") -> "Phase2Coord":
        return m.act(self)

    def heuristic(self) -> int:
        # decode corner/edge/ud-slice permutation
        corners = CubieState.decode_perm(self.corner_perm, 8)
        edges = CubieState.decode_perm(self.edge_perm, 8)
        ud_edges = CubieState.decode_perm(self.ud_slice_perm, 4)

        # 保守估计：统计不在原位的数量
        h_corners = sum(1 for i, c in enumerate(corners) if c != i)
        h_edges = sum(1 for i, e in enumerate(edges) if e != i)
        h_ud = sum(1 for i, u in enumerate(ud_edges) if u != i)

        # Phase2 中最少需要的 move 至少等于最大的错位数
        return max(h_corners, h_edges, h_ud)


@dataclass(frozen=True)
class Phase2Action(QuotientMove):
    corner_perm_map: np.ndarray  # shape (40320,)
    edge_perm_map: np.ndarray  # shape (40320,)
    ud_slice_perm_map: np.ndarray  # shape (24,)

    def act(self, c: Phase2Coord) -> Phase2Coord:
        """DFS 用, m ⋅ s """
        return Phase2Coord(
            corner_perm=int(self.corner_perm_map[c.corner_perm]),
            edge_perm=int(self.edge_perm_map[c.edge_perm]),
            ud_slice_perm=int(self.ud_slice_perm_map[c.ud_slice_perm]),
        )

    def compose(self, other: "Phase2Action") -> "Phase2Action":
        """
        self ∘ other
        """
        return Phase2Action(
            corner_perm_map=self.corner_perm_map[other.corner_perm_map],
            edge_perm_map=self.edge_perm_map[other.edge_perm_map],
            ud_slice_perm_map=self.ud_slice_perm_map[other.ud_slice_perm_map],
            cubie_move=self.cubie_move.compose(other.cubie_move),
        )

    def __eq__(self, other):
        if not isinstance(other, Phase2Action):
            return NotImplemented
        return (
                np.array_equal(self.corner_perm_map, other.corner_perm_map)
                and np.array_equal(self.edge_perm_map, other.edge_perm_map)
            # and np.array_equal(self.ud_slice_perm_map, other.ud_slice_perm_map)
        )

    @classmethod
    def phi(cls, m: CubieMove) -> "Phase2Action":
        """
        直接从 CubieMove 诱导,坐标转换
        G₂ = ⟨U, D, R², L², F², B²⟩
        """
        solved = CubieState.solved()

        def induce_corner_perm_map() -> np.ndarray:
            """诱导角块置换表 (8! = 40320)"""
            size = factorial(8)  # 40320
            perm_map = np.zeros(size, dtype=np.int32)

            for idx in range(size):
                # decode 成 corner permutation
                rel_perm = CubieState.decode_perm(idx, 8)  # 0~7
                corner_perm = np.array(rel_perm, dtype=np.int8)  # shape (8,) 直接就是 cubie id
                s = solved.with_(corners_perm=corner_perm)
                out = m.act(s)
                perm_map[idx] = CubieState.encode_perm(out.corners_perm.tolist())  # perm8: shape (8,), values 0..7

            return perm_map

        def induce_edge_perm_map(positions: list[int], cubies: list[int] | tuple[int]) -> np.ndarray:
            """
            诱导边块子置换表（非 slice 8! 或 slice 4!）  8! = 40320
            cubies 编号集合（piece space）NON_SLICE_EDGES/UD_SLICE_EDGES
            """
            n = len(positions)
            size = factorial(n)  # 40320/24
            # assert n == len(cubies) and factorial(n) == size
            perm_map = np.zeros(size, dtype=np.int32 if n == 8 else np.int8)
            cubie_to_rel = {cubie: i for i, cubie in enumerate(cubies)}

            for idx in range(size):
                rel_perm = CubieState.decode_perm(idx, n)  # 0 ~ n-1,0..7/0..3
                actual = [cubies[i] for i in rel_perm]  # perm8/perm4,真实 cubie id,values in cubies
                # 嵌回到 solved 的完整状态
                new_edges_perm = solved.edges_perm.copy()
                for pos, cubie in zip(positions, actual):  # slice 保持 solved 顺序
                    new_edges_perm[pos] = cubie

                out = m.act(solved.with_(edges_perm=new_edges_perm))

                out_rel = [cubie_to_rel[int(out.edges_perm[p])] for p in positions]  # -> 0..7/ 0..3, 长度 8,
                perm_map[idx] = CubieState.encode_perm(out_rel)

            return perm_map

        # 三个坐标分别调用
        corner_perm_map = induce_corner_perm_map()
        edge_perm_map = induce_edge_perm_map(positions=CubeBase.NON_SLICE_POSITIONS,
                                             cubies=CubieState.non_slice_edges())
        ud_slice_perm_map = induce_edge_perm_map(positions=CubeBase.SLICE_POSITIONS,
                                                 cubies=CubieState.ud_slice_edges())

        return cls(
            corner_perm_map=corner_perm_map,
            edge_perm_map=edge_perm_map,
            ud_slice_perm_map=ud_slice_perm_map,
            cubie_move=m
        )


@dataclass(frozen=True)
class Phase15Coord:
    """
    Phase-1.5 quotient 坐标
    在 G / H₁ 的基础上，再 quotient 一个 H₂ ⊂ H₁
    Phase1 已经 quotient 掉 orientation 与 slice membership。
    Phase1.5 再 quotient 掉 slice 内排列与角块内部排列，只保留 coset 与 edge parity
    状态空间大小 3360,压缩状态
    """
    slice_perm: int  # 4! = 24 slice_perm ∈ [0,24)
    corner_coset: int  # quotient 掉 U-layer + D-layer 内部的排列 8! / (4!·4!) = C(8,4)  ∈ [0,70)
    parity: int  # Z2 守恒 0 / 1,edge_parity' = edge_parity XOR Δ(m) / edge_parity XOR corner_parity

    @property
    def index(self) -> int:
        return ((self.slice_perm * 70 + self.corner_coset) << 1) | self.parity

    @classmethod
    def from_index(cls, i: int) -> "Phase15Coord":
        N_CORNER_COSET = 70
        parity = i & 1
        i >>= 1
        corner_coset = i % N_CORNER_COSET
        slice_perm = i // N_CORNER_COSET
        return cls(slice_perm=slice_perm, corner_coset=corner_coset, parity=parity)

    @property
    def key(self) -> tuple:
        return self.slice_perm, self.corner_coset, self.parity

    @classmethod
    def project(cls, s: CubieState) -> "Phase15Coord":
        """ encode 状态空间裁剪 """
        # 1. slice_perm（Phase1 已保证 membership）
        slice_perm = CubieState.encode_ud_slice_perm(s.edges_perm.tolist())  # 0..23

        # 2. corner_coset（U 层是哪 4 个 corner）
        corner_coset = CubieState.encode_corner_coset(s.corners_perm.tolist())
        # 3. edge parity
        edge_parity = s.edge_parity  # 0/1

        return cls(
            slice_perm=slice_perm,
            corner_coset=corner_coset,
            parity=edge_parity
        )

    def is_solved(self) -> bool:
        return self.index == 0  # return self.slice_perm == 0 and self.corner_coset == 0 and self.parity == 0

    @staticmethod
    def tight(s: CubieState) -> "Phase15Coord":
        """
        corner parity 还在 Phase-2 可达 kernel
        在 Phase-2 中，corner parity 与 edge parity 必须匹配
        """
        p = Phase15Coord.project(s)
        return Phase15Coord(slice_perm=p.slice_perm, corner_coset=p.corner_coset,
                            parity=s.edge_parity ^ s.corner_parity)

    def decode(self) -> CubieState:
        """
         canonical representative
         仅用于 φ 构造 不用于搜索 replay,搜索过程中 从不调用 decode 再 project
        """
        s = CubieState.solved()
        # 1. slice_perm
        # 2. corner_coset（只放层，不管层内排列）
        s.with_(edges_perm=CubieState.canonical_ud_slice_edges(self.slice_perm),
                corners_perm=CubieState.canonical_corner_coset(self.corner_coset))

        # 3. edge parity
        if s.edge_parity != self.parity:  # decode 出来的状态，其 parity 必须天然一致
            raise AssertionError("Illegal Phase15Coord: parity mismatch")

        return s


@dataclass(frozen=True)
class Phase15Action(QuotientMove):
    """
    Phase-1.5 上的群作用投影
    φ : CubieMove → End(Phase15Coord)
    """
    slice_perm_map: np.ndarray  # shape (24,)
    corner_coset_map: np.ndarray  # shape (70,)
    edge_parity_map: np.ndarray  # shape (2,)

    def act(self, c: Phase15Coord) -> Phase15Coord:
        return Phase15Coord(
            slice_perm=int(self.slice_perm_map[c.slice_perm]),
            corner_coset=int(self.corner_coset_map[c.corner_coset]),
            parity=int(self.edge_parity_map[c.parity]),
        )

    def act_index(self, idx: int) -> int:
        c = Phase15Coord.from_index(idx)
        return self.act(c).index

    @classmethod
    def phi(cls, m: CubieMove) -> "Phase15Action":
        # 固定 orientation & slice membership
        # 枚举 quotient coord
        # m.act(s) → re-encode
        solved = CubieState.solved()

        slice_perm_map = np.zeros(24, dtype=np.int8)
        corner_coset_map = np.zeros(70, dtype=np.int16)
        edge_parity_map = np.zeros(2, dtype=np.int8)

        # slice_perm
        for i in range(24):
            # membership 固定 + slice 内排列 = i
            s = solved.with_(edges_perm=CubieState.canonical_ud_slice_edges(i))
            s2 = m.act(s)
            slice_perm_map[i] = CubieState.encode_ud_slice_perm(s2.edges_perm.tolist())

        # corner_coset
        for i in range(70):
            s = solved.with_(corners_perm=CubieState.canonical_corner_coset(i))
            s2 = m.act(s)
            corner_coset_map[i] = CubieState.encode_corner_coset(s2.corners_perm.tolist())

        # edge_parity
        delta = m.edge_parity_delta  # 0 or 1 群论一致性条件
        for p in (0, 1):
            edge_parity_map[p] = p ^ delta

        return cls(
            slice_perm_map=slice_perm_map,
            corner_coset_map=corner_coset_map,
            edge_parity_map=edge_parity_map,
            cubie_move=m
        )


class StickerMove:
    def __init__(self, perm: np.ndarray):
        """对 sticker index 的一维置换"""
        self.perm = perm.astype(np.int32)  # 一维贴纸置换

    def act(self, sticker_idx: np.ndarray) -> np.ndarray:
        """右作用：new[i] = old[perm[i]] new_state[i] = old_state[perm[i]]"""
        flat = sticker_idx.reshape(-1)
        return flat[self.perm].reshape(sticker_idx.shape)

    def replay(self, s: StickerCube | np.ndarray) -> StickerCube:
        """等价于 act"""
        arr = s if isinstance(s, np.ndarray) else s.cube  # get_state[self.perm]
        return StickerCube(state=self.act(arr), n=arr.shape[1])

    @staticmethod
    def act_moves(state: np.ndarray, moves: list[CubieMove]) -> np.ndarray:
        '''
        replay_cubie_moves,CubieMove → StickerMove → replay
        state' = state ∘ m1 ∘ m2 ∘ ... ∘ mn
        '''
        n = state.shape[1]
        flat = state.reshape(-1).copy()
        for m in moves:
            perm = m.to_sticker_perm(n)  # 返回正向置换
            flat = flat[perm]
        return flat.reshape(6, n, n)

    @classmethod
    def from_rotation(cls, n: int, axis: int, layer: int, direction: int):
        """
        CubieMove ⊂ StickerMove,  CubieMove → StickerMove
        perm[i] = j  表示 new_flat[i] = old_flat[j]
        perm[i] = j 表示 i 号贴纸 → j 号位置
        """
        state_idx = np.arange(6 * n * n, dtype=np.int32).reshape(6, n, n)
        CubeBase.rotate_core(state_idx, axis, layer, direction)
        return cls(perm=state_idx.reshape(-1))  # flatten

    def center_perm(self, n: int) -> np.ndarray:
        """
           返回一维 center_perm：
           index = center sticker 全局编号
           value = 被 move 后送来的 sticker 编号
        """
        perm = self.perm.reshape(6, n, n)
        centers = CubeBase.get_center_rings(n)

        flat = []
        for fidx, rings in enumerate(centers):
            for ring in rings:
                for r, c, _ in ring:
                    flat.append(perm[fidx, r, c])
        return np.array(flat, dtype=np.int32)

    @classmethod
    def identity(cls, n: int) -> "StickerMove":
        return cls(perm=np.arange(6 * n * n, dtype=np.int32))

    def inverse(self) -> "StickerMove":
        inv = np.empty_like(self.perm)
        inv[self.perm] = np.arange(len(self.perm))
        return StickerMove(inv)

    def compose(self, other: "StickerMove") -> "StickerMove":
        # self ∘ other,先 other，再 self
        return StickerMove(self.perm[other.perm])


class CycleLibrary:
    """
    合法群元素的组合模板,构造一个群元素，它在贴纸表示下呈现为 cycle
    """

    @staticmethod
    def cycle3(A, B, P):
        """
        experimental
        在 P(A,B) 定义的位置制造一个 3-cycle
        A, B: 产生 3-cycle 的基元
        P: 定位用的 conjugate
        P · [A, B] · P⁻¹
        """
        base = CubeBase.commutator(A, B)
        return CubeBase.conjugate(P, base)

    @staticmethod
    def verify_cycle(state: np.ndarray, moves: list):
        """
        sticker 3-cycle → 3 或 6 :assert verify_cycle(cube0, cycle) in (3, 6)
        edge 3-cycle → 6（3 条边 × 2 贴纸）
        corner 3-cycle → 9（3 × 3）
        """
        arr = state.copy()
        CubeBase.act_moves(arr, moves)
        diff = np.where(arr != state)
        return len(diff[0])  # np.sum(arr != state)

    @staticmethod
    def sticker_3cycle_base():
        """
        一个已知稳定的贴纸 3-cycle（固定位置）
        支撑很小，parity 合法
        """
        A = [(0, +1, 1)]  # R
        B = [(1, +1, 1)]  # U
        return CubeBase.commutator(A, B)

    @staticmethod
    def edge_3cycle_base():
        """
        3 条边的循环，不翻转方向
        """
        A = [(0, +1, 1)]  # R
        B = [(2, +1, 1)]  # F
        return CubeBase.commutator(A, B)

    @staticmethod
    def corner_3cycle_base():
        """
        3 个角块的循环，不扭转
        """
        A = [(0, +1, 1)]  # R
        B = [(1, +1, 1)]  # U
        C = [(0, +1, -1)]  # R'
        return CubeBase.commutator(A, CubeBase.commutator(B, C))

    @staticmethod
    def at(position_moves: list, base_cycle: list):
        """
        在指定位置制造一个贴纸 3-cycle
        position_moves: 把目标贴纸搬到工作区的 moves
        base_cycle: 已知在固定工作区的 cycle
        cycle_at = P · base · P⁻¹
        """
        return CubeBase.conjugate(position_moves, base_cycle)

    @staticmethod
    def sticker_3cycle(position_moves: list):
        base = CycleLibrary.sticker_3cycle_base()
        return CycleLibrary.at(position_moves, base)

    @staticmethod
    def edge_3cycle(position_moves: list):
        base = CycleLibrary.edge_3cycle_base()
        return CycleLibrary.at(position_moves, base)

    @staticmethod
    def corner_3cycle(position_moves: list):
        base = CycleLibrary.corner_3cycle_base()
        return CycleLibrary.at(position_moves, base)


class CubieBase(CubeBase):
    AXIS_NAME = ('X', 'Y', 'Z')

    def __init__(self, n: int = 3):
        super().__init__(n)
        self.CORNER_REF_AXIS = self.build_corner_reference()
        # assert np.all(self.corner_orientation(self.solved) == 0) ,self.corner_orientation(self.solved)

    def cubie_state(self, state: np.ndarray) -> CubieState:
        """
        sticker_to_cubie_state from_stickers
        state = [
          corners_perm (8)       ∈ [0..7]
          corners_ori  (8)       ∈ [0..2]
          edges_perm   (12)      ∈ [0..11]
          edges_ori    (12)      ∈ [0..1]
        ] 最小充分状态
        定义并修正 ori
        """
        assert self.n == state.shape[1]
        edges_perm, edges_ori = self.edge_ids_ori(state)
        corners_perm, corners_ori = self.corner_ids_ori(state)
        # 修正 parity
        corner_parity = self.permutation_parity(corners_perm)
        edge_parity = self.permutation_parity(edges_perm)
        if corner_parity != edge_parity:
            print(f'Fixing odd parity: corner {corner_parity} != edge {edge_parity}')
            non_slice = CubieState.non_slice_edges()
            i, j = non_slice[-2], non_slice[-1]  # 交换最后两条非 slice 边
            edges_perm[i], edges_perm[j] = edges_perm[j], edges_perm[i]
            assert self.permutation_parity(edges_perm) == corner_parity, "Parity fix failed"

        return CubieState(
            corners_perm=corners_perm,
            corners_ori=corners_ori,  # self.corner_orientation(state)
            edges_perm=edges_perm,
            edges_ori=edges_ori,
        )

    @class_status('参考方法')
    def build_cubie_move_from_stickers(self, state_arr: np.ndarray, axis: int, layer: int, direction: int) -> CubieMove:
        """
        构建 CubieMove：orientation 信息被抹平过,只作为“验证 / 校准工具” 不做 parity 修正,不保证 is_solvable
        不依赖贴纸索引顺序来算 delta，直接从 CubieState 计算。
        - cube: 当前 Cube 对象，需提供 cubie_state() 和 rotate_state()
        - axis, layer, direction: move 定义: (axis, layer, dir)
        """
        s0: CubieState = self.cubie_state(state_arr)  # 原始 CubieState
        rotated_arr = self.rotate_state(state_arr, axis, layer, direction)  # 贴纸级旋转
        s1: CubieState = self.cubie_state(rotated_arr)  # 旋转后状态

        mv = CubieMove.build(s0, s1)

        assert s1.is_solvable()
        assert mv.edges_ori_delta.sum() % 2 == 0
        assert mv.corners_ori_delta.sum() % 3 == 0, f'{axis},{mv.corners_ori_delta}'

        return mv

    @class_status('参考方法')
    def build_primitive_moves(self) -> dict[tuple, CubieMove]:
        """
        生成所有 primitive move 对应的 CubieMove,手工定义 / 程序生成（基于坐标）
        sticker rotation → CubieState → delta (right action)
        m.act(s) == rotate_state(s)
         | 项目       | corner | edge  |
         | --------- | ------ | ----- |
         | 群         | Z₃     | Z₂    |
         | reference | U/D 色  | F/B 色 |
         | U/D move  | 不变    | 不变   |
         | R/L move  | 变      | 不变   |
         | F/B move  | 变      | 变     |
        """
        prim_moves = {}
        for move in self.basic_generators():
            prim_moves[move] = self.build_cubie_move_from_stickers(self.solved, *move)
        return prim_moves  # self.PRIM_MOVES

    @staticmethod
    def build_phase_graph(start: Phase0Coord | Phase1Coord | Phase2Coord, max_depth: int = 2, max_nodes: int = 10000):
        """
        Schreier graph
        nodes: set[Phase1Coord]
        edges: list[(src, label, dst)]
        I explicitly constructed the Phase-1 Schreier graph of the Rubik’s Cube quotient
        and verified generator degeneracies, involutions, and identity actions.
        """
        moves = CubieMove.phase0_moves if isinstance(start, Phase0Coord) \
            else CubieMove.phase1_moves if isinstance(start, Phase1Coord) \
            else CubieMove.phase2_moves
        queue = deque([(start, 0)])
        nodes = {start.key: start}
        edges = []
        while queue and len(nodes) < max_nodes:
            s, d = queue.popleft()
            if d >= max_depth:
                continue

            for label, m in moves.items():  # (axis,side,dir)
                s2 = m.act(s)
                k2 = s2.key
                edges.append((s.key, label, k2))
                if k2 not in nodes:
                    nodes[k2] = s2  # d+1
                    queue.append((s2, d + 1))

        return nodes, edges

    @staticmethod
    def build_phase15_pruning() -> np.ndarray:
        N_PHASE15 = 24 * 70 * 2
        PHASE15_MOVES: list[Phase15Action] = list(CubieMove.phase15_moves.values())
        INF = np.int8(127)
        dist = np.full(N_PHASE15, INF, dtype=np.int8)

        solved = Phase15Coord.project(CubieState.solved()).index
        dist[solved] = 0

        queue = np.empty(N_PHASE15, dtype=np.int16)
        head = 0
        tail = 1
        queue[0] = solved

        while head < tail:
            cur = int(queue[head])
            head += 1

            d = dist[cur]
            nd = np.int8(d + 1)

            for act in PHASE15_MOVES:
                nxt = act.act_index(cur)
                if dist[nxt] == INF:
                    dist[nxt] = nd
                    queue[tail] = nxt
                    tail += 1

        return dist

    @classmethod
    def build_pruning_table(cls):
        # 离线构建器
        # PRIM_MOVES = CubieMove.prim_moves()
        PHASE0_MOVES: list[Phase0Action] = list(CubieMove.phase0_moves.values())
        PHASE1_MOVES: list[Phase1Action] = list(CubieMove.phase1_moves.values())
        PHASE2_MOVES: list[Phase2Action] = list(CubieMove.phase2_moves.values())
        print('build_pruning_table', len(PHASE1_MOVES), len(PHASE2_MOVES))
        import os
        if os.path.exists('data/phase1_pruning.npz'):
            data = np.load("data/phase1_pruning.npz")
            cls.CO_EO_PRUNE = data["CO_EO"]
            cls.UD_PRUNE = data["UD"]
        else:
            # PHASE0_PRUNE 两维联合（CO × EO），固定 UD-slice = 0,深度 ≤ 7
            cls.CO_EO_PRUNE = CubieMove.build_pruning_table(
                moves=PHASE0_MOVES,  # PHASE1_MOVES
                apply_move=lambda m, c: (m.corner_ori_map[c[0]], m.edge_ori_map[c[1]]),
                start_coord=(0, 0),
                table_shape=(2187, 2048)  # (3**7, 2**11)
            )  # dist((corner_ori, edge_ori)) ≈ 4.5M
            # 最大深度 7
            cls.UD_PRUNE = CubieMove.build_pruning_table(
                moves=PHASE1_MOVES,  # List[Phase1Action]
                apply_move=lambda m, ud: m.ud_slice_map[ud],
                start_coord=CubieState.solved_ud,
                table_shape=495
            )  # dist(ud_slice) C(12,4)=495
            np.savez(
                "data/phase1_pruning.npz",
                CO_EO=cls.CO_EO_PRUNE,  # (2187, 2048)
                UD=cls.UD_PRUNE  # (495,)
            )

        if os.path.exists('data/phase2_pruning.npz'):
            data = np.load("data/phase2_pruning.npz")
            cls.CO_PRUNE = data["CO"]
            cls.EDGE_PRUNE = data["EDGE"]
            cls.SLICE_PRUNE = data["SLICE"]
        else:
            # Corner pruning
            cls.CO_PRUNE = CubieMove.build_pruning_table(
                moves=PHASE2_MOVES,
                apply_move=lambda m, idx: m.corner_perm_map[idx],
                start_coord=CubieState.encode_perm(list(range(8))),
                table_shape=40320  # 8!
            )
            # Non-slice edge pruning
            cls.EDGE_PRUNE = CubieMove.build_pruning_table(
                moves=PHASE2_MOVES,
                apply_move=lambda m, idx: m.edge_perm_map[idx],
                start_coord=CubieState.encode_perm(list(range(8))),  # 0
                table_shape=40320
            )
            # Slice pruning,进去收益极小，但复杂度翻倍
            cls.SLICE_PRUNE = CubieMove.build_pruning_table(
                moves=PHASE2_MOVES,
                apply_move=lambda m, idx: m.ud_slice_perm_map[idx],
                start_coord=0,
                table_shape=24
            )
            np.savez(
                "data/phase2_pruning.npz",
                CO=cls.CO_PRUNE,  # 40320
                EDGE=cls.EDGE_PRUNE,  # 40320
                SLICE=cls.SLICE_PRUNE
            )

        cls.PHASE15_PRUNE = cls.build_phase15_pruning()
        # h_15(s) = dist[Phase15Coord.project(s).index]

        print(cls.CO_PRUNE.max())  # 最大深度 ≈ 11  14
        print(cls.EDGE_PRUNE.max())  # ≈ 10~11  10
        print(np.sum(cls.CO_PRUNE >= 0))  # < 40320 一半以上的状态必然是 -1  40320
        print(cls.CO_EO_PRUNE.shape, cls.CO_PRUNE.shape, cls.EDGE_PRUNE.shape, cls.PHASE15_PRUNE.shape)

    @classmethod
    def phase1_search(cls, state: CubieState, depth_limit: int = 8) -> list[tuple] | None:
        """
        IDA* 搜索 Phase-1：目标是进入 G₁（EO=0, CO=0, UD-slice 在中层）
        使用右作用 CubieMove.act + Phase1Action.project
        BFS 解决 18 个基本转动,直径 diam(G / G₁) ≈ 7，通常 depth_limit=7 或 8 足够
        12:9~12
        """
        PHASE1_MOVES = CubieMove.phase1_moves()

        def dfs(coord, depth: int, last_move: tuple | None):
            h = max(cls.CO_EO_PRUNE[coord.corner_ori, coord.edge_ori],
                    cls.UD_PRUNE[coord.ud_slice])  # admissible phase1_heuristic（reference）
            if depth + h > depth_limit:
                return None

            if coord.is_solved():  # EO=0, CO=0, UD-slice=solved, current_state.is_ud_slice_separated()
                print('phase1_search', depth, h, coord.ud_slice)  # 69,current_state.ud_slice_coord()
                return []  # 判断 phase1 solved

            for k, m in PHASE1_MOVES.items():
                if CubieMove.is_redundant(last_move, k):
                    continue

                next_coord = m.act(coord)
                # next_state = m.replay(current_state) next_coord = Phase1Coord.project(next_state)
                res = dfs(next_coord, depth + 1, k)  # 递归 moves
                if res is not None:
                    return [(k, m)] + res

            return None

        initial_coord = Phase1Coord.project(state)
        return dfs(initial_coord, 0, None)

    @classmethod
    def phase2_search(cls, state: CubieState, depth_limit: int = 14) -> list[tuple] | None:
        """
        10～12个基本转动 diam(G₁ / G₂) = 10
        IDDFS 18 limit ≈ 10~11, 12:14-20
        """
        PHASE2_MOVES = CubieMove.phase2_moves()

        def dfs(coord, depth: int, last_move: tuple | None) -> list[tuple] | None:
            h = max(
                cls.CO_PRUNE[coord.corner_perm],
                cls.EDGE_PRUNE[coord.edge_perm],
                cls.SLICE_PRUNE[coord.ud_slice_perm]
            )
            if depth + h > depth_limit:
                return None

            if coord.is_solved():
                print('phase2_search', depth, h, coord.ud_slice_perm)
                return []

            for k, m in PHASE2_MOVES.items():
                if CubieMove.is_redundant(last_move, k):
                    continue

                next_coord = m.act(coord)
                res = dfs(next_coord, depth + 1, k)
                if res is not None:
                    return [(k, m)] + res

            return None

        initial_coord = Phase2Coord.project(state)
        return dfs(initial_coord, 0, None)

    @classmethod
    def phase15_search(cls, state: CubieState, depth_limit: int = 8) -> list[tuple] | None:
        """
        IDA* 搜索 Phase-1.5
        目标：slice_perm = solved, corner_coset = solved, parity = 0
        move 集：G₁
        """
        PHASE15_MOVES = CubieMove.phase15_moves()

        def dfs(coord: Phase15Coord, depth: int, last_move: tuple | None):
            h = cls.PHASE15_PRUNE[coord.index]
            if depth + h > depth_limit:
                return None

            if coord.is_solved():
                print('phase15_search', depth, h, coord.index)
                return []

            for k, m in PHASE15_MOVES.items():
                if CubieMove.is_redundant(last_move, k):
                    continue

                next_coord = m.act(coord)
                res = dfs(next_coord, depth + 1, k)
                if res is not None:
                    return [(k, m)] + res

            return None

        initial_coord = Phase15Coord.project(state)
        return dfs(initial_coord, 0, None)

    @classmethod
    def solve_kociemba(cls, s: CubieState) -> tuple[list[tuple], CubieMove]:
        """
            Kociemba 两阶段求解：
            Phase1: 解决 EO + CO + UD-slice separation
            Phase2: 解决剩余 CP + EP（在 G1 内）
            返回：(move_keys 序列, 总复合 move)
            G -Phase-1 -> H -Phase-2 -> {e}
            Gₙ
             ├─ Phase-0：orientation only
             ├─ Phase-1：slice class
             ├─ Phase-1.5：center orbits
             └─ Phase-2：permutation class
        """
        assert s.is_solvable(), "Unsolvable cube state"
        moves1, moves2 = None, None
        d = 6  # 9
        while d < 12:  # 9/13
            moves1 = cls.phase1_search(s, d)
            if moves1 is not None:
                break
            print('phase1 depth', d)
            d += 1

        assert moves1 is not None, f"Phase1 failed to solve: {s}"
        mv1, state = CubieMove.act_moves(s, [x[1].cubie_move for x in moves1])
        assert state.is_phase1_solved(), f'Phase1 invalid: len={len(moves1)}, ud_slice={state.ud_slice_coord()}'
        # assert Phase15Coord.project(state).parity == 0

        d = 9
        while d < 19:  # 12,20
            moves2 = cls.phase2_search(state, d)
            if moves2 is not None:
                break
            print('phase2 depth', d)
            d += 1

        mv2, state = CubieMove.act_moves(state, [x[1].cubie_move for x in moves2])
        assert state.is_phase2_solved(), f'Phase2 invalid: len={len(moves2)}, state={state}'
        assert state == s.solved(), f'Not fully solved,state:{state}'
        return [a for a, _ in moves1 + moves2], mv1.compose(mv2)

    def solve_sticker(self, state: np.ndarray) -> list[tuple]:
        if not hasattr(self, 'CO_EO_PRUNE'):
            self.build_pruning_table()
        s = self.cubie_state(state)
        moves, mv = self.solve_kociemba(s)
        act = [(axis, side * self.mid, dir) for axis, side, dir in moves]
        s0 = state.copy()
        self.act_moves(s0, act)
        print(self.is_solved(s0), s0, '\n', self.cubie_state(s0), '\n', s)
        return act

    @class_status('参考实现')
    def permutation_parity_ok(self, state):
        corner_coords = self.corner_coords(self.n)
        edge_coords = self.edge_coords(self.n)
        solved_corners = [self.get_data(self.solved, c) for c in corner_coords]
        solved_edges = [self.get_data(self.solved, e) for e in edge_coords]

        def corner_perm(state):
            perm = []
            for c in corner_coords:
                cid = self.get_data(state, c)
                perm.append(solved_corners.index(cid))
            return perm

        def edge_perm(state):
            perm = []
            for e in edge_coords:
                eid = self.get_data(state, e)
                perm.append(solved_edges.index(eid))
            return perm

        return self.permutation_parity(corner_perm(state)) == self.permutation_parity(edge_perm(state))

    @class_status('参考方法')
    def corner_orientation(self, state: np.ndarray) -> np.ndarray:
        """
        返回每个角块的朝向 0,1,2 (Z3),只看 U / D 颜色在哪个贴纸位置（Z₃）,全局状态,cubie 自身坐标系 vs 世界坐标系 的相对关系
        朝向定义：需要旋转几次（沿角块到中心的径向）才能使 U/D 颜色回到“标准位置”（即 cycle[0] 位置）
        每个角块有 3 种姿态,orientation 的定义 必须对所有 move 保持一致,U/D 是魔方的“上下极轴”
        角块的朝向信息,在当前架构(在贴纸级表示)下，corner_orientation 是“冗余状态”，已经被贴纸的空间位置完全决定,当前魔方状态：贴纸级真实旋转
        ---rotate 不会改变它，因为 rotate 已经体现在贴纸位置里了,用于验证或接口兼容传统求解器时才需要。
        """
        U, D = self.face_idx['U'], self.face_idx['D']  # face_to_color 隐含假设：颜色编号 == 面编号 0,1
        corner_pos = self.corner_coords(self.n)
        ori = np.zeros(8, dtype=np.int8)
        for i, corner in enumerate(corner_pos):
            dst_colors = [state[f, r, c] for f, r, c in corner]
            # 找到 U/D 色在角块内部的索引，哪个物理位置（0,1,2）
            ud_idx = next(j for j, c in enumerate(dst_colors) if c in (U, D))  # physical
            cycle = self.corner_face_cycle[i]  # 映射到标准 cycle_faces 中的逻辑位置
            ud_logical_idx = next(j for j, f in enumerate(cycle) if f in ('U', 'D'))  # U/D 始终在索引 0！
            # orientation = 旋转次数使 U/D 色在 cycle[0] 位置(内部参考顺序)
            ori[i] = (ud_idx - ud_logical_idx) % 3  # 注意：方向要正确 —— 通常是顺时针为正 ori[i] = (3 - ud_idx) % 3
        return ori

    @class_status('已废弃')
    def corner_orientation_delta(self, s0: np.ndarray, s1: np.ndarray) -> np.ndarray:
        """
        局部增量,生成元,这个 move 对“被搬到 i 位置的角块”额外施加了多少扭转
        perm 表示 piece 的重排，ori_delta 表示局部坐标系里的 twist / flip
        需要 s0 (原始), s1 (旋转后) 来算 delta
        face_to_color 隐含假设：颜色编号 == 面编号 0,1(这是关键假设）
        """
        U, D = self.face_idx['U'], self.face_idx['D']
        corner_pos = self.corner_coords(self.n)
        corner_perm, corner_ori = self.corner_ids_ori(s1)
        ori_delta = np.zeros(8, dtype=np.int8)
        for i, corner in enumerate(corner_pos):
            dst_colors = [s1[f, r, c] for f, r, c in corner]
            src_pos = np.where(corner_perm == i)[0][0]  # 这个块在 s0 中的位置（通过 perm 反推）
            # orientation delta（Z₃）
            src_colors = [s0[f, r, c] for f, r, c in corner_pos[src_pos]]
            # 找到 U/D 色在角块内部的索引，哪个物理位置（0,1,2）
            dst_ud = next(j for j, c in enumerate(dst_colors) if c in (U, D))
            src_ud = next(j for j, f in enumerate(src_colors) if f in (U, D))
            ori_delta[i] = (dst_ud - src_ud) % 3
        return ori_delta

    @class_status('参考方法')
    def edge_orientation(self, state: np.ndarray) -> np.ndarray:
        """
        稳定子内的剩余自由度
        返回 12 条边块的朝向 0/1，shape = (12,), edge orientation 定义基于：颜色 == 面编号 == 几何语义
        规则：
            orientation = 0 时，优先让 U/D 颜色在 U/D 面上
            赤道边（无 U/D）：orientation = 0 时 F/B 颜色在 F/B 面上
        定义：
          - U/D 色边：在 U/D 面为 0，否则为 1
          - F/B 色边：在 F/B 面为 0，否则为 1
          - color == face_id
          - 标准 Singmaster U/D, F/B 轴定义
        """
        U, D, F, B = self.face_idx['U'], self.face_idx['D'], self.face_idx['F'], self.face_idx['B']
        ori = np.zeros(12, dtype=np.uint8)  # edges_ori (12)
        for i, edge_def in enumerate(self.edge_coords(self.n)):
            (f1, r1, c1), (f2, r2, c2) = edge_def
            c1v, c2v = state[f1, r1, c1], state[f2, r2, c2]
            # 找 F 或 B 色 在角块内部的索引，哪个物理位置,edge_flip_index
            if c1v in (U, D):
                ori[i] = 0 if f1 in (U, D) else 1
            elif c2v in (U, D):
                ori[i] = 0 if f2 in (U, D) else 1
            elif c1v in (F, B):  # F/B 色的边（不含 U/D）
                ori[i] = 0 if f1 in (F, B) else 1
            elif c2v in (F, B):
                ori[i] = 0 if f2 in (F, B) else 1
            else:
                ori[i] = 0  # R/L 纯边，默认 0
        return ori  # [0 0 0 0 0 0 0 0 0 0 0 0]

    @class_status('参考方法')
    def build_edge_reference(self):
        """
        ref 被抹掉,投影态不可逆
        为每条 edge 确定 orientation = 0 的 reference sticker
        """
        U = self.face_idx['U']  # 0
        D = self.face_idx['D']  # 1
        F = self.face_idx['F']  # 2
        B = self.face_idx['B']  # 3

        ref = np.empty(12, dtype=np.int8)
        for i, edge in enumerate(self.edge_coords(self.n)):
            (f1, r1, c1), (f2, r2, c2) = edge
            # key = (self.FACES[f1], self.FACES[f2])
            col1 = self.solved[f1, r1, c1]
            col2 = self.solved[f2, r2, c2]

            # 优先选 U/D 颜色
            if col1 in (U, D):
                ref[i] = 0
            elif col2 in (U, D):
                ref[i] = 1
            else:
                ref[i] = 0 if col1 in (F, B) else 1  # 否则选 F/B
        return ref

    @class_status('参考方法')
    def build_corner_reference(self):
        """
        在 solved 状态下，记录每个 corner 的 U/D 色所在轴,corner 只看 U/D 是否在顶部/底部位置
        每个 corner 必须有一个参考循环
        """
        U = self.face_idx['U']
        D = self.face_idx['D']

        def corner_ud_index(colors):
            for i, c in enumerate(colors):
                if c == U or c == D: return i
            raise ValueError("corner missing U/D color")

        ref = np.empty(8, dtype=np.int8)
        for i, corner in enumerate(self.corner_coords(self.n)):
            colors = [self.solved[f, r, c] for f, r, c in corner]
            ref[i] = corner_ud_index(colors)

        return ref  # [0 0 0 0 0 0 0 0]

    @class_status('参考方法')
    def build_corner_reference_axis(self):
        """
        在 solved 状态下，记录每个 corner 的 U/D 色所在轴,corner 只看 U/D 是否在顶部/底部位置
        corner 的 twist 循环 ≠ 空间轴的排列循环.而是“相对于参考循环顺序的位移”,axis 本身没有方向性、也没有旋向。
        """
        U = self.face_idx['U']
        D = self.face_idx['D']

        ref = np.empty(8, dtype=np.int8)
        for i, corner in enumerate(self.corner_coords(self.n)):
            for (f, r, c) in corner:
                cv = self.solved[f, r, c]  # shape (6,n,n)
                if cv in (U, D):
                    axis, _ = self.face_axis[self.FACES[f]]
                    ref[i] = axis
                    break
            else:
                raise RuntimeError(f"Solved corner {i} without U/D color")

        return ref

    @staticmethod
    @class_status('实验用')
    def canonicalize_ud_slice(s: 'CubieState') -> 'CubieState':
        """
        Phase-1 → Phase-2
        phase2_start = canonicalize_ud_slice(phase1_state)
        1. 将所有 UD-slice 边放回标准中层位置（SLICE_POSITIONS）
        2. 内部顺序按 solved 顺序排列
        """
        s = s.clone()

        slice_pos = CubeBase.SLICE_POSITIONS
        non_slice_pos = CubeBase.NON_SLICE_POSITIONS
        slice_cubies = s.ud_slice_edges  # (4,5,6,7)

        original_edges = s.edges_perm.copy()

        # 当前 slice cubies
        slice_edges = sorted(e for e in original_edges if e in slice_cubies)  # 按 solved 顺序
        # 非 slice cubies
        non_slice_cubies = [e for e in original_edges if e not in slice_cubies]

        new_edges = np.zeros(12, dtype=np.int8)  # 重写 edges_perm
        for pos, cubie in zip(slice_pos, slice_edges):  # 放 slice
            new_edges[pos] = cubie

        for pos, cubie in zip(non_slice_pos, non_slice_cubies):  # 放非 slice
            new_edges[pos] = cubie

        s.edges_perm[:] = new_edges
        s.corners_ori[:] = 0
        s.edges_ori[:] = 0

        return s

    @classmethod
    @class_status('参考实现')
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
    cube = CubieBase(n=4)
    cube = CubieBase(n=3)

    print('cr', cube.build_corner_reference())
    print('er', cube.build_edge_reference())

    print(CubieState.non_slice_edges)
    print(CubieState.ud_slice_edges)
    print(CubieState.solved_ud)  # 69
    print(Phase1Coord.project(CubieState.solved()))

    print('rotate_map', cube.build_rotate_map())
    s_i = cube.solved_idx.copy()
    s: CubieState = cube.cubie_state(cube.solved)
    s0 = CubieState.solved()
    assert s == s0

    assert cube.cubie_state(s0.to_stickers(n=3)) == s0

    s_idx = cube.solved_idx.copy()
    s_idx1 = cube.rotate_state(s_idx, 0, 1, 1)

    # 看这个 corner 的 3 个贴纸，来自哪里
    corner = cube.corner_coords(cube.n)[1]
    print([s_idx1[f, r, c] for (f, r, c) in corner])


    def test_all_primitive_moves_solvable(cube):
        """
        cube : 你的 StickerCube / state-index 体系
        s0   : CubieState，对应 s_i0
        s_i0 : 贴纸/索引状态（solved 或任意）
        CubieState 是真值层（ground truth）
        Sticker 只是一个表示（representation）
        """
        s_i = cube.solved_idx.copy()
        s0 = CubieState.solved()

        failed = []
        d_i = 0

        for ma, move in CubieMove.prim_moves.items():
            # 贴纸级旋转
            s_i1 = cube.rotate_state(s_i, *ma)
            s1 = move.act(s0)
            assert CubieMove.build(s0, move.act(s0)) == move
            # 转回 CubieState
            s_i2 = cube.idx_to_state(s_i1)
            s11 = cube.cubie_state(s_i2)  # CubieMove.from_rotation(2, 1, 1)
            s_i3 = s1.to_stickers(n=cube.n)
            s13 = cube.cubie_state(s_i3)  # cubie → 贴纸 → cubie
            assert s13 == s1, f'{s13},{s1}'
            if not np.array_equal(s_i3, s_i2):
                print(f"{(s_i3 != s_i2).sum()}\n {np.argwhere(s_i3 != s_i2)}")
                # 总是差 8 个贴纸,8 个角块 → 正好 8 个贴纸不同 几何事实

            if not s11.is_solvable():
                failed.append(ma)
                print(f"[FAIL] {ma}", s1.is_solvable(), s11.is_solvable())
                print("s11 =", s11)
                print("corner_ori_sum =", s11.corners_ori.sum() % 3)
                print("edge_ori_sum   =", s11.edges_ori.sum() % 2)
                print(
                    "parity(corner, edge) =",
                    CubeBase.permutation_parity(s11.corners_perm),
                    CubeBase.permutation_parity(s11.edges_perm),
                )

            if s1 != s11:
                if not (np.array_equal(s1.edges_perm, s11.edges_perm) and
                        np.array_equal(s1.corners_perm, s11.corners_perm)):
                    print(s1.is_solvable(), s11.is_solvable())
                    raise
                if not np.array_equal(s1.edges_ori, s11.edges_ori):
                    print(cube.corner_orientation(s_i2))
                    print(s1.edges_ori, s11.edges_ori)
                if not np.array_equal(s1.corners_ori, s11.corners_ori):
                    print("s1 corners_ori :", s1.corners_ori.tolist())
                    print("s11 corners_ori:", s11.corners_ori.tolist())
                    diff = (s1.corners_ori - s11.corners_ori) % 3
                    print("diff (mod 3)   :", diff.tolist())
                    print("-" * 40)
                    d_i += 1

        if not failed:
            print("✅ All primitive moves produce solvable CubieState", d_i)  # 这边也是8
        else:
            print("❌ Failed moves:", failed)


    test_all_primitive_moves_solvable(cube)


    def test_outer_moves_only():
        s = cube.solved.copy()

        for axis in (2,):
            for layer in (-1, 1):  # 只转最外层
                for d in (1, -1):
                    s1 = cube.rotate_state(s, axis, layer, d)
                    try:
                        s11 = cube.cubie_state(s1)
                        if not s11.is_solvable():
                            print(s11)
                            raise AssertionError

                    except ValueError as e:
                        print("❌ outer move illegal!", axis, layer, d)
                        raise
        print("✔ outer moves all cubie-valid")


    test_outer_moves_only()

    m = StickerMove.from_rotation(3, 0, 0, 1)
    m_inv = m.inverse()
    assert np.all(m_inv.perm[m.perm] == np.arange(54))

    cube0 = StickerCube(n=3)

    s = cube0.solved_idx  # s = cube0.solved  # StickerCube
    K1 = random.choice(list(CubieMove.prim_moves.keys()))
    K2 = random.choice(list(CubieMove.prim_moves.keys()))
    print(K1, K2)

    m1 = CubieMove.prim_moves[K1]
    m2 = CubieMove.prim_moves[K2]
    m1_s = m1.to_sticker_perm(cube0.n)
    m2_s = m2.to_sticker_perm(cube0.n)
    cube1 = StickerMove(m1_s).replay(cube0)
    cube2 = StickerMove(m2_s).replay(cube1)

    cube3 = StickerMove.act_moves(cube0.get_state(), [m1, m2])
    assert np.all(cube2.cube == cube3)
    print(cube3)
    cube0.reset()
    xx = []
    for move in [K1, K2]:
        axis, side, direction = move
        layer = side * cube0.mid
        xx.append((axis, layer, -direction))  # 用 -direction 对齐

    cube0.apply(xx)
    print(np.all(cube0.cube == cube3))

    s1 = StickerMove(m2_s).act(StickerMove(m1_s).act(s))
    s2 = StickerMove(m1_s).compose(StickerMove(m2_s)).act(s)
    assert np.all(s1 == s2), f'{s1}\n{s2}'

    s_i1 = cube.rotate_state(s_i, 1, 1, 1)
    I = CubieMove.identity()
    M = CubieMove.from_rotation(1, 1, 1)
    s1 = M.act(s0)
    s11 = cube.cubie_state(cube.idx_to_state(s_i1))
    if s1 != s11:
        print('test 1')
        print('s1', s1)
        print('s11', s11)
        print(cube.corner_orientation(cube.idx_to_state(s_i1)))

    s_i1 = cube.rotate_state(s_i, 0, 1, 1)
    s1 = CubieMove.from_rotation(0, 1, 1).act(s0)
    s11 = cube.cubie_state(cube.idx_to_state(s_i1))
    if s1 != s11:
        print('test 2')
        print('s1', s1)
        print('s11', s11)
        print(cube.corner_orientation(cube.idx_to_state(s_i1)))

    s_i1 = cube.rotate_state(s_i, 2, -1, 1)
    s1 = CubieMove.from_rotation(2, -1, 1).act(s0)
    s11 = cube.cubie_state(cube.idx_to_state(s_i1))
    if s1 != s11:
        print('test 3')
        print('s1', s1)
        print('s11', s11)
        print(cube.corner_orientation(cube.idx_to_state(s_i1)))

    s0 = CubieState.solved()
    for move in cube.basic_generators():
        """
        把一个 move 映射到“同一个物理效果但在不同参考系/ gauge 下的等价元素”
        参考系变换元素 gauge element,g 属于 orientation 子群的“平移”部分（常量加法），是正规子群中的元素
        g = s0 ∘ m_truth ∘ m_theory⁻¹ ∘ s0⁻¹
        g 是 m_truth ⋅ m_theory⁻¹ 这个“差异元素”被 s0 共轭后的结果
        是 s0 共轭后的差异元素
        """
        axis, layer, direction = move
        if layer == 0:
            continue
        # 用 from_rotation 得到一个“理论 move”
        m_theory = CubieMove.from_rotation(*move)

        # 用贴纸构建一个“真值 move”
        m_truth = cube.build_cubie_move_from_stickers(cube.solved, *move)

        print(m_theory, '\n', m_truth)
        assert CubieMove.build(s0, m_theory.act(s0)) == m_theory, f'{m_theory}'

        assert np.array_equal(m_theory.corners_perm,
                              m_truth.corners_perm), f'{m_theory.corners_perm}, {m_truth.corners_perm}'
        assert np.array_equal(m_theory.edges_perm, m_truth.edges_perm), f'{m_theory.edges_perm}, {m_truth.edges_perm}'

        if m_theory == m_truth:
            print(f"{move} pass ✓")
            continue

        # assert m_theory.apply(s0)==m_truth.act(s0)
        # 计算 gauge 修正
        s_theory = m_theory.act(s0)  # _left
        s_truth = m_truth.act(s0)
        g = CubieMove.build(s_theory, s_truth)  # g = CubieMove.build(m_theory.act(s0), m_truth.act(s0))
        print(f"{move} gauge g:", g)  # 看 g 的 ori_delta 是否统

        # 修正 from_rotation
        # m_fixed = g.compose(m_truth)
        m_fixed = m_theory.compose(g)

        # 验证修正后一致

        assert m_fixed == m_truth, f'fixed != truth for {move}\n{m_fixed}\n{m_truth}'
        assert m_fixed.act(s0) == m_truth.act(s0), f'fixed act mismatch for {move}'

        g = CubieMove.build(m_truth.act(s0), m_theory.act(s0))
        print(f"{move} gauge g2:", g)
        m_fixed = m_truth.compose(g)  # m_fixed = g.inverse().compose(m_truth)
        assert m_fixed == m_theory, f'fixed != truth for {move}\n{m_fixed}\n{m_theory}'

        print(f"{move} fixed ✓")
        """
        g 需要看 axis / side
        真实贴纸状态里拿不到 move（axis / side)
        同一个 cubie permutation + orientation delta 可以由 多个不同的基本旋转组合产生
        orientation gauge 抹掉了“旋转方向”的信息
        U/D 轴 (axis=1)：g 全 0（无 twist 差异），完全一致
        R/L 轴 (axis=0)：
        side = -1 (L 层)：g = [0,0,1,0,0,1,0,1]（1 在 2,5,7）
        side = 1 (R 层)：g = [0,0,0,1,1,0,0,1]（1 在 3,4,7）或 [1,0,0,0,0,0,0,2]（类似但位置不同）
        
        F/B 轴 (axis=2)：
        side = -1 (B 层)：g 有 2 在 2/3/6/7
        side = 1 (F 层)：g 有 2 在 0/1/4/5/7
        """

    s: CubieState = CubieState.solved()  # cube.cubie_state(cube.solved)
    M = CubieMove.from_rotation(1, 1, 1)
    I = CubieMove.identity()

    # assert m.compose(m.inverse()) == I ,f'{m},{m.inverse()}'
    # 1. 逆元与恒等元
    assert M.compose(I) == M
    assert I.compose(M) == M
    assert M.compose(M.inverse()) == I, f'{M},{M.inverse()}'
    assert M.inverse().compose(M) == I

    for mv in CubieMove.prim_moves.values():
        # ---------- 基本 act ----------
        s1 = mv.act(s)
        assert mv.act(s) == s1
        assert I.act(s) == s
        assert mv.convert().convert() == mv  # 反转左右作用 复原
        assert mv.convert().inverse() == mv.inverse().convert()
        s2 = mv.act_left(s)
        assert mv.convert().act(s) == s2
        assert mv.act(s) == mv.convert().act_left(s)

        assert mv.compose(I) == mv and I.compose(mv) == mv

        # ---------- 逆元消去 ----------
        assert mv.inverse().act(s1) == s
        assert mv.act(mv.inverse().act(s)) == s

        # ---------- 群乘法 ----------
        assert mv.compose(mv.inverse()) == I
        assert mv.inverse().compose(mv) == I

        # ---------- 结合律（抽测） ----------
        # (s ∘ mv) ∘ mv⁻¹ == s ∘ (mv ∘ mv⁻¹)
        assert mv.inverse().act(mv.act(s)) == I.act(s)

        assert mv.corners_ori_delta.sum() % 3 == 0
        assert mv.edges_ori_delta.sum() % 2 == 0

        assert mv.act(s).is_solvable()  # 所有 prim move 保持可解

    I = CubieMove.identity()
    s = CubieState.solved()

    # 1. identity
    assert I.act(s) == s

    # 2. inverse
    for m in CubieMove.prim_moves.values():
        assert m.compose(m.inverse()) == I

    # 3. action consistency
    for x, m in CubieMove.prim_moves.items():
        s1 = m.act(s)
        assert CubieMove.build(s, s1) == m, f'{x},{s1}'
        print(f"{x} ✓")

    # 4. group action 右作用复合一致性
    for m1 in CubieMove.prim_moves.values():
        for m2 in CubieMove.prim_moves.values():
            assert m1.compose(m2).act(s) == m2.act(m1.act(s)), f"compose/act inconsistency: m1={m1}, m2={m2}"


    def check_homomorphism(m: CubieMove, s: CubieState):
        # 验证 phi(m.act(s)) == phi(m).act(phi(s))
        lhs = Phase1Coord.project(m.act(s))  # left
        rhs = Phase1Action.phi(m).act(Phase1Coord.project(s))  # right
        assert lhs == rhs, f"Homomorphism broken!{lhs}_{rhs}"


    # 随机测试
    for _ in range(100):
        m = random.choice(list(CubieMove.prim_moves.values()))
        s = CubieMove.scramble_state(50)
        check_homomorphism(m, s)

    s = CubieState.solved()

    m1 = random.choice(list(CubieMove.phase1_moves().values()))
    m2 = random.choice(list(CubieMove.phase1_moves().values()))

    # 路径 A：先 apply 再 project
    sA = m2.replay(m1.replay(s))
    coordA = Phase1Coord.project(sA)

    # 路径 B：先 project 再 act
    coordB = m2.act(m1.act(Phase1Coord.project(s)))

    assert coordA == coordB

    for c in range(495):
        edges_perm = CubieState.decode_ud_slice(c)
        coord = CubieState.encode_ud_slice(edges_perm.tolist())
        assert coord == c, f'{c},{coord},{edges_perm}'

    s = CubieState.solved()

    for m in CubieMove.phase1_moves.values():
        phi = Phase1Action.phi(m.cubie_move)
        for c in range(495):
            s = s.with_(edges_perm=s.decode_ud_slice(c))
            out = m.cubie_move.act(s)
            assert phi.ud_slice_map[c] == out.ud_slice_coord()

    # for a, m1 in CubieMove.phase1_moves.items():
    #     for b, m2 in CubieMove.phase1_moves.items():
    #         lhs = Phase1Action.phi(m1.cubie_move.compose(m2.cubie_move))
    #         rhs = Phase1Action.phi(m1.cubie_move).compose(
    #             Phase1Action.phi(m2.cubie_move))
    #         assert lhs == rhs, (a, b)
    # # φ₂ 的同态性
    # for a, m1 in CubieMove.phase2_moves().items():
    #     for b, m2 in CubieMove.phase2_moves().items():
    #         lhs = Phase2Action.phi(m1.cubie_move.compose(m2.cubie_move))
    #         rhs = Phase2Action.phi(m1.cubie_move).compose(
    #             Phase2Action.phi(m2.cubie_move)
    #         )
    #         assert lhs == rhs, (a, b)
    sticker_idx = CubieBase(n=5).solved_idx.copy()
    sm = StickerMove.identity(5)
    sm_perm = sm.act(sticker_idx)

    v, e = cube.build_phase_graph(Phase1Coord.solved(), 2)
    print(v)
    print(e)

    # v, e = cube.build_phase_graph(Phase2Coord.solved(), 2)
    # print(v)
    # print(e)

    cube.build_pruning_table()

    # 随机 prim_moves 扰动
    s0 = CubieState.solved()

    for _ in range(3):
        s0 = random.choice(list(CubieMove.phase1_moves.values())).replay(s0)

    moves = CubieBase.phase1_search(s0, depth_limit=7)

    # 应该几乎总能找到
    assert moves is not None

    phase1_coord = Phase1Coord.project(s0)
    s1 = s0.clone()
    s2 = s0.clone()
    m2 = CubieMove.identity()
    for move in moves:
        phase1_coord = move[1].act(phase1_coord)
        s2 = move[1].replay(s2)
        m2 = m2.compose(move[1].cubie_move)
        print(phase1_coord)

    # 应用后
    s1 = CubieMove.apply(s1, [x[0] for x in moves])
    assert np.all(s1.corners_ori == 0)
    assert np.all(s1.edges_ori == 0)
    assert s1 == s2, f'{s1},{s2}'
    s3 = m2.act(s0)
    assert s1 == s3, f'{s1},{s3}'

    sampled_items = random.sample(list(CubieMove.prim_moves().items()), 9)
    s0 = CubieState.solved()
    m0 = CubieMove.identity()
    for x, m in sampled_items:
        print(x)
        m0 = m0.compose(m)

    print(m0)
    # s1 = m0.act(s0)
    # for _ in range(7):
    #     s0 = random.choice(list(CubieMove.prim_moves().values())).act(s0)
    # s0 = CubieMove.scramble_state(7)

    for _ in range(10):
        s0 = random.choice(list(CubieMove.phase1_moves().values())).replay(s0)

    moves_1 = CubieBase.phase1_search(s0, 15)
    phase1_state = s0.clone()
    phase11_state = s0.clone()
    if moves_1:
        phase1_state = CubieMove.apply(phase1_state, [x[0] for x in moves_1])
        for a, m in moves_1:
            phase11_state = m.replay(phase11_state)
        assert phase1_state == phase11_state

    else:
        print('no moves phase1')
    #     moves_1 =  m0.inverse()
    #     phase1_state = moves_1.act(phase1_state)
    #     print(moves_1)

    print(phase1_state.is_phase1_solved(), phase1_state.edges_perm)
    if not phase1_state.is_phase1_solved():
        print(phase1_state.ud_slice_coord())
        phase1_state = CubieBase.canonicalize_ud_slice(phase1_state)
    print(phase1_state.ud_slice_coord(), phase1_state.edges_perm)

    moves_2 = CubieBase.phase2_search(phase1_state, 20)
    phase2_state = phase1_state.clone()
    for a, m in moves_2:
        phase2_state = m.replay(phase2_state)

    _, phase22_state = CubieMove.act_moves(phase1_state, [x[1].cubie_move for x in moves_2])
    assert phase2_state == phase22_state

    print(phase2_state.ud_slice_coord())
    print(phase2_state)

    print(phase2_state.is_phase1_solved())
    assert phase2_state.corners_ori.sum() == 0
    assert phase2_state.edges_ori.sum() == 0
    print(phase2_state == CubieState.solved())
    print([a for a, _ in moves_1 + moves_2])
    print(len(moves_1 + moves_2))

    m22, m2 = CubieBase.solve_kociemba(s0)
    print(m22)
    s20 = m2.act(s0)
    assert s20 == CubieState.solved()

    for m in CubieMove.prim_moves.values():
        for sp in range(24):
            s = Phase15Coord(
                slice_perm=sp,
                corner_coset=0,
                parity=0
            ).decode()

            s2 = m.act(s)
            sp2 = Phase15Coord.project(s2).slice_perm
            assert 0 <= sp2 < 24

    ori_before = cube.corner_orientation(cube.solved)


    def test_random_path_consistency(cube, steps=20):
        s0_st = cube.solved_idx.copy()
        s0_cu = CubieState.solved()  # 参考世界
        sm = StickerMove.identity(cube.n)
        path = []
        for _ in range(steps):
            ma = random.choice(list(CubieMove.prim_moves.keys()))
            path.append(ma)

            sm = sm.compose(sm.from_rotation(cube.n, *ma))

            s0_st = cube.rotate_state(s0_st, *ma)
            # s0_cu =  CubieMove.from_rotation(*ma).act(s0_cu)
            s0_cu = CubieMove.prim_moves[ma].act(s0_cu)

        st11 = cube.idx_to_state(s0_st)
        st123 = sm.act(cube.solved)
        if not np.array_equal(st123, st11):
            print(f"sm {(st123 != st11).sum()}")
        s1 = cube.cubie_state(st11)
        # m = CubieMove.build(s0_cu, s1)
        # assert m.act(s0_cu) == s1, f"delta wrong for {m}"
        s2 = s0_cu

        s_cu = s0_cu.to_stickers(n=cube.n)
        s22 = cube.cubie_state(s_cu)
        assert s22 == s0_cu, f'{s22},{s0_cu}'

        if not np.array_equal(s_cu, st11):
            print(f"cu {(s_cu != st11).sum()}")  # 17

        # permutation 必须一致
        assert np.array_equal(s1.corners_perm, s2.corners_perm), f'{s1.corners_perm}, {s2.corners_perm}'
        assert np.array_equal(s1.edges_perm, s2.edges_perm)

        # solvable 必须一致
        assert s1.is_solvable()
        assert s2.is_solvable()

        moves_cu, mv_cu = CubieBase.solve_kociemba(s2)  # 参考世界
        moves_st, mv_st = CubieBase.solve_kociemba(s1)  # 实际世界
        print(f"{len(moves_st)},{len(moves_cu)}")
        print(mv_st, mv_cu)

        assert mv_st.act(s1) == CubieState.solved()

        st1 = mv_cu.act(s0_cu)
        assert st1 == CubieState.solved()

        act_cu = [(axis, side * cube.mid, dir) for axis, side, dir in moves_cu]
        act_st = [(axis, side * cube.mid, dir) for axis, side, dir in moves_st]
        st10 = st11.copy()
        st21 = st11.copy()
        st31 = st11.copy()
        cube.act_moves(st10, act_st)
        cube.act_moves(st21, act_cu)
        cube.act_moves(st31, cube.invert_moves(path))
        assert cube.is_solved(st31), f'{st31}'

        st12 = cube.cubie_state(st10)
        if st12 != CubieState.solved():
            print(st12)  # corners_ori canonicalize
        st22 = cube.cubie_state(st21)
        if st22 != CubieState.solved():
            print(st22)

        if not cube.is_solved(st10) or not cube.is_solved(st21):
            print(f'{st10}\n{st21}')


    for t in range(3):
        print('test_random_path_consistency', t)
        test_random_path_consistency(cube, steps=30)

    # pm = cube.build_primitive_moves()
    # print(pm)
    #
    # U_move = cube.build_cubie_move_from_stickers(cube.solved, axis=1, layer=cube.mid, direction=1)
    # s: CubieState = cube.cubie_state(cube.solved)
    #
    # # 测试 U^4 = identity
    # for _ in range(4):
    #     s = U_move.apply(s)
    #     assert np.all((s.corners_ori >= 0) & (s.corners_ori < 3))
    #     assert np.all((s.edges_ori >= 0) & (s.edges_ori < 2))
    #
    # assert np.all(s.corners_perm == np.arange(8))
    # assert np.all(s.edges_perm == np.arange(12))
    # assert np.all(s.corners_ori == 0)
    # assert np.all(s.edges_ori == 0)
    # assert np.all(s.edges_ori == 0)

    # for layer in (-cube.mid, cube.mid):
    #     for axis in (0, 1, 2):
    #         move: CubieMove = cube.build_cubie_move_from_stickers(cube.solved, axis, layer, 1)
    #         # cube.AXIS_FACE[axis]
    #         print(axis, layer, move)

    print('.................')


    def test_single_move_physical(n=5):
        cube = CubieBase(n=n)
        for axis in range(3):
            for layer in (-2, -1, 0, 1, 2):
                for d in [1, 2, 3]:
                    s = cube.rotate_state(cube.solved, axis, layer, d)
                    # cube.rotate(axis, layer, d)
                    # assert bool(np.array_equal(cube.cube, s))
                    # cube.rotate(axis, layer, -d)

                    ori_after = cube.corner_orientation(s)
                    ori_delta = cube.corner_orientation_delta(cube.solved, s)
                    corner_perm, _ = cube.corner_ids_ori(s)
                    edge_perm, _ = cube.edge_ids_ori(s)
                    orbit_perm = cube.orbit_perm(s)
                    print("ori sum before:", np.sum(ori_before))  # 0
                    print(f"axis", axis)
                    print("corner_perm after:", corner_perm)
                    print("ori after:", ori_after, 'ori delta:', ori_delta)
                    print("ori sum after:", np.sum(ori_after) % 3)
                    print(cube.heuristic_corner_old(s), cube.heuristic_corner_perm(s),
                          cube.edge_orientation(s))
                    print(orbit_perm)
                    assert edge_perm.shape == (12,)
                    assert corner_perm.shape == (8,)


    test_single_move_physical(5)

    print('.................')

    # print('phase2_moves:', phase2_moves)
    # print(len(phase2_moves))
    # for x in phase2_moves:
    #     s = cube.solved
    #     for mv in x:
    #         s = cube.rotate_state(s, *mv)
    #         assert cube.corner_orientation_ok(s)
    #         assert cube.edge_orientation_ok(s)

    from rime.base import check_class_status

    print(check_class_status(CubieBase))
