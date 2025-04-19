import itertools
import random
import numpy as np
from collections import defaultdict, deque, Counter


def get_axes_by_allele_vector(allele_vector_mapping: dict) -> tuple:
    """
    根据 MAPPING 自动推导出轴（即各个抗原的顺序）。转换为抗原集合。
    这里假设非零向量对应的抗原字母即为轴，按字母顺序排列。
    例如，{'A': (1, 0), 'B': (0, 1), 'O': (0, 0)} 得到 ('A','B')
    """
    # 取向量维度（所有向量长度相同）
    dim = len(next(iter(allele_vector_mapping.values())))
    # 从 mapping 中挑选出非零的等位基因对应的字母
    axes = {allele for allele, vec in allele_vector_mapping.items()
            if vec != (0,) * dim}  # != (0, 0)
    return tuple(sorted(axes))


def get_original_states(state_vector, dim=2):
    """
    反推出量子态的原始基态表示
    """
    if not dim:
        dim = int(np.round(np.sqrt(len(state_vector))))
    index = np.argmax(np.abs(state_vector))  # 找到非零元素的索引
    q1 = index // dim
    q2 = index % dim
    return q1, q2  # f'|{q1}{q2}⟩'


class Allele:
    @staticmethod
    def generate_allele_vector_mapping(axes: list | tuple = ('A', 'B'), outer: str = 'O') -> dict:
        """
        根据给定的等位基因列表和全局轴自动生成向量映射。
        如果等位基因出现在轴中，则返回 one-hot 向量，N*(N+1)
         O 型 返回全零向量。

        例如:
          alleles = ['A', 'B', 'O']
          axes = ('A','B')
        则生成：
          {'A': (1, 0), 'B': (0, 1), 'O': (0, 0)}
        """
        # one-hot 编码：在轴中当前位置为 1，其余为 0
        mapping = {
            allele: tuple(1 if allele == axis else 0 for axis in axes)  # vector[i] = 1
            for allele in axes
        }
        if outer:
            mapping[outer] = (0,) * len(axes)  # tuple(0 for _ in axes) other 设置为全零向量
        return mapping

    AXES = ('A', 'B')  # 定义全局轴，表示所有正表达抗原的字母（顺序也就是向量各分量的定义顺序）
    OUTER = 'O'
    SYSTEM = ''.join(AXES) + OUTER  # alleles
    Allele_Vector_Mapping = generate_allele_vector_mapping(AXES, OUTER)

    def __init__(self, name: str):
        # random.seed(seed)
        # np.random.seed(seed)

        self.name = name
        self.antigens = self.get_antigens()
        self.antibodies = self.get_antibodies()

        """
        'AXES', 
        'Allele_State_Mapping',
        'Allele_Vector_Mapping', 
        'Antigen_To_Vector_Mapping', 
        'Genotype_To_Phenotype_Mapping', 
        'Genotype_To_Vector_Mapping', 
        'Genotype_Transfusion_Matrix', 
        'OUTER', 
        'Phenotype_To_Genotypes_Mapping', 
        'Phenotype_Transfusion_Mapping'
        
        Allele_Vector_Mapping = {
            'A': (1, 0),
            'B': (0, 1),
            'O': (0, 0)
        }
        血型系统是经典的孟德尔遗传，ABO血型由三个等位基因（IA、IB、i）控制，表现为共显性或隐性。
        血型抗原抗体基础规则
        IA（显性）、IB（显性）、i（隐性）构成复等位基因,核基因控制（位于第9号染色体）
        利用血型的抗原和抗体规则构造输血兼容性矩阵
            基础规则：
             - A: 抗原：{'A'}，抗体：{'B'}
             - B: 抗原：{'B'}，抗体：{'A'}
             - O: 抗原：set()，抗体：{'A', 'B'}
             - AB: 抗原：{'A', 'B'}，抗体：set()
         
        antigen_antibody = {
            'A': {'antigens': {'A'}, 'antibodies': {'B'}},
            'B': {'antigens': {'B'}, 'antibodies': {'A'}},
            'O': {'antigens': set(), 'antibodies': {'A', 'B'}},
            'AB': {'antigens': {'A', 'B'}, 'antibodies': set()}
        }
        输血相容性矩阵
        Phenotype_Transfusion_Mapping = {
            'O':  ['O', 'A', 'B', 'AB'],  # 万能供血者
            'A':  ['A', 'AB'],
            'B':  ['B', 'AB'],
            'AB': ['AB']                 # 万能受血者
        }
        基因型到表现型转换,基因型决定表型,血型由 IA/IB/i 组合决定
        Genotype_To_Phenotype_Mapping = {
            'AA': 'A', 'AO': 'A',
            'BB': 'B', 'BO': 'B',
            'OO': 'O',
            'AB': 'AB'
        }
        表现型到可能基因型的映射
        Phenotype_To_Genotypes_Mapping = {
            A: ['AO', 'AA']
            O: ['OO']
            B: ['BB', 'BO']
            AB: ['AB']
        }
        #血型遗传规则
        RULES = {
            'A': [['A', 'O'], ['A', 'A']],
            'B': [['B', 'O'], ['B', 'B']],
            'O': [['O', 'O']],
            'AB': [['A', 'B']]
        }
        """

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"

    def get_antigens(self):
        raise NotImplementedError("子类必须实现 get_antigens 方法")

    def get_antibodies(self):
        raise NotImplementedError("子类必须实现 get_antibodies 方法")

    def get_allele_vector(self):
        return getattr(self, 'value', self.allele_vector(self.name))

    def get_allele_state(self):
        return getattr(self, 'state', self.allele_vector(self.name))

    @classmethod
    def get_vars(cls):
        """获取类中的变量名"""
        return [name for name, value in vars(cls).items() if not callable(value) and not name.startswith("__")]

    # --- 基础映射接口 ---
    @classmethod
    def allele_vector(cls, allele: str) -> tuple:
        """从映射中获取等位基因的二维向量表示 {'A': (1, 0), 'B': (0, 1), 'O': (0, 0)}"""
        return cls.Allele_Vector_Mapping.get(allele)

    @classmethod
    def allele_state(cls, allele: str) -> np.ndarray:
        """从映射中获取等位基因的量子态表示"""
        if hasattr(cls, 'Allele_State_Mapping'):
            return cls.Allele_State_Mapping.get(allele)
        return cls.vector_to_state(cls.allele_vector(allele))

    @classmethod
    def allele_bitwise(cls, allele: str) -> int:
        """从映射中获取等位基因的二维向量表示,转换成二进制编码"""
        if hasattr(cls, 'Allele_Bitwise_Mapping'):
            return cls.Allele_Bitwise_Mapping.get(allele)
        return cls.vector_to_binary(cls.allele_vector(allele))

    # --- 向量操作 ---
    @classmethod
    def combine_vectors(cls, v1: tuple, v2: tuple) -> tuple:
        """组合两个向量，按分量求和后取阈值（大于0则为1，否则0）"""
        # (1 if (v1[0] + v2[0] > 0) else 0, 1 if (v1[1] + v2[1] > 0) else 0)
        # (min(v1[0] + v2[0], 1), min(v1[1] + v2[1], 1))
        return tuple(int(x + y > 0) for x, y in zip(v1, v2))

    @classmethod
    def combine_vectors_np(cls, v1: tuple, v2: tuple) -> tuple:
        arr = np.array(v1) + np.array(v2)  # np.kron() 扩展维度,高维展开
        return tuple((arr > 0).astype(int))  # 使用阈值判断（大于0返回 True，再转换为 int）

    @classmethod
    def combine_quantum(cls, v1: tuple, v2: tuple) -> np.ndarray:
        """
        将 one-hot 向量转换为量子态，量子态合并（叠加态）。
        - O 仍然是叠加态,不会影响 A/B。array([0.707+0.j, 0.707+0.j])
        - A 和 B 组合后应变成 AB 而非 O 的状态。
         A+B=AB，A+O=A，O+O=O
        state1 = array([1.+0.j, 0.+0.j])
        state2 = array([0.+0.j, 1.+0.j])
        combine_quantum(state1, state2) -> [1.+0.j 1.+0.j]
        """
        state1 = cls.vector_to_state(v1)
        state2 = cls.vector_to_state(v2)
        return np.maximum(state1, state2)  # 使用最大值合并
        # combined = state1 + state2
        # norm = np.linalg.norm(combined)  # 向量相加计算范数
        # return combined / norm if norm != 0 else combined # Hadamard 叠加

    @classmethod
    def combine_bitwise(cls, v1: tuple, v2: tuple) -> int:
        """合并等位基因（显性遗传）按位或运算合并,合并后二进制编码: 0b11 (3)"""
        bin1 = cls.vector_to_binary(v1)
        bin2 = cls.vector_to_binary(v2)
        return bin1 | bin2  # (bin1 << len(cls.AXES)) | bin2

    @classmethod
    def set_to_vector(cls, antigens: set) -> tuple:
        """
        将抗原集合转换为二维向量，第一位表示 A, 第二位表示 B ,{'A'} 返回 (1, 0)，{'A','B'} 返回 (1, 1)。
        int('A' in antigens), int('B' in antigens)
        """
        return tuple(int(ag in antigens) for ag in cls.AXES)  # AXES:get_axes_by_allele_vector

    @classmethod
    def vector_to_set(cls, vector: tuple | list) -> set:
        """将二维向量转换为抗原集合 (1, 0):{'A'},(0, 1):{'B'},(0, 0):set(),(1, 1):{'A', 'B'}"""
        # antigens = set()
        # if vector[0]: antigens.add('A')
        # if vector[1]: antigens.add('B')
        # {key for key, value in cls.Allele_Vector_Mapping.items() if any(v and v == value[i] for i, v in enumerate(vector))}
        return {cls.AXES[i] for i, val in enumerate(vector) if val}

    @classmethod
    def binary_to_set(cls, binary: int) -> set:
        antigens = set()
        for i in range(len(cls.AXES)):
            mask = 1 << (len(cls.AXES) - 1 - i)  # 从高位到低位扫描
            if binary & mask:
                antigens.add(cls.AXES[i])
        return antigens

    @classmethod
    def vector_to_phenotype(cls, vector: tuple | list) -> str:
        """
        将向量推导表现型    (0, 0): 'O',(1, 0): 'A',(0, 1): 'B',(1, 1): 'AB'
        """
        # {v: k for k, v in cls.Allele_Vector_Mapping.items()}.get(vector, 'AB')
        antigens = cls.vector_to_set(vector)
        return ''.join(sorted(antigens)) if antigens else cls.OUTER

    @classmethod
    def state_to_phenotype(cls, state: np.ndarray) -> str:
        """
        将量子态转换为表现型名称（'A', 'B', 'AB', 'O'）。
        """
        # 获取1分量对应的抗原
        antigens = {cls.AXES[i] for i, val in enumerate(state) if np.isclose(val, 1)}
        return ''.join(sorted(antigens)) if antigens else cls.OUTER

    @classmethod
    def binary_to_phenotype(cls, binary: int) -> str:
        """将二进制编码转换为表现型名称"""
        antigens = cls.binary_to_set(binary)
        return ''.join(sorted(antigens)) if antigens else cls.OUTER

    @classmethod
    def antigens_to_antibodies_set(cls, antigens: set) -> set:
        """
        利用向量取反的方法：抗体向量 = (1 - v[0], 1 - v[1])，再转换为集合
        """
        antigens_vector = cls.set_to_vector(antigens)
        antibodies_vector = tuple(1 - comp for comp in antigens_vector)  # 按位取反+掩码 (~bin) & mask
        return cls.vector_to_set(antibodies_vector)

    @classmethod
    def vector_to_state(cls, vector: tuple):
        """
        将等位基因向量转换为复数数组表示。
        将 one-hot 抗原向量转换为二复数数组表示。
        (1, 0) -> array([1.+0.j, 0.+0.j])
        (0, 0) -> array([0.70710678+0.j, 0.70710678+0.j])  # 叠加态  O = [1/sqrt(2), 1/sqrt(2)]
        """
        return np.array(vector, dtype=complex) if any(vector) else (
                np.ones(len(vector), dtype=complex) / np.sqrt(len(vector)))

    @classmethod
    def vector_to_binary(cls, vector: tuple) -> int:
        """
        将 one-hot 抗原向量转换为二进制数值编码（按 axes 顺序编码）。
        vector = (1, 0) -> 0b10 (即 2)
        vector = (0, 1) -> 0b01 (即 1)
        vector = (1, 1) -> 0b11 (即 3)
        vector = (0, 0) -> 0b00 (即 0)
        mask = (1 << len(cls.AXES)) - 1,0b11
        """
        # num_axes = len(vector)
        # binary = 0
        # for i, val in enumerate(vector):
        #     if val != 0:
        #         binary |= 1 << (num_axes - 1 - i)  # 高位在左
        return sum(bit << (len(vector) - 1 - i) for i, bit in enumerate(vector) if bit)

    @classmethod
    def binary_to_vector(cls, binary_value: int) -> tuple:
        """
        将二进制数值转换为抗原向量（按 axes 顺序解码,从高位到低位解析） 将二进制表示的等位基因转换为 one-hot 向量。
        binary_value = 0b10, num_axes = 2 : (1, 0)
        """
        num_axes = len(cls.AXES)  # len(next(iter(cls.Allele_State_Mapping.values())))
        return tuple((binary_value >> (num_axes - 1 - i)) & 1 for i in range(num_axes))

    @classmethod
    def union_antigens(cls, allele1: 'Allele', allele2: 'Allele'):
        """合并两个等位基因的抗原信息 :set(antigens)"""
        return allele1.antigens.union(allele2.antigens)

    @classmethod
    def genotype_state(cls, allele1: str, allele2: str) -> np.ndarray:
        """利用量子态表示构造个体的基因型（两个等位基因的张量积）"""
        state1 = cls.allele_state(allele1)
        state2 = cls.allele_state(allele2)
        return np.maximum(state1, state2)

    @classmethod
    def allele_to_antigen_vector(cls, allele1: str, allele2: str) -> tuple:
        """根据两个等位基因生成抗原向量（如 'A' 和 'O' → (1,0)）。"""
        vec1 = cls.allele_vector(allele1)
        vec2 = cls.allele_vector(allele2)
        return cls.combine_vectors(vec1, vec2)

    @classmethod
    def allele_to_phenotype_vector(cls, allele1: str, allele2: str):
        """根据两个等位基因推导表现型,抗原向量:基因型转表现型,根据抗原集合确定表现型"""
        combine_vectors = cls.allele_to_antigen_vector(allele1, allele2)
        return cls.vector_to_phenotype(combine_vectors), combine_vectors

    @classmethod
    def allele_to_phenotype(cls, allele1: str, allele2: str) -> str:
        """基因型转表现型,根据抗原集合确定表现型"""
        return cls.allele_to_phenotype_vector(allele1, allele2)[0]

    @classmethod
    def genotype_to_phenotype(cls, genotype: str) -> str:
        """
        经典的基因型到表现型映射函数：如果有 A，则贡献 A；有 B，则贡献 B,O 不贡献抗原
        """
        alleles = sorted(genotype)
        if hasattr(cls, 'Genotype_To_Phenotype_Mapping'):
            return getattr(cls, 'Genotype_To_Phenotype_Mapping', {}).get((alleles[0], alleles[1]))
        return cls.allele_to_phenotype(alleles[0], alleles[1])
        # if set(alleles) == {'A', 'B'}: return 'AB'  # self.antigens
        # if 'A' in alleles: return 'A'
        # if 'B' in alleles: return 'B'
        # return 'O'

    @classmethod
    def generate_allele_binary_mapping(cls) -> dict:
        """
        生成等位基因到二进制编码的映射（如 A->0b10, B->0b01,'O' → 0b00 (0)）
        {'A': 2, 'B': 1, 'O': 0}
        """
        if not hasattr(cls, 'Allele_Bitwise_Mapping'):
            num_axes = len(cls.AXES)
            # 每个抗原对应一个二进制位，axes顺序决定位权重,左移运算符实现位分配
            mapping = {allele: 1 << (num_axes - 1 - i) for i, allele in enumerate(cls.AXES)}
            if cls.OUTER:
                mapping[cls.OUTER] = 0  # O型对应全0
            cls.Allele_Bitwise_Mapping = mapping
        return cls.Allele_Bitwise_Mapping

    @classmethod
    def generate_allele_state_mapping(cls) -> dict:
        """
        将等位基因转换为量子态映射 ,生成等位基因的向量映射：
        - `axes` 中的等位基因按单位基向量表示
        - `outer` 代表的等位基因是所有 `axes` 叠加的归一化态
        {'A': 基态 |0>, 'B': |1⟩, 'O': 均匀叠加态 (|0> + |1>)/sqrt(2)}, Qubit（2×2） / Qutrit（3×3）
        {'A': array([1.+0.j, 0.+0.j]), 'B': array([0.+0.j, 1.+0.j]), 'O': array([0.70710678+0.j, 0.70710678+0.j])}
        """
        if not hasattr(cls, 'Allele_State_Mapping'):
            num_axes = len(cls.AXES)
            mapping = {
                allele: np.array([1 if i == idx else 0 for idx in range(num_axes)], dtype=complex)
                for i, allele in enumerate(cls.AXES)
            }
            if cls.OUTER:
                mapping[cls.OUTER] = np.full(num_axes, 1 / np.sqrt(num_axes), dtype=complex)  # 叠加态
            cls.Allele_State_Mapping = mapping
        return cls.Allele_State_Mapping

    @classmethod
    def genotype_to_vector_mapping(cls) -> dict[tuple, tuple]:
        """
        生成所有唯一的基因型,每个基因型的抗原向量:
        {'AB': (1, 1), 'OO': (0, 0), 'AA': (1, 0), 'BB': (0, 1), 'BO': (0, 1), 'AO': (1, 0)} ''.join
        {('A', 'A'): (1, 0),
        ('A', 'B'): (1, 1),
        ('A', 'O'): (1, 0),
        ('B', 'B'): (0, 1),
        ('B', 'O'): (0, 1),
        ('O', 'O'): (0, 0)}
        """
        if not hasattr(cls, 'Genotype_To_Vector_Mapping'):
            cls.Genotype_To_Vector_Mapping = {
                tuple(sorted(gt)): cls.allele_to_antigen_vector(*gt) for gt in
                itertools.combinations_with_replacement(cls.Allele_Vector_Mapping.keys(), r=2)}
        return cls.Genotype_To_Vector_Mapping

    @classmethod
    def antigen_to_vector_mapping(cls) -> dict[str, tuple]:
        """
        生成所有表现型:Phenotype 及其抗原向量
        O 型血无抗原（既不表达 A 也不表达 B）,AB 型：同时表达 A 和 B 抗原
        {'A': (1, 0), 'AB': (1, 1), 'B': (0, 1), 'O': (0, 0)}
        """
        if not hasattr(cls, 'Antigen_To_Vector_Mapping'):
            cls.Antigen_To_Vector_Mapping = {
                pheno: vec for p in itertools.combinations_with_replacement(cls.Allele_Vector_Mapping.keys(), r=2)
                for pheno, vec in (cls.allele_to_phenotype_vector(*p),)}
        return cls.Antigen_To_Vector_Mapping

    @classmethod
    def antibody_to_vector_mapping(cls) -> dict[str, tuple]:
        """ 生成所有抗体向量 抗体向量 = 1 - 抗原向量 {'A': (0, 1), 'AB': (0, 0), 'B': (1, 0), 'O': (1, 1)}"""
        if not hasattr(cls, 'Antibody_To_Vector_Mapping'):
            cls.Antibody_To_Vector_Mapping = {k: tuple(1 - x for x in v)
                                              for k, v in cls.antigen_to_vector_mapping().items()}
        return cls.Antibody_To_Vector_Mapping

    @classmethod
    def genotype_antigens(cls, unique=True):
        """
        列举所有可能的基因型（Genotype_antigen）  Genotype_To_Phenotype_Mapping
        ['AA', 'AO', 'BB', 'BO', 'AB', 'OO']
        {('B', 'O'), ('A', 'A'), ('O', 'O'), ('A', 'B'), ('A', 'O'), ('B', 'B')}
        [('A', 'A'), ('A', 'B'), ('A', 'O'), ('B', 'A'), ('B', 'B'), ('B', 'O'), ('O', 'A'), ('O', 'B'), ('O', 'O')]
        """
        if unique:  # frozenset,去重,使用 itertools.product 列举所有长度为 2 的组合
            if hasattr(cls, 'Genotype_To_Vector_Mapping'):
                return set(getattr(cls, 'Genotype_To_Vector_Mapping'))
            return {tuple(sorted(p)) for p in
                    itertools.combinations_with_replacement(cls.Allele_Vector_Mapping.keys(), r=2)}
        return list(itertools.product(cls.Allele_Vector_Mapping.keys(), repeat=2))

    @classmethod
    def genotype_to_phenotype_mapping(cls) -> dict[tuple, str]:
        """
        自动枚举所有可能的基因型，并根据抗原贡献推导出表现型,genotype_to_phenotype,3+2+1
        对 alleles ['A','B','O'] 生成：
            {'AA': 'A', 'AO': 'A', 'BB': 'B', 'BO': 'B', 'AB': 'AB', 'OO': 'O'}  ''.join(sorted(p))
            {('A', 'A'): 'A', ('A', 'B'): 'AB', ('A', 'O'): 'A', ('B', 'B'): 'B', ('B', 'O'): 'B', ('O', 'O'): 'O'}
        """
        if not hasattr(cls, 'Genotype_To_Phenotype_Mapping'):
            cls.Genotype_To_Phenotype_Mapping = {
                tuple(sorted(p)): cls.allele_to_phenotype(*p) for p in
                itertools.combinations_with_replacement(cls.Allele_Vector_Mapping.keys(), r=2)}
        return cls.Genotype_To_Phenotype_Mapping

    @classmethod
    def phenotype_to_genotypes_mapping(cls) -> dict[str, list]:
        """
        表现型到可能基因型的映射，反向构建表现型到所有可能基因型的映射, genotype_to_phenotype
        {'A': ['AA', 'AO'], 'AB': ['AB'], 'B': ['BB', 'BO'], 'O': ['OO']}
        {'A': [('A', 'A'), ('A', 'O')],
        'AB': [('A', 'B')],
        'B': [('B', 'B'), ('B', 'O')],
        'O': [('O', 'O')]}
        """
        if not hasattr(cls, 'Phenotype_To_Genotypes_Mapping'):
            phenotype_map = defaultdict(list)
            if hasattr(cls, 'Genotype_To_Phenotype_Mapping'):
                for gt, pheno in getattr(cls, 'Genotype_To_Phenotype_Mapping').items():
                    phenotype_map[pheno].append(gt)
            else:
                genotype_to_vector = cls.genotype_to_vector_mapping()  # 获取 Genotype -> Vector 映射
                antigen_to_vector = cls.antigen_to_vector_mapping()  # 获取 Phenotype -> Vector 映射
                for genotype, vector in genotype_to_vector.items():  # 反向查找，匹配 vector
                    for pheno, antigen_vector in antigen_to_vector.items():
                        if vector == antigen_vector:  # 如果抗原向量匹配
                            phenotype_map[pheno].append(genotype)

            cls.Phenotype_To_Genotypes_Mapping = dict(phenotype_map)

        return cls.Phenotype_To_Genotypes_Mapping

    @classmethod
    def allele_pairs(cls, pheno_type: str) -> list[tuple]:
        """表现型转基因型，获取血型对应的等位基因组合,如 A:[('A', 'A'), ('A', 'O')] """
        if hasattr(cls, 'Phenotype_To_Genotypes_Mapping'):
            return getattr(cls, 'Phenotype_To_Genotypes_Mapping').get(pheno_type)  # [tuple(g) for g in ]

        genotype_to_vector = cls.genotype_to_vector_mapping()
        antigen_vector = cls.antigen_to_vector_mapping().get(pheno_type)
        return [genotype for genotype, vector in genotype_to_vector.items() if vector == antigen_vector]

    @classmethod
    def is_compatible(cls, donor: 'Allele', recipient: 'Allele') -> bool:
        """
        输血相容性检查：
        检查捐赠者血型 donor 与受血者血型 recipient 是否兼容：
        如果 donor 的抗原出现在 recipient 的抗体中，则不兼容
        """
        donor_antigens = donor.get_antigens()  # set(donor.replace('O', ''))
        recipient_antibodies = recipient.get_antibodies()
        # not any(antigen in recipient_antibodies for antigen in donor_antigens)
        # (donor_bin & antibody_bin) == 0
        # donor in transfusion_rules[recipient]
        # donor_antigens.isdisjoint(recipient_antibodies)
        return len(donor_antigens.intersection(recipient_antibodies)) == 0

    @classmethod
    def genotype_transfusion_matrix(cls):
        """
        生成基因型相容性矩阵,输血相容性逻辑：捐赠者的抗原如果出现在受血者的抗体中，则会被排斥。
        {'AO': ['AO', 'AB', 'AA'], 'AB': ['AB'], 'BO': ['AB', 'BO', 'BB'], 'AA': ['AO', 'AB', 'AA'], 'BB': ['AB', 'BO', 'BB'], 'OO': ['AO', 'AB', 'BO', 'AA', 'BB', 'OO']}
        {('B', 'B'): [('B', 'B'), ('B', 'O'), ('A', 'B')],
        ('B', 'O'): [('B', 'B'), ('B', 'O'), ('A', 'B')],
        ('A', 'B'): [('A', 'B')],
        ('O', 'O'): [('B', 'B'),
        ('B', 'O'), ('A', 'B'), ('O', 'O'), ('A', 'O'), ('A', 'A')],
        ('A', 'O'): [('A', 'B'), ('A', 'O'), ('A', 'A')],
        ('A', 'A'): [('A', 'B'), ('A', 'O'), ('A', 'A')]}
        """
        if not hasattr(cls, 'Genotype_Transfusion_Matrix'):
            # 预计算每个基因型的抗原向量
            genotype_antigen_vector = cls.genotype_to_vector_mapping()
            matrix = {}
            for donor_gt in genotype_antigen_vector.keys():
                matrix[donor_gt] = []
                donor_antigen_vec = genotype_antigen_vector[donor_gt]  # 供体抗原向量
                for recipient_gt in genotype_antigen_vector.keys():
                    recipient_vec = genotype_antigen_vector[recipient_gt]  # 受体抗原向量
                    recipient_antibody_vec = tuple(1 - x for x in recipient_vec)  # 受体抗体向量
                    # 判断供体抗原是否触发受体抗体（按位与全零则相容）
                    if all((d & a) == 0 for d, a in zip(donor_antigen_vec, recipient_antibody_vec)):  # is_compatible
                        matrix[donor_gt].append(recipient_gt)
            cls.Genotype_Transfusion_Matrix = matrix
        return cls.Genotype_Transfusion_Matrix

    @classmethod
    def phenotype_transfusion_mapping(cls):
        """
        输血相容性逻辑：捐赠者的抗原如果出现在受血者的抗体中，则会被排斥。关键原则：受血者的抗体不能与供血者的抗原发生反应。
        因此：如果 donor.antigens 与 recipient.antibodies 有交集，则不兼容。
        {'A': ['A', 'AB'], 'AB': ['AB'], 'B': ['AB', 'B'], 'O': ['A', 'AB', 'B', 'O']}
        """
        if not hasattr(cls, 'Phenotype_Transfusion_Mapping'):
            phenotype_to_vector = cls.antigen_to_vector_mapping()
            blood_types = phenotype_to_vector.keys()  # sorted,{'A', 'B', 'AB', 'O'}
            matrix = {}  # defaultdict(list)
            for donor in blood_types:
                matrix[donor] = []
                antigen_vec = phenotype_to_vector[donor]  # 供体的抗原
                for recipient in blood_types:
                    antibody_vec = tuple(1 - x for x in phenotype_to_vector[recipient])  # 计算受体抗体向量：1 - 抗原向量
                    # 检查供体抗原是否触发受体抗体：按位与后是否为全零 is_compatible
                    if all((d & a) == 0 for d, a in zip(antigen_vec, antibody_vec)):
                        matrix[donor].append(recipient)
            cls.Phenotype_Transfusion_Mapping = matrix
        return cls.Phenotype_Transfusion_Mapping

    @staticmethod
    def get_probability(data: list | tuple, output_format: str = "probs") -> list | set | dict:
        """
        统计列表中的元素频率，并支持不同的输出格式。

        :param data: 输入的列表,possibilities
        :param output_format: 输出格式，可选值：
            - "list": 返回原始列表
            - "set": 返回唯一元素的集合
            - "counter": 返回 Counter 统计的字典
            - "probability": 返回归一化的概率字典,normalize
        :return: 对应格式的统计结果
        """
        if output_format == "list":
            return list(data)
        if output_format == "set":
            return set(data)

        ct = Counter(data)
        if output_format == "counter":
            return dict(ct)
        if output_format == "probs":
            total = sum(ct.values())
            return {key: value / total for key, value in ct.items()} if total > 0 else {}

        raise ValueError("Invalid output_format. Choose from 'list', 'set', 'counter', or 'probability'.")

    @classmethod
    def set_genotype_freq(cls, allele_freq: dict | list | tuple = None):
        """
        计算各基因型频率（基于 Hardy–Weinberg 原理）p² + 2pr + q² + 2qr + 2pq + r² = 1
        allele_freq = {"I^A": 0.3, "I^B": 0.1, "i": 0.6}
        genotype_freq = {
            "AA": allele_freq["I^A"] ** 2,
            "AO": 2 * allele_freq["I^A"] * allele_freq["i"],
            "BB": allele_freq["I^B"] ** 2,
            "BO": 2 * allele_freq["I^B"] * allele_freq["i"],
            "AB": 2 * allele_freq["I^A"] * allele_freq["I^B"],
            "OO": allele_freq["i"] ** 2,
        }
        GENOTYPE_FREQ = {
          ('A', 'A'): 0.09,
          ('A', 'B'): 0.06,
          ('A', 'O'): 0.36,
          ('B', 'B'): 0.01,
          ('B', 'O'): 0.12,
          ('O', 'O'): 0.36
          }
        """
        if not allele_freq:
            # ABO 血型系统等位基因的频率
            allele_freq = {'A': 0.3, 'B': 0.1, 'O': 0.6}
        if isinstance(allele_freq, list | tuple):
            allele_freq = cls.get_probability(allele_freq, output_format="probs")
        if sum(allele_freq.values()) != 1.0:
            raise ValueError("需满足 IA+IB+i=1")
        genotype_freq = defaultdict(float)
        for p in itertools.product(allele_freq.keys(), repeat=2):  # 3*3
            genotype = tuple(sorted(p))
            freq = allele_freq[p[0]] * allele_freq[p[1]]
            genotype_freq[genotype] += freq
        cls.GENOTYPE_FREQ = {k: round(v, 4) for k, v in genotype_freq.items()}
        return cls.GENOTYPE_FREQ

    @classmethod
    def get_genotype_freq(cls):
        """
        # 基因型在人群中的分布频率,根据人群基因频率数据调整（基于 Hardy–Weinberg 原理）
        GENOTYPE_FREQ = {
            ('A', 'A'): 0.03, ('A', 'O'): 0.22,
            ('B', 'B'): 0.02, ('B', 'O'): 0.15,
            ('A', 'B'): 0.04, ('O', 'O'): 0.54
        }
        """
        if not hasattr(cls, 'GENOTYPE_FREQ'):
            return cls.set_genotype_freq()  # setattr(cls, 'GENOTYPE_FREQ', cls.set_genotype_freq())
        return getattr(cls, 'GENOTYPE_FREQ')

    @staticmethod
    def clear_genotype_freq():
        setattr(Allele, "GENOTYPE_FREQ", None)
        # import gc
        # gc.collect()

    @classmethod
    def get_population_freq(cls):
        """
        根据基因型频率 GENOTYPE_FREQ 来计算人群中表现型的分布频率。
        POPULATION_FREQ = {
                'A': 0.31,  # 在人群中的频率
                'B': 0.24,
                'O': 0.39,
                'AB': 0.06
            }
        {'A': 0.45, 'AB': 0.06, 'B': 0.13, 'O': 0.36}
        """
        phenotype_freq = defaultdict(float)
        for genotype, freq in cls.get_genotype_freq().items():
            phenotype = cls.allele_to_phenotype(*genotype)
            phenotype_freq[phenotype] += freq
        return {k: round(v, 4) for k, v in phenotype_freq.items()}  # dict(phenotype_freq)

    @classmethod
    def get_child_phenotypes_by_alleles(cls, parent1_alleles: tuple, parent2_alleles: tuple) -> list:
        """
        根据父母的基因型，计算子代各表现型。blood_types
        ('B', 'A') ('B', 'O') -> {'A', 'B', 'AB'};('B', 'O') ('O', 'B') -> {'B', 'O'}
        """
        return [cls.allele_to_phenotype(*p) for p in itertools.product(parent1_alleles, parent2_alleles)]

    @classmethod
    def genotype_probs_mapping(cls) -> dict[tuple, dict]:
        """
        ABO血型遗传概率矩阵(父母基因型→子代表现型概率)
        父母的顺序不影响子代的概率，这两种组合的概率是相同的，不需要重复存储
        6*6,C(6,2) + 6 = 15 + 6 = 21
        {(('A', 'O'), ('A', 'O')): {'A': 0.75, 'O': 0.25},
        (('A', 'A'), ('A', 'O')): {'A': 1.0},
        (('A', 'O'), ('B', 'B')): {'AB': 0.5, 'B': 0.5},
        (('A', 'B'), ('A', 'O')): {'A': 0.5, 'AB': 0.25, 'B': 0.25},
        (('A', 'O'), ('O', 'O')): {'A': 0.5, 'O': 0.5},
        (('A', 'O'), ('B', 'O')): {'AB': 0.25, 'B': 0.25, 'A': 0.25, 'O': 0.25},
        (('A', 'A'), ('A', 'A')): {'A': 1.0},
        (('A', 'A'), ('B', 'B')): {'AB': 1.0},
        (('A', 'A'), ('A', 'B')): {'A': 0.5, 'AB': 0.5},
        (('A', 'A'), ('O', 'O')): {'A': 1.0},
        (('A', 'A'), ('B', 'O')): {'AB': 0.5, 'A': 0.5},
        (('B', 'B'), ('B', 'B')): {'B': 1.0},
        (('A', 'B'), ('B', 'B')): {'AB': 0.5, 'B': 0.5},
        (('B', 'B'), ('O', 'O')): {'B': 1.0},
        (('B', 'B'), ('B', 'O')): {'B': 1.0},
        (('A', 'B'), ('A', 'B')): {'A': 0.25, 'AB': 0.5, 'B': 0.25},
        (('A', 'B'), ('O', 'O')): {'A': 0.5, 'B': 0.5},
        (('A', 'B'), ('B', 'O')): {'AB': 0.25, 'B': 0.5, 'A': 0.25},
        (('O', 'O'), ('O', 'O')): {'O': 1.0},
        (('B', 'O'), ('O', 'O')): {'B': 0.5, 'O': 0.5},
        (('B', 'O'), ('B', 'O')): {'B': 0.75, 'O': 0.25}}
        """
        if not hasattr(cls, 'Genotype_Probs_Mapping'):
            cls.Genotype_Probs_Mapping = {
                tuple(sorted(gt)): cls.get_probability(cls.get_child_phenotypes_by_alleles(*gt), output_format="probs")
                for gt in itertools.combinations_with_replacement(cls.genotype_antigens(unique=True), r=2)}
        return cls.Genotype_Probs_Mapping  # allele_to_phenotype(*gt[0])

    @classmethod
    def get_child_phenotypes(cls, parent1_phenotype: str, parent2_phenotype: str) -> list:
        """
        根据父母的表现型，计算子代各表现型。通过血清学方法（如凝集反应）直接观察红细胞表面的抗原类型,无法区分基因型中的显隐性组合（如A型可能是AA或AO基因型）
        A + AB ->['A', 'AB', 'A', 'AB', 'A', 'AB', 'A', 'B']-> {'A': 0.5, 'AB': 0.375, 'B': 0.125}
        """
        possibilities = [
            cls.get_child_phenotypes_by_alleles(*g)
            for g in itertools.product(cls.allele_pairs(parent1_phenotype), cls.allele_pairs(parent2_phenotype))]
        return [phenotype for sublist in possibilities for phenotype in sublist]

    @classmethod
    def phenotype_probs_mapping(cls) -> dict[tuple, dict]:
        """
        计算子代血型概率
        根据父母的表现型，计算子代各表现型的概率。C(4,2)+4
        1. 根据表现型找出可能的基因型。
        2. 列举每个基因型的所有可能配子组合，得到子代基因型。
        3. 将子代基因型转换为表现型，并统计出现频次。
        ABO血型遗传概率矩阵(父母表现型→子代概率)
        {('A', 'A'): {'A': 0.9375, 'O': 0.0625},
        ('A', 'AB'): {'A': 0.5, 'AB': 0.375, 'B': 0.125},
        ('A', 'B'): {'AB': 0.5625, 'B': 0.1875, 'A': 0.1875, 'O': 0.0625},
        ('A', 'O'): {'A': 0.75, 'O': 0.25},
        ('AB', 'AB'): {'A': 0.25, 'AB': 0.5, 'B': 0.25},
        ('AB', 'B'): {'AB': 0.375, 'B': 0.5, 'A': 0.125},
        ('AB', 'O'): {'A': 0.5, 'B': 0.5},
        ('B', 'B'): {'B': 0.9375, 'O': 0.0625},
        ('B', 'O'): {'B': 0.75, 'O': 0.25},
        ('O', 'O'): {'O': 1.0}}
        """
        if not hasattr(cls, 'Phenotype_Probs_Mapping'):
            cls.Phenotype_Probs_Mapping = {
                tuple(sorted(p)): cls.get_probability(cls.get_child_phenotypes(*p), output_format="probs")
                for p in itertools.combinations_with_replacement(cls.antigen_to_vector_mapping().keys(), r=2)}
        return cls.Phenotype_Probs_Mapping

    @classmethod
    def get_child_probability(cls, parent1_phenotype: str, parent2_phenotype: str):
        """父母表现型→子代概率"""
        # 生成所有可能的等位基因组合
        p1_combos = cls.allele_pairs(parent1_phenotype)
        p2_combos = cls.allele_pairs(parent2_phenotype)

        # 计算所有组合可能性,穷举父母所有可能的基因型组合和配子组合
        outcomes = defaultdict(int)
        for p1g, p2g in itertools.product(p1_combos, p2_combos):
            for allele1, allele2 in itertools.product(p1g, p2g):
                # child_geno = ''.join(sorted([allele1, allele2]))  # 为了简化，将基因型排序（例如 AO 与 OA 均归为 'AO'）
                # child_pheno = cls.genotype_to_phenotype(child_geno)
                child_pheno = cls.allele_to_phenotype(allele1, allele2)
                outcomes[child_pheno] += 1
                # print({ 'parent1_genotype': p1g, 'parent2_genotype': p2g,'child_genotype': child_geno, 'child_pheno': child_pheno})

        total = sum(outcomes.values())
        return {bt: count / total for bt, count in outcomes.items()}  # 转换为概率

    @classmethod
    def parent_candidates(cls, child_phenotype: str, weights: dict | float | int = None) -> dict:
        """根据子代表现型返回可能的父母组合及概率"""
        # 处理权重参数
        if weights is None:
            weights = cls.get_genotype_freq()

        pheno_comb_probs = defaultdict(float)
        # 遍历所有唯一的父母基因型组合
        for (gt1, gt2), child_dist in cls.genotype_probs_mapping().items():
            if child_phenotype not in child_dist:
                continue
            # 计算基因型组合的先验概率
            prior = weights[gt1] * weights[gt2] if isinstance(weights, dict) else weights
            # 条件概率
            cond_prob = child_dist[child_phenotype]
            # 联合概率
            joint_prob = prior * cond_prob
            # 转换为表现型组合
            p1_pheno = cls.allele_to_phenotype(*gt1)  # genotype_to_phenotype
            p2_pheno = cls.allele_to_phenotype(*gt2)
            # 表现型组合去重
            key = tuple(sorted([p1_pheno, p2_pheno]))
            # 累加到对应表现型组合
            pheno_comb_probs[key] += joint_prob

        total = sum(pheno_comb_probs.values())
        sorted_probs = sorted(pheno_comb_probs.items(), key=lambda x: x[1], reverse=True)
        return {comb: prob / total for comb, prob in sorted_probs}  # 归一化处理

    @classmethod
    def generate_genotype(cls, weights: dict | list | tuple | float | int = None) -> tuple:
        """
        随机生成基因型
         weights (dict): 基因型权重，格式如 {('A','O'):0.22, ...}
                        需包含所有可能的基因型组合
        """
        if isinstance(weights, (float, int)):
            return random.choice(cls.genotype_antigens(unique=False))
        if weights is None:
            weights = cls.get_genotype_freq()

        genotypes = list(cls.genotype_to_vector_mapping().keys())
        if isinstance(weights, (list, tuple)):
            probabilities = list(weights)
        else:  # 处理缺失的血型权重
            probabilities = [weights.get(gt, 0) for gt in genotypes]
        if any(w < 0 for w in probabilities):
            raise ValueError("权重必须为非负数")

        return random.choices(genotypes, weights=probabilities, k=1)[0]  # weights 相对权重,只需要是正数，相对大小决定了选择概率,会自动归一化

    @classmethod
    def generate_phenotype(cls, weights: dict | list | tuple | float | int = None, size=1) -> str | list:
        """
        随机生成血型 blood_types ('A', 'B', 'O', 'AB')
        # get_population_freq()
        """
        phenotypes = tuple(cls.antigen_to_vector_mapping().keys())
        if isinstance(weights, (float, int)):
            return random.choice(phenotypes)
        if weights is None:
            weights = cls.get_population_freq()  # 获取表现型概率分布

        if isinstance(weights, (list, tuple)):
            probabilities = list(weights)
        else:  # 处理缺失的血型权重
            probabilities = [weights.get(pheno, 0) for pheno in phenotypes]
        if any(w < 0 for w in probabilities):
            raise ValueError("权重必须为非负数")
        result = np.random.choice(phenotypes, size=size, p=probabilities)
        return result if size > 1 else result.item()


class AlleleA(Allele):
    def __init__(self):
        super().__init__('A')
        # super().__init__(*args, **kwargs)
        self.value = self.Allele_Vector_Mapping.get('A', (1, 0))

    def get_antigens(self):
        # A 等位基因表达抗原 A
        return {'A'}

    def get_antibodies(self):
        # 依据抗原抗体规则，A 型的抗体为 B
        return {'B'}


class AlleleB(Allele):
    def __init__(self):
        super().__init__('B')
        self.value = self.Allele_Vector_Mapping.get('B', (0, 1))

    def get_antigens(self):
        # B 等位基因表达抗原 B
        return {'B'}

    def get_antibodies(self):
        # B 型的抗体为 A
        return {'A'}


class AlleleO(Allele):
    def __init__(self):
        super().__init__('O')
        self.value = self.Allele_Vector_Mapping.get('B', (0, 0))

    def get_antigens(self):
        # O 等位基因不表达抗原
        return set()

    def get_antibodies(self):
        # O 型没有对应的抗原，但体内通常会产生抗 A 和抗 B 抗体
        return {'A', 'B'}

    def get_allele_vector(self):
        return self.value


class BloodType:
    """血型类，处理等位基因和表现型逻辑"""

    def __init__(self, alleles: tuple[Allele | str, Allele | str], cytoplasm, system: str = 'ABO'):
        """
        :param alleles: 核基因的等位基因对,等位基因组合，如 ('A', 'O'),如 ('Rf', 'rf')
        :param system: 血型系统，默认ABO
        :param cytoplasm: 细胞质类型，'S' 或 'N'
        """
        self.alleles = tuple(
            a if isinstance(a, Allele) else AlleleB for a in alleles)  # 标准化排序，如 ('A', 'O') → ('A', 'O')
        self.system = system
        self.cytoplasm = cytoplasm

    @property
    def phenotype(self) -> str:
        return Allele.allele_to_phenotype(self.alleles[0].name, self.alleles[1].name)

    def get_gametes(self) -> tuple:
        return self.alleles[0].name, self.alleles[1].name

    def is_male_sterile(self):
        """
        判断是否为雄性不育：细胞质为 S 且核基因为 rf/rf
        """
        return self.cytoplasm == 'S' and self.alleles == ('rf', 'rf')

    def cross(self, other):
        """
        与另一植株杂交，生成子代
        :return: 子代基因型 (RicePlant 对象)
        """
        # 子代细胞质来自母本（假设当前植株为母本）
        child_cytoplasm = self.cytoplasm

        # 子代核基因：从父母各随机取一个等位
        child_alleles = (
            random.choice(self.alleles),  # 母本贡献
            random.choice(other.alleles)  # 父本贡献
        )
        # 排序等位基因以便统一表示（如 ('Rf', 'rf') 和 ('rf', 'Rf') 视为相同）
        child_alleles = tuple(sorted(child_alleles, key=lambda x: x[::-1]))

        return BloodType(child_alleles, child_cytoplasm, self.system)

    def __repr__(self):
        return f"BloodType(alleles={self.alleles}, phenotype='{self.phenotype}')"


if __name__ == "__main__":
    genotype_map = Allele.genotype_to_phenotype_mapping()
    genotype_db = Allele.phenotype_to_genotypes_mapping()
    print(Allele.SYSTEM, genotype_map, genotype_db)
    print('A', Allele.allele_pairs('A'),
          Allele.allele_vector('O'),
          'AB:', Allele.allele_to_phenotype('A', 'B'), Allele.allele_to_phenotype('B', 'A'))
    print(Allele.vector_to_state((0, 0)))
    print("Combined AB:", Allele.combine_quantum((1, 0), (0, 1)), Allele.genotype_state('A', 'B'))
    print("Combined AO:", Allele.combine_quantum((1, 0), (0, 0)), Allele.genotype_state('A', 'O'))
    print("Combined OO:", Allele.combine_quantum((0, 0), (0, 0)), Allele.genotype_state('O', 'O'))
    print(Allele.state_to_phenotype(Allele.genotype_state('A', 'B')))
    print(Allele.state_to_phenotype(Allele.genotype_state('A', 'O')))
    print(Allele.state_to_phenotype(Allele.genotype_state('O', 'O')))

    print('phenotype', Allele.generate_phenotype(), 'genotypes', Allele.genotype_antigens(False))
    print('ab antigens', Allele.vector_to_set((1, 1)), 'antibodies',
          Allele.antigens_to_antibodies_set({'A'}), Allele.antigens_to_antibodies_set({'O'}),
          Allele.antigens_to_antibodies_set({'A', 'B'}))

    a1 = Allele.generate_genotype()
    a2 = Allele.generate_genotype()
    p1 = Allele.allele_to_phenotype(*a1)
    p2 = Allele.allele_to_phenotype(*a2)
    c = Allele.get_child_phenotypes_by_alleles(a1, a2)
    print(a1, a2, 'then', c)
    print(Allele.get_child_phenotypes(p1, p2))
    c1 = Allele.get_child_probability(p1, p2)
    print(p1, 'and', p2, 'then', c1)

    child_probability = {','.join(sorted(p)): Allele.get_child_probability(*p)
                         for p in itertools.product(Allele.antigen_to_vector_mapping().keys(), repeat=2)}
    print(child_probability)
    print(Allele.phenotype_probs_mapping())

    print(AlleleB.union_antigens(AlleleA(), AlleleO()))
    print(Allele.genotype_transfusion_matrix())
    print('phenotype_transfusion', Allele.phenotype_transfusion_mapping())
    print('allele_state', Allele.generate_allele_state_mapping())
    print(dir(AlleleB), '\n', AlleleO.__dict__)
    print('repr', Allele.__repr__)
    print(Allele.antibody_to_vector_mapping())
    print('AlleleB', AlleleB.allele_bitwise('B'), Allele.generate_allele_binary_mapping(),
          AlleleB.allele_bitwise('A'))
    print(AlleleB.Genotype_Transfusion_Matrix)
    print(AlleleB.genotype_probs_mapping())
    print('parent_candidates', AlleleB.parent_candidates(child_phenotype='A', weights=None))

    print(Allele.get_population_freq())
    print(Allele.set_genotype_freq(allele_freq={'A': 0.2, 'B': 0.1, 'O': 0.7}))
    print(Allele.get_population_freq())
    print(Allele.genotype_to_phenotype('AB'))
    print(Allele.antigen_to_vector_mapping())
    print(Allele.genotype_to_vector_mapping())
    print(Allele.phenotype_to_genotypes_mapping())
    print(Allele.allele_pairs('A'))
    print(Allele.get_vars())
    print(Allele.generate_phenotype(size=30))
