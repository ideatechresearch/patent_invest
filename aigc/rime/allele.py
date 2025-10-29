import itertools
import random
import numpy as np
from collections import defaultdict, Counter


def dice_feature(dice: list | tuple) -> list:
    """
       分析三个骰子的数值特征。

       参数:
           dice: 包含三个整数（1-6）的列表或元组，代表骰子点数。

       返回:
           list: 一个包含特征字符串的列表。
    for dice in itertools.combinations_with_replacement(range(1,7), r=3)：
        f = dice_feature(dice)
    """
    a, b, c = sorted(dice)  # 排序便于判断连续和相同
    features = []

    # 1. 判断是否为顺子（连续）
    if b - a == 1 and c - b == 1:
        features.append("连续")
    if c - a >= 5:
        features.append("夸大")

    # 2. 判断是否三同
    if a == b == c:
        features.append("全同")
    # 3. 判断是否两同一异 (Pair)
    elif a == b or b == c:
        # 确定那个成对的数字和单独的数字
        # pair_num = a if a == b else c
        features.append("两同")

    # 4. 判断是否为质数组合 (三个点数都是质数)
    # 骰子中的质数面：2, 3, 5
    prime_faces = {2, 3, 5}
    if all(die in prime_faces for die in dice):
        features.append("质升")

    if all(die % 2 == 1 for die in dice):  # 全奇数
        features.append("全奇")
    elif all(die % 2 == 0 for die in dice):  # 全偶数
        features.append("全偶")

    # 和值大小 (例如，和值大于12算大)
    sum_dice = sum(dice)
    if sum_dice > 12:
        features.append("大数")
    elif sum_dice < 6:
        features.append("小数")

    # 如果没有显著特征，则标记为“杂色”
    if not features:
        features.append("杂花")

    return features


class AlleleBase:
    _expressed_cache = {}  # 缓存的懒加载类属性

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

    @staticmethod
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

    @staticmethod
    def class_property(attr_name: str):
        """
        缓存装饰器，支持首次调用生成值并缓存到类属性。
        自定义缓存属性名，适合无参或固定参数的懒加载。
        用于懒加载（lazy load）型类属性。仅检查当前类。
        @AlleleBase.class_property("cached_value")
        """

        def decorator(func) -> classmethod:
            def wrapper(cls, *args):
                if not hasattr(cls, attr_name):
                    setattr(cls, attr_name, func(cls, *args))
                return getattr(cls, attr_name)

            return classmethod(wrapper)

        return decorator

    @classmethod
    def set_cache(cls, attr_name: str, factory, *args, **kwargs):
        """
        通用类属性缓存：若类属性不存在，则调用 factory() 生成并缓存。
        factory 可以是可调用函数，也可以是直接的值。
        实际写入属性的是 cls,当前调用的类对象,缓存绑定到 Sub 自身
        """
        if callable(factory):
            value = factory(cls, *args, **kwargs)
            src = factory.__name__
        else:
            value = factory
            src = repr(factory)
        if isinstance(value, list):
            value = tuple(value)
        elif isinstance(value, set):
            value = frozenset(value)
        # if isinstance(value, dict):
        #     value = MappingProxyType(value)  # read_only
        setattr(cls, attr_name, value)
        cls._expressed_cache[attr_name] = src
        return value

    @classmethod
    def re_cache(cls, attr_name: str, *, rebuild: bool = False):
        """触发懒加载,重新构建"""
        builder = cls._expressed_cache.get(attr_name)
        if isinstance(builder, str):
            builder = getattr(cls, builder, None)
        if callable(builder):
            if rebuild and hasattr(cls, attr_name):
                delattr(cls, attr_name)
            return builder()  # cls.set_cache(attr_name, builder)
        return getattr(cls, attr_name, None)

    @classmethod
    def get_cache(cls, attr_name: str = None, key=None):
        """
        若 key 不为 None，则从缓存中取值（通常是 dict），不存在则抛 KeyError。
        """
        if attr_name is None:
            return {name: getattr(cls, name, None) for name in cls._expressed_cache}
        cache = getattr(cls, attr_name, None)
        if cache is None:
            return None
        if key is None:
            return cache
        try:
            return cache[key]
        except (KeyError, TypeError):
            raise KeyError(f"[{cls.__name__}] Key {key} not found in {attr_name} or invalid.")

    @classmethod
    def is_expressed(cls, name: str) -> bool:
        return name in cls._expressed_cache or hasattr(cls, name)

    @classmethod
    def suppress_expressed(cls, names: list[str]):
        """撤销表达"""
        for attr in set(names):
            if cls._expressed_cache.pop(attr, None) is not None:  # del cls._expressed_cache[attr]
                if hasattr(cls, attr):
                    delattr(cls, attr)

    @classmethod
    def get_vars(cls):
        """获取类中的变量名"""
        return [name for name, value in vars(cls).items() if
                not (callable(value) or isinstance(value, (classmethod, staticmethod)) or name.startswith("_"))]

    @staticmethod
    def sorted_frozen(*args, unique: bool = False) -> tuple | frozenset:
        """接受任意多个参数并返回排序后的 tuple"""
        if len(args) == 1 and isinstance(args[0], (list, tuple, set)):
            items = args[0]
        else:
            items = list(args)
        return frozenset(items) if unique else tuple(sorted(items))  # dict.fromkeys(items)

    @staticmethod
    def vector_to_state(vector: tuple):
        """
        将等位基因向量转换为复数数组表示。
        将 one-hot 抗原向量转换为二复数数组表示。
        状态 |0> (1, 0) -> array([1.+0.j, 0.+0.j])
        状态 |1> (0, 0) -> array([0.70710678+0.j, 0.70710678+0.j])
        状态 (|0> + |1>)/√2 叠加态  O = [1/sqrt(2), 1/sqrt(2)]
        """
        if any(vector):
            return np.array(vector, dtype=complex)
        return np.ones(len(vector), dtype=complex) / np.sqrt(len(vector))

    @staticmethod
    def vector_to_binary(vector: tuple) -> int:
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

    @staticmethod
    def equal_vectors(vecs: list[tuple], vector: tuple) -> bool:
        # (np.array(v1) + np.array(v2) > 0).astype(int) bit 比对 使用阈值判断（大于0返回 True，再转换为 int）
        target = np.asarray(vector, dtype=bool)
        merged = np.logical_or.reduce([np.array(v, dtype=bool) for v in vecs])
        return np.array_equal(merged, target)

    @staticmethod
    def tensor_product(state1, state2) -> np.ndarray:
        """
        计算两个量子态的张量积
        np.kron() 扩展维度,高维展开
        """
        return np.kron(state1, state2)

    @staticmethod
    def density_matrix(state) -> np.ndarray:
        """从状态向量计算密度矩阵 纯态密度矩阵"""
        return np.outer(state, np.conj(state))

    @staticmethod
    def is_quantum_state(vector, tolerance=1e-10):
        """检查向量是否表示有效的量子态（是否归一化）"""
        norm = np.sum(np.abs(vector) ** 2)
        return abs(norm - 1.0) < tolerance

    @staticmethod
    def is_entangled(state, tolerance=1e-10):
        """
        简单判断两量子比特纯态是否纠缠
        通过检查是否可以被写成两个单量子比特状态的张量积
        # 贝尔态 |Φ+> = (|00> + |11>)/√2
        bell_phi_plus = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
        # 贝尔态 |Ψ-> = (|01> - |10>)/√2
        bell_psi_minus = np.array([0, 1/np.sqrt(2), -1/np.sqrt(2), 0])
        # GHZ态: (|000⟩ + |111⟩)/√2
        # W态: (|001⟩ + |010⟩ + |100⟩)/√3
        """
        psi_matrix = state.reshape(2, 2)  # 将状态向量重塑为 2x2 矩阵
        U, S, Vh = np.linalg.svd(psi_matrix)  # 计算奇异值分解

        # 如果只有一个非零奇异值，则不纠缠
        # 如果有多个非零奇异值，则纠缠
        if np.sum(S > tolerance) > 1:
            return True
        # 计算两量子比特纯态的并发度，并发度 concurrence > 0 表示纠缠
        a, b, c, d = state
        return 2 * abs(a * d - b * c) > tolerance

    @staticmethod
    def separate_product_state(state, dims: list = [2, 2], tolerance=1e-10):
        """
        尝试分离可分离的张量积状态 composite_state
        """
        # 重塑为矩阵
        M = state.reshape(dims[0], dims[1])
        U, S, Vh = np.linalg.svd(M)  # SVD分解

        # 如果只有一个非零奇异值，则是可分离的
        if np.sum(S > tolerance) == 1:
            state_1 = U[:, 0] * np.sqrt(S[0])
            state_2 = Vh[0, :] * np.sqrt(S[0])
            return state_1, state_2
        # 状态是纠缠的，无法分离为纯态张量积,纠缠态不可逆
        return None, None

    @staticmethod
    def original_states(state, dims: list = None) -> tuple:
        """
        反推出量子态的原始基态表示,从量子态向量反推出最可能的基态分量
        单量子比特 → N=2
        双量子比特 → N=4 [2, 2]
        三量子比特 → N=8 [2,2,2]
        """
        state_vector = np.asarray(state)
        N = len(state_vector)
        if dims is None:
            if np.log2(N).is_integer():
                n_qubits = int(np.log2(N))
                subsystem_dims = [2] * n_qubits  # 多个量子比特的系统
            else:
                root = int(np.round(np.sqrt(N)))
                subsystem_dims = [root, root] if root * root == N else [N]
        else:
            if np.prod(dims) != N:
                raise ValueError(f"子系统维度乘积不等于状态向量长度 {N}")
            subsystem_dims = dims

        index = np.argmax(np.abs(state_vector))  # 找到非零元素的索引,幅度最大的主基态
        amplitude = state_vector[index]  # probability = np.abs(amplitude)**2,phase = np.angle(amplitude)
        indices = []
        for d in reversed(subsystem_dims):  # 从低位到高位，各子系统的基态索引
            indices.append(index % d)
            index //= d
        indices = list(reversed(indices))  # ket=f"|{''.join(map(str, indices))}⟩"
        return subsystem_dims, tuple(indices), amplitude  # f'|{q1}{q2}⟩'

    @staticmethod
    def get_probability(data: list | tuple | dict, output_format: str = "probs", sort: bool = False) -> dict:
        """
        统计列表中的元素频率，并支持不同的输出格式。

        :param data: 输入的列表,possibilities
        :param output_format: 输出格式，可选值：
            - "counter": 返回 Counter 统计的字典
            - "probability": 返回归一化的概率字典,normalize
        :param sort: 按值从大到小排序
        :return: 对应格式的统计结果
        """
        ct = data.copy() if isinstance(data, dict) else Counter(data)
        if output_format == "counter":
            if sort:
                ct = sorted(ct.items(), key=lambda x: x[1], reverse=True)
            return dict(ct)

        if output_format in ("probs", "probability"):
            total = sum(ct.values())
            if total == 0:
                return {}
            probs = {key: value / total for key, value in ct.items()}
            if sort:
                return dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))
            return probs

        raise ValueError("Invalid output_format. Choose from  'counter' or 'probability'.")

    @staticmethod
    def normalize_weights(items: list | tuple, weights: dict | list | tuple | float | int) -> list:
        """
        将多种形式的权重输入统一为与 items 对齐的概率列表。

        :param items: 要采样的元素序列
        :param weights: 输入权重，可以是 dict / list / tuple / float / int / None
        :return: 概率（未必归一化，但可直接用于 random.choices / np.random.choice）
        """
        n = len(items)
        # weights 是数字 → 均匀随机
        if isinstance(weights, (float, int)):
            probabilities = [1.0 / n] * n
        elif isinstance(weights, (list, tuple)):
            if len(weights) != n:
                raise ValueError(f"权重长度 {len(weights)} 与元素数 {n} 不匹配")
            probabilities = list(weights)
        elif isinstance(weights, dict):  # 处理缺失的权重
            probabilities = [weights.get(x, 0.0) for x in items]
        else:
            raise TypeError(f"不支持的权重类型: {type(weights)}")

        if any(w < 0 for w in probabilities):
            raise ValueError("权重必须为非负数")
        if sum(probabilities) <= 0:
            raise ValueError("权重总和必须大于 0")
        return probabilities  # weights 相对权重,只需要是正数，相对大小决定了选择概率,会自动归一化

    def __init_subclass__(cls, **kwargs):
        cls.__doc__ = cls.__name__ + """:
        'AXES', 'OUTER', 'ALLELES', 'GENOTYPES'
        'Allele_Vector_Mapping', 
        'Genotype_To_Phenotype_Mapping',
        'Phenotype_To_Genotypes_Mapping', 
        'Phenotype_To_Antigen_Mapping', 
        'GENOTYPE_FREQ', 
        'Genotype_To_Vector_Mapping', 
        'Phenotype_Probs_Mapping', 
        'Genotype_Transfusion_Mapping', 
        'Allele_State_Mapping', 
        'Allele_Binary_Mapping', 
        'Vector_To_Antigen_Mapping', 
        'Vector_To_Antibody_Mapping', 
        'Phenotype_Transfusion_Mapping', 
        
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
        Genotype = Allele × Allele 一个基因型由两个等位基因组成
        Allele:str 单个字符或枚举值，基本基因型元素 allele->genotype
        Genotype:tuple 二元组合，字符或枚举值，两个等位基因组成
        Antigens:set 字符或枚举值 NOT OUTER. antigens->phenotype
        Phenotype:str 表现型，基因表达决定,可观测特征
        """
        super().__init_subclass__(**kwargs)


class Allele(AlleleBase):
    AXES = ('A', 'B')  # 定义全局轴，表示所有正表达抗原的字母（顺序也就是向量各分量的定义顺序）
    OUTER = 'O'
    DEFAULT_FREQ = {'A': 0.3, 'B': 0.1, 'O': 0.6}
    Allele_Vector_Mapping = AlleleBase.generate_allele_vector_mapping(AXES, OUTER)  # 等位基因 -> 向量映射

    def __init__(self, name: str = None) -> None:
        # random.seed(seed)
        # np.random.seed(seed)
        if name is None:
            self.name = self.generate_allele()
        elif name in self.alleles():
            self.name = name
        else:
            raise ValueError(f"无效的等位基因名称: {name}")

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name}), SYSTEM={self.system()}"

    def __eq__(self, other):
        """
        判断两个等位基因是否相等：
        - 类型必须相同；
        - 名称相同或向量完全一致；
        - 允许通过哈希缓存简化比较。
        """
        if self is other:
            return True
        if not isinstance(other, self.__class__):
            return NotImplemented
        if hasattr(self, "vector") and hasattr(other, "vector"):  # 优先比较向量
            return tuple(self.vector) == tuple(other.vector)
        return self.name == other.name

    def get_antigens(self) -> set:
        # O 等位基因不表达抗原
        # B 等位基因表达抗原 B
        # A 等位基因表达抗原 A
        return self.vector_to_antigens(self.get_vector())

    def get_antibodies(self) -> set:
        # 依据抗原抗体规则
        # A 型的抗体为 B
        # B 型的抗体为 A
        # O 型没有对应的抗原，但体内通常会产生抗 A 和抗 B 抗体
        return self.antigens_to_antibodies(self.get_antigens())

    def get_vector(self) -> tuple:
        return getattr(self, 'value', self.allele_vector(self.name))

    def get_state(self) -> np.ndarray:
        return getattr(self, 'state', self.allele_state(self.name))

    def get_freq(self) -> float:
        if hasattr(self, 'freq'):
            return getattr(self, 'freq')
        if hasattr(self, 'ALLELE_FREQ'):
            return self.ALLELE_FREQ.get(self.name)
        return self.DEFAULT_FREQ.get(self.name, 0.0)

    @staticmethod
    def union_antigens(allele1: 'Allele', allele2: 'Allele') -> set:
        """合并两个等位基因的抗原信息 :set(antigens)| """
        return allele1.get_antigens().union(allele2.get_antigens())

    @classmethod
    def rebuild_constants(cls, axes: list | tuple = None, outer: str = None):
        """
        重建核心常量和缓存：
        - 可传入新的 axes 和 outer
        - 清空 _expressed_cache
        - 重新生成 SYSTEM 和 Allele_Vector_Mapping
        """
        if axes is not None:
            cls.AXES = tuple(axes)
        if outer is not None:
            cls.OUTER = outer

        cls._expressed_cache.clear()

        # 清空其他缓存懒加载属性
        for attr, value in tuple(vars(cls).items()):
            if attr not in ('AXES', 'OUTER', 'DEFAULT_FREQ', '_expressed_cache'):
                if not (attr.startswith('_') or callable(value) or isinstance(value, (classmethod, staticmethod))):
                    delattr(cls, attr)

        cls.allele_vector_mapping()  # 'Allele_Vector_Mapping'
        print(f"重建完成，SYSTEM={cls.system()}")

    @staticmethod
    def class_property(attr_name: str):
        """
        缓存装饰器，支持首次调用生成值并缓存到类属性。
        自定义缓存属性名，适合无参或固定参数的懒加载。
        用于懒加载（lazy load）型类属性。仅检查当前类。
        @class_property("cached_value")
        """

        def decorator(func) -> classmethod:
            def wrapper(cls, *args):
                if not cls.is_expressed(attr_name):
                    print(f"{attr_name} -> {cls.__name__}.{func.__name__}")
                    return cls.set_cache(attr_name, func, *args)
                return cls.get_cache(attr_name)

            return classmethod(wrapper)

        return decorator

    @class_property('ALLELES')
    def alleles(cls) -> tuple:
        if hasattr(cls, 'Allele_Vector_Mapping'):
            return tuple(sorted(cls.Allele_Vector_Mapping, key=cls.Allele_Vector_Mapping.get))
        return *cls.AXES, cls.OUTER

    @classmethod
    def system(cls):
        return ''.join(cls.AXES) + cls.OUTER

    # --- 基础映射接口 ---
    @classmethod
    def allele_vector(cls, allele: str) -> tuple:
        """从映射中获取等位基因的二维向量表示 {'A': (1, 0), 'B': (0, 1), 'O': (0, 0)}"""
        if hasattr(cls, 'Allele_Vector_Mapping'):
            return cls.Allele_Vector_Mapping.get(allele)
        return tuple(1 if allele == axis else 0 for axis in cls.AXES)  # one-hot

    @classmethod
    def allele_state(cls, allele: str) -> np.ndarray:
        """从映射中获取等位基因的量子态表示"""
        if hasattr(cls, 'Allele_State_Mapping'):
            return cls.Allele_State_Mapping.get(allele)
        return cls.vector_to_state(cls.allele_vector(allele))

    @classmethod
    def allele_binary(cls, allele: str) -> int:
        """从映射中获取等位基因的二维向量表示,转换成二进制编码"""
        if hasattr(cls, 'Allele_Binary_Mapping'):
            return cls.Allele_Binary_Mapping.get(allele)
        return cls.vector_to_binary(cls.allele_vector(allele))

    # --- 向量操作 ---
    @classmethod
    def combine_vectors(cls, v1: tuple, v2: tuple) -> tuple:
        """组合两个向量，按分量求（大于0则为取原始和，不超过 num_axes，否则0）"""
        # x | y, tuple(int(x + y > 0) for x, y in zip(v1, v2))
        # (min(v1[0] + v2[0], 1), min(v1[1] + v2[1], 1))
        num_axes = len(cls.AXES)
        return tuple(min(x + y, num_axes) for x, y in zip(v1, v2))

    @classmethod
    def combine_quantum(cls, v1: tuple, v2: tuple) -> np.ndarray:
        """
        将 one-hot 向量转换为量子态，量子态合并（叠加态）。
        - O 仍然是叠加态,不会影响 A/B。array([0.707+0.j, 0.707+0.j])
        - A 和 B 组合后应变成 AB 而非 O 的状态。
         A+B=AB，A+O=A，O+O=O
        state1 psi_0 = array([1.+0.j, 0.+0.j])
        state2 psi_1 = array([0.+0.j, 1.+0.j])
        combine_quantum(state1, state2) -> [1.+0.j 1.+0.j]
        """
        state1 = cls.vector_to_state(v1)
        state2 = cls.vector_to_state(v2)
        return np.maximum(state1, state2)  # 使用最大值合并,中间状态
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
    def antigens_to_vector(cls, antigens: set) -> tuple:
        """
        将抗原集合转换为二维向量，第一位表示 A, 第二位表示 B ,{'A'} 返回 (1, 0)，{'A','B'} 返回 (1, 1), {'O'} 返回 (0, 0)
        int('A' in antigens), int('B' in antigens)
        """
        return tuple(int(axis in antigens) for axis in cls.AXES)  # AXES=get_axes_by_allele_vector

    @classmethod
    def vector_to_antigens(cls, vector: tuple | list) -> set:
        """将二维向量转换为抗原集合 (1, 0):{'A'},(0, 1):{'B'},(0, 0):set(),(1, 1):{'A', 'B'}"""
        # antigens = set()
        # if vector[0]: antigens.add('A')
        # if vector[1]: antigens.add('B')
        # {key for key, value in cls.Allele_Vector_Mapping.items() if any(v and v == value[i] for i, v in enumerate(vector))}
        return {cls.AXES[i] for i, val in enumerate(vector) if val}

    @classmethod
    def antigens_to_antibodies(cls, antigens: set) -> set:
        """
        利用抗原向量取反的方法：抗体向量 = (1 - v[0], 1 - v[1])，再转换为集合
        """
        vector = cls.antigens_to_vector(antigens)  # set(phenotype)
        inverted = tuple(1 - v for v in vector)  # 按位取反+掩码 (~bin) & mask
        return cls.vector_to_antigens(inverted)

    @classmethod
    def binary_to_antigens(cls, binary: int) -> set:
        """将二进制编码转换为抗原集合,从高位到低位扫描"""
        num_axes = len(cls.AXES)
        return {cls.AXES[i] for i in range(num_axes) if binary & (1 << (num_axes - 1 - i))}

    @classmethod
    def state_to_antigens(cls, state: np.ndarray) -> set:
        """获取1分量对应的抗原"""
        return {cls.AXES[i] for i, val in enumerate(state) if np.isclose(val, 1)}

    @classmethod
    def antigens_to_phenotype(cls, antigens: set) -> str:
        """抗原集合 → 表现型"""
        return ''.join(sorted(antigens)) if antigens else cls.OUTER

    @classmethod
    def vector_to_phenotype(cls, vector: tuple | list) -> str:
        """
        将向量推导表现型    (0, 0): 'O',(1, 0): 'A',(0, 1): 'B',(1, 1): 'AB'
        """
        return cls.antigens_to_phenotype(cls.vector_to_antigens(vector))

    @classmethod
    def state_to_phenotype(cls, state: np.ndarray) -> str:
        """
        将量子态转换为表现型名称（'A', 'B', 'AB', 'O'）。
        """
        return cls.antigens_to_phenotype(cls.state_to_antigens(state))

    @classmethod
    def binary_to_phenotype(cls, binary: int) -> str:
        """将二进制编码转换为表现型名称"""
        return cls.antigens_to_phenotype(cls.binary_to_antigens(binary))

    @classmethod
    def binary_to_vector(cls, binary_value: int) -> tuple[int, ...]:
        """
        将二进制数值转换为抗原向量（按 axes 顺序解码,从高位到低位解析） 将二进制表示的等位基因转换为 one-hot 向量。
        binary_value = 0b10, num_axes = 2 : (1, 0)
        """
        num_axes = len(cls.AXES)  # len(next(iter(cls.Allele_State_Mapping.values())))
        return tuple((binary_value >> (num_axes - 1 - i)) & 1 for i in range(num_axes))

    @classmethod
    def genotype_state(cls, allele1: str, allele2: str) -> np.ndarray:
        """利用量子态表示构造个体的基因型（两个等位基因的张量积）,默认取实部进行比较"""
        state1 = cls.allele_state(allele1)
        state2 = cls.allele_state(allele2)
        return np.maximum(state1, state2)

    @classmethod
    def allele_combine_vector(cls, allele1: str, allele2: str) -> tuple:
        """根据两个等位基因生成抗原向量（如 'A' 和 'O' → (1,0)）。"""
        vec1 = cls.allele_vector(allele1)
        vec2 = cls.allele_vector(allele2)
        return cls.combine_vectors(vec1, vec2)

    @classmethod
    def allele_to_antigens_vector(cls, allele1: str, allele2: str) -> (set, tuple):
        """
        根据两个等位基因计算表现型向量。allele_to_antigens_vector('A','B')->{'A', 'B'}, (1, 1)
        1. 将两个等位基因映射为抗原向量。
        2. 将抗原向量按阈值转换为二值向量（0/1）。
        3. 根据向量映射得到表现型。
        """
        antigen_vector = tuple(int(v > 0) for v in cls.allele_combine_vector(allele1, allele2))  # 转为二值向量
        return cls.vector_to_antigens(antigen_vector), antigen_vector

    @classmethod
    def allele_to_phenotype(cls, allele1: str, allele2: str) -> str:
        """基因型转表现型,根据抗原集合确定表现型"""
        antigen_vector = cls.allele_to_antigens_vector(allele1, allele2)[1]
        return cls.vector_to_phenotype(antigen_vector)

    @classmethod
    def genotype_to_phenotype(cls, *args, genotype: tuple | str = None) -> str:
        """
        经典的基因型到表现型映射函数：如果有 A，则贡献 A；有 B，则贡献 B,O 不贡献抗原
        """
        gt = tuple(sorted(genotype) if genotype else sorted(args))
        if hasattr(cls, 'Genotype_To_Phenotype_Mapping'):
            return getattr(cls, 'Genotype_To_Phenotype_Mapping', {}).get(gt)
        return cls.allele_to_phenotype(*gt)
        # if set(alleles) == {'A', 'B'}: return 'AB'
        # if 'A' in alleles: return 'A'
        # if 'B' in alleles: return 'B'
        # return 'O'

    @classmethod
    def phenotype_to_antigens(cls, phenotype: str) -> set:
        """表现型 → 抗原集合"""
        return cls.phenotype_to_antigen_mapping().get(''.join(sorted(phenotype)), set(phenotype) & set(cls.AXES))

    # ---映射函数 f()----
    @class_property('Allele_Binary_Mapping')
    def allele_binary_mapping(cls) -> dict:
        """
        生成等位基因到二进制编码的映射（如 A->0b10, B->0b01,'O' → 0b00 (0)）
        {'A': 2, 'B': 1, 'O': 0}
        """
        num_axes = len(cls.AXES)
        # 每个抗原对应一个二进制位，axes顺序决定位权重,左移运算符实现位分配
        mapping = {allele: 1 << (num_axes - 1 - i) for i, allele in enumerate(cls.AXES)}
        if cls.OUTER:
            mapping[cls.OUTER] = 0  # O型对应全0
        return mapping

    @class_property('Allele_State_Mapping')
    def allele_state_mapping(cls) -> dict:
        """
        将等位基因转换为量子态映射 ,生成等位基因的向量映射：
        - `axes` 中的等位基因按单位基向量表示
        - `outer` 代表的等位基因是所有 `axes` 叠加的归一化态
        {'A': 基态 |0>, 'B': |1⟩, 'O': 均匀叠加态 (|0> + |1>)/sqrt(2)}, Qubit（2×2） / Qutrit（3×3）
        {'A': array([1.+0.j, 0.+0.j]), 'B': array([0.+0.j, 1.+0.j]), 'O': array([0.70710678+0.j, 0.70710678+0.j])}
        """
        num_axes = len(cls.AXES)
        mapping = {allele: np.array(np.eye(num_axes, dtype=complex)[i]) for i, allele in enumerate(cls.AXES)}
        if cls.OUTER:
            mapping[cls.OUTER] = np.full(num_axes, 1 / np.sqrt(num_axes), dtype=complex)  # 叠加态
        return mapping

    @class_property('Allele_Vector_Mapping')
    def allele_vector_mapping(cls) -> dict:
        return cls.generate_allele_vector_mapping(cls.AXES, cls.OUTER)

    @class_property('Genotype_To_Vector_Mapping')
    def genotype_to_vector_mapping(cls) -> dict[tuple, tuple]:
        """
        生成所有唯一的基因型,每个基因型的抗原向量:
        {'AB': (1, 1), 'OO': (0, 0), 'AA': (1, 0), 'BB': (0, 1), 'BO': (0, 1), 'AO': (1, 0)} ''.join
        {('A', 'A'): (2, 0), ('A', 'B'): (1, 1), ('A', 'O'): (1, 0), ('B', 'B'): (0, 2), ('B', 'O'): (0, 1), ('O', 'O'): (0, 0)}
        """
        if hasattr(cls, 'Allele_Vector_Mapping'):
            return {(a, b): cls.combine_vectors(cls.Allele_Vector_Mapping[a], cls.Allele_Vector_Mapping[b])
                    for a, b in itertools.combinations_with_replacement(cls.Allele_Vector_Mapping, r=2)}
        return {cls.sorted_frozen(gt): cls.allele_combine_vector(*gt) for gt in
                itertools.combinations_with_replacement(cls.alleles(), r=2)}

    @class_property('GENOTYPES')
    def genotypes(cls) -> list[tuple]:
        """按 vector 排序  genotype → index 映射 """
        mapping = cls.genotype_to_vector_mapping()
        return sorted(mapping, key=mapping.get)

    @class_property('Phenotype_To_Antigen_Mapping')
    def phenotype_to_antigen_mapping(cls) -> dict[str, set]:
        """{'A': {'A'}, 'AB': {'A', 'B'}, 'B': {'B'}, 'O': set()}"""
        mapping = {}
        for combo in itertools.combinations_with_replacement(cls.alleles(), 2):
            vector = cls.antigens_to_vector(set(combo))
            phenotype = cls.vector_to_phenotype(vector)
            mapping[phenotype] = cls.vector_to_antigens(vector)
        return mapping
        # {cls.antigens_to_phenotype(antigens): antigens  for vec,antigens in cls.vector_to_antigen_mapping().items()}

    @class_property('PHENOTYPES')
    def phenotypes(cls) -> list:
        # 'O', 'B', 'A', 'AB' phenotype_to_genotypes_mapping()
        return sorted(cls.phenotype_to_antigen_mapping().keys())

    @class_property('Vector_To_Antigen_Mapping')
    def vector_to_antigen_mapping(cls) -> dict[tuple, set]:
        """
        生成所有表现型:Phenotype 及其抗原向量
        O 型血无抗原（既不表达 A 也不表达 B）,AB 型：同时表达 A 和 B 抗原,这里多了(1, 1)
        {'A': (1, 0), 'AB': (1, 1), 'B': (0, 1), 'O': (0, 0)}
        {(1, 0): {'A'}, (1, 1): {'B', 'A'}, (0, 1): {'B'}, (0, 0): set()}
        """
        return {vec: antigens for p in itertools.combinations_with_replacement(cls.alleles(), r=2)
                for antigens, vec in (cls.allele_to_antigens_vector(*p),)}

    @class_property('Vector_To_Antibody_Mapping')
    def vector_to_antibody_mapping(cls) -> dict[tuple, set]:
        """
        生成所有抗体向量 抗体向量 = 1 - 抗原向量
        {'A': (0, 1), 'AB': (0, 0), 'B': (1, 0), 'O': (1, 1)}
        {(1, 0): {'B'}, (1, 1): set(), (0, 1): {'A'}, (0, 0): {'B', 'A'}}
        """
        return {vec: cls.antigens_to_antibodies(antigens) for vec, antigens in cls.vector_to_antigen_mapping().items()}

    @classmethod
    def genotype_iter(cls, unique: bool = True) -> set | list:
        """
        列举所有可能的基因型（Genotype_antigen）  Genotype_To_Phenotype_Mapping
        ['AA', 'AO', 'BB', 'BO', 'AB', 'OO']
        {('A', 'A'), ('A', 'B'), ('A', 'O'), ('B', 'B'), ('B', 'O'), ('O', 'O')}
        [('A', 'A'), ('A', 'B'), ('A', 'O'), ('B', 'A'), ('B', 'B'), ('B', 'O'), ('O', 'A'), ('O', 'B'), ('O', 'O')]
        """
        if unique:  # frozenset,去重,使用 itertools.product 列举所有长度为 2 的组合
            if hasattr(cls, 'GENOTYPES'):
                return set(getattr(cls, 'GENOTYPES'))
            return {cls.sorted_frozen(gt) for gt in itertools.combinations_with_replacement(cls.alleles(), r=2)}
        return list(itertools.product(cls.alleles(), repeat=2))

    @class_property('Genotype_To_Phenotype_Mapping')
    def genotype_to_phenotype_mapping(cls) -> dict[tuple, str]:
        """
        自动枚举所有可能的基因型，并根据抗原贡献推导出表现型,genotype_to_phenotype,3+2+1
        对 alleles ['A','B','O'] 生成：
            {'AA': 'A', 'AO': 'A', 'BB': 'B', 'BO': 'B', 'AB': 'AB', 'OO': 'O'}  ''.join(sorted(p))
            {('A', 'A'): 'A', ('A', 'B'): 'AB', ('A', 'O'): 'A', ('B', 'B'): 'B', ('B', 'O'): 'B', ('O', 'O'): 'O'}
        """
        return {cls.sorted_frozen(gt): cls.allele_to_phenotype(*gt) for gt in
                itertools.combinations_with_replacement(cls.alleles(), r=2)}

    @class_property('Phenotype_To_Genotypes_Mapping')
    def phenotype_to_genotypes_mapping(cls) -> dict[str, dict]:
        """
        表现型到可能基因型的映射，反向构建表现型到所有可能基因型的映射, genotype_to_phenotype
        {'A': ['AA', 'AO'], 'AB': ['AB'], 'B': ['BB', 'BO'], 'O': ['OO']}
        {'O': {('O', 'O'): 1.0},
        'B': {('B', 'O'): 0.5, ('B', 'B'): 0.5},
        'A': {('A', 'O'): 0.5, ('A', 'A'): 0.5},
        'AB': {('A', 'B'): 1.0}}
        """
        phenotype_map = defaultdict(list)
        if hasattr(cls, 'Genotype_To_Phenotype_Mapping'):
            for gt, pheno in getattr(cls, 'Genotype_To_Phenotype_Mapping').items():
                phenotype_map[pheno].append(gt)
        else:
            genotype_to_vector = cls.genotype_to_vector_mapping()  # 获取 Genotype -> Vector 映射
            vector_to_antigen = cls.vector_to_antigen_mapping()  # 获取  Vector -> Antigens 映射
            for genotype, vector in genotype_to_vector.items():
                antigen_vector = tuple(int(v > 0) for v in vector)
                if antigen_vector in vector_to_antigen:  # 如果抗原向量匹配,反向查找，匹配 vector
                    pheno = cls.antigens_to_phenotype(vector_to_antigen[antigen_vector])
                    phenotype_map[pheno].append(genotype)
            # equals = [g for  g,v in genotype_to_vector.items() if cls.equal_vectors([vector], antigen_vector)]
        return {p: {g: 1 / len(gs) for g in gs} for p, gs in phenotype_map.items()}  # dict(phenotype_map) 等概率假设uniform

    @classmethod
    def phenotype_to_genotypes(cls, phenotype: str) -> list:
        """表现型转基因型，获取血型对应的等位基因组合,如 A:[('A', 'A'), ('A', 'O')] """
        if hasattr(cls, 'Phenotype_To_Genotypes_Mapping'):
            return list(getattr(cls, 'Phenotype_To_Genotypes_Mapping').get(''.join(sorted(phenotype)), {}))

        target_antigens = cls.phenotype_to_antigens(phenotype)
        return [g for g, v in cls.genotype_to_vector_mapping().items() if cls.vector_to_antigens(v) == target_antigens]

    @classmethod
    def get_random_genotype(cls, phenotype: str, prob: bool = False) -> tuple[tuple, float]:
        """
        根据给定表现型随机返回一个可能的基因型。prob=True 按条件概率 P(genotype | phenotype) 加权抽样；
        返回 (随机选中的基因型, 该表现型下可能的基因型概率, 用于判断唯一)
        """
        if prob:  # 加权抽样 conditional
            cond_map = cls.phenotype_conditional_mapping()
            weights = cond_map.get(''.join(sorted(phenotype)), {})
            chosen = random.choices(list(weights.keys()), weights=list(weights.values()), k=1)[0]
            return chosen, weights.get(chosen)

        genotypes = cls.phenotype_to_genotypes(phenotype)
        return random.choice(genotypes), 1 / len(genotypes)  # 均匀抽样 uniform

    @classmethod
    def is_compatible_phenotype(cls, pheno_donor: str, pheno_recipient: str) -> bool:
        """
        输血相容性检查（基于表现型）：
        检查捐赠者血型 pheno_donor 与受血者血型 pheno_recipient 是否兼容。

        规则：
        - donor 的抗原不能出现在 recipient 的抗体中；
        - 若类中已定义 Phenotype_Transfusion_Mapping，则优先使用；
        - 否则按抗原/抗体规则动态计算。
        """
        if hasattr(cls, 'Phenotype_Transfusion_Mapping'):
            return pheno_recipient in getattr(cls, 'Phenotype_Transfusion_Mapping').get(pheno_donor, [])

        donor_antigens = cls.phenotype_to_antigens(pheno_donor)
        recipient_antibodies = cls.antigens_to_antibodies(set(pheno_recipient))
        return donor_antigens.isdisjoint(recipient_antibodies)

    @class_property('Genotype_Transfusion_Mapping')
    def genotype_transfusion_mapping(cls):
        """
        生成基因型相容性矩阵,输血相容性逻辑：捐赠者的抗原如果出现在受血者的抗体中，则会被排斥。
        {'AO': ['AO', 'AB', 'AA'], 'AB': ['AB'], 'BO': ['AB', 'BO', 'BB'], 'AA': ['AO', 'AB', 'AA'], 'BB': ['AB', 'BO', 'BB'], 'OO': ['AO', 'AB', 'BO', 'AA', 'BB', 'OO']}
        {('A', 'A'): [('A', 'A'), ('A', 'B'), ('A', 'O')],
        ('A', 'B'): [('A', 'B')],
        ('A', 'O'): [('A', 'A'), ('A', 'B'), ('A', 'O')],
        ('B', 'B'): [('A', 'B'), ('B', 'B'), ('B', 'O')],
        ('B', 'O'): [('A', 'B'), ('B', 'B'), ('B', 'O')],
        ('O', 'O'): [('A', 'A'), ('A', 'B'), ('A', 'O'), ('B', 'B'), ('B', 'O'), ('O', 'O')]}
        """
        # 预计算每个基因型的抗原向量
        genotype_antigen_vector = cls.genotype_to_vector_mapping()
        matrix = {}
        for donor_gt, donor_vector in genotype_antigen_vector.items():
            donor_antigen_vec = [v != 0 for v in donor_vector]  # 供体抗原向量
            matrix[donor_gt] = []
            for recipient_gt, recipient_vector in genotype_antigen_vector.items():
                recipient_antibody_vec = [1 - (v != 0) for v in recipient_vector]  # 受体抗体向量
                # 判断供体抗原是否触发受体抗体（按位与全零则相容）
                if all((d & r) == 0 for d, r in zip(donor_antigen_vec, recipient_antibody_vec)):  # is_compatible
                    matrix[donor_gt].append(recipient_gt)
        return matrix

    @class_property('Phenotype_Transfusion_Mapping')
    def phenotype_transfusion_mapping(cls):
        """
        输血相容性逻辑：捐赠者的抗原如果出现在受血者的抗体中，则会被排斥。关键原则：受血者的抗体不能与供血者的抗原发生反应。
        {'A': ['A', 'AB'], 'AB': ['AB'], 'B': ['B', 'AB'], 'O': ['B', 'O', 'A', 'AB']}
        """
        vector_to_antigen = cls.vector_to_antigen_mapping()
        vector_to_antibody = cls.vector_to_antibody_mapping()
        matrix = {}  # defaultdict(list)
        for antigen_vec, antigens in vector_to_antigen.items():  # 供体的抗原
            donor = cls.antigens_to_phenotype(antigens)
            matrix[donor] = []
            for antibody_vec, antibodies in vector_to_antibody.items():  # 受体抗体
                recipient = cls.antigens_to_phenotype(antibodies)
                # 检查供体抗原是否触发受体抗体：按位与后是否为全零 is_compatible
                if all((d & r) == 0 for d, r in zip(antigen_vec, antibody_vec)):
                    matrix[donor].append(recipient)
        return matrix

    @classmethod
    def set_allele_freq(cls, freq: dict = None, prior_size: int = None, data: list | tuple | dict = None) -> dict:
        """
         allele_freq = {"I^A": 0.3, "I^B": 0.1, "i": 0.6}
        :param freq: 手动指定的频率分布ABO 血型系统等位基因的频率
        :param prior_size: 先验对应的等效样本量（必须与 data 同时提供）
        :param data: 可选的观测数据（基因型拆分为等位基因）
        :return:
        """
        alleles = cls.alleles()
        allele_freq = cls.DEFAULT_FREQ if freq is None else {a: float(freq.get(a, 0.0)) for a in alleles}
        if bool(data) ^ bool(prior_size):
            raise ValueError("data 与 prior_size 必须同时提供或同时为空")
        if data and prior_size:
            if isinstance(data, dict):
                if any(not isinstance(v, (int, float)) or v < 0 for v in data.values()):
                    raise ValueError("data 需为正数计数")
                ct = data.copy()
            else:
                ct = Counter(data)
            combined_counts = {a: ct.get(a, 0) + allele_freq.get(a, 0) * prior_size for a in alleles}
            allele_freq = Allele.get_probability(combined_counts, output_format="probs")

        if any(v < 0 for v in allele_freq.values()):
            raise ValueError("等位基因频率不能为负数")
        if abs(sum(allele_freq.values()) - 1.0) > 1e-6:
            raise ValueError("需满足 IA+IB+i=1")

        cls.suppress_expressed(["PHENOTYPE_FREQ", "GENOTYPE_FREQ"])
        setattr(cls, 'ALLELE_FREQ', allele_freq)
        return allele_freq

    @classmethod
    def allele_freq(cls):
        """获取当前类级别的等位基因频率（懒加载）"""
        if not hasattr(cls, 'ALLELE_FREQ'):
            return cls.set_allele_freq()
        return getattr(cls, 'ALLELE_FREQ')

    @class_property('GENOTYPE_FREQ')
    def genotype_freq(cls) -> dict:
        """
        基因型在人群中的分布频率,根据人群基因频率数据调整
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
        allele_freq = cls.allele_freq()
        genotype_freq = defaultdict(float)
        for g in itertools.product(allele_freq.keys(), repeat=2):  # 3*3
            genotype = cls.sorted_frozen(g)
            freq = allele_freq[g[0]] * allele_freq[g[1]]
            genotype_freq[genotype] += freq
        return {k: round(v, 6) for k, v in genotype_freq.items()}

    @class_property('PHENOTYPE_FREQ')
    def phenotype_freq(cls) -> dict:
        """
        根据基因型频率 GENOTYPE_FREQ 来计算人群中表现型的分布频率, phenotype 概率 P(p)。
        POPULATION_FREQ = {
                'A': 0.31,  # 在人群中的频率
                'B': 0.24,
                'O': 0.39,
                'AB': 0.06
            }
        {'A': 0.45, 'AB': 0.06, 'B': 0.13, 'O': 0.36}
        """
        phenotype_freq = defaultdict(float)
        for genotype, freq in cls.genotype_freq().items():
            phenotype_freq[cls.genotype_to_phenotype(genotype=genotype)] += freq
        return {k: round(v, 6) for k, v in phenotype_freq.items()}  # dict(phenotype_freq)

    @classmethod
    def phenotype_conditional_mapping(cls) -> dict:
        """
        构建条件概率映射：P(g | p), 表现型->genotype freq
        {'O': {('O', 'O'): 1.0},
        'B': {('B', 'O'): 0.923076923076923, ('B', 'B'): 0.07692307692307693},
        'A': {('A', 'O'): 0.7999999999999999, ('A', 'A'): 0.19999999999999998},
        'AB': {('A', 'B'): 1.0}}
        """
        genotype_freq = cls.genotype_freq()
        pheno_freq = cls.phenotype_freq()
        cond = {}
        for p, prob in cls.phenotype_to_genotypes_mapping().items():
            cond[p] = {}
            for g in prob:
                cond[p][g] = genotype_freq[g] / pheno_freq[p]
        return cond

    @staticmethod
    def child_genotype_distribution(parent1_genotype: tuple, parent2_genotype: tuple) -> dict:
        """
        根据父母的基因型，计算子代各基因型，简单孟德尔模型，穷举父母所有可能的基因型组合和配子组合，子代基因型分布。无群体修正
        如 ('A','B'), ('O','O') -> [('A', 'O'), ('A', 'O'), ('B', 'O'), ('B', 'O')] 并数量统计 child_g_counter
        """
        return dict(Counter([tuple(sorted(gt)) for gt in itertools.product(parent1_genotype, parent2_genotype)]))

    @class_property('Genotype_Probs_Mapping')
    def genotype_probs_mapping(cls) -> dict[tuple, dict]:
        """
        ABO血型遗传概率矩阵(父母基因型→子代基因型概率)

        父母的顺序不影响子代的概率，这两种组合的概率是相同的，不需要重复存储
        6*6,C(6,2) + 6 = 15 + 6 = 21
        {(('O', 'O'), ('O', 'O')): {('O', 'O'): 1.0},
        (('B', 'O'), ('O', 'O')): {('B', 'O'): 0.5, ('O', 'O'): 0.5},
        (('B', 'B'), ('O', 'O')): {('B', 'O'): 1.0},
        (('A', 'O'), ('O', 'O')): {('A', 'O'): 0.5, ('O', 'O'): 0.5},
        (('A', 'B'), ('O', 'O')): {('A', 'O'): 0.5, ('B', 'O'): 0.5},
        (('A', 'A'), ('O', 'O')): {('A', 'O'): 1.0},
        (('B', 'O'), ('B', 'O')): {('B', 'O'): 0.5, ('B', 'B'): 0.25, ('O', 'O'): 0.25},
        (('B', 'B'), ('B', 'O')): {('B', 'B'): 0.5, ('B', 'O'): 0.5},
        (('A', 'O'), ('B', 'O')): {('A', 'B'): 0.25, ('B', 'O'): 0.25, ('A', 'O'): 0.25, ('O', 'O'): 0.25},
        (('A', 'B'), ('B', 'O')): {('A', 'B'): 0.25, ('B', 'B'): 0.25, ('A', 'O'): 0.25, ('B', 'O'): 0.25},
        (('A', 'A'), ('B', 'O')): {('A', 'B'): 0.5, ('A', 'O'): 0.5},
        (('B', 'B'), ('B', 'B')): {('B', 'B'): 1.0},
        (('A', 'O'), ('B', 'B')): {('A', 'B'): 0.5, ('B', 'O'): 0.5},
        (('A', 'B'), ('B', 'B')): {('A', 'B'): 0.5, ('B', 'B'): 0.5},
        (('A', 'A'), ('B', 'B')): {('A', 'B'): 1.0},
        (('A', 'O'), ('A', 'O')): {('A', 'O'): 0.5, ('A', 'A'): 0.25, ('O', 'O'): 0.25},
        (('A', 'B'), ('A', 'O')): {('A', 'A'): 0.25, ('A', 'B'): 0.25, ('A', 'O'): 0.25, ('B', 'O'): 0.25},
        (('A', 'A'), ('A', 'O')): {('A', 'A'): 0.5, ('A', 'O'): 0.5},
        (('A', 'B'), ('A', 'B')): {('A', 'B'): 0.5, ('A', 'A'): 0.25, ('B', 'B'): 0.25},
        (('A', 'A'), ('A', 'B')): {('A', 'A'): 0.5, ('A', 'B'): 0.5},
        (('A', 'A'), ('A', 'A')): {('A', 'A'): 1.0}}
        """
        return {cls.sorted_frozen(gt):
                    cls.get_probability(data=cls.child_genotype_distribution(*gt), output_format="probs", sort=True)
                for gt in itertools.combinations_with_replacement(cls.genotypes(), r=2)}

    @classmethod
    def is_valid_child_by_alleles(cls, child_alleles: tuple, parent1_alleles: tuple, parent2_alleles: tuple) -> bool:
        """检查此人是否可能是两个父母的子女, 检查子女的每个等位基因是否来自父母"""
        if hasattr(cls, 'Genotype_Probs_Mapping'):
            key = cls.sorted_frozen(cls.sorted_frozen(parent1_alleles), cls.sorted_frozen(parent2_alleles))
            child_probs = getattr(cls, 'Genotype_Probs_Mapping').get(key, {})
            return cls.sorted_frozen(child_alleles) in child_probs if child_probs else False

        p1_count = Counter(parent1_alleles)
        p2_count = Counter(parent2_alleles)
        for allele in child_alleles:
            if p1_count[allele] > 0:
                p1_count[allele] -= 1
            elif p2_count[allele] > 0:
                p2_count[allele] -= 1
            else:
                return False
        return True  # tuple(sorted(child_alleles)) in product_child_genotype_by_alleles(parent1_alleles,parent2_alleles)

    @class_property('Phenotype_Probs_Mapping')
    def phenotype_probs_mapping(cls) -> dict[tuple, dict]:
        """
        计算所有父母表现型组合对应的子代表现型概率分布（无群体修正,基因型主导下的遗传分布投影） 计算子代血型概率。C(4,2)+4
        1. 根据表现型找出可能的基因型。考虑了 A=AO 的隐性组合权重
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
        pheno_comb_probs = defaultdict(lambda: defaultdict(float))
        # 遍历所有唯一的父母基因型组合
        for (gt1, gt2), child_dist in cls.genotype_probs_mapping().items():
            # 转换为表现型组合
            p1_pheno = cls.genotype_to_phenotype(*gt1)
            p2_pheno = cls.genotype_to_phenotype(*gt2)
            key = cls.sorted_frozen(p1_pheno, p2_pheno)  # 表现型组合去重
            for cg, prob in child_dist.items():  # 将子基因型分布转为表现型概率
                child_pheno = cls.genotype_to_phenotype(*cg)
                pheno_comb_probs[key][child_pheno] += prob  # 累加

        return {k: cls.get_probability(v, output_format="probs", sort=True) for k, v in pheno_comb_probs.items()}

    @class_property('Phenotype_Probs_Equal_Mapping')
    def phenotype_probs_equal_mapping(cls):
        '''等权平均所有可能父母基因型
        {('O', 'O'): {'O': 1.0},
         ('B', 'O'): {'B': 0.75, 'O': 0.25},
         ('A', 'O'): {'A': 0.75, 'O': 0.25},
         ('AB', 'O'): {'A': 0.5, 'B': 0.5},
         ('B', 'B'): {'B': 0.9375, 'O': 0.0625},
         ('A', 'B'): {'AB': 0.5625, 'B': 0.1875, 'A': 0.1875, 'O': 0.0625},
         ('AB', 'B'): {'AB': 0.375, 'B': 0.5, 'A': 0.125},
         ('A', 'A'): {'A': 0.9375, 'O': 0.0625},
         ('A', 'AB'): {'A': 0.5, 'AB': 0.375, 'B': 0.125},
         ('AB', 'AB'): {'A': 0.25, 'AB': 0.5, 'B': 0.25}}
        '''
        pheno_comb_probs = defaultdict(lambda: defaultdict(float))
        phenotype_to_genotypes = cls.phenotype_to_genotypes_mapping()
        for p1, p2 in itertools.combinations_with_replacement(phenotype_to_genotypes.keys(), r=2):
            key = cls.sorted_frozen(p1, p2)
            gts1 = phenotype_to_genotypes[p1]
            gts2 = phenotype_to_genotypes[p2]
            combos = list(itertools.product(gts1, gts2))
            for g1, g2 in combos:
                child_dist = cls.child_genotype_distribution(g1, g2)
                for cg, prob in child_dist.items():
                    pheno = cls.genotype_to_phenotype(*cg)
                    pheno_comb_probs[key][pheno] += prob / len(combos)  # 等权平均

        return {k: cls.get_probability(v, output_format='probs', sort=True) for k, v in pheno_comb_probs.items()}

    @classmethod
    def child_phenotype_distribution(cls, parent1_phenotype: str, parent2_phenotype: str, equal: bool = True,
                                     prob: bool = False) -> dict:
        """
        根据父母的表现型，计算子代各表现型。通过血清学方法（如凝集反应）直接观察红细胞表面的抗原类型,无法区分基因型中的显隐性组合
        生成所有可能的等位基因组合，计算所有组合可能性，这里有两次 product
        A + AB ->['A', 'AB', 'A', 'AB', 'A', 'AB', 'A', 'B']-> {'A': 0.5, 'AB': 0.375, 'B': 0.125}
        """
        if equal:  # 等权枚举
            pheno_counter = defaultdict(int)
            # 生成所有可能的等位基因组合,假设 phenotype_to_genotypes 等权重（blood_types 表现型推到基因型时信息丢失）
            # 这边有逻辑偏差，表现型到可能基因型的映射信息丢失，会扭曲整体分布，表现型只是基因型的结果（如A型可能是AA或AO基因型）
            p1_combos = cls.phenotype_to_genotypes(parent1_phenotype)
            p2_combos = cls.phenotype_to_genotypes(parent2_phenotype)
            for p1g, p2g in itertools.product(p1_combos, p2_combos):
                for cg in itertools.product(p1g, p2g):
                    pheno_counter[cls.genotype_to_phenotype(*cg)] += 1  # 不同基因型在群体中的先验概率通常不同，phenotype_freq
        else:  # 计算子代表现型在父母表现型给定下的条件概率,严谨的贝叶斯推导，以及父母基因型组合对子代的真实遗传概率。
            pheno_counter = Counter()
            # 计算 P(g | p)
            cond_map = cls.phenotype_conditional_mapping() if prob else cls.phenotype_to_genotypes_mapping()
            # 枚举 g1,g2 并累加子代表现型概率
            for p1g, w1 in cond_map[parent1_phenotype].items():
                for p2g, w2 in cond_map[parent2_phenotype].items():
                    for cg, pc in cls.child_genotype_distribution(p1g, p2g).items():  # map genotype->prob
                        pheno_counter[cls.genotype_to_phenotype(*cg)] += w1 * w2 * pc

        return cls.get_probability(pheno_counter, output_format="probs", sort=True)  # pheno_counter already sums to 1

    @classmethod
    def get_child_probability(cls, parent1_phenotype: str, parent2_phenotype: str, equal: bool = True) -> dict:
        """父母表现型→子代概率 """
        if equal:
            if hasattr(cls, 'Phenotype_Probs_Equal_Mapping'):
                key = cls.sorted_frozen(parent1_phenotype, parent2_phenotype)
                return cls.Phenotype_Probs_Equal_Mapping.get(key, {})
        else:
            if hasattr(cls, 'Phenotype_Probs_Mapping'):
                key = cls.sorted_frozen(parent1_phenotype, parent2_phenotype)
                return cls.Phenotype_Probs_Mapping.get(key, {})

        return cls.child_phenotype_distribution(parent1_phenotype, parent2_phenotype, equal, prob=False)

    @classmethod
    def get_parent_phenotypes(cls, child_phenotype: str, parent_phenotype: str = None) -> set[str]:
        """根据子女血型和另一个父母血型计算当前父母可能的血型"""
        # 单亲情况：返回所有可能的父母表现型
        if parent_phenotype is None:
            return {pheno for key, child_dist in cls.phenotype_probs_equal_mapping().items() if
                    child_dist.get(child_phenotype, 0) > 0 for pheno in key}
        # 双亲情况：反向应用遗传规则
        return {potential_phenotype for potential_phenotype in cls.phenotypes()
                if child_phenotype in cls.get_child_probability(potential_phenotype, parent_phenotype, equal=True)}

    @classmethod
    def get_parent_candidates(cls, child_phenotype: str, weights: dict | float | int = None) -> dict:
        """
        根据子代表现型返回可能的父母表现组合及概率,遗传层面,血型表现型是基因型的映射
        P(父母表现型 | 子表现型) ∝ P(父母基因型) * P(子表现型 | 父母基因型),P(p1,p2∣c)∝g1,g2∑P(g1,g2)P(c∣g1,g2)
        基因型层级的真实遗传分布，是更符合遗传规律的模型。表现型与基因型之间有多对一关系
        """
        # 处理权重参数
        if weights is None:
            weights = cls.genotype_freq()

        pheno_comb_probs = defaultdict(float)
        # 遍历所有唯一的父母基因型组合,计算联合概率 P(父母组合, 子表现型)
        for (gt1, gt2), child_dist in cls.genotype_probs_mapping().items():
            # 子表现型概率分布缓存
            phenotype_probs = defaultdict(float)
            for cg, prob in child_dist.items():
                child_pheno = cls.genotype_to_phenotype(*cg)
                phenotype_probs[child_pheno] += prob

            cond_prob = phenotype_probs.get(child_phenotype, 0)  # 条件概率
            if cond_prob <= 0:
                continue
            # 计算基因型组合的先验概率
            prior = weights[gt1] * weights[gt2] if isinstance(weights, dict) else weights
            joint_prob = prior * cond_prob  # 联合概率
            # 转换为表现型组合
            p1_pheno = cls.genotype_to_phenotype(*gt1)
            p2_pheno = cls.genotype_to_phenotype(*gt2)

            key = cls.sorted_frozen(p1_pheno, p2_pheno)  # 表现型组合去重
            pheno_comb_probs[key] += joint_prob  # 累加到对应表现型组合

        return cls.get_probability(pheno_comb_probs, output_format='probs', sort=True)

    @classmethod
    def get_parent_candidates_equal(cls, child_phenotype: str, weights: dict | float | int = None) -> dict:
        """
        根据子代表现型返回可能的父母表现组合及概率,观测层面, 等权平均模型：假设父母在其表现型下所有基因型等可能
        P(父母表现型 | 子表现型) ∝ P(C | P1, P2) * P(P1) * P(P2)
        各可能基因型等权,更贴合“只知表现型”的推理
        """
        # 处理权重参数
        if weights is None:
            weights = cls.phenotype_freq()

        pheno_comb_probs = defaultdict(float)
        # 遍历所有唯一的父母表现型组合,计算联合概率 P(父母组合, 子表现型)
        for key, child_dist in cls.phenotype_probs_equal_mapping().items():
            p1_pheno, p2_pheno = key
            cond_prob = child_dist.get(child_phenotype, 0)  # 条件概率
            if cond_prob <= 0:
                continue
            # 计算基因型组合的先验概率
            prior = weights[p1_pheno] * weights[p2_pheno] if isinstance(weights, dict) else weights
            joint_prob = prior * cond_prob  # 联合概率
            pheno_comb_probs[key] += joint_prob  # 累加到对应表现型组合

        return cls.get_probability(pheno_comb_probs, output_format='probs', sort=True)

    @classmethod
    def generate_allele(cls, weights: dict | list | tuple | float | int = None) -> str:
        """
        随机生成基因型
            weights (dict | list | tuple | float | int | None):
            - dict: 基因权重，格式如 {'A':0.22, ...}
            - list/tuple: 与基因型列表顺序对应的权重
            - float/int: 表示随意选择，不使用权重，均匀随机取一个
            - None: 使用默认频率 allele_freq()
        """
        alleles = cls.alleles()
        if isinstance(weights, (float, int)):
            return random.choice(alleles)
        probs = cls.normalize_weights(alleles, weights or cls.allele_freq())
        return random.choices(list(alleles), weights=probs, k=1)[0]

    @classmethod
    def generate_genotype(cls, weights: dict | list | tuple | float | int = None, size: int = 1) -> tuple | list[tuple]:
        """
        随机生成基因型
            weights (dict | list | tuple | float | int | None):
            - dict: 基因型权重，格式如 {('A','O'):0.22, ...}，需包含所有可能的基因型组合
            - list/tuple: 与基因型列表顺序对应的权重
            - float/int: 表示随意选择，不使用权重，均匀随机取多个/均匀随机取一个
            - None: 使用默认频率 genotype_freq()
        """
        if isinstance(weights, (float, int)):
            result = random.choices(cls.genotype_iter(unique=False), k=size)
            return result if size > 1 else result[0]
        genotypes = cls.genotypes()
        probs = cls.normalize_weights(genotypes, weights or cls.genotype_freq())
        result = random.choices(list(genotypes), weights=probs, k=size)
        return result if size > 1 else result[0]

    @classmethod
    def generate_phenotype(cls, weights: dict | list | tuple | float | int = None, size: int = 1) -> str | list:
        """
        随机生成血型 blood_types ('A', 'B', 'O', 'AB')
        """
        phenotypes = cls.phenotypes()
        probs = cls.normalize_weights(phenotypes, weights or cls.phenotype_freq())  # 获取表现型概率分布
        result = np.random.choice(phenotypes, size=size, p=np.array(probs))
        return result.tolist() if size > 1 else result.item()

    @classmethod
    def genotype_iter_by_freq(cls, last_size: int = 10 ** 6, size: int = 360, callback=None):
        """
        基于当前频率作为先验，生成若干 genotype，
        增量更新等位基因频率（基于最新频率作为先验）
        支持多次调用自动累积先验规模
        last_size: int 上一次累计的样本数
        """
        old_freq = cls.allele_freq().copy()
        genotypes = cls.generate_genotype(weights=cls.genotype_freq(), size=size)
        data = []
        for combo in genotypes:
            yield combo
            data.extend(combo)

        if data:
            new_freq = cls.set_allele_freq(freq=old_freq, prior_size=last_size * 2, data=data)
            total = last_size + len(data) / 2  # 每个 genotype 有两个 allele
            if callback:
                callback(total, old_freq, new_freq)

    @classmethod
    def genotype_combinations(cls):
        '''
        按需生成所有基因型组合 C(n+k−1,k)
        n=len(key_to_index)
        k=num_axes+1
        math.comb(n + k - 1, k)
        '''
        num_axes = len(cls.AXES)
        for combo in itertools.combinations_with_replacement(cls.genotypes(), r=num_axes + 1):
            yield combo


class AlleleA(Allele):
    def __init__(self):
        super().__init__('A')
        # super().__init__(*args, **kwargs)
        self.value = self.allele_vector('A')  # (1, 0)


class AlleleB(Allele):
    def __init__(self):
        super().__init__('B')
        self.value = self.allele_vector('B')  # (0, 1)


class AlleleO(Allele):
    def __init__(self):
        super().__init__('O')
        self.value = self.allele_vector('O')  # (0, 0)

    def get_antigens(self):
        # O 等位基因不表达抗原
        return set()

    def get_antibodies(self):
        # O 型没有对应的抗原，但体内通常会产生抗 A 和抗 B 抗体
        return {'A', 'B'}

    def get_vector(self):
        return self.value


class BloodType:
    """血型类，处理等位基因和表现型逻辑"""

    def __init__(self, alleles: tuple[Allele | str, Allele | str] = None, phenotype: str = None, cytoplasm: str = None):
        """
        :param alleles: 核基因的等位基因对,等位基因组合，如 ('A', 'O'),如 ('Rf', 'rf')
        :param cytoplasm: 细胞质类型，'S' 或 'N'
        """
        if alleles is not None:
            self.reveal = True  # 当前对象的基因型是否可以被完全确定
        elif phenotype:
            alleles, w = Allele.get_random_genotype(phenotype)
            self.reveal = (w == 1)
        else:
            alleles = Allele.generate_genotype(size=1)  # 标准化排序，如 ('A', 'O') → ('A', 'O')
            self.reveal = False

        self.alleles = tuple(a if isinstance(a, Allele) else Allele(a) for a in alleles)
        self._phenotype = phenotype  # 保留原始表型
        self.system = Allele.system()  # 血型系统，默认ABO
        self.cytoplasm = cytoplasm

    @property
    def genotype(self) -> tuple | None:
        return Allele.sorted_frozen(self.alleles[0].name, self.alleles[1].name) if self.reveal else None

    @property
    def phenotype(self) -> str | None:
        return Allele.genotype_to_phenotype(*self.genotype) if self.reveal else getattr(self, "_phenotype", None)

    @property
    def antigens(self) -> set[str]:
        if self.reveal:
            return self.alleles[0].get_antigens() | self.alleles[1].get_antigens()
        return Allele.phenotype_to_antigens(self.phenotype)

    @property
    def antibodies(self) -> set[str]:
        return Allele.antigens_to_antibodies(self.antigens)

    def get_gamete(self) -> str:
        if self.reveal:
            return random.choice([self.alleles[0].name, self.alleles[1].name])
        genotype, _ = Allele.get_random_genotype(self.phenotype)
        return random.choice(genotype)

    def is_male_sterile(self):
        """
        判断是否为雄性不育：细胞质为 S 且核基因为 rf/rf
        """
        return self.cytoplasm == 'S' and self.alleles == ('rf', 'rf')

    @staticmethod
    def child_alleles(blood_type1: 'BloodType', blood_type2: 'BloodType') -> tuple[str, ...]:
        # 从父母处各随机获取一个等位基因
        gamete_self = blood_type1.get_gamete()  # 母本贡献
        gamete_partner = blood_type2.get_gamete()  # 父本贡献
        return Allele.sorted_frozen(gamete_self, gamete_partner)

    def cross(self, other: 'BloodType'):
        """
        与另一植株杂交，生成子代
        :return: 子代基因型 (RicePlant 对象)
        """
        # 子代核基因：从父母各随机取一个等位
        child_alleles = (
            random.choice(self.alleles),  # 母本贡献
            random.choice(other.alleles)  # 父本贡献
        )
        # 排序等位基因以便统一表示（如 ('Rf', 'rf') 和 ('rf', 'Rf') 视为相同）
        child_alleles = tuple(sorted(child_alleles, key=lambda a: a.name))
        # 子代细胞质来自母本（假设当前植株为母本）
        return BloodType(child_alleles, cytoplasm=self.cytoplasm)

    def __repr__(self):
        return f"BloodType(alleles={self.alleles}, phenotype='{self.phenotype}')"

    def is_valid_child(self, parent1: 'BloodType', parent2: 'BloodType') -> bool:
        if not self.genotype or not parent1.genotype or not parent2.genotype:
            return True  # 如果基因型未知，无法验证
        return Allele.is_valid_child_by_alleles(self.genotype, parent1.genotype, parent2.genotype)

    @staticmethod
    def is_compatible(donor: 'BloodType', recipient: 'BloodType') -> bool:
        """
        输血相容性检查：检查捐赠者血型 donor 与受血者血型 recipient 是否兼容
        donor 的抗原不能出现在 recipient 的抗体中 如果 donor 的抗原出现在 recipient 的抗体中，则不兼容
        因此：如果 donor.antigens 与 recipient.antibodies 有交集，则不兼容。
        # not any(antigen in recipient_antibodies for antigen in donor_antigens) donor in transfusion_rules[recipient]
        # (donor_bin & antibody_bin) == 0
        """
        return donor.antigens.isdisjoint(recipient.antibodies)


if __name__ == "__main__":
    print(Allele.__doc__)
    print(Allele.system(), Allele.alleles())
    print('axes', Allele.get_axes_by_allele_vector(Allele.allele_vector_mapping()))
    genotype_map = Allele.genotype_to_phenotype_mapping()
    print('genotype_to_phenotype', genotype_map)
    genotype_db = Allele.phenotype_to_genotypes_mapping()
    print('phenotype_to_genotypes', genotype_db)
    print('A', Allele.phenotype_to_genotypes('A'),
          Allele.allele_vector('O'),
          'AB:', Allele.genotype_to_phenotype('A', 'B'), Allele.genotype_to_phenotype('B', 'A'))
    print(Allele.vector_to_state((0, 0)))
    t1 = Allele.tensor_product(Allele.allele_state('A'), Allele.allele_state('B'))
    print("Combined AB:", Allele.combine_quantum(Allele.allele_vector('A'), Allele.allele_vector('B')), t1,
          Allele.separate_product_state(t1), Allele.is_entangled(t1), Allele.genotype_state('A', 'B'))
    t1 = Allele.tensor_product(Allele.allele_state('A'), Allele.allele_state('A'))
    print("Combined AA:", Allele.combine_quantum(Allele.allele_vector('A'), Allele.allele_vector('A')), t1,
          Allele.separate_product_state(t1), Allele.is_entangled(t1), Allele.genotype_state('A', 'A'))
    t1 = Allele.tensor_product(Allele.allele_state('A'), Allele.allele_state('O'))
    print("Combined AO:", Allele.combine_quantum(Allele.allele_vector('A'), Allele.allele_vector('O')), t1,
          Allele.separate_product_state(t1), Allele.is_entangled(t1), Allele.original_states(t1),
          Allele.genotype_state('A', 'O'))
    t1 = Allele.tensor_product(Allele.allele_state('O'), Allele.allele_state('O'))
    print("Combined OO:", Allele.combine_quantum(Allele.allele_vector('O'), Allele.allele_vector('O')), t1,
          Allele.separate_product_state(t1), Allele.is_entangled(t1), Allele.original_states(t1),
          Allele.genotype_state('O', 'O'))
    print(Allele.state_to_phenotype(Allele.genotype_state('A', 'B')))
    print(Allele.state_to_phenotype(Allele.genotype_state('A', 'O')))
    print(Allele.state_to_phenotype(Allele.genotype_state('O', 'O')))

    print('phenotype', Allele.generate_phenotype(), 'genotypes', Allele.genotype_iter(False))
    print('frq', Allele.genotype_freq())
    print(Allele.phenotype_freq())
    print('ab antigens', Allele.vector_to_antigens((1, 1)), 'antibodies',
          Allele.antigens_to_antibodies({'A'}), Allele.antigens_to_antibodies({'O'}),
          Allele.antigens_to_antibodies({'A', 'B'}))

    a1 = Allele.generate_genotype()
    a2 = Allele.generate_genotype()
    p1 = Allele.allele_to_phenotype(*a1)
    p2 = Allele.allele_to_phenotype(*a2)
    print(a1, a2, 'then', Allele.child_genotype_distribution(a1, a2))
    print(p1, 'and', p2, 'then', Allele.child_phenotype_distribution(p1, p2, True))
    print(p1, 'and', p2, 'then', Allele.child_phenotype_distribution(p1, p2, False, False))
    print(p1, 'and', p2, 'then', Allele.child_phenotype_distribution(p1, p2, False, True))
    print(p1, 'and', p2, 'then', Allele.get_child_probability(p1, p2))

    print(p1, 'c and p1', p2, 'then parent2', Allele.get_parent_phenotypes(p1, p2))
    print('parent_possible')
    print('A', Allele.get_parent_phenotypes('A'))
    print('O', Allele.get_parent_phenotypes('O'))
    print('AB', Allele.get_parent_phenotypes('AB'))

    child_probability = {','.join(sorted(p)): Allele.get_child_probability(*p)
                         for p in itertools.product(Allele.phenotypes(), repeat=2)}
    print(child_probability)
    print('genotype_probs', Allele.genotype_probs_mapping())
    print('phenotype_probs', Allele.phenotype_probs_mapping())
    print('phenotype_probs_equal', Allele.phenotype_probs_equal_mapping())

    print('union', AlleleB.union_antigens(AlleleA(), AlleleO()))
    print('compatible O->A', Allele.is_compatible_phenotype('O', 'A'))
    print('genotype_transfusion', Allele.genotype_transfusion_mapping())
    print('allele_state', Allele.allele_state_mapping())

    print('AlleleB', AlleleB.allele_binary('B'), Allele.allele_binary_mapping(),
          AlleleB.allele_binary('A'))

    print('parent_candidates', AlleleB.get_parent_candidates(child_phenotype='A', weights=None))
    print('parent_candidates_equal', AlleleB.get_parent_candidates_equal(child_phenotype='A', weights=None))
    print('phenotype_conditional', AlleleB.phenotype_conditional_mapping())

    Allele.set_allele_freq(freq={'A': 0.2, 'B': 0.1, 'O': 0.7})

    print(Allele.phenotype_freq(), Allele.genotype_freq(), Allele.allele_freq())
    print('AO:', Allele.genotype_to_phenotype(genotype='AO'))

    print('vector_to_antigen', Allele.vector_to_antigen_mapping())
    print(Allele.vector_to_antibody_mapping())

    print('genotype_to_vector_mapping', Allele.genotype_to_vector_mapping())
    print('phenotype_to_genotypes_mapping', Allele.phenotype_to_genotypes_mapping())
    print('phenotype_transfusion', Allele.phenotype_transfusion_mapping())
    print(Allele.phenotype_to_antigens('A'))
    print(Allele.phenotype_to_genotypes('A'))

    print(Allele.generate_phenotype(size=30))
    print('generate_phenotype', Allele())
    print(list(Allele.genotype_combinations()))

    print(Allele._expressed_cache)
    print('vars', Allele.get_vars())
    print('cache', Allele.get_cache())
    print(Allele.re_cache('ALLELES', rebuild=True))

    print(dir(AlleleB), '\n', AlleleO.__dict__)
    Allele.rebuild_constants()
    print(Allele.get_vars())

    print('repr', Allele.__repr__)

    genotypes_iter = Allele.genotype_iter_by_freq(1000, 360)
