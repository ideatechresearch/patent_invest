from rime.allele import *
import uuid
import random
from typing import List, Optional, Tuple


class Human:
    """基础人类类，支持遗传特征继承"""
    _id_counter = 0  # 类级ID计数器
    _registry: dict[str, 'Human'] = {}  # 全局注册表

    def __init__(self,
                 generation: int = 0,
                 parents: Optional[tuple['Human', 'Human']] = None,
                 gender: Optional[str] = None,
                 blood_type: Optional[BloodType] = None):
        """
        :param generation: 代数（0为初始代）
        :param parents: 父代对象元组
        :param gender: 性别（'male'/'female'）
        :param blood_type: 血型对象，若未提供则随机生成
        """
        self.id = self._generate_id()
        self.generation = generation
        self.parents = parents
        self.children: list['Human'] = []  # 子代对象列表

        self.gender = gender if gender else random.choice(['male', 'female'])
        self.blood_type = blood_type if blood_type else BloodType()
        self.possible_types = {self.blood_type.phenotype} if blood_type else set(Allele.phenotypes())

        self.members: dict[str, dict] = {}  # {id: {'pheno':str, 'parents':[id1,id2], ...}}# 存储待解决的家族树问题
        self._register()

    def _register(self):
        """注册实例到全局注册表"""
        self._registry[self.id] = self

    @classmethod
    def _generate_id(cls) -> str:
        """生成唯一ID：类名前缀+计数器+UUID"""
        cls._id_counter += 1
        return f"Human_{cls._id_counter}_{uuid.uuid4().hex[:6]}"

    @classmethod
    def get_by_id(cls, human_id: str) -> Optional['Human']:
        """通过ID获取人类实例"""
        return cls._registry.get(human_id)

    def add_parent(self, parent: 'Human'):
        if not self.parents:
            self.parents = (parent,)
        elif parent not in self.parents:
            self.parents = self.parents + (parent,)
        if self not in parent.children:
            parent.children.append(self)

    def familytree(self, member_id: str, pheno: str, parents=None):
        self.members[member_id] = {'pheno': pheno, 'parents': parents or []}

    @staticmethod
    def generate_familytree(family_size: int) -> dict:
        """生成随机家族树（简化实现）"""
        # 实际实现需要更复杂的家族树生成逻辑
        # 这里返回一个简化的结构
        tree = {}
        phenotypes = Allele.phenotypes()
        for i in range(family_size):
            # 随机生成血型
            blood_type = random.choice(phenotypes)
            tree[i] = {
                'blood_type': blood_type,
                'parents': [] if i < 2 else [i - 2, i - 1],  # 简化的父母关系
                'children': [] if i >= family_size - 2 else [i + 1, i + 2]  # 简化的子女关系
            }
        return tree

    @staticmethod
    def select_revealed_nodes(family_tree: dict, ratio: float) -> dict:
        """选择要披露的血型节点"""
        revealed = {}
        node_count = len(family_tree)
        reveal_count = max(1, int(node_count * ratio))

        # 优先选择叶子节点（最年轻一代）
        leaf_nodes = [i for i in family_tree if not family_tree[i]['children']]
        random.shuffle(leaf_nodes)

        for i in range(min(reveal_count, len(leaf_nodes))):
            node_id = leaf_nodes[i]
            revealed[node_id] = family_tree[node_id]['blood_type']

        return revealed

    @staticmethod
    def get_blood_type_constraints(person: 'Human') -> set[str]:
        """根据遗传规则获取可能的血型"""
        if person.blood_type is not None:
            return {person.blood_type.phenotype}

        # 如果没有父母信息，返回所有可能血型
        if not person.parents:
            return set(Allele.phenotypes())

        # 根据父母血型推断可能血型
        possible_types = set()

        # 获取父母所有可能的血型组合
        parent1, parent2 = person.parents
        for p1_type in parent1.possible_types:
            for p2_type in parent2.possible_types:
                # 根据父母血型组合计算子女可能血型
                child_possible = Allele.get_child_probability(p1_type, p2_type)
                possible_types.update(child_possible.keys())

        return possible_types

    def reproduce(self, partner: 'Human', num_children: int = 1) -> List['Human']:
        """与配偶繁殖后代"""
        if self.gender == partner.gender:
            raise ValueError("Same gender cannot reproduce biologically")

        children = []
        for _ in range(num_children):
            # 从父母处各随机获取一个等位基因
            child_alleles = BloodType.child_alleles(self.blood_type, partner.blood_type)

            # 创建子代对象
            child = Human(
                generation=max(self.generation, partner.generation) + 1,
                parents=(self, partner),
                gender=random.choice(['male', 'female']),
                blood_type=BloodType(alleles=child_alleles)
            )
            self.children.append(child)
            partner.children.append(child)
            children.append(child)

        return children

    def get_ancestry(self) -> dict:
        """获取家族树结构"""
        return {
            'self': self.id,
            'gender': self.gender,
            'generation': self.generation,
            'blood_type': self.blood_type.phenotype,
            'parents': [p.get_ancestry() if p else None for p in self.parents] if self.parents else None,
            'children': [c.get_ancestry() for c in self.children]
        }

    def __repr__(self):
        return f"<Human {self.id} Gen{self.generation} {self.gender} {self.blood_type.phenotype}>"


# ======================
# 算法参数配置
# ======================
POPULATION_SIZE = 100  # 种群大小
GENOME_LENGTH = 20  # 基因长度（移动步数）
ARCHIVE_SIZE = 200  # 档案最大容量
K_NEAREST = 15  # 计算新颖性时考虑的最近邻居数
THRESHOLD = 1.25  # 初始新颖性阈值
MUTATION_RATE = 0.1  # 变异概率


# ======================
# 个体类定义
# ======================
class Individual:
    def __init__(self):
        self.genome = np.random.choice([0, 1, 2, 3], size=GENOME_LENGTH)  # 基因编码（0:上,1:下,2:左,3:右）
        self.behavior = None  # 行为特征（最终坐标）
        self.novelty = 0.0  # 新颖性得分

    def calculate_behavior(self):
        """ 按需计算行为特征（避免重复计算） """
        if self.behavior is None:
            self.behavior = simulate_movement(self.genome)


# ======================
# 迷宫环境模拟
# ======================
def simulate_movement(genome):
    """
    模拟个体在迷宫中的移动
    返回最终坐标(x,y)作为行为特征
    """
    x, y = 0, 0  # 起点
    for move in genome:
        # 简单迷宫边界约束（假设迷宫大小为5x5）
        prev_x, prev_y = x, y  # 记录移动前坐标
        if move == 0:
            y = min(y + 1, 4)  # 上
        elif move == 1:
            y = max(y - 1, 0)  # 下
        elif move == 2:
            x = max(x - 1, 0)  # 左
        elif move == 3:
            x = min(x + 1, 4)  # 右

    return (x, y)


# ======================
# 新颖性计算
# ======================
def calculate_novelty(individual, population, archive):
    """
    计算个体新颖性得分
    """
    distances = []

    # 计算与种群中所有个体的距离
    for ind in population:
        if ind != individual:
            dx = individual.behavior[0] - ind.behavior[0]
            dy = individual.behavior[1] - ind.behavior[1]
            distances.append(np.sqrt(dx ** 2 + dy ** 2))

    # 计算与档案中个体的距离
    for record in archive:
        dx = individual.behavior[0] - record[0]
        dy = individual.behavior[1] - record[1]
        distances.append(np.sqrt(dx ** 2 + dy ** 2))

    if not distances:
        return 10.0  # 初始奖励探索

    # 取k个最近邻的平均距离
    distances.sort()
    k_nearest = distances[:K_NEAREST]
    return np.mean(k_nearest) if k_nearest else 0


# ======================
# 进化操作
# ======================
def evolve(population, archive):
    # 评估所有个体
    # 第一阶段：统一生成所有个体的行为特征
    for ind in population:
        ind.behavior = simulate_movement(ind.genome)  # 确保所有behavior已初始化

    # new_individuals = [ind for ind in population if ind.behavior is None]
    # for ind in new_individuals:
    #     ind.calculate_behavior()

    # 第二阶段：统一计算新颖性
    for ind in population:
        ind.novelty = calculate_novelty(ind, population, archive)  # 此时所有behavior已就绪

    # # 动态更新阈值
    # if len(archive) < ARCHIVE_SIZE * 0.3:
    #     THRESHOLD *= 0.95  # 加快阈值下降
    # elif len(archive) > ARCHIVE_SIZE * 0.8:
    #     THRESHOLD *= 1.05

    # 更新档案（先进先出队列）
    for ind in population:
        if ind.novelty > THRESHOLD:
            archive.append(ind.behavior)
    while len(archive) > ARCHIVE_SIZE:
        archive.popleft()

    # 选择（基于新颖性排序）
    population.sort(key=lambda x: x.novelty, reverse=True)
    selected = population[:int(POPULATION_SIZE * 0.5)]

    # 交叉与变异
    new_pop = []
    while len(new_pop) < POPULATION_SIZE:
        # 选择父母
        # parents = np.random.choice(selected, 2, replace=False)
        parent1 = np.random.choice(selected)
        parent2 = np.random.choice(selected)

        # 单点交叉
        crossover_point = np.random.randint(1, GENOME_LENGTH)
        child_genome = np.concatenate((
            parent1.genome[:crossover_point],
            parent2.genome[crossover_point:]
        ))

        # 变异
        for i in range(len(child_genome)):
            if np.random.rand() < MUTATION_RATE:
                child_genome[i] = np.random.choice([0, 1, 2, 3])

        # 创建新个体
        new_ind = Individual()
        new_ind.genome = child_genome
        new_pop.append(new_ind)

    return new_pop, archive


def print_debug(population, generation):
    print(f"\n=== Generation {generation} ===")
    print("Behaviors:", [ind.behavior for ind in population[:3]])
    print("Novelties:", [round(ind.novelty, 2) for ind in population[:3]])
    print("Archive samples:", list(archive)[-3:])


class NoveltySearch:
    def __init__(self, pop_size=50, k=5, archive_threshold=0.3, mutation_rate=0.1):
        self.pop_size = pop_size  # 种群大小
        self.k = k  # 近邻数量
        self.archive_threshold = archive_threshold  # 存档新奇性阈值
        self.mutation_rate = mutation_rate  # 变异率
        self.archive = []  # 存档用于记录新奇个体

    def initialize_population(self):
        return [np.random.rand(2) for _ in range(self.pop_size)]

    def mutate(self, individual):
        return individual + np.random.normal(0, self.mutation_rate, size=individual.shape)

    def compute_novelty(self, individual, population):
        from scipy.spatial import distance
        distances = [distance.euclidean(individual, other) for other in population]
        distances.sort()
        return np.mean(distances[:self.k])  # 计算k近邻的平均距离

    def evolve(self, generations=100):
        population = self.initialize_population()
        for gen in range(generations):
            novelties = [self.compute_novelty(ind, population + self.archive) for ind in population]

            # 选择最具新奇性的个体
            selected_indices = np.argsort(novelties)[-self.pop_size // 2:]
            new_population = [self.mutate(population[i]) for i in selected_indices]

            # 存档高新奇性个体
            for i in selected_indices:
                if novelties[i] > self.archive_threshold:
                    self.archive.append(population[i])

            population = new_population + self.initialize_population()[:self.pop_size - len(new_population)]

            print(f'Generation {gen + 1}, Avg Novelty: {np.mean(novelties):.4f}, Archive Size: {len(self.archive)}')


if __name__ == "__main__":
    # 初始化
    # ns = NoveltySearch()
    # archive = ns.evolve(generations=50)
    from collections import deque
    population = [Individual() for _ in range(POPULATION_SIZE)]
    archive = deque(maxlen=ARCHIVE_SIZE)
    # 进化循环
    for generation in range(100):
        population, archive = evolve(population, archive)
        # 输出统计信息
        avg_novelty = np.mean([ind.novelty for ind in population])
        print(f"Gen {generation:3d} | Avg Novelty: {avg_novelty:.2f} | Archive: {len(archive)}")

    print("首次进化后示例:", population[0].behavior, population[0].genome)
    print([ind.novelty for ind in population])
