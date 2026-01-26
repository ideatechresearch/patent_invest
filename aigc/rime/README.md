# RIME - 多领域数学建模与计算框架

RIME 是一个跨领域的 Python 数学建模与计算框架，涵盖魔方求解双重建模（sticker-level + cubie-level）、遗传学计算、环形数据结构、金字塔神经网络等多个方向的算法实现与可视化。

## 项目结构

```
rime/
├── base.py          # 基础工具类：属性代理、类属性缓存装饰器
├── allele.py        # 遗传学：ABO血型系统建模、基因型/表现型计算
├── circular.py      # 环形数据结构、矩阵面操作、金字塔投影
├── pyramid.py       # 金字塔神经网络、旋转对称矩阵变换
├── cube.py          # 魔方建模：NxN魔方贴纸级状态表示与基础求解
├── cubie.py         # 魔方建模：块级群论建模、Kociemba两阶段算法
├── cubedraw.py      # 魔方可视化：Pygame 3D渲染与交互
├── dice.py          # 骰子特征分析、游戏触发器规则
└── body.py          # 遗传进化：人类血型遗传、新颖性搜索算法
```

## 主要模块

### 1. 魔方贴纸级系统 ([cube.py](rime/cube.py))

完整的 NxN 魔方数学建模与求解系统。

**核心特性：**
- 支持 3x3 到 NxN 阶魔方的完整状态表示
- 贴纸级 (sticker-level) 状态表示，保证物理可达性
- 基于群论的旋转操作与状态验证
- 多种求解算法：BFS、IDA*、十字求解、中心块修正
- 标准魔方记法解析：`U`, `U'`, `U2`, `Rw`, `2Rw2` 等

**核心类：**
- `CubeBase`: 魔方数学基础类，包含坐标系统、几何变换、群论约束
- `RubiksCube`: 魔方实现类，提供旋转、打乱、求解接口

```python
from rime.cube import RubiksCube

# 创建 3x3 魔方
cube = RubiksCube(n=3)

# 应用标准记法动作
cube.apply_move("R")
cube.apply_move("U'")

# 打乱并求解
scramble = cube.scramble(20)
cube.apply(scramble)
solution = cube.solve()
```

**数学特性：**
- 坐标系：右手笛卡尔坐标系，+X→R, +Y→U, +Z→F
- 群论约束：角块朝向 (Z₃)、边块朝向 (Z₂)、排列奇偶性
- 状态嵌入：20 维向量 (8 角块 + 12 边块)
- 启发式函数：角块置换、中心块误差评估

### 2. 魔方块级系统 ([cubie.py](rime/cubie.py))

基于群论的魔方块级建模与 Kociemba 两阶段算法实现。

**核心特性：**
- 块级状态表示：`CubieState` 包含角块/边块的置换与朝向
- 群论建模：`G = (S₈ × S₁₂) ⋉ (ℤ₃⁷ × ℤ₂¹¹)`，状态空间约 4.3×10¹⁹
- 角块 S₈ / 边块 S₁₂ 是置换群，ℤ₃⁷ / ℤ₂¹¹ 是朝向群，半直积表示方向和排列的半独立性。
- 两阶段算法：Phase-1 (方向修正 + 边块分层) → Phase-2 (排列求解)
- 剪枝表：CO×EO (2187×2048)、UD-slice (495)、角块/边块排列 (40320)
- 同态投影：`CubieMove → Phase1Move/Phase2Move` 坐标映射

**魔方块级状态流程图：**
```scss
CubieState (角块+边块)
        │
        │ Phase1_project → Phase1Coord (CO, EO, UD-slice)
        │             ↑
        │       Phase1Move φ
        │
        │ Phase1_search (CO=0, EO=0, UD-slice separated)
        │
        │ Phase2_project → Phase2Coord (corner_perm, edge_perm, slice_perm)
        │             ↑
        │       Phase2Move φ
        │
        └─ Phase2_search → Solved
```

**核心类：**
- `CubieState`: 块级状态，包含 8 角块 + 12 边块的置换与朝向
- `CubieMove`: 群元素，支持半直积作用、合成、逆元
- `Phase1Coord`: Phase-1 坐标 (CO, EO, UD-slice)
- `Phase2Coord`: Phase-2 坐标 (角块/边块排列)
- `CubieBase`: 继承自 `CubeBase`，提供两阶段搜索接口


```python
from rime.cubie import CubieState, CubieMove, CubieBase

# 创建已解状态
state = CubieState.solved()

# 应用基本动作
move = CubieMove.from_rotation(axis=0, layer=1, direction=1)  # R
state = move.act(state)

# 两阶段求解
CubieBase.build_pruning_table()  # 构建剪枝表（首次运行）
moves, cubie_move, final_state = CubieBase.solve_kociemba(state)
print(f"Solution: {moves}")
```

**算法细节：**
- **Phase-1 目标**: CO=0, EO=0, UD-slice 分离（边块回到中层）
- **Phase-2 目标**: 角块/边块排列还原（在 G₁ = ⟨U,D,R²,L²,F²,B²⟩ 子群内）
- **剪枝策略**: 联合 CO×EO + UD-slice 启发式，限制搜索深度
- **群同态**: `φ: CubieMove → Phase1Move/Phase2Move` 满足 `φ(m₁∘m₂) = φ(m₁)∘φ(m₂)`

**数学基础：**
- 状态空间: |G| ≈ 4.3×10¹⁹
- 群结构: `G = (S₈ × S₁₂) ⋉ (ℤ₃⁷ × ℤ₂¹¹)`
- Phase-1 子群: `G₁ = ⟨U,D,R²,L²,F²,B²⟩`, |G|/|G₁| ≈ 1.95×10¹⁰
- 剪枝表大小: CO_EO (4.5M) + UD (495) + CO (40320) + EDGE (40320) + SLICE (24)

**剪枝表构建**
- CO×EO (2187×2048) → Phase1 方向搜索
- UD-slice (495) → Phase1 中层边
- 角块排列 (40320)、非 slice 边排列 (40320) → Phase2 搜索
- slice 内部排列 (24) → Phase2 局部自由度

> 首次构建剪枝表约耗时 10-20 秒（视 CPU 而定），数据会缓存到 `data/` 目录


### 3. 魔方可视化 ([cubedraw.py](rime/cubedraw.py))

基于 Pygame 的 3D 魔方可视化与交互系统。

**功能：**
- 3D 立方体实时渲染
- 局部旋转动画
- 鼠标拖拽交互（视角旋转、层转动）
- 自动播放打乱序列
- 2D 展开图显示

```python
from rime.cube import RubiksCube
from rime.cubedraw import RubiksCubeDraw

cube = RubiksCube(n=5)
app = RubiksCubeDraw(cube)
app.run()
```

**控制说明：**
- 左键拖拽：旋转视角
- 右键拖拽：推断并执行层转动
- `A`: 切换自动旋转
- `Space`: 暂停/恢复动画
- `P`: 执行单次随机动作
- `S`: 生成并播放打乱序列
- `R`: 重置魔方

### 4. 遗传学系统 ([allele.py](rime/allele.py))

完整的 ABO 血型系统遗传学建模。

**核心类：**
- `AlleleBase`: 遗传学基础工具，向量映射、量子态表示
- `ABOSystem`: ABO 血型系统定义
- `Allele`: 等位基因与基因型操作
- `BloodType`: 血型类，支持输血相容性检查

```python
from rime.allele import Allele, BloodType

# 基因型转表现型
phenotype = Allele.genotype_to_phenotype('A', 'O')  # 'A'

# 子代概率计算
prob = Allele.get_child_probability('A', 'B')
# {'A': 0.1875, 'AB': 0.5625, 'B': 0.1875, 'O': 0.0625}

# 输血相容性
compatible = Allele.is_compatible_phenotype('O', 'A')  # True
```

**数据结构：**
- 等位基因向量：`A=(1,0)`, `B=(0,1)`, `O=(0,0)`
- 抗原/抗体映射：自动从向量推导
- 遗传概率矩阵：支持基因型和表现型两层概率
- 群体频率：Hardy-Weinberg 平衡计算

### 5. 环形数据结构 ([circular.py](rime/circular.py))

支持动态游标、容量限制、持久化的环形数据结构。

**核心类：**
- `CircularBand`: 环形缓冲区，支持旋转、转置、镜像等操作
- `MatrixFace`: 矩阵面操作，旋转对称、分块投影

```python
from rime.circular import CircularBand

band = CircularBand(['A', 'B', 'C'], capacity=5)
band.append('D')
band.rotate(1)  # 循环移动
band.transpose(4)  # 块转置
```

**金字塔投影：**
- 多层环形数据投影到方阵
- 支持 4 象限旋转对称切分
- 可逆变换：band ↔ matrix ↔ 3d blocks

### 6. 金字塔神经网络 ([pyramid.py](rime/pyramid.py))

基于递增环和旋转对称结构的混合神经网络架构。

**核心类：**
- `PYRAMID`: 金字塔数据结构，多层 band 管理
- `PyramidNN`: Transformer + GNN 混合架构
- `NextBandHead`: 层间预测头

```python
from rime.pyramid import PYRAMID, PyramidNN

pyramid = PYRAMID(max_layers=9)
pyramid.build(genotypes_iter)

nn = PyramidNN(dim=2, hidden=48)
outputs = nn.forward_pyramid(encoded_bands)
```

**特性：**
- 层内环状传播（邻域聚合）
- 层间交叉注意力
- 支持 band 编码与嵌入学习

### 7. 骰子特征系统 ([dice.py](rime/dice.py))

三骰子游戏特征分析与触发器系统。

**功能：**
- 骰子特征提取：顺子、三同、质数等
- 游戏触发器：优先级排序的效果系统
- 批量特征计算与规则匹配

```python
from rime.dice import dice_feature_game

features = dice_feature_game((4, 4, 4))
# ['四之恶', '三相之力']
```

**触发器示例：**
| 特征 | 条件 | 效果 |
|------|------|------|
| 三相之力 | 三枚相同 | 启用三种塔系 |
| 极限呈现 | 总和 > 16 | 极限表现 |
| 保底 | 1,2,3 顺子 | 稳定输出 |

### 8. 进化算法 ([body.py](rime/body.py))

遗传进化与新颖性搜索算法实现。

**核心类：**
- `Human`: 人类遗传模型，支持血型继承
- `NoveltySearch`: 新颖性搜索算法
- `Individual`: 进化个体

## 依赖项

```bash
pip install numpy pygame scipy
```

- `numpy`: 数值计算、数组操作
- `pygame`: 可视化渲染
- `scipy`: 科学计算（可选，用于距离计算）

## 快速开始

### 魔方贴纸级操作

```bash
python -m rime.cube
python -m rime.cubedraw
```

### 魔方块级求解

```python
from rime.cubie import CubieState, CubieMove, CubieBase

# 构建剪枝表（首次运行，数据会保存到 data/ 目录）
CubieBase.build_pruning_table()

# 打乱魔方
state = CubieState.solved()
for _ in range(10):
    state = random.choice(list(CubieMove.phase1_moves().values())).apply(state)

# 两阶段求解
moves, cubie_move, final_state = CubieBase.solve_kociemba(state)
print(f"Solution moves: {len(moves)}")
print(f"Final state solved: {final_state == CubieState.solved()}")
# 检查最终状态
assert final_state.is_solved()
```

### 遗传学计算

```python
from rime.allele import Allele

# 打印系统信息
print(Allele.system())  # 'ABO'
print(Allele.alleles())  # ('A', 'B', 'O')

# 计算子代概率
prob = Allele.get_child_probability('A', 'B')
print(f'A x B → {prob}')
```

### 金字塔投影

```python
from rime.pyramid import PYRAMID
from rime.allele import Allele

genotypes_iter = Allele.genotype_iter_by_freq(1000, 360)
pyramid = PYRAMID(max_layers=9)
pyramid.build(genotypes_iter)

# 获取 19x19 矩阵投影
matrix = pyramid.to_matrix(fill_center_with=('O', 'O'))
```

## 数学模型

### 魔方贴纸级状态空间

- **状态表示**: `(6, n, n)` 数组，贴纸级编码
- **群约束**:
  - 角块朝向和 ≡ 0 (mod 3)
  - 边块朝向和 ≡ 0 (mod 2)
  - 排列奇偶性一致

### 魔方块级群结构

- **状态表示**: `CubieState = (corners_perm, corners_ori, edges_perm, edges_ori)`
- **群结构**: `G = (S₈ × S₁₂) ⋉ (ℤ₃⁷ × ℤ₂¹¹)`
  - 角块: 8! × 3⁷ = 88,179,840 种状态（受朝向约束）
  - 边块: 12! × 2¹¹ = 495,466,560,000 种状态（受翻转约束）
  - 总状态: |G| ≈ 4.3 × 10¹⁹
- **Phase-1 投影**: `(CO, EO, UD-slice)`，维度 2187 × 2048 × 495
- **Phase-2 投影**: `(corner_perm, edge_perm, slice_perm)`，维度 40320 × 40320 × 24
- **Kociemba 算法**: 分两阶段降维求解
  - Phase-1: 恢复方向 + 边块分层（搜索空间 ~10¹⁰）
  - Phase-2: 排列还原（搜索空间 ~10⁸）

### ABO 血型遗传

- **Hardy-Weinberg 平衡**:
  - P(AA) = p², P(AO) = 2pr
  - P(BB) = q², P(BO) = 2qr
  - P(AB) = 2pq
  - P(OO) = r²

### 金字塔投影

- **网格大小**: `2n + 1` (n 层)
- **元素总数**: `(2n + 1)² = 4n(n+1) + 1`
- **环形编码**: 第 i 层 `8i` 个元素

## 项目特点

1. **跨学科融合**: 涵盖群论、遗传学、金字塔神经网络、游戏触发器
2. **双重建模**: 贴纸级与块级魔方建模，从直观表示到抽象群论
3. **严格的数学基础**: 基于群论的魔方建模、基于孟德尔定律的遗传计算
4. **高效算法**: Kociemba 两阶段算法、剪枝表优化、群同态投影
5. **可视化支持**: Pygame 3D 渲染、实时交互
6. **扩展性设计**: 支持自定义阶魔方、血型系统、金字塔层数配置

## 许可证

本项目为学术研究项目，仅供学习和研究使用。

## 贡献

欢迎提交 Issue 和 Pull Request。
