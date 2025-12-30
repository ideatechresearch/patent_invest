# AIGC

<div align="center">

![Version](https://img.shields.io/badge/version-v1.3.0-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)
![License](https://img.shields.io/badge/license-MIT-orange)

**以大语言模型为核心的智能分析工具 + 魔方可视化与求解系统**

[特性](#-特性) • [快速开始](#-快速开始) • [文档](#-核心模块) • [API](#-web-api) • [贡献](#-贡献)

</div>

---

## 📖 项目简介

AIGC 是一个功能强大的 Python 项目，包含两大核心模块：

1. **智能专利与企业分析工具**：基于大语言模型，聚焦于信息抽取、任务图调度、企业搜索、多维交叉比对等功能，辅助投资分析与尽调
2. **魔方可视化与求解系统**：支持 N×N 阶魔方的 3D 可视化、交互操作、自动求解（BFS/IDA*算法）

### ✨ 特性 Highlights

#### 🤖 AI 智能分析
- 🔍 **智能搜索与企业分析**：支持自然语言提问、模糊匹配、上下文记忆
- 🧠 **Agent 多角色协作**：基于任务图组织 Agent，自动完成复杂任务流
- 🔗 **结构化任务管理**：支持任务节点、依赖边、状态更新，配合 Neo4j 图数据库
- 📄 **PDF、合同、专利内容处理**：具备本地文档摘要、对比与风险提示能力
- 🧰 **工具插件体系**：AI 工具统一调度，支持 RAG 检索、多模型评估
- 🌐 **多模型支持**：集成 30+ AI 服务商（Qwen、DeepSeek、Moonshot、GLM、GPT、Claude 等）

#### 🎲 魔方系统
- 🎨 **3D 可视化渲染**：基于 Pygame 的实时 3D 魔方渲染引擎
- 🖱️ **交互式操作**：支持鼠标拖拽旋转视角和层旋转
- 🧩 **自动求解算法**：BFS 广度优先搜索、IDA* 迭代加深 A* 算法
- 📐 **支持任意阶数**：3×3、4×4、5×5、6×6 等 N×N 阶魔方
- 🎯 **打乱与还原**：随机打乱、自动求解、操作回放

---

## 🚀 快速开始

### 📦 安装

```bash
# 克隆项目
git clone https://github.com/ideatechresearch/patent_invest.git
cd patent_invest/aigc

# 安装依赖
pip install -r requirements.txt
```

### 🐳 Docker 构建

```bash
cd docker
docker build -t aigc-agent .
```

### 🎮 运行

```bash
# 启动 FastAPI 服务
python main.py

# 或使用 Windows 批处理
win deploy.bat

# 运行魔方可视化
python -m rime.cubedraw
```

---

## 🧠 核心模块

### AI 智能体模块

| 模块 | 说明 |
|------|------|
| `agents/ai_tasks.py` | 图结构任务调度核心，TaskNode、TaskEdge、任务图管理 |
| `agents/ai_vectors.py` | 向量化处理与召回重排（Qdrant 支持） |
| `agents/ai_search.py` | 网络搜索、上下文引导 |
| `agents/ai_agents.py` | 多 Agent 协作执行逻辑 |
| `agents/ai_company.py` | 企业搜索与分析 |
| `agents/ai_tools.py` | AI 工具统一调度 |
| `agents/ai_prompt.py` | 提示词模板管理 |
| `agents/ai_multi.py` | 多模型融合推理 |
| `generates.py` | 通用生成与 AI 工具处理 |
| `config.py` | 配置管理（支持多环境切换、YAML 加载） |

### 魔方模块

| 模块 | 说明 |
|------|------|
| `rime/cube.py` | 魔方核心逻辑（状态表示、旋转操作、求解算法） |
| `rime/cubedraw.py` | 3D 渲染引擎（Pygame 可视化） |
| `rime/circular.py` | 装饰器工具（链式方法、类缓存） |

### 核心功能

#### 🤖 AI 功能
- **任务图调度**：基于 Neo4j 的任务依赖管理
- **向量检索**：Qdrant 向量数据库集成
- **多模型推理**：统一接口调用 30+ LLM 服务
- **企业分析**：企业信息抽取、关联分析
- **文档处理**：PDF 解析、摘要生成、风险提示

#### 🎲 魔方功能
- **状态表示**：6×N×N 数组表示魔方状态
- **旋转操作**：支持任意层的顺/逆时针旋转
- **求解算法**：
  - BFS 广度优先搜索（深度 ≤ 6）
  - IDA* 迭代加深 A* 启发式搜索（深度 ≤ 25）
  - 贪心算法（中心块修正）
- **可视化渲染**：
  - 3D 旋转投影
  - 实时动画
  - 鼠标交互

---

## 📂 项目结构

```text
aigc/
├── agents/                # AI 智能体模块
│   ├── ai_tasks.py        # 任务图调度
│   ├── ai_vectors.py      # 向量检索
│   ├── ai_search.py       # 网络搜索
│   ├── ai_agents.py       # Agent 协作
│   ├── ai_company.py      # 企业分析
│   └── ...
├── rime/                  # 魔方模块
│   ├── cube.py            # 魔方核心逻辑
│   ├── cubedraw.py        # 3D 渲染引擎
│   └── circular.py        # 工具装饰器
├── script/                # 测试与示例脚本
├── data/                  # 数据与配置缓存
├── docker/                # Docker 构建文件
├── router/                # FastAPI 路由
├── utils/                 # 工具函数
├── config.py              # 主配置文件
├── main.py                # 启动入口
├── generates.py           # AI 生成逻辑
├── database.py            # 数据库操作
├── structs.py             # 数据结构定义
└── requirements.txt       # 依赖清单
```

---

## 🎮 使用示例

### 🤖 AI 智能分析

```python
from generates import llm_generate
from agents.ai_tasks import TaskGraph

# 创建任务图
graph = TaskGraph()
graph.add_task("search_company", ...)
graph.add_task("analyze_patents", ...)
result = graph.execute()

# 调用 LLM
response = llm_generate(
    prompt="分析这家公司的技术实力",
    model="qwen-max"
)
```

### 🎲 魔方操作

```python
from rime.cube import RubiksCube
from rime.cubedraw import RubiksCubeDraw

# 创建 3×3 魔方
cube = RubiksCube(n=3)

# 打乱
scramble_moves = cube.scramble(20)
cube.apply(scramble_moves)

# 自动求解
solution = cube.solve()
print(f"求解步骤: {solution}")

# 启动可视化
app = RubiksCubeDraw(cube)
app.run()
```

### 🎨 魔方可视化交互

**键盘控制**：
- `A` - 切换自动旋转
- `Space` - 暂停/恢复动画
- `P` - 单步执行
- `S` - 随机打乱（25 步）
- `R` - 重置魔方
- `C` - 清空队列

**鼠标控制**：
- 左键拖拽 - 旋转视角
- 右键拖拽 - 旋转层

---

## 🔧 配置说明

配置文件支持多种格式（Python 类、YAML）：

```python
# config.py
class Config:
    # AI 模型配置
    DEFAULT_MODEL = 'qwen'
    DEFAULT_MODEL_EMBEDDING = 'BAAI/bge-large-zh-v1.5'

    # 数据库配置
    SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://...'

    # 向量数据库
    QDRANT_HOST = 'qdrant'
    QDRANT_PORT = 6333

    # API Keys
    QIANFAN_Service_Key = '***'
    Moonshot_Service_Key = '***'
    # ... 更多配置
```

**敏感信息保护**：
- 使用 `***` 占位符
- 运行时从 YAML 加载真实配置
- 支持配置版本管理与备份

```bash
# 保存配置到 YAML
python -c "from config import Config; Config.save()"

# 脱敏处理
python -c "from config import Config; Config.mask_sensitive()"
```

---

## 🌐 Web API

FastAPI 自动生成 API 文档（OpenAPI 规范）：

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

### 主要端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/chat` | POST | AI 对话接口 |
| `/api/search` | POST | 企业搜索 |
| `/api/tasks` | POST | 创建任务 |
| `/health` | GET | 健康检查 |

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 📝 更新日志

### v1.3.0 (当前)
- 新增魔方可视化渲染引擎
- 支持 N×N 阶魔方
- 新增 IDA* 求解算法
- 优化配置管理系统
- 新增 30+ AI 模型支持

### v1.2.9
- 增强企业分析功能
- 优化向量检索性能
- 修复已知问题

---

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

---

## 🙏 致谢

本项目基于以下开源框架构建：

- **AI 框架**：LangGraph、FastMCP、OpenAI SDK
- **数据库**：Neo4j、Qdrant、Redis、MySQL、ClickHouse
- **Web 框架**：FastAPI、Pydantic
- **数据处理**：Dask、NumPy
- **可视化**：Pygame
- **容器化**：Docker

感谢所有开源贡献者！

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给一个 Star！**

Made with ❤️ by [ideatechresearch](https://github.com/ideatechresearch)

</div>
