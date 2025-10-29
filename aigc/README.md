# AIGC

This is a Python project for AIGC.

AIGC 模块是一个以大语言模型为核心的智能专利与企业分析工具，聚焦于信息抽取、任务图调度、企业搜索、多维交叉比对等功能，辅助投资分析与尽调。

## ✨ 特性 Highlights

- 🔍 **智能搜索与企业分析**：支持自然语言提问、模糊匹配、上下文记忆。
- 🧠 **Agent 多角色协作**：基于任务图组织 Agent，自动完成复杂任务流。
- 🔗 **结构化任务管理**：支持任务节点、依赖边、状态更新，配合 Neo4j 图数据库。
- 📄 **PDF、合同、专利内容处理**：具备本地文档摘要、对比与风险提示能力。
- 🧰 **工具插件体系**：AI 工具统一调度，支持 RAG 检索、多模型评估。

---

## 📦 安装 Installation

```bash
git clone https://github.com/ideatechresearch/patent_invest.git
cd patent_invest/aigc
pip install -r requirements.txt
```

## 如需构建容器环境：
```bash
cd docker
docker build -t aigc-agent .
```

## 🚀 使用方式 Usage
```bash
python main.py  # 启动主入口，运行测试或命令行任务
win deoloy.bat
```


## 🧠 核心模块说明

| 模块名                    | 说明                            |
| ---------------------- | ----------------------------- |
| `agents/ai_tasks.py`   | 图结构任务调度核心，TaskNode、TaskEdge、任务图管理，节点与边支持动态添加与触发条件     |
| `agents/ai_vectors.py` | 向量化处理与召回重排（Qdrant 支持）         |
| `agents/ai_search.py`  | 网络搜索、上下文引导                |
| `agents/ai_agents.py`  | 多 Agent 协作执行逻辑                |
| `agents/ai_company.py` | 企业搜索                 |
| `config*.py`           | 配置管理，支持多环境切换                  |
| `database.py`          | 数据库 MySQL 操作支持                   |
| `generates.py`         | 通用生成与AI工具处理，多模型融合              |
| `script/*.py`          | 各类 AIGC 调用示例、后端服务脚本、独立测试脚本      |
| `script/aigc.py`       | 通用调度测试脚本                   |
| `script/knowledge.py ` | 知识提取/加载模块                  |
| `data/*.yaml / *.pkl`  | 配置缓存或中间结果文件                   |
| `docker/`              | 容器构建配置 Dockerfile                  |
| `structs.py`           | FastAPI 数据结构体定义 |
| `utils.py`             | 工具函数集合，如 tokenizer、模糊匹配、模糊匹配、时间戳处理、序列化、转换等     |

## 📂 文件结构概览
```text
aigc/
├── agents/                # 核心智能 agent 模块，封装任务执行、工具调用与决策策略
├── script/                # 脚本目录，用于实验、批处理、自动化工具或临时任务
├── rime/                  # 功能性业务模块，实现领域核心逻辑
├── data/                  # 中间数据与配置缓存
├── docker/                # Docker 构建环境
├── templates/             # 网页模板文件（如 Jinja2/HTML），用于渲染前端页面
├── router/                # FastAPI 路由定义模块，负责接口的 URL 组织与分发
├── static/                # 静态资源目录（CSS、JS、图片等）
├── utils/                 # 工具函数与通用方法集合（如日志、配置、数据转换等）
│
├── config.py              # 主配置（环境变量、系统参数、路径、模型、数据库连接等）
├── main.py                # 启动入口，FastAPI 主实例与路由注册
├── generates.py           # 通用生成逻辑与工具调用实现（如内容生成、分析等）
|── service.py             # 服务层逻辑，封装业务调用、任务调度或服务对接
├── database.py            # 数据库操作与连接管理（连接池、ORM封装、CRUD接口）
├── structs.py             # 数据结构与 Pydantic 模型定义（输入输出结构约束）
├── secure.py              # 安全与权限相关逻辑（加密、鉴权、签名等）
├── requirements.txt       # Python依赖包清单
│
├── config.yaml            # YAML格式的项目配置（模型、路径、环境参数等）
├── nginx.conf             # Nginx 反向代理或静态服务配置文件
├── openapi.json           # 自动生成的 API 文档定义文件（FastAPI/OpenAPI 规范）
├── api-types.ts           # TypeScript 类型定义文件（用于前端API类型校验）
│
├── *.sh                   # Shell脚本（Linux/Mac 环境部署、启动、同步、清理等）
├── *.bat                  # Windows批处理脚本（快速启动或工具命令）
├── *.log                  # 日志输出文件，用于调试与运行记录
└── README.md              # 项目说明文档（结构说明、部署指南、使用示例等）
```


🤝 致谢

本项目基于 Neo4j、Qdrant、Redis、MySQL、ClickHouse、Dask、FastAPI、FastMCP、LangGraph 等现代 AI 框架构建，感谢所有开源组件贡献者。
