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
```

也可调用各模块：

    agents/ai_search.py：企业/专利搜索

    agents/ai_tasks.py：任务图管理

    script/aigc.py：通用调度测试脚本

## 🧠 核心模块说明

| 模块名                    | 说明                             |
| ---------------------- | ------------------------------ |
| `agents/ai_agents.py`  | 多 Agent 协作执行逻辑                 |
| `agents/ai_tasks.py`   | 图结构任务调度核心，节点与边支持动态添加与触发条件      |
| `agents/ai_search.py`  | 企业搜索、模糊匹配、上下文引导                |
| `agents/ai_vectors.py` | 向量化处理与召回重排（Qdrant 支持）          |
| `config*.py`           | 配置管理，支持多环境切换                   |
| `database.py`          | MySQL 异步操作支持                   |
| `script/*.py`          | 各类 AIGC 调用示例、后端服务脚本            |
| `data/*.yaml / *.pkl`  | 配置缓存或中间结果文件                    |
| `docker/`              | 容器构建配置                         |
| `structs.py`           | TaskNode、TaskEdge、任务状态等核心结构体定义 |
| `utils.py`             | 工具函数集合，如 tokenizer、时间、转换等      |

## 📂 文件结构概览
```text
aigc/
├── agents/                # 核心智能 agent 模块
├── script/                # 实验性与部署脚本
│   └── rime/              # 专利语义处理模块
├── data/                  # 中间数据与配置缓存
├── docker/                # Docker 构建环境
├── config.py              # 主配置
├── main.py                # 启动入口
├── generates.py           # 通用生成与总结函数
├── knowledge.py           # 知识提取/加载模块
├── database.py            # 数据库操作支持
├── structs.py             # 数据结构定义（任务图）
├── utils.py               # 工具方法
└── requirements.txt       # 依赖包
```


🤝 致谢

本项目基于 Neo4j、Qdrant、FastAPI、LangGraph 等现代 AI 框架构建，感谢所有开源组件贡献者。
