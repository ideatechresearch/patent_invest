# TechNet Master Project

## 项目简介
本项目是一个综合性的项目，旨在通过网站和文档目录的协同工作，提供数据整理、转换和分析等功能。  
本项目拟通过研究机器学习、AI和大模型技术在金融领域的最新发展趋势和应用，深入研究并开发人工智能和大模型技术在金融场景中的应用平台。研究内容涵盖新兴技术的趋势分析、金融数据的收集、处理与商业价值挖掘、AI模型的定制开发与优化，以及构建集成这些技术的金融实操应用平台。  

其中technet-master是基于TechNet 文本分析方法的技术相似专利挖掘方法及应用

## 目录结构
项目的目录结构如下：

| 文件/目录              | 描述         | 用途                               |
|-----------------------|--------------|------------------------------------|
| /technet-master       | 网站目录     | 存放项目的源代码                   |
| /technet-master/static/| 网站目录     | 存放静态预预置文件的源代码          |
| /technet-master/static/js| 网站目录     | 存放JavaScript的源代码           |
| /technet-master/templates| 网站目录     | 存放html页面的源代码             |
| /technet              | 其他目录     | 包含历史版本及一些方法             |
| /data                 | 文档目录     | 包含项目的文档和使用手册           |
| ipynb                 | 数据整理文件 | 包含数据整理、转换和分析的 Jupyter Notebooks |

## 使用指南
以下是如何使用该项目的基本信息和指导：

1. 克隆项目:
   ```bash
   git clone https://github.com/ideatechresearch/patent_invest/technet-master.git 
2. 进入项目目录:
    ```bash
    cd technet-master
3. 安装依赖:
    ```bash
    pip install -r requirements.txt
4. 运行项目:
    根据具体项目的配置和运行方式，提供详细的步骤和命令。
    1. 启动入口: 

        
        ```flask_app.py``` 是项目的启动入口。
        启动命令：
        ```bash
        python /technet-master/flask_app.py 
        ```
    2. 选用停用词部分:

        /technet-master/select_stop_words.py 是处理选用停用词的部分。
    3. 向量数据库搜索部分:

        /technet-master/qdrant_net.py 处理向量数据库的搜索功能。
    4. 数据库定义部分:

        /technet-master/database.py 包含数据库的定义。

    5. 配置文件:

        config_debug.py 是配置文件的样例格式，可以用此文件生成 ```/technet-master/config.py``` 来运行项目。
        复制命令：
        ```bash
        cp /technet-master/config_debug.py /technet-master/config.py
        ```



## 贡献
欢迎对本项目进行贡献，请遵循以下步骤：

1. Fork 本仓库
2. 创建你的分支 (git checkout -b feature/your-feature)
3. 提交你的修改 (git commit -m 'Add some feature')
4. 推送到分支 (git push origin feature/your-feature)
5. 提交 Pull Request

## 许可证
本项目采用 [MIT 许可证](LICENSE).


## 联系我
如有任何问题或建议，请通过以下方式与我们联系：

电子邮件: 303056474@139.com

## GitHub Issues:
我们使用 GitHub Issues 来跟踪任务、错误和功能请求。您可以通过以下步骤来报告问题或提出功能请求：

1. **报告错误**：
   - 如果发现项目中的错误或问题，请在 [GitHub Issues](https://github.com/yourusername/technet-master/issues) 页面中创建一个新的 Issue。
   - 请详细描述错误，包括重现步骤、预期结果和实际结果。

2. **提出功能请求**：
   - 如果您有新功能的建议，请在 [GitHub Issues](https://github.com/yourusername/technet-master/issues) 页面中创建一个新的 Issue，并选择 "Feature request" 标签。
   - 请详细描述功能，包括用途、预期效果和实现思路。

3. **参与讨论**：
   - 您可以在现有的 Issues 中参与讨论，提供反馈和建议。
   - 对于重要的决策和讨论，我们会在 Issues 中记录并与大家分享。

我们鼓励大家积极参与，通过报告问题和提出建议来帮助我们改进项目。感谢您的支持！