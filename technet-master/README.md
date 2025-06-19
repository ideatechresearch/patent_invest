## 目录结构
项目的目录结构如下：

| 文件/目录              | 描述         | 用途                               |
|-----------------------|--------------|------------------------------------|
| /technet-master       | 网站目录     | 存放项目的源代码                   |
| /technet-master/static/| 网站目录     | 存放静态预置文件的源代码           |
| /technet-master/static/js| 网站目录     | 存放JavaScript的源代码           |
| /technet-master/templates| 网站目录     | 存放Html页面的源代码             |

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
    6. 数据库：  
        需要安装sql数据库和qdrant数据库。  
        如果使用文件方式运行，可以查看历史版本中/technet中w2v_neo4j_net.py,vec_neo4j_net.py,matrix_neo4j_net.py这些方法构建。

5. technet说明:  
    #### 选取停用词部分(select_stop_words.py)，主要是用位运算对不同用户给选取的词做标记，其他还包括清零和置位。
        如:StopWordsFlag::set_stop_flag,StopWordsFlag::set_words_flag
    #### 向量搜索功能部分(qdrant_net.py)，使用qdrant向量数据库。
        1. 提供通用字段等筛选方法，如：field_match,empty_match,ids_match
        2. 提供嵌入模型查询方法，如：get_bge_embeddings,most_similar_embeddings
        3. 节点匹配重复关系优化方法：rerank_similar_by_search,rerank_similar_by_recommend
        4. 提供ID和Name映射记忆模块，如VDBSimilar::get_id,get_ids,get_name,get_names主要使用names_to_ids方法
            获取向量数据方法：get_vec,get_vecs
            可选字段载荷数据方法：get_payload
            搜索方法,Recommend使用向量数据服务做ID推荐：Search,Recommend,SearchBatch,RecommendBatch
        5. 提供关系查询模块：VDBRelationships,对不同的查询筛选类别创建VDBSimilar
            基础获取数据方法：get_id,get_ids,get_names,get_payload
            查询和关系方法：similar,similar_by_names,create_relationship,create_relationships,Relation,Relations主要是返回数据构造不同
            多项网络查询方法：create_relationships_breadth,create_relationships_depth,RelationsBreadth,RelationsDepth主要是广度和深度区别
            通用网络关系方法：SimilarRelationships,SimulationNodes,SimilarRelations,可以提供不同参数自动使用相应方法,并给路由返回的数据打包节点和关系
    #### Web应用部分(flask_app.py)，使用Flask框架。
        1.定义/search提供查询功能
        2.定义/get_words提供词浏览页面及标记停用词功能
        3.定义/relationship提供关系网络查询页面功能
        4.定义/show_relationships,/node_relations提供关系网络显示功能
        5.定义/similar响应点击返回相似词
        6.定义/details提供数据库查询及表格页面显示功能
        7.定义companys_detail_links,patents_detail_links,words_detail_links,words_detail_absrtact提供表格数据转换附加链接等功能
        8.定义relationship_params主要用做页面参数转换附加节点限制等
    #### Web前端部分(draw.graph.js)，主要是画力导向图。
        1.在relationship.html中，当按钮点击时，传页面参数，然后给fetchDataAndDraw
        2.fetchDataAndDraw请求数据到/show_relationships
        3.show_relationships通过relationship_params转换参数并添加其他信息
        4.接下来传给VDBRelationships::SimilarRelations
        5.SimilarRelations判断各层次layers和batch,传给RelationsBreadth或RelationsDepth
        6.判断一些break条件如最大深度、计算次数、节点数量，根据广度和深度区别使用Relations循环或Relation递归
        7.接下来每个节点数据调用VDBSimilar::RecommendBatch或VDBSimilar::Recommend
        8.然后请求qdrant_client,得到节点向量，并添加一些动态的过滤条件查找相似值,其中VDBSimilar::RecommendBatch需要给rerank_similar_by_recommend把重复关系优化一下
        9.得到(id,score)更新ids_depth和累加relationships_edges
        10.然后根据draw类型,增加payloads.name并改变格式组装打包数据
        11.使用json格式返回请求给 JavaScript fetch 
        12.获取数据后根据各个节点关系(source,target)匹配，给drawGraph绘制图形
        13.新建力导向图，根据(nodes, edges)数据建立节点和边，并根据属性添加文字、颜色，给节点添加ticked、ClickNode函数
        14.当点击节点时触发ClickNode事件,然后给fetchNodeRelations
        15.fetchNodeRelations请求数据到/node_relations，并传递页面参数，节点id,name和当前已有的Nodes信息
        16.node_relations转换参数后给VDBRelationships::SimilarRelation
        17.Relation根据已有节点和其所在层数信息给VDBSimilar::Recommend
        18.Recommend动态的过滤一些节点id后请求qdrant_client.recommend得到点击节点id的相似值
        19.得到(id,score)后转换一下，使用(new_depth, new_relationships)打包(nodes, edges),其中这边source,target传的是节点向量数据库id
        20.使用json格式返回请求或对新节点填下缺失后push给前端js的(nodes, edges),然后使用updateSimulation更新图形创建新节点和边
