import qdrant_client.models as qcm
import json, re, uuid, time
import asyncio
import aiomysql
import httpx
from contextlib import asynccontextmanager
from qdrant_client import AsyncQdrantClient
from openai import AsyncOpenAI
from typing import Optional, Literal, Dict, List, Tuple, Union, Any, Callable
from bm25 import BM25, load_jieba


async def fetch_url(url, httpx_client, max_retries=3, delay=3):
    for attempt in range(max_retries):
        try:
            response = await httpx_client.get(url, timeout=30)
            response.raise_for_status()
            result = response.json()
            # if result.get('status')=='000103':
            #     return
            if result.get('status') == "fail" and result.get("code") == "000205":  # "数据正在计算，请稍后再试
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)
                    continue
                else:
                    print(f"[{url}] 接口多次返回 fail:{result.get('reason', response.text)}，放弃重试。")

            return result

        except httpx.HTTPError as e:
            if attempt < max_retries - 1:
                print(f"[{url}] 请求异常 {e}，第 {attempt + 1} 次尝试后等待 {delay} 秒重试...")
                await asyncio.sleep(delay)
            else:
                raise e

    return None


async def ai_analyze(system_prompt: str, results, client, desc: str = None, model='deepseek-reasoner', max_tokens=4096,
                     temperature: float = 0.2, **kwargs):
    user_request = json.dumps(results, ensure_ascii=False)
    if desc:
        user_request = f"{desc}: {user_request}"

    messages = [{"role": "system", "content": system_prompt}, {'role': 'user', 'content': user_request}]
    payload = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
    )
    payload.update(kwargs)
    completion = await client.chat.completions.create(**payload)
    content = completion.choices[0].message.content
    print(f'</{desc}>: {content}')
    return content.split("</think>")[-1]


async def ai_embeddings(inputs: str | list[str], client, model: str = 'BAAI/bge-large-zh-v1.5', batch_size=25,
                        get_embedding: bool = True, **kwargs):
    if not inputs:
        return []

    if isinstance(inputs, (list, tuple)) and len(inputs) > batch_size:
        tasks = [client.embeddings.create(
            model=model, input=inputs[i:i + batch_size],
            encoding_format="float",
            **kwargs
        ) for i in range(0, len(inputs), batch_size)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        if not get_embedding:
            results_data = results[0]
            all_data = []
            for i, result in enumerate(results):
                for idx, item in enumerate(result.data):
                    item.index = len(all_data)
                    all_data.append(item)

                if i > 0:
                    results_data.usage.prompt_tokens += result.usage.prompt_tokens
                    results_data.usage.total_tokens += result.usage.total_tokens

            results_data.data = all_data
            return results_data.model_dump()

        embeddings = [None] * len(inputs)
        input_idx = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Error encountered: {result}")
                continue
            for idx, item in enumerate(result.data):
                embeddings[input_idx] = item.embedding
                input_idx += 1
    else:
        # await asyncio.to_thread(client.embeddings.create
        results = await client.embeddings.create(
            model=model, input=inputs, encoding_format="float", **kwargs)

        if not get_embedding:
            return results.model_dump()

        embeddings = [item.embedding for item in results.data]

    return embeddings


async def create_collection(collection_name, client, size, vector_name: str = None,
                            vectors_on_disk=True, payload_on_disk=True, hnsw_on_disk=True, recreate=False):
    """
    创建 Qdrant 集合。如果 `new` 为 True 且集合已经存在，则删除现有集合并重新创建。
    """
    # https://api.qdrant.tech/api-reference/collections/create-collection
    try:
        if recreate:
            if await client.collection_exists(collection_name=collection_name):
                print(f"Collection {collection_name} exists. Deleting and recreating...")
                await client.delete_collection(collection_name=collection_name)
        # 如果集合不存在，创建新的集合
        if not await client.collection_exists(collection_name=collection_name):
            print(f"Creating new collection: {collection_name}")
            # 向量配置,  "Cosine","Euclid","Dot","Manhattan" ,datatype='float32'
            vector_params = qcm.VectorParams(size=size, distance=qcm.Distance.COSINE, on_disk=vectors_on_disk)
            await client.create_collection(
                collection_name=collection_name,
                # map from strings to objects
                vectors_config={vector_name: vector_params} if vector_name else vector_params,  # 多向量模式 "text","image"
                # HNSW 配置：是否将 HNSW 索引保存在磁盘上
                hnsw_config=qcm.HnswConfigDiff(on_disk=hnsw_on_disk),
                on_disk_payload=payload_on_disk
                # 如果需要禁用索引，可以取消注释以下配置：
                # optimizer_config=OptimizersConfigDiff(indexing_threshold=0),
            )
        # 返回集合中的点数
        res = await client.get_collection(collection_name=collection_name)
        return res.model_dump()

    except Exception as e:
        print(f"Error while creating or managing the collection {collection_name}: {e}")
        return None


async def upsert_points(payloads: list[dict], vectors: list, collection_name: str, client, vector_name: str = None):
    """
    插入单个点到 Qdrant 集合。
    """
    assert len(payloads) == len(vectors), "Payloads and vectors must have the same length."
    current_count = await client.count(collection_name=collection_name, exact=True)
    # dense_doc=qcm.Document(text="bye world", model="sentence-transformers/all-MiniLM-L6-v2")#DENSE
    points: list[qcm.PointStruct] = [qcm.PointStruct(id=current_count.count + idx, payload=payload,
                                                     vector={vector_name: vector} if vector_name else vector)
                                     for idx, (payload, vector) in enumerate(zip(payloads, vectors))]

    try:
        operation_info = await client.upsert(collection_name=collection_name, points=points)
        assert operation_info.status == qcm.UpdateStatus.COMPLETED
        return operation_info.operation_id
    except Exception as e:
        # client.upload_points(collection_name=collection_name, points=points, parallel=parallel, wait=True)
        print(f"Error upserting points to collection {collection_name}: {e}")
        return None


def field_match(field_key, match_values):
    """匹配字段值，支持单值和多值匹配"""
    if not field_key or not match_values:
        return []
    if isinstance(match_values, (str, int, float)):
        return [qcm.FieldCondition(key=field_key, match=qcm.MatchValue(value=match_values, ), )]
    if isinstance(match_values, (list, tuple, set)):
        return [qcm.FieldCondition(key=field_key, match=qcm.MatchAny(any=list(match_values)))]
    return [qcm.FieldCondition(key=field_key, match=qcm.MatchValue(value=v), ) for v in match_values]


async def search_by_vecs(query_vectors: list[list[float]], collection_name: str, client,
                         payload_key: str = 'title', vector_name: str = None,
                         match: list = [], not_match: list = [],
                         topn: int = 10, score_threshold: float = 0.0, exact: bool = False):
    """
    使用 Qdrant 批量查询查询词的嵌入，返回与查询最相似的标记和得分。

    :param query_vectors: 预计算的向量列表，每个向量表示一个查询。
    :param collection_name: Qdrant 集合的名称。
    :param client: Qdrant 客户端实例。
    :param payload_key: 返回的 payload 中的键名，默认是 'word'。
    :param match: 搜索时需要匹配的过滤条件。
    :param not_match: 搜索时不匹配的过滤条件。
    :param topn: 返回的最相似结果数量。
    :param score_threshold: 返回的最小得分阈值。
    :param exact: 是否进行精确搜索（True）或近似搜索（False）。

    :return: 每个查询返回一个 `ScoredPoint` 结果列表。
    """
    query_filter = qcm.Filter(must=match, must_not=not_match)  # 缩小查询范围
    search_queries = [
        qcm.SearchRequest(vector={"vector": vec, "name": vector_name} if vector_name else vec, filter=query_filter,
                          limit=topn, score_threshold=score_threshold,
                          with_payload=[payload_key] if payload_key else True,
                          params=qcm.SearchParams(exact=exact), )
        for vec in query_vectors]

    return await client.search_batch(collection_name=collection_name, requests=search_queries)


async def search_by_embeddings(querys: list[str] | tuple[str], collection_name: str, client, emb_client, model: str,
                               payload_key: str | list = 'title', vector_name: str = None,
                               match: list = [], not_match: list = [], topn: int = 10, score_threshold: float = 0.0,
                               exact: bool = False, get_dict=True, **kwargs):
    """
    使用 Qdrant 批量查询查询词的嵌入，返回与查询最相似的标记和得分。

    :return: 返回一个包含查询和匹配结果的列表，每个查询对应一个列表，包含匹配标记的 payload、得分和 ID。
    -> List[dict]
    -> List[Tuple[str, dict]]
    -> List[Tuple[Any, float, int]]:
    -> List[Tuple[str, List[Tuple[Any, float, int]]]]
    """
    query_vectors: list[list[float]] = await ai_embeddings(querys, emb_client, model, **kwargs)
    if not query_vectors:
        return []

    assert len(querys) == len(query_vectors), "Query list and embedding vector list size mismatch."

    def extract_payload(p, keys):
        if get_dict:
            return p.model_dump(exclude={'vector', 'version', 'shard_key', 'order_value'}, exclude_none=True)
        if keys is True or keys is None:
            return p.payload, p.score, p.id
        if isinstance(keys, str):
            return p.payload.get(keys, None), p.score, p.id
        if isinstance(keys, list):
            return {k: p.payload.get(k) for k in keys}, p.score, p.id
        return p.payload, p.score, p.id

    if len(querys) == 1:
        with_payload = True
        if payload_key:
            if isinstance(payload_key, str):
                with_payload = [payload_key]
            if isinstance(payload_key, list):
                with_payload = payload_key

        query_filter = qcm.Filter(must=match, must_not=not_match)
        query_vector = {"vector": query_vectors[0], "name": vector_name} if vector_name else query_vectors[0]
        search_hit = await client.search(collection_name=collection_name,
                                         query_vector=query_vector,  # tolist() Named Vector
                                         query_filter=query_filter,
                                         limit=topn,
                                         score_threshold=score_threshold,
                                         with_payload=with_payload,
                                         # params=qcm.SearchParams(exact=exact),
                                         # with_prefix_cache=True  # 第二次查询会直接命中缓存,在并发查询较多时表现更好
                                         )

        for h in search_hit:
            if not hasattr(h, 'payload') or h.payload is None:  # fallback：用 retrieve 填充
                id_record = await client.retrieve(collection_name, [h.id], with_payload=with_payload)
                h.payload = id_record[0].payload
        # {k: p.payload.get(k) for k in payload_key}
        return [extract_payload(p, payload_key) for p in search_hit]

    search_hit = await search_by_vecs(query_vectors, collection_name, client, payload_key, vector_name,
                                      match, not_match, topn, score_threshold, exact)  # ScoredPoint

    return [(item, [extract_payload(p, payload_key) for p in hit])
            for item, hit in zip(querys, search_hit)]


def extract_json_array(input_str) -> list | None:
    """
    提取并解析 markdown 包裹的 JSON 数组（尤其是 ```json ... ``` 格式）
    """
    # 提取 ```json ... ``` 块中内容
    match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', input_str, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
    return None


class OperationMysql:
    def __init__(self, host, user, password, db_name, port=3306, charset="utf8mb4"):
        self.host = host
        self.user = user
        self.password = password
        self.db_name = db_name
        self.charset = charset
        self.port = port
        self.pool = None

    async def __aenter__(self):
        if self.pool is None:
            await self.init_pool()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close_pool()

    async def init_pool(self, minsize=1, maxsize=30, autocommit=True):
        try:
            self.pool = await aiomysql.create_pool(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                db=self.db_name,
                charset=self.charset,
                autocommit=autocommit,
                minsize=minsize,
                maxsize=maxsize
            )
        except Exception as e:
            print(f"[Async] 创建连接池失败: {e}")
            self.pool = None

    async def close_pool(self):
        if self.pool:
            self.pool.close()
            await self.pool.wait_closed()
            self.pool = None

    async def run(self, sql: str, params: tuple | dict | list = None):
        if self.pool is None:
            await self.init_pool()

        sql_type = (sql or "").strip().split()[0].lower()
        result = None
        try:
            async with self.pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cur:
                    if isinstance(params, list):
                        await cur.executemany(sql, params)
                    else:
                        await cur.execute(sql, params or ())

                    if sql_type == "select":
                        result = await cur.fetchall()
                    elif sql_type in {"insert", "update", "delete", "replace"}:
                        await conn.commit()  # 显式保险,autocommit=True
                        if sql_type == "insert":
                            result = cur.lastrowid or int(conn.insert_id())
                        else:
                            result = True

        except Exception as e:
            print(f"[Async] SQL执行错误: {e}, SQL={sql}")

        return result

    async def execute(self, sql_list: list[tuple[str, tuple | dict | list]]):
        if self.pool is None:
            await self.init_pool()

        conn = await self.pool.acquire()
        try:
            async with conn.cursor() as cur:
                for sql, params in sql_list:
                    if not sql.strip():
                        continue
                    if isinstance(params, list):
                        await cur.executemany(sql, params)
                    else:
                        await cur.execute(sql, params or ())
                await conn.commit()
                return True
        except Exception as e:
            print(f"[Async] 批量 SQL 执行失败: {e}")
            try:
                await conn.rollback()
            except Exception as rollback_err:
                print(f"[Async] 回滚失败: {rollback_err}")
            return False
        finally:
            self.pool.release(conn)

    async def query(self, query_list: list[tuple[str, tuple | dict]], fetch_all: bool = True) -> list:
        if self.pool is None:
            await self.init_pool()

        results = []
        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                for sql, params in query_list:
                    await cur.execute(sql, params)
                    result = await cur.fetchall() if fetch_all else await cur.fetchone()
                    results.append(result)
        return results

    async def insert(self, table_name: str, params_data: dict):
        fields = tuple(params_data.keys())
        values = tuple(params_data.values())
        field_str = ', '.join(f"`{field}`" for field in fields)
        sql = f"INSERT INTO `{table_name}` ({field_str}) VALUES ({', '.join(['%s'] * len(fields))})"
        return await self.run(sql, values)

    async def get_conn(self):
        if self.pool is None:
            await self.init_pool()
        return await self.pool.acquire()


async def consumer_worker(queue: asyncio.Queue[tuple[Any, int]], process_task: Callable, max_retries: int = 0,
                          delay: float = 1.0, **kwargs):
    while True:
        task = await queue.get()
        if task is None:
            queue.task_done()
            break

        success = await process_task(task, **kwargs)
        if not success and max_retries > 0:
            retry_count = task[-1]
            task_data = task[:-1]
            if retry_count < max_retries:
                await asyncio.sleep(delay)
                new_task = (*task_data, retry_count + 1)  # 重建任务，重试次数+1
                await queue.put(new_task)
                print(f"[Task Retry] {task_data} (attempt {retry_count + 1})")
            else:
                print(f"[Task Failed] {task_data}")

        queue.task_done()  # 必须调用，标记任务完成


async def stop_worker(queue: asyncio.Queue, worker_tasks: list):
    '''优雅停止所有 worker'''
    try:
        await queue.join()  # 等待队列清空

        print("All tasks processed. Stopping consumers...")
        for _ in worker_tasks:
            await queue.put(None)  # 发送停止信号
    except Exception as e:
        print(f"[Tasks Error] {e}, attempting to cancel workers...")
        for c in worker_tasks:
            c.cancel()

    finally:
        # 统一回收所有任务
        await asyncio.gather(*worker_tasks, return_exceptions=True)


from fastapi import FastAPI

Ideatech_Host = '***.ideatech.info'
Ideatech_API_Key = '***'
DeepSeek_API_Key = 'sk-***'
DashScope_Service_Key = 'sk-***'
MODEL_EMBEDDING = 'text-embedding-v2'
Collection_Name = 'co_analysis'

ai_client = AsyncOpenAI(base_url='https://api.deepseek.com', timeout=300, api_key=DeepSeek_API_Key)
emb_client = AsyncOpenAI(base_url='https://dashscope.aliyuncs.com/compatible-mode/v1', timeout=300,
                         api_key=DashScope_Service_Key)
qd_client = AsyncQdrantClient(host="10.10.10.5", grpc_port=6334, prefer_grpc=True)
dbop = OperationMysql(
    # host="localhost",
    # port=3307,
    # user="root",
    # password="123456",
    # db_name="hammer"
    host="10.10.10.5",
    port=3306,
    user="dooven",
    password="***",
    db_name="***",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up.")
    print(f"[DEBUG] 配置信息:")
    print(f"Host: {Ideatech_Host}")
    print(f"API Key: {Ideatech_API_Key}")

    await dbop.init_pool(minsize=10, maxsize=50)
    # app.state.db_pool = _dbop.pool
    app.state.httpx_client = httpx.AsyncClient(limits=httpx.Limits(max_connections=100, max_keepalive_connections=30),
                                               timeout=httpx.Timeout(timeout=100, read=60.0, write=30.0, connect=5.0))

    if not await qd_client.collection_exists(collection_name=Collection_Name):
        embeddings = await ai_embeddings(inputs=Collection_Name, client=emb_client, model=MODEL_EMBEDDING)
        if embeddings and embeddings[0]:
            size = len(embeddings[0])  # 1536
            collection_info = await create_collection(Collection_Name, client=qd_client, size=size)
            print(collection_info)
    else:
        pass
        # await qd_client.create_payload_index(collection_name=Collection_Name,
        #                                      field_name='batch_no', field_schema='keyword')
        # await qd_client.create_payload_index(collection_name=Collection_Name,
        #                                      field_name='insert_id', field_schema='integer')
    load_jieba()

    try:
        yield
    finally:
        print("Shutting down.")
        await dbop.close_pool()
        await app.state.httpx_client.aclose()


QUERY_TASKS = [
    {
        "url_template": "https://{host}/cloudidp/api/company-exception-list?key={api_key}&keyWord={keyword}",
        "desc": "经营异常名录",
        "question": "企业是否存在经营异常记录？若有，列出异常原因与时间。"
    },
    {
        "url_template": "https://{host}/cloudidp/api/saic-basic-info?key={api_key}&keyWord={keyword}",
        "desc": "工商基本信息",
        "question": "请判断企业基本注册信息中是否存在异常，例如注册资本异常波动、频繁变更法人或地址等情况。"
    },
    {
        "url_template": 'https://{host}/cloudidp/outer/exactSaicInfo?key={api_key}&keyWord={keyword}',
        "desc": "工商全维度信息",
        "question": '''请基于企业在工商系统中的全量信息（包括注册信息、股东结构、历史变更、年报数据、行政处罚、经营异常、对外投资与担保等），从以下角度综合判断企业是否存在潜在经营或合规风险：
        1. 基本信息是否稳定，是否存在频繁的注册资本变动、法定代表人或注册地址变更；
        2. 股东结构是否复杂，是否频繁变更股东或实控人；
        3. 是否存在行政处罚记录、列入经营异常或严重违法失信名单等情形；
        4. 年报数据是否合理，包括：从业人数为 0、营业收入为负、长期亏损、社保与纳税金额为 0 或与规模不符；
        5. 是否存在对外投资/担保过多，形成复杂的关联链条或疑似壳公司结构；
        6. 是否存在分支机构异常、短期设立多个分支、经营范围与主业偏离等特征；
        请结合企业年报信息与工商登记数据是否一致，分析其经营合规性与潜在风险。"
        '''
    },
    {
        "url_template": "https://{host}/cloudidp/api/company-out-investment?key={api_key}&keyWord={keyword}",
        "desc": "对外投资情况",
        "question": "企业是否存在对外投资？若有，请分析投资行业是否集中，是否涉及高风险领域。"
    },
    {
        "url_template": "https://{host}/cloudidp/api/base-account-record?key={api_key}&keyWord={keyword}",
        "desc": "基本账户履历",
        "question": "请分析企业基本账户是否存在频繁变更、账户被撤销等风险迹象。"
    },
    {
        "url_template": "https://{host}/cloudidp/api/annual-report-info?key={api_key}&keyWord={keyword}",
        "desc": "企业年报信息",
        "question": '''请检查企业年报信息是否存在以下问题，并简要说明可能原因或风险：
        1. 年报是否存在缺失或未按时披露（如某些年份缺报或滞后严重）；
        2. 财务数据是否异常（如收入为负、连续多年亏损、资产负债结构异常等）；
        3. 年报信息与工商登记数据是否存在不一致（如注册资本、从业人数、对外投资等不匹配）。'''
    },
    {
        "url_template": "https://{host}/cloudidp/api/company-black-list?key={api_key}&keyWord={keyword}",
        "desc": "严重违法信息",
        "question": "企业是否被列入工商严重违法失信名单？若有，请说明原因及时间。"
    },
    {
        "url_template": "https://{host}/cloudidp/outer/finalBeneficiary?key={api_key}&keyWord={keyword}",
        "desc": "最终受益人识别",
        "question": "请说明企业的最终受益人结构是否复杂，是否涉及疑似隐名股东或关联人。"
    },
    {
        "url_template": "https://{host}/cloudidp/outer/equityShareList?key={api_key}&keyWord={keyword}",
        "desc": "股权结构",
        "question": "请分析企业股权结构是否集中，是否存在交叉持股或频繁变更现象。"
    },
    # {
    #     "url_template": "https://{host}/cloudidp/api/identity/microEnt?key={api_key}&keyWord={keyword}",
    #     "desc": "小微企业识别",
    #     "question": "该企业是否为小微企业？如是，请简要说明识别依据。"
    # },
    {
        "url_template": "https://{host}/cloudidp/api/simple-cancellation?key={api_key}&keyWord={keyword}",
        "desc": "简易注销公告",
        "question": "企业是否已申请简易注销？若有，请说明公告信息。"
    },
    # {
    #     "url_template": "https://{host}/cloudidp/api/tax-arrears-info?key={api_key}&keyWord={keyword}",
    #     "desc": "欠税信息",
    #     "question": "企业是否存在欠税记录？如有，涉及哪些税种与欠税余额？"
    # },
    {
        "url_template": "https://{host}/cloudidp/api/court-notice?key={api_key}&keyWord={keyword}",
        "desc": "开庭公告",
        "question": "是否存在开庭公告？若有，请列出案件简要信息、开庭时间"
    },
    {
        "url_template": "https://{host}/cloudidp/api/judgment-doc?key={api_key}&keyWord={keyword}",
        "desc": "裁判文书",
        "question": "企业是否涉及已判决的法律纠纷？若有，请简述判决内容与案件类型等关键信息。"
    },
    {
        "url_template": "https://{host}/cloudidp/api/court-announcement?key={api_key}&keyWord={keyword}",
        "desc": "法院公告",
        "question": "是否存在法院公告信息？若有，请简要说明公告内容。"
    },
    {
        "url_template": "https://{host}/cloudidp/api/dishonesty?key={api_key}&keyWord={keyword}",
        "desc": "失信信息",
        "question": "企业是否被列入失信被执行人名单？如有，请说明失信行为。"
    },
    {
        "url_template": "https://{host}/cloudidp/api/implements?key={api_key}&keyWord={keyword}",
        "desc": "被执行信息",
        "question": "企业是否存在被执行记录？若有，请说明执行金额和执行法院。"
    },
    {
        "url_template": "https://{host}/cloudidp/api/stock-freeze?key={api_key}&keyWord={keyword}",
        "desc": "股权冻结信息",
        "question": "企业是否存在股权被冻结情况？请说明冻结股东及金额。"
    },
    {
        "url_template": "https://{host}/cloudidp/api/case-filing?key={api_key}&keyWord={keyword}",
        "desc": "立案信息",
        "question": "是否存在立案记录？若有，请说明案件类型和立案时间。"
    },
    {
        "url_template": "https://{host}/cloudidp/api/shellCompany?key={api_key}&keyWord={keyword}",
        "desc": "空壳企业识别",
        "question": "该企业是否疑似为空壳企业？请说明判断依据。"
    }
]


async def background_consumer(batch_no: str, queue):
    """消费者：从队列获取并执行任务"""
    system_prompt = '''你是一个结构化数据处理与信息切片助手，任务是将 JSON 返回的数据转化为适合做文本嵌入（embedding）的自然语言段落/句子，保留原始字段信息以支持后续溯源分析，并合理控制每个分段的长度。

    【处理目标】
    - 输入是一段结构化的接口返回 JSON 数据。
    - 你需要：
      - 识别有价值的信息字段（如法院信息、处罚记录、年报详情等）；
      - 将它们转换为**自然语言段落或句子列表**；
      - 每个段落或句子尽量控制在适中长度（50~300字以内），便于后续做嵌入。
    - 输出应为一个扁平化 JSON 数组，每项为一条句子或段落。

    【处理原则】
    1. **溯源性**：尽可能保留推断的字段标签或字段名，例如“企业名称为：北京科技有限公司”；
    2. **合并简洁字段**：如“类型”“编号”“时间”等可合并成一句简洁描述；
    3. **智能展开结构化字段**：
       - 对于嵌套结构（如年报、法院文书、处罚记录等），每条记录形成一个段落；
       - 段落中保留关键字段内容，不必每个字段一句；
    4. **忽略无效或无数据字段**：
       - 如状态码、无数据提示、无权限等，可省略或统一表述如“未查询到相关处罚信息”；
    5. **控制长度，便于嵌入**：
       - 每段内容不应过长（一般 <300字），必要时可适当切分；
       - 不要碎片化拆成一句话一句字段，避免信息丢失。

    【输出格式】
    返回一个扁平化 JSON 数组，每项为一个段落或句子（字符串），示例如下：
    ```json
    [
      "企业名称：北京科技有限公司，成立于2008年，注册资本为5000万元。",
      "存在一条法院判决记录：案件编号为2019京0102民初999号，文书类型为判决书，发布日期为2019年5月3日。",
      "2022年度年报显示，企业从业人数为120人，营业收入达1.2亿元，实缴资本为3000万元。",
      "未查询到相关的行政处罚信息。"
    ]
    '''
    while True:
        task = await queue.get()
        if task is None:
            queue.task_done()
            break
        origin_question, interface_result = task
        try:
            desc = f'请根据上述要求，对以下{origin_question}接口返回的 JSON 数据进行处理：'
            result = await ai_analyze(system_prompt, interface_result, client=emb_client, desc=desc, model="qwen3-32b",
                                      max_tokens=8192, top_p=0.85, extra_body={"enable_thinking": False})

            parsed = extract_json_array(result)
            chunk_lists = [s.strip() for s in parsed if isinstance(s, str) and s.strip()] if isinstance(parsed,
                                                                                                        list) else []
            if chunk_lists:
                embeddings = await ai_embeddings(chunk_lists, client=emb_client, model=MODEL_EMBEDDING, batch_size=25)
                payloads = [{'text': chunk, 'chunk_id': i, 'batch_no': batch_no, 'desc': origin_question} for i, chunk
                            in enumerate(chunk_lists)]
                operation_id = await upsert_points(payloads, vectors=embeddings, collection_name=Collection_Name,
                                                   client=qd_client, vector_name=None)
                if not operation_id:
                    pass
            else:
                print(f"[Chunk Fail] {origin_question} => {result}")
        except Exception as e:
            print(f"[Consumer Error] {origin_question} => {str(e)}")
        finally:
            queue.task_done()  # 必须调用！标记任务完成


async def process_embedding_task(task: tuple, **kwargs):
    system_prompt_phase1 = '''
    你是一个结构化数据理解助手，任务是将结构化的 JSON 数据转化为通顺、连贯的自然语言描述文章，用于后续语义理解与信息切片。

    【输入说明】
    - 输入是一段结构化接口返回数据，可能包括：企业基本信息、法院记录、年报数据、处罚情况等；
    - 数据格式为标准 JSON，字段名称明确，可能包含嵌套结构或数组列表；

    【你的任务】
    - 提取所有有价值的信息字段，整合生成一篇连贯、自然、可阅读的中文文章；
    - 每段内容表达应清晰、完整，便于人理解和后续自动处理；
    - **保持字段原意或字段名称所代表的含义，确保信息可溯源**；
    - 无需控制段落数量或字数长度，优先保证信息完整表达；

    【处理规则】
    1. **字段标签溯源性与语义表达明确性**：
       - 每条信息应尽量保留字段名称或其语义推断，例如：
         - `"ent_name": "北京科技有限公司"` → “企业名称为：北京科技有限公司”；
         - `"case_no": "(2019)京0102民初999号"` → “案件编号为：（2019）京0102民初999号”；
       - 如字段名为拼音/英文/缩写等不直观，**可根据字段值智能猜测并补充中文语义描述**；

    2. **结构字段合并**：
       - 可将“成立日期”“注册资本”“企业类型”等基础字段合并为一句简洁表达；
       - 同一类信息（如地址、股东、高管）可归类成段表述；

    3. **嵌套结构展开成自然段**：
       - 对于年报、法院文书、处罚记录等嵌套结构，应将每条记录展开为独立句子或段落；
       - 不必逐字段一行，但需确保关键信息都表达清楚（如编号、时间、类型、金额等）；

    4. **缺失数据统一处理**：
       - 如某类数据为空，可统一使用：“未查询到相关处罚信息” 等自然语言描述；

    5. **最终输出格式**
       - 输出为一整篇自然语言文章，分段表达，适合人类阅读和后续信息切片；
       - 不要返回 Markdown、JSON 或代码格式，仅返回自然语言内容。

    【输出示例】
    企业名称为：北京科技有限公司，成立于2008年，注册资本为5000万元。企业类型为有限责任公司，统一社会信用代码为911101087875XXXXXX，法定代表人为张三。

    该企业存在一条法院判决记录，案件编号为（2019）京0102民初999号，判决时间为2019年5月3日，文书类型为判决书。

    根据2022年度年报，企业从业人数为120人，营业收入为1.2亿元，实缴资本为3000万元。

    未查询到相关的行政处罚信息。
    '''

    system_prompt_phase2 = '''
    你是一个信息切片助手，任务是将一段自然语言文章切分为便于嵌入的句子或段落列表。

    【输入说明】
    - 输入是一段经过结构化处理后的自然语言文章，已经表达了结构化 JSON 中的所有关键信息；
    - 内容格式清晰、语义完整，涵盖如企业信息、年报、法院记录等内容。

    【切片目标】
    - 将文章合理切分为多个**语义完整、内容连续的段落或句子**；
    - **切片必须覆盖原始文章的全部内容，不遗漏任何句子或信息**；
    - 每段控制在 50~300 字之间，适合用于文本嵌入；
    - **不得改变原文表述或语义，不做任何转换、总结、概括或润色**；
    - 切片后的段落内容需保持原文顺序，便于溯源。

    【处理原则】
    1. **保留原文内容**：每个切片必须从原文中截取，不做改写；
    2. **完整表达**：确保每段是一个语义完整的句子或段落，不能中断句意；
    3. **不丢信息**：切片需覆盖原文全部内容，不跳段、不合并；
    4. **按逻辑或语义断点切分**：优先按句号、段落等自然边界切分；
    5. **格式要求**：输出一个 JSON 数组，每项为一个字符串，即一段原文，如：

    ```json
    [
      "企业名称为：北京科技有限公司，成立于2008年，注册资本为5000万元。",
      "企业类型为有限责任公司，统一社会信用代码为911101087875XXXXXX，法定代表人为张三。",
      "存在法院判决记录，案号为（2019）京0102民初999号，发布日期为2019年5月3日，文书类型为判决书。",
      "2022年度年报显示，从业人数为120人，营业收入为1.2亿元，实缴资本为3000万元。"
    ]
    '''
    batch_no, insert_id, origin_question, interface_result = task
    try:
        await dbop.insert('task_question_content',
                          {
                              'id': insert_id, 'batch_no': batch_no, 'origin_question': origin_question,
                              'status': 'running'
                          })

        desc_1 = f'请将以下“{origin_question}”接口返回的 JSON 数据，转化为一篇完整的自然语言文章，便于后续语义处理'
        result_1 = await ai_analyze(system_prompt_phase1, interface_result, client=emb_client, desc=desc_1,
                                    model="qwen3-32b", max_tokens=8192, top_p=0.85,
                                    extra_body={"enable_thinking": False})
        await dbop.run("""
                         UPDATE task_question_content SET status='done',content=%s
                         WHERE id=%s
                     """, (result_1, insert_id))

        desc_2 = f'请将下面针对“{origin_question}”生成的自然语言文章，进行合理信息切片，每条切片用于后续文本嵌入（embedding）'
        result_2 = await ai_analyze(system_prompt_phase2, {"文章内容": result_1}, client=emb_client, desc=desc_2,
                                    model="qwen2.5-32b-instruct", max_tokens=8192, top_p=0.85)

        parsed = extract_json_array(result_2)
        chunk_lists = [s.strip() for s in parsed if isinstance(s, str) and s.strip()] if isinstance(parsed,
                                                                                                    list) else []
        if chunk_lists:
            embeddings = await ai_embeddings(chunk_lists, client=emb_client, model=MODEL_EMBEDDING, batch_size=25)
            payloads = [
                {'text': chunk, 'chunk_id': i, 'batch_no': batch_no, 'insert_id': insert_id, 'desc': origin_question}
                for i, chunk in enumerate(chunk_lists)]
            operation_id = await upsert_points(payloads, vectors=embeddings, collection_name=Collection_Name,
                                               client=qd_client, vector_name=None)
            if operation_id:
                await dbop.run("""
                           UPDATE task_question_content SET status='completed',operation_id=%s
                           WHERE id=%s
                       """, (operation_id, insert_id))

                return True
        else:
            print(f"[Chunk Fail] {origin_question} => {result_2}")
    except Exception as e:
        print(f"[Consumer Error] {origin_question} => {str(e)}")
        await dbop.run("""
                   UPDATE task_question_content SET status='failed'
                   WHERE id=%s
               """, (insert_id,))

    return False


async def run_enterprise_analysis_task(batch_no: str, company_name: str, host: str, api_key: str, httpx_client) -> str:
    await dbop.insert(
        'task_batch',
        {'batch_no': batch_no, 'title': f"企业分析任务 - {company_name}", 'status': 'running'}
    )

    analysis_prompt = f'''
    请分析该企业（{company_name}）的运营状况，重点判断是否存在异常情况：

    - 如查询无结果，或接口返回异常状态码（如 status: 000207）或错误提示（如“未申请此接口”），请直接给出简明结论，例如：
      - “当前系统中未显示贵公司存在股权被冻结的情况”
      - “当前系统中未显示您的立案记录”
      - “贵公司目前未被列入工商严重违法失信名单”
      - “贵公司当前未被列入经营异常名录”

    - 如查询成功，请结合返回信息进行简洁明晰的分析，判断是否存在风险，并说明分析依据。

    - 请勿引用原始字段名、标签或数据结构代码，仅提供自然语言表达的分析结论。

    下面是本项查询所需关注的问题：
    '''
    queue = asyncio.Queue()

    # 异步执行所有子任务
    async def run_subtask(i, task_conf):
        question_no = f"Q{i + 1:03}"
        url = task_conf["url_template"].format(
            host=host, api_key=api_key, keyword=company_name
        )

        prompt = task_conf["question"]
        origin_question = task_conf["desc"]

        insert_id = await dbop.run("""
                      INSERT INTO task_question (batch_no, question_no, origin_question, formatted_prompt, status)
                      VALUES (%s, %s, %s, %s, %s)
                  """, (batch_no, question_no, origin_question, prompt, "created"))

        # 发起 HTTP 请求
        try:
            print(f"\n[DEBUG] 完整请求URL: {url}")
            interface_result = await fetch_url(url, httpx_client, max_retries=3, delay=3)
            task = (batch_no, insert_id, origin_question, interface_result)
            await queue.put(task)
            #  await asyncio.create_task(process_embedding_task(task))
            print(f"已接收请求: {origin_question} | 队列大小: {queue.qsize()}")

            await dbop.run("""
                        UPDATE task_question SET status='running', interface_result=%s
                        WHERE batch_no=%s AND question_no=%s
                    """, (json.dumps(interface_result, ensure_ascii=False), batch_no, question_no))

            # 让 AI 进行分析
            prompt = analysis_prompt + '\n' + prompt
            result = await ai_analyze(prompt, interface_result, ai_client, origin_question, model='deepseek-chat')

            await dbop.run("""
                        UPDATE task_question SET status='completed', result=%s
                        WHERE batch_no=%s AND question_no=%s
                    """, (result, batch_no, question_no))

            return result

        except Exception as e:
            err_msg = f"[{question_no}] Error during processing: {str(e)}"

            await dbop.run("""
                            UPDATE task_question SET status='failed', result=%s
                            WHERE batch_no=%s AND question_no=%s
                        """, (err_msg, batch_no, question_no))
            print(err_msg)
            return err_msg

    worker_consumers_background = [asyncio.create_task(consumer_worker(queue, process_embedding_task, 0)) for _ in
                                   range(len(QUERY_TASKS))]
    all_results = await asyncio.gather(*(run_subtask(i, task_conf) for i, task_conf in enumerate(QUERY_TASKS)))

    await dbop.insert('task_summary_question', {'batch_no': batch_no, 'status': 'created'})

    rows = await dbop.run("SELECT origin_question,result FROM task_question WHERE batch_no=%s", (batch_no,))
    all_results = [row["result"] for row in rows]
    joined_questions = "、".join(row["origin_question"] for row in rows)

    # 汇总总结
    # summary_prompt = f"请根据以下关于企业（{company_name}）{joined_questions}的多项分析，做一个整体风险与运营状况总结。"
    summary_prompt = f"""
    你是一位企业风控分析专家，请根据以下关于企业【{company_name}】的全维度分析结果，生成一篇《整体风险与运营状况总结》报告。

    分析来源维度包括（但不限于）：
    {joined_questions}

    请遵循以下要求进行撰写：

    1. **结构清晰，内容完整**，建议使用如下四级结构：
       - 核心优势与稳定性（如股东结构、合规记录、经营连续性）
       - 重大风险与问题（如治理混乱、法律诉讼、财务隐患等）
       - 风险关联性分析（建议以表格形式，列出风险类型、表现、潜在影响）
       - 结论与建议（突出重大隐患，提出可执行优化措施，并注明优先级）

    2. **维度评分与总分**：请按照下表格式稳定输出五个核心维度的得分，字段请完整输出为：维度、得分（0-100）、得分依据、权重、加权得分，并计算总分。注意表格字段顺序与命名保持一致，便于系统解析：
    
    **维度评分与总分**
    
    基于多维度分析，对企业进行量化评估。权重默认设定如下：治理稳定性25%、法律风险程度25%、财务透明度20%、主业聚焦程度15%、运营规范性15%。总分通过加权平均计算：
    
    | 维度         | 得分（0-100） | 得分依据（简要说明） | 权重 | 加权得分 |
    |--------------|---------------|------------------------|------|-----------|
    | 治理稳定性   | XX            | ……                     | 25%  | XX        |
    | 法律风险程度 | XX            | ……                     | 25%  | XX        |
    | 财务透明度   | XX            | ……                     | 20%  | XX        |
    | 主业聚焦程度 | XX            | ……                     | 15%  | XX        |
    | 运营规范性   | XX            | ……                     | 15%  | XX        |
    | **总分**     | **XX**        | **示例：70×0.25 + … = 74** |      |           |
    
    **总分说明**：请总结该分值所代表的风险等级，并指出主导风险维度，如有必要可说明是否调整权重。示例：“74 分表示中等风险水平，主要由财务与法律维度拉低，总体可控，仍建议优化以提升至 80+。”
       
   3. **推荐非柜面开户额度区间**：
   - 请基于企业整体评分，结合以下财务与基础信息，综合推导企业的开户额度建议区间：
     - 财务指标：注册资本、实缴资本、近年纳税额、营业收入、交易流水等；
     - 基本信息：是否新设、注册时长、变更频率、法人是否一致、合规与风控状况等；
   - 核算逻辑请包含以下两步（可自然表达，不必公式化）：
     1. **计算基础额度**：基于财务与业务规模估算的合理初始额度；
     2. **风险调整**：参考风险评分对基础额度进行调整（如使用风险系数递减），使最终额度匹配企业风险水平；
   - 请输出推荐额度区间，并说明信源比逻辑（即额度推导所依据的核心数值或风险因子）

   **示例输出格式（仅供参考）**：
   - **推荐开户额度区间**：300 万元 ~ 500 万元
   - **信源比逻辑说明**：该企业注册资本为 1000 万元，实缴 600 万元，年营收约 1500 万元，近一年交易流水显著增长，合规记录良好。基础额度估算为 600 万元，因存在一定治理不稳与财务披露滞后，整体评分中等（74分），按风险系数 0.8 调整，最终建议额度区间为 300 万~500 万元。

    4. 风格要求：专业、冷静、结构化，不要逐条复述原始数据，要做总结和归纳，突出**风险优先级与应对建议**。

    请生成完整报告正文，包括各项维度评分、整体评分及非柜面开户额度区间说明。
    """
    await dbop.run("""
                UPDATE task_summary_question SET summary_question=%s,status='running'
                WHERE batch_no=%s
            """, (summary_prompt, batch_no))

    try:
        summary_result = await ai_analyze(summary_prompt, all_results, ai_client, desc='企业各项分析',
                                          model='deepseek-reasoner', max_tokens=8192)
        print(summary_result)

        await dbop.execute([
            ("""
                UPDATE task_summary_question 
                SET summary_answer=%s, model=%s, status='completed'
                WHERE batch_no=%s
                """,
             (summary_result, 'deepseek-reasoner', batch_no)
             ),
            ("""
                UPDATE task_batch SET status='completed', completed_at=NOW()
                WHERE batch_no=%s
                """,
             (batch_no,)
             )
        ])


    except Exception as e:
        summary_result = f"分析失败：{str(e)}"

        await dbop.execute([
            ("""
            UPDATE task_summary_question 
            SET summary_answer=%s, model=%s, status='failed'
            WHERE batch_no=%s
            """, (summary_result, 'deepseek-reasoner', batch_no)
             ),
            ("""
            UPDATE task_batch SET status='failed'
            WHERE batch_no=%s
            """, (batch_no,)
             )
        ])

    await stop_worker(queue, worker_consumers_background)

    return summary_result


async def run_enterprise_analysis_task_background(company_name: str, httpx_client):
    batch_no = uuid.uuid4().hex[:16]
    # 异步后台调度任务，不阻塞主流程
    task = asyncio.create_task(
        run_enterprise_analysis_task(batch_no, company_name, Ideatech_Host, Ideatech_API_Key, httpx_client))
    '''
    ✅ 1. 治理稳定性（25%）
    
    可映射字段：
    
        法定代表人、董事高管变更记录（来自：工商基本信息 / 工商全维度信息）
    
        股东结构变化频次（来自：股权结构 / 工商信息）
    
        简易注销公告（是否准备退出）（来自：简易注销公告）
    
        最终受益人结构是否集中、是否频繁变化（来自：最终受益人识别）
    
        基本账户履历变更频次（来自：基本账户履历）
    
    评估逻辑：
    
        多次变更 → 不稳定
    
        国资或大股东稳定持股 → 稳定
    
        注销意图或实缴资本缺失 → 潜在治理放弃
    
    ✅ 2. 法律风险程度（25%）
    
    可映射字段：
    
        开庭公告、裁判文书、法院公告、立案信息（诉讼频次、案由、金额）
    
        被执行信息、失信信息、股权冻结信息（司法风险强度）
    
        严重违法记录（如重大违规处罚）
    
        对外投资公司是否涉及上述司法信息（穿透式风控）
    
    评估逻辑：
    
        法律纠纷频繁，执行金额大 → 风险高
    
        出现失信 / 被执行 → 高风险
    
        法院公告+股权冻结 → 联合判断是否存在资产冻结困境
    
    ✅ 3. 财务透明度（20%）
    
    可映射字段：
    
        企业年报信息（是否完整公示财务数据，是否连续年报）
    
        纳税等级与缴税额（如年报中包含）
    
        基本账户信息稳定性（间接判断财务账户是否频繁调整）
    
        空壳企业识别（如“无人员、无经营”可判断其财务真实性存疑）
    
    评估逻辑：
    
        年报未披露营收/利润 → 低透明度
    
        连续多年缺年报或数据缺失 → 极低透明度
    
        被标记为空壳企业 → 严重财务信息失真
    
    ✅ 4. 主业聚焦程度（15%）
    
    可映射字段：
    
        工商经营范围（如包含无关行业扩张、主业漂移）
    
        对外投资行业是否偏离主业（来自：对外投资情况）
    
        企业自身标签 / 所属行业 与 实际业务偏离（可结合上下游分析）
    
    评估逻辑：
    
        新增经营项目为不相关行业 → 主业不聚焦
    
        投资多个异业公司 → 资源分散
    
        无明确主业描述/注册范围含糊 → 判断为“主业模糊”
    
    ✅ 5. 运营规范性（15%）
    
    可映射字段：
    
        是否列入经营异常名录（主要）
    
        严重违法信息（如工商处罚等）
    
        年报连续性、社保人数与门店匹配性（企业年报）
    
        分支机构是否齐全、负责人是否兼任多个门店（工商全维度）
    
        是否空壳（实际办公地址、人员、经营情况）
    
    评估逻辑：
    
        被列入经营异常名录 → 明显不规范
    
        年报不及时、社保/门店数据不一致 → 操作不规范
    
        实体空壳 / 负责人“一肩挑多店” → 内控薄弱
    '''
    return batch_no, task
