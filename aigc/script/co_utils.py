import qdrant_client.models as qcm
import json, re, uuid, time
import asyncio
import aiomysql
import httpx
import difflib
import logging

from contextlib import asynccontextmanager
from qdrant_client import AsyncQdrantClient
from openai import AsyncOpenAI
from typing import Optional, Literal, Dict, List, Tuple, Union, Any, Callable, AsyncIterator, Type, Awaitable
from functools import wraps
from collections import defaultdict
from datetime import datetime

from bm25 import BM25, load_jieba
from co_prompt import SYS_PROMPT
from co_tasks import QUERY_TASKS


def async_error_logger(max_retries: int = 0, delay: int | float = 1, backoff: int | float = 2,
                       exceptions: Type[Exception] | tuple[Type[Exception], ...] = Exception,
                       extra_msg: str = None, logger_name: Optional[str] = None, log_level: int = logging.ERROR):
    """
    异步函数的错误重试和日志记录装饰器

    参数:
        max_retries (int): 最大重试次数（不含首次尝试），默认为 0，表示不重试；设为 1 表示失败后重试一次（共尝试 2 次）。
        delay (int/float): 初始延迟时间(秒)，默认为1
        backoff (int/float): 延迟时间倍增系数，默认为2
        exceptions (Exception/tuple): 要捕获的异常类型，默认为所有异常
        logger_name (Logger): 自定义logger，默认使用模块logger
        log_level (int): 日志级别
    """
    logger = logging.getLogger(logger_name)

    def decorator(func: Callable[..., Awaitable]):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 优先使用调用时的参数，如果没有就用装饰器默认值
            _max_retries = kwargs.pop("max_retries", max_retries)
            _delay = kwargs.pop("delay", delay)
            _backoff = kwargs.pop("backoff", backoff)
            _extra_msg = kwargs.pop("extra_msg", extra_msg)

            retry_count = 0
            current_delay = _delay

            while True:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    retry_count += 1
                    msg = f"Async function {func.__name__} failed with error: {str(e)}."
                    if _extra_msg:
                        msg += f" | Extra: {_extra_msg}"
                    if retry_count > _max_retries:
                        logger.log(log_level, f"{msg} After {_max_retries} retries", exc_info=True)
                        raise  # 重试次数用尽后重新抛出异常

                    logger.log(log_level, f"{msg} Retrying {retry_count}/{_max_retries} in {current_delay} seconds...")
                    await asyncio.sleep(current_delay)
                    current_delay *= _backoff  # 指数退避

        return wrapper

    return decorator


@async_error_logger(max_retries=2, delay=3, exceptions=(Exception, httpx.HTTPError))
async def fetch_url(url, httpx_client, timeout=30):
    response = await httpx_client.get(url, timeout=timeout)
    response.raise_for_status()
    try:
        result = response.json()
    except json.JSONDecodeError:
        result = response.text

    if result.get('status') == "fail" and result.get("code") == "000205":  # "数据正在计算，请稍后再试
        raise Exception(f"[{url}] 接口失败，原因：{result.get('reason', response.text)}")  # 主动抛出业务异常，让装饰器重试

    return result


# async def fetch_url(url, httpx_client, max_retries=3, delay=3):
#     for attempt in range(max_retries):
#         try:
#             response = await httpx_client.get(url, timeout=30)
#             response.raise_for_status()
#             result = response.json()
#             # if result.get('status')=='000103':
#             #     return
#             if result.get('status') == "fail" and result.get("code") == "000205":
#                 if attempt < max_retries - 1:
#                     await asyncio.sleep(delay)
#                     continue
#                 else:
#                     print(f"[{url}] 接口多次返回 fail:{result.get('reason', response.text)}，放弃重试。")
#
#             return result
#
#         except httpx.HTTPError as e:
#             if attempt < max_retries - 1:
#                 print(f"[{url}] 请求异常 {e}，第 {attempt + 1} 次尝试后等待 {delay} 秒重试...")
#                 await asyncio.sleep(delay)
#             else:
#                 raise e
#
#     return None


@async_error_logger(1)
async def ai_analyze(system_prompt: str, results: dict | list | str, client, desc: str = None,
                     model='deepseek-reasoner', max_tokens=4096, temperature: float = 0.2, **kwargs):
    user_request = results if isinstance(results, str) else json.dumps(results, ensure_ascii=False)
    if desc:
        user_request = f"{desc}:\n{user_request}"

    messages = [{"role": "system", "content": system_prompt}, {'role': 'user', 'content': user_request}]
    payload = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
    )
    if Local_Base_Url:
        payload['user'] = 'infr'
    payload.update(kwargs)
    completion = await client.chat.completions.create(**payload)
    content = completion.choices[0].message.content
    print(f'</{desc}>: {content}')  # reasoning_content
    return content.split("</think>")[-1]


@async_error_logger(1)
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


async def upsert_points(payloads: list[dict], vectors: list, collection_name: str, client, vector_name: str = None,
                        hash_id=True, **kwargs):
    """
    插入单个点到 Qdrant 集合。
    """
    assert len(payloads) == len(vectors), "Payloads and vectors must have the same length."

    # dense_doc=qcm.Document(text="bye world", model="sentence-transformers/all-MiniLM-L6-v2")#DENSE
    if hash_id:
        points: list[qcm.PointStruct] = [qcm.PointStruct(id=str(uuid.uuid4()), payload=payload,
                                                         vector={vector_name: vector} if vector_name else vector)
                                         for payload, vector in zip(payloads, vectors)]
    else:
        current_count = await client.count(collection_name=collection_name, exact=True)
        points: list[qcm.PointStruct] = [qcm.PointStruct(id=current_count.count + idx, payload=payload,
                                                         vector={vector_name: vector} if vector_name else vector)
                                         for idx, (payload, vector) in enumerate(zip(payloads, vectors))]

    try:
        operation_info = await client.upsert(collection_name=collection_name, points=points, **kwargs)
        assert operation_info.status == qcm.UpdateStatus.COMPLETED, operation_info.model_dump()
        return operation_info.operation_id, [p.id for p in points]
    except Exception as e:
        # client.upload_points(collection_name=collection_name, points=points, parallel=parallel, wait=True)
        # client.upload_records(collection_name=collection_name,
        #                       records=(qcm.Record(id=idx, vector=vec.tolist()) for idx, vec in enumerate(vectors)),
        #                       parallel=parallel, wait=True)
        print(f"Error upserting points to collection {collection_name}: {e}")
        return None, []


def field_match(field_key, match_values):
    """匹配字段值，支持单值和多值匹配"""
    if not field_key or not match_values:
        return []
    if isinstance(match_values, (str, int, float)):
        return [qcm.FieldCondition(key=field_key, match=qcm.MatchValue(value=match_values, ), )]
    if isinstance(match_values, (list, tuple, set)):
        return [qcm.FieldCondition(key=field_key, match=qcm.MatchAny(any=list(match_values)))]
    return [qcm.FieldCondition(key=field_key, match=qcm.MatchValue(value=v), ) for v in match_values]


@async_error_logger(1)
async def fix_missing_payload(hits, collection_name, client, with_payload: bool | list[str] = True):
    missing_hits = [h for h in hits if not getattr(h, 'payload', None)]  # or h.payload is None
    if missing_hits:
        id_records = await client.retrieve(collection_name, [h.id for h in missing_hits], with_payload=with_payload)
        # fallback：用 retrieve 填充 回填对应的 payload
        id_map = {r.id: r for r in id_records}
        for h in missing_hits:
            record = id_map.get(h.id)
            if record:
                h.payload = record.payload


async def search_by_vecs(query_vectors: list[list[float]], collection_name: str, client,
                         payload_key: str | list = 'title', vector_name: str = None,
                         match: list = [], not_match: list = [],
                         topn: int = 10, score_threshold: float = 0.0, exact: bool = False, **kwargs):
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

    return await client.search_batch(collection_name=collection_name, requests=search_queries, **kwargs)


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
        if keys is True:
            return p.payload, p.score, p.id
        if p.payload:
            if isinstance(keys, str):
                return p.payload.get(keys, None), p.score, p.id
            if isinstance(keys, list):
                return {k: p.payload.get(k) for k in keys}, p.score, p.id
        return p.payload, p.score, p.id

    if len(querys) == 1:
        with_payload = False
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
        if with_payload:
            await fix_missing_payload(search_hit, collection_name, client, with_payload=with_payload)
        # {k: p.payload.get(k) for k in payload_key}
        return [extract_payload(p, payload_key) for p in search_hit]

    search_hit = await search_by_vecs(query_vectors, collection_name, client, payload_key, vector_name,
                                      match, not_match, topn, score_threshold, exact)  # ScoredPoint

    return [(item, [extract_payload(p, payload_key) for p in hit])
            for item, hit in zip(querys, search_hit)]


async def recommend_by_id(ids, collection_name, client, payload_key: str | list = 'word', match=[], not_match=[],
                          not_ids=[], topn=10, score_threshold: float = 0.0, get_dict=True, **kwargs):
    not_match_ids = [qcm.HasIdCondition(has_id=not_ids)]
    query_filter = qcm.Filter(must=match, must_not=not_match_ids + not_match)
    with_payload = False
    if payload_key:
        if isinstance(payload_key, str):
            with_payload = [payload_key]
        if isinstance(payload_key, list):
            with_payload = payload_key

    recommend_hit = await client.recommend(collection_name=collection_name,
                                           positive=ids,  # [ID]
                                           query_filter=query_filter,
                                           limit=topn, score_threshold=score_threshold,
                                           with_payload=with_payload,
                                           **kwargs)  # ScoredPoint
    if with_payload:
        await fix_missing_payload(recommend_hit, collection_name, client, with_payload=with_payload)

    def extract_payload(p, keys):
        if get_dict:
            return p.model_dump(exclude={'vector', 'version', 'shard_key', 'order_value'}, exclude_none=True)
        if keys is True:
            return p.payload, p.score, p.id
        if p.payload:
            if isinstance(keys, str):
                return p.payload.get(keys, None), p.score, p.id
            if isinstance(keys, list):
                return {k: p.payload.get(k) for k in keys}, p.score, p.id
        return p.payload, p.score, p.id

    return [extract_payload(p, payload_key) for p in recommend_hit]


def extract_json_array(input_data) -> list | None:
    """
    提取并解析 markdown 包裹的 JSON 数组（尤其是 ```json ... ``` 格式）
    """
    # 提取 ```json ... ``` 块中内容
    md_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', input_data, re.DOTALL)
    if md_match:
        json_str = md_match.group(1)
    else:
        # 查找字符串中最外层 JSON 数组（首次出现）
        array_match = re.search(r"\[[\s\S]*?\]", input_data)
        if array_match:
            json_str = array_match.group(0)
        else:
            json_str = input_data.strip()  # 若本身就是 JSON 字符串（无包裹）

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
    return None


def extract_json_struct(input_data: str | dict) -> dict | None:
    """
    从输入中提取结构化 JSON 对象，兼容以下格式：
    支持多种格式（Markdown JSON 块、普通 JSON 字符串、字典等）,支持已经是字典的输入
    - 直接为 dict 类型
    - 标准 JSON 字符串
    - Markdown JSON 块（```json ... ```)
    - 字符串中嵌入 JSON 部分（提取第一个 {...} 段）
    """

    if isinstance(input_data, dict):
        return input_data

    # Markdown JSON 格式：```json\n{...}\n```
    md_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", input_data, re.DOTALL)
    if md_match:
        json_str = md_match.group(1)
    else:
        # 尝试提取最外层 JSON：匹配最先出现的大括号包裹段，但可能不处理嵌套的 JSON
        brace_match = re.search(r"\{.*\}", input_data, re.DOTALL)
        if brace_match:
            json_str = brace_match.group(0)
        else:
            # 输入可能是标准 JSON 字符串（无包裹）
            json_str = input_data.strip()

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"invalid json format, JSON 解析失败: {e}\n原始输入: {input_data}")
    return None


def extract_json_str(json_code: str) -> str:
    """
    模型返回的内容，其中 JSON 数据通常被包裹在 Markdown 的代码块标记中（即以 json 开始，以 结束）
    如果未找到起始或结束标记，尝试直接解析整个字符串为 JSON
    :param json_code:
    :return:
    """
    start = json_code.find("```json")
    # 从start开始找到下一个```结束
    end = json_code.find("```", start + 1)
    if start == -1 or end == -1:
        try:
            json.loads(json_code)
            return json_code
        except Exception as e:
            print("Error:", e)
        return ""
    return json_code[start + 7:end]


def get_origin_fild(index: int, data: dict[str, list] | list[tuple[str, list]]):
    items = data.items() if isinstance(data, dict) else data
    count = 0
    for k, v in items:
        count += len(v)
        if index < count:
            return k
    return None


def map_fields(data: dict, mapping: dict) -> dict:
    '''递归遍历字典，根据提供的映射规则修改键名'''

    def translate_key(prefix, key):
        full_key = f"{prefix}.{key}" if prefix else key
        return mapping.get(full_key, mapping.get(key, key))

    def recursive_map(obj, prefix=""):
        if isinstance(obj, dict):
            return {translate_key(prefix, k): recursive_map(v, f"{prefix}.{k}" if prefix else k)
                    for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recursive_map(item, prefix) for item in obj]
        else:
            return obj

    return recursive_map(data)


def extract_interface_map(data: dict, field_path: list, mapping: dict):
    """
    从 data 中按 field_path 提取字段，并用 mapping 映射字段。
    支持 field_path 为 str 或 list。
    """

    def get_nested_value(d, path):
        if isinstance(path, str):
            path = [path]
        for key in path:
            if isinstance(d, dict):
                d = d.get(key, {})
            else:
                return {}
        return d  # data.get("data", data.get("result", data))

    extracted = get_nested_value(data, field_path) if field_path else data
    if not mapping:
        return extracted

    if isinstance(extracted, list):
        return [map_fields(item, mapping) for item in extracted]
    elif isinstance(extracted, dict):
        return map_fields(extracted, mapping)

    return extracted  # 兜底处理


def find_best_matches(query: str, template_list: list[str], top_n=3, cutoff=0.8, best=True) -> list[tuple]:
    # 获取满足 cutoff 的匹配
    matches = difflib.get_close_matches(query, template_list, n=top_n, cutoff=cutoff)
    # 计算每个匹配项与查询词的相似度
    if matches:
        return [(match, difflib.SequenceMatcher(None, query, match).ratio(), template_list.index(match))
                for match in matches]
    # 如果没有匹配，则强制返回最相似的 1 个
    if best and template_list:
        scores = [(text, difflib.SequenceMatcher(None, query, text).ratio(), i)
                  for i, text in enumerate(template_list)]
        return [max(scores, key=lambda x: x[1])]
        # sorted( [item for item in scores if item[1] >= cutoff], key=lambda x: -x[1])[:top_n]

    return []  # text, score, idx [(匹配文本, 相似度, 对应原始idx)]


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

    async def run(self, sql: str, params: tuple | dict | list = None, conn=None):
        async def _run(c):
            sql_type = (sql or "").strip().split()[0].lower()

            async with c.cursor(aiomysql.DictCursor) as cur:
                if isinstance(params, list):
                    await cur.executemany(sql, params)
                else:
                    await cur.execute(sql, params or ())

                if sql_type == "select":
                    return await cur.fetchall()
                elif sql_type in {"insert", "update", "delete", "replace"}:
                    await c.commit()  # 显式保险,autocommit=True
                    if sql_type == "insert":
                        return cur.lastrowid or int(c.insert_id())
                    else:
                        return True

                return None

        try:
            if conn:
                return await _run(conn)

            if self.pool is None:
                await self.init_pool()
            async with self.pool.acquire() as conn:
                return await _run(conn)

        except Exception as e:
            print(f"[Async] SQL执行错误: {e}, SQL={sql}")

        return None

    async def execute(self, sql_list: list[tuple[str, tuple | dict | list | None]], conn=None):
        """
        批量执行多条 SQL 并自动提交或回滚（同一个事务）
        """
        if conn:
            should_release = False
        else:
            if self.pool is None:
                await self.init_pool()

            conn = await self.pool.acquire()
            should_release = True

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
            if should_release:
                self.pool.release(conn)

    async def query(self, query_list: list[tuple[str, tuple | dict]], fetch_all: bool = True, cursor=None) -> list:
        async def _run(c) -> list:
            results = []
            for sql, params in query_list:
                await c.execute(sql, params or ())
                result = await c.fetchall() if fetch_all else await c.fetchone()
                results.append(result)
            return results

        if cursor:
            return await _run(cursor)

        if self.pool is None:
            await self.init_pool()

        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                return await _run(cur)

    async def query_one(self, sql: str, params: tuple | dict = (), cursor=None) -> dict | None:
        results = await self.query([(sql, params)], fetch_all=False, cursor=cursor)
        return results[0] if results else None

    async def merge(self, table_name: str, params_data: dict, update_fields: list[str] = None, conn=None):
        """
        插入或更新数据（根据主键或唯一键自动合并）

        Args:
            table_name (str): 表名
            params_data (dict): 要插入的数据（更新必须包含主键/唯一索引字段）
            update_fields (list): 需要更新的字段列表，默认为除了主键以外的字段,在发生冲突时被更新的字段列表,[]为插入
            conn:可选外部传入连接
        """
        if not params_data:
            raise ValueError("参数数据不能为空")

        fields = list(params_data.keys())
        values = tuple(params_data.values())
        field_str = ', '.join(f"`{field}`" for field in fields)
        placeholder_str = ', '.join(['%s'] * len(fields))
        sql = f"INSERT INTO `{table_name}` ({field_str}) VALUES ({placeholder_str})"
        if update_fields is None:
            update_fields = [f for f in fields if f.lower() not in ("id", "created_at", "created_time")]
        if update_fields:
            sql += " AS new"
            update_str = ', '.join(f"`{field}` = new.`{field}`" for field in update_fields)
            # update_str += ", `updated_at` = CURRENT_TIMESTAMP"
            sql += f" ON DUPLICATE KEY UPDATE {update_str}"
        return await self.run(sql, values, conn=conn)

    async def insert(self, table_name: str, params_data: dict, conn=None):
        return await self.merge(table_name, params_data, update_fields=[], conn=conn)

    async def get_table_columns(self, table_name: str, cursor=None) -> List[dict]:
        async def _run(c) -> list:
            await c.execute(f"DESCRIBE {table_name}")
            columns = await c.fetchall()
            # 转换列信息为更友好的格式
            formatted_columns = [{
                "name": col["Field"],
                "type": col["Type"],
                "nullable": col["Null"] == "YES",
                "key": col["Key"],
                "default": col["Default"],
                "extra": col["Extra"]
            } for col in columns]
            return formatted_columns

        if cursor:
            return await _run(cursor)

        if self.pool is None:
            await self.init_pool()

        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                return await _run(cur)

    async def get_conn(self):
        if self.pool is None:
            await self.init_pool()
        return await self.pool.acquire()

    def release(self, conn):
        if self.pool is not None:
            self.pool.release(conn)

    @asynccontextmanager
    async def get_cursor(self, conn=None) -> AsyncIterator[aiomysql.Cursor]:
        should_release = False
        if conn is None:
            conn = await self.get_conn()
            should_release = True

        try:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                yield cursor
        finally:
            if should_release:
                self.release(conn)

    @classmethod
    async def get_async_conn(cls, **kwargs):
        async with cls(**kwargs) as dbop:
            yield dbop

    @staticmethod
    def prepare_value(value) -> Union[str, int, float, bool, None]:
        """
        准备数据库写入值，根据不同类型进行转换：
        - dict → JSON 字符串
        - list/tuple（全字符串）→ \n\n 连接字符串
        - set（全字符串）→ 分号连接字符串
        - 其他类型 → 原样返回（如 int、float、str、bool、None）
        """
        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)  # 处理字典类型 → JSON序列化, indent=2

        if isinstance(value, (tuple, list)):
            if all(isinstance(item, str) for item in value):
                return "\n\n".join(value)

            return json.dumps(value, ensure_ascii=False)  # 非全字符串元素则JSON序列化

        if isinstance(value, set):
            if all(isinstance(item, str) for item in value):
                return ";".join(sorted(value))
            return json.dumps(list(value), ensure_ascii=False)

        # 其他类型保持原样 (None等)
        return str(value)


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

Ideatech_Host = 'matrix.***.info'
Local_Base_Url = 'http://**:7000/v1'
Ideatech_API_Key = '***'
DeepSeek_API_Key = '***'
DashScope_Service_Key = '***'
MODEL_EMBEDDING = 'text-embedding-v2'
Collection_Name = 'co_analysis'
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.WARNING,
                    handlers=[
                        logging.StreamHandler(),  # 输出到终端,控制台输出
                        logging.FileHandler('errors.log'),
                    ])

ai_client = AsyncOpenAI(base_url=Local_Base_Url or 'https://api.deepseek.com', timeout=300, api_key=DeepSeek_API_Key)
emb_client = AsyncOpenAI(base_url=Local_Base_Url or 'https://dashscope.aliyuncs.com/compatible-mode/v1', timeout=300,
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
    user="***",
    password="***",
    db_name="kettle",
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


async def background_consumer(batch_no: str, queue):
    """消费者：从队列获取并执行任务"""
    system_prompt = SYS_PROMPT.get('chunking_prompt')
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
                operation_id, ids = await upsert_points(payloads, vectors=embeddings, collection_name=Collection_Name,
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
    system_prompt = SYS_PROMPT.get('chunking')
    batch_no, insert_id, origin_question, interface_data, field_mapping = task

    print(origin_question)
    print(format_content_str(interface_data, level=0))

    try:
        await dbop.insert('task_question_content',
                          {'id': insert_id, 'batch_no': batch_no, 'origin_question': origin_question,
                           'interface_data': dbop.prepare_value(interface_data), 'status': 'running'
                           })

        desc = f'''请将以下 JSON 数据，生成自然语言段落列表,合理信息切片，每条切片用于后续文本嵌入（embedding）；
        其中相关字段映射为：{field_mapping}（诺含义或标签不明确，可以此辅助推断）；
         ”{origin_question}” 接口数据'''

        result = await ai_analyze(system_prompt, interface_data, client=emb_client, desc=desc,
                                  model="qwen3-32b", max_tokens=8192, top_p=0.85,
                                  extra_body={"enable_thinking": False})
        # await dbop.run("""
        #                  UPDATE task_question_content SET status='done',content=%s
        #                  WHERE id=%s
        #              """, (result_1, insert_id))
        #
        # desc_2 = f'请将下面针对“{origin_question}”生成的自然语言文章，进行合理信息切片，每条切片用于后续文本嵌入（embedding）'
        # result_2 = await ai_analyze(system_prompt_phase2, {"文章内容": result_1}, client=emb_client, desc=desc_2,
        #                             model="qwen2.5-32b-instruct", max_tokens=8192, top_p=0.85)

        parsed = extract_json_array(result)
        chunk_lists = [s.strip() for s in parsed if isinstance(s, str) and s.strip()] if isinstance(parsed,
                                                                                                    list) else []

        combined_text = dbop.prepare_value(chunk_lists) if chunk_lists else result
        await dbop.run("UPDATE task_question_content SET status='done', content=%s WHERE id=%s",
                       (combined_text, insert_id))
        if chunk_lists:
            embeddings = await ai_embeddings(chunk_lists, client=emb_client, model=MODEL_EMBEDDING, batch_size=25)
            payloads = [
                {'text': chunk, 'chunk_id': i, 'batch_no': batch_no, 'insert_id': insert_id, 'desc': origin_question}
                for i, chunk in enumerate(chunk_lists)]
            operation_id, ids = await upsert_points(payloads, vectors=embeddings, collection_name=Collection_Name,
                                                    client=qd_client, vector_name=None, hash_id=False)
            if operation_id:
                await dbop.run("""
                           UPDATE task_question_content SET status='completed',operation_id=%s
                           WHERE id=%s
                       """, (operation_id, insert_id))

                return True
        else:
            print(f"[Chunk Fail] {origin_question} => {result}")

    except Exception as e:
        print(f"[Consumer Error] {origin_question} => {str(e)}")
        await dbop.run("UPDATE task_question_content SET status='failed' WHERE id=%s", (insert_id,))

    return False


async def company_stock_deep_relation(company_name, httpx_client, host, api_key, limit: int = 5, timeout=30, **kwargs):
    """
    企业法人对外投资信息深度分析数据源获取实现：
    1. 获取法人信息；
    2. 查询法人担任法人公司的信息（companyLegal）；
    3. 提取前 N 个法人公司名称；
    4. 查询每个企业的工商基本信息；
    """
    saic_basic_info_conf = next((item for item in QUERY_TASKS if item['desc'] == '工商基本信息'), None)
    url_basic = saic_basic_info_conf['url_template'].format(host=host, api_key=api_key, keyword=company_name)
    data = await fetch_url(url_basic, httpx_client, timeout=timeout, **kwargs)
    raw_data = data.get('data', {})
    legal = raw_data.get('legalPerson') or raw_data.get('legalPersonName')
    legal_person = legal.strip() if legal else None
    if not legal_person:
        return {'error': '无法从工商基本信息中提取法人信息'}, []
    company_stock_relation_conf = next((item for item in QUERY_TASKS if item['desc'] == '人员对外投资信息'), None)
    url_relation = company_stock_relation_conf['url_template'].format(host=host, api_key=api_key, keyword=company_name,
                                                                      person=legal_person)
    data = await fetch_url(url_relation, httpx_client, timeout=timeout, **kwargs)
    company_legal = [item.get('name') for item in data.get('data', {}).get('companyLegal', []) if item.get('name')]
    company_names = list(dict.fromkeys(company_legal))[:limit]  # 去重 + 截取前N个
    print(f"\n[DEBUG] 完整请求URL: {url_relation}")
    if not company_names:
        return {'result': '未找到法人企业列表'}, []

    urls = [saic_basic_info_conf['url_template'].format(host=host, api_key=api_key, keyword=company) for company in
            company_names]
    results = await asyncio.gather(*(fetch_url(url, httpx_client, timeout=timeout, **kwargs) for url in urls))
    data['relation_company'] = results
    results_data = [extract_interface_map(d, saic_basic_info_conf.get("field_path", []),
                                          saic_basic_info_conf.get("field_mapping", {})) for d in results]
    return data, results_data


async def run_enterprise_analysis_task(batch_no: str, company_name: str, host: str, api_key: str, httpx_client) -> str:
    conn = await dbop.get_conn()
    await dbop.insert(
        'task_batch',
        {'batch_no': batch_no, 'title': f"企业分析任务 - {company_name}", 'status': 'running'}, conn)
    now = datetime.now()
    date_str = now.strftime('%Y年%m月%d日')
    analysis_prompt = SYS_PROMPT.get('analysis_prompt').format(company_name=company_name, date_str=date_str)
    default_mapping = {
        'status': '返回结果状态',
        'message': '返回结果消息',
        'data': '结果数据',
        'result': '结果数据',
        'total': '总记录条数',
        'items': '信息列表',
        'currentTime': '当前时间',
        'updateTime': '数据更新时间',
        'code': '消息码',
        'reason': '消息说明',
        'pageIndex': '当前页码',
    }
    queue = asyncio.Queue()

    # 异步执行所有子任务
    async def run_subtask(i, task_conf):
        question_no = f"Q{i + 1:03}"
        template = task_conf["url_template"]
        format_kwargs = {"host": host, "api_key": api_key, "keyword": company_name}
        if "{person}" in template:
            format_kwargs["person"] = 'legal_person'
        url = template.format(**format_kwargs)

        field_mapping = task_conf.get("field_mapping", {})
        # 让 AI 进行分析
        prompt = task_conf["question"]
        # 专家知识注入、细化研究范围、互补认知
        # user_knowledge = '\n当用户拥有明确的领域知识、特定的参考文献或个人见解时，补充具体技术方向，主动引导 AI 关注关键信息源，补充信息在多个维度上适应用户的偏好和要求，提供个性化的研究辅助。'
        # prompt += user_knowledge
        # http://www.capitallaw.cn/newsinfo/7676415.html

        origin_question = task_conf["desc"]

        insert_id = await dbop.run("""
                      INSERT INTO task_question (batch_no, question_no, origin_question, formatted_prompt, status)
                      VALUES (%s, %s, %s, %s, %s)
                  """, (batch_no, question_no, origin_question, prompt, "created"))

        # 发起 HTTP 请求
        try:
            print(f"\n[DEBUG] 完整请求URL: {url}")
            if task_conf.get('exec') == 'company_stock_deep_relation':
                interface_result, interface_data = await company_stock_deep_relation(company_name, httpx_client, host,
                                                                                     api_key, limit=5, timeout=30)
                if not interface_data:
                    interface_data = extract_interface_map(interface_result, task_conf.get("field_path", []),
                                                           field_mapping) or map_fields(interface_result,
                                                                                        default_mapping)
            else:
                interface_result = await fetch_url(url, httpx_client, max_retries=3, delay=3)
                interface_data = extract_interface_map(interface_result, task_conf.get("field_path", []),
                                                       field_mapping) or map_fields(interface_result, default_mapping)

            task = (batch_no, insert_id, origin_question, interface_data, field_mapping)
            await queue.put(task)
            #  await asyncio.create_task(process_embedding_task(task))
            print(f"已接收请求: {origin_question} | 队列大小: {queue.qsize()}")

            await dbop.run("""
                        UPDATE task_question SET status='running', interface_result=%s
                        WHERE batch_no=%s AND question_no=%s
                    """, (dbop.prepare_value(interface_result), batch_no, question_no))

            if field_mapping:
                prompt += f'\n相关字段映射为：{field_mapping}，（诺含义或标签不明确，可以此辅助推断）'
            result = await ai_analyze(analysis_prompt + '\n' + prompt, interface_data or interface_result, ai_client,
                                      origin_question, model='deepseek-chat')

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
    try:
        attention_origins = ['严重违法信息', '失信信息', '被执行信息', '空壳企业识别', '经营异常名录', '简易注销公告']
        summary_data, summary_text = await run_summary_result(batch_no, company_name, attention_origins,
                                                              'deepseek-reasoner', conn)

        field_map = {
            "core_strengths": "核心优势与稳定性",
            "major_risks": "重大风险与问题",
            "risk_table": "风险关联性分析",
            "conclusion_advice": "结论与建议",
            "score_table": "维度评分与总分",
            "account_limit_suggestion": "推荐非柜面开户额度区间"
        }
        summary_result = render_summary_text(field_map, summary_data) if summary_data else summary_text
        print(summary_result)
        # await run_question_summary(batch_no,company_name)
        await stop_worker(queue, worker_consumers_background)
        keyword_origin_matches = await run_summary_embedding(batch_no, summary_data, summary_text, company_name, conn)
        print(keyword_origin_matches)

    except Exception as e:
        print(e)
        summary_result = f"分析失败：{str(e)}"

    finally:
        dbop.release(conn)
        print(f'[enterprise_analysis]:{batch_no},{company_name}')

    return summary_result


async def run_enterprise_analysis_task_background(company_name: str, httpx_client):
    batch_no = uuid.uuid4().hex[:16]
    # 异步后台调度任务，不阻塞主流程
    task = asyncio.create_task(
        run_enterprise_analysis_task(batch_no, company_name, Ideatech_Host, Ideatech_API_Key, httpx_client))

    return batch_no, task


async def run_summary_result(batch_no, company_name: str, attention_origins: list[str],
                             model: str = 'deepseek-reasoner', conn=None):
    insert_id = await dbop.insert('task_summary_question', {'batch_no': batch_no, 'status': 'created'}, conn)
    rows = await dbop.run("SELECT origin_question,result FROM task_question WHERE batch_no=%s",
                          (batch_no,), conn)
    all_results = [(row["origin_question"], row["result"]) for row in rows]
    joined_questions = "、".join(row["origin_question"] for row in rows if row["origin_question"] in attention_origins)
    now = datetime.now()
    date_str = now.strftime('%Y年%m月%d日')
    # 汇总总结
    # summary_prompt = f"请根据以下关于企业（{company_name}）{joined_questions}的多项分析，做一个整体风险与运营状况总结。"
    summary_prompt = SYS_PROMPT.get('summary_prompt').format(company_name=company_name,
                                                             joined_questions=joined_questions, date_str=date_str)

    await dbop.run("""
                UPDATE task_summary_question SET summary_question=%s,status='running'
                WHERE id=%s
            """, (summary_prompt, insert_id), conn)
    try:
        summary_result = await ai_analyze(summary_prompt, all_results, ai_client, desc='企业多维数据各项分析',
                                          model=model, max_tokens=8192)
        parsed = extract_json_struct(summary_result)
        summary_data = {k: d for k, d in parsed.items() if
                        isinstance(d, (str, dict, list)) and str(d).strip()} if isinstance(parsed, dict) else {}

        summary_text = dbop.prepare_value(summary_data) if summary_data else (
                extract_json_str(summary_result) or summary_result)

        await dbop.execute([
            ("""
                UPDATE task_summary_question 
                SET summary_answer=%s, model=%s, status='completed'
                WHERE id=%s
                """,
             (summary_text, model, insert_id)
             ),
            ("""
                UPDATE task_batch SET status='completed', completed_at=NOW()
                WHERE batch_no=%s
                """,
             (batch_no,)
             )
        ], conn)

        return summary_data, summary_text

    except Exception as e:
        print(e)
        summary_result = f"分析失败：{str(e)}"
        await dbop.execute([
            ("""
            UPDATE task_summary_question 
            SET summary_answer=%s, model=%s, status='failed'
            WHERE id=%s
            """,
             (summary_result, model, insert_id)
             ),
            ("""
            UPDATE task_batch SET status='failed'
            WHERE batch_no=%s
            """,
             (batch_no,)
             )
        ], conn)

    return {}, ''


async def run_summary_embedding(batch_no, summary_data: dict, summary_text: str, company_name, conn=None):
    chunk_prompt = SYS_PROMPT.get('chunk_prompt_1')
    reranker_prompt = SYS_PROMPT.get('reranker_prompt')
    succeed = False
    try:
        await dbop.insert('task_summary_chunks',
                          {'batch_no': batch_no, 'company_name': company_name, 'chunk_type': '正文',
                           'status': 'running'}, conn)
        # 以下是系统根据公开数据与内部资料提取出的关键字段摘要
        fields = ["core_strengths", "major_risks", "risk_table", "conclusion_advice"]
        summary_data_filter = {k: v for k, v in summary_data.items() if k in fields}
        now = datetime.now()
        date_str = now.strftime('%Y年%m月%d日')
        desc = f'今天是{date_str}，请将下面关于企业【{company_name}】的全维度分析结果生成的自然语言文章，进行合理信息切片、关键信息提取、相似语句生成，用于后续与基础数据库进行比对与反查'
        chunk_result = await ai_analyze(chunk_prompt, {"总结报告文章内容": summary_data_filter or summary_text},
                                        client=ai_client, desc=desc, model='deepseek-chat', max_tokens=8192, top_p=0.85)
        parsed = extract_json_array(chunk_result)
        keywords_list: list[dict] = [d.replace("\n", " ").strip() if
                                     isinstance(d, str) else d for d in parsed] if isinstance(parsed, list) else []
        combined_text = dbop.prepare_value(keywords_list) if keywords_list else chunk_result
        await dbop.run(
            "UPDATE task_summary_chunks SET status=%s,summary_chunk=%s,model=%s WHERE batch_no=%s",
            ('ready', combined_text, "deepseek-chat", batch_no), conn)

        # keywords: 用户实际关注的原始事实片段，是核心语义锚点，用于前端显示高亮；
        # query: 围绕该keyword衍生出的多个语义等价表达，用于embedding检索（召回用）；
        summary_chunks_list = [line for d in keywords_list for line in d['query'] if isinstance(d['query'], list)]
        print(f'summary_chunks_list:{len(summary_chunks_list)},keywords_list:{len(keywords_list)}')
        if not summary_chunks_list:
            print(f"[Chunk Fail] 整体风险与运营状况总结 =>{chunk_result}\n{keywords_list}")
            return []

        embeddings = await ai_embeddings(summary_chunks_list, client=emb_client, model=MODEL_EMBEDDING,
                                         batch_size=25)
        payloads = [{'text': chunk, 'chunk_id': i, 'batch_no': batch_no, 'desc': '整体风险与运营状况总结'}
                    for i, chunk in enumerate(summary_chunks_list)]
        operation_id, ids = await upsert_points(payloads, vectors=embeddings, collection_name=Collection_Name,
                                                client=qd_client, vector_name=None, hash_id=True)
        summary_vector_ids = list(zip(ids, summary_chunks_list))
        if operation_id:
            await dbop.run("""
                       UPDATE task_summary_chunks SET status='completed',vector_ids=%s,operation_id=%s
                       WHERE batch_no=%s
                   """, (dbop.prepare_value(summary_vector_ids), operation_id, batch_no), conn)
            succeed = True

        inserted_ids = [await dbop.run("""
            INSERT INTO keyword_summary_origin (batch_no, `keywords`,`topic`,status) VALUES (%s, %s, %s, %s)
        """, (batch_no, d.get('keywords'), d.get('topic'), "created"), conn) for d in keywords_list]

        question_content_rows, question_analysis_rows = await dbop.query([
            ("SELECT origin_question,content FROM task_question_content WHERE batch_no=%s AND status=%s",
             (batch_no, 'completed')),
            ("SELECT origin_question,result FROM task_question WHERE batch_no=%s AND status=%s",
             (batch_no, 'completed'))
        ])

        sentence = [(r["origin_question"], r.get("content", '').split('\n\n')) for r in question_content_rows]
        segments = [(r["origin_question"], split_summary_chunks(r.get("result", ''))) for r in question_analysis_rows]
        corpus = [line.replace("\n", " ").strip() for chunks in sentence for line in chunks[1]]
        corpus_analyze = [line for chunks in segments for line in chunks[1]]
        bm25 = BM25(corpus)
        print(f'segments:{len(segments)},corpus:{len(corpus)},corpus_analyze:{len(corpus_analyze)}')

        keyword_origin_matches: list[dict] = []
        adjacency_list: dict[int, list[int]] = defaultdict(list)  # kw_id->[query_id...]
        summary_origins_map: dict[int, set[str]] = defaultdict(set)  # defaultdict(<class 'set'>
        for i, d in enumerate(keywords_list):
            keywords = d.get('keywords')
            adjacency_list[i] = [summary_chunks_list.index(q) for q in d.get('query', [])]
            # re.search(r'\b' + re.escape(keywords) + r'\b', context):精确正则匹配（\b词边界）
            matches = [(text, len(keywords) / len(text), i) for i, text in enumerate(corpus_analyze)
                       if keywords in text]  # filter like 精确子串匹配,keywords 与分析的语料里面更接近
            if not matches:  # 模糊近似匹配，无语义理解，长文本比短句长很多时，编辑距离天然就会大，无法公平对比
                matches = find_best_matches(keywords, corpus_analyze, top_n=2, cutoff=0.5, best=True)
            analyze_matches = [(text, round(score, 3), get_origin_fild(idx, segments))
                               for text, score, idx in matches]
            # 基于关键词覆盖的检索，适合初筛
            scores = bm25.rank_documents(keywords, top_k=5, normalize=True)
            origin_matches = [(corpus[idx], round(score, 3), get_origin_fild(idx, sentence))
                              for idx, score in scores]

            origin_name = {match[2] for match in origin_matches} | {match[2] for match in analyze_matches}
            for ctx_id in adjacency_list[i]:
                summary_origins_map[ctx_id] |= origin_name  # update

            keyword_origin_matches.append({**d, 'analyze_matches': analyze_matches,
                                           'origin_matches': origin_matches, 'dense_matches': []})

        batch_match = field_match(field_key='batch_no', match_values=batch_no)

        async def recommend_context(i, t):
            vec_id, query = t
            origins = summary_origins_map.get(i, [])
            origins_match = field_match(field_key='desc', match_values=origins)

            recommend_matches = await recommend_by_id([vec_id], Collection_Name, qd_client,
                                                      payload_key=['text', 'desc'],  # ['text', 'desc', 'chunk_id']
                                                      match=batch_match + origins_match, not_match=[], not_ids=ids,
                                                      topn=15, score_threshold=0.0, get_dict=False)

            recommend_matches = [(payload.get('text'), score, payload.get('desc')) for payload, score, _id in
                                 recommend_matches]

            return {'query': query, 'origins': origins, 'recommend_matches': recommend_matches}

        query_recommend: list[dict] = await asyncio.gather(
            *[recommend_context(ctx_id, t) for ctx_id, t in enumerate(summary_vector_ids)], return_exceptions=True)

        for kw_idx, query_idxs in adjacency_list.items():
            combined = {
                'query': [],
                'origins': set(),
                'recommend_matches': []
            }
            for ctx_id in query_idxs:
                entry = query_recommend[ctx_id]
                combined['query'].append(entry['query'])
                combined['origins'].update(entry.get('origins', []))
                combined['recommend_matches'].extend(entry.get('recommend_matches', []))

            # 汇总推荐匹配项,推荐结果按得分降序排列，保留 top 30,max: 15*n(q)
            combined['recommend_matches'] = sorted(combined['recommend_matches'], key=lambda x: x[1], reverse=True)[:30]
            combined['origins'] = list(combined['origins'])
            keyword_origin_matches[kw_idx]['dense_matches'] = combined  # [chunks_dense[j] for j in chunk_idxs]

        params_list = [(dbop.prepare_value(d['analyze_matches']),
                        dbop.prepare_value(d['origin_matches']),
                        dbop.prepare_value(d['dense_matches']), i) for i, d in
                       zip(inserted_ids, keyword_origin_matches) if i]

        await dbop.run("""
                  UPDATE keyword_summary_origin SET analyze_matches=%s, origin_matches=%s, dense_matches=%s, status="ready" 
                  WHERE id=%s
              """, params_list, conn)

        async def llm_origin(insert_id, d, sem):
            data = {'dense_matches': d['dense_matches'].get('recommend_matches', []),
                    'origin_matches': d['origin_matches']}

            desc = f"请根据以上规则处理，并返回一个 JSON 数组（保留相关结果并重排）。keywords:{d.get('keywords')},query:{d.get('query', [])}\n 参考如下 search_hit"
            async with sem:
                reranker_result = await ai_analyze(reranker_prompt, data, emb_client, desc, model="qwen3-32b",
                                                   max_tokens=8192, top_p=0.85, extra_body={"enable_thinking": False})
                parsed = extract_json_array(reranker_result)
                reranker_list = [d.replace("\n", " ").strip() if
                                 isinstance(d, str) else d for d in parsed] if isinstance(parsed, list) else []
                if reranker_list and insert_id:  # 分别 completed
                    await dbop.run("""
                        UPDATE keyword_summary_origin SET reranker_matches=%s, status="completed" 
                        WHERE id=%s
                         """, (dbop.prepare_value(reranker_list), insert_id))

                return {'reranker_matches': reranker_list or reranker_result}

        semaphore = asyncio.Semaphore(50)
        tasks = [llm_origin(i, d, semaphore) for i, d in zip(inserted_ids, keyword_origin_matches)]
        llm_reranker_origin: list[dict] = await asyncio.gather(*tasks, return_exceptions=True)

        for i, d in enumerate(llm_reranker_origin):
            keyword_origin_matches[i].update(d)

        return keyword_origin_matches

    except Exception as e:
        print(f"[Chunk Error] => {str(e)}")
        if not succeed:
            await dbop.run("UPDATE task_summary_chunks SET status='failed' WHERE batch_no=%s", (batch_no,), conn)
        return []


def split_summary_chunks(text: str) -> list[str]:
    """
     清洗大模型输出的文本，分成自然段 + 清除 bullet/markdown/符号前缀
    """

    def clean_lines(line: str) -> str:
        return re.sub(r"^[-–•\d\)\.\s]+", "", line).strip()  # 清除 bullet/数字/多余符号

    normalized = re.sub(r'\n{2,}', '\n\n', text.strip())
    return [clean_lines(chunk.replace("\n", " ")) for chunk in normalized.split("\n\n") if chunk.strip()]


def format_content_str(content, level=0, exclude_null=True) -> str:
    def format_table(data: list[dict]) -> str:
        headers = list(data[0].keys())
        lines = ["| " + " | ".join(headers) + " |",  # header_line
                 "| " + " | ".join(["---"] * len(headers)) + " |"]  # divider_line
        for row in data:  # rows
            lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
        return "\n".join(lines)

    if exclude_null and content is None:
        return ""  # "*（空内容）*"

    if isinstance(content, (tuple, list)):
        if exclude_null and not content:
            return ""  # "*（空列表）*"
        if all(isinstance(x, dict) for x in content):
            heading_prefix = "\n" if level > 0 else ''
            return heading_prefix + format_table(content)
        elif all(isinstance(x, str) for x in content):
            return "\n".join(content)
        else:
            return json.dumps(content, ensure_ascii=False, indent=2)
    if isinstance(content, dict):
        if exclude_null and not content:
            return ""
        if all(isinstance(x, str) for x in content.keys()):
            heading_prefix = ("\n" + " " * level) if level > 0 else ''  # "#" * (level + 2)
            lines = [f"{heading_prefix}**{key}**:{format_content_str(val, level + 1)}" for key, val in content.items()
                     if not exclude_null or val]
            return "\n".join(lines)
        else:
            return json.dumps(content, ensure_ascii=False, indent=2)
    if isinstance(content, (int, float, bool)):
        return str(content)
    return str(content).strip()


def render_summary_text(field_map: dict, summary_data: dict) -> str:
    """
    将结构化 summary_data 转为分节展示文本（markdown 风格）
    """
    num_iter = iter("一二三四五六七八九十")
    sections = []

    def split_numbered_items(text: str) -> str:
        """
        将段落中编号项（1）2）...）前插入换行，避免破坏案号等结构。
        匹配规则：在中文句子中，若出现编号“1）”、“2）”等开头，强制断行。
        """
        pattern = r'(?<![\d\(])(\d{1,2}）)'  # 前缀不能是数字或 (
        return re.sub(pattern, r'\n\1', text)

    def remove_markdown_block(text: str) -> str:
        """
        如果文本以 ```markdown 开头并以 ``` 结尾，则移除这两个标记，返回中间内容。
        否则返回原始文本。
        """
        match = re.match(r"^```markdown\s*\n(.*?)\n?```$", text.strip(), re.DOTALL)
        if match:
            return match.group(1).strip()
        return text

    if not field_map:
        field_map = {k: k for k in summary_data}
    for i, (key, title) in enumerate(field_map.items()):
        content = summary_data.get(key)
        if not content:
            continue
        num = next(num_iter, str(i + 1))
        content_str = remove_markdown_block(format_content_str(content))  # 自动转为字符串（支持字典或表格结构）
        # 数字项强制断行split_numbered_items(str(content)).strip()
        sections.append(f"#### {num}、{title}\n\n{content_str}\n")  # f"**{num}.{title}**\n\n{content_str}\n\n"

    return '\n\n---\n\n'.join(sections)


if __name__ == "__main__":
    async def test():
        await dbop.execute([
            ("""
        CREATE TABLE `task_question_content` (
          `id` bigint NOT NULL COMMENT '与 task_question 的 id 相同',
          `batch_no` varchar(64) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '外键，关联 task_batch，用于辅助查询',
          `interface_data` LONGTEXT DEFAULT NULL COMMENT '接口返回数据初步映射转换',
          `origin_question` text COLLATE utf8mb4_unicode_ci COMMENT '原始问题描述',
          `content` longtext COLLATE utf8mb4_unicode_ci COMMENT 'LLM 给出的最终回答',
          `status` varchar(32) COLLATE utf8mb4_unicode_ci DEFAULT 'pending' COMMENT '状态',
          `operation_id` int DEFAULT NULL COMMENT '向量数据库操作记录 ID',
          `created_at` datetime DEFAULT CURRENT_TIMESTAMP,
          `updated_at` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
          PRIMARY KEY (`id`),
          KEY `batch_no` (`batch_no`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""", ()),
            ("""
        CREATE TABLE `due_diligence_questions` (
            `id` BIGINT AUTO_INCREMENT PRIMARY KEY,
            `batch_no` VARCHAR(64) NOT NULL,
            `question_text` LONGTEXT DEFAULT NULL COMMENT '生成的问题清单，格式为JSON数组',
            `question_type` VARCHAR(32) DEFAULT '尽调提问' COMMENT '可区分场景，如风险对抗/尽调等',
            `question_id` VARCHAR(64) DEFAULT NULL,
            `status` VARCHAR(32) DEFAULT 'pending',
            `retry_count` INT DEFAULT 0,
            `model` VARCHAR(64) DEFAULT NULL COMMENT '生成来源模型',
            `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP,
            `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            INDEX (`batch_no`)
            );""", ()),
            ("""
        CREATE TABLE `task_summary_chunks` (
            `id` BIGINT AUTO_INCREMENT PRIMARY KEY,
            `batch_no` VARCHAR(64) NOT NULL,
            `company_name` VARCHAR(255) NOT NULL COMMENT '公司名称',
            `summary_chunk` LONGTEXT DEFAULT NULL COMMENT '提取出的自然语言结构化句子',
            `chunk_type` VARCHAR(32) DEFAULT 'default' COMMENT '可用于标注类型如正文/结尾/表格前内容',
            `status` VARCHAR(32) DEFAULT 'pending',
            `model` VARCHAR(64) DEFAULT NULL COMMENT '生成来源模型',
            `operation_id` INTEGER DEFAULT NULL COMMENT '向量数据库操作记录 ID',
            `vector_ids` JSON  DEFAULT NULL  COMMENT '向量id',
            `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP,
            `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            INDEX (`batch_no`)
            );""", ()),
            ("""
        CREATE TABLE due_diligence_answer (
            id INT PRIMARY KEY AUTO_INCREMENT COMMENT '主键',
            `batch_no` VARCHAR(64) NOT NULL,
            question_id VARCHAR(64) NOT NULL COMMENT '原始问题 ID',
            seq INT NOT NULL COMMENT '问题在任务列表中的序号',
            company_name VARCHAR(255) NOT NULL COMMENT '公司名称',
            question TEXT NOT NULL COMMENT '问题内容',
            answer LONGTEXT COMMENT 'AI 回答内容',
            role VARCHAR(32) DEFAULT NULL,
            action TEXT DEFAULT NULL COMMENT '可引导行动',
            topic VARCHAR(100) DEFAULT NULL COMMENT '主题/风险维度',
            evaluate LONGTEXT COMMENT 'AI 评价',
            status VARCHAR(32) DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
            `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        ) COMMENT='尽调问题 AI 回答表';
            """, ()),
            ("""
        CREATE TABLE `keyword_summary_origin` (
          `id` INT PRIMARY KEY AUTO_INCREMENT COMMENT '主键',
          `batch_no` VARCHAR(64) NOT NULL COMMENT '批次号',
          `keywords` TEXT NOT NULL COMMENT '关键词',
          `topic` VARCHAR(64)    COMMENT  '摘要主题',
          `analyze_matches` JSON     COMMENT '分析来源匹配结果',
          `origin_matches`  JSON     COMMENT '来源匹配结果',
          `dense_matches`   JSON     COMMENT '密集匹配结果',
          `reranker_matches`   JSON     COMMENT 'AI 结果',
          `status` VARCHAR(32) DEFAULT 'pending' COMMENT '状态',
          `created_at` TIMESTAMP    DEFAULT CURRENT_TIMESTAMP            COMMENT '创建时间',
          `updated_at` DATETIME     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间'
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """, ()),
        ])
