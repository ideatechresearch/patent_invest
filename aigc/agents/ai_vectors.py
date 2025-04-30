from qdrant_client import QdrantClient, AsyncQdrantClient, grpc
from qdrant_client.client_base import QdrantBase
import asyncio
# from qdrant_client.qdrant_remote import QdrantRemote
# from qdrant_client.models import Filter, FieldCondition, IsEmptyCondition, HasIdCondition, MatchValue,PointStruct,DiscoverQuery,ContextQuery,NearestQuery,RecommendQuery
import qdrant_client.models as qcm
from typing import Callable, Any


def empty_match(field_key):
    # 检查字段是否为空
    if not field_key:
        return []
    return [qcm.IsEmptyCondition(is_empty=qcm.PayloadField(key=field_key), )]


def null_match(field_key):
    # 检查字段值是否为 null
    if not field_key:
        return []
    return [qcm.IsNullCondition(is_null=qcm.PayloadField(key=field_key), )]


def field_match(field_key, match_values):
    """匹配字段值，支持单值和多值匹配"""
    if not field_key or not match_values:
        return []
    if isinstance(match_values, (str, int, float)):
        return [qcm.FieldCondition(key=field_key, match=qcm.MatchValue(value=match_values, ), )]
    if isinstance(match_values, (list, tuple, set)):
        return [qcm.FieldCondition(key=field_key, match=qcm.MatchAny(any=list(match_values)))]
    return [qcm.FieldCondition(key=field_key, match=qcm.MatchValue(value=v), ) for v in match_values]


def field_range(field_key, **kwargs):
    """创建字段的范围查询条件，支持 lt, lte, gt, gte"""
    range_params = {}
    for op in ["lt", "lte", "gt", "gte"]:
        if op in kwargs:
            range_params[op] = kwargs[op]

    if not field_key or not range_params:
        return []

    if isinstance(next(iter(range_params.values())), (float, int)):
        return [qcm.FieldCondition(key=field_key, range=qcm.Range(**range_params))]
    return [qcm.FieldCondition(key=field_key, range=qcm.DatetimeRange(**range_params))]  # 日期范围


def geo_match(field_key="geo", lat: float = 40.7128, lon: float = -74.0060, radius=10000):
    """Geo Radius 限制半径内搜索,半径 10km"""
    return [qcm.FieldCondition(key=field_key,
                               geo_radius=qcm.GeoRadius(center=qcm.GeoPoint(lat=lat, lon=-lon),
                                                        radius=radius, ),
                               )]
    # "geo": {"lat": 34.0522, "lon": -118.2437},
    # geo_bounding_box=models.GeoBoundingBox(
    #                    top_left=models.GeoPoint(lat=41, lon=-75),
    #                    bottom_right=models.GeoPoint(lat=40, lon=-73),
    #                ),


async def ids_to_names(ids, collection_name, client, key_name='word'):
    id_record = await client.retrieve(collection_name=collection_name, ids=ids, with_payload=[key_name])
    return {p.id: p.payload[key_name] for p in id_record}


async def names_to_ids(names: list, collection_name, client, match=[], key_name='word'):
    shoulds = [qcm.FieldCondition(key=key_name, match=qcm.MatchValue(value=w)) for w in names]
    scroll_filter = qcm.Filter(must=match, should=shoulds) if match else qcm.Filter(
        should=shoulds)  # must_not,min_should,
    scroll_result, next_offset = await client.scroll(collection_name=collection_name,
                                                     scroll_filter=scroll_filter,
                                                     with_payload=[key_name],
                                                     # with_vectors=True,
                                                     limit=len(names))

    return {i.payload[key_name]: i.id for i in scroll_result}
    # [(i.payload[key_name], i.id) for i in scroll_result[0]]


async def create_collection_aliases(collection_name, alias_name, client):
    await client.update_collection_aliases(
        change_aliases_operations=[
            qcm.CreateAliasOperation(
                create_alias=qcm.CreateAlias(collection_name=collection_name, alias_name=alias_name)
            )
        ]
    )
    return await client.get_collection_aliases(collection_name)  # client.get_aliases()


async def delete_collection(collection_name, client):
    # Drops the specified collection and all associated data in it.
    if await client.collection_exists(collection_name=collection_name):
        print(f"Collection {collection_name} exists. Deleting and recreating...")
        return await client.delete_collection(collection_name=collection_name)
    return True


async def create_collection(collection_name, client, size, alias_name: str = None, vector_name: str = None,
                            vectors_on_disk=True, payload_on_disk=True, hnsw_on_disk=True, recreate=False):
    """
        创建 Qdrant 集合。如果 `new` 为 True 且集合已经存在，则删除现有集合并重新创建。

        :param collection_name: 集合名称
        :param client: Qdrant 客户端实例
        :param size: 向量的维度大小
        :param alias_name:别名
        :param vector_name
        :param vectors_on_disk: 向量是否保存在磁盘上
        :param hnsw_on_disk: HNSW 索引是否保存在磁盘上
        :param payload_on_disk
        :param recreate
        :return: 集合中的点数
        """
    # https://api.qdrant.tech/api-reference/collections/create-collection
    try:
        if recreate:
            await delete_collection(collection_name, client)
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
            if alias_name:
                await create_collection_aliases(collection_name, alias_name, client)
        # 返回集合中的点数
        res = await client.get_collection(collection_name=collection_name)
        return res.model_dump()

    except Exception as e:
        print(f"Error while creating or managing the collection {collection_name}: {e}")
        return None
    # vectors_config = {
    #     "image": qcm.VectorParams(
    #         size=20,
    #         distance=qcm.Distance.DOT,
    #         hnsw_config=qcm.HnswConfigDiff(
    #             m=9,
    #             ef_construct=99,
    #             full_scan_threshold=42,
    #             max_indexing_threads=4,
    #             on_disk=True,
    #             payload_m=5,
    #         ),
    #         quantization_config=qcm.ScalarQuantization(
    #             scalar=qcm.ScalarQuantizationConfig(
    #                 type=qcm.ScalarType.INT8, quantile=0.69, always_ram=False
    #             )
    #         ),
    #         on_disk=True,
    #     ),
    # }


async def export_all_points(collection_name, client, batch=1000):
    all_points = []
    scroll_offset = None

    while True:
        result, scroll_offset = await client.scroll(
            collection_name=collection_name,
            limit=batch,
            with_vectors=True,
            with_payload=True,
            offset=scroll_offset
        )
        all_points.extend(result)
        if scroll_offset is None:
            break

    return all_points


# async def update_collection(collection_name, vector_name, client):
#     vector_params = qcm.VectorParamsDiff(hnsw_config=qcm.HnswConfigDiff(on_disk=True), on_disk=True)
#     vectors_config = {vector_name if vector_name else '': vector_params}
#     return await client.update_collection(
#         vectors_config=vectors_config,
#         collection_name=collection_name,
#     )


async def create_payload_index(collection_name, client, field_name, field_schema='keyword'):
    # 适用于加速基于 filter 的查询
    res = await client.create_payload_index(collection_name=collection_name,
                                            field_name=field_name,  # 需要索引的字段
                                            field_schema=field_schema,  # 分类、数值筛选、时间查询、全文搜索
                                            # keyword、integer、float、datetime、text
                                            )
    return res.model_dump()


async def upsert_points(payloads: list[dict], vectors: list, collection_name: str, client, vector_name: str = None):
    """
    插入单个点到 Qdrant 集合。

    :param payloads: 数据负载
    :param vectors: 向量
    :param collection_name: 集合名称
    :param client: Qdrant 客户端实例
    :param vector_name
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


async def upsert_points_batch(payloads: list[dict], vectors: list, collection_name: str, client,
                              vector_name: str = None, batch_size: int = 1000):
    """
    批量插入数据到 Qdrant 集合。

    :param payloads: 数据负载列表
    :param vectors: 向量列表
    :param collection_name: 集合名称
    :param client: Qdrant 客户端实例
    :param vector_name:
    :param batch_size: 每次插入的批次大小
    """
    assert len(payloads) == len(vectors), "Payloads and vectors must have the same length."
    current_count = await client.count(collection_name=collection_name)
    index = current_count.count
    for i in range(0, len(payloads), batch_size):
        batch_payloads = payloads[i:i + batch_size]
        batch_vectors = vectors[i:i + batch_size]
        points_size = len(batch_payloads)
        batch_ids = list(range(index, index + points_size))

        # 创建 Batch 对象
        points = qcm.Batch(ids=batch_ids, payloads=batch_payloads,
                           vectors={vector_name: batch_vectors} if vector_name else batch_vectors)

        try:
            print(f"Processing batch {i // batch_size + 1} with size {points_size}...")
            operation_info = await client.upsert(collection_name=collection_name, points=points)
            assert operation_info.status == qcm.UpdateStatus.COMPLETED, 'NOT COMPLETED'
            index += points_size
        except StopIteration:
            break
        except Exception as e:
            print(e)
            print(f"Error upserting batch {i // batch_size + 1}: {e}")
            continue
    return index


async def update_points_batch(ids: list, vectors: list, collection_name: str, client, vector_name: str = None,
                              payload: dict = None):
    operations = []
    if not ids:  # upsert
        current_count = await client.count(collection_name=collection_name)
        ids = list(range(current_count.count, current_count.count + len(vectors)))
        # new_points = [qcm.PointStruct(id=_id, vector=vector) for _id, vector in zip(ids,vectors)]# payload={},
        # upsert_operation = qcm.UpsertOperation(upsert=qcm.PointsList(points=new_points))
        batch = qcm.Batch(ids=ids, vectors={vector_name: vectors} if vector_name else vectors)
        upsert_operation = qcm.UpsertOperation(upsert=qcm.PointsBatch(batch=batch))
        operations.append(upsert_operation)
    else:  # update
        if vectors:
            update_points = [qcm.PointVectors(id=_id, vector=vector) for _id, vector in zip(ids, vectors)]
            update_vectors_operation = qcm.UpdateVectorsOperation(
                update_vectors=qcm.UpdateVectors(points=update_points))
            operations.append(update_vectors_operation)
        if payload:  # for specified points
            set_payload_operation = qcm.SetPayloadOperation(
                set_payload=qcm.SetPayload(payload=payload, points=ids, ))
            operations.append(set_payload_operation)
    return await client.batch_update_points(collection_name, operations)


async def update_vectors(vec_ids: list[tuple[str, list[int]]] | list[int],
                         vecs_list: list[list[list[float]]] | list[list[float]], client,
                         collection_name: str, vector_name: str = None):
    points_all = []  # Points with named vectors
    if isinstance(vec_ids[0], tuple):
        assert len(vecs_list) == len(vec_ids)

        for (key, ids), vecs in zip(vec_ids, vecs_list):
            points = [qcm.PointVectors(id=_id, vector={key: vec}, ) for _id, vec in zip(ids, vecs)]
            points_all.extend(points)
    else:
        points_all = [qcm.PointVectors(id=_id, vector={vector_name: vec} if vector_name else vec) for _id, vec in
                      zip(vec_ids, vecs_list)]

    return await client.update_vectors(points_all, collection_name)


# async def inference(q,collection_name: str, dense_model:str,client, vector_name: str ="text"):
#     inference_object_dense_doc_1 = qcm.InferenceObject(
#         object="hello world",
#         model=dense_model,
#         options={"lazy_load": True},
#     )
#
#     client.query_points(collection_name, inference_object_dense_doc_1, using=vector_name)


async def payload_facet(collection_name, client, field_key, limit=10, match=[], exact=False):
    # 统计某个 payload 字段的分布情况,主要用于 分类统计
    facet_filter = qcm.Filter(must=match)
    facets = await client.facet(collection_name=collection_name,
                                key=field_key,
                                limit=limit,
                                facet_filter=facet_filter,
                                exact=exact,  # 精确计数
                                )
    return facets


async def get_payload(names, collection_name, client, ids=[], fields=[], match=[], key_name='word'):
    if not ids:
        result = await names_to_ids(names, collection_name, client, match, key_name)
        ids = [result[n] for n in names if n in result]
    id_record = await client.retrieve(collection_name=collection_name, ids=ids,
                                      with_payload=fields if len(fields) else True)
    return [(p.id, p.payload) for p in id_record]  # [(id,{payload}),]


async def get_vecs(names, collection_name, client, ids=[], match=[], key_name='word'):
    if not ids:
        result = await names_to_ids(names, collection_name, client, match, key_name)
        ids = [result[n] for n in names if n in result]
    id_record = await client.retrieve(collection_name=collection_name, ids=ids, with_vectors=True,
                                      with_payload=[key_name] if key_name else False)
    return {p.payload[key_name] if key_name else p.id: p.vector for p in id_record}


async def get_vec(name, collection_name, client, _id=-1, match=[], key_name='word'):
    if _id > 0:
        point_result = await client.retrieve(collection_name=collection_name, ids=[_id], with_vectors=True)
    else:
        match_name = qcm.FieldCondition(key=key_name, match=qcm.MatchValue(value=name, ), )
        scroll_filter = qcm.Filter(must=match + [match_name])
        point_result, next_offset = await client.scroll(collection_name=collection_name,
                                                        scroll_filter=scroll_filter,
                                                        with_vectors=True,
                                                        # with_payload=[key_name],
                                                        limit=1,  # 只获取一个匹配的结果
                                                        # order_by=OrderBy(key='df',direction='desc')
                                                        )  # (points,next_page_offset)
        if not point_result:
            return []

    return point_result[0].vector


# Recommend
async def recommend_by_id(ids, collection_name, client, key_name='word', match=[], not_match=[], not_ids=[], topn=10,
                          score_threshold: float = 0.0, ret_name=False):
    not_match_ids = [qcm.HasIdCondition(has_id=not_ids)]
    query_filter = qcm.Filter(must=match, must_not=not_match_ids + not_match)

    hit = await client.recommend(collection_name=collection_name,
                                 positive=ids,  # [ID]
                                 query_filter=query_filter,
                                 limit=topn, score_threshold=score_threshold,
                                 with_payload=[key_name] if (ret_name and key_name) else False,
                                 )  # ScoredPoint

    return [(p.payload[key_name] if (ret_name and key_name) else p.id, p.score) for p in hit]


async def recommend_by_id_group(ids, collection_name, client, group_key="document_id", match=[], not_match=[],
                                not_ids=[], topn=10, group_size=3,
                                score_threshold: float = 0.0):
    not_match_ids = [qcm.HasIdCondition(has_id=not_ids)]
    query_filter = qcm.Filter(must=match, must_not=not_match_ids + not_match)

    hit = await client.recommend_groups(collection_name=collection_name,
                                        positive=ids,  # [ID]
                                        # negative=[718, [0.2, 0.3, 0.4, 0.5]],
                                        query_filter=query_filter,
                                        group_by=group_key,
                                        limit=topn,
                                        group_size=group_size,
                                        score_threshold=score_threshold,
                                        with_payload=[group_key] if group_key else False,
                                        )  # ScoredPoint

    return [(p.payload[group_key] if group_key else p.id, p.score) for p in hit]


async def recommend_by_ids(ids: list | tuple, collection_name: str, client, payload_key='word', match=[], not_match=[],
                           topn=10, score_threshold: float = 0.0):
    not_match_ids = [qcm.HasIdCondition(has_id=ids)]
    query_filter = qcm.Filter(must=match, must_not=not_match_ids + not_match)  # 缩小查询范围

    recommend_queries = [
        qcm.RecommendRequest(positive=[_id], filter=query_filter, limit=topn, score_threshold=score_threshold,
                             with_payload=[payload_key] if payload_key else True)
        for _id in ids]

    recommend_hit = await client.recommend_batch(collection_name=collection_name,
                                                 requests=recommend_queries)  # ScoredPoint

    return [(_id, [(p.payload[payload_key] if payload_key else p.payload, p.id, p.score) for p in hit]) for _id, hit
            in zip(ids, recommend_hit)]


# Search
async def search_by_id(name, collection_name, client, _id=-1, key_name='word', match=[], exclude=[], not_ids=[],
                       topn=10, score_threshold: float = 0.0, exact=False):
    query_vector = await get_vec(name, collection_name, client, _id=_id, match=match, key_name=key_name)
    if not query_vector:
        return []

    not_match_ids = [qcm.HasIdCondition(has_id=not_ids)]
    not_match_name = [qcm.FieldCondition(key=key_name, match=qcm.MatchValue(value=v), ) for v in
                      exclude] if key_name else []
    query_filter = qcm.Filter(must=match, must_not=not_match_ids + not_match_name)

    hit = await client.search(collection_name=collection_name,
                              query_vector=query_vector,  # neighbor_vector then result_vector tolist()
                              query_filter=query_filter,
                              with_payload=True,
                              limit=topn,  # max_neighbors
                              score_threshold=score_threshold,
                              offset=0,
                              search_params=qcm.SearchParams(exact=exact,  # Turns on the exact search mode
                                                             # quantization=QuantizationSearchParams(rescore=True),
                                                             ),
                              )  # ScoredPoint

    return [(p.payload[key_name], p.score) for p in hit]


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

    return await client.search_batch(collection_name=collection_name, requests=search_queries)  # ScoredPoint


# SimilarByIds
async def search_by_ids(ids, collection_name, client, payload_key='word', match=[], not_match=[], not_ids=[], topn=10,
                        score_threshold: float = 0.0, exact=False):
    """  返回一个包含查询和匹配结果的列表，每个查询对应一个列表，包含匹配标记的 payload、得分和 ID。"""
    id_record = await client.retrieve(collection_name=collection_name, ids=ids, with_vectors=True)
    not_match_ids = [qcm.HasIdCondition(has_id=not_ids)]  # ids +
    query_vectors = [p.vector for p in id_record]
    search_hit = await search_by_vecs(query_vectors, collection_name, client,
                                      payload_key=payload_key, match=match, not_match=not_match_ids + not_match,
                                      topn=topn, score_threshold=score_threshold, exact=exact)

    return [(id, [(p.payload[payload_key], p.score, p.id) for p in hit]) for id, hit in zip(ids, search_hit)]  # [:topn]


async def search_by_embeddings(querys: list[str] | tuple[str], collection_name: str, client,
                               payload_key: str | list = 'title', vector_name: str = None,
                               match: list = [], not_match: list = [], topn: int = 10, score_threshold: float = 0.0,
                               exact: bool = False, get_dict=True,
                               embeddings_calls: Callable[..., Any] = lambda *args, **kwargs: [], **kwargs):
    """
    使用 Qdrant 批量查询查询词的嵌入，返回与查询最相似的标记和得分。

    :param querys: 查询字符串或查询字符串列表。
    :param collection_name: Qdrant 集合的名称。
    :param client: Qdrant 客户端实例。
    :param payload_key: 返回的 payload 中的键名，默认是 'word'。
    :param vector_name:
    :param match: 搜索时需要匹配的过滤条件。
    :param not_match: 搜索时不匹配的过滤条件。
    :param topn: 返回的最相似结果数量。
    :param score_threshold: 返回的最小得分阈值。
    :param exact: 是否进行精确搜索（True）或近似搜索（False）。
    :param get_dict: 控制返回 dict 还是 tuple
    :param embeddings_calls: 用于生成嵌入向量的可调用函数，默认为 ai_embeddings。
    :param kwargs: 其他参数，传递给 embeddings_calls。

    :return: 返回一个包含查询和匹配结果的列表，每个查询对应一个列表，包含匹配标记的 payload、得分和 ID。
    -> List[dict]
    -> List[Tuple[str, dict]]
    -> List[Tuple[Any, float, int]]:
    -> List[Tuple[str, List[Tuple[Any, float, int]]]]
    """
    query_vectors: list[list[float]] = await embeddings_calls(querys, **kwargs)
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
        query_vector = {"vector": query_vectors[0], "name": vector_name} if vector_name else query_vectors[0],
        search_hit = await client.search(collection_name=collection_name,
                                         query_vector=query_vector,  # tolist() Named Vector
                                         query_filter=query_filter,
                                         limit=topn,
                                         score_threshold=score_threshold,
                                         with_payload=with_payload,
                                         # params=qcm.SearchParams(exact=exact),
                                         # with_prefix_cache=True #第二次查询会直接命中缓存,在并发查询较多时表现更好
                                         )
        # {k: p.payload.get(k) for k in payload_key}
        return [extract_payload(p, payload_key) for p in search_hit]

    search_hit = await search_by_vecs(query_vectors, collection_name, client, payload_key, vector_name,
                                      match, not_match, topn, score_threshold, exact)  # ScoredPoint

    return [(item, [extract_payload(p, payload_key) for p in hit])
            for item, hit in zip(querys, search_hit)]  # [:topn]


def rerank_similar_by_recommend(ids, recommend_hit, topn=10, duplicate=0):
    """
    重排序函数，支持不同类型的数据处理（推荐或搜索）。
    :param ids: ID 或名称列表
    :param recommend_hit: 推荐或搜索的结果
    :param topn: 返回的前 topn 个结果
    :param duplicate: 去重方式，0为不去重，1为去重，2为基于节点排序去重，3为全局排序去重
    :return: 排序后的相似数据
    """
    similar_next = {}  # 数据格式:{_id:[(next_id,score)*topn]}
    # 处理无重复的情况
    if duplicate == 0:
        similar_next = {i: [(p.id, p.score) for p in hit[:topn]] for i, hit in zip(ids, recommend_hit)}
        # return [(_id, [(p.id, p.score) for p in hit]) for _id, hit in zip(ids, search_hit)]

    # 处理去重情况
    if duplicate == 1:
        for i, hit in zip(ids, recommend_hit):
            y = [(p.id, p.score) for p in hit]
            y2 = [_[0] for x in similar_next.values() for _ in x]
            similar_next[i] = [y3 for y3 in y if y3[0] not in y2][:topn]

    # 处理排序并去重的情况（按节点排序后去除重复）各自排序后,由下一节点各自选最近的上节点,最后由上节点各自匹配重复关系
    if duplicate == 2:
        similar_with = {}
        for i, hit in zip(ids, recommend_hit):
            y = [(p.id, p.score) for p in hit]
            for z in y:
                if z[0] not in similar_with or z[1] > similar_with[z[0]][1]:  # 对关系去重选最大的,反向选择
                    similar_with[z[0]] = (i, z[1])
        for y, x in similar_with.items():
            similar_next[x[0]] = sorted(similar_next.get(x[0], []) + [(y, x[1])],
                                        key=lambda z: z[1], reverse=True)[:topn]
    # 处理全局排序并去重的情况
    if duplicate == 3:
        similar_sort = sorted(
            [(i, p.id, p.score) for i, hit in zip(ids, recommend_hit) for p in hit],
            key=lambda x: x[2], reverse=True)
        pos = []
        for ix, iy, v in similar_sort:
            y = [p[1] for p in pos]
            x = [p[0] for p in pos if p[0] == ix]
            if len(x) < topn and iy not in y:
                pos.append((ix, iy))
                similar_next[ix] = similar_next.get(ix, []) + [(iy, v)]

    # print(similar_next)
    return [(i, similar_next[i]) for i in ids if i in similar_next]


def rerank_similar_by_search(names, search_hit, topn=10, duplicate=0, key_name='word'):
    similar_next = {}  # 数据格式:{co:[(next,val)*topn]}
    if duplicate == 0:
        similar_next = {name: [(p.payload[key_name], p.score) for p in hit[:topn]] for name, hit in
                        zip(names, search_hit)}
        # return [(name,[(p.payload['word'],p.score) for p in hit[:topn]]) for name,hit in zip(names,search_hit)]

    if duplicate == 1:
        for name, hit in zip(names, search_hit):
            y = [(p.payload[key_name], p.score) for p in hit]
            y2 = [_[0] for x in similar_next.values() for _ in x]
            similar_next[name] = [y3 for y3 in y if y3[0] not in y2][:topn]

    if duplicate == 2:  # 各自排序后,由下一节点各自选最近的上节点,最后由上节点各自匹配重复关系
        similar_with = {}
        for name, hit in zip(names, search_hit):
            y = [(p.payload[key_name], p.score) for p in hit]
            for z in y:
                if z[0] not in similar_with or z[1] > similar_with[z[0]][1]:  # 对关系去重选最大的,反向选择
                    similar_with[z[0]] = (name, z[1])
        for y, x in similar_with.items():
            similar_next[x[0]] = sorted(similar_next.get(x[0], []) + [(y, x[1])],
                                        key=lambda z: z[1], reverse=True)[:topn]

    if duplicate == 3:
        similar_sort = sorted(
            [(name, p.payload[key_name], p.score) for name, hit in zip(names, search_hit) for p in hit],
            key=lambda x: x[2], reverse=True)
        pos = []
        for ix, iy, v in similar_sort:
            y = [p[1] for p in pos]
            x = [p[0] for p in pos if p[0] == ix]
            if len(x) < topn and iy not in y:
                pos.append((ix, iy))
                similar_next[ix] = similar_next.get(ix, []) + [(iy, v)]

    # print(similar_next)
    return [(w, similar_next[w]) for w in names if w in similar_next]


# RecommendBatch
async def recommend_batch_by_ids(ids, collection_name, client, match=[], not_match=[], not_ids=[], topn=10, duplicate=0,
                                 score_threshold: float = 0.0):
    not_match_ids = [qcm.HasIdCondition(has_id=not_ids)]
    query_filter = qcm.Filter(must=match, must_not=not_match_ids + not_match)  # 缩小查询范围

    recommend_queries = [
        qcm.RecommendRequest(positive=[_id], filter=query_filter, limit=topn * len(ids),
                             score_threshold=score_threshold, with_payload=False)
        for _id in ids]

    recommend_hit = await client.recommend_batch(collection_name=collection_name,
                                                 requests=recommend_queries)  # ScoredPoint

    return rerank_similar_by_recommend(ids, recommend_hit, topn=topn, duplicate=duplicate)


# SearchBatch
async def search_batch_by_ids(ids, collection_name, client, key_name='word', match=[], not_ids=[], topn=10, duplicate=0,
                              score_threshold: float = 0.0):
    id_record = await  client.retrieve(collection_name=collection_name, ids=ids, with_vectors=True)

    names = [p.payload[key_name] for p in id_record]
    not_match_ids = [qcm.HasIdCondition(has_id=ids + not_ids)]
    query_filter = qcm.Filter(must=match, must_not=not_match_ids)  # 缩小查询范围

    search_queries = [
        qcm.SearchRequest(vector=p.vector, filter=query_filter, limit=topn * len(names),
                          score_threshold=score_threshold,
                          with_payload=[key_name])
        for p in id_record]

    search_hit = await client.search_batch(collection_name=collection_name, requests=search_queries)  # ScoredPoint

    return rerank_similar_by_search(names, search_hit, topn=topn, duplicate=duplicate, key_name=key_name)


# Relation
async def node_relations(collection_name, client, width, depth, _id, ids_depth: dict = {}, match=[], not_match=[],
                         key_name='word', exclude_all=True, score_threshold=0.0):
    relationships = []
    try:
        not_ids = list(ids_depth) if exclude_all else [k for k in ids_depth if ids_depth[k] <= depth]
        similar_next = await recommend_by_id([_id], collection_name, client, key_name=key_name, match=match,
                                             not_match=not_match, not_ids=not_ids, topn=width,
                                             score_threshold=score_threshold, ret_name=False)
        ids = [y[0] for y in similar_next]  # p.payload[self.key_name] if use_name else p.id
        if len(ids) == 0:
            return ids_depth, relationships  # 返回递归

        # print("Depth:", depth, "Similar:", _id, "->", similar_next)
        new_depth = {i: depth + 1 for i in ids if i not in ids_depth}
        ids_depth.update(new_depth)  # 当前层去重

        relationships = [
            {'source': _id, 'target': y[0], 'relation': str(round(y[1], 5)), 'value': 1.0 - y[1], 'rank': i}
            for i, y in enumerate(similar_next)]

    except KeyError:
        id_first = [k for k, v in ids_depth.items() if v == 0]
        print(f"Data '{id_first, _id}' not in vocabulary.")
        # new_depth = {}

    return new_depth, relationships


# Relations
async def nodes_relations(collection_name, client, width=3, depth=0, ids=[], match=[], not_match=[], ids_depth={},
                          relationships_edges=[], duplicate=3, score_threshold: float = 0.0):
    try:
        similar_next = await recommend_batch_by_ids(ids, collection_name, client, match=match, not_match=not_match,
                                                    not_ids=list(ids_depth), topn=width, duplicate=duplicate,
                                                    score_threshold=score_threshold)  # 分支会有合并,下层关系去重

        ids = list(set(y[0] for x in similar_next for y in x[1]))
        # print("Depth:", depth, "Similar:", [x[0] for x in similar_next], "->", similar_next)
        ids_depth.update({i: depth + 1 for i in ids if i not in ids_depth})
        relationships = [
            {'source': x[0], 'target': y[0], 'relation': str(round(y[1], 5)), 'value': 1.0 - y[1], 'rank': i}
            for x in similar_next for i, y in enumerate(x[1])]
        relationships_edges += relationships

    except KeyError:
        id_first = [k for k, v in ids_depth.items() if v == 0]  # list(ids_depth)[0]
        print(f"Data '{id_first, ids}' not in vocabulary.")
        ids = []

    return ids


# RelationsDepth
async def relations_depth(collection_name, client, _id, match=[], not_match=[], layers=[3, 3, 3], max_calc=30,
                          max_node=0,
                          score_threshold=0.0, exclude_all=True, depth=0, ids_depth={}, relationships_edges=[]):
    if ids_depth is None:
        ids_depth = {}
    if relationships_edges is None:
        relationships_edges = []
    if depth == 0:
        relationships_edges.clear()
        ids_depth.clear()
        ids_depth[_id] = 0

    if depth >= len(layers):
        return ids_depth, relationships_edges

    if len(ids_depth) + layers[depth] > max_node > 0:
        return ids_depth, relationships_edges

    if max_calc > 0 and len(ids_depth) > 1:
        count = 1  # len(relationships_edges)
        for d, w in enumerate(layers):
            n_depth = [dn for dn in ids_depth.values() if dn == d + 1]
            count += len(n_depth) // w
        if count >= max_calc:
            return ids_depth, relationships_edges

    new_depth, relationships = await node_relations(collection_name, client, width=layers[depth], depth=depth, _id=_id,
                                                    ids_depth=ids_depth, match=match, not_match=not_match, key_name='',
                                                    exclude_all=exclude_all, score_threshold=score_threshold)

    relationships_edges += relationships
    tasks = [
        relations_depth(collection_name, client, i, match, not_match, layers, max_calc, max_node, score_threshold,
                        exclude_all, depth + 1, ids_depth, relationships_edges)
        for i in new_depth]

    await asyncio.gather(*tasks)

    return ids_depth, relationships_edges


# RelationsBreadth
async def relations_breadth(collection_name, client, _id, match=[], not_match=[], layers=[3, 3, 3], duplicate=3,
                            max_calc=30,
                            max_node=0, score_threshold=0.0):
    ids_depth = {_id: 0}
    relationships_edges = []

    ids = [_id]
    depth = 0
    count = 0

    for width in layers:
        if len(ids_depth) + len(ids) * width > max_node > 0:
            cut_len = (max_node - len(ids_depth)) // width
            if cut_len <= 0:
                break
            ids = ids[:cut_len]

        if count + len(ids) > max_calc > 0:  # calc_layers_total
            cut_len = (max_calc - count) // width
            if cut_len <= 0:
                break
            ids = ids[:cut_len]

        count += len(ids)

        ids = await nodes_relations(collection_name, client, width=width, depth=depth, ids=ids, match=match,
                                    not_match=not_match, ids_depth=ids_depth, relationships_edges=relationships_edges,
                                    duplicate=duplicate, score_threshold=score_threshold)

        depth += 1

        if len(ids) == 0:
            break

    return ids_depth, relationships_edges


# 多层次关系探索:max_neighbors max_depth search_tree append(children)
# SimilarRelations
async def graph_relations(collection_name, client, name, width=3, max_depth=3, layers=[], batch=True,
                          duplicate=3, max_calc=30, max_node=0, score_threshold=0.0, exclude_name=[],
                          key_name='word', field_key='行业', match_values=[]):
    match_first = field_match(field_key, match_values)
    not_match_name = field_match(key_name, exclude_name)
    name_id = await names_to_ids([name], collection_name, client, match_first, key_name)
    _id = name_id.get(name, -1)
    if _id == -1:
        return {}, []

    if not layers:
        layers = [width] * max_depth

    if batch:
        ids_depth, relationships_edges = await relations_breadth(collection_name, client, _id, match_first,
                                                                 not_match_name, layers,
                                                                 duplicate=duplicate,
                                                                 max_calc=max_calc, max_node=max_node,
                                                                 score_threshold=score_threshold)

    else:
        ids_depth, relationships_edges = await relations_depth(collection_name, client, _id, match_first,
                                                               not_match_name, layers,
                                                               max_calc=max_calc, max_node=max_node,
                                                               score_threshold=score_threshold,
                                                               exclude_all=(duplicate > 0))

    payloads = await get_payload([], collection_name, client, ids=list(ids_depth))
    nodes = [{"id": i, "name": p[key_name], "depth": ids_depth[i], 'properties': {**p}}
             for idx, (i, p) in enumerate(payloads)]

    key_nodes = {n['id']: n for n in nodes}
    edges = relationships_edges.copy()
    for i, rel in enumerate(edges):
        rel['source'] = key_nodes.get(rel['source'], rel['source'])
        rel['target'] = key_nodes.get(rel['target'], rel['target'])
        # rel['id'] = str(i)
        # edges[i] = {"data": rel}

    return {"nodes": nodes, "edges": edges}


# SimilarRelation
async def similar_node_relations(collection_name, client, node_id, node_name, existing_nodes, width=3,
                                 duplicate=3, score_threshold=0.0, exclude_name=[], key_name='word', field_key='行业',
                                 match_values=[]):
    match_first = field_match(field_key, match_values)
    not_match_name = field_match(key_name, exclude_name)
    ids_depth = {n['id']: n['depth'] for n in existing_nodes}
    new_depth, new_relationships = await node_relations(collection_name, client, width=width,
                                                        depth=ids_depth.get(node_id, 1),
                                                        _id=node_id, ids_depth=ids_depth, match=match_first,
                                                        not_match=not_match_name, key_name=key_name,
                                                        exclude_all=(duplicate > 0),
                                                        score_threshold=score_threshold)

    ids_name = await  ids_to_names(list({**ids_depth, **new_depth}), collection_name, client, key_name)
    nodes = [{"id": i, "name": ids_name.get(i, 'None'), "depth": d} for i, d in new_depth.items()]
    return {"nodes": nodes, "edges": new_relationships}



def recreate_with_named_vector(collection_name, named_vector, client, new_collection=None, batch_size=1000):
    from tqdm import tqdm
    collection_info = client.get_collection(collection_name)
    print(collection_info.dict())
    vector_params = collection_info.config.params.vectors
    print(vector_params)
    if named_vector in vector_params:
        vector_size = vector_params[named_vector].size
        vector_distance = vector_params[named_vector].distance
        raise ValueError(f"Vector {named_vector} found in collection {collection_name}")

    vector_size = vector_params.size
    vector_distance = vector_params.distance
    all_points = []
    scroll_offset = None

    while True:
        result, scroll_offset = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            with_vectors=True,
            with_payload=True,
            offset=scroll_offset
        )
        all_points.extend(result)
        if scroll_offset is None:
            break

    if not new_collection:
        new_collection = collection_name

    if new_collection == collection_name and client.collection_exists(collection_name):
        client.delete_collection(collection_name)  # 删除现有集合

    # 重新创建 collection，并设置新的 vector 配置
    client.create_collection(
        collection_name=new_collection,
        vectors_config={
            named_vector: qcm.VectorParams(size=vector_size, distance=vector_distance)
        }
    )

    named_points = []
    for point in all_points:
        named_vector_data = {named_vector: point.vector}
        named_points.append(
            qcm.PointStruct(
                id=point.id,
                vector=named_vector_data,
                payload=point.payload
            )
        )
    for i in tqdm(range(0, len(named_points), batch_size)):
        batch = named_points[i:i + batch_size]
        client.upsert(collection_name=new_collection, points=batch)


if __name__ == "__main__":
    # qc = AsyncQdrantClient(host='47.110.156.41', grpc_port=6334, prefer_grpc=True)
    # http://47.110.156.41:6333/dashboard#/collections
    qc = QdrantClient(url="http://47.110.156.41:6333")
    print(qc.get_collections())
    collection_names = [collection.name for collection in qc.get_collections().collections]
    print(collection_names)

    collection_aliases = qc.get_collection_aliases("拜访记录")
    if len(collection_aliases.aliases):
        model_name_1 = collection_aliases.aliases[0].alias_name.split("_")[-1]
        print(model_name_1)

    print(qc.retrieve(collection_name="places_bank", ids=[2], with_vectors=True))

    # print(qc.update_collection(
    #     collection_name="places_bank",
    #     vectors_config={
    #         "location": qcm.VectorParamsDiff(),
    #     }
    # ))

    # recreate_with_named_vector("places_bank", "location", qc, batch=1000)

    print(qc.search(collection_name="places_bank",
                    query_vector={"vector": [29.86192, 121.60988], "name": "location"},
                    # neighbor_vector then result_vector tolist()
                    with_payload=True,
                    limit=4,  # max_neighbors
                    score_threshold=0))

    # response = qc.query_batch_points(
    #     collection_name="places_bank",
    #     requests=[
    #         qcm.QueryRequest(
    #             query={"location": [29.86192, 121.60988]},  # 第一个查询向量[29.86192, 121.60988],
    #             limit=1,  # 取最近的 5 个结果
    #             # with_payload=True,
    #         ),
    #         # qcm.QueryRequest(
    #         #     query=[0.4, 0.5],  # 第二个查询向量
    #         #     limit=3,  # 取最近的 3 个结果
    #         #     filter=qcm.Filter(must=[qcm.FieldCondition(key="color", match=qcm.MatchValue(value="red"))]),
    #         # ),
    #
    #     ]
    # )
    # print(response)
    #
    # doc_1 = qcm.Document(text="123", model="Qdrant/bm25")

    import asyncio


    async def main():
        from generates import ai_embeddings, init_ai_clients
        import numpy as np
        import pickle
        from config import AI_Models, API_KEYS

        qc = AsyncQdrantClient(host='47.110.156.41')  # , grpc_port=6334, prefer_grpc=True

        # data = np.load('../data/embeddings_拜访_MiniLM_2.npz')
        data = np.load('../data/embeddings_拜访_bge-large-zh.npz')
        text_embeddings = data['embeddings']
        index = data['index']

        with open("../data/ideatech_跟进记录_record_text.pkl", "rb") as f:
            payloads = pickle.load(f)
        assert len(payloads) == text_embeddings.shape[0]

        print(text_embeddings.shape)

        collection_name = '拜访记录'
        model_name = 'BAAI/bge-large-zh-v1.5'  # 'BAAI/bge-base-zh-v1.5'  # 'all-MiniLM-L6-v2'
        size = text_embeddings.shape[1]
        print(size)  # 1024
        await init_ai_clients(AI_Models, API_KEYS, get_data=False)
        embeddings = await ai_embeddings(inputs='测试', model_name=model_name, model_id=0)
        print(embeddings)
        assert embeddings and embeddings[0]
        assert size == len(embeddings[0])  # 384

        alias_name = collection_name + f"_{model_name}"  # suffix
        res = await create_collection(collection_name, client=qc, size=size, alias_name=alias_name)

        if res:
            res['model_name'] = model_name
            res['model_size'] = size
            print(res)

        collection_aliases = await qc.get_collection_aliases(collection_name)
        if len(collection_aliases.aliases):
            model_name_1 = collection_aliases.aliases[0].alias_name.split("_")[-1]
            assert model_name_1 == model_name

        # if text_field and not inputs:
        #     inputs = [p.get(text_field) for p in payloads]
        # embeddings = await ai_embeddings(inputs=inputs, model_name=model_name, model_id=0, get_embedding=True)
        # if embeddings:
        #     return {'operation_id': await upsert_points(payloads, vectors=embeddings,
        #                                                 collection_name=collection_name, client=QD_Client),
        #             'embeddings_size': len(embeddings),
        #             'embeddings_model': model_name}

        # await upsert_points(payloads, vectors=embeddings,collection_name=collection_name, client=qc)

        r = await  upsert_points_batch(payloads, text_embeddings.tolist(), collection_name, client=qc, batch_size=1000)
        print(r)  # 18952

        ret = await qc.get_collection(collection_name=collection_name)
        print(ret.model_dump())

        # {
        #     'status': < CollectionStatus.GREEN: 'green' >, 'optimizer_status': < OptimizersStatusOneOf.OK: 'ok' >, 'vectors_count': None, 'indexed_vectors_count': 0, 'points_count': 18952, 'segments_count': 4, 'config': {
        #     'params': {'vectors': {'size': 384, 'distance'
        #                            : < Distance.COSINE: 'Cosine' >, 'hnsw_config': None, 'quantization_config': None, 'on_disk': True, 'datatype': None, 'multivector_config': None}, 'shard_number': 1, 'sharding_method': None, 'replication_factor': 1, 'write_consistency_factor': 1, 'read_fan_out_factor': None, 'on_disk_payload': True, 'sparse_vectors': None}, 'hnsw_config': {
        #     'm': 16, 'ef_construct': 100, 'full_scan_threshold': 10000, 'max_indexing_threads': 0, 'on_disk': True,
        #     'payload_m': None}, 'optimizer_config': {'deleted_threshold': 0.2, 'vacuum_min_vector_number': 1000,
        #                                              'default_segment_number': 0, 'max_segment_size': None,
        #                                              'memmap_threshold': None, 'indexing_threshold': 20000,
        #                                              'flush_interval_sec': 5,
        #                                              'max_optimization_threads': None}, 'wal_config': {
        #     'wal_capacity_mb': 32, 'wal_segments_ahead': 0}, 'quantization_config': None}, 'payload_schema': {}
        # }

    # asyncio.run(main())
