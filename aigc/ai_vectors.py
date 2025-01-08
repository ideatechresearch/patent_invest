from qdrant_client import QdrantClient, AsyncQdrantClient
import asyncio
# from qdrant_client.models import Filter, FieldCondition, IsEmptyCondition, HasIdCondition, MatchValue
from typing import List, Dict, Any, Union, Tuple, Callable, Optional
import qdrant_client.models as qcm
from generates import ai_embeddings


def empty_match(field_key):
    if not field_key:
        return []
    return [qcm.IsEmptyCondition(is_empty=qcm.PayloadField(key=field_key), )]


def field_match(field_key, match_values):
    if not field_key or not match_values:
        return []
    if isinstance(match_values, (str, int, float)):
        return [qcm.FieldCondition(key=field_key, match=qcm.MatchValue(value=match_values, ), )]
    return [qcm.FieldCondition(key=field_key, match=qcm.MatchValue(value=v), ) for v in match_values]


async def ids_to_names(ids, collection_name, client, key_name='word'):
    id_record = await client.retrieve(collection_name=collection_name, ids=ids, with_payload=[key_name])
    return {p.id: p.payload[key_name] for p in id_record}


async def names_to_ids(names, collection_name, client, match=[], key_name='word'):
    shoulds = [qcm.FieldCondition(key=key_name, match=qcm.MatchValue(value=w)) for w in names]
    scroll_filter = qcm.Filter(must=match, should=shoulds)
    scroll_result = await client.scroll(collection_name=collection_name, scroll_filter=scroll_filter,
                                        with_payload=[key_name],
                                        # with_vectors=True,
                                        limit=len(names))

    return {i.payload[key_name]: i.id for i in scroll_result[0]}
    # [(i.payload[key_name], i.id) for i in scroll_result[0]]


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
        scroll_result = await client.scroll(collection_name=collection_name,
                                            scroll_filter=scroll_filter,
                                            with_vectors=True,
                                            # with_payload=[key_name],
                                            limit=1,  # 只获取一个匹配的结果
                                            # order_by=OrderBy(key='df',direction='desc')
                                            )
        point_result = scroll_result[0]  # (points,next_page_offset)
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


async def recommend_by_ids(ids, collection_name, client, key_name='word', match=[], not_match=[], topn=10,
                           score_threshold: float = 0.0, ret_name=False):
    not_match_ids = [qcm.HasIdCondition(has_id=ids)]
    query_filter = qcm.Filter(must=match, must_not=not_match_ids + not_match)  # 缩小查询范围

    recommend_queries = [
        qcm.RecommendRequest(positive=[_id], filter=query_filter, limit=topn, score_threshold=score_threshold,
                             with_payload=[key_name] if (ret_name and key_name) else False)
        for _id in ids]

    recommend_hit = await client.recommend_batch(collection_name=collection_name,
                                                 requests=recommend_queries)  # ScoredPoint

    return [(_id, [(p.payload[key_name] if (ret_name and key_name) else p.id, p.score) for p in hit]) for _id, hit in
            zip(ids, recommend_hit)]


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


# SimilarByIds
async def search_by_ids(ids, collection_name, client, key_name='word', match=[], not_match=[], not_ids=[], topn=10,
                        score_threshold: float = 0.0, exact=False):
    id_record = await client.retrieve(collection_name=collection_name, ids=ids, with_vectors=True)

    not_match_ids = [qcm.HasIdCondition(has_id=ids + not_ids)]
    query_filter = qcm.Filter(must=match, must_not=not_match_ids + not_match)  # 缩小查询范围

    search_queries = [
        qcm.SearchRequest(vector=p.vector, filter=query_filter, limit=topn, score_threshold=score_threshold,
                          with_payload=[key_name], params=qcm.SearchParams(exact=exact), )
        for p in id_record]

    search_hit = await client.search_batch(collection_name=collection_name, requests=search_queries)  # ScoredPoint

    return [(id, [(p.payload[key_name], p.score) for p in hit]) for id, hit in zip(ids, search_hit)]  # [:topn]


def rerank_similar_by_recommend(ids, recommend_hit, topn=10, duplicate=0):
    similar_next = {}  # 数据格式:{_id:[(next_id,score)*topn]}
    if duplicate == 0:
        similar_next = {i: [(p.id, p.score) for p in hit[:topn]] for i, hit in zip(ids, recommend_hit)}
        # return [(_id, [(p.id, p.score) for p in hit]) for _id, hit in zip(ids, search_hit)]

    if duplicate == 1:
        for i, hit in zip(ids, recommend_hit):
            y = [(p.id, p.score) for p in hit]
            y2 = [_[0] for x in similar_next.values() for _ in x]
            similar_next[i] = [y3 for y3 in y if y3[0] not in y2][:topn]

    if duplicate == 2:  # 各自排序后,由下一节点各自选最近的上节点,最后由上节点各自匹配重复关系
        similar_with = {}
        for i, hit in zip(ids, recommend_hit):
            y = [(p.id, p.score) for p in hit]
            for z in y:
                if z[0] not in similar_with or z[1] > similar_with[z[0]][1]:  # 对关系去重选最大的,反向选择
                    similar_with[z[0]] = (i, z[1])
        for y, x in similar_with.items():
            similar_next[x[0]] = sorted(similar_next.get(x[0], []) + [(y, x[1])],
                                        key=lambda z: z[1], reverse=True)[:topn]

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


async def similar_by_embeddings(query, collection_name, client, topn=10, score_threshold=0.0, query_vector=[],
                                match=[], not_match=[], embeddings_calls: Callable[[...], Any] = ai_embeddings,
                                **kwargs):
    if not query_vector:
        query_vector = await embeddings_calls(query, **kwargs)
        if not query_vector:
            return []

    query_filter = qcm.Filter(must=match, must_not=not_match)
    search_hit = await client.search(collection_name=collection_name,
                                     query_vector=query_vector,  # tolist()
                                     query_filter=query_filter,
                                     limit=topn,
                                     score_threshold=score_threshold,
                                     )
    return [(p.payload, p.score) for p in search_hit]


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
            ndepth = [dn for dn in ids_depth.values() if dn == d + 1]
            count += len(ndepth) // w
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
async def graph_relations(collection_name, client, name, key_radius='', width=3, max_depth=3, layers=[], batch=True,
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
