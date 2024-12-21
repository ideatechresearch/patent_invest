from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, HasIdCondition, IsEmptyCondition, MatchValue, PayloadField, \
    SearchRequest, SearchParams, QuantizationSearchParams, VectorParams, CollectionStatus, PointStruct, OrderBy, Batch, \
    Distance, Range
from py2neo import Graph, Node, Relationship, Subgraph
from embeddings_api import *
# import asyncio


def cosine_sim(A, B):
    import numpy as np
    dot_product = np.dot(A, B)
    similarity = dot_product / (np.linalg.norm(A) * np.linalg.norm(B))
    return similarity


# 计算次数：13
def calc_layers_total(layers=[3, 3, 3]):
    n = 0
    product = 1
    for layer in layers:
        n += product
        product *= layer
    return n
    # import math
    # for i in range(len(layers)):
    #     n += math.prod(layers[:i], start=1)


# 最大节点数：40
def calc_nodes_total(layers=[3, 3, 3]):
    n = 1
    product = 1
    for layer in layers:
        product *= layer
        n += product
    return n


def scale_to_range(numbers, new_min, new_max):
    min_value = min(numbers)
    max_value = max(numbers)

    # 避免除以零的情况
    if min_value == max_value:
        return [new_min + (new_max - new_min) / 2] * len(numbers)

    scaled = (new_max - new_min) / (max_value - min_value)
    scaled_numbers = [
        new_min + (num - min_value) * scaled for num in numbers
    ]

    return scaled_numbers


def download_file(url, local_filename):
    import requests
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(local_filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
    return local_filename


def recover_snapshots_upload(host, collection_name, file_path):
    import requests
    url = f"http://{host}:6333/collections/{collection_name}/snapshots/upload"
    with open(file_path, 'rb') as f:
        files = {'snapshot': f}
        # headers={"Content-Type": "multipart/form-data"}
        response = requests.post(url, files=files)

    return response.json()


def del_collections_snapshots(client):
    for collection in client.get_collections().collections:
        snapshots_names = client.list_snapshots(collection.name)
        print(collection.name, snapshots_names)
        if snapshots_names:
            snapshots_name = snapshots_names[0].name
            client.delete_snapshot(collection_name=collection.name, snapshot_name=snapshots_name)


def create_vdb_collection(collection_name, size, new=False):
    if new and client.collection_exists(collection_name=collection_name):
        client.delete_collection(collection_name=collection_name)

    if not client.collection_exists(collection_name=collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=size, distance=Distance.COSINE),  # { "size": 128, "distance": "Cosine" }
            # optimizer_config=OptimizersConfigDiff(indexing_threshold=0),#上传期间禁用索引,避免不必要的向量索引，这些向量将被下一批覆盖
        )
    return client.count(collection_name=collection_name, exact=True)  # get_collection(collection_name)


def get_vdb_point(host, collection_name, ids):
    import requests
    response = requests.get(f'http://{host}:6333/collections/{collection_name}/points/{ids}')  # 从单个点检索所有详细信息
    return response.json()


def most_similar_embeddings(query, collection_name, client, topn=10, score_threshold=0.0,
                            match=[], not_match=[], access_token=''):
    query_vector = get_bge_embeddings(query, access_token)
    if not query_vector:
        return []
    query_filter = Filter(must=match, must_not=not_match)
    search_hit = client.search(collection_name=collection_name,
                               query_vector=query_vector,  # tolist()
                               query_filter=query_filter,
                               limit=topn,
                               score_threshold=score_threshold,
                               search_params=SearchParams(exact=True,  # Turns on the exact search mode
                                                          # quantization=QuantizationSearchParams(rescore=True),
                                                          ),
                               # offset=1
                               )
    return [(p.payload, p.score) for p in search_hit]


def most_similar_by_name(name, collection_name, client, match=[], exclude=[], topn=10, score_threshold=0.5):
    match_name = FieldCondition(key='word', match=MatchValue(value=name, ), )
    not_match_name = [FieldCondition(key='word', match=MatchValue(value=w), ) for w in exclude]
    scroll_filter = Filter(must=match + [match_name])
    scroll_result = client.scroll(collection_name=collection_name,
                                  scroll_filter=scroll_filter,
                                  with_vectors=True,
                                  with_payload=True,
                                  limit=1,  # 只获取一个匹配的结果
                                  # order_by=OrderBy(key='df',direction='desc')
                                  )

    query_vector = scroll_result[0][0].vector

    query_filter = Filter(must=match, must_not=[match_name] + not_match_name)  # 缩小查询范围
    search_hit = client.search(collection_name=collection_name,
                               query_vector=query_vector,  # tolist()
                               query_filter=query_filter,
                               limit=topn,
                               score_threshold=score_threshold,
                               search_params=SearchParams(exact=True,  # Turns on the exact search mode
                                                          # quantization=QuantizationSearchParams(rescore=True),
                                                          ),
                               # offset=1
                               )

    return [(p.payload['word'], p.score) for p in search_hit]


def most_similar_by_ids(ids, collection_name, client, match=[], exclude=[], topn=10, score_threshold=0.0):
    id_record = client.retrieve(collection_name=collection_name, ids=ids, with_vectors=True)

    names = [p.payload['word'] for p in id_record]
    not_match_name = [FieldCondition(key='word', match=MatchValue(value=w), ) for w in names + exclude]
    query_filter = Filter(must=match, must_not=not_match_name)  # 缩小查询范围

    search_queries = [SearchRequest(vector=p.vector, filter=query_filter, limit=topn,
                                    score_threshold=score_threshold,
                                    with_payload=['word'],
                                    # params=SearchParams(exact=True),
                                    # offset=1,
                                    )
                      for p in id_record]

    search_hit = client.search_batch(collection_name=collection_name, requests=search_queries)  # ScoredPoint
    return [(id, name, [(p.payload['word'], p.score) for p in hit]) for id, name, hit in
            zip(ids, names, search_hit)]  # [:topn]


def names_to_ids(names, collection_name, client, match=[], key_name='word'):
    shoulds = [FieldCondition(key=key_name, match=MatchValue(value=w)) for w in names]
    scroll_filter = Filter(must=match, should=shoulds)

    scroll_result = client.scroll(collection_name=collection_name,
                                  scroll_filter=scroll_filter,
                                  with_payload=True,
                                  # with_vectors=True,
                                  limit=len(names))

    return {i.payload[key_name]: i.id for i in scroll_result[0]}


def field_match(field_key, match_values):
    if not field_key or match_values == 'all':
        return []
    if isinstance(match_values, (str, int, float)):
        return [FieldCondition(key=field_key, match=MatchValue(value=match_values, ), )]
    return [FieldCondition(key=field_key, match=MatchValue(value=v), ) for v in match_values]


def empty_match(field_key):
    if not field_key:
        return []
    return [IsEmptyCondition(is_empty=PayloadField(key=field_key), )]


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


def SimilarByIds(ids, collection_name, client, match=[], exclude=[], topn=10, duplicate=0, score_threshold=0):
    id_record = client.retrieve(collection_name=collection_name, ids=ids, with_vectors=True)

    names = [p.payload['word'] for p in id_record]
    not_match_name = [FieldCondition(key='word', match=MatchValue(value=w), ) for w in names + exclude]
    query_filter = Filter(must=match, must_not=not_match_name)  # 缩小查询范围

    search_queries = [
        SearchRequest(vector=p.vector, filter=query_filter, limit=topn * len(names), score_threshold=score_threshold,
                      with_payload=True)
        for p in id_record]

    search_hit = client.search_batch(collection_name=collection_name, requests=search_queries)  # ScoredPoint

    return rerank_similar_by_search(names, search_hit, topn=topn, duplicate=duplicate, key_name='word')


class VDBSimilar:
    client = None
    collection_name = ''
    key_name = 'word'
    match_first = []

    def __init__(self, client, collection_name, key_name='word', match_first=[]):
        self.client = client
        self.collection_name = collection_name
        self.key_name = key_name  # .lower()
        self.match_first = match_first

        self.name_ids = {}  # unique,记忆

    def get_ids(self, names):
        finds = set(names) - self.name_ids.keys()
        if len(finds):  # 更新未记录id
            result = names_to_ids(finds, self.collection_name, self.client, self.match_first, self.key_name)
            if result:
                self.name_ids.update(result)

        return [self.name_ids[n] for n in names if n in self.name_ids]

    def get_id(self, name):
        if name not in self.name_ids:
            result = names_to_ids([name], self.collection_name, self.client, self.match_first, self.key_name)
            if result:
                self.name_ids.update(result)

        return self.name_ids.get(name, None)

    def get_menory(self):
        return self.name_ids

    def get_vecs(self, names):
        ids = self.get_ids(names)
        id_record = self.client.retrieve(collection_name=self.collection_name, ids=ids, with_vectors=True)
        return {p.payload[self.key_name]: p.vector for p in id_record}

    def get_vec(self, name):
        if name in self.name_ids:
            point_result = self.client.retrieve(collection_name=self.collection_name, ids=[self.name_ids[name]],
                                                with_vectors=True)
        else:
            match_name = FieldCondition(key=self.key_name, match=MatchValue(value=name, ), )
            scroll_filter = Filter(must=self.match_first + [match_name])
            scroll_result = self.client.scroll(collection_name=self.collection_name,
                                               scroll_filter=scroll_filter,
                                               with_vectors=True,
                                               with_payload=True,
                                               limit=1,  # 只获取一个匹配的结果
                                               # order_by=OrderBy(key='df',direction='desc')
                                               )

            point_result = scroll_result[0]  # (points,next_page_offset)
            if not point_result:
                return []

            self.name_ids[point_result[0].payload[self.key_name]] = point_result[0].id

        return point_result[0].vector

    def get_payload(self, names, fields=[]):
        ids = self.get_ids(names)
        id_record = self.client.retrieve(collection_name=self.collection_name, ids=ids,
                                         with_payload=fields if len(fields) else True)
        return [(p.id, p.payload) for p in id_record]  # [(,{}),]

    def SimilarByNames(self, names, exclude=[], not_ids=[], topn=10, duplicate=0, score_threshold=0.0):
        ids = self.get_ids(names)
        id_record = self.client.retrieve(collection_name=self.collection_name, ids=ids, with_vectors=True)

        names = [p.payload[self.key_name] for p in id_record]
        not_match_name = [FieldCondition(key=self.key_name, match=MatchValue(value=w), ) for w in names + exclude]
        not_match_ids = [HasIdCondition(has_id=ids + not_ids)]
        query_filter = Filter(must=self.match_first, must_not=not_match_ids + not_match_name)  # 缩小查询范围

        search_queries = [
            SearchRequest(vector=p.vector, filter=query_filter, limit=topn * len(names), with_payload=True,
                          score_threshold=score_threshold)
            for p in id_record]

        search_hit = self.client.search_batch(collection_name=self.collection_name,
                                              requests=search_queries)  # ScoredPoint

        self.name_ids.update({p.payload[self.key_name]: p.id for hit in search_hit for p in hit})

        return rerank_similar_by_search(names, search_hit, topn=topn, duplicate=duplicate, key_name=self.key_name)

    def SimilarByName(self, name, exclude=[], not_ids=[], topn=10, score_threshold=0.0):
        query_vector = self.get_vec(name)
        if not query_vector:
            return []

        not_match_name = [FieldCondition(key=self.key_name, match=MatchValue(value=w), ) for w in exclude]
        not_match_ids = [HasIdCondition(has_id=not_ids)]
        query_filter = Filter(must=self.match_first, must_not=not_match_ids + not_match_name)  # [match_name] + 缩小查询范围
        hit = self.client.search(collection_name=self.collection_name,
                                 query_vector=query_vector,  # tolist()
                                 query_filter=query_filter,
                                 limit=topn,
                                 score_threshold=score_threshold,
                                 offset=(0 if name in exclude else 1),
                                 # search_params=SearchParams(exact=True,  # Turns on the exact search mode
                                 #                            # quantization=QuantizationSearchParams(rescore=True),
                                 #                            ),
                                 )  # ScoredPoint

        self.name_ids.update({p.payload[self.key_name]: p.id for p in hit})

        return [(p.payload[self.key_name], p.score) for p in hit]


def CreateNode(word, depth, graph, node_name='Word', **kwargs):
    nodes = graph.nodes.match(node_name)
    node = nodes.where(name=word).first()  # Node("Word", name=word,id=mid)
    if not node:
        node = Node(node_name, name=word, depth=depth, nid=len(nodes), **kwargs)
        graph.create(node)
    return node


class VDBRelationships:
    graph = None
    vdb = {}  # functions.get(key)(,)

    def load(self, graph, client, collection_name='专利_w2v', prefix='Word', field_key='行业', match_values=['all']):
        self.graph = graph
        self.append(client, collection_name=collection_name, prefix=prefix,
                    field_key=field_key, match_values=match_values)

    def append(self, client, collection_name='专利_w2v', prefix='Word', field_key='行业', match_values=['all']):
        if not field_key:
            match_values = ['all']
        for k in match_values:
            match = [] if k == 'all' else [FieldCondition(key=field_key, match=MatchValue(value=k, ))]
            vdb_key = f"{prefix}_{k}"
            self.vdb[vdb_key] = VDBSimilar(client, collection_name=collection_name, key_name=prefix.lower(),
                                           match_first=match)
            # count = client.count(collection_name=collection_name,
            #                      count_filter=Filter(must=match), exact=True)
            # print(vdb_key, ':', count)

    def get_payload(self, vdb_key, names, fields=[]):
        if vdb_key not in self.vdb:  # self.vdb.get(vdb_key, None)
            return []
        return self.vdb[vdb_key].get_payload(names, fields)

    def get_ids(self, vdb_key, names):
        if vdb_key not in self.vdb:
            return None
        name_ids = self.vdb[vdb_key].get_menory()
        return {n: name_ids.get(n, self.vdb[vdb_key].get_id(n)) for n in names}

    def get_id(self, vdb_key, name):
        if vdb_key not in self.vdb:  # self.vdb.get(vdb_key, None)
            return None
        return self.vdb[vdb_key].get_id(name)

    def similar(self, tokens, vdb_key='Word_all', topn=10, duplicate=0, exclude=[], score_threshold=0.0):
        tokens = [token.strip().lower() for token in tokens]
        if vdb_key not in self.vdb:
            return tokens, []
        # tokens = [token for token in tokens if token in self.data.index]  # data.index.isin(tokens)
        if len(tokens) == 0:
            return [], []
        return tokens, self.vdb[vdb_key].SimilarByNames(tokens, exclude=exclude, topn=topn, duplicate=duplicate,
                                                        score_threshold=score_threshold)

    def match_relationships(self, name, prefix='Word'):
        if self.graph:
            node_name = f"{prefix}_{name}"
            return self.graph.run(
                "MATCH (source:" + node_name + " {name:\"" + name + "\"})-[relation:SIMILAR_TO]->(target) RETURN source,relation,target").data()
        return []

    def create_relationships(self, width=3, depth=0, similar_names=[], names_depth={}, relationships_edges=[],
                             vdb_key='Word_all', duplicate=3, create=0, score_threshold=0.0, exclude=[], **kwargs):

        name_first = [k for k, v in names_depth.items() if v == 0]  # list(words_depth)[0]
        try:
            not_ids = self.vdb[vdb_key].get_ids(list(names_depth))
            similar_next = self.vdb[vdb_key].SimilarByNames(similar_names,
                                                            exclude=exclude,  # +list(names_depth)
                                                            not_ids=not_ids,
                                                            topn=width,
                                                            duplicate=duplicate,
                                                            score_threshold=score_threshold)  # 分支会有合并,下层关系去重

            similar_names = list(set(y[0] for x in similar_next for y in x[1]))  # 迭代词组
            print("Depth:", depth, "Similar:", [x[0] for x in similar_next], "->", similar_next)
            names_depth.update({w: depth + 1 for w in similar_names if w not in names_depth})

            if self.graph and create:
                prefix = vdb_key.split('_')[0]
                node_name = f"{prefix}_{name_first}"

                for x in similar_next:
                    word_node = CreateNode(x[0], names_depth[x[0]], self.graph, node_name)  # w=x[0]
                    relationships = [Relationship(word_node, "SIMILAR_TO",
                                                  CreateNode(y[0], names_depth[y[0]], self.graph, node_name),
                                                  rank=i, similarity=float(y[1]))
                                     for i, y in enumerate(x[1])]

                    self.graph.create(Subgraph(relationships=relationships))  # 创建关系
                    relationships_edges += relationships
            else:
                relationships = [
                    {'source': x[0], 'target': y[0], 'relation': str(round(y[1], 5)), 'value': 1.0 - y[1]}
                    for x in similar_next for i, y in enumerate(x[1])]
                relationships_edges += relationships

        except KeyError:
            print(f"Data '{name_first, similar_names}' not in vocabulary.")
            similar_names = []

        return similar_names

    def create_relationship(self, width, depth, name, names_depth={}, vdb_key='Word_all', exclude_all=True, create=0,
                            score_threshold=0.0, exclude=[]):
        name_first = [k for k, v in names_depth.items() if v == 0]
        relationships = []
        try:
            exclude_names = list(names_depth) if exclude_all else [k for k in names_depth if
                                                                   names_depth[k] <= depth]  # 深度遍历时,排除上几层出现的
            not_ids = self.vdb[vdb_key].get_ids(exclude_names)
            similar_next = self.vdb[vdb_key].SimilarByName(name, exclude=exclude,
                                                           not_ids=not_ids,
                                                           topn=width,
                                                           score_threshold=score_threshold)
            similar_names = [y[0] for y in similar_next]
            if len(similar_names) == 0:
                return names_depth, relationships  # 返回递归

            print("Depth:", depth, "Similar:", name, "->", similar_next)
            new_depth = {w: depth + 1 for w in similar_names if w not in names_depth}
            names_depth.update(new_depth)  # 当前层去重

            if self.graph and create:
                prefix = vdb_key.split('_')[0]
                node_name = f"{prefix}_{name_first}"
                word_node = CreateNode(name, names_depth[name], self.graph, node_name)  # 创建节点
                relationships = [Relationship(word_node, "SIMILAR_TO",
                                              CreateNode(y[0], names_depth[y[0]], self.graph, word_node),
                                              similarity=y[1], rank=i)
                                 for i, y in enumerate(similar_next)]
                self.graph.create(Subgraph(relationships=relationships))  # 创建关系
            else:
                relationships = [
                    {'source': name, 'target': y[0], 'relation': str(round(y[1], 5)), 'value': 1.0 - y[1]}
                    for i, y in enumerate(similar_next)]

        except KeyError:
            print(f"Data '{name_first, name}' not in vocabulary.")
            # new_depth = {}

        return new_depth, relationships

    def create_relationships_breadth(self, name, vdb_key='Word_all', layers=[3, 3, 3], duplicate=3,
                                     max_calc=30, max_node=0, create=0, score_threshold=0.0, exclude=[], **kwargs):
        if vdb_key not in self.vdb:
            return {}, []

        names_depth = {name: 0}
        relationships_edges = []

        similar_names = [name]
        depth = 0
        count = 0

        for width in layers:  # while depth <= max_depth and len(names_depth) <= max_node
            if max_node > 0 and len(names_depth) + len(similar_names) * width > max_node:
                cut_len = (max_node - len(names_depth)) // width
                if cut_len <= 0:
                    break
                similar_names = similar_names[:cut_len]

            if max_calc > 0 and count + len(similar_names) > max_calc:  # calc_layers_total
                cut_len = (max_calc - count) // width
                if cut_len <= 0:
                    break
                similar_names = similar_names[:cut_len]

            count += len(similar_names)

            similar_names = self.create_relationships(width=width, depth=depth,
                                                      similar_names=similar_names,
                                                      names_depth=names_depth,
                                                      relationships_edges=relationships_edges,
                                                      vdb_key=vdb_key, duplicate=duplicate, create=create,
                                                      score_threshold=score_threshold, exclude=exclude, **kwargs)

            depth += 1

            if len(similar_names) == 0:
                break

        return names_depth, relationships_edges

    def create_relationships_depth(self, name, vdb_key='Word_all', layers=[3, 3, 3], max_calc=30, max_node=0,
                                   create=0, score_threshold=0.0, exclude_all=True, exclude=[],
                                   depth=0, names_depth={}, relationships_edges=[]):
        if vdb_key not in self.vdb:
            return {}, []

        if names_depth is None:
            names_depth = {}
        if relationships_edges is None:
            relationships_edges = []
        if depth == 0:
            relationships_edges.clear()
            names_depth.clear()
            names_depth[name] = 0

        if depth >= len(layers):
            return names_depth, relationships_edges

        if max_node > 0 and len(names_depth) + layers[depth] > max_node:
            return names_depth, relationships_edges

        if max_calc > 0 and len(names_depth) > 1:
            count = 1  # len(relationships_edges)
            for d, w in enumerate(layers):
                ndepth = [dn for dn in names_depth.values() if dn == d + 1]
                count += len(ndepth) // w
            if count >= max_calc:
                return names_depth, relationships_edges

        new_depth, relationships = self.create_relationship(width=layers[depth], depth=depth,
                                                            name=name, names_depth=names_depth,
                                                            vdb_key=vdb_key, exclude_all=exclude_all,
                                                            create=create, score_threshold=score_threshold,
                                                            exclude=exclude)

        relationships_edges += relationships

        # 递归创建更深层次的关系,只找新的
        for w in new_depth:
            self.create_relationships_depth(w, vdb_key, layers, max_calc, max_node,
                                            create, score_threshold, exclude_all, exclude,
                                            depth + 1, names_depth, relationships_edges)

        # return [name] + [
        #     self.create_relationships_depth(w, vdb_key, names_depth, relationships_edges,
        #                                     width, depth + 1, max_depth, max_node,
        #                                     create, score_threshold, exclude_all)
        #     if w in new_depth else [w] for w in similar_words]

        return names_depth, relationships_edges

    # functions
    def SimilarRelationships(self, name, vdb_key='Word_all', width=3, max_depth=3, layers=[], batch=True,
                             duplicate=3, create=0, max_calc=30, max_node=0, score_threshold=0.0, exclude=[], **kwargs):

        if not layers:
            layers = [width] * max_depth

        if batch:
            return self.create_relationships_breadth(name, vdb_key, layers, duplicate=duplicate,
                                                     max_calc=max_calc, max_node=max_node, create=create,
                                                     score_threshold=score_threshold, exclude=exclude, **kwargs)
        else:
            return self.create_relationships_depth(name, vdb_key, layers,
                                                   max_calc=max_calc, max_node=max_node, create=create,
                                                   score_threshold=score_threshold, exclude=exclude,
                                                   exclude_all=(duplicate > 0))

    def SimulationNodes(self, names_depth, relationships_edges, vdb_key='Word_all', key_radius=''):
        if vdb_key not in self.vdb:  # self.vdb.get(vdb_key, None)
            key_idx = {w: i for i, w in enumerate(names_depth)}  # w: depth-> w,idx
            nodes = [{"name": w, "id": i + 1, 'depth': names_depth[w]} for w, i in key_idx.items()]
            edges = relationships_edges.copy()
            for r in edges:
                r['source'] = nodes[key_idx.get(r['source'])]
                r['target'] = nodes[key_idx.get(r['target'])]
            return {"nodes": nodes, "edges": edges}

        ps = vdb_key.split('_')
        if len(ps) and key_radius:
            key_prefix = ps[0].lower()  # 'word'
            i_payloads = self.get_payload(vdb_key, list(names_depth), fields=[key_prefix, key_radius])
            radius = scale_to_range([p.get(key_radius, 1) for i, p in i_payloads], 10, 30)
            nodes = [{"id": i, "name": p[key_prefix], "depth": names_depth[p[key_prefix]], "radius": radius[idx]}
                     for idx, (i, p) in enumerate(i_payloads)]  # zip(*i_payloads)
        else:
            name_ids = self.get_ids(vdb_key, list(names_depth))
            nodes = [{"id": i, "name": n, "depth": names_depth[n], "radius": 20} for n, i in name_ids.items()]

        key_nodes = {n['name']: n for n in nodes}  # {p['word']: i for i, p in i_payloads}
        edges = relationships_edges.copy()
        for r in edges:
            r['source'] = key_nodes.get(r['source'])
            r['target'] = key_nodes.get(r['target'])

        return {"nodes": nodes, "edges": edges}


if __name__ == '__main__':
    client = QdrantClient(host="10.10.10.5", grpc_port=6334, prefer_grpc=True)
    match_first = field_match("行业", '金融')
    vdbs = VDBSimilar(client, collection_name='专利_w2v', key_name='word', match_first=match_first)
    res = vdbs.SimilarByNames(['计算机', '训练方法', '电子设备'], exclude=[], topn=3, duplicate=3, score_threshold=0.3)
    print(res)
    res = vdbs.SimilarByName('领域', exclude=[], topn=10, score_threshold=0.5)
    print(res)

    vdr = VDBRelationships()
    vdr.load(None, client, collection_name='专利_w2v', prefix='Word',
             field_key='行业', match_values=['金融', '传统制造'])
    words_depth, relationships_edges = vdr.create_relationships_breadth('计算机', 'Word_金融', [2, 3, 4],
                                                                        duplicate=3,
                                                                        create=0, max_node=400,
                                                                        score_threshold=0.0)
    print(words_depth, '\n', relationships_edges)

    words_depth, relationships_edges = vdr.create_relationships_depth('计算机', 'Word_金融',
                                                                      [2, 3, 3],
                                                                      max_node=500, create=0, score_threshold=0.0,
                                                                      exclude_all=False)

    print(words_depth, '\n', relationships_edges, '\n')

    words_depth, relationships_edges = vdr.SimilarRelationships('计算机', 'Word_金融', width=3, max_depth=3, layers=[],
                                                                batch=False)
    print(words_depth, '\n', relationships_edges)

    name = '电子装置'
    next = vdbs.SimilarByName(name, exclude=words_depth.keys(), not_ids=[], topn=10, score_threshold=0.0)
    print(next)

    new_depth, relationships = vdr.create_relationship(1, words_depth.get(name, 1), name, names_depth=words_depth,
                                                       vdb_key='Word_金融', exclude_all=True, create=0,
                                                       score_threshold=0.0, exclude=[])
    print(new_depth, relationships)

    # collection_name='专利_w2v_188_37'
    # client.create_snapshot(collection_name=collection_name)
    # snapshots_name=client.list_snapshots(collection_name)[0].name
    # url=f'http://10.10.10.5:6333/collections/{collection_name}/snapshots/{snapshots_name}'
    # local_filename=f'E:\Downloads\{snapshots_name}'
    # download_file(url, local_filename)
    # recover_snapshots_upload('47.110.156.41', collection_name, local_filename)
    # client.delete_snapshot(
    # collection_name=collection_name, snapshot_name=snapshots_name
    # )
