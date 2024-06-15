from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, SearchRequest, SearchParams, VectorParams, \
    PointStruct, CollectionStatus, Distance, Range
from py2neo import Graph, Node, Relationship, Subgraph
import numpy as np
import asyncio


def cosine_sim(A, B):
    dot_product = np.dot(A, B)
    similarity = dot_product / (np.linalg.norm(A) * np.linalg.norm(B))
    return similarity


def most_similar_by_name(name, collection_name, client, mutch=[], exclude=[], topn=10, score_threshold=0.5):
    match_name = FieldCondition(key='word', match=MatchValue(value=name, ), )
    match_not_name = [FieldCondition(key='word', match=MatchValue(value=w), ) for w in exclude]
    scroll_filter = Filter(must=mutch + [match_name])
    scroll_result = client.scroll(collection_name=collection_name,
                                  scroll_filter=scroll_filter,
                                  with_vectors=True,
                                  with_payload=True,
                                  limit=1)  # 只获取一个匹配的结果

    query_vector = scroll_result[0][0].vector

    query_filter = Filter(must=mutch, must_not=[match_name] + match_not_name)  # 缩小查询范围
    search_hit = client.search(collection_name=collection_name,
                               query_vector=query_vector,  # tolist()
                               query_filter=query_filter,
                               limit=topn,
                               score_threshold=score_threshold,
                               search_params=SearchParams(exact=True, ),
                               # Turns on the exact search mode
                               )

    return [(p.payload['word'], p.score) for p in search_hit]


def most_similar_by_ids(ids, collection_name, client, mutch=[], exclude=[], topn=10, score_threshold=0.0):
    id_record = client.retrieve(collection_name=collection_name, ids=ids, with_vectors=True)

    names = [p.payload['word'] for p in id_record]
    match_not_name = [FieldCondition(key='word', match=MatchValue(value=w), ) for w in names + exclude]
    query_filter = Filter(must=mutch, must_not=match_not_name)  # 缩小查询范围

    search_queries = [SearchRequest(vector=p.vector, filter=query_filter, limit=topn, score_threshold=score_threshold,
                                    with_payload=True)
                      for p in id_record]

    search_hit = client.search_batch(collection_name=collection_name, requests=search_queries)  # ScoredPoint
    return [(id, name, [(p.payload['word'], p.score) for p in hit]) for id, name, hit in
            zip(ids, names, search_hit)]  # [:topn]


def name_to_ids(names, collection_name, client, mutch=[], key_name='word'):
    shoulds = [FieldCondition(key=key_name, match=MatchValue(value=w)) for w in names]
    scroll_filter = Filter(must=mutch, should=shoulds)

    scroll_result = client.scroll(collection_name=collection_name,
                                  scroll_filter=scroll_filter,
                                  with_payload=True,
                                  # with_vectors=True,
                                  limit=len(names))

    return {i.payload[key_name]: i.id for i in scroll_result[0]}


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


def SimilarByIds(ids, collection_name, client, mutch=[], exclude=[], topn=10, duplicate=0, score_threshold=0):
    id_record = client.retrieve(collection_name=collection_name, ids=ids, with_vectors=True)

    names = [p.payload['word'] for p in id_record]
    match_not_name = [FieldCondition(key='word', match=MatchValue(value=w), ) for w in names + exclude]
    query_filter = Filter(must=mutch, must_not=match_not_name)  # 缩小查询范围

    search_queries = [
        SearchRequest(vector=p.vector, filter=query_filter, limit=topn * len(names), score_threshold=score_threshold,
                      with_payload=True)
        for p in id_record]

    search_hit = client.search_batch(collection_name=collection_name, requests=search_queries)  # ScoredPoint

    return rerank_similar_by_search(names, search_hit, topn=topn, duplicate=duplicate, key_name='word')


class VDB_Similar:
    client = None
    collection_name = ''
    key_name = 'word'
    match_first = []
    name_ids = {}  # unique,记忆

    def __init__(self, client, collection_name, key_name='word', match_first=[]):
        self.client = client
        self.collection_name = collection_name
        self.key_name = key_name  # .lower()
        self.match_first = match_first

    def get_ids(self, names):
        finds = set(names) - self.name_ids.keys()
        if len(finds):  # 更新未记录id
            result = name_to_ids(finds, self.collection_name, self.client, self.match_first, self.key_name)
            if result:
                self.name_ids.update(result)

        return [self.name_ids[n] for n in names if n in self.name_ids]

    def get_menory(self):
        return self.name_ids

    def get_vec(self, names):
        ids = self.get_ids(names)
        id_record = self.client.retrieve(collection_name=self.collection_name, ids=ids, with_vectors=True)
        return {p.payload[self.key_name]: p.vector for p in id_record}

    def get_payload(self, names):
        ids = self.get_ids(names)
        id_record = self.client.retrieve(collection_name=self.collection_name, ids=ids, with_payload=True)
        return [p.payload for p in id_record]  # [{},]

    def SimilarByNames(self, names, exclude=[], topn=10, duplicate=0, score_threshold=0.0):
        ids = self.get_ids(names)
        id_record = self.client.retrieve(collection_name=self.collection_name, ids=ids, with_vectors=True)

        names = [p.payload[self.key_name] for p in id_record]
        match_not_name = [FieldCondition(key=self.key_name, match=MatchValue(value=w), ) for w in names + exclude]
        query_filter = Filter(must=self.match_first, must_not=match_not_name)  # 缩小查询范围

        search_queries = [
            SearchRequest(vector=p.vector, filter=query_filter, limit=topn * len(names), with_payload=True,
                          score_threshold=score_threshold)
            for p in id_record]

        search_hit = self.client.search_batch(collection_name=self.collection_name,
                                              requests=search_queries)  # ScoredPoint
        self.name_ids.update({p.payload[self.key_name]: p.id for hit in search_hit for p in hit})

        return rerank_similar_by_search(names, search_hit, topn=topn, duplicate=duplicate, key_name=self.key_name)


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

    def load(self, graph, client, collection_name='专利_w2v', prefix='Word', match_hy=['all']):
        self.graph = graph
        self.append(client, collection_name=collection_name, prefix=prefix, match_hy=match_hy)

    def append(self, client, collection_name='专利_w2v', prefix='Word', match_hy=['all']):
        for k in match_hy:
            match = [FieldCondition(key="行业", match=MatchValue(value=k, ))] if k != 'all' else []
            count = client.count(collection_name=collection_name,
                                 count_filter=Filter(must=match), exact=True)

            vdb_key = f"{prefix}_{k}"
            print(vdb_key, ':', count)
            self.vdb[vdb_key] = VDB_Similar(client, collection_name=collection_name, key_name=prefix.lower(),
                                            match_first=match)

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

    def create_relationships(self, width=3, depth=0, similar_words=[], names_depth={}, relationships_edges=[],
                             vdb_key='Word_all', duplicate=3, create=0, score_threshold=0.0, exclude=[], **kwargs):

        name_first = [k for k, v in names_depth.items() if v == 0]  # list(words_depth)[0]
        try:
            similar_words_next = self.vdb[vdb_key].SimilarByNames(similar_words,
                                                                  exclude=list(names_depth) + exclude,
                                                                  topn=width,
                                                                  duplicate=duplicate,
                                                                  score_threshold=score_threshold)  # 分支会有合并,下层关系去重

            similar_words = list(set(y[0] for x in similar_words_next for y in x[1]))  # 迭代词组
            print("Depth:", depth, "Similar:", [x[0] for x in similar_words_next], "->", similar_words_next)
            names_depth.update({w: depth + 1 for w in similar_words if w not in names_depth})

            if not (self.graph and create):
                relationships = [
                    {'source': x[0], 'target': y[0], 'relation': str(round(y[1], 5)), 'value': 1.0 - y[1]}
                    for x in similar_words_next for i, y in enumerate(x[1])]
                relationships_edges += relationships
            else:
                prefix = vdb_key.split('_')[0]
                node_name = f"{prefix}_{name_first}"

                for x in similar_words_next:
                    word_node = CreateNode(x[0], names_depth[x[0]], self.graph, node_name)  # w=x[0]
                    relationships = [Relationship(word_node, "SIMILAR_TO",
                                                  CreateNode(y[0], names_depth[y[0]], self.graph, node_name),
                                                  rank=i, similarity=float(y[1]))
                                     for i, y in enumerate(x[1])]

                    self.graph.create(Subgraph(relationships=relationships))  # 创建关系
                    relationships_edges += relationships

        except KeyError:
            print(f"Data '{name_first, similar_words}' not in vocabulary.")
            similar_words = []

        return similar_words

    def create_relationships_layer(self, name, vdb_key='Word_all', layers=[3, 3, 3], duplicate=3, create=0,
                                   max_node=400, score_threshold=0.0, exclude=[], **kwargs):
        if vdb_key not in self.vdb:
            return {}, []

        names_depth = {name: 0}
        relationships_edges = []

        similar_words = [name]
        depth = 0

        for width in layers:  # while depth <= max_depth and len(names_depth) <= max_node
            if len(names_depth) > max_node:
                break

            similar_words = self.create_relationships(width=width, depth=depth,
                                                      similar_words=similar_words,
                                                      names_depth=names_depth,
                                                      relationships_edges=relationships_edges,
                                                      vdb_key=vdb_key, duplicate=duplicate, create=create,
                                                      score_threshold=score_threshold, exclude=exclude, **kwargs)

            depth += 1
            if len(similar_words) == 0:
                break

        return names_depth, relationships_edges

    def create_relationships_layer_wd(self, name, vdb_key='Word_all', width=3, max_depth=3, duplicate=3, create=0,
                                      max_node=400, score_threshold=0.0, exclude=[], **kwargs):

        layers = [width] * max_depth
        return self.create_relationships_layer(name, vdb_key, layers, duplicate=duplicate, create=create,
                                               max_node=max_node, score_threshold=score_threshold,
                                               exclude=exclude,**kwargs)


# def create_relationships_depth(word, model, graph=None, words_depth=None, width=3, depth=0, max_depth=5, wid=0,
#                                         max_node=1000, exclude_all=False):
#     if words_depth is None:
#         words_depth = {}
#     if depth == 0:
#         words_depth[word] = 0
#     if depth > max_depth or len(words_depth) > max_node:
#         return [word]
#     else:
#         try:
#             exclude = list(words_depth) if exclude_all else [w for w in words_depth if
#                                                              words_depth[w] <= depth]  # 排除上几层出现的
#             if len(exclude) > 1:
#                 similar_words = similar_by_aword(word, model, width, exclude)
#             else:
#                 similar_words = model.wv.most_similar(word, topn=width)
#
#             if len(similar_words) > 0:
#                 similar_words_depth = {w[0]: depth + 1 for w in similar_words if
#                                        w[0] not in words_depth}  # 当前层去重相似度最高的词
#                 words_depth.update(similar_words_depth)
#                 print("Depth:", depth, "Similar:", word, "->", similar_words)
#
#                 if graph:
#                     word_node = create_word_node(word, words_depth[word], graph, wid)  # 创建节点
#                     relationships = [
#                         Relationship(word_node, "SIMILAR_TO", create_word_node(w[0], words_depth[w[0]], graph, wid),
#                                      similarity=w[1], rank=i)
#                         for i, w in enumerate(similar_words)]
#                     print(relationships)
#                     graph.create(Subgraph(relationships=relationships))  # 创建关系
#
#                 return [word] + [
#                     create_word_similarity_relationship(w[0], model, graph, words_depth, width, depth + 1, max_depth,
#                                                         wid, exclude_all)
#                     if w[0] in similar_words_depth else [w[0]] for w in similar_words]
#             else:
#                 return [word]  # 返回递归
#         except KeyError:
#             print(f"Word '{word}' not in vocabulary.")
#             return [word]

if __name__ == '__main__':
    client = QdrantClient(host="10.10.10.5", grpc_port=6334, prefer_grpc=True)
    match_first = [FieldCondition(key="行业", match=MatchValue(value=k, ), ) for k in ['金融']]
    vdbs = VDB_Similar(client, collection_name='专利_w2v', key_name='word', match_first=match_first)
    res = vdbs.SimilarByNames(['计算机', '训练方法', '电子设备'], exclude=[], topn=3, duplicate=3, score_threshold=0.3)
    print(res)

    vdr = VDBRelationships()
    vdr.load(None, client, collection_name='专利_w2v', prefix='Word', match_hy=['金融', '传统制造'])
    words_depth, relationships_edges = vdr.create_relationships_layer('计算机', 'Word_金融', [2, 3, 4],
                                                                      duplicate=3,
                                                                      create=0, max_node=400,
                                                                      score_threshold=0.0)

    print(words_depth, '\n', relationships_edges)

    words_depth, relationships_edges = vdr.create_relationships_layer_wd('计算机', 'Word_金融', width=3, max_depth=3)
    print(words_depth, '\n', relationships_edges)
