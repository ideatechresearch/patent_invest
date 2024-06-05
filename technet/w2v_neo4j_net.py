# coding=utf-8
from py2neo import Graph, Node, Relationship, Subgraph
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import time
import os, re


# 创建节点
def create_word_node(word, depth, graph, wid, **kwargs):
    nodes = graph.nodes.match(f"Word_{wid}")
    node = nodes.where(name=word).first()
    if not node:
        node = Node(f"Word_{wid}", name=word, depth=depth, nid=len(nodes), **kwargs)
        graph.create(node)
    return node


# 创建关系
def similarity_relationship(word1, word2, depth1, depth2, similarity, graph, wid=0, **kwargs):
    word1_node = create_word_node(word1, depth1, graph, wid)
    word2_node = create_word_node(word2, depth2, graph, wid)
    relationship = Relationship(word1_node, "SIMILAR_TO", word2_node, similarity=similarity, **kwargs)
    return relationship  # graph.create(relationship)


# 排除某些词
def similar_by_aword(word, model, topn=10, exclude=[]):
    eidx = [model.wv.key_to_index[w] for w in exclude + [word]]
    return list(pd.Series(
        cosine_similarity(model.wv[word].reshape(1, -1), np.delete(model.wv.vectors, eidx, axis=0)).reshape(-1),
        index=np.delete(model.wv.index_to_key, eidx), name=word
    ).sort_values(ascending=False)[:topn].items())  # .to_dict()


def similar_by_words(words, model, topn=10, exclude=[]):
    eidx = [model.wv.key_to_index[w] for w in words + exclude]
    sim = pd.DataFrame(cosine_similarity(model.wv[words], np.delete(model.wv.vectors, eidx, axis=0)).T,
                       index=np.delete(model.wv.index_to_key, eidx), columns=words)
    return [(w, list(sim[w].sort_values(ascending=False)[:topn].items())) for w in words]


def create_word_similarity_relationship(word, model, graph=None, words_depth=None, width=3, depth=0, max_depth=5, wid=0,
                                        max_node=1000, exclude_all=False):
    if words_depth is None:
        words_depth = {}
    if depth == 0:
        words_depth[word] = 0
    if depth > max_depth or len(words_depth) > max_node:
        return [word]
    else:
        try:
            exclude = list(words_depth) if exclude_all else [w for w in words_depth if
                                                             words_depth[w] <= depth]  # 排除上几层出现的
            if len(exclude) > 1:
                similar_words = similar_by_aword(word, model, width, exclude)
            else:
                similar_words = model.wv.most_similar(word, topn=width)

            if len(similar_words) > 0:
                similar_words_depth = {w[0]: depth + 1 for w in similar_words if
                                       w[0] not in words_depth}  # 当前层去重相似度最高的词
                words_depth.update(similar_words_depth)
                print("Depth:", depth, "Similar:", word, "->", similar_words)

                if graph:
                    word_node = create_word_node(word, words_depth[word], graph, wid)  # 创建节点
                    relationships = [
                        Relationship(word_node, "SIMILAR_TO", create_word_node(w[0], words_depth[w[0]], graph, wid),
                                     similarity=w[1], rank=i)
                        for i, w in enumerate(similar_words)]
                    print(relationships)
                    graph.create(Subgraph(relationships=relationships))  # 创建关系

                return [word] + [
                    create_word_similarity_relationship(w[0], model, graph, words_depth, width, depth + 1, max_depth,
                                                        wid, exclude_all)
                    if w[0] in similar_words_depth else [w[0]] for w in similar_words]
            else:
                return [word]  # 返回递归
        except KeyError:
            print(f"Word '{word}' not in vocabulary.")
            return [word]


def SimilarByWords(words, model, topn=10, exclude=[], duplicate=0):
    eidx = [model.wv.key_to_index[w] for w in words + exclude]
    sim = pd.DataFrame(cosine_similarity(model.wv[words], np.delete(model.wv.vectors, eidx, axis=0)).T,
                       index=np.delete(model.wv.index_to_key, eidx), columns=words)

    if duplicate == 0:
        return [(w, list(sim[w].sort_values(ascending=False)[:topn].items())) for w in words]  # 不对相同节点关系去重,保存排序顺序

    mx = len(words)
    similar_next = {}
    if duplicate == 1:  # 由上一节点顺序优先选择
        for w in words:
            y = list(sim[w].sort_values(ascending=False)[:topn * mx].items())
            y2 = [_[0] for x in similar_next.values() for _ in x]
            similar_next[w] = [y3 for y3 in y if y3[0] not in y2][:topn]

    if duplicate == 2:  # 各自排序后,由下一节点各自选最近的上节点,最后由上节点各自匹配重复关系
        similar_with = {}  # {y:(x,v)}
        for w in words:
            y = list(sim[w].sort_values(ascending=False)[:topn * mx].items())  # 最多重复目标数量
            for z in y:
                if z[0] not in similar_with or z[1] > similar_with[z[0]][1]:  # 对关系去重选最大的,反向选择
                    similar_with[z[0]] = (w, z[1])

        for y, x in similar_with.items():
            similar_next[x[0]] = sorted(similar_next.get(x[0], []) + [(y, x[1])], key=lambda z: z[1], reverse=True)[
                                 :topn]

    if duplicate == 3:  # 整体排序,关系值大的先选,相对来说这个计算较慢
        similar_sort = sorted(enumerate(sim.values.reshape(-1)), key=lambda x: x[1], reverse=True)  # 所有数据值排序(i,value)
        pos = []  # 储存行列位置数据,统计数量与去重[(,),]
        for i, v in similar_sort:
            y = [p[0] for p in pos]
            if len(y) >= topn * mx:
                break
            ix = i % mx
            iy = i // mx
            x = [p[1] for p in pos if p[1] == ix]
            if len(x) < topn and iy not in y:
                pos.append((iy, ix))
                similar_next[sim.columns[ix]] = similar_next.get(sim.columns[ix], []) + [(sim.index[iy], v)]

    return [(w, similar_next[w]) for w in words if w in similar_next]  # 恢复顺序，list(similar_next.items())


# 创建节点
def CreateNode(word, depth, graph, node_name='Word', **kwargs):
    nodes = graph.nodes.match(node_name)
    node = nodes.where(name=word).first()  # Node("Word", name=word,id=mid)
    if not node:
        node = Node(node_name, name=word, depth=depth, nid=len(nodes), **kwargs)
        graph.create(node)
    return node


def CreateWordSimilarityRelationship(word, model, graph=None, width=3, max_depth=3, wid=0, duplicate=0,
                                     stop_words=[]):  # 层级搜索扩散
    words_depth = {}
    similar_words = []
    depth = 0
    while depth <= max_depth:
        try:
            if depth == 0:
                words_depth[word] = 0
                similar_words_next = [(word, model.wv.most_similar(word, topn=width))]
            else:
                similar_words_next = SimilarByWords(similar_words, model, topn=width,
                                                    exclude=list(words_depth) + stop_words,
                                                    duplicate=duplicate)  # 分支会有合并,下层关系去重,

            similar_words = list(set(y[0] for x in similar_words_next for y in x[1]))  # 迭代词组
            print("Depth:", depth, "Similar:", [x[0] for x in similar_words_next], "->", similar_words_next)
            depth += 1
            words_depth.update({w: depth for w in similar_words if w not in words_depth})

            if not graph:
                continue

            node_name = f"Word_{wid}"
            for x in similar_words_next:
                word_node = CreateNode(x[0], words_depth[x[0]], graph, node_name,
                                       similar=float(model.wv.similarity(word, x[0])))  # w=x[0]
                relationships = [Relationship(word_node, "SIMILAR_TO",
                                              CreateNode(y[0], words_depth[y[0]], graph, node_name,
                                                         similar=float(model.wv.similarity(word, y[0]))), rank=i,
                                              similarity=float(y[1]))
                                 for i, y in enumerate(x[1])]
                print(relationships)
                graph.create(Subgraph(relationships=relationships))  # 创建关系

        except KeyError:
            print(f"Word '{word, similar_words}' not in vocabulary.")
            break

    return words_depth


class WordRelationships:
    graph = None
    wo = None
    wo_sg = None

    def load(self, graph, wo, wo_sg=None):
        self.graph = graph
        self.wo = wo
        self.wo_sg = wo_sg
        print(wo.wv.most_similar("数据"))

    def similar(self, words, sg=0, topn=10, exclude=[], duplicate=0):
        tokens = re.split(r'[^\w\s]| ', words)
        tokens = [token.strip().lower() for token in tokens]
        model = self.wo_sg if sg else self.wo
        if not model:
            return tokens, []
        tokens = [token for token in tokens if token in model.wv.key_to_index]
        if len(tokens) == 0:
            return [], []
        if duplicate or len(exclude) > 0:
            return tokens, SimilarByWords(tokens, model, topn, exclude, duplicate)
            # similar_by_aword(word, self.wo_sg if sg else self.wo, topn, exclude)
        return tokens, model.wv.most_similar(tokens, topn=topn)

    def match_relationships(self, word):
        if self.graph:
            wid = f"Word_{word}"
            return self.graph.run(
                "MATCH (source:" + wid + " {name:\"" + word + "\"})-[relation:SIMILAR_TO]->(target) RETURN source,relation,target").data()
            # self.graph.run("MATCH (n1:person {name:\"" + entity + "\"})-[rel:" + relation + "]->(n2) RETURN n1,rel,n2").data()
            # self.graph.run("MATCH (n1)- [rel] -> (n2:major {name:\"" + entity1 + "\"}) RETURN n1,rel,n2").data()
            nodes = self.graph.nodes.match(wid)  # nodes.all())
            if len(nodes) > 0:
                return self.graph.relationships.match({self.graph.nodes.match(wid, name=word).first()},
                                                      r_type='SIMILAR_TO').all()

        return []

    def create_relationships(self, word, width=3, max_depth=3, duplicate=3, sg=0, cyc=0, create=0, max_node=400,
                             exclude=[]):
        max_depth -= 1
        model = self.wo_sg if sg else self.wo
        if not model:
            return {}, []

        if cyc:
            words_depth = {}
            lrs = create_word_similarity_relationship(word, model, self.graph, words_depth, width, 0, max_depth, word,
                                                      max_node, exclude_all=duplicate)
            print("cyc --->", lrs)
            return words_depth, lrs

        return self.CreateWordSimilarityRelationship(word, sg, width, max_depth, duplicate,
                                                     create, max_node, stop_words=exclude)
        # CreateWordSimilarityRelationship(word, model, self.graph,
        #                                  width, max_depth, word, duplicate, exclude)

        # rid = f'{word}_{uid}'
        # if cyc:
        #     lrs = self.create_word_similarity_relationship(word, rid, sg,
        #                                                    words_depth, width, 0, max_depth, exclude_all=duplicate)
        #     print("cyc --->", lrs)
        # else:
        #     words_depth = self.CreateWordSimilarityRelationship(word, rid, sg, width, max_depth, duplicate)
        #
        #     model = self.wo_sg if sg else self.wo

    def CreateWordSimilarityRelationship(self, word, sg=0, width=3, max_depth=3, duplicate=0, create=0,
                                         max_node=1000, stop_words=[]):  # 层级搜索扩散
        words_depth = {}
        relationships_edges = []
        similar_words = []
        depth = 0
        model = self.wo_sg if sg else self.wo
        while depth <= max_depth and len(words_depth) <= max_node:
            try:
                if depth == 0:
                    words_depth[word] = 0
                    similar_words_next = [(word, model.wv.most_similar(word, topn=width))]
                else:
                    similar_words_next = SimilarByWords(similar_words, model, topn=width,
                                                        exclude=list(words_depth) + stop_words,
                                                        duplicate=duplicate)  # 分支会有合并,下层关系去重,

                similar_words = list(set(y[0] for x in similar_words_next for y in x[1]))  # 迭代词组
                print("Depth:", depth, "Similar:", [x[0] for x in similar_words_next], "->", similar_words_next)
                depth += 1
                words_depth.update({w: depth for w in similar_words if w not in words_depth})

                if not (self.graph and create):
                    relationships = [
                        {'source': x[0], 'target': y[0], 'relation': str(round(y[1], 5)), 'value': 1.0 / (y[1] + 1)}
                        for x in similar_words_next for i, y in enumerate(x[1])]
                    relationships_edges += relationships
                else:
                    node_name = f"Word_{word}"
                    for x in similar_words_next:
                        word_node = CreateNode(x[0], words_depth[x[0]], self.graph, node_name,
                                               similar=float(model.wv.similarity(word, x[0])))  # w=x[0]
                        relationships = [Relationship(word_node, "SIMILAR_TO",
                                                      CreateNode(y[0], words_depth[y[0]], self.graph, node_name,
                                                                 similar=float(model.wv.similarity(word, y[0]))),
                                                      rank=i, similarity=float(y[1]))
                                         for i, y in enumerate(x[1])]

                        self.graph.create(Subgraph(relationships=relationships))  # 创建关系
                        relationships_edges += relationships

            except KeyError:
                print(f"Word '{word, similar_words}' not in vocabulary.")
                break

        return words_depth, relationships_edges
    #
    # def create_word_similarity_relationship(self, word, rid, sg=0, words_depth=None, width=3, depth=0, max_depth=5,
    #                                         exclude_all=False):
    #     model = self.wo_sg if sg else self.wo
    #     if words_depth is None:
    #         words_depth = {}
    #     if depth == 0:
    #         words_depth[word] = 0
    #     if depth > max_depth:
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
    #                 if self.graph:
    #                     word_node = CreateNode(word, words_depth[word], self.graph)  # 创建节点
    #                     relationships = [
    #                         Relationship(word_node, "SIMILAR_TO", CreateNode(w[0], words_depth[w[0]], self.graph),
    #                                      similarity=w[1], rank=i, rid=rid)
    #                         for i, w in enumerate(similar_words)]
    #                     print(relationships)
    #                     self.graph.create(Subgraph(relationships=relationships))  # 创建关系
    #
    #                 return [word] + [
    #                     self.create_word_similarity_relationship(w[0], sg, rid, words_depth, width, depth + 1,
    #                                                              max_depth, exclude_all)
    #                     if w[0] in similar_words_depth else [w[0]] for w in similar_words]
    #             else:
    #                 return [word]  # 返回递归
    #         except KeyError:
    #             print(f"Word '{word}' not in vocabulary.")
    #             return [word]


if __name__ == '__main__':
    # graph= Graph("bolt://127.0.0.1:7687", auth=("neo4j", "77"))
    graph= Graph("bolt://10.10.10.5:7687", auth=("d", "77"))
    DATA_DIR = "DATA"

    wo = Word2Vec.load(os.path.join(DATA_DIR, "patent_w2v.model"))
    wo_sg = Word2Vec.load(os.path.join(DATA_DIR, "patent_w2v_sg.model"))
    width = 3
    max_depth = 3

    print(wo.wv.most_similar("数据"))

    while True:
        word = input('请输入关键词：')
        word = word.strip().lower()
        if not word:
            continue
        if word == 'none':
            break

        nodes = graph.nodes.match(f"Word_{word}")
        if len(nodes) > 0:
            print(graph.relationships.match({graph.nodes.match(f"Word_{word}", name=word).first()},
                                            r_type='SIMILAR_TO').all(), '\n', nodes.all())
            continue

        if word not in wo.wv.key_to_index:
            print('找不到关键字：', word)
            continue

        for i in range(4):
            words_depth = CreateWordSimilarityRelationship(word, wo, graph, width, max_depth, word + f'_{i}',
                                                           i)  # Cbow
            print(word, "--->", words_depth)

            if len(words_depth) > 1:
                words_depth = CreateWordSimilarityRelationship(word, wo_sg, graph, width, max_depth, word + f'_{i}_sg',
                                                               i)  # skip-gram
                print(word, " sg --->", words_depth)

        for i in range(2):
            dix = {}
            create_word_similarity_relationship(word, wo, graph, dix, width, 0, max_depth, word + f'_cyc_{i}',
                                                exclude_all=i)
            print("cyc --->", dix)
            create_word_similarity_relationship(word, wo_sg, graph, dix, width, 0, max_depth, word + f'_cyc_{i}_sg',
                                                exclude_all=i)
            print("cyc sg --->", dix)

        time.sleep(1)
