from py2neo import Graph, Node, Relationship, Subgraph
import pandas as pd
import re


# 创建节点
def CreateNode(word, depth, graph, node_name='Word', **kwargs):
    nodes = graph.nodes.match(node_name)
    node = nodes.where(name=word).first()  # Node("Word", name=word,id=mid)
    if not node:
        node = Node(node_name, name=word, depth=depth, nid=len(nodes), **kwargs)
        graph.create(node)
    return node


def SimilarByCol(col, data, topn=10, exclude=[], duplicate=0):
    sim = data.loc[~data.index.isin(col + exclude), col]
    if duplicate == 0:
        return [(co, list(sim[co].sort_values(ascending=False)[:topn].items())) for co in col]  # 不对相同节点关系去重,保存排序顺序

    similar_next = {}  # 数据格式:{co:[(next,val)*topn]}
    if duplicate == 1:  # 由上一节点顺序优先选择
        for co in col:
            y = list(sim[co].sort_values(ascending=False)[:topn * len(col)].items())
            y2 = [_[0] for x in similar_next.values() for _ in x]
            similar_next[co] = [y3 for y3 in y if y3[0] not in y2][:topn]

    if duplicate == 2:  # 各自排序后,由下一节点各自选最近的上节点,最后由上节点各自匹配重复关系
        similar_with = {}  # {y:(x,v)}
        for co in col:
            y = list(sim[co].sort_values(ascending=False)[:topn * len(col)].items())  # 最多重复目标数量:(n*m)*m
            for z in y:
                if z[0] not in similar_with or z[1] > similar_with[z[0]][1]:  # 对关系去重选最大的,反向选择
                    similar_with[z[0]] = (co, z[1])

        for y, x in similar_with.items():
            similar_next[x[0]] = sorted(similar_next.get(x[0], []) + [(y, x[1])], key=lambda z: z[1], reverse=True)[
                                 :topn]

    if duplicate == 3:
        # [(row.Index, col, sim.at[row.Index, col]) for row in sim.itertuples() for col in sim.columns]
        # similar_sort=sorted([(i, j, sim.iloc[i, j]) for i in range(sim.shape[0]) for j in range(sim.shape[1])],
        #         key=lambda x:x[2],reverse=True)#所有数据值排序(row,col,value)
        similar_sort = sorted(enumerate(sim.values.reshape(-1)), key=lambda x: x[1], reverse=True)  # 所有数据值排序(i,value)
        pos = []  # 储存行列位置数据,统计数量与去重
        mx = len(col)
        for i, v in similar_sort:
            y = [p[0] for p in pos]
            if len(y) >= topn * mx:
                break
            ix = i % mx
            iy = i // mx
            z = [p[1] for p in pos if p[1] == ix]
            if len(z) < topn and iy not in y:
                pos.append((iy, ix))  # (x[0],x[1])
                similar_next[sim.columns[ix]] = similar_next.get(sim.columns[ix], []) + [(sim.index[iy], v)]

    return [(co, similar_next[co]) for co in col if co in similar_next]  # 恢复顺序 list(similar_next.items())
    # 返回数据格式:[(co:[(next,value)*topn])*len(col)],皆有排序


def CreateCoSimilarityRelationship(co, data, graph=None, width=3, max_depth=3, wid=0, duplicate=0):  # 层级扩散
    words_depth = {co: 0}
    similar_words = [co]
    depth = 0
    while depth <= max_depth:
        try:
            similar_words_next = SimilarByCol(similar_words, data, topn=width, exclude=list(words_depth),
                                              duplicate=duplicate)  # 分支会有合并,下层关系去重
            similar_words = list(set(y[0] for x in similar_words_next for y in x[1]))  # 迭代词组
            print("Depth:", depth, "Similar:", [x[0] for x in similar_words_next], "->", similar_words_next)
            depth += 1
            words_depth.update({w: depth for w in similar_words if w not in words_depth})

            if not graph:
                continue

            node_name = f"Co_{wid}"
            for x in similar_words_next:
                word_node = CreateNode(x[0], words_depth[x[0]], graph, node_name,
                                       similar=float(data.loc[co, x[0]]))  # w=x[0]
                relationships = [Relationship(word_node, "SIMILAR_TO",
                                              CreateNode(y[0], words_depth[y[0]], graph, node_name,
                                                         similar=float(data.loc[co, y[0]])), rank=i,
                                              similarity=float(y[1]))
                                 for i, y in enumerate(x[1])]
                print(relationships)
                graph.create(Subgraph(relationships=relationships))  # 创建关系

        except KeyError:
            print(f"Data '{co, similar_words}' not in vocabulary.")
            break

    return words_depth


class XYRelationships:
    graph = None
    data = None

    def load(self, graph, data):
        self.graph = graph
        self.data = data
        print('数据导入:', self.data.columns, self.data.shape)

    def similar(self, txt, topn=10, duplicate=0):
        tokens = re.split(r'[^\w\s]| ', txt)
        tokens = [token.strip().lower() for token in tokens]
        if self.data is None or self.data.empty:
            return tokens, []
        tokens = [token for token in tokens if token in self.data.index]
        if len(tokens) == 0:
            return [], []
        return tokens, SimilarByCol(tokens, self.data, topn=topn, duplicate=duplicate)

    def match_relationships(self, co):
        if self.graph:
            wid = f"Co_{co}"
            return self.graph.run(
                "MATCH (source:" + wid + " {name:\"" + co + "\"})-[relation:SIMILAR_TO]->(target) RETURN source,relation,target").data()
        return []

    def create_relationships(self, co, width=3, max_depth=3, duplicate=3, create=0, max_node=400):
        if self.data is None or self.data.empty:
            return {}, []
        max_depth -= 1
        # return CreateCoSimilarityRelationship(co, self.data, self.graph,
        #                                       width, max_depth, co, duplicate)
        words_depth = {co: 0}
        relationships_edges = []
        similar_words = [co]
        depth = 0
        while depth <= max_depth and len(words_depth) <= max_node:
            try:
                similar_words_next = SimilarByCol(similar_words, self.data, topn=width, exclude=list(words_depth),
                                                  duplicate=duplicate)  # 分支会有合并,下层关系去重
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
                    node_name = f"Co_{co}"
                    for x in similar_words_next:
                        word_node = CreateNode(x[0], words_depth[x[0]], self.graph, node_name,
                                               similar=float(self.data.loc[co, x[0]]))  # w=x[0]
                        relationships = [Relationship(word_node, "SIMILAR_TO",
                                                      CreateNode(y[0], words_depth[y[0]], self.graph, node_name,
                                                                 similar=float(self.data.loc[co, y[0]])), rank=i,
                                                      similarity=float(y[1]))
                                         for i, y in enumerate(x[1])]

                        self.graph.create(Subgraph(relationships=relationships))  # 创建关系
                        relationships_edges += relationships

            except KeyError:
                print(f"Data '{co, similar_words}' not in vocabulary.")
                break

        return words_depth, relationships_edges


if __name__ == '__main__':
    graph = Graph("bolt://10.10.10.5:7687", auth=("d", "77"))
    co_sim = pd.read_excel('data/co_cosine_sim_3.xlsx', index_col=0)
    CreateCoSimilarityRelationship('通联数据股份公司', co_sim, width=3, max_depth=2, wid='通联数据', duplicate=0,
                                   graph=graph)
