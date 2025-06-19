from collections import Counter, deque
import os, re
import math
import jieba
from utils import get_tokenizer


def read2list(path, encoding='UTF-8', **kwargs):
    with open(path, encoding=encoding, **kwargs) as f:
        l = [line.strip('\n') for line in f.readlines() if line.strip()]
    return l


def cut_text(text, tokenizer=None, model_name="gpt-3.5-turbo"):
    # 去除标点/数字/空格
    text = re.sub(r'[^\u4e00-\u9fa5]', '', str(text))
    tokenizer = tokenizer or get_tokenizer(model_name)
    if tokenizer:
        token_ids = tokenizer.encode(text)
        tokens = [tokenizer.decode([tid]) for tid in token_ids]  # tokenizer.tokenize
    else:
        tokens = jieba.lcut(text, cut_all=False)
    return tokens  # ' '.join(tokens)


def load_jieba(dict_path=None, stop_path=None):
    # 初始化（可选，提前加载）
    jieba.initialize()

    # 加载领域词典（强烈建议用于专利/行业语料）
    if dict_path and os.path.exists(dict_path):
        jieba.load_userdict(dict_path)  # .txt 自定义词典文件,提取高频词来构建词典,对中文检索质量影响很大

    if stop_path and os.path.exists(stop_path):
        stop_words = set(read2list(stop_path))  # .txt
        return stop_words
    return None


def rerank_search_jieba(query: str, results: list):
    query_tokens = set(jieba.cut(query))  # jieba.cut_for_search(text)
    scored = []
    for hit in results:
        doc_text = hit.payload["text"]
        doc_tokens = set(jieba.cut(doc_text))
        overlap_score = len(query_tokens & doc_tokens)  # 简单词交集数
        final_score = hit.score + 0.1 * overlap_score  # 融合打分
        scored.append((hit.payload, final_score))

    return sorted(scored, key=lambda x: x[1], reverse=True)


class BM25:
    """
    TF-IDF算法的改进版，通过引入词频（TF）和文档频率（DF）的函数来计算文档与查询的相关性得分。

    技术原理：BM25算法考虑了词频和文档长度，通过参数调整可以优化长文档和短文档的检索效果。
    参数调整：BM25算法中的参数k1和b可以调整，以适应不同的检索需求和数据集特性。
    在处理大规模数据集时效率较高，尤其是在数据量较大时效果更好，但在语义理解方面可能存在局限。
    """

    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus = [jieba.lcut(doc) for doc in corpus]  # 使用 jieba 对文档进行分词, cut_all=False
        self.doc_lengths = [len(doc) for doc in self.corpus]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.corpus else 0
        self.doc_count = len(self.corpus)
        self.doc_term_freqs = [Counter(doc) for doc in self.corpus]
        self.inverted_index = self._build_inverted_index()
        self.idf_cache = {}  # 增加一个 IDF 缓存，提高效率

    def _build_inverted_index(self):
        inverted_index = {}
        for doc_id, doc_term_freq in enumerate(self.doc_term_freqs):
            for term, freq in doc_term_freq.items():
                if term not in inverted_index:
                    inverted_index[term] = []
                inverted_index[term].append((doc_id, freq))
        return inverted_index

    def _idf(self, term):
        if term in self.idf_cache:
            return self.idf_cache[term]
        doc_freq = len(self.inverted_index.get(term, []))  # 动态更新语料
        if doc_freq == 0:
            self.idf_cache[term] = 0
        else:  # 加1避免除以0，符合BM25标准公式
            self.idf_cache[term] = math.log((self.doc_count - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
        return self.idf_cache[term]

    def add_document(self, doc: str):
        """增量添加文档，并同步更新内部结构"""
        tokens = jieba.lcut(doc)
        doc_id = self.doc_count
        term_freq = Counter(tokens)

        # 更新核心结构
        self.corpus.append(tokens)
        self.doc_lengths.append(len(tokens))
        self.doc_term_freqs.append(term_freq)
        self.doc_count += 1
        self.avg_doc_length = sum(self.doc_lengths) / self.doc_count

        # 更新倒排索引
        for term, freq in term_freq.items():
            self.inverted_index.setdefault(term, []).append((doc_id, freq))
            if term in self.idf_cache:
                del self.idf_cache[term]  # 失效相关 IDF 缓存

    def get_sparse_vector(self, doc_id):
        doc_tf = self.doc_term_freqs[doc_id]
        sparse = {term: tf * self._idf(term) for term, tf in doc_tf.items()}
        return sparse

    def get_sparse_vectors(self):
        """将文档 term -> tf * idf 转换为稀疏向量，稀疏向量不是通用 encoder，必须先基于“训练语料”构建词表和 IDF 权重，构建后这套词表就应保持固定，否则新文档/查询的向量维度、IDF 就会不一致，导致不可用。文档集合（corpus）发生变化重新初始化"""
        return [self.get_sparse_vector(i) for i in range(self.doc_count)]

    def get_score(self, query_terms, doc_id):
        score = 0.0
        doc_tf = self.doc_term_freqs[doc_id]
        doc_length = self.doc_lengths[doc_id]
        for term in query_terms:
            tf = doc_tf.get(term, 0)
            if tf == 0:
                continue
            idf = self._idf(term)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
            score += idf * (numerator / denominator)
        return score

    def get_scores(self, query: str, normalize=True) -> list:
        query_terms = list(jieba.cut(query))  # 对查询进行分词
        scores = [(doc_id, self.get_score(query_terms, doc_id)) for doc_id in range(self.doc_count)]
        if normalize:
            max_score = max(scores, key=lambda x: x[1])[1]
            min_score = min(scores, key=lambda x: x[1])[1]
            if max_score != min_score:
                scores = [(doc_id, (score - min_score) / (max_score - min_score)) for doc_id, score in scores]

        return scores

    def rank_documents(self, query: str, top_k: int = None, normalize=False):
        scores = self.get_scores(query, normalize)
        ranked = sorted(scores, key=lambda x: x[1], reverse=True)
        return ranked[:top_k] if top_k else ranked


# class BM25:
#     def __init__(self, corpus, k1=1.5, b=0.75):
#         self.k1 = k1
#         self.b = b
#         self.corpus = [list(jieba.cut(doc)) for doc in corpus]  # doc.split()
#         self.N = len(self.corpus)  # 语料库中文档总数
#         self.avgdl = sum(len(doc) for doc in self.corpus) / self.N  # 文档的平均长度
#         self.df = self._calculate_df()  # 每个词项的文档频率
#         self.idf = self._calculate_idf()  # 每个词项的逆文档频率
#
#     def _calculate_df(self):
#         """计算词项的文档频率"""
#         df = {}
#         for doc in self.corpus:
#             unique_words = set(doc)
#             for word in unique_words:
#                 df[word] = df.get(word, 0) + 1
#         return df
#
#     def _calculate_idf(self):
#         """计算词项的逆文档频率"""
#         idf = {}
#         for word, freq in self.df.items():
#             idf[word] = math.log((self.N - freq + 0.5) / (freq + 0.5) + 1)
#         return idf
#
#     def _score(self, query, doc):
#         """计算单个文档对查询的 BM25 得分"""
#         score = 0.0
#         doc_len = len(doc)
#         term_frequencies = Counter(doc)
#         for word in query:
#             if word in term_frequencies:
#                 freq = term_frequencies[word]
#                 numerator = self.idf.get(word, 0) * freq * (self.k1 + 1)
#                 denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
#                 score += numerator / denominator
#         return score
#
#     def get_scores(self, query):
#         """计算语料库中每个文档对查询的 BM25 得分"""
#         query = list(jieba.cut(query))  # 使用 jieba 对查询进行分词
#         scores = []
#         for doc in self.corpus:
#             scores.append(self._score(query, doc))
#         return scores
if __name__ == "__main__":
    # from rank_bm25 import BM25Okapi
    load_jieba(dict_path='data/patent_thesaurus.txt', stop_path=None)
    corpus = [
        "快速的棕色狐狸跳过了懒狗",
        "懒狗躺下了",
        "狐狸很快速并且跳得很高",
        "快速的棕色狐狸",
        "猫跳过了狗"
    ]
    query = "快速的狐狸"
    bm25 = BM25(corpus)
    scores = bm25.get_scores(query, normalize=True)
    scores2 = bm25.rank_documents(query, top_k=3)
    print(scores, scores2, bm25.corpus)
    # print(BM25(corpus).rank_documents(query))
