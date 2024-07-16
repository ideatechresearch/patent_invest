from gensim import corpora
from gensim.models import Word2Vec,LdaModel,TfidfModel
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter,OrderedDict,defaultdict
import jieba
import multiprocessing
import pandas as pd
import numpy as np
import re,os,sys,pickle

class W2vLda:
    args = {
        'vector_size': 250,  # 定义向量维度大小
        'sg': 0,
        'window': 5,
        'len_below': 2,  # word
        'no_below': 2, # 最少频次.3
        'min_count': 2, #同上
        'count_fifter': 10,  # doc

        'num_topics': 20,  # 定义主题数.50
        'top_n_words': 10,  # 每个主题显示几个词.30
        'top_n_topics': 5, # 每个文档前几个主题,None,数量截取
        'weight_threshold_topics': 0.05,  # 文档主题权重大于阈值的,0.1,阈值截取
        'minimum_probability':0.05 #同上
    }
    # 初始化数据，不随改变参数改变
    sentences = pd.Series() # cleaned document
    id_data = None  # index
    group_ids = None # values,[arr.tolist() for arr in array_data]
    # public
    stop_words = set()
    dictionary = None
    corpus = []
    wo = None
    lda = None  # LDA主题模型可以通过单词概率进行解释,LsiModel,log_perplexity(corpus)
    word2vecs = {}  # Vector
    dict2vecs = {}  # 只需要lda词典有的词向量
    document_topics = None  # 所有文档的主题分布
    topic_documents_dict = {}  # 每个主题的文档集合的索引

    new_filter = False  # 控制停用词筛选
    suffix = ''

    #导入数据及分词 
    def __init__(self, sentences,wd_data=None,co_data=None,suffix='',stop_words=None,**kwargs)-> None:
        self.args = {**self.args, **kwargs}
        self.suffix = suffix

        if stop_words:
            self.stop_words = set(stop_words)#self.reset_stop_words(stop_words)

        if wd_data is None:  #导入分词数据
            self.sentences = sentences[sentences.apply(len) >= self.args.get('count_fifter', 0)].dropna()
        else:
            # self.wd_data=wd_data.dropna().apply(lambda x:
            # [w.strip() for w in jieba.lcut(clean_doc(x)) if len(w) >= self.args.get('len_below',2) and w not in self.stop_words]).dropna()
            # flag_data=wd_data.dropna().apply(lambda x:
            # [ i.flag for i in jieba.posseg.cut(clean_doc(x.lower())) if not i.word.isdigit()]).dropna()
            assert '序号' in wd_data.columns,'需要索引!'
                
            self.sentences = wd_data['标题 (中文)'].str.cat(wd_data['摘要 (中文)'].replace(np.nan,'')).dropna().apply(lambda x: [w.strip().lower() for w in jieba.lcut(clean_doc(x))
                                                   if not (len(w) < self.args.get('len_below', 2) or
                                                           w.isdigit() or re.match('\d+\.\d+$', w) or
                                                           w in self.stop_words)]).dropna()

            self.id_data = wd_data['序号'].copy()
            pd.concat([self.id_data, self.sentences.rename('词语')], axis=1).to_parquet(
                f'data\patent_cut_doc_{self.suffix}.parquet', index=False)  # 序号,[word,]
            # .to_pickle('data\patent_cut_doc.pkl') #to_csv('data\patent_cut_doc.csv',encoding="utf_8_sig")
            self.sentences = self.sentences[self.sentences.apply(len) >= self.args.get('count_fifter', 0)]

            if not co_data:
                co_data=wd_data['申请人'].str.split(';',expand=True).stack().str.strip().reset_index(level=1,drop=True).rename('Co')
                
            co_unstack = pd.merge(co_data, wd_data.loc[self.sentences.index,'序号'], left_index=True,
                                  right_index=True)  # Co,序号(wd_data dropna)
            
           
            self.group_ids = co_unstack.groupby('Co')['序号'].apply(lambda x: x.to_list()).reset_index()  # Co,['序号',]
            self.group_ids.to_parquet(f'data\patent_co_ids_{self.suffix}.parquet', index=False)
    
            self.new_filter = False

        
    # 中途可能推出，需要备份
    def save(self):
        if self.dictionary:
            self.dictionary.save(f'data\patent_dictionary_{self.suffix}.dict')
        if self.wo:
            self.wo.save(f'data\patent_w2v_{self.suffix}.model')
        if self.lda:
            self.lda.save(f'data\patent_lda_{self.suffix}.model')

    def load(self,dictionary=True, wo=True, lda=True):
        if dictionary:
            self.dictionary = corpora.Dictionary.load(f'data\patent_dictionary_{self.suffix}.dict')
        if wo:
            self.wo = Word2Vec.load(f'data\patent_w2v_{self.suffix}.model')
        if lda:
            self.lda = LdaModel.load(f'data\patent_lda_{self.suffix}.model')

    def update(self, params: dict):  # argparse.ArgumentParser() add_argument
        self.args.update(params)

    def params(self):  # parser.parse_args()
        return self.args  # |.__dict__

    def on_init(self,sentences,suffix=''):
        self.sentences = sentences[sentences.apply(len) >= self.args.get('count_fifter', 0)].dropna()
        self.new_filter = True
        if suffix:
            self.suffix = suffix
        
    def reset_stop_words(self, stop_words):  # 初始化停用词,搭配filter_stop_words
        self.stop_words = set(stop_words)
        self.dictionary = None  # 重建词典,dictionary.cfs
        self.new_filter = True
        return len(self.stop_words)

    # 过滤去除停用词，新增
    def filter_stop_words(self, stoplist):  # ["is", "a", "for", "in", "be", "the", "and", "should"]
        if not self.dictionary:
            return self.reset_stop_words(stoplist)

        stop_ids = [self.dictionary.token2id[w] for w in stoplist if
                    w in self.dictionary.token2id]  # [tokenid for tokenid, docfreq in self.dictionary.dfs.items() if docfreq == 4]
        if len(stop_ids) > 0:  # 词典中找到新停用词，重新计算
            self.stop_words |= set([self.dictionary.id2token[id] for id in stop_ids])
            self.dictionary.filter_tokens(bad_ids=stop_ids)

            self.wo = None  # 初始化,以便后续重新计算，重建词向量，否则最后相似度变化不大
            self.lda = None  # self.lda.update(self.corpus, chunksize=self.args.get('chunksize',1000), passes=self.args.get('passes',3)) #decay=0.5, iterations=50, gamma_threshold=0.0001
            self.new_filter = True

        return len(stop_ids), len(self.dictionary)


    # 词汇表及特征提取向量化:sentences,corpus,document_topics,documents_vec
    def on_corpus(self, **kwargs):
        if kwargs:
            self.args.update(kwargs)  # 更新参数

        processed_doc = self.sentences.map(
            lambda x: [w for w in x if w not in self.stop_words]) if self.new_filter else self.sentences  
        # .to_frame().applymap()不需要重新分词

        if not self.dictionary:
            self.dictionary = corpora.Dictionary(processed_doc)  # 单词ID到单词的映射
            self.dictionary.filter_extremes(no_below=self.args.get('no_below', 2),
                                            no_above=self.args.get('no_above', 0.99),
                                            keep_n=self.args.get('keep_n', 800000))  # 删除出现少于3个文档的单词或在95％以上文档中出现的单词

        self.corpus = [self.dictionary.doc2bow(text) for text in
                       processed_doc]  # new_corpus,整个语料库的词袋表示,文档集合,[(字典中词索引,词频)]：Counter(model.sentences[i])
        print(self.args)

        if not self.wo:
            self.wo = Word2Vec(processed_doc,
                               sg=self.args.get('sg', 0), vector_size=self.args['vector_size'],
                               window=self.args.get('window', 5), min_count=self.args.get('min_count', 2),
                               workers=multiprocessing.cpu_count())
        # self.word2vecs = {key:v for key,v in zip(self.wo.wv.index_to_key,self.wo.wv.vectors)}
        self.dict2vecs = {word: self.wo.wv[word] for word in
                          (set(self.wo.wv.index_to_key) & set(self.dictionary.values()))}


        if not self.lda:
            self.lda = LdaModel(corpus=self.corpus, id2word=self.dictionary, num_topics=self.args['num_topics'],
                                chunksize=self.args.get('chunksize', 1000), passes=self.args.get('passes', 3),
                                eta='auto', random_state=1234)  # iterations

        new_filter = False

        return len(self.wo.wv), len(self.dictionary), len(self.dict2vecs)  
        # len(self.word2vecs),wo.wv.vectors.shape



    def on_topics(self, **kwargs):
        if kwargs:
            self.args.update(kwargs)

        self.document_topics = self.lda.get_document_topics(self.corpus,
                                                    minimum_probability=self.args.get('minimum_probability',0.05))  
        # 所有文档的主题分布,列表
        # [lda[doc] for doc in corpus]\ [lda.get_document_topics(doc) for doc in corpus]

        topics_vec = []  # 基于词向量的主题向量表示
        for topic_id in range(self.lda.num_topics):  # lda.get_topics():num_topics,dictionary
            word_distribute = self.lda.show_topic(topicid=topic_id, topn=self.args['top_n_words'])  # 主题的词分布

            v_words = np.array(
                [self.dict2vecs.get(word) for word, _ in word_distribute])  # 词向量 句子向量,self.word2vecs[word]
            distribute = np.array([x[1] for x in word_distribute])
            w_distribute = (distribute / np.sum(distribute)).reshape(-1, 1)  # 归一化处理
            v_topic = (w_distribute * v_words).sum(axis=0)  # 词向量分别乘以其权重并加和
            topics_vec.append(v_topic)

        self.topic_documents_dict = {}  # 每个主题的文档集合的索引 topic:[index]
        documents_vec = []  # 基于词向量的专利文档向量
        for doc_index, topic_weight in enumerate(self.document_topics):
            for topic, prob in topic_weight:
                self.topic_documents_dict.setdefault(topic, []).append(doc_index)  # idx:0++
            if len(topic_weight) == 0:  # get 不到主题：空摘要,'',nan,[],去除停用词后词太少
                documents_vec.append(np.zeros(self.args['vector_size']))
                continue

            topic_n = sorted(topic_weight, key=lambda x: x[1], reverse=True)[:self.args.get('top_n_topics', None)]
            distribute = np.array(topic_n)[:, 1]
            mask = distribute >= self.args.get('weight_threshold_topics', 0.0)  # 主题权重大于阈值

            v_topics = np.array([topics_vec[t[0]] for t in topic_n])[mask]  # 主题向量
            w_distribute = (distribute[mask] / np.sum(distribute[mask])).reshape(-1, 1)  # 将归一化结果作为每个技术主题的权重
            v_doc = (w_distribute * v_topics).sum(axis=0)  # 专利文档中前个技术主题向量分别乘以其权重后加和
            documents_vec.append(v_doc)

        return topics_vec, np.array(documents_vec)  # <-document_topics<-corpus<-sentences


    def union_vec(self, documents_vec):
        df_lda_w2v = pd.DataFrame({'vec': [vec for vec in documents_vec]}, index=self.sentences.index)
        df_lda_w2v = df_lda_w2v[df_lda_w2v['vec'].apply(np.sum) != 0]
        df_lda_w2v['序号'] = self.id_data.loc[df_lda_w2v.index]  # iloc
        df_lda_w2v.set_index('序号', inplace=True)

        #.apply(lambda x: df_lda_w2v.loc[df_lda_w2v['序号'].isin(x), 0: self.args['vector_size'] - 1].mean(axis=0)) 
        co_ids_vec = self.group_ids['序号'].apply(lambda x: df_lda_w2v.loc[df_lda_w2v.index.isin(x),'vec'].apply(pd.Series).mean(axis=0))
        co_ids_vec.index = self.group_ids['Co']
        return co_ids_vec[co_ids_vec.sum(axis=1) != 0]  # Co:vec,以此做group相似度计算

    def docs_vec(self, documents_vec,wd_data=None):
        doc_lda_w2v = pd.DataFrame(documents_vec, index=self.sentences.index)
        doc_lda_w2v = doc_lda_w2v[doc_lda_w2v.sum(axis=1) != 0]
        if wd_data is not None:
            doc_lda_w2v=doc_lda_w2v.join(wd_data['序号']).set_index('序号')
            # doc_lda_w2v.merge(wd_data['序号'], left_index=True, right_index=True, how='left')
            # doc_lda_w2v['序号'] = wd_data.loc[doc_lda_w2v.index,'序号'] 
            # doc_lda_w2v.set_index('序号', inplace=True)
  
        doc_lda_w2v.to_parquet(f'data\\documents_vec_{self.suffix}.parquet')    
        return doc_lda_w2v

    def group_vec(self, df_documents_vec,group_ids):
        ids_vec = group_ids['序号'].apply(lambda x: df_documents_vec.loc[df_documents_vec.index.isin(x)].mean(axis=0))
        ids_vec.index = group_ids['Co']
        return ids_vec[ids_vec.sum(axis=1) != 0]

    def topic_vec(self,doc,topics_vec):
        processed_doc=[w.strip().lower() for w in jieba.lcut(clean_doc(doc))
                                     if not (len(w) < self.args.get('len_below', 2) or w.isdigit() 
                                             or re.match('\d+\.\d+$', w) or w in self.stop_words)]

        bow=self.dictionary.doc2bow(processed_doc)
        topic_distribution=self.lda.get_document_topics(bow,minimum_probability=self.args.get('minimum_probability',0.05))

        if len(topic_distribution) == 0:  # get 不到主题：空摘要,'',nan,[],去除停用词后词太少
            return np.zeros_like(topics_vec[0])

        topic_n = sorted(topic_distribution, key=lambda x: x[1], reverse=True)[:self.args.get('top_n_topics', None)]
        distribute = np.array(topic_n)[:, 1]
        mask = distribute >= self.args.get('weight_threshold_topics', 0.0)  # 主题权重大于阈值

        v_topics = np.array([topics_vec[t[0]] for t in topic_n])[mask]  # 主题向量
        if np.sum(distribute[mask]) == 0:
            return np.zeros_like(v_topics[0])
            
        w_distribute = (distribute[mask] / np.sum(distribute[mask])).reshape(-1, 1)  # 将归一化结果作为每个技术主题的权重
        v_doc = (w_distribute * v_topics).sum(axis=0) 

        return v_doc #然后与documents_vec找相似

    def topic_search(self,doc,topics_vec,documents_vec,wd_data=None):
        v_doc=self.topic_vec(doc,topics_vec)
        scores =cosine_similarity(documents_vec,v_doc.reshape(1, -1))
        scores_map= pd.Series(scores,index=self.sentences.index,name='score')
        if wd_data is not None:
            scores_map=scores_map.to_frame().join(wd_data['序号']).set_index('序号')
        return scores_map.sort_values(by='score', ascending=False)
        
    def df_topics(self, topn=30):  # lda.print_topics(num_topics=lda.num_topics, num_words=20)
        return pd.concat(
            [pd.DataFrame(self.lda.show_topic(topicid=topic_id, topn=topn if topn>0 else self.args['top_n_words']), columns=['word', topic_id]) for topic_id in
             range(self.lda.num_topics)], axis=1)

    def topics_words(self, topn=30):
        words=[]
        for topic_id in range(self.lda.num_topics):
            word_distribute=self.lda.show_topic(topicid=topic_id, topn=topn if topn>0 else self.args['top_n_words'])
            words +=[word for word, _ in word_distribute]#(word,topic_id) 
        return pd.Series(words,name='word')# value_counts()[:30],通用词停用

    def topic_cross(self):  # 计算主题之间的交叉度
        topic_cross_dists = np.zeros((self.lda.num_topics, self.lda.num_topics))
        for i in range(self.lda.num_topics):
            for j in range(i + 1, self.lda.num_topics):
                dists = []
                for doc_topic_dist in self.document_topics:
                    dists.append(distance.jensenshannon([x[1] for x in doc_topic_dist[i]],
                                                        [x[1] for x in doc_topic_dist[j]]))
                topic_cross_dists[i, j] = np.mean(dists)
                topic_cross_dists[j, i] = topic_cross_dists[i, j]
        return topic_cross_dists

    def lda_avg_kl(self):  # 主题的可区分性，衡量的是包含所有词的主题之间的距离
        topics_matrix = self.lda.get_topics()  # 每个元素表示相应主题中相应词汇的权重(主题，词汇在主题中的权重)
        return np.log2(average_jensen_shannon_kl_distance(topics_matrix) / topics_matrix.shape[
            1])  # ａｖｇ＿ＫＬ′值越大，表明主题与主题之间的距离越远，主题的可区分性越大

    def topic_word_id_entropy(self):  # 从语料库corpus,dictionary,计算快
        topic_wid_df_dict = {}  # wid:[df]
        topic_wid_tf_entropy_sum = Counter()
        for topic, doc_index in self.topic_documents_dict.items():
            # topic_doc_data=model.sentences.iloc[topic_doc_index]#每个主题的文档内容
            wid_tf_dict = {}  # wid:[tf]
            wid_df_list = []  # Counter([wid for idx in topic_doc_index for wid, _ in model.corpus[idx]])
            for idx in doc_index:
                for wid, count in self.corpus[idx]:
                    wid_tf_dict.setdefault(wid, []).append(count)  # 表示词 ｗｉ在领域 ｃｔ的文档 ｄｊ中的词频
                    wid_df_list.append(wid)

            topic_wid_tf_entropy_sum += Counter(
                {wid: calc_entropy(np.array(wid_tf_dict[wid]) / sum(wid_tf_dict[wid])) for wid in
                 wid_tf_dict})  # 类别内信息熵EIC

            for wid, count in Counter(wid_df_list).items():
                topic_wid_df_dict.setdefault(wid, []).append(count)  # 表示词 ｗｉ在类别 ｃｔ中的文档频次

        topic_wid_df_entropy = {wid: calc_entropy(np.array(topic_wid_df_dict[wid]) / sum(topic_wid_df_dict[wid])) for
                                wid in topic_wid_df_dict}  # 类别间信息熵EBC

        return topic_wid_df_entropy, topic_wid_tf_entropy_sum  # 类间熵,类别熵之和
        # len(topic_wid_df_entropy),len(topic_wid_tf_entropy_sum),len(self.dictionary)
        # [self.dictionary[wid] for wid in topic_wid_df_entropy.keys()-set(topic_wid_tf_entropy_sum)]

    def topic_word_entropy_corpus(self):  # 词的类别信息熵,从语料库中,已删除<no_below部分词汇
        topic_wid_df_entropy, topic_wid_tf_entropy_sum = self.topic_word_id_entropy()
        word_entropy = pd.concat(
            [pd.Series(topic_wid_df_entropy).rename('ebc'), pd.Series(topic_wid_tf_entropy_sum).rename('eic_sum')],
            axis=1)
        word_entropy['ec'] = word_entropy['ebc'] * word_entropy['eic_sum']  # 词的类别熵 ＥＣ
        word_entropy['word'] = [self.dictionary[wid] for wid in word_entropy.index]
        return word_entropy.sort_values('ec', ascending=False)

    def topic_word_entropy_docs(self):  # 词的类别信息熵,从文档中,已删除<len_below部分词汇
        # [index for index, topics in enumerate(model.document_topics) if any(topic[0] == 1 for topic in topics)]
        word_df = pd.Series()
        word_tf_eics = pd.Series()
        for topic, doc_index in self.topic_documents_dict.items():  # 每个主题的文档集合的索引
            topic_doc_data = self.sentences.iloc[doc_index]  # 每个主题的文档内容
            print(topic, len(doc_index))  # lib_c,tf_c,df_c =calc_tf_df(topic_doc_data)
            word_df = pd.concat([word_df, pd.Series(Counter([val for row in topic_doc_data for val in set(row)]))],
                                axis=0)  # 词 ｗｉ在类别 ｃｔ中的文档频次
            word_tf = pd.concat([pd.Series(Counter(row)) for row in topic_doc_data], axis=0)  # ｗｉ在类别 ｃｔ中的总词频
            word_tf_eics = pd.concat(
                [word_tf_eics, word_tf.groupby(word_tf.index).apply(lambda x: calc_entropy(list(x) / x.sum()))], axis=0)

        word_entropy = pd.concat([self.sentences.explode().value_counts().rename('tf'),
                                  self.sentences.apply(set).explode().value_counts().rename('df'),
                                  word_df.groupby(word_df.index).apply(
                                      lambda x: calc_entropy(list(x) / x.sum())).rename('ebc'),
                                  word_tf_eics.groupby(word_tf_eics.index).sum().rename('eic_sum')], axis=1)
        word_entropy['ec'] = word_entropy['ebc'] * word_entropy['eic_sum']  # 词的类别熵 ＥＣ
        return word_entropy.sort_values('ec', ascending=False)

    def word_entropy(self, word):  # 按词计算类别信息熵
        word_df = []
        word_tfs = []
        for topic, topic_doc_index in self.topic_documents_dict.items():  # 每个主题的文档集合的索引
            topic_doc_data = self.sentences.iloc[topic_doc_index]  # 每个主题的文档内容
            word_df.append(sum(1 for d in topic_doc_data if word in d))  # 词 ｗｉ在类别 ｃｔ中的文档频次
            word_tfs.append([d.count(word) for d in topic_doc_data if word in d])  # ｗｉ在类别 ｃｔ中的总词频

        ebc = calc_entropy(np.array(word_df) / sum(word_df))
        eic_sum = sum(calc_entropy(np.array(word_tf) / sum(word_tf)) for word_tf in word_tfs)
        return ebc, eic_sum, ebc * eic_sum



def cosine_sim_arr(co_ids_vec):
    matrix = cosine_similarity(co_ids_vec.values)
    return matrix[np.triu_indices(matrix.shape[1], k=1)]  # upper_triangle


def cosine_sim_df(co_ids_vec, triu=False):
    matrix = cosine_similarity(co_ids_vec.values)
    if triu:
        matrix[np.triu_indices(matrix.shape[0], k=0)] = np.nan
    return pd.DataFrame(matrix, index=co_ids_vec.index, columns=co_ids_vec.index)
    # co_cosine_sim.to_numpy()[~np.isnan(co_cosine_sim)]


def cosim_sim_top(co_ids_vec, top_n=50):
    matrix = cosine_similarity(co_ids_vec.values)
    upper_indices = np.triu_indices(co_ids_vec.shape[0], k=1)  # upper_triangle
    upper_values = matrix[upper_indices]
    sorted_indices = np.argsort(upper_values)  # sorted(enumerate(arr), key=lambda x: x[1], reverse=True)
    top_indices = sorted_indices[-top_n:]
    top_table = np.stack((co_ids_vec.index[upper_indices[0][top_indices]],
                          co_ids_vec.index[upper_indices[1][top_indices]],
                          upper_values[top_indices]), axis=1)
    return pd.DataFrame(top_table[::-1])


def restore_matrix(cosine_sim_arr):
    n = np.sqrt(2 * len(cosine_sim_arr) + 1 / 4) + 1 / 2  # 阶数
    if n <= 0 or n * (n - 1) != 2 * len(cosine_sim_arr):  # 确保数据长度合法
        raise ValueError(f"数据长度:{len(cosine_sim_arr)},不符合上半角矩阵的要求:{n}")

    n = int(n)  # n.is_integer()
    matrix = np.zeros((n, n))  # np.full((n, n), np.nan)

    idx = 0
    for i in range(n):
        matrix[i, i] = 1  # np.diag(np.ones(n))
        for j in range(i + 1, n):
            matrix[i, j] = cosine_sim_arr[idx]  # 填充上半角数据
            idx += 1

    matrix += matrix.T - np.diag(matrix.diagonal())  # 对称填充下半角
    return matrix

def arr2_cosine_sim(self, arr, columns):#group_ids.Co
    return pd.DataFrame(restore_matrix(arr), index=columns, columns=columns)

def get_max_count(word, word_counter, doc=None):
    s1 = word_counter.apply(lambda x: x.get(word))
    max_index = s1.idxmax()
    max_values = s1[max_index]  # int(s1.max())
    max_values_index = s1.index[s1.values == max_values]  # .tolist()
    max_values_2 = max_values
    try:
        s2 = doc[max_values_index].dropna().str.count(word)  # .iloc
        if len(max_values_index) > 1:
            max_index = s2.idxmax()
        max_values_2 = s2[max_index]
    except:
        print(word, doc[max_values_index])
        pass
    return max_values, max_values_2, max_index


def similar_by_words(words, model, topn=10, exclude=[]):
    eidx = [model.wv.key_to_index[w] for w in words + exclude]
    sim = pd.DataFrame(cosine_similarity(model.wv[words], np.delete(model.wv.vectors, eidx, axis=0)).T,
                       index=np.delete(model.wv.index_to_key, eidx), columns=words)
    return [(w, list(sim[w].sort_values(ascending=False)[:topn].items())) for w in words]


# 距离指标定量描述主题的区分度
def average_jensen_shannon_kl_distance(distributions):  # 计算多个概率分布之间的平均Jensen-Shannon距离。
    k = len(distributions)
    total_distance = 0.0

    # 遍历所有分布对
    for i in range(k):
        for j in range(k):
            kl_distance = np.sum(distributions[i] * np.log(distributions[i] / distributions[j]))
            total_distance += kl_distance

    # 计算平均值
    average_distance = total_distance / (k ** 2)
    return average_distance


def calc_entropy(class_probs):  # 计算类别间的信息熵,类别内的信息熵。(类别概率,样本概率)
    return -np.sum(class_probs * np.log2(class_probs))  # np.multiply


def calc_tf_df(documents):
    libs = set()
    for document in documents:
        libs.update(document)

    # 初始化 TF 和 DF 字典
    tf_dict = {word: 0 for word in libs}
    df_dict = {word: 0 for word in libs}

    # 计算 TF 和 DF
    for i, document in enumerate(documents):
        word_counts = Counter(document)  # df_li.append(dict(word_counts))

        for word, count in word_counts.items():
            tf_dict[word] += count
            df_dict[word] += 1
    return list(libs), tf_dict, df_dict  # sorted(tf_dict.items(), key=lambda x: x[1], reverse=True)


def c_tf_df(documents):
    return set([val for row in documents for val in row]), Counter([val for row in documents for val in row]), Counter(
        [val for row in documents for val in set(row)])


def clean_doc(text):
    pure_text = text.replace('\n', " ")
    pure_text = re.sub(r"-", " ", pure_text)

    pure_text = re.sub(r"\d+/\d+/\d+", "", pure_text)  # 剔除IP
    pure_text = re.sub(r"[0-2]?[0-9]:[0-6][0-9]", "", pure_text)  # 剔除时间

    # #URL，为了防止对中文的过滤，所以使用[a-zA-Z0-9]而不是\w
    url_regex = re.compile(r"""
        (https?://)?
        ([a-zA-Z0-9]+)
        (\.[a-zA-Z0-9]+)
        (\.[a-zA-Z0-9]+)*
        (/[a-zA-Z0-9]+)*
    """, re.VERBOSE | re.IGNORECASE)
    pure_text = url_regex.sub(r"", pure_text)

    # #剔除日期
    # data_regex = re.compile(u"""        #utf-8编码
    #     年 |月 |日 |
    #     (周一) | (周二) | (周三) | (周四) | (周五) | (周六)
    # """, re.VERBOSE)
    # decimal_regex = re.compile(r"[^a-zA-Z]\d+")    #剔除所有数字
    # space_regex = re.compile(r"\s+")    # 剔除空格

    # pure_text = re.sub("@([\s\S]*?):","",pure_text)  # 去除@ ...：
    # pure_text = re.sub("\[([\S\s]*?)\]","",pure_text)  # [...]：
    # pure_text = re.sub("@([\s\S]*?)","",pure_text)  # 去除@...
    # pure_text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+","",pure_text)  # 去除标点及特殊符号
    # pure_text = re.sub("[^\u4e00-\u9fa5]","",pure_text)  #  去除所有非汉字内容（英文数字）

    # cn_regex=re.compile(pattern='[\u4e00-\u9fa5]+')
    # pure_text =re.findall(pattern=cn_regex,string=pure_text)

    # pure_text[~pure_text.isin(stopwords)]不对原始文本处理停用词，分词后再排除
    return pure_text  # ' '.join(pure_text)


def read2list(file, encoding='UTF-8', **kwargs):
    with open(file, encoding=encoding, **kwargs) as file:
        l = [line.strip('\n') for line in file.readlines()]
    return l

#TF-IDF 改进
class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_lengths = [len(doc) for doc in corpus]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)
        self.doc_count = len(corpus)
        self.doc_term_freqs = [Counter(doc) for doc in corpus]
        self.inverted_index = self.build_inverted_index()

    def build_inverted_index(self):
        inverted_index = {}
        for doc_id, doc_term_freq in enumerate(self.doc_term_freqs):
            for term, freq in doc_term_freq.items():
                if term not in inverted_index:
                    inverted_index[term] = []
                inverted_index[term].append((doc_id, freq))
        return inverted_index

    def idf(self, term):
        doc_freq = len(self.inverted_index.get(term, []))
        if doc_freq == 0:
            return 0
        return math.log((self.doc_count - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

    def bm25_score(self, query_terms, doc_id):
        score = 0
        doc_length = self.doc_lengths[doc_id]
        for term in query_terms:
            tf = self.doc_term_freqs[doc_id].get(term, 0)
            idf = self.idf(term)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
            score += idf * (numerator / denominator)
        return score

    def rank_documents(self, query):
        query_terms = query.split()
        scores = [(doc_id, self.bm25_score(query_terms, doc_id)) for doc_id in range(self.doc_count)]
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        return sorted_scores