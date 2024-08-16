from gensim import corpora
from gensim.models import LdaModel
import numpy as np
import re
import jieba


class LdaTopics:

    def __init__(self):
        self.lda = None
        self.dictionary = None
        self.suffix = ''
        self.stop_words = set()
        self.args = {
            'len_below': 2,  # word
            # 'num_topics': 20,  # 定义主题数
            # 'top_n_words': 10,  # 每个主题显示几个词
            'top_n_topics': 5,  # 每个文档前几个主题,None,数量截取
            'weight_threshold_topics': 0.05,  # 文档主题权重大于阈值的,0.1,阈值截取
            'minimum_probability': 0.05  # 同上
        }

    def load(self, suffix='', **kwargs):
        self.args = {**self.args, **kwargs}
        self.suffix = suffix
        jieba.initialize()
        jieba.load_userdict('data/patent_thesaurus.txt')  # 自定义词典文件
        self.stop_words = set(read2list('data/patent_stoppages.txt'))
        self.dictionary = corpora.Dictionary.load(f'data/patent_dictionary_{self.suffix}.dict')
        self.lda = LdaModel.load(f'data/patent_lda_{self.suffix}.model')
        self.topics_vec = np.load(f"data/topics_vec_{self.suffix}.npy")
        print('load lda topics:', self.topics_vec.shape)

    def notload(self):
        return self.lda is None

    def encode(self, doc, **kwargs):
        if kwargs:
            self.args.update(kwargs)

        processed_doc = [w.strip().lower() for w in jieba.lcut(clean_doc(doc))
                         if not (len(w) < self.args.get('len_below', 2) or w.isdigit()
                                 or re.match('\d+\.\d+$', w) or w in self.stop_words)]

        bow = self.dictionary.doc2bow(processed_doc)
        topic_distribution = self.lda.get_document_topics(bow, minimum_probability=self.args.get('minimum_probability',
                                                                                                 0.05))

        if len(topic_distribution) == 0:  # get 不到主题：空摘要,'',nan,[],去除停用词后词太少
            return np.zeros_like(self.topics_vec[0])

        topic_n = sorted(topic_distribution, key=lambda x: x[1], reverse=True)[:self.args.get('top_n_topics', None)]
        distribute = np.array(topic_n)[:, 1]
        mask = distribute >= self.args.get('weight_threshold_topics', 0.0)  # 主题权重大于阈值

        v_topics = np.array([self.topics_vec[t[0]] for t in topic_n])[mask]  # 主题向量
        if np.sum(distribute[mask]) == 0:
            return np.zeros_like(v_topics[0])

        w_distribute = (distribute[mask] / np.sum(distribute[mask])).reshape(-1, 1)  # 将归一化结果作为每个技术主题的权重
        v_doc = (w_distribute * v_topics).sum(axis=0)

        return v_doc  # 然后与documents_vec找相似


def read2list(file, encoding='UTF-8', **kwargs):
    with open(file, encoding=encoding, **kwargs) as file:
        l = [line.strip('\n') for line in file.readlines()]
    return l


# def encode_text(text):
#     inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
#     with torch.no_grad():
#         embeddings = model(**inputs).last_hidden_state.mean(dim=1)
#     return embeddings.cpu().numpy()

# AutoModelForSeq2SeqLM.from_pretrained('/bart-large-cnn/bart-base-chinese')
#  inputs = generator_tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
#  summary_ids = generator.generate(inputs['input_ids'], max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
#  generated_text = generator_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

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


if __name__ == '__main__':
    lt = LdaTopics()
    lt.load(suffix='xjzz', len_below=2, top_n_topics=4, minimum_probability=0.03, weight_threshold_topics=0.03)
    vec = lt.encode('一种电风扇摇头装置，其特征在于由两个同轴安 装的上传动盘和下传动盘组成，上传动盘的下表面设')
    print(vec)
