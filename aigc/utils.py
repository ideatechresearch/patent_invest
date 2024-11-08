import re, json
import inspect
import xml.etree.ElementTree as ET
from difflib import get_close_matches, SequenceMatcher
from collections import OrderedDict, Counter
import math
import jieba


class LRUCache:
    def __init__(self, capacity: int):
        self.stack = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key in self.stack:
            self.stack.move_to_end(key)
            return self.stack[key]
        else:
            return None

    def put(self, key, value) -> None:
        if key in self.stack:
            self.stack[key] = value
            self.stack.move_to_end(key)
        else:
            self.stack[key] = value
        if len(self.stack) > self.capacity:
            self.stack.popitem(last=False)

    def change_capacity(self, capacity):
        self.capacity = capacity
        for i in range(len(self.stack) - capacity):
            self.stack.popitem(last=False)

    def delete(self, key):
        if key in self.stack:
            del self.stack[key]

    def keys(self):
        return self.stack.keys()

    def __len__(self):
        return len(self.stack)

    def __contains__(self, key):
        return key in self.stack


def get_function_parameters(func):
    signature = inspect.signature(func)
    parameters = signature.parameters
    return [param for param in parameters]


def get_function_string(func):
    try:
        function_code = inspect.getsource(func)
        return function_code
    except Exception as e:
        return f"{e}"


# 递归地将字典中的所有键名转换为首字母大写的驼峰命名
def convert_keys_to_pascal_case(d):
    if isinstance(d, dict):
        new_dict = {}
        for k, v in d.items():
            new_key = ''.join(x.title() for x in k.split('_'))
            new_dict[new_key] = convert_keys_to_pascal_case(v)
        return new_dict
    elif isinstance(d, list):
        return [convert_keys_to_pascal_case(item) for item in d]
    else:
        return d


def extract_json_from_string(input_str):
    match = re.search(r'\{.*}', input_str, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
    return None


def execute_code_blocks(text):
    code_blocks = extract_python_code(text)
    for code in code_blocks:
        try:
            exec(code)
        except Exception as e:
            print(f"Error executing code block: {e}")


# 调整缩进,修复代码缩进，确保最小的缩进被移除，以避免缩进错误
def fix_indentation(code):
    lines = code.splitlines()

    min_indent = min(len(line) - len(line.lstrip()) for line in lines if line.strip())
    fixed_lines = [line[min_indent:] if len(line) >= min_indent else line for line in lines]
    return "\n".join(fixed_lines)


def extract_python_code(text):
    code_blocks = re.findall(r'```(?:python)?(.*?)```', text, re.DOTALL)
    if not code_blocks:
        code_blocks = re.findall(r'((?: {4}.*\n)+)', text)

    return [fix_indentation(block) for block in code_blocks]  # [block.strip()]


def extract_sql_code(text):
    sql_blocks = re.findall(r'```(?:sql)?(.*?)```', text, re.DOTALL)
    if not sql_blocks:
        sql_blocks = re.findall(r'((?:SELECT|INSERT|UPDATE|DELETE).*?;)', text, re.DOTALL)
    return [block.strip() for block in sql_blocks]


def extract_html_code(text):
    html_blocks = re.findall(r'```(?:html)?(.*?)```', text, re.DOTALL)
    if not html_blocks:
        html_blocks = re.findall(r'(<html.*?</html>)', text, re.DOTALL | re.IGNORECASE)
    return [block.strip() for block in html_blocks]


def extract_cpp_code(text):
    cpp_blocks = re.findall(r'```(?:cpp|c\+\+)?(.*?)```', text, re.DOTALL)
    return [block.strip() for block in cpp_blocks]


def extract_java_code(text):
    java_blocks = re.findall(r'```(?:java)?(.*?)```', text, re.DOTALL)
    return [block.strip() for block in java_blocks]


def extract_bash_code(text):
    bash_blocks = re.findall(r'```(?:bash|sh)?(.*?)```', text, re.DOTALL)
    return [block.strip() for block in bash_blocks]


def extract_table_data(text):
    table_blocks = re.findall(r'```(?:table)?(.*?)```', text, re.DOTALL)
    if not table_blocks:
        table_blocks = re.findall(r'((?:\|.*?\|)+)', text)  # 简单匹配 Markdown 表格，如 | A | B |
    return [block.strip() for block in table_blocks]


def extract_list_data(text):
    list_blocks = re.findall(r'```(?:list)?(.*?)```', text, re.DOTALL)
    if not list_blocks:
        list_blocks = re.findall(r'(\n\s*[-*].*?(\n\s{2,}.*?)*\n)', text)  # 纯文本列表
    return [block.strip() for block in list_blocks]


def extract_json_data(text):
    # 提取 JSON 格式的代码块
    json_blocks = re.findall(r'```(?:json)?(.*?)```', text, re.DOTALL)
    return [block.strip() for block in json_blocks]


def extract_code_blocks(text, lag=''):
    funcs = {
        "sql": extract_sql_code,
        "html": extract_html_code,
        "python": extract_python_code,
        "cpp": extract_cpp_code,
        "java": extract_java_code,
        "bash": extract_bash_code,

        "table": extract_table_data,
        "list": extract_list_data,
        "json": extract_json_data,
    }
    if lag in funcs:
        return funcs[lag](text)

    # 提取 ``` 包裹的代码块
    code_blocks = re.findall(r'```(.*?)```', text, re.DOTALL)
    if lag:
        code_blocks = [block for block in code_blocks if block.startswith(lag)]
        return code_blocks  # 过滤出指定语言的代码块

    return {k: f(text) for k, f in funcs.items()}


def extract_jsons(input_str, n=None):
    # 1,None,-1
    matches = re.findall(r'\{.*?\}', input_str, re.DOTALL)
    if not matches:
        return None
    json_objects = []
    for match in matches:
        try:
            json_objects.append(json.loads(match))
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e} - Skipping this fragment: {match}")

    if not json_objects:
        return None

    return json_objects if n is None else json_objects[:n]


def extract_headers(text):
    # 提取 ## 或 ### 等标题
    headers = re.findall(r'^(#{1,6})\s+(.*)', text, re.MULTILINE)
    return [{'level': len(header[0]), 'text': header[1]} for header in headers]


def extract_links(text):
    # 提取 Markdown 格式的链接 [链接文字](链接地址)
    links = re.findall(r'\[([^\]]+)\]\((https?:\/\/[^\s]+)\)', text)
    return [{'text': link[0], 'url': link[1]} for link in links]


def extract_bold(text):
    # 提取 Markdown 格式的 **粗体**
    bold_texts = re.findall(r'\*\*(.*?)\*\*', text)
    return bold_texts


def extract_italic(text):
    # 提取 Markdown 格式的 __斜体__ 或 *斜体*
    italic_texts = re.findall(r'__(.*?)__|\*(.*?)\*', text)
    return [italic[0] or italic[1] for italic in italic_texts]  # 处理两个捕获组


def ordinal_generator():
    ordinals = ['①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧', '⑨', '⑩']
    for ordinal in ordinals:
        yield ordinal


def remove_markdown(text):
    # 去除 Markdown 的常见标记
    """
    **粗体文本**
    _斜体文本_
    ![图片描述](image_url)
    [链接文本](url)
    ### 标题文本
    > 引用块
    * 无序列表项
    1. 有序列表项
    ~~删除线文本~~
    __下划线文本__
    """
    text = re.sub(r'(`{1,3})(.*?)\1', r'\2', text)  # 去除反引号代码块
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # 去除粗体
    text = re.sub(r'\*(.*?)\*', r'\1', text)  # 去除斜体
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)  # 去除图片
    text = re.sub(r'\[.*?\]\((.*?)\)', r'\1', text)  # 去除链接，但保留 URL
    text = re.sub(r'#{1,6}\s*(.*)', r'\1', text)  # 去除标题
    text = re.sub(r'>\s*(.*)', r'\1', text)  # 去除引用块
    text = re.sub(r'(\*|-|\+)\s+(.*)', r'\2', text)  # 去除无序列表符号
    text = re.sub(r'\d+\.\s+(.*)', r'\1', text)  # 去除有序列表符号
    text = re.sub(r'~~(.*?)~~', r'\1', text)  # 去除删除线
    text = re.sub(r'_{2}(.*?)_{2}', r'\1', text)  # 去除下划线标记
    text = re.sub(r'\[(.*?)\]\((.*?)\)', r'\1', text)  # 去除链接和 URL
    text = re.sub(r'\n{2,}', '\n', text)  # 将多余的空行替换为单个换行符
    return text.strip()


def format_for_wechat(text):
    formatted_text = text
    formatted_text = re.sub(r'\*\*(.*?)\*\*', r'✦\1✦', formatted_text)  # **粗体** 转换为 ✦粗体✦样式
    formatted_text = re.sub(r'__(.*?)__', r'※\1※', formatted_text)  # __斜体__ 转换为星号包围的样式
    formatted_text = re.sub(r'!!(.*?)!!', r'❗\1❗', formatted_text)  # !!高亮!! 转换为 ❗符号包围
    formatted_text = re.sub(r'~~(.*?)~~', r'_\1_', formatted_text)  # ~~下划线~~ 转换为下划线包围
    formatted_text = re.sub(r'\^\^(.*?)\^\^', r'||\1||', formatted_text)  # ^^重要^^ 转换为 ||重要|| 包围
    formatted_text = re.sub(r'######\s+(.*?)(\n|$)', r'[\1]\n', formatted_text)  # ###### 六级标题
    formatted_text = re.sub(r'#####\s+(.*?)(\n|$)', r'《\1》\n', formatted_text)  # ##### 五级标题
    formatted_text = re.sub(r'####\s+(.*?)(\n|$)', r'【\1】\n', formatted_text)  # #### 标题转换
    formatted_text = re.sub(r'###\s+(.*?)(\n|$)', r'— \1 —\n', formatted_text)  # ### 三级标题
    formatted_text = re.sub(r'##\s+(.*?)(\n|$)', r'—— \1 ——\n', formatted_text)  # ## 二级标题
    formatted_text = re.sub(r'#\s+(.*?)(\n|$)', r'——— \1 ———\n', formatted_text)  # # 一级标题
    # formatted_text = re.sub(r'```([^`]+)```',
    #                         lambda m: '\n'.join([f'｜ {line}' for line in m.group(1).splitlines()]) + '\n',
    #                         formatted_text)
    # formatted_text = re.sub(r'`([^`]+)`', r'「\1」', formatted_text)  # `代码` 转换为「代码」样式
    # formatted_text = re.sub(r'>\s?(.*)', r'💬 \1', formatted_text)  # > 引用文本，转换为聊天符号包围
    # formatted_text = re.sub(r'^\s*[-*+]\s+', '• ', formatted_text, flags=re.MULTILINE)  # 无序列表项
    # formatted_text = re.sub(r'^\s*\d+\.\s+',f"{next(ordinal_iter)} ", formatted_text, flags=re.MULTILINE)  # 有序列表项
    formatted_text = re.sub(r'\n{2,}', '\n\n', formatted_text)  # 转换换行以避免多余空行

    return formatted_text.strip()


def extract_string(text, extract, **kwargs):
    if not extract:
        return None
    funcs = {
        "json": extract_json_from_string,
        "jsons": extract_jsons,
        "header": extract_headers,
        "links": extract_links,
        "bold": extract_bold,
        "italic": extract_italic,
        "wechat": format_for_wechat,
    }
    try:
        if extract in funcs:
            return funcs[extract](text, **kwargs)

        extract_type = extract.split('.')
        if extract_type[0] == 'code':
            transform = extract_code_blocks(text, lag=extract_type[1], **kwargs)
            return transform

        return {k: f(text, **kwargs) for k, f in funcs.items()}  # "type": "all"
    except Exception as e:
        print(e)

    return None


def dict_to_xml(tag, d):
    """将字典转换为 XML 字符串"""
    elem = ET.Element(tag)
    for key, val in d.items():
        child = ET.SubElement(elem, key)
        if isinstance(val, list):
            for item in val:
                item_elem = ET.SubElement(child, "item")
                item_elem.text = str(item)
        else:
            child.text = str(val)
    return ET.tostring(elem, encoding='unicode')


def list_to_xml(tag, lst):
    """将列表转换为 XML 字符串"""
    elem = ET.Element(tag)
    for item in lst:
        item_elem = ET.SubElement(elem, "item")
        item_elem.text = str(item)
    return ET.tostring(elem, encoding='unicode')


def find_similar_word(target_keyword, tokens):
    max_ratio = 0
    similar_word_index = -1
    for i, token in enumerate(tokens):
        ratio = SequenceMatcher(None, target_keyword, token).ratio()
        if ratio > max_ratio:
            max_ratio = ratio
            similar_word_index = i
    return similar_word_index


def find_similar_words(query, tokens, top_n=3):
    matches = get_close_matches(query, tokens, n=top_n)
    # 计算每个匹配项与查询词的相似度
    results = []
    for match in matches:
        matcher = SequenceMatcher(None, query, match)
        results.append((match, matcher.ratio(), tokens.index(match)))

    return results


def contains_chinese(text):
    # 检测字符串中是否包含中文字符
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    return bool(chinese_pattern.search(text))


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus = [jieba.lcut(doc) for doc in corpus]  # 使用 jieba 对文档进行分词
        self.doc_lengths = [len(doc) for doc in self.corpus]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)
        self.doc_count = len(self.corpus)
        self.doc_term_freqs = [Counter(doc) for doc in self.corpus]
        self.inverted_index = self.build_inverted_index()
        self.idf_cache = {}  # 增加一个 IDF 缓存，提高效率

    def build_inverted_index(self):
        inverted_index = {}
        for doc_id, doc_term_freq in enumerate(self.doc_term_freqs):
            for term, freq in doc_term_freq.items():
                if term not in inverted_index:
                    inverted_index[term] = []
                inverted_index[term].append((doc_id, freq))
        return inverted_index

    def idf(self, term):
        if term in self.idf_cache:
            return self.idf_cache[term]
        doc_freq = len(self.inverted_index.get(term, []))
        if doc_freq == 0:
            self.idf_cache[term] = 0
        else:
            self.idf_cache[term] = math.log((self.doc_count - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
        return self.idf_cache[term]

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
        query_terms = list(jieba.cut(query))  # 对查询进行分词
        scores = [(doc_id, self.bm25_score(query_terms, doc_id)) for doc_id in range(self.doc_count)]
        return sorted(scores, key=lambda x: x[1], reverse=True)


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
    jieba.initialize()
    # jieba.load_userdict('data/patent_thesaurus.txt')
    corpus = [
        "快速的棕色狐狸跳过了懒狗",
        "懒狗躺下了",
        "狐狸很快速并且跳得很高",
        "快速的棕色狐狸",
        "猫跳过了狗"
    ]
    query = "快速的狐狸"
    bm25 = BM25(corpus)
    scores = bm25.rank_documents(query)
    print(scores, bm25.corpus)
    # print(BM25(corpus).rank_documents(query))
