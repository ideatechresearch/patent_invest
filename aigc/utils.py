import re, json, io, os
import inspect
from pathlib import Path
from contextlib import redirect_stdout
import xml.etree.ElementTree as ET
from difflib import get_close_matches, SequenceMatcher
from collections import OrderedDict, Counter
import math
import jieba
from pypinyin import lazy_pinyin
from langdetect import detect, detect_langs
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


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


def extract_method_calls(text):
    # 匹配方法调用（方法名+括号内容）
    pattern = r"\b[a-zA-Z_][a-zA-Z0-9_]*\b"
    matches = re.findall(pattern, text)
    # 检查匹配项，返回最后一个方法名
    if matches:
        return matches[-1]
    return None


def execute_code_blocks(text):
    code_blocks = extract_python_code(text)
    results = []
    global_namespace = globals()  # 引用全局命名空间
    for code in code_blocks:
        local_namespace = {}  # 用于存储代码的局部变量
        captured_output = io.StringIO()  # 用于捕获 `print` 输出
        try:
            with redirect_stdout(captured_output):  # 重定向 `print` 输出
                exec(code, global_namespace, local_namespace)
                # exec(code, globals=None, locals=None)用于动态执行较复杂的代码块,不返回结果,需要通过全局或局部变量获取结果
            output = captured_output.getvalue()  # 获取 `print` 的内容
            results.append({
                "output": output.strip(),
                "namespace": local_namespace,
                "error": None
            })
        except Exception as e:
            results.append({
                "output": captured_output.getvalue().strip(),
                "namespace": local_namespace,
                "error": f"Error executing code block: {e}"
            })
        finally:
            captured_output.close()

    return results


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
    formatted_text = re.sub(r'!!(.*?)!!', r'❗\1❗', formatted_text)  # !!高亮!! 转换为 ❗符号包围
    # formatted_text = re.sub(r'__(.*?)__', r'※\1※', formatted_text)  # __斜体__ 转换为星号包围的样式
    formatted_text = re.sub(r'~~(.*?)~~', r'_\1_', formatted_text)  # ~~下划线~~ 转换为下划线包围
    formatted_text = re.sub(r'\^\^(.*?)\^\^', r'||\1||', formatted_text)  # ^^重要^^ 转换为 ||重要|| 包围
    formatted_text = re.sub(r'######\s+(.*?)(\n|$)', r'[\1]\n', formatted_text)  # ###### 六级标题
    formatted_text = re.sub(r'#####\s+(.*?)(\n|$)', r'《\1》\n', formatted_text)  # ##### 五级标题
    formatted_text = re.sub(r'####\s+(.*?)(\n|$)', r'【\1】\n', formatted_text)  # #### 标题转换
    formatted_text = re.sub(r'###\s+(.*?)(\n|$)', r'— \1 —\n', formatted_text)  # ### 三级标题
    formatted_text = re.sub(r'##\s+(.*?)(\n|$)', r'—— \1 ——\n', formatted_text)  # ## 二级标题
    formatted_text = re.sub(r'#\s+(.*?)(\n|$)', r'※ \1 ※\n', formatted_text)  # # 一级标题
    # formatted_text = re.sub(r'```([^`]+)```',
    #                         lambda m: '\n'.join([f'｜ {line}' for line in m.group(1).splitlines()]) + '\n',
    #                         formatted_text)
    # formatted_text = re.sub(r'`([^`]+)`', r'「\1」', formatted_text)  # `代码` 转换为「代码」样式
    # formatted_text = re.sub(r'>\s?(.*)', r'💬 \1', formatted_text)  # > 引用文本，转换为聊天符号包围
    # formatted_text = re.sub(r'^\s*[-*+]\s+', '• ', formatted_text, flags=re.MULTILINE)  # 无序列表项
    # formatted_text = re.sub(r'^\s*\d+\.\s+',f"{next(ordinal_iter)} ", formatted_text, flags=re.MULTILINE)  # 有序列表项
    formatted_text = re.sub(r'\n{2,}', '\n\n', formatted_text)  # 转换换行以避免多余空行

    return formatted_text.strip()


def format_for_html(text):
    try:
        import markdown
        return markdown.markdown(text)
    except:
        return remove_markdown(text)


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
        "html": format_for_html
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
    # detect(text)=='zh-cn'


def convert_to_pinyin(text):
    # 检查输入是否为中国城市名称（仅中文），然后转换为拼音
    if all('\u4e00' <= char <= '\u9fff' for char in text):
        return ''.join(lazy_pinyin(text))
    return text


def split_sentences(text,
                    pattern=r'(?<=[。！？])|(?=\b[一二三四五六七八九十]+\、)|(?=\b[（(][一二三四五六七八九十]+[）)])|(?=\b\d+\、)',
                    merged_pattern=r'\b[一二三四五六七八九十]+\、|\b[（(][一二三四五六七八九十]+[）)]|\b\d+\、'):  # r'(?<=。|！|？|\r\n)'
    """
    分句函数，支持按标点符号和结构化序号进行分句。
    :param text: 输入的文本
    :param pattern: 正则表达式匹配分隔符
    :param merged_pattern: 正则表达式匹配结构化序号（如“一、二、三”或“（一）”、“4、”）
    :return: 分割后的句子列表
    """
    # 基于句号、感叹号、问号进行分句
    sentences = re.split(pattern, text)
    # 去掉空白句子并返回
    if merged_pattern:
        merged_sentences = []
        temp = ""
        for sentence in sentences:
            if re.match(merged_pattern, sentence):
                if temp:
                    merged_sentences.append(temp.strip())
                temp = sentence
            else:
                temp += sentence
        if temp:
            merged_sentences.append(temp.strip())

        return [sentence for sentence in merged_sentences if sentence.strip()]

    return [sentence for sentence in sentences if sentence.strip()]


def split_paragraphs(sentences, max_length=512):
    paragraphs = []
    current_paragraph = ""

    for sentence in sentences:
        # 如果当前段落加上新句子的长度未超标，直接添加
        if len(current_paragraph) + len(sentence) <= max_length:
            current_paragraph += sentence
        else:
            # 超过 max_length，优先寻找标点符号处分割
            if len(current_paragraph) > 0:
                paragraphs.append(current_paragraph)
            current_paragraph = sentence

    # 添加最后一段
    if current_paragraph:
        paragraphs.append(current_paragraph)

    return paragraphs


#
# def split_paragraphs(text, max_length=256):
#     sentences = split_sentences(text)
#     paragraphs = []
#     current_paragraph = ""
#
#     for sentence in sentences:
#         # 如果当前段落加上新句子的长度未超标，直接添加
#         if len(current_paragraph) + len(sentence) <= max_length:
#             current_paragraph += sentence
#         else:
#             # 超过 max_length，优先寻找标点符号处分割
#             if len(current_paragraph) > 0:
#                 paragraphs.append(current_paragraph)
#             current_paragraph = sentence
#
#     # 最后一段加入
#     if current_paragraph:
#         paragraphs.append(current_paragraph)
#
#     # 处理超过长度的段落，优先按标点或换行分段
#     final_paragraphs = []
#     for paragraph in paragraphs:
#         if len(paragraph) > max_length:
#             # 查找标点或换行符位置
#             sub_paragraphs = re.split(r'(?<=[。！？])', paragraph)
#             buffer = ""
#             for sub in sub_paragraphs:
#                 if len(buffer) + len(sub) <= max_length:
#                     buffer += sub
#                 else:
#                     final_paragraphs.append(buffer.strip())
#                     buffer = sub
#             if buffer:
#                 final_paragraphs.append(buffer.strip())
#         else:
#             final_paragraphs.append(paragraph.strip())
#
#     return final_paragraphs

# 实现小到大分块逻辑
def organize_segments(tokens, small_chunk_size: int = 175, large_chunk_size: int = 512, overlap: int = 20):
    '''
    小块适合用于查询匹配，提高查询的精准度。
    大块划分，将包含上下文信息的多个小块合并为较大的片段。
    滑动窗口：为了保持上下文关系，在小块和大块之间添加一定的重叠区域，确保边缘信息不丢失。这样，查询结果能保持更高的连贯性。
    '''

    # 小块分割
    small_chunks = []
    for i in range(0, len(tokens), small_chunk_size - overlap):
        small_chunks.append(tokens[i:i + small_chunk_size])

    # 组织大片段
    large_chunks = []
    for i in range(0, len(small_chunks), large_chunk_size // small_chunk_size):
        large_chunk = []
        for j in range(i, min(i + large_chunk_size // small_chunk_size, len(small_chunks))):
            large_chunk.extend(small_chunks[j])
        large_chunks.append(large_chunk[:large_chunk_size])

    return small_chunks, large_chunks


# 支持的扩展名
def get_local_suffix(folder_path, supported_suffix=None, recursive=False):
    supported_extensions = (ext.lower() for ext in supported_suffix or [".jpg", ".jpeg", ".png", ".bmp"])
    folder = Path(folder_path)
    if not folder.is_dir():
        raise ValueError(f"The path '{folder_path}' is not a valid directory.")
    pattern = "**/*" if recursive else "*"
    return [str(f_path) for f_path in folder.glob(pattern) if f_path.suffix.lower() in supported_extensions]


def get_file_type(object_name: str) -> str:
    """
    根据文件名或路径判断文件类型。

    :param object_name: 文件名或路径
    :return: 文件类型（如 'image', 'audio', 'video', 'text', 'compressed', '*'）
    """
    if not object_name:
        return ""

    _, file_extension = os.path.splitext(object_name.lower())

    # 定义文件类型分类
    file_types = {
        "image": [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".webp", ".tiff", ".heic", ".heif"],
        "audio": [".mp3", ".wav", ".ogg", ".aac", ".flac", ".m4a"],
        "video": [".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".3gp"],
        "text": [".txt", ".csv", ".md", ".html", ".json", ".xml"],
        "document": [".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".numbers"],
        "compressed": [".zip", ".rar", ".7z", ".tar", ".gz", ".bz2"],
        "code": [".py", ".java", ".c", ".cpp", ".js", ".ts", ".html", ".css", ".sql"]
    }

    for file_type, extensions in file_types.items():
        if file_extension in extensions:
            return file_type

    return "*"

def get_file_type_wx(object_name: str) -> str:
    if not object_name:  # object_name.endswith()
        return ""
    '''
    文档：DOC、DOCX、XLS、XLSX、PPT、PPTX、PDF、Numbers、CSV
    图片：JPG、JPG2、PNG、GIF、WEBP、HEIC、HEIF、BMP、PCD、TIFF
    文件上传大小限制：每个文件最大512MB。
    '''
    _, file_extension = os.path.splitext(object_name.lower())
    # 根据文件后缀判断类型
    if file_extension in [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".webp"]:
        return "image"
    elif file_extension in [".mp3", ".wav", ".ogg", ".aac", ".flac"]:
        return "audio"
    elif file_extension in [".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm"]:
        return "video"
    elif file_extension in [".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".txt", ".csv",
                            '.zip', '.rar', '.html']:
        return "*"
    return ""


def format_date(date_str):
    # 直接使用 strptime 来解析日期并格式化为目标格式
    return datetime.strptime(date_str, "%Y/%m/%d %H:%M:%S").date().strftime("%Y-%m-%d")


class Url:
    def __init__(this, host, path, schema):
        this.host = host
        this.path = path
        this.schema = schema
        pass


def parse_url(requset_url):
    stidx = requset_url.index("://")
    host = requset_url[stidx + 3:]
    schema = requset_url[:stidx + 3]
    edidx = host.index("/")
    if edidx <= 0:
        raise Exception("invalid request url:" + requset_url)
    path = host[edidx:]
    host = host[:edidx]
    return Url(host, path, schema)


def format_date_type(date=None):
    # 如果没有传入日期，使用当前日期
    if not date:
        date = datetime.now()
    elif isinstance(date, str):
        supported_formats = [
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
        ]
        for date_format in supported_formats:
            try:
                date = datetime.strptime(date, date_format)
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"Invalid date format: {date}. Supported formats are {supported_formats}.")

    return date


def get_times_shift(days_shift: int = 0, hours_shift: int = 0):
    '''
    :param days_shift: 偏移的天数，>0 表示未来，<0 表示过去，0 表示当前日期。
    :param hours_shift: 偏移的小时数，>0 表示未来，<0 表示过去，0 表示当前时间。
    :return: 格式化后的时间，格式为 'YYYY-MM-DD HH:MM:SS'。
    '''
    current_datetime = datetime.now()
    adjusted_time = current_datetime + timedelta(days=days_shift, hours=hours_shift)
    return adjusted_time.strftime('%Y-%m-%d %H:%M:%S')


def get_day_range(date=None, shift: int = 0, count: int = 1):
    date = format_date_type(date)
    start_date = date - timedelta(days=shift)
    end_date = start_date + timedelta(days=count)
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")


def get_week_range(date=None, shift: int = 0, count: int = 1):
    """
    获取指定日期所在周的开始和结束日期。
    支持通过 shift 参数偏移周。

    :param date: 指定的日期（默认为当前日期）。
    :param shift: 偏移周数，>0 表示未来的周，<0 表示过去的周，0 表示当前周。
    :param count: 控制返回的周数范围，默认为 1，表示返回一个周的日期范围。
    :return: 返回指定周的开始和结束日期，格式为 ('YYYY-MM-DD', 'YYYY-MM-DD')
    """
    date = format_date_type(date)
    # 根据 shift 参数调整日期
    date = date + relativedelta(weeks=shift)
    # 获取今天是周几 (0 是周一, 6 是周日)
    weekday = date.weekday()
    # 计算周一的日期 (开始日期)
    start_of_week = date - timedelta(days=weekday)
    # 计算周日的日期 (结束日期)
    # end_of_week = start_of_week + timedelta(days=6)

    end_date = start_of_week + timedelta(weeks=count) - timedelta(days=1)  # start_of_week + timedelta(days=6)

    return start_of_week.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")


def get_month_range(date=None, shift: int = 0, count: int = 1):
    """
    获取指定日期所在月的开始和结束日期。
    支持通过 shift 参数偏移月数，和通过 count 控制返回的月份范围。

    :param date: 指定的日期（默认为当前日期）。
    :param shift: 偏移的月数，>0 表示未来的月，<0 表示过去的月，0 表示当前月。
    :param count: 控制返回的月份范围，默认为 1，表示返回一个月的开始和结束日期。
    :return: 返回指定月份的开始和结束日期，格式为 ('YYYY-MM-DD', 'YYYY-MM-DD')
      """
    date = format_date_type(date)
    # 根据 shift 参数调整日期
    start_date = (date + relativedelta(months=shift)).replace(day=1)
    # 计算下个月的第一天，然后减去一天
    end_date = (start_date + relativedelta(months=count)).replace(day=1) - timedelta(days=1)  # + timedelta(days=32)
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')  # '%Y-%m-01'


def get_quarter_range(date=None, shift: int = 0, count: int = 1):
    """
    获取指定日期所在季度的开始和结束日期。
    支持通过 shift 参数偏移季度数，和通过 count 控制返回的季度范围。

    :param date: 指定的日期（默认为当前日期）。
    :param shift: 偏移的季度数，>0 表示未来的季度，<0 表示过去的季度，0 表示当前季度。
    :param count: 控制返回的季度范围，默认为 1，表示返回一个季度的开始和结束日期。
    :return: 返回指定季度的开始和结束日期，格式为 ('YYYY-MM-DD', 'YYYY-MM-DD')
    """
    date = format_date_type(date)

    # 确定当前日期所在季度的起始月份
    start_month = 3 * ((date.month - 1) // 3) + 1
    start_date = (date.replace(month=start_month, day=1)
                  + relativedelta(months=3 * shift))

    # 计算季度结束日期
    end_date = (start_date + relativedelta(months=3 * count)).replace(day=1) - timedelta(days=1)

    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')


def get_quarter_month_range(date=None, shift: int = 0, count: int = 1):
    """
    获取指定日期所在季度的月份范围。
    支持通过 shift 参数偏移季度数，和通过 count 控制返回的季度范围。

    :param date: 指定的日期（默认为当前日期）。
    :param shift: 偏移季度数，>0 表示未来的季度，<0 表示过去的季度，0 表示当前季度。
    :param count: 控制返回的季度范围，默认为 1，表示返回一个季度的开始和结束日期。
    :return: 返回季度的开始月和结束月以及起始年份，格式为 ('YYYY','MM', 'MM')
    """
    date = format_date_type(date)

    current_year = date.year

    # 计算当前季度的起始月份
    quarter_start = (date.month - 1) // 3 * 3 + 1

    # 根据 shift 偏移季度
    quarter_start += shift * 3

    # 处理跨年情况：如果起始月份超出了12月，需要调整年份
    if quarter_start > 12:
        quarter_start -= 12
        current_year += 1
    elif quarter_start < 1:
        quarter_start += 12
        current_year -= 1

    # 计算季度的结束月份
    quarter_end = quarter_start + 3 * count - 1

    # 处理结束月份跨年情况：如果结束月份超过12月，需要调整年份
    if quarter_end > 12:
        quarter_end -= 12

    return current_year, quarter_start, quarter_end


def get_year_range(date=None, shift: int = 0, count: int = 1):
    """
    获取指定日期所在年的开始和结束日期。
    支持通过 shift 参数偏移年数，和通过 count 控制返回的年度范围。

    :param date: 指定的日期（默认为当前日期）。
    :param shift: 偏移的年数，>0 表示未来的年，<0 表示过去的年，0 表示当前年。
    :param count: 控制返回的年度范围，默认为 1，表示返回一年的开始和结束日期。
    :return: 返回指定年的开始和结束日期，格式为 ('YYYY-MM-DD', 'YYYY-MM-DD')
    """
    date = format_date_type(date)

    # 计算年份的开始日期
    start_date = date.replace(month=1, day=1) + relativedelta(years=shift)

    # 计算年份的结束日期
    end_date = (start_date + relativedelta(years=count)).replace(day=1, month=1) - timedelta(days=1)

    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')


def get_half_year_range(date=None, shift: int = 0, count: int = 1):
    """
    获取指定日期所在的半年（前半年或后半年）范围。
    支持通过 shift 参数偏移半年数，和通过 count 控制返回的半年数范围。

    :param date: 指定的日期（默认为当前日期）。
    :param shift: 半年偏移量，0 表示当前半年，-1 表示前一半年，1 表示下一半年。
    :param count: 返回的半年范围，默认为 1，表示返回一个半年的开始和结束日期。
    :return: 返回指定半年的开始和结束日期，格式为 ('YYYY-MM-DD', 'YYYY-MM-DD')
    """
    date = format_date_type(date)

    # 判断当前是前半年还是后半年
    if date.month <= 6:
        start_date = date.replace(month=1, day=1)
    else:
        start_date = date.replace(month=7, day=1)

    # 调整日期到指定的半年
    start_date += relativedelta(months=6 * shift)
    # 计算半年结束日期
    end_date = (start_date + relativedelta(months=6 * count)).replace(day=1) - timedelta(days=1)

    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')


def date_range_calculator(period_type: str, date=None, shift: int = 0, count: int = 1) -> dict:
    """
    计算基于参考日期的时间范围。

    :param period_type: 时间周期类型，'days'、'weeks'、'months' 等
    :param date: 基准日期，格式为 'YYYY-MM-DD'
    :param shift: 半年偏移量，0 表示当前半年，-1 表示前一半年，1 表示下一半年。
    :param count: 时间周期数量，表示从参考日期向前或向后的时长
    :return: 返回计算出的日期范围，包含 'start_date' 和 'end_date'
    """
    period_map = {'days': get_day_range,
                  'weeks': get_week_range,
                  'month': get_month_range,
                  'quarters': get_quarter_range,
                  'half_year': get_half_year_range,
                  'year': get_year_range,
                  }

    handler = period_map.get(period_type)
    if handler:
        start_date, end_date = handler(date, shift, count)
    else:
        raise ValueError(f"不支持的时间单位: {period_type}")

    # 返回结果字典，包含开始和结束日期
    return {'start_date': start_date, 'end_date': end_date}


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

    def rank_documents(self, query, sort=True, normalize=False):
        query_terms = list(jieba.cut(query))  # 对查询进行分词
        scores = [(doc_id, self.bm25_score(query_terms, doc_id)) for doc_id in range(self.doc_count)]
        if normalize:
            max_score = max(scores, key=lambda x: x[1])[1]
            min_score = min(scores, key=lambda x: x[1])[1]
            if max_score != min_score:
                scores = [(doc_id, (score - min_score) / (max_score - min_score)) for doc_id, score in scores]

        return sorted(scores, key=lambda x: x[1], reverse=True) if sort else scores


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
