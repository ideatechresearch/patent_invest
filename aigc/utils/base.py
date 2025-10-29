import json, re, random
from enum import Enum
import base64, hashlib
from collections import defaultdict
from urllib.parse import urlparse, urlencode, parse_qs, unquote_plus
from langdetect import detect, detect_langs

_tokenizer_cache = {}


def get_tokenizer(model_name: str = "gpt-3.5-turbo"):
    if not model_name:
        return None
    if model_name in _tokenizer_cache:
        return _tokenizer_cache[model_name]

    try:
        import tiktoken
        tokenizer = tiktoken.encoding_for_model(model_name)
        _tokenizer_cache[model_name] = tokenizer
        return tokenizer
    except ImportError:
        print("[Tokenizer Init] tiktoken not installed.")
    except Exception as e:
        print(f"[Tokenizer Init] Failed to get tokenizer for model '{model_name}': {e}")

    return None


def lang_token_size(text: str, tokenizer=None, model_name="gpt-3.5-turbo"):
    """
    计算文本的大致 token 数。
    - 优先使用提供的 tokenizer 或根据 model_name 获取的 tokenizer。
    - 若无 tokenizer，使用语言检测 + 启发式估算。

    参数：
        text: 文本内容
        tokenizer: tiktoken tokenizer（可选）
        model_name: 模型名称，用于获取默认 tokenizer（若未传入）

    返回：
        token 数量（估算或精确）
    """
    if not text:
        return 0
    tokenizer = tokenizer or get_tokenizer(model_name)
    if tokenizer:
        return len(tokenizer.encode(text))
    # 中文平均1字≈1 token，英文≈4字=1 token，粗略估算
    lang = detect(text[:100])
    if lang in ('en', 'fr', 'es', 'de'):
        return len(text.split())  # 对于英文、法语、西班牙语、德语等，使用空格分词,基于空格分词的近似 token 数
        # len(text.encode('utf-8')) // 3

    # 'zh-cn','zh-hk','zh-tw','ja','ar',简体中文,日语,阿拉伯语，返回字符数
    return len(text)


def get_max_tokens_from_string(text: str, max_tokens: int, tokenizer) -> str:
    """
        Extract max tokens from string using the specified encoding (based on openai compute)
        从一个字符串中提取出符合最大 token 数限制的部分
    """
    # from transformers import AutoTokenizer
    # encoding = AutoTokenizer.from_pretrained(model_id)
    # encoding = tiktoken.encoding_for_model(encoding_name)  # tiktoken.model.MODEL_TO_ENCODING
    tokens = tokenizer.encode(text)
    token_bytes = [tokenizer.decode_single_token_bytes(token) for token in tokens[:max_tokens]]
    return b"".join(token_bytes).decode()


def convert_keys_to_pascal_case(d):
    '''递归地将字典中的所有键名转换为首字母大写的驼峰命名'''
    if isinstance(d, dict):  # user_name->UserName
        return {''.join(x.title() for x in k.split('_')): convert_keys_to_pascal_case(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [convert_keys_to_pascal_case(item) for item in d]
    else:
        return d


def convert_keys_to_lower_case(d):
    """递归地将字典的所有键转换为小写"""
    if isinstance(d, dict):
        return {k.lower(): convert_keys_to_lower_case(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [convert_keys_to_lower_case(item) for item in d]
    else:
        return d


# 自动转换数值类型的辅助函数
def convert_num_value(value: str) -> str | int | float:
    """尝试将字符串转换为int或float，失败则返回原字符串"""
    value = value.strip()

    # 尝试转换为整数
    if re.match(r'^-?\d+$', value):
        try:
            return int(value)
        except ValueError:
            pass

    # 尝试转换为浮点数
    if re.match(r'^-?\d*\.\d+$', value) or re.match(r'^-?\d+\.\d*$', value):
        try:
            return float(value)
        except ValueError:
            pass

    # 尝试科学计数法表示的浮点数
    if re.match(r'^-?\d+(?:\.\d*)?[eE][-+]?\d+$', value):
        try:
            return float(value)
        except ValueError:
            pass

    # 尝试带符号的浮点数
    if re.match(r'^[+-]?\d*\.?\d+$', value):
        try:
            return float(value)
        except ValueError:
            pass

    return value


def serialize(obj):
    '''将复杂对象（包括自定义类、Pydantic模型、Enum等）递归转换为 可序列化的Python原生数据结构（dict/list/primitive types）'''
    if isinstance(obj, (str, int, float, bool)):  # 优先处理简单类型
        return obj
    if isinstance(obj, Enum):  # 处理 Enum 类型
        return obj.value
    elif isinstance(obj, (list, tuple, set, frozenset)):  # 处理列表
        return [serialize(item) for item in obj]
    elif isinstance(obj, dict):  # 处理字典
        return {key: serialize(value) for key, value in obj.items()}
    elif hasattr(obj, "dict") and callable(obj.dict):  # 处理 Pydantic 模型
        return obj.dict()
    elif hasattr(obj, "model_dump") and callable(obj.model_dump):  # 兼容 Pydantic v2 的 BaseMode
        return obj.model_dump()
    elif hasattr(obj, "__dict__"):  # 递归转换对象
        return {key: serialize(value) for key, value in obj.__dict__.items()}
    else:
        return obj  # asdict


def map_fields(data: dict, field_mapping: dict) -> dict:
    """
    递归遍历字典，根据提供的映射规则修改键名，支持对指定字段的值进行映射替换。

    :param data: 原始数据
    :param field_mapping: 键名映射（支持扁平路径）
    :return: 映射后的字典
    """

    def translate_key(prefix, key):
        full_key = f"{prefix}.{key}" if prefix else key
        return field_mapping.get(full_key, field_mapping.get(key, key))

    def recursive_map(obj, prefix=""):
        if isinstance(obj, dict):
            return {translate_key(prefix, k): recursive_map(v, f"{prefix}.{k}" if prefix else k)
                    for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recursive_map(item, prefix) for item in obj]
        else:
            return obj

    return recursive_map(data)


def extract_interface_map(data: dict, field_path: list, mapping: dict):
    """
    从 data 中按 field_path 提取字段，并用 mapping 映射字段。
    支持 field_path 为 str 或 list。
    """

    def get_nested_value(d, path):
        if isinstance(path, str):
            path = [path]
        for key in path:
            if isinstance(d, dict):
                d = d.get(key, {})
            else:
                return {}
        return d  # data.get("data", data.get("result", data))

    extracted = get_nested_value(data, field_path) if field_path else data
    if not mapping:
        return extracted

    if isinstance(extracted, list):
        return [map_fields(item, mapping) for item in extracted]
    elif isinstance(extracted, dict):
        return map_fields(extracted, mapping)

    return extracted  # 兜底处理


def slice_json(data, prefix='') -> dict:
    '''将嵌套结构完全展开为扁平字典（键变为路径字符串）'''
    slices = {}
    if isinstance(data, dict):
        for key, value in data.items():
            slices.update(slice_json(value, f"{prefix}{key}."))
    elif isinstance(data, list):
        for i, item in enumerate(data):
            slices.update(slice_json(item, f"{prefix}{i}."))
    else:
        slices[prefix[:-1]] = data

    return slices


def flatten_json(y):
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(y)
    return out


def get_origin_fild(index: int, data: dict[str, list | str] | list[tuple[str, list | str]]):
    items = data.items() if isinstance(data, dict) else data
    count = 0
    for k, v in items:
        count += len(v)
        if index < count:
            return k
    return None


def make_hashable(obj):
    if isinstance(obj, dict):
        return frozenset((k, make_hashable(v)) for k, v in obj.items())
    elif isinstance(obj, (list, tuple)):
        return tuple(make_hashable(i) for i in obj)
    elif isinstance(obj, set):
        return frozenset(make_hashable(i) for i in obj)
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return repr(obj)


def generate_hash_key(*args, **kwargs):
    """
    根据任意输入参数生成唯一的缓存键。
    :param args: 任意位置参数（如模型名称、模型 ID 等）
    :param kwargs: 任意关键字参数（如其他描述性信息）
    :return: 哈希键
    """
    # (id(_func), tuple(args), make_hashable(kwargs))frozenset(kwargs.items())
    # 将位置参数和关键字参数统一拼接成一个字符串
    inputs = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            inputs.extend(map(str, arg))  # 如果是列表，逐个转换为字符串
        elif isinstance(arg, (float, int, str, bool)):
            inputs.append(str(arg))
        else:
            inputs.append(json.dumps(arg, sort_keys=True, ensure_ascii=True, default=str))

    for key, value in kwargs.items():
        if isinstance(value, (list, dict, tuple)):
            value_str = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
        else:
            value_str = str(value)
        inputs.append(f"{key}:{value_str}")  # 格式化关键字参数为 key:value

    joined_key = "|".join(inputs)[:5000]
    # 返回 MD5 哈希
    return hashlib.md5(joined_key.encode()).hexdigest()


def key_from_inputs(*args, **kwargs) -> str:
    """当 generate_hash_key 不可用时的回退 key 生成器"""
    payload = {
        "inputs": make_hashable(args),
        **{k: make_hashable(v) for k, v in kwargs.items()}
    }
    raw = repr(payload).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def generate_random_hash(length=16):
    import string
    chars = string.ascii_letters + string.digits  # 字母 + 数字
    return ''.join(random.choice(chars) for _ in range(length))


def defaultdict_to_dict(d) -> dict:
    if isinstance(d, defaultdict):
        d = {k: defaultdict_to_dict(v) for k, v in d.items()}
    elif isinstance(d, dict):
        d = {k: defaultdict_to_dict(v) for k, v in d.items()}
    return d


def records_to_list(records: list[dict]) -> dict[str, list]:
    result = defaultdict(list)
    for row in records:
        for key, value in row.items():
            result[key].append(value)
    return dict(result)


def extract_json_struct(input_data: str | dict) -> dict | None:
    """
    从输入中提取结构化 JSON 对象，兼容以下格式：
    支持多种格式（Markdown JSON 块、普通 JSON 字符串、字典等）,支持已经是字典的输入
    - 直接为 dict 类型
    - 标准 JSON 字符串
    - Markdown JSON 块（```json ...)
    - 字符串中嵌入 JSON 部分（提取第一个 {...} 段）
    """

    if isinstance(input_data, dict):
        return input_data

    # Markdown JSON 格式：```json\n{...}\n```
    md_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", input_data, re.DOTALL)
    if md_match:
        json_str = md_match.group(1)
    else:
        # 尝试提取最外层 JSON：匹配最先出现的大括号包裹段，但可能不处理嵌套的 JSON
        brace_match = re.search(r"\{.*\}", input_data, re.DOTALL)
        json_str = brace_match.group(0) if brace_match else input_data

    try:
        return json.loads(json_str.strip())
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}\n invalid json format: {input_data}")
    return None


def extract_json_array(input_data: str | list) -> list | None:
    """
    提取并解析 markdown 包裹的 JSON 数组（尤其是 ```json ... ``` 格式）
    """
    if isinstance(input_data, list):
        return input_data
    # 提取 ```json ... ``` 块中内容
    md_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', input_data, re.DOTALL)
    if md_match:
        json_str = md_match.group(1)
    else:
        # 查找字符串中最外层 JSON 数组（首次出现）
        array_match = re.search(r"\[[\s\S]*?\]", input_data)  # r'\[\s*\{.*?\}\s*\]'
        if array_match:
            json_str = array_match.group(0)
        else:
            json_str = input_data.strip()  # 若本身就是 JSON 字符串（无包裹）

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
    return None


def extract_method_calls(text):
    """
    提取 Python 代码中的方法调用（方法名+括号）。
    支持：
    - 普通函数调用 func()
    - 对象/类方法调用 obj.method()
    Args:
        text (str): 代码文本
    Returns:
        list: 代码中的方法调用列表
    """
    # 匹配方法调用（方法名+括号内容）
    pattern = r"\b(?:[a-zA-Z_][a-zA-Z0-9_]*\.)?([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)"  # r"\b[a-zA-Z_][a-zA-Z0-9_]*\b"
    # 检查匹配项
    return re.findall(pattern, text)


def extract_date_strings(text):
    """
    从文本中提取符合格式的日期时间字符串。

    Args:
        text (str): 输入文本

    Returns:
        list: 提取出的日期时间字符串列表
    """
    date_regex = r'\b\d{4}[-/]\d{2}[-/]\d{2}(?:\s\d{2}:\d{2}:\d{2})?\b'
    return re.findall(date_regex, text)


def parse_database_uri(uri):
    parsed = urlparse(uri)
    query = parse_qs(parsed.query)
    # parsed.scheme  # e.g., mysql+aiomysql
    return {
        "host": parsed.hostname or 'localhost',
        "port": parsed.port or 3306,
        "user": unquote_plus(parsed.username),
        "password": unquote_plus(parsed.password),
        "db_name": parsed.path.lstrip('/'),  # 去掉前面的 /,parsed.path[1:]
        "charset": query.get("charset", ["utf8mb4"])[0]
    }


def extract_text_urls(text):
    # 提取所有 http(s) URL
    urls = re.findall(r'https?://[^\s)]+', text)
    return list(set(urls))


def is_url(url: str) -> bool:
    """更准确地判断是否为URL"""
    # url.startswith("http://") or url.startswith("https://")
    parsed = urlparse(url)
    # return all([parsed.scheme, parsed.netloc])
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


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
