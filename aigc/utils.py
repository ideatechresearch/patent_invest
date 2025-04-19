import re, json, io, os, sys, threading, time, pickle, struct
import httpx, aiohttp, aiofiles, asyncio, joblib
import inspect, importlib, ast, requests
from itertools import groupby
import yaml
from pathlib import Path
from contextlib import redirect_stdout
import xml.etree.ElementTree as ET
from difflib import get_close_matches, SequenceMatcher
from functools import partial, wraps  # cache, lru_cache, partial, wraps
from collections import OrderedDict, Counter, deque
import numpy as np
from enum import Enum
import math
from pypinyin import lazy_pinyin
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from langdetect import detect, detect_langs
import jieba


# import tiktoken
# tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

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


def serialize(obj):
    if isinstance(obj, Enum):  # 处理 Enum 类型
        return obj.value
    elif hasattr(obj, "dict"):  # 处理 Pydantic 模型
        return obj.dict()
    elif hasattr(obj, "__dict__"):  # 递归转换对象
        return {key: serialize(value) for key, value in obj.__dict__.items()}
    elif isinstance(obj, list):  # 处理列表
        return [serialize(item) for item in obj]
    elif isinstance(obj, dict):  # 处理字典
        return {key: serialize(value) for key, value in obj.items()}
    else:
        return obj  # asdict


def extract_json_from_string(input_str):
    # 从一个普通字符串中提取 JSON 结构，但可能不处理嵌套的 JSON
    match = re.search(r'\{.*}', input_str, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
    return None


def parse_json(inputs) -> dict:
    # 支持多种格式（Markdown JSON 块、普通 JSON 字符串、字典等）,支持已经是字典的输入
    if not isinstance(inputs, dict):
        try:
            match = re.search(r'^\s*(```json\n)?(.*)\n```\s*$', inputs, re.S)
            if match:
                inputs = match.group(2).strip()
            inputs = json.loads(inputs)
        except json.JSONDecodeError as exc:
            raise Exception(f'invalid json format: {inputs}') from exc

    return inputs


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


def execute_code_results(text):
    code_blocks = extract_python_code(text)
    results = []
    global_namespace = globals()  # 引用全局命名空间 {"__builtins__": dict(__builtins__)}
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


def pip_install(*args):
    import subprocess  # nosec
    # import platform
    cli_args = []
    for arg in args:
        cli_args.extend(str(arg).split(" "))  # export_command.split(" "),

    subprocess.run([sys.executable, "-m", "pip", "install", *cli_args],
                   # shell=(platform.system() == "Windows"),
                   check=True)


def git_repo_clone(repo_url: str, revision: str = None, add_to_sys_path: bool = True) -> Path:
    # 自动化环境部署、动态加载 Git 仓库代码
    repo_path = Path(repo_url.split("/")[-1].replace(".git", ""))
    import subprocess
    if not repo_path.exists():  # 本地没有该仓库
        try:
            subprocess.run(["git", "clone", repo_url], check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except Exception as exc:
            print(f"Failed to clone the repository: {exc.stderr}")
            raise

        if revision:  # 切换分支或提交
            subprocess.Popen(["git", "checkout", revision], cwd=str(repo_path))
    if add_to_sys_path and str(repo_path.resolve()) not in sys.path:
        sys.path.insert(0, str(repo_path.resolve()))
        # 加入 Python 的 sys.path，以便 import该仓库的模块

    return repo_path


async def download_by_aiohttp(url: str, save_path, chunk_size=4096, in_decode=False):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            # response = await asyncio.wait_for(session.get(url), timeout=timeout)
            if response.status == 200:
                if save_path:
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    async with aiofiles.open(save_path, mode='wb') as f:
                        # response.content 异步迭代器（流式读取）,iter_chunked 非阻塞调用，逐块读取
                        async for chunk in response.content.iter_chunked(chunk_size):
                            if isinstance(chunk, (bytes, bytearray)):
                                await f.write(chunk)
                            elif isinstance(chunk, str):
                                await f.write(chunk.encode('utf-8'))  # 将字符串转为字节
                            else:
                                raise TypeError(
                                    f"Unexpected chunk type: {type(chunk)}. Expected bytes or bytearray.")

                    return save_path

                content = await response.read()  # 单次异步读取完整内容，小文件,await response.content.read(chunk_size)
                return content.decode('utf-8') if in_decode else content  # 将字节转为字符串,解码后的字符串 await response.text()

            print(f"Failed to download {url}: {response.status}")

    return None


async def download_by_httpx(url: str, save_path, chunk_size=4096, in_decode=False):
    async with httpx.AsyncClient() as client:
        async with client.stream("GET", url) as response:  # 长时间的流式请求
            # response = await client.get(url, stream=True)
            response.raise_for_status()  # 如果响应不是2xx  response.status_code == 200:
            if save_path:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                async with aiofiles.open(save_path, mode="wb") as f:
                    # response.aiter_bytes() 异步迭代器
                    async for chunk in response.aiter_bytes(chunk_size=chunk_size):
                        await f.write(chunk)

                return save_path

            content = bytearray()  # 使用 `bytearray` 更高效地拼接二进制内容  b""  # raw bytes
            async for chunk in response.aiter_bytes(chunk_size=chunk_size):
                content.extend(chunk)
                # content += chunk
                # yield chunk

            return content.decode(response.encoding or 'utf-8') if in_decode else bytes(content)
            # return response.text if in_decode else response.content


def download_by_requests(url: str, save_path, chunk_size=4096, in_decode=False):
    """
    同步下载的流式方法
    如果目标是保存到文件，直接使用 content（无需解码）。（如图片、音频、视频、PDF 等）
    如果目标是处理和解析文本数据，且确定编码正确，使用 text。（如 HTML、JSON）
    """
    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()  # 确保请求成功
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as f:
                # requests iter_content 同步使用,阻塞调用
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:  # <class 'bytes'>
                        f.write(chunk)

            return save_path

        return response.text if in_decode else response.content  # 直接获取文本,同步直接返回全部内容


async def upload_by_httpx(url: str, files_path=('example.txt', b'Hello World')):
    files = {'file': files_path}
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, files=files)
        return resp.json()


def upload_by_requests(url: str, file_path, file_key='snapshot'):
    with open(file_path, "rb") as f:
        files = {file_key: f}
        response = requests.post(url, files=files)
    return response.json()


# 保存所有模型到文件
def save_models(models: dict, model_dir='data/models/'):
    """
    保存训练好的模型到指定路径
    :param models: 一个字典，包含所有需要保存的模型
    :param model_dir: 保存模型的路径
    """
    os.makedirs(model_dir, exist_ok=True)
    for model_name, model in models.items():
        if model is not None:
            joblib.dump(model, f'{model_dir}/{model_name}.pkl')
    print(f"模型已保存至 {model_dir}")


# 加载模型
def load_models(model_names: list, model_dir='data/models/'):
    """
    加载保存的模型
    :param model_names: 模型名称列表
    :param model_dir: 保存模型的路径
    :return: 加载的模型字典
    """
    models = {}
    for model_name in model_names:
        model_path = f'{model_dir}/{model_name}.pkl'
        if os.path.exists(model_path):
            models[model_name] = joblib.load(model_path)
    print(f"模型已从 {model_dir} 加载")
    return models


async def call_http_request(url, headers=None, time_out=100.0, **kwargs):
    async with httpx.AsyncClient() as cx:
        try:
            response = await cx.get(url, headers=headers, timeout=time_out, **kwargs)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(e)
            return None


async def post_http_json(url, js=None, headers: dict = None, time_out=100, **kwargs):
    connector = aiohttp.TCPConnector(limit=100, limit_per_host=50)
    timeout = aiohttp.ClientTimeout(total=time_out, sock_read=60.0, sock_connect=5.0)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout, headers=headers or {}) as session:
        async with session.post(url, json=js, **kwargs) as resp:
            return await resp.json()  # await resp.text()


# @asynccontextmanager
async def get_httpx_client(time_out=100.0):
    limits = httpx.Limits(max_connections=100, max_keepalive_connections=50)
    timeout = httpx.Timeout(timeout=time_out, read=60.0, write=30.0, connect=5.0)
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as cx:
        yield cx


# 调整缩进,修复代码缩进，确保最小的缩进被移除，以避免缩进错误
def fix_indentation(code):
    lines = code.splitlines()

    min_indent = min(len(line) - len(line.lstrip()) for line in lines if line.strip())
    fixed_lines = [line[min_indent:] if len(line) >= min_indent else line for line in lines]
    return "\n".join(fixed_lines)


def extract_python_code(text):
    """
    提取 Markdown 代码块中的 Python 代码，同时支持缩进代码块
    """
    code_blocks = re.findall(r'```(?:python)?(.*?)```', text, re.DOTALL)
    if not code_blocks:
        # 查找缩进代码块，即每行前 4 个空格的代码， 无 ``` 包围的代码块
        code_blocks = re.findall(r'((?: {4}.*\n)+)', text)

    return [fix_indentation(block) for block in code_blocks]  # [block.strip()]


def extract_any_code(markdown_string: str):
    # Regex pattern to match Python code blocks,匹配 Python与其他代码块
    pattern = r"```[\w\s]*python\n([\s\S]*?)```|```([\s\S]*?)```"
    # Find all matches in the markdown string
    matches = re.findall(pattern, markdown_string, re.IGNORECASE)
    # Extract the Python code from the matches
    code_blocks = []
    for match in matches:
        code = match[0] or match[1]  # 如果是 Python 代码块，取 ```python 之后的代码,如果是其他代码块，取代码内容
        code_blocks.append(code.strip())

    return code_blocks


def remove_function_decorators(func):
    """
    获取函数源码，去掉所有装饰器
    """
    source = inspect.getsource(func)  # 获取原始代码
    tree = ast.parse(source)  # 解析 AST

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):  # 找到函数定义
            node.decorator_list = []  # 清空装饰器列表

    return ast.unparse(tree)  # 重新转换成代码
    # import textwrap
    # textwrap.dedent(source)  # 处理缩进


def extract_function_metadata(code, r=None):
    # 用于解析Python函数并提取元数据,从源代码级别提取详细的代码信息
    # 使用AST来解析Python代码,将 Python 源代码（字符串形式）转换为抽象语法树（AST）
    tree = ast.parse(code)
    functions = []

    # 遍历AST节点，找到函数定义
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            function_name = node.name
            arguments = [arg.arg for arg in node.args.args]
            docstring = ast.get_docstring(node) or ""
            function_code = ast.unparse(node) if hasattr(ast, 'unparse') else ast.dump(node)
            # if generate_docstring and not docstring:
            #     docstring = generate_docstring(function_name, function_code)

            metadata = {
                'name': function_name,
                'args': arguments,
                'docstring': docstring,
                'code': function_code
            }
            functions.append(metadata)
            if r:
                key = f"function:{function_name}"  # 用函数名作为Redis的键
                r.set(key, str(metadata))  # 存储为JSON格式

    return functions


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


def extract_table_code(text):
    # 提取整个表格块
    table_blocks = re.findall(r'```(?:table)?(.*?)```', text, re.DOTALL)
    if not table_blocks:
        table_blocks = re.findall(r'((?:\|.*?\|)+)', text)  # 简单匹配 Markdown 表格，如 | A | B |
    return [block.strip() for block in table_blocks]


def extract_table_data(text):
    """提取 Markdown 格式的表格数据,返回按表格行分组的表格列表"""
    table_pattern = re.findall(r'(\|.*\|)', text)
    tables = []
    current_table = []

    for line in text.split("\n"):
        if line.strip() in table_pattern:
            current_table.append(line.strip())
        elif current_table:
            tables.append("\n".join(current_table))
            current_table = []

    if current_table:
        tables.append("\n".join(current_table))

    return tables


def extract_yaml_data(text):
    """提取 Markdown 中的 YAML 数据"""
    yaml_blocks = re.findall(r'```yaml\n(.*?)\n```', text, re.DOTALL)
    parsed_data = []

    for block in yaml_blocks:
        try:
            parsed_data.append(yaml.safe_load(block))  # 解析 YAML
        except yaml.YAMLError:
            parsed_data.append(None)  # 解析失败则返回 None

    return parsed_data


def extract_list_data(text):
    list_blocks = re.findall(r'```(?:list)?(.*?)```', text, re.DOTALL)
    if not list_blocks:
        list_blocks = re.findall(r'(\n\s*[-*].*?(\n\s{2,}.*?)*\n)', text)  # 纯文本列表
    return [block.strip() for block in list_blocks]


def extract_json_data(text):
    # 提取 JSON 格式的代码块
    json_blocks = re.findall(r'```(?:json)?(.*?)```', text, re.DOTALL)
    return [block.strip() for block in json_blocks]


def extract_json_str(json_code):
    start = json_code.find("```json")
    # 从start开始找到下一个```结束
    end = json_code.find("```", start + 1)
    if start == -1 or end == -1:
        try:
            json.loads(json_code)
            return json_code
        except Exception as e:
            print("Error:", e)
        return ""
    return json_code[start + 7:end]


def extract_code_blocks(text, lag='python', **kwargs):
    # 从文本中提取特定格式的代码块，支持不同的编程语言（如 Python、SQL、HTML 等）以及表格、JSON、列表等数据类型
    funcs = {
        "sql": extract_sql_code,
        "html": extract_html_code,
        "python": extract_python_code,
        "cpp": extract_cpp_code,
        "java": extract_java_code,
        "bash": extract_bash_code,
        "code": extract_any_code,
        "method": extract_method_calls,

        "table": extract_table_code,
        "yaml": extract_yaml_data,
        "list": extract_list_data,
        "json": extract_json_data,
    }
    if lag in funcs:
        return funcs[lag](text)

    # 提取 ``` 包裹的代码块
    code_blocks = re.findall(r'```(\w+)?\n(.*?)```', text, re.DOTALL)  # r'```(.*?)```'
    if lag:
        code_blocks = [block for block in code_blocks if block.lstrip().startswith(lag)]
        return code_blocks  # 过滤出指定语言的代码块

    # try:
    #     for k, f in funcs.items():
    #         print(k,f(text))
    # except Exception as e:
    #     print(k,e)

    return {k: f(text) for k, f in funcs.items()}


def extract_jsons(input_str, n=None):
    # 1,None,-1
    matches = re.findall(r'\{.*?\}', input_str, re.DOTALL)
    if not matches:
        return None
    json_objects = []  # [var.strip() for var in matches if '{' not in var]
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


def extract_tagged_content(text, tag="answer"):
    """
    提取指定标签最后一个匹配
    Extracts the value from the last occurrence of a specified tag in the text.

    Args:
        text (str): The input text containing the tagged content.
        tag (str): The tag to extract content from (default is 'answer').

    Returns:
        str or None: The extracted content, or None if no valid content is found.
    """
    pattern = rf"<{tag}>(.*?)</{tag}>"  # 正则匹配 <tag>...</tag>
    matches = re.findall(pattern, text, re.DOTALL)  # 获取所有匹配项"<answer> </answer>""

    if matches:
        last_match = matches[-1].strip()  # 获取最后一个匹配的内容并去除首尾空格
        return None if last_match == "..." else last_match
    return None


# reasoning
def extract_tagged_split(text, tag="think"):
    """
    Splits the text into two parts: the content inside the specified tag
    and the remaining text outside the tag.

    Args:
        text (str): The input text containing the tagged content.
        tag (str): The tag to extract content from (default is 'think',reasoning).

    Returns:
        list: A list containing [tag_content, remaining_text].
    """
    pattern = re.compile(rf"<{tag}>(.*?)</{tag}>\s*(.*)", re.DOTALL)
    match = pattern.search(text)

    if match:
        think_content = match.group(1).strip()  # 提取 <think> 内的内容,
        output_content = match.group(2).strip()  # 提取最终输出内容
        return [think_content, output_content]

    return [None, text]


def process_assistant_think(content):
    if '<think>' in content and '</think>' in content:
        content = re.sub(r'(<think>)(.*?)(</think>)',
                         r'<details style="font-style: italic; background: rgba(222, 222, 222, 0.5); padding: 10px; border-radius: 10px;"><summary style="font-weight:bold;">推理内容（展开）</summary>\2</details>',
                         content,
                         flags=re.DOTALL)

    if '<think>' in content and '</think>' not in content:
        content = re.sub(r'<think>(.*?)$',
                         r'<details open style="font-style: italic; background: rgba(222, 222, 222, 0.5); padding: 10px; border-radius: 10px;"><summary style="font-weight:bold;">推理中...</summary>\1</details>',
                         content,
                         flags=re.DOTALL)

    if '<think>' not in content and '</think>' in content:
        content = re.sub(r'(.*?)</think>',
                         r'<details style="font-style: italic; background: rgba(222, 222, 222, 0.5); padding: 10px; border-radius: 10px;"><summary style="font-weight:bold;">推理内容（展开）</summary>\1</details>',
                         content,
                         flags=re.DOTALL)

    return content


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
    formatted_text = extract_tagged_split(text, tag="think")[1]  # text
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


def df_to_markdown(df, index=False):
    # df.fillna('N/A').to_markdown()
    headers = df.columns.tolist()
    md = "| " + " | ".join(headers) + " |\n"
    md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for _, row in df.iterrows():
        md += "| " + " | ".join(row.astype(str)) + " |\n"

    return md if index else md.replace("| Index |", "|")  # 可选移除索引列


def format_for_html(text):
    # Markdown 格式的文本转换为 HTML 的字符串,渲染 Markdown 文章
    import markdown
    return markdown.markdown(text)
    # from IPython.display import Markdown, display
    # display(Markdown(f"`{export_command}`"))


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
        "tables": extract_table_data,
        "date": extract_date_strings,
        "answer": extract_tagged_content,
        "think": partial(extract_tagged_split, tag="think"),
        "reasoning": partial(extract_tagged_split, tag="reasoning"),
        'sentence': split_sentences,
        "wechat": format_for_wechat,
        'remark': remove_markdown,
        "html": format_for_html,
    }
    try:
        if extract in funcs:
            return funcs[extract](text, **kwargs)

        extract_type = extract.split('.')
        if extract_type[0] == 'code':
            transform = extract_code_blocks(text, lag=extract_type[1] if len(extract_type) > 1 else '', **kwargs)
            return transform

        return {k: f(text, **kwargs) for k, f in funcs.items()}  # "type": "all"
    except Exception as e:
        print(e)

    return None


# class Partial:
#     def __init__(self, func, *args, **kwargs):
#         self.func = func
#         self.args = args
#         self.kwargs = kwargs
#
#     def __call__(self, *more_args, **more_kwargs):
#         # 合并固定参数和新参数
#         all_args = self.args + more_args
#         all_kwargs = {**self.kwargs, **more_kwargs}
#         return self.func(*all_args, **all_kwargs)

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


def df2doc(data, use_index=True):
    """
    将 DataFrame 中每一行转换为一段文本
    :param data: 输入 DataFrame
    :param use_index: 是否在文本前增加行索引
    :return: 文本记录列表
    """
    docs = []
    try:
        import pandas as pd
        if use_index:
            for item in zip(data.index, data.to_dict(orient='records')):
                docs.append(f'{item[0]}\t' + '|'.join(
                    f'{k}#{v.strip() if isinstance(v, str) else v}' for k, v in item[1].items() if pd.notna(v)).strip())
        else:
            for item in data.to_dict(orient='records'):
                docs.append('|'.join(
                    f'{k}#{v.strip() if isinstance(v, str) else v}' for k, v in item.items() if pd.notna(v)).strip())
    except ImportError:
        if not isinstance(data, list) or not all(isinstance(d, dict) for d in data):
            raise ValueError("输入数据应为列表的字典格式，例如 [{'key1': 'value1', 'key2': 'value2'}, ...]")

        for idx, record in enumerate(data):  # data.iterrows()
            # 拼接每个字段，跳过 None 值，并对字符串做 strip 处理
            doc_line = '|'.join(
                f"{k}#{v.strip() if isinstance(v, str) else v}"
                for k, v in record.items() if v is not None
            )
            # 如果 use_index=True，则在前面加上索引
            if use_index:
                doc_line = f"{idx}\t" + doc_line

            docs.append(doc_line)
    except Exception as e:
        print(e)

    return docs


def get_last_entries_records(records: list[dict], fields, use_index=False, max_tokens: int = 8000, tokenizer=None):
    texts = []
    total_chars = 0
    # 从最新记录开始拼接，直到总字符数超过 max_tokens 时停止添加（返回最后不足 max_chars 字符的部分）
    for idx, record in enumerate(records):
        use_fields = fields or list(record.keys())
        prefix = f"{idx}\t" if use_index else ""
        item_str = prefix + '|'.join(
            f"{k}#{(str(record[k]).strip() if isinstance(record[k], str) else record[k])}"
            for k in use_fields if record.get(k) is not None
        )
        entry_length = lang_token_size(item_str, tokenizer)  # len(item_str)
        if total_chars + entry_length > max_tokens:
            break
        texts.append(item_str)
        total_chars += entry_length

    # 如果有多个记录，倒序拼接（保证最早的记录在最前面）
    return list(reversed(texts))  # "\n\n".join(reversed(texts))


def get_max_items_from_list(data: list, max_tokens: int = 4000, tokenizer=None):
    """
        Get max items from list of items based on defined max tokens (based on openai compute)
        根据给定的最大 token 数，从一组字典数据中选取适合的项目，直到达到 token 限制为止
        :param data: 包含字典的列表，每个字典表示一个项目
        :param max_tokens: 允许的最大 token 数
        :param tokenizer: 可选的 tokenizer（如果没有提供，则根据语言自动处理）
        :return: 适合的项目列表
        List[Dict[str, str]]
    """
    result = []
    current_tokens = 0
    # encoding = tiktoken.encoding_for_model(encoding_name)
    # tiktoken.get_encoding("cl100k_base")
    for item in data:
        item_str = json.dumps(item)
        item_tokens = lang_token_size(item_str, tokenizer)
        if current_tokens + item_tokens > max_tokens:
            break

        result.append(item)
        current_tokens += item_tokens

    return result


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
    # return detect(text)=='zh-cn'


def contains_hebrew_arabic(text):
    return bool(re.search(r'[\u0590-\u05FF\u0600-\u06FF]', text))


def contains_cjk(text):
    """检测是否包含 CJK（中文、日文、韩文）字符"""
    return bool(re.search(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', text))


def convert_to_pinyin(text):
    # 检查输入是否为中国城市名称（仅中文），然后转换为拼音
    if all('\u4e00' <= char <= '\u9fff' for char in text):
        return ''.join(lazy_pinyin(text))
    return text


def lang_detect_to_trans(text):
    t = detect(text)
    if t == 'zh-cn':
        t = 'zh'
    if t == 'no':
        t = 'zh' if contains_chinese(text) else 'auto'
    return t


def lang_token_size(text, tokenizer=None):
    if tokenizer:
        return len(tokenizer.encode(text))
    # 中文平均1字≈1 token，英文≈4字=1 token，粗略估算
    lang = detect(text)
    if lang in ('en', 'fr', 'es', 'de'):
        # 对于英文、法语、西班牙语、德语等，使用空格分词
        return len(text.split())
        # len(text) // 3

    # 'zh-cn','zh-hk','zh-tw','ja','ar',简体中文,日语,阿拉伯语，返回字符数
    return len(text)


def cut_text(text, tokenizer=None):
    # 去除标点/数字/空格
    text = re.sub(r'[^\u4e00-\u9fa5]', '', str(text))
    if tokenizer:
        token_ids = tokenizer.encode(text)
        tokens = [tokenizer.decode([tid]) for tid in token_ids]
    else:
        tokens = jieba.lcut(text, cut_all=False)
    return tokens  # ' '.join(tokens)


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


def build_prompt(messages: list, use_role=False) -> str:
    """
    Build a single prompt string from a list of messages.
    Each message is expected to be a dictionary with 'role' and 'content' keys.
    This function concatenates all message contents, preserving the training format.
    """
    if use_role:
        # OpenAI-style messages are transformed to a structured conversation format for Ollama.
        return "\n".join(
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content'].strip()}"
            for msg in messages)
    return "\n".join([msg["content"].strip() for msg in messages])


def alternate_chat_history(messages: list):
    # 确保 user 和 assistant 消息交替出现，插入默认消息或删除多余消息
    i = 0
    while i < len(messages) - 1:
        # if (
        #     isinstance(message, dict) and
        #     message.get("role") in ["user", "assistant"] and
        #     isinstance(message.get("content"), str) and
        #     message["content"].strip()  # 确保 content 非空
        # ):
        message = messages[i]
        next_message = messages[i + 1]
        # 处理连续相同角色的情况
        if message['role'] == next_message['role']:  # messages.insert(0, messages.pop(i))
            if i % 2 == 0:
                if message['role'] == 'user':
                    messages.insert(i + 1, {'role': 'assistant', 'content': '这是一个默认的回答。'})
                else:
                    messages.insert(i + 1, {'role': 'user', 'content': '请问您有什么问题？'})
            else:
                del messages[i + 1]
                i -= 1
        i += 1
    return messages


def split_whitespace_nonwhitespace(s, max_len=5):
    # 按照 空白/非空白 交替拆分字符串，控制每段的最大长度，预切割
    for k, g in groupby(s, key=str.isspace):
        chunk = list(g)
        for i in range(0, len(chunk), max_len):
            yield ''.join(chunk[i:i + max_len])


LINE_STOP_FLAG = (
    '.', '!', '?', '。', '！', '？', ')', '）', '"', '”', ':', '：', ';', '；', ']', '】', '}', '}', '>', '》', '、', ',', '，',
    '-',
    '—', '–',)
LINE_START_FLAG = ('(', '（', '"', '“', '【', '{', '《', '<', '「', '『', '【', '[',)


def find_last_punctuation(text, punctuations=("。", "？", "！", "；", "：")):
    """找到文本中最后一个有效的标点符号位置"""
    return max(text.rfind(p) for p in punctuations)


def is_punctuation_or_emoji(char):
    """检查字符是否为空格、指定标点或表情符号"""
    # 定义需要去除的中英文标点（包括全角/半角）
    punctuation_set = {
        '，', ',',  # 中文逗号 + 英文逗号
        '。', '.',  # 中文句号 + 英文句号
        '！', '!',  # 中文感叹号 + 英文感叹号
        '-', '－',  # 英文连字符 + 中文全角横线
        '、'  # 中文顿号
    }
    if char.isspace() or char in punctuation_set:
        return True
    # 检查表情符号（保留原有逻辑）
    code_point = ord(char)
    emoji_ranges = [
        (0x1F600, 0x1F64F), (0x1F300, 0x1F5FF),
        (0x1F680, 0x1F6FF), (0x1F900, 0x1F9FF),
        (0x1FA70, 0x1FAFF), (0x2600, 0x26FF),
        (0x2700, 0x27BF)
    ]
    return any(start <= code_point <= end for start, end in emoji_ranges)


def get_string_no_punctuation_or_emoji(s):
    """去除字符串首尾的空格、标点符号和表情符号,只清理首尾，不影响中间的内容"""
    chars = list(s)
    # 处理开头的字符
    start = 0
    while start < len(chars) and is_punctuation_or_emoji(chars[start]):
        start += 1
    # 处理结尾的字符
    end = len(chars) - 1
    while end >= start and is_punctuation_or_emoji(chars[end]):
        end -= 1
    return ''.join(chars[start:end + 1])


LLM_Abort_Event = asyncio.Event()  # threading.Event() 线程安全


async def llm_abort_stop():
    LLM_Abort_Event.set()  # 触发终止,是否提前终止


async def process_llm_stream(llm_responses_stream, token_size=20):
    """
    处理大模型返回的文本流，并按标点符号分割交给 TTS 朗读。
    :param llm_responses_stream: 大模型返回的文本流
    :param token_size: 标点不足时，允许的最小缓冲区长度
    """
    response_message = []
    text_index = 0
    processed_chars = 0
    async for content in llm_responses_stream:
        response_message.append(content)
        if LLM_Abort_Event.is_set():  # 实时检查是否终止
            break

        # 获取当前未处理的文本
        full_text = "".join(response_message)
        current_text = full_text[processed_chars:]

        # 查找最后一个有效标点
        last_punct_pos = find_last_punctuation(current_text)
        if last_punct_pos != -1 or lang_token_size(current_text) > token_size:
            split_pos = last_punct_pos if last_punct_pos != -1 else token_size  # 选取最合适的切割点
            segment_text_raw = current_text[:split_pos + 1]
            segment_text = get_string_no_punctuation_or_emoji(segment_text_raw)  # 处理无效字符
            if segment_text:
                text_index += 1
                yield segment_text, text_index
                processed_chars += len(segment_text_raw)  # 更新已处理字符位置

    # 处理剩余未分割的文本
    remaining_text = "".join(response_message)[processed_chars:]
    if remaining_text:
        segment_text = get_string_no_punctuation_or_emoji(remaining_text)
        if segment_text:
            text_index += 1
            yield segment_text, text_index

    yield response_message, -1  # finish_task


async def start_llm_stream(new_llm_stream):
    """复位终止信号，并重新启动大模型流"""
    LLM_Abort_Event.clear()  # 重新启动前复位
    async for text, idx in process_llm_stream(new_llm_stream):
        if idx > 0:
            print(f"🔊 朗读: {text}")


def split_sentences(text,
                    pattern=(r'[^一二三四五六七八九十\d\r\n]*\b[一二三四五六七八九十]+\、'  # 中文序号 "一、二、"
                             r'|[^（(）)]*\b[（(][一二三四五六七八九十]+[）)]'  # 括号内的中文序号 "(一)(二)"
                             r'|[^\d\r\n]*\b\d+\、'  # 数字序号 "1、2、"
                             r'|[^。！？]*[。！？]'  # 句号、感叹号、问号
                             r'|[^\r\n]*\r?\n'  # 换行符（支持 Windows 的 \r\n 和 Unix 的 \n）
                             )
                    ):
    """
    分句函数，支持按标点符号和结构化序号进行分句，分隔符会保留在前一句结尾。
    :param text: 输入的文本
    :param pattern: 正则表达式匹配分隔符
    :return: 分割后的句子列表
    """
    if not pattern:
        pattern = r'(?=[。！？])'
    sentences = re.findall(pattern, text)
    # re.findall re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]


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
    .pdf .txt .csv .doc .docx .xls .xlsx .ppt .pptx .md .jpeg .png .bmp .gif .svg .svgz .webp .ico .xbm .dib .pjp .tif .pjpeg .avif .dot .apng .epub .tiff .jfif .html .json .mobi .log .go .h .c .cpp .cxx .cc .cs .java .js .css .jsp .php .py .py3 .asp .yaml .yml .ini .conf .ts .tsx
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


def parse_tool_text(text):
    # 定义正则表达式模式来匹配 <tags>, <tool_call>, <content> 及其内容
    tags_pattern = r'<tags>(.*?)</tags>'
    tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
    content_pattern = r'<content>(.*?)</content>'
    # 使用正则表达式查找匹配的内容
    tags_match = re.search(tags_pattern, text, re.DOTALL)
    tool_call_match = re.search(tool_call_pattern, text, re.DOTALL)
    content_match = re.search(content_pattern, text, re.DOTALL)
    # 提取匹配的内容，如果没有匹配到则返回空字符串
    tags = tags_match.group(1).strip() if tags_match else ""
    tool_call = tool_call_match.group(1).strip() if tool_call_match else ""
    content = content_match.group(1).strip() if content_match else ""
    # 将提取的内容存储在字典中
    result = {
        "tags": tags,
        "tool_call": tool_call,
        "content": content
    }
    return result


def get_clock(t, speed=10):
    return "🕐🕑🕒🕓🕔🕕🕖🕗🕘🕙🕚🕛"[int(t * speed) % 12]


def parse_time(val):
    # 解析时间字符串，并转换为 datetime 对象。如果解析失败，返回一个默认值 datetime.min（0001-01-01 00:00:00）。
    if isinstance(val, datetime):
        return val
    try:
        return datetime.strptime(val, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return datetime.min


def format_date(date_str):
    # 直接使用 strptime 来解析日期并格式化为目标格式
    return datetime.strptime(date_str, "%Y/%m/%d %H:%M:%S").date().strftime("%Y-%m-%d")


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
        for fmt in supported_formats:
            try:
                date = datetime.strptime(date, fmt)
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"Invalid date format: {date}. Supported formats are {supported_formats}.")

    return date


def get_times_shift(days_shift: int = 0, hours_shift: int = 0):
    """
    :param days_shift: 偏移的天数，>0 表示未来，<0 表示过去，0 表示当前日期。
    :param hours_shift: 偏移的小时数，>0 表示未来，<0 表示过去，0 表示当前时间。
    :return: 格式化后的时间，格式为 'YYYY-MM-DD HH:MM:SS'。
    """
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


def is_finite(value) -> bool:
    """判断是否为有限数字"""
    try:
        float_val = float(value)
        return not (float_val == float('inf') or float_val == float('-inf') or float_val != float_val)
    except (TypeError, ValueError):
        return False


def cosine_sim(vecs1, vecs2):
    # 两个向量（1D 数组）之间的余弦相似度
    dot_product = np.dot(vecs1, vecs2)
    similarity = dot_product / (np.linalg.norm(vecs1) * np.linalg.norm(vecs2))
    return similarity


def fast_dot_np(vecs1, vecs2):
    # 用 NumPy 批量计算点积,形状相同的 2D 数组逐行点积,矩阵逐元素相乘后按行求和
    return np.einsum('ij,ij->i', vecs1, vecs2)  # np.sum(A * B, axis=1)


def normalize_np(vecs) -> list[float]:
    # 手动归一化
    # norms = np.sqrt(np.einsum('ij,ij->i', vecs, vecs)) #模长,L2 范数 ||ndarr1|| for each row
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def normalize_embeddings(vectors: list[list[float]], to_list=False):
    normalized = [vec / np.linalg.norm(vec) if np.linalg.norm(vec) != 0 else vec for vec in np.array(vectors)]
    return [vec.tolist() if isinstance(vec, np.ndarray) else vec for vec in normalized] if to_list else normalized


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def cosine_similarity_np(ndarr1, ndarr2):
    denominator = np.outer(np.linalg.norm(ndarr1, axis=1), np.linalg.norm(ndarr2, axis=1))
    dot_product = np.dot(ndarr1, ndarr2.T)  # np.einsum('ik,jk->ij', ndarr1, ndarr2)
    with np.errstate(divide='ignore', invalid='ignore'):
        similarity = np.where(denominator != 0, dot_product / denominator, 0)
    return similarity


def get_similar_nodes(embeddings, base_nodes, top_k=3):
    """
    计算 base_nodes（初步召回的记录）与所有记录之间的余弦相似度，找到最相似的 top_k 记录
    :param embeddings: 所有节点的嵌入矩阵 (Tensor)
    :param base_nodes: 需要查询相似记录的节点索引列表
    :param top_k: 每个节点要找的相似记录数
    :return: 召回的相似记录索引列表
    """
    # 提取 base_nodes 对应的向量
    base_embeddings = embeddings[base_nodes]
    # all_embeddings = embeddings.cpu().detach().numpy()

    # 计算余弦相似度 (sklearn)
    similarity_matrix = cosine_similarity_np(base_embeddings, embeddings)
    # 对每个 base_node 取最相似的 top_k 记录（排除自身）
    similar_nodes = set()
    for i, node in enumerate(base_nodes):
        sorted_indices = np.argsort(-similarity_matrix[i])  # 获取该记录的相似度排序,降序排序
        for idx in sorted_indices:
            if idx != node:  # 排除自身
                similar_nodes.add(idx)
            if len(similar_nodes) >= top_k:
                break

    return list(similar_nodes)


def float16_to_bin(num):
    # 将float16数打包为2字节16位，使用struct.pack 处理二进制数据的模块
    packed_num = struct.pack('e', num)  # e 半精度浮点数（float16,16-bit) b'\x00<'
    # 解包打包后的字节以获取整数表示
    int_value = struct.unpack('H', packed_num)[0]
    # 将整数表示转换为二进制
    binary_representation = bin(int_value)[2:].zfill(16)
    return binary_representation


def functions_registry(functions_list: list, safe_path=True, module_name: str = None) -> dict:
    """
    根据函数名称列表,创建全局函数注册表,或者指定模块中动态加载
    1. 从当前全局作用域查找函数名；
    2. 指定 module_name，批量从该模块加载；
    3. 使用 'module.path:func' 格式，单个动态加载。

    :param functions_list: 需要注册的函数名列表
    :param safe_path: 取消不检查是否可调用，适合从一个模块中加载多个函数。
    :param module_name: 模块名称（字符串形式），适合从一个模块中加载多个函数。
    :return: Dict[str, Callable[..., Any]]
    """
    module = importlib.import_module(module_name) if module_name else None
    if not safe_path:
        return {name: getattr(module, name) if module else globals().get(name) for name in functions_list}

    registry = {}
    for name in functions_list:
        try:
            if ":" in name:
                module_path, func_name = name.rsplit(":", 1)
                module = importlib.import_module(module_path)
                func_obj = getattr(module, func_name, None)
            else:
                func_obj = globals().get(name)

            if callable(func_obj):
                registry[name] = func_obj
            else:
                raise ValueError(f"函数 {name} 不存在或不是可调用对象,未在当前作用域中找到,可能未导入或模块未指定。")
        except Exception as e:
            registry[name] = None
            print(f"[⚠️] 加载函数失败: {name} → {e}")

    return registry
    # get_function_parameters


def function_registry_dynamic(functions_list: list, module_names: list):
    """
    动态加载模块并注册函数
    :param functions_list: 需要注册的函数名列表
    :param module_names: 模块名称列表（字符串形式）
    :return: 函数注册表
    """
    registry = {}
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)  # 动态加载模块
            for name in functions_list:
                if name not in registry:  # 避免重复覆盖
                    func = getattr(module, name, None)
                    if func is not None:
                        registry[name] = func
        except ModuleNotFoundError:
            print(f"Module '{module_name}' not found.")
    return registry


# import psutil,signal,contextlib
# def kill_process_tree(pid: int):
#     """
#     Kills all descendant processes of the given pid by sending SIGKILL.
#
#     Args:
#         pid (int): Process ID of the parent process
#     """
#     try:
#         parent = psutil.Process(pid)
#     except psutil.NoSuchProcess:
#         return
#
#     # Get all children recursively
#     children = parent.children(recursive=True)
#
#     # Send SIGKILL to all children first
#     for child in children:
#         with contextlib.suppress(ProcessLookupError):
#             os.kill(child.pid, signal.SIGKILL)
#
#     # Finally kill the parent
#     with contextlib.suppress(ProcessLookupError):
#         os.kill(pid, signal.SIGKILL)


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus = [jieba.lcut(doc) for doc in corpus]  # 使用 jieba 对文档进行分词, cut_all=False
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

    with open("utils.py", 'r', encoding='utf-8') as file:
        code = file.read()
    functions = extract_function_metadata(code, r=None)
    for func in functions:
        print(func)
