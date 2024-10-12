import re, json
import inspect
import xml.etree.ElementTree as ET

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