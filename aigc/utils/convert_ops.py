import ast
import markdown2
import yaml
from functools import partial
from .base import *


# convert/extract/format/fix/clean


def fix_unbalanced_brackets(name):
    left_count = name.count('(')
    right_count = name.count(')')

    # 如果左括号比右括号多，补齐右括号
    if left_count > right_count:
        name += ')' * (left_count - right_count)
    # 如果右括号比左括号多，补齐左括号
    elif right_count > left_count:
        name = '(' * (right_count - left_count) + name

    return name


def fix_indentation(code: str) -> str:
    """
    调整缩进,修复代码缩进，确保最小的缩进被移除，以避免缩进错误
    忽略空行，提取每行左侧空格数；
    找出所有非空行中的 最小缩进量（min_indent）；
    对每行去掉这个缩进量（保持相对缩进结构）；
    最后拼接为新的代码字符串。
    """
    lines = code.replace("\t", "    ").splitlines()
    non_empty_lines = [line for line in lines if line.strip()]
    if not non_empty_lines:
        return code

    min_indent = min(len(line) - len(line.lstrip()) for line in lines if line.strip())
    fixed_lines = [(line[min_indent:] if len(line) >= min_indent else line).rstrip() for line in lines]
    return "\n".join(fixed_lines)


def fix_invalid_backslashes(match):
    char = match.group(1)
    if char in '"\\/bfnrtu':  # JSON 里合法的转义字符只有这些： " \ bfnrtu
        return '\\' + char  # 合法保留
    else:
        return '\\\\' + char  # 非法的补成 \\ + 字符


def extract_python_code(text) -> list[str]:
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


def extract_json_str_md(json_code: str) -> str:
    """
    模型返回的内容，其中 JSON 数据通常被包裹在 Markdown 的代码块标记中（即以 json 开始，以 结束）
    如果未找到起始或结束标记，尝试直接解析整个字符串为 JSON
    :param json_code:
    :return:
    """
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


def extract_json_str(text: str) -> str:
    """
    尝试从输入字符串中提取 JSON 数组或对象字符串。
    - 支持去除 markdown 的 ```json 代码块包装
    - 优先提取 JSON 数组（[...]），其次为对象（{...}）
    - 若无明确结构，则尝试解析整体内容是否为合法 JSON
    """

    # 去除 markdown 包裹
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)

    # 查找 JSON 数组
    start_index = text.find('[')
    end_index = text.rfind(']')
    if start_index != -1 and end_index != -1 and end_index > start_index:
        return text[start_index:end_index + 1]

    # 查找 JSON 对象
    start_index = text.find('{')
    end_index = text.rfind('}')
    if start_index != -1 and end_index != -1 and end_index > start_index:
        return text[start_index:end_index + 1]

    # 尝试整体是否为合法 JSON
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError as e:
        print(f"提取失败，输入非合法 JSON: {e}")
        return ""


def parse_json(inputs: dict | str) -> dict:
    #  Markdown 中完整包裹的 JSON 块，支持多种格式（Markdown JSON 块、普通 JSON 字符串、字典等）,支持已经是字典的输入
    if not isinstance(inputs, dict):
        try:
            match = re.search(r'^\s*(```json\n)?(.*)\n```\s*$', inputs, re.S)
            if match:
                inputs = match.group(2).strip()
            inputs = json.loads(inputs)
        except json.JSONDecodeError as exc:
            raise Exception(f'invalid json format: {inputs}') from exc

    return inputs


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


def extract_markdown_blocks(text: str) -> list[str]:
    """
    返回内容块列表，去除包裹的 markdown 标记
    """
    blocks = re.findall(r"```markdown\n(.*?)```", text, flags=re.DOTALL)
    return [block.strip() for block in blocks]


def extract_table_code(text):
    # 提取整个表格块
    table_blocks = re.findall(r'```(?:table)?(.*?)```', text, re.DOTALL)
    if not table_blocks:
        table_blocks = re.findall(r'((?:\|.*?\|)+)', text)  # 简单匹配 Markdown 表格，如 | A | B |
    return [block.strip() for block in table_blocks]


def extract_table_blocks(text) -> list[str]:
    """
    提取 Markdown 格式的表格数据，返回完整的表格块列表
    table_blocks: List[str]，每个元素都是一段完整的表格（含多行）
    """
    table_pattern = re.compile(r'(?:^\|[^\n]*\|\s*$\n?)+', re.MULTILINE)
    table_blocks = table_pattern.findall(text)
    if table_blocks:
        return [block.strip() for block in table_blocks if block.strip()]

    # 如果没有找到，尝试更宽松的匹配（处理不规范的表格,逐行找以 | 开头并包含 | 的行）
    table_blocks = []
    current_table = []

    for line in text.splitlines():
        if re.match(r'^\|.*\|$', line.strip()):
            current_table.append(line.strip())
        elif current_table:
            table_blocks.append("\n".join(current_table))
            current_table = []

    if current_table:
        table_blocks.append("\n".join(current_table))

    return [block.strip() for block in table_blocks if block.strip()]


def extract_table_segments(raw_text) -> list[tuple[str, str]]:
    """
    从长文本中提取所有连续的“|...|”表格块，作为一个整体段落返回，
    并返回去除了这些表格块后的纯正文。
    :param raw_text: 原始多行合同文本
    :return: [('table',table_blocks), ('text',remaining_text)]
      - table_blocks: str，每个元素都是一段完整的表格（含多行）
      - remaining_text: str，没有表格块的正文
    """
    # ordered
    # 1. 正则匹配连续多行“|…|”表格块
    table_pattern = re.compile(r'(?:^\|[^\n]*\|\s*$\n?)+', re.MULTILINE)

    segments = []
    last_end = 0
    # 2. 遍历所有表格块
    for m in table_pattern.finditer(raw_text):
        start, end = m.span()
        # 2a. 先把表格块前面的正文片段收集下来
        if start > last_end:
            text_segment = raw_text[last_end:start]
            segments.append(('text', text_segment))
        # 2b. 再把这个表格块本身收集下来
        table_block = m.group()
        segments.append(('table', table_block))
        last_end = end

    # 2c. 最后收集表格块后剩余的正文
    if last_end < len(raw_text):
        segments.append(('text', raw_text[last_end:]))

    return segments


def extract_markdown_table(text: str, convert: bool = True) -> list[dict]:
    """
    从Markdown格式文本中提取表格并转换为字典列表
    保留原始字符串格式，只做必要的清理
    """
    block_match = re.search(r'(?:^\s*\|.*\|\s*$\n?)+', text, re.MULTILINE)  # 匹配Markdown表格模式
    if not block_match:
        return []

    lines = block_match.group(0).strip().splitlines()
    if len(lines) < 2:
        return []

    def clean_cell(content: str) -> str | None:
        """清理单元格内容，只移除基本的Markdown格式"""
        if content is None:
            return None

        content = re.sub(r'\*\*([^*]+)\*\*', r'\1', content)  # 移除简单的加粗标记（**text**），但保留内容
        content = re.sub(r'\*([^*]+)\*', r'\1', content)  # 移除简单的斜体标记（*text*），但保留内容
        content = re.sub(r'`([^`]*)`', r'\1', content)  # 移除 `code`
        return content.strip() or None  # 移除行首行尾多余空格

    # 清理和提取表头
    headers = [clean_cell(col.strip() or f"col{idx}") for idx, col in enumerate(lines[0].strip('|').split('|'))]
    sep_re = re.compile(r'^\s*\|?[-:\s|]+\|?\s*$')
    data_lines = [ln for ln in lines[1:] if not sep_re.match(ln)]  # 数据行从第一行之后，跳过所有分隔线

    # 处理数据行
    result = []
    for ln in data_lines:
        cells = [clean_cell(col.strip()) for col in ln.strip('|').split('|')]
        cells += [None] * (len(headers) - len(cells))
        cells = cells[:len(headers)]  # 截断多余列
        row = {hdr: convert_num_value(val) if convert and val is not None else val
               for hdr, val in zip(headers, cells)}  # 创建行数据
        result.append(row)

    return result


def parse_table_block(block: str) -> list[list[str]]:
    """
    将一个连续的表格块（多行以 | 开头和结尾）拆成行和字段列表。
    :param block: str, 形如:
      "| 列A | 列B |\n| --- | --- |\n| 1  | x   |\n| 2  | y   |\n"
    :return: List[List[str]]，如 [["列A","列B"], ["1","x"], ["2","y"]]
    """
    rows = []
    for line in block.strip().split('\n'):
        line = line.strip()
        if not line or not line.startswith('|') or line.count('|') < 2:
            continue
        # 按 | 分，丢掉第一个和最后一个空字符串
        cells = [cell.strip() for cell in line.split('|')[1:-1]]
        rows.append(cells)
    return rows


def extract_web_content(html):
    # 提取<title>内容
    title_match = re.search(r"<title.*?>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    title = title_match.group(1).strip() if title_match else ""

    # 提取<body>内容，去除脚本、样式等标签
    body_match = re.search(r"<body.*?>(.*?)</body>", html, re.IGNORECASE | re.DOTALL)
    body_content = body_match.group(1).strip() if body_match else ""

    # 移除<script>和<style>标签及其内容
    body_content = re.sub(r"<(script|style).*?>.*?</\1>", "", body_content, flags=re.IGNORECASE | re.DOTALL)

    # 移除所有HTML标签，只保留文本
    text_content = re.sub(r"<[^>]+>", "", body_content)
    text_content = re.sub(r"\s+", " ", text_content).strip()

    return {"title": title, "content": text_content}


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


def extract_transcription_segments(text: str) -> str:
    core = text.strip()  # 第二部分就是我们要的识别内容
    lines = []
    last_line = None
    for line in core.splitlines():
        line = line.strip()  # 可选：去掉多余空行
        if not line:
            continue
        # 去掉每行的 "[xxs - xxs]"
        line = re.sub(r"^\[[^\]]+\]\s*", "", line)
        # 合并和上一行重复的内容
        if line == last_line:
            continue
        lines.append(line)
        last_line = line

    return "\n".join(lines)


def extract_list_data(text) -> list[str]:
    list_blocks = re.findall(r'```(?:list)?(.*?)```', text, re.DOTALL)
    if not list_blocks:
        list_blocks = re.findall(r'(\n\s*[-*].*?(\n\s{2,}.*?)*\n)', text)  # 纯文本列表
    return [block.strip() for block in list_blocks]


def extract_json_data(text) -> list[str]:
    # 提取 JSON 格式的代码块
    json_blocks = re.findall(r'```(?:json)?(.*?)```', text, re.DOTALL)
    return [block.strip() for block in json_blocks]


def extract_jsons(input_str) -> list[dict]:
    """
    处理包含多个 JSON 对象的文本数据,成功解析了 JSON 对象，返回一个包含所有解析结果的列表
    :param input_str:
    :return: list[dict]
    """
    # 1,None,-1
    matches = re.findall(r'\{.*?\}', input_str, re.DOTALL)  # regex.findall(r'\{(?:[^{}]|(?R))*\}', input_str)
    if not matches:
        return None
    json_objects = []  # [var.strip() for var in matches if '{' not in var]
    for match in matches:
        try:
            json_objects.append(json.loads(match))
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e} - Skipping this fragment: {match}")

    return json_objects


def clean_any_string(raw_text):
    """
    文本清洗函数，支持多种文本格式预处理，包括邮件、URL、时间、日期、异常空格等。
    """
    pure_text = raw_text.replace('\n', " ")
    # 替换不可见字符（如 \xa0）为普通空格
    pure_text = pure_text.replace('\xa0', ' ').strip()
    pure_text = re.sub(r'[-–—]', ' ', pure_text)

    pure_text = re.sub(r"\d+/\d+/\d+", "", pure_text)  # 剔除日期
    pure_text = re.sub(r"[0-2]?[0-9]:[0-6][0-9]", "", pure_text)  # 剔除时间
    pure_text = re.sub(r"\S+@\S+", "", pure_text)  # 去除电子邮件
    # #URL，为了防止对中文的过滤，所以使用[a-zA-Z0-9]而不是\w
    url_regex = re.compile(r"""
        (https?://)?             # 协议部分可选
        ([a-zA-Z0-9-]+\.)+       # 域名部分
        [a-zA-Z]{2,}             # 顶级域名
        (/[a-zA-Z0-9-]*)*        # 路径部分可选
    """, re.VERBOSE | re.IGNORECASE)
    pure_text = url_regex.sub(r"", pure_text)
    # pure_text = re.sub("[^\u4e00-\u9fa5]","",pure_text)  #  去除所有非汉字内容（英文数字）
    # pure_text = re.sub(r"\s+", " ", pure_text).strip()  # 多余空格合并
    # 去掉多余空格和连续空格
    pure_text = re.sub(r'[^\S\r\n]+', ' ', pure_text)  # r'\s+'
    return pure_text.strip()


def clean_json_string(json_str):
    # 1. 去除 // 注释
    json_str = re.sub(r'//.*', '', json_str)
    # 2. 修复非法反斜杠：把非法的 \x 转为 x
    json_str = re.sub(r'\\(.)', fix_invalid_backslashes, json_str)
    # 3. 替换 HTML 标签、伪标签、非法换行符
    json_str = json_str.replace('<br>', '\n')  # 替换 HTML 标签或伪标签
    json_str = json_str.replace('<', '《').replace('>', '》')  # 修复 <ucam.xxx> 造成的错误
    json_str = json_str.replace('\\"', '"')  # 修复被转义的双引号（如 \\"）

    json_str = re.sub(r'"(reason|suggest)":\s*"([^"]+?)(?=\n\s*")', lambda m: f'"{m.group(1)}": "{m.group(2).strip()}"',
                      json_str)  # 尝试补全 "reason": "... \n  "suggest"

    return json_str


def clean_escaped_string(text: str, force_str: bool = True, decode_unicode: bool = False) -> str:
    """
    尝试去除外层引号，并反转义，返回字符串。
    force_str=True 时，非字符串结果也会转成 str。
    decode_unicode=True 时，尝试把 \\uXXXX 形式解码成真实字符。
    """
    try:
        result = ast.literal_eval(text)  # 自动处理 \" \\n 等
        if force_str and not isinstance(result, str):
            result = str(result)
        return result
    except:
        # fallback 处理
        text = text.strip()
        if re.fullmatch(r'''["'].*["']''', text) and text[0] == text[-1]:
            text = text[1:-1]
        if decode_unicode:
            try:
                return text.encode('utf-8').decode('unicode_escape')
            except:
                pass

    return text


def extract_json_from_string(input_str):
    # 从一个普通字符串中提取 JSON 结构，但可能不处理嵌套的 JSON
    parsed = extract_json_struct(input_str)
    if parsed:
        return parsed

    match = re.search(r'\{.*}', input_str, re.DOTALL)
    if match:
        json_str = clean_json_string(match.group(0))
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e},{input_str}")

    return extract_json_array(input_str)


def extract_links(text):
    # 提取 Markdown 格式的链接 [链接文字](链接地址)
    pattern = r'\[([^\]]+)\]\((https?://[^\s)]+)\)'
    links = re.findall(pattern, text)
    return [{'text': link[0], 'url': link[1]} for link in links]


def extract_headers(text):
    # 提取 ## 或 ### 等标题
    headers = re.findall(r'^(#{1,6})\s+(.*)', text, re.MULTILINE)
    return [{'level': len(header[0]), 'text': header[1].strip()} for header in headers]


def extract_imports(text):
    bold_texts = re.findall(r'\*\*(.*?)\*\*', text)  # 提取 Markdown 格式的 **粗体**
    italic_texts = re.findall(r'__(.*?)__|\*(.*?)\*', text)  # 提取 Markdown 格式的 __斜体__ 或 *斜体*
    italic = [italic[0] or italic[1] for italic in italic_texts]
    return {"bold": bold_texts, "italic": italic, "header": extract_headers(text)}


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


def ordinal_generator():
    ordinals = ['①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧', '⑨', '⑩']
    for ordinal in ordinals:
        yield ordinal


def remove_markdown_block(text: str) -> str:
    """
    如果文本以 ```markdown 开头并以 ``` 结尾，则移除这两个标记，返回中间内容。
    否则返回原始文本。
    """
    match = re.match(r"^```markdown\s*\n(.*?)\n?```$", text.strip(), re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


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
    text = remove_markdown_block(text)
    text = re.sub(r'(`{1,3})(.*?)\1', r'\2', text, flags=re.DOTALL)  # 去除反引号代码块
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # 去除粗体
    text = re.sub(r'\*(.*?)\*', r'\1', text)  # 去除斜体
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)  # 去除图片
    # text = re.sub(r'\[.*?\]\((.*?)\)', r'\1', text)  # 去除链接，但保留 URL
    text = re.sub(r'\[(.*?)\]\((.*?)\)', r'\1', text)  # 去除链接和 URL
    text = re.sub(r'#{1,6}\s*(.*)', r'\1', text, flags=re.MULTILINE)  # 去除标题
    text = re.sub(r'>\s*(.*)', r'\1', text, flags=re.MULTILINE)  # 去除引用块
    text = re.sub(r'(\*|-|\+)\s+(.*)', r'\2', text)  # 去除无序列表符号
    text = re.sub(r'\d+\.\s+(.*)', r'\1', text)  # 去除有序列表符号
    text = re.sub(r'~~(.*?)~~', r'\1', text)  # 去除删除线
    text = re.sub(r'_{2}(.*?)_{2}', r'\1', text)  # 去除下划线标记

    text = re.sub(r'\n{2,}', '\n', text)  # 将多余的空行替换为单个换行符,压缩空行
    return text.strip()


def format_for_wechat(text):
    formatted_text = text.split("</think>")[-1]  # text:extract_tagged_split(text, tag="think")[1]
    formatted_text = re.sub(r'\*\*(.*?)\*\*', r'✦\1✦', formatted_text)  # **粗体** 转换为 ✦粗体✦样式
    formatted_text = re.sub(r'!!(.*?)!!', r'❗\1❗', formatted_text)  # !!高亮!! 转换为 ❗符号包围
    # formatted_text = re.sub(r'__(.*?)__', r'※\1※', formatted_text)  # __斜体__ 转换为星号包围的样式
    formatted_text = re.sub(r'~~(.*?)~~', r'_\1_', formatted_text)  # ~~下划线~~ 转换为下划线包围
    formatted_text = re.sub(r'\^\^(.*?)\^\^', r'||\1||', formatted_text)  # ^^重要^^ 转换为 ||重要|| 包围
    formatted_text = re.sub(r'######\s+(.*?)(\n|$)', r'[\1]\n', formatted_text)  # ###### 六级标题
    formatted_text = re.sub(r'#####\s+(.*?)(\n|$)', r'《\1》\n', formatted_text)  # ##### 五级标题
    formatted_text = re.sub(r'####\s+(.*?)(\n|$)', r'【\1】\n', formatted_text)  # #### 标题转换
    formatted_text = re.sub(r'###\s+(.*?)(\n|$)', r'=== \1 ===\n', formatted_text)  # ### 三级标题
    formatted_text = re.sub(r'##\s+(.*?)(\n|$)', r'— \1 —\n', formatted_text)  # ## 二级标题
    formatted_text = re.sub(r'#\s+(.*?)(\n|$)', r'※ \1 ※\n', formatted_text)  # # 一级标题
    # formatted_text = re.sub(r'```([^`]+)```',
    #                         lambda m: '\n'.join([f'｜ {line}' for line in m.group(1).splitlines()]) + '\n',
    #                         formatted_text)
    # formatted_text = re.sub(r'`([^`]+)`', r'「\1」', formatted_text)  # `代码` 转换为「代码」样式
    # formatted_text = re.sub(r'>\s?(.*)', r'💬 \1', formatted_text)  # > 引用文本，转换为聊天符号包围
    # formatted_text = re.sub(r'^\s*[-*+]\s+', '• ', formatted_text, flags=re.MULTILINE)  # 无序列表项
    # formatted_text = re.sub(r'^\s*\d+\.\s+',f"{next(ordinal_iter)} ", formatted_text, flags=re.MULTILINE)  # 有序列表项
    formatted_text = re.sub(r'\n---+\n', '\n——————————————\n', formatted_text, flags=re.MULTILINE)  # 替换水平线 r'^---+$'
    formatted_text = re.sub(r'\?{4}', '✨', formatted_text)
    formatted_text = re.sub(r'\n{2,}', '\n\n', formatted_text)  # 转换换行以避免多余空行

    return formatted_text.strip()


def format_to_spans(text: str, codes: bool = True, tables: bool = True) -> str:
    """把自定义标记语法转换为 HTML span"""
    import html
    def parse_table(md: str) -> str:
        lines = [l for l in md.splitlines() if l.strip()]
        if not lines or "|" not in lines[0]:
            return md  # 不是表格
        html_table = ["<table>"]
        for i, line in enumerate(lines):
            cols = [c.strip() for c in line.split("|") if c.strip()]
            if not cols:
                continue
            tag = "th" if i == 0 else "td"
            html_table.append("<tr>" + "".join(f"<{tag}>{html.escape(c)}</{tag}>" for c in cols) + "</tr>")
        html_table.append("</table>")
        return "\n".join(html_table)

    def format_text(t: str) -> str:
        # --- 或 *** 或 ___ 作为水平分割线
        t = re.sub(r'^\s*(---|\*\*\*|___)\s*$', '<hr>', t, flags=re.M)
        # #### 或更多的标题 -> small-caps
        t = re.sub(r'^(#{4,7})\s+(.*?)(\n|$)', r'<span class="bold important small-caps">\2</span>\n',
                   t, flags=re.M)
        # ### 标题 -> large-text uppercase
        t = re.sub(r'###\s(.*?)(\n|$)', r'<span class="bold large-text uppercase">\1</span>\n', t)
        # 文本标记替换
        t = re.sub(r'\*\*(.*?)\*\*', r'<span class="bold">\1</span>', t)  # **粗体**
        t = re.sub(r'!!(.*?)!!', r'<span class="highlight">\1</span>', t)  # !!高亮!!
        t = re.sub(r'__(.*?)__', r'<span class="italic">\1</span>', t)  # __斜体__
        t = re.sub(r'~~(.*?)~~', r'<span class="underline">\1</span>', t)  # ~~下划线~~
        t = re.sub(r'\^\^(.*?)\^\^', r'<span class="important">\1</span>', t)  # ^^重要^^

        # t = re.sub(r'\n+', '\n', t).strip()
        t = re.sub(r'\n{2,}', '</p><p>', t)  # 处理多余空行 -> 段落
        t = t.replace("\n", "<br>")  # 把换行转成 <br>（避免浏览器忽略）
        # # 包裹在 <p> 中，保证 HTML 结构完整
        # if not t.startswith("<p>"):
        #     t = f"<p>{t}</p>"
        return t

    def repl_codeblock(match):
        code_content = html.escape(match.group(1))
        return f'<pre><code>{code_content}</code></pre>'

    if codes:
        # 处理 ```code```，HTML 转义内部
        text = re.sub(r'```(.*?)```', repl_codeblock, text, flags=re.S)
    if tables:
        text = parse_table(text)  # 先处理表格

    return format_text(text)


def normalize_markdown(text: str) -> str:
    # 确保标题后至少有一个空行
    def repl(m):
        title = m.group(1)
        rest = m.group(2)
        if rest.startswith("\n\n"):
            return title + rest
        else:
            return title + "\n\n" + rest

    text = text.replace("\\n", "\n")
    pattern = re.compile(r'^(#{1,6} .+?)(\n.*)', re.MULTILINE | re.DOTALL)  # 确保标题后至少有一个空行
    return pattern.sub(repl, text)


def format_for_html(text: str, html: bool = False):
    # Markdown 格式的文本转换为 HTML 的字符串,渲染 Markdown 文章
    # markdown.markdown(text, extensions=['tables', 'codehilite'])
    # from IPython.display import Markdown, display
    # display(Markdown(f"`{export_command}`"))
    style = """
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans", Arial, sans-serif; }
        table { border-collapse: collapse; width: 100%; max-width: 100%; overflow-x: auto; display: block; }
        th, td { border: 1px solid #999; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        pre { background-color: #1e1e1e; color: #d4d4d4; padding: 8px; border-radius: 5px; overflow-x: auto; font-size: 0.9em;}
        code { font-family: "Courier New", monospace; background-color: #1e1e1e; color: #d4d4d4; padding: 2px 4px; border-radius: 3px;}
    </style>
    """
    text = normalize_markdown(text)
    html_content = markdown2.markdown(text, extras=["tables", "fenced-code-blocks", "break-on-newline"])
    return f"<html><head>{style}</head><body>{html_content}</body></html>" if html else html_content


def safe_convert_html(text):
    '''过滤危险标签：防止 XSS 攻击'''
    from markdown import Markdown
    from bs4 import BeautifulSoup
    html = Markdown(extensions=['tables']).convert(text)
    soup = BeautifulSoup(html, 'html.parser')
    # 移除 script 等危险标签
    for tag in soup.find_all(['script', 'iframe']):
        tag.decompose()
    return str(soup)


def format_table_md(records: list[dict]) -> str:
    seen = []
    for r in records:
        for k in r.keys():
            if k not in seen:
                seen.append(k)
    headers = seen  # list(records[0].keys())

    lines = ["| " + " | ".join(headers) + " |",  # header_line
             "| " + " | ".join(["---"] * len(headers)) + " |"]  # divider_line
    for row in records:  # rows
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    return "\n".join(lines)


def format_content_str(content, level=0, exclude_null=True) -> str:
    if exclude_null and content is None:
        return ""  # "*（空内容）*"

    if isinstance(content, dict):
        if exclude_null and not content:
            return ""
        if all(isinstance(x, str) for x in content.keys()):
            heading_prefix = ("\n" + " " * level) if level > 0 else ''  # "#" * (level + 2)
            lines = [f"{heading_prefix}**{key}**:{format_content_str(val, level + 1, exclude_null)}"
                     for key, val in content.items()
                     if not exclude_null or val is not None]  # 统一走递归多行输出
            return "\n".join(lines)
        return json.dumps(content, ensure_ascii=False, indent=2)

    if isinstance(content, (list, tuple)):
        if exclude_null and not content:
            return ""  # "*（空列表）*"
        return "\n".join(format_content_str(x, level, exclude_null) for x in content)  # format_table/str/any

    if isinstance(content, (int, float, bool)):
        return str(content)

    return str(content).strip()


def format_summary_text(summary_data: dict | list[dict], title_map: dict = None, base_level=3) -> str:
    """
    字段中文标题映射
    将结构化 summary_data 转为分节展示文本（markdown 风格）
    - 渲染顺序以 summary_data 本身为准
    """
    title_map = title_map or {}
    numerals = "一二三四五六七八九十"

    def render_one(data: dict, index: int = None):
        values = list(data.values())
        if all(isinstance(v, (int, float, bool)) for v in values) or all(
                isinstance(v, str) and len(v) < 50 for v in values):
            return f"```json\n{json.dumps(map_fields(data, title_map), ensure_ascii=False, indent=2)}```"

        num_iter = iter(numerals)
        sections = []
        for i, (key, content) in enumerate(data.items()):
            if not content:
                continue
            if title_map and key not in title_map:
                continue
            # 自动转为字符串（支持字典或表格结构）
            num = next(num_iter, str(i + 1))
            prefix = f"{'#' * (base_level + 1)} {num}、{title_map.get(key, key)}"
            if isinstance(content, list) and all(isinstance(x, dict) for x in content):
                content_str = format_table_md(content)
            else:
                content_str = remove_markdown_block(format_content_str(content))
            sections.append(f"{prefix}\n\n{content_str}\n")

        md_text = "\n\n---\n\n".join(sections)
        if index is not None:
            return f"{'#' * base_level}  第 {index + 1} 条\n\n" + md_text
        return md_text

    if isinstance(summary_data, list):
        return "\n\n---\n\n".join(render_one(item, i) for i, item in enumerate(summary_data))
    return render_one(summary_data)


def split_summary_chunks(text: str) -> list[str]:
    """
     清洗大模型输出的文本，分成自然段 + 清除 bullet/markdown/符号前缀
    """

    def clean_lines(line: str) -> str:
        return re.sub(r"^[-–•\d\)\.\s]+", "", line).strip()  # 清除 bullet/数字/多余符号

    normalized = re.sub(r'\n{2,}', '\n\n', text.strip())
    return [clean_lines(chunk.replace("\n", " ")) for chunk in normalized.split("\n\n") if chunk.strip()]


def split_text_into_sentences(raw_text: str) -> list[str]:
    # 逐字找标点分割，使用常见的标点符号分割文本，生成句子列表
    sentence_endings = ['。', '！', '？', '；', '.', '!', '?', ';']  # 常见中文/英文标点
    sentences = []
    current_sentence = ""

    for char in raw_text:
        current_sentence += char
        if current_sentence[-1] in sentence_endings:
            sentences.append(current_sentence.strip())
            current_sentence = ""

    # 如果有残留的文本，加入句子列表
    if current_sentence.strip():
        sentences.append(current_sentence.strip())

    return sentences


def remove_parentheses(entity):
    keys = {'［', '(', '[', '（'}
    symbol = {'］': '［', ')': '(', ']': '[', '）': '（'}
    stack = []
    remove = []
    for index, s in enumerate(entity):
        if s in keys:
            stack.append((s, index))
        if s in symbol:
            if not stack: continue
            temp_v, temp_index = stack.pop()
            if entity[index - 1] == '\\':
                t = entity[temp_index - 1:index + 1]
                remove.append(t)
            else:
                remove.append(entity[temp_index:index + 1])

    for r in remove:
        entity = entity.replace(r, '')
    return entity


def split_sentences(text,
                    pattern=(r'[^一二三四五六七八九十\d\r\n]*\b[一二三四五六七八九十]+\、'  # 中文序号 "一、二、"
                             r'|[^（(）)]*\b[（(][一二三四五六七八九十]+[）)]'  # 括号内的中文序号 "(一)(二)"
                             r'|[^\d\r\n]*\b\d+\、'  # 数字序号 "1、2、"
                             r'|[^。！？]*[。！？]'  # 句号、感叹号、问号
                             r'|[^\r\n]*\r?\n'  # 换行符（支持 Windows 的 \r\n 和 Unix 的 \n）
                             )
                    ) -> list[str]:
    """
    分句函数，支持按标点符号和结构化序号进行分句，分隔符会保留在前一句结尾。结构化比较清晰的合同、制度文件。粗粒度分句（以自然语言的标点/序号为主）
    :param text: 输入的文本
    :param pattern: 正则表达式匹配分隔符
    :return: 分割后的句子列表
    """
    if not pattern:
        pattern = r'(?=[。！？])'
    sentences = re.findall(pattern, text)
    # re.split(r'[。！？\n]', text)
    return [s.strip() for s in sentences if s.strip()]


def split_sentences_clean(text, h_symbols=True, h_tables=True) -> list[str]:
    """
    合同、规章、带大量编号、条款、表格的文本,分句建模、摘要、切块处理
    篇章分句，额外支持：
      - 第X条（中国式条款）
      - (一)、(1)、(a) 等括号编号
      - 1.1、2.3.4 等多级小数编号
    :param text: str, 整段原始文本
    :param h_symbols: bool, 是否处理连续符号和换行符标准化
    :param h_tables: bool, 是否处理表格符号“|”
    :return: list of sentences
    """
    # 1. 统一换行符
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\n{2,}', '\n\n', text.strip())
    if h_symbols:
        # 2. 在各种序号后面加空格，避免与正文粘连
        # （1）中国式条款：第X条
        text = re.sub(r'(第[一二三四五六七八九十]+条)', r'\1 ', text)
        # （2）括号编号：(一)、(1)、(a)……
        text = re.sub(r'(\([一二三四五六七八九十\dA-Za-z]+\))', r'\1 ', text)
        # （3）多级小数编号：1.1、2.3.4……
        text = re.sub(r'(\d+(?:\.\d+)+)', r'\1 ', text)

        # 3. 特殊处理表格“|序号.”、“|序号、”
        text = re.sub(r'(\|\s*\d+[\.、])', r'\1 ', text)
        text = re.sub(r'(^|\n)\s*(\d+[\.、])', r'\1\2 ', text)

    if h_tables:
        # 4. 把表格分隔符 ‘|’ 看作句号
        text = text.replace('|', '。')

    # 5. 合并连续中文标点
    text = re.sub(r'[。！？；]{2,}', '。', text)

    # 6. 按中文句号、问号、叹号、分号切句
    sentences = re.split(r'(?<=[。！？；])', text)

    # 7. 去空白，过滤太短的
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 1]


def structure_aware_chunk(text, max_size: int = 1000) -> list[str]:
    """
    按结构优先切分文本，保证每块尽量不超过 max_size。
    - 按分隔符递归切分,从“自然边界”到“硬切割”，渐进式分割
    - 如果仍超过 max_size，则继续细分
    """

    # 优先级分隔符（根据经验排序）
    separators = [
        "\n\n",  # 段落边界 - 最自然
        "。\n",  # 中文句子 + 换行
        ".\n",  # 英文句子 + 换行
        "。",  # 中文句号
        ".",  # 英文句号
        "；",  # 中文分号
        ";",  # 英文分号
        "，",  # 中文逗号
        ",",  # 英文逗号（最后的手段）
    ]

    def recursive_split(t: str, sep_index: int = 0) -> list[str]:
        if len(t) <= max_size:
            return [t]

        if sep_index >= len(separators):
            # 最后一层硬切
            return [t[i:i + max_size] for i in range(0, len(t), max_size)]

        sep = separators[sep_index]
        if sep in t:
            parts = t.split(sep)
            result = []
            for part in parts:
                if not part.strip():
                    continue
                sub_chunks = recursive_split(part, sep_index + 1)
                # 把分隔符加回到每个 sub_chunk 的结尾（除了最后一个）
                for i, sc in enumerate(sub_chunks):
                    if i < len(sub_chunks) - 1 or part.endswith(sep):
                        result.append(sc + sep)
                    else:
                        result.append(sc)
            return result

        return recursive_split(t, sep_index + 1)

    chunks = recursive_split(text)
    return [c.strip() for c in chunks if c.strip()]


def cross_sentence_chunk(sentences: list[str], chunk_size=5, overlap_size=2, max_length=512,
                         model_name="gpt-3.5-turbo"):
    """
    滑动窗口分块 + 最大长度截断（只测长度，不用 tokenizer.decode）
    :param sentences: 分句后的句子列表
    :param chunk_size: 每块包含几个句子
    :param overlap_size: 相邻块重叠几个句子
    :param max_length: 最大长度（token数或字符数）
    :param  model_name: 用于计算 token 长度的分词器
    :return: List[str] 每块一个字符串
    """
    tokenizer = get_tokenizer(model_name)
    chunks = []
    step = max(chunk_size - overlap_size, 1)

    for i in range(0, len(sentences), step):
        window = sentences[i: i + chunk_size]
        text = " ".join(window)

        if tokenizer:
            token_count = len(tokenizer.encode(text))
            if token_count > max_length:
                # 直接按字符截断，可能句子割裂
                text = text[: max_length]
        else:
            # 用字符长度作为 fallback
            if len(text) > max_length:
                text = text[: max_length]

        chunks.append(text)

    return chunks


def organize_segments_chunk(sentences: list[str], chunk_size=7, overlap_size=2, max_length=1024,
                            tokenizer=None) -> list[list[str]]:
    """
    交叉分块函数，将句子列表按块划分，并在块之间保持一定重叠，并根据max_length控制每个段落的最大长度。
    :param sentences: 分句后的句子列表 split_sentences_clean
    :param chunk_size: 每个块的句子数量
    :param overlap_size: 块之间的重叠句子数
    :param max_length: 每个块的最大长度（token数）
    :param tokenizer: 用于计算token长度的分词器模型（Tokenizer）
    :return: 交叉分块后的句子块列表
    """
    # Step 1: 构建基础块
    base_chunks = []
    current = []
    for sent in sentences:
        if lang_token_size(" ".join(current + [sent]), tokenizer=tokenizer) <= max_length:
            current.append(sent)
        else:
            if current:
                base_chunks.append(current)
            # 单句过长 -> 再硬切， 用 tokenizer 只测长度，不 decode
            if lang_token_size(sent, tokenizer=tokenizer) > max_length:
                sub_chunks = structure_aware_chunk(sent, max_size=max_length)  # 单句也超长，按标点强制成块
                base_chunks.extend([[sc] for sc in sub_chunks])
                current = []
            else:
                current = [sent]
    if current:
        base_chunks.append(current)  # 添加当前块

    # Step 2: 滑动窗口，处理滑动窗口重叠，组织大片段
    overlapped = []
    step = chunk_size - overlap_size
    for i in range(0, len(base_chunks), step):
        merged = []
        for j in range(i, min(i + chunk_size, len(base_chunks))):
            merged.extend(base_chunks[j])

        # 确保 merged 不超过 max_length
        temp = []
        for s in merged:
            if lang_token_size(" ".join(temp + [s]), tokenizer=tokenizer) > max_length:
                overlapped.append(temp)
                temp = [s]
            else:
                temp.append(s)
        if temp:
            overlapped.append(temp)

    return overlapped


# 实现小到大分块逻辑
def organize_segments(tokens: list[int | str], small_chunk_size: int = 175, large_chunk_size: int = 512,
                      overlap: int = 20):
    '''
    小块适合用于查询匹配，提高查询的精准度。
    大块划分，将包含上下文信息的多个小块合并为较大的片段。
    滑动窗口：为了保持上下文关系，在小块和大块之间添加一定的重叠区域，确保边缘信息不丢失。这样，查询结果能保持更高的连贯性。
    '''

    # 小块分割
    small_chunks = []
    for i in range(0, len(tokens), small_chunk_size - overlap):
        small_chunks.append(tokens[i:i + small_chunk_size])  # ''.join()

    # 组织大片段
    large_chunks = []
    for i in range(0, len(small_chunks), large_chunk_size // small_chunk_size):
        large_chunk = []
        for j in range(i, min(i + large_chunk_size // small_chunk_size, len(small_chunks))):
            large_chunk.extend(small_chunks[j])
        large_chunks.append(large_chunk[:large_chunk_size])

    return small_chunks, large_chunks


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

        "md": extract_markdown_blocks,
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


def extract_string(text, extract: str | list, **kwargs):
    if not extract or extract == 'raw':
        return None
    funcs = {
        "jsons": extract_jsons,
        "json": extract_json_struct,
        "json_array": extract_json_array,
        "json_object": extract_json_from_string,

        "header": extract_headers,
        "imports": extract_imports,
        "links": extract_links,
        'urls': extract_text_urls,

        "tables": extract_table_blocks,
        "table_segments": extract_table_segments,
        "table_data": extract_markdown_table,
        "answer": extract_tagged_content,
        "think": partial(extract_tagged_split, tag="think"),
        "reasoning": partial(extract_tagged_split, tag="reasoning"),
        'clean_any': clean_any_string,
        'sentence': split_sentences,
        'sentences_clean': split_sentences_clean,
        'split_chunks': split_summary_chunks,
        "wechat": format_for_wechat,
        'remark': remove_markdown,
        "html": format_for_html,
        "spans": format_to_spans,
        "web": extract_web_content,
    }

    def run_extract(e):
        if e in funcs:
            return funcs[e](text, **kwargs)  # str,list,dict,set
        extract_type = e.split('.')
        if extract_type[0] == 'code':
            return extract_code_blocks(text, lag=extract_type[1] if len(extract_type) > 1 else '', **kwargs)
        return None

    try:
        if isinstance(extract, str):
            result = run_extract(extract)
            if result:
                return result

            if extract not in funcs:
                return {k: f(text, **kwargs) for k, f in funcs.items()}  # "type": "all", dict

        elif isinstance(extract, list):
            results = {}
            for e in extract:
                val = run_extract(e)
                if val is not None:
                    results[e] = val
            if results:
                return results  # dict

    except Exception as e:
        print(e)

    return None
