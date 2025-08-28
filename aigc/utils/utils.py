import re, json, io, os, threading, time
import  aiofiles, asyncio
from itertools import groupby
from pathlib import Path
from difflib import get_close_matches, SequenceMatcher
from collections import Counter, deque, defaultdict
from pypinyin import lazy_pinyin
from utils.base import *


async def embed_images_as_base64(md_content, image_dir):
    """异步将Markdown中的图片转换为Base64并嵌入到Markdown中"""
    lines = md_content.split('\n')
    new_lines = []

    for line in lines:
        if line.startswith("![") and "](" in line and ")" in line:
            start_idx = line.index("](") + 2
            end_idx = line.index(")", start_idx)
            img_rel_path = line[start_idx:end_idx]

            img_name = os.path.basename(img_rel_path)
            img_path = os.path.join(image_dir, img_name)

            if os.path.exists(img_path):
                # 异步读取并转换图片为Base64
                async with aiofiles.open(img_path, 'rb') as img_file:
                    img_data = await img_file.read()
                    img_base64 = base64.b64encode(img_data).decode('utf-8')

                img_extension = os.path.splitext(img_name)[-1].lower()
                # 根据扩展名确定 MIME 类型
                if img_extension in ['.jpg', '.jpeg']:
                    mime_type = 'image/jpeg'
                elif img_extension == '.gif':
                    mime_type = 'image/gif'
                else:
                    mime_type = 'image/png'
                # 修改Markdown中的图片路径为Base64编码
                new_line = f'{line[:start_idx]}data:{mime_type};base64,{img_base64}{line[end_idx:]}'
                new_lines.append(new_line)
            else:  # 图片文件不存在，保留原始Markdown格式
                new_lines.append(line)
        else:  # 保留非图片链接的原始行
            new_lines.append(line)

    return '\n'.join(new_lines)


def qwen3_think_index(output_ids):
    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0
    return index


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

def dict2xml(tag, d):
    """将字典转换为 XML 字符串"""
    import xml.etree.ElementTree as ET
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


def list2xml(tag, lst):
    """将列表转换为 XML 字符串"""
    import xml.etree.ElementTree as ET
    elem = ET.Element(tag)
    for item in lst:
        item_elem = ET.SubElement(elem, "item")
        item_elem.text = str(item)
    return ET.tostring(elem, encoding='unicode')


def df2markdown(df, index=False):
    # df.fillna('N/A').to_markdown()
    headers = df.columns.tolist()
    md = "| " + " | ".join(headers) + " |\n"
    md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for _, row in df.iterrows():
        md += "| " + " | ".join(row.astype(str)) + " |\n"

    return md if index else md.replace("| Index |", "|")  # 可选移除索引列


def df2doc(data, use_index=True) -> list[str]:
    """
    将 DataFrame 中每一行转换为一段文本，跳过 None 值
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


def df2doc_batch(records, batch_size: int = 5):
    """
    将 DataFrame 或列表数据按 batch_size 分批，yield 每个批次的记录（列表 of dicts）。
    """
    try:
        import pandas as pd
        if isinstance(records, pd.DataFrame):
            records = records.to_dict(orient='records')
    except ImportError:
        if not isinstance(records, list) or not all(isinstance(d, dict) for d in records):
            raise ValueError("输入数据应为列表的字典格式，例如 [{'key1': 'value1', 'key2': 'value2'}, ...]")
    except Exception as e:
        print(e)

    batch = []
    for i, item in enumerate(records):
        batch.append(item)
        # 每 batch_size 组一个 batch
        if (i + 1) % batch_size == 0 or i == len(records) - 1:
            yield batch
            batch = []


def df2doc_split(records, max_tokens: int = 4000, tokenizer=None):
    """
    将 DataFrame 或列表数据按 token 数分块，yield 每个分块（list[str]）。
    for block in df2doc_split(...
    :param records: pandas.DataFrame 或 list[dict]
    :param max_tokens: 每个分块允许的最大 token 数
    :param tokenizer: tokenizer 对象（可选），如果没有则用 len(str) 近似
    :return: generator，每次 yield 一个 list[dict]，即一块记录
    """
    try:
        import pandas as pd
        if isinstance(records, pd.DataFrame):
            records = records.to_dict(orient="records")
    except ImportError:
        if not isinstance(records, list) or not all(isinstance(d, dict) for d in records):
            raise ValueError("输入数据应为列表的字典格式，例如 [{'key1': 'value1'}, ...]")

    chunk = []
    current_tokens = 0

    for item in records:
        item_str = json.dumps(item, ensure_ascii=False)
        item_tokens = lang_token_size(item_str, tokenizer=tokenizer)

        # 如果当前记录本身就超过 max_tokens，则单独作为一个块输出
        if item_tokens > max_tokens:
            if chunk:
                yield chunk
                chunk = []
                current_tokens = 0
            yield [item]
            continue

        # 累加后超过 max_tokens，先输出当前块
        if current_tokens + item_tokens > max_tokens:
            yield chunk
            chunk = []
            current_tokens = 0

        chunk.append(item)
        current_tokens += item_tokens

    if chunk:
        yield chunk


def get_max_items_from_list(records: list[dict], max_tokens: int = 4000, tokenizer=None) -> list[dict]:
    """
        Get max items from list of items based on defined max tokens (based on openai compute)
        根据给定的最大 token 数，从一组字典数据中选取适合的项目，直到达到 token 限制为止,从头开始
        :param records: 包含字典的列表，每个字典表示一个项目
        :param max_tokens: 允许的最大 token 数
        :param tokenizer: 可选的 tokenizer（如果没有提供，则根据语言自动处理）
        :return: 适合的项目列表
        List[Dict[str, str]]
    """
    # encoding = tiktoken.encoding_for_model(encoding_name)
    # tiktoken.get_encoding("cl100k_base")
    return next(df2doc_split(records, max_tokens=max_tokens, tokenizer=tokenizer))


def get_last_entries_records(records: list[dict], fields: list = None, use_index: bool = False, max_tokens: int = 8000,
                             tokenizer=None) -> list[str]:
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
        entry_length = lang_token_size(item_str, tokenizer=tokenizer)  # len(item_str)
        if total_chars + entry_length > max_tokens:
            break

        texts.append(item_str)
        total_chars += entry_length

    # 如果有多个记录，倒序拼接（保证最早的记录在最前面）
    return list(reversed(texts))  # "\n\n".join(reversed(texts))


def find_similar_word(target_keyword, template_list: list[str]) -> int:
    max_ratio = 0
    similar_word_index = -1
    for i, token in enumerate(template_list):
        ratio = SequenceMatcher(None, target_keyword, token).ratio()
        if ratio > max_ratio:
            max_ratio = ratio
            similar_word_index = i
    return similar_word_index


def find_best_matches(query: str, template_list: list[str], top_n=3, cutoff=0.8, best=True) -> list[tuple]:
    # 获取满足 cutoff 的匹配
    matches = get_close_matches(query, template_list, n=top_n, cutoff=cutoff)
    # 计算每个匹配项与查询词的相似度
    if matches:
        return [(match, SequenceMatcher(None, query, match).ratio(), template_list.index(match))
                for match in matches]
    # 如果没有匹配，则强制返回最相似的 1 个
    if best and template_list:
        scores = [(text, SequenceMatcher(None, query, text).ratio(), i)
                  for i, text in enumerate(template_list)]
        return [max(scores, key=lambda x: x[1])]
        # sorted( [item for item in scores if item[1] >= cutoff], key=lambda x: -x[1])[:top_n]

    return []  # text, score, idx [(匹配文本, 相似度, 对应原始idx)]


def fuzzy_match_template(query, template_list: list[str], threshold=0.8):
    if not isinstance(query, str):
        return None

    matches = get_close_matches(query, template_list, n=1, cutoff=threshold)
    return matches[0] if matches else None


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


def build_prompt(messages: list, use_role=False) -> str:
    """
    Build a single prompt string from a list of messages.
    Each message is expected to be a dictionary with 'role' and 'content' keys.
    This function concatenates all message contents, preserving the training format.
    """
    if all(isinstance(msg, str) for msg in messages):
        return '\n\n'.join(msg.strip() for msg in messages if msg.strip())
    if use_role:
        # OpenAI-style messages are transformed to a structured conversation format for Ollama.
        return "\n".join(
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content'].strip()}"
            for msg in messages if msg.get('role', 'user') != 'system')
    return "\n\n".join(msg["content"].strip() for msg in messages)


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


def cut_chat_history(user_history: list[dict], max_size=33000, max_pairs=0, model_name="gpt-3.5-turbo"):
    """
    根据 token 数截断对话历史，保留最近的上下文。

    :param user_history: 完整的消息列表，每项 {'role':..., 'content':...}
    :param max_size: 最大允许的 token 数
    :param max_pairs: 最大允许的 消息对数
    :param  model_name: tokenizer model
    :return: 截断后的消息列表
    32K tokens
    64K tokens
    128K token
    """

    if max_size <= 0 and max_pairs <= 0:
        return user_history

    tokenizer = get_tokenizer(model_name)

    pair_count = 0
    total_size = 0
    last_records = []

    i = len(user_history) - 1
    while i >= 0:
        if i >= 1:
            # 尝试组成 user+assistant 对
            if user_history[i - 1].get("role") == "user" and user_history[i].get("role") == "assistant":
                pair = user_history[i - 1:i + 1]
                pair_len = sum(lang_token_size(record.get("content", ""), tokenizer) for record in pair)

                if pair_count >= max_pairs > 0:
                    break
                if total_size + pair_len > max_size > 0:
                    break

                last_records = pair + last_records
                total_size += pair_len
                pair_count += 1
                i -= 2
                continue

        # 否则单条处理（如开头或非成对）
        pair = [user_history[i]]
        pair_len = lang_token_size(pair[0].get("content", ""), tokenizer)

        if total_size + pair_len > max_size > 0:
            break

        last_records = pair + last_records
        total_size += pair_len
        i -= 1

    return last_records


def split_whitespace_nonwhitespace(s, max_len=5):
    # 按照 空白/非空白 交替拆分字符串，控制每段的最大长度，预切割
    for k, g in groupby(s, key=str.isspace):
        chunk = list(g)
        for i in range(0, len(chunk), max_len):
            yield ''.join(chunk[i:i + max_len])


LINE_STOP_FLAG = (
    '.', '!', '?', '。', '！', '？', ')', '）', '"', '”', ':', '：', ';', '；', ']', '】', '}', '}', '>', '》', '、', ',', '，',
    '-', '—', '–',)
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


class AsyncAbortController:
    def __init__(self):
        self._abort_event = asyncio.Event()  # 线程安全 threading.Event()

    def should_abort(self) -> bool:
        """查询是否已触发终止信号,实时检查是否终止"""
        return self._abort_event.is_set()

    async def wait_abort(self):
        """等待终止信号（可用于并发 await）,可 await 等待中断"""
        await self._abort_event.wait()

    def abort(self):
        """外部触发终止信号,触发终止,是否提前终止,可供外部触发"""
        self._abort_event.set()

    def reset(self):
        """清除终止信号，为下一轮任务做准备,重新启动前复位,可多轮复用"""
        self._abort_event.clear()


LLM_Controller = AsyncAbortController()


async def process_llm_stream(llm_responses_stream, token_size=20, model_name="gpt-3.5-turbo"):
    """
    处理大模型返回的文本流，并按标点符号分割交给 TTS 朗读。
    :param llm_responses_stream: 大模型返回的文本流
    :param token_size: 标点不足时，允许的最小缓冲区长度
    :param  model_name: tokenizer model
    """
    response_message = []
    text_index = 0
    processed_chars = 0
    tokenizer = get_tokenizer(model_name)
    async for content in llm_responses_stream:
        response_message.append(content)
        if LLM_Controller.should_abort():  # 实时检查是否终止
            break

        # 获取当前未处理的文本
        full_text = "".join(response_message)
        current_text = full_text[processed_chars:]

        # 查找最后一个有效标点
        last_punct_pos = find_last_punctuation(current_text)
        if last_punct_pos != -1 or lang_token_size(current_text, tokenizer) > token_size:
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
    LLM_Controller.reset()  # 重新启动前复位
    async for text, idx in process_llm_stream(new_llm_stream):
        if idx > 0:
            print(f"🔊 朗读: {text}")


# 支持的扩展名
def get_local_suffix(folder_path, supported_suffix=None, recursive=False):
    supported_extensions = (ext.lower() for ext in supported_suffix or [".jpg", ".jpeg", ".png", ".bmp"])
    folder = Path(folder_path)
    if not folder.is_dir():
        raise ValueError(f"The path '{folder_path}' is not a valid directory.")
    pattern = "**/*" if recursive else "*"
    return [str(f_path) for f_path in folder.glob(pattern) if f_path.suffix.lower() in supported_extensions]


def get_doc_type(text):
    doc_type = 'text'
    sample = text[:5000].strip()

    # 1. 代码块（Markdown 风格）
    if '```' in sample and sample.count('```') > 2:
        return 'code'

    # 2. Markdown 文档
    if sample.count('#') > 5 or re.search(r'\[.*?\]\(.*?\)', sample):
        return 'md'

    # 3. JSON 格式
    if sample.startswith('{') or sample.startswith('['):
        try:
            import json
            json.loads(sample)
            return 'json'
        except Exception:
            pass

    # 4. HTML/XML
    if re.search(r'<(html|div|span|head|body|title)[\s>]', sample, re.IGNORECASE):
        return 'html'

    # 5. CSV / TSV
    if sample.count(',') > 5 and re.search(r'(?:\n|^)([^,\n]+,){2,}', sample):
        return 'csv'
    if '\t' in sample and re.search(r'(?:\n|^)([^\t\n]+\t){2,}', sample):
        return 'tsv'

    # 6. 日志文件
    if re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', sample):
        return 'log'

    # 默认文本
    return doc_type


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
