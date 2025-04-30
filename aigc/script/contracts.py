import script.aigc as aigc

aigc.AIGC_HOST = '47.110.156.41'

import pandas as pd
from openai import OpenAI, AsyncOpenAI
from config import Config
from utils import *
from tqdm.asyncio import tqdm_asyncio

max_concurrent_tasks = 5
sema = asyncio.Semaphore(max_concurrent_tasks)


async def files_messages(file_path_obj, question, client, name='qwen-long', **kwargs):
    messages = []
    async with sema:
        file_object = await client.files.create(file=file_path_obj, purpose="file-extract")
        messages.append({"role": "system", "content": f"fileid://{file_object.id}", })
        messages.append({"role": "user", "content": question})
        print(file_object.id, str(file_path_obj))
        await asyncio.sleep(0.05)
        completion = await client.chat.completions.create(model=name, messages=messages, **kwargs)
        if not completion.choices or not hasattr(completion.choices[0], "message"):
            raise ValueError(f"Incomplete response for file {file_path_obj}")
        bot_response = completion.choices[0].message.content
        messages.append({"role": "assistant", "content": bot_response})

    # file_content =client.files.content(file_id=file_object.id)  # 文件内容
    # file_content.text
    return messages, completion.model_dump()


def get_pdf_files(path):
    matched_files = []
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(('.pdf')):
                full_path = os.path.join(dirpath, filename)
                matched_files.append(full_path)
    return matched_files


async def process_one_contract(file_path_obj, question, client, folder_path):
    json_path = file_path_obj.with_suffix(".json")  # 自动将 .pdf 替换为 .json
    # if json_path.exists():
    #     async with aiofiles.open(json_path, 'r', encoding='utf-8') as f:
    #         content = await f.read()
    #         json.loads(content)
    try:
        messages, completion = await files_messages(file_path_obj, question, client, name='qwen-long')

        # print(i, file_path_obj)
        # all_msg[str(file_path_obj)] = messages

        content = messages[-1]['content']
        match = re.search(r'\{.*}', content, re.DOTALL)
        json_data = {}
        if match:
            json_str = match.group(0)
            try:
                json_data = json.loads(json_str)
            except json.JSONDecodeError as e1:
                try:
                    json_str = re.sub(r'//.*', '', json_str)  # 1. 去掉 // 注释

                    def fix_invalid_backslashes(match):
                        char = match.group(1)
                        if char in '"\\/bfnrtu':  # JSON 里合法的转义字符只有这些： " \ bfnrtu
                            return '\\' + char  # 合法的保留
                        else:
                            return '\\\\' + char  # 非法的补成 \\ + 字符

                    json_str = re.sub(r'\\(.)', fix_invalid_backslashes, json_str)
                    json_data = json.loads(json_str)
                except json.JSONDecodeError as e2:
                    print(f"Error decoding JSON: {e2},{content}")

        relative_path = os.path.relpath(file_path_obj, folder_path)
        first_folder = relative_path.split(os.sep)[0] if os.sep in relative_path else ''

        data = {
            'fileid': messages[0]['content'],
            'path': relative_path,
            'folder': first_folder,
            'content': content,
            '合同标题': json_data.get('合同标题', ''),
            '甲方': json_data.get('甲方', ''),
            '签署日期': json_data.get('签署日期', ''),
            '合同期限': json_data.get('合同期限', ''),
            '需求模块': json_data.get('需求模块', ''),
            '需求摘要': json_data.get('需求摘要', '')
        }
        # res_data.append(data)
        await  asyncio.sleep(0.05)

        async with aiofiles.open(json_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(messages, ensure_ascii=False, indent=2))
            # json.dump(messages, f, ensure_ascii=False, indent=2)

        return str(file_path_obj), messages, data

    except Exception as e:
        print(f"[ERROR] {file_path_obj}: {e}")
        return str(file_path_obj), [], {}


async def get_contracts(client, folder_path):
    matched_files = get_pdf_files(folder_path)
    question = r'''
    你是一位合同分析专家，擅长从复杂的合同文本中提取关键信息。现在，请你完成以下任务：

    1. 从合同中提取“需求功能模块”部分（这部分可能位于服务范围、功能需求、技术条款、项目描述中），并尽量保留原文。
    2. 基于“需求功能模块”的原文，生成一段简明的需求摘要。
    3. 提取合同的基本信息，包括：
       - 合同标题
       - 甲方
       - 合同期限
       - 签署日期

    请将提取结果按以下 JSON 格式返回，字段缺失可以留空，但不要改变字段名：

    注意事项：
    - 输出必须是合法 JSON
    - 所有字符串中不能包含非法的 `\` 字符（如 `\<br>`，应改为 `\\<br>` 或换成 HTML 实体）
    - 不要出现多余注释、markdown 代码块或多余的符号，只输出 JSON 本体。


    ```json
    {
      "合同标题": "",
      "甲方": "",
      "合同期限": "",
      "签署日期": "",
      "需求模块": "",
      "需求摘要": ""
    }
    '''

    tasks = []
    for i, file_path in enumerate(matched_files):
        file_path_obj = Path(file_path)
        # if i > 3:  # do try
        #     break
        if file_path_obj.exists():  # .is_file()
            tasks.append(process_one_contract(file_path_obj, question, client, folder_path))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    all_msg = {path: msg for path, msg, _ in results if msg}
    res_data = [data for _, _, data in results if data]

    os.makedirs('data', exist_ok=True)
    with open("data/合同解析消息.pkl", "wb") as f:
        pickle.dump(all_msg, f)

    with open("data/合同解析结果.pkl", "wb") as f:
        pickle.dump(res_data, f)

    return all_msg, res_data


def build_ordered_sentences(raw_text):
    """
    返回一个列表，按原文顺序：正文句子 和 表格行句子 相互交替
    """
    segments = extract_table_segments(raw_text)
    ordered = []
    for typ, seg in segments:
        if typ == 'text':
            ordered.extend(split_sentences_clean(seg))
        else:  # typ == 'table'
            table = parse_table_block(seg)
            # 把表格每行拼成一句话
            for row in table:
                # 跳过纯分隔行
                if all(cell.strip().startswith('-') for cell in row):
                    continue
                ordered.append(' | '.join(row) + '。')
    return ordered


async def contracts_sentences_split(raw_text):
    system = '''你是一个合同分句助手。
    任务：将给定的合同原文拆分成独立的句子或表格块，并返回一个扁平的 JSON 数组，数组元素为字符串，方便后续做 embedding。

    规则：
    1. **表格识别**
       - 先扫描全文，连续多行以竖线“|”开头或结尾的内容，整块识别为一个“表格”，不在表格内做分句。
       - 将表格块中第一行（表头）单独作为一条输出，请保持表格的结构性和连贯性。表格的每一行应包括表头，并且应视每行内容为一个完整的信息单元。
       - 保留整行文本及其中的“|”分隔符。保留表格的表头，并为每个表格行附带表头信息。

    2. **普通文本分句**
       - 对非表格区域，按中文句号（。）、问号（？）、感叹号（！）、分号（；）、英文分号（;）或换行符为边界拆分，保留标点在句尾。
       - 各类结构化编号（如“第1条”、“2.1”、“(一)”、“1)”、“(2)”、“1、”）必须与其后文本保持在同一句，不可拆分编号与正文的关联。
       - 对于复杂的条款或条目，应保持条目之间的逻辑关系。
       - 去除每条句子首尾多余空白。

    3. **输出格式**
       - 一旦文本处理完成，按适当的分句进行处理，确保每个分句可以直接用作文本嵌入（embedding）。每个分句应包含相关上下文信息（如条款名称或表头信息），确保后续处理时能保持上下文。
       - 返回一个 JSON 数组，扁平列出所有句子和表格行，示例如下：

    示例输入：
    2.1功能性需求 (一)建设全国统一集中账户管理前置系统(单位账户) 本系统建立目标是采集行内单位账户数据、完善数据校验、实现行内数据处理流程、与人民银行数据交换管理平台(DEMP)对接，实现单位账户直联报送。主要功能有：1、源数据采集 系统需要提供两种方式完成与苏州银行内部的源数据采集，包括：1)文件接口方式：我行系统按照人行接口规范的要求生成相应的报送数据文件，并将文件放入文件监控目录区，本系统文件监控模块通过任务调度程序，实时监听文件，并对文件名的格式按照接口规范要求进行校验。2)接口实时采集方式：本系统提供web接口与我行柜面系统对接，实时交互需要报送的源数据，同时系统按照人行的接口规范要求，对数据进行校验。2、影像信息采集 对接行内影像平台，实现账户信息配套影像文件的抓取工作。
    第1条 合同目的：本合同旨在明确权利义务。
        序号	项目	说明
        1	功能A	实现A功能；
        2	功能B	实现B功能。...
    示例输出：
    ```json
    [
      "2.1功能性需求 (一)建设全国统一集中账户管理前置系统(单位账户) 本系统建立目标是采集行内单位账户数据、完善数据校验、实现行内数据处理流程、与人民银行数据交换管理平台(DEMP)对接，实现单位账户直联报送。",
      "主要功能有：1、源数据采集 系统需要提供两种方式完成与苏州银行内部的源数据采集，包括：1)文件接口方式：我行系统按照人行接口规范的要求生成相应的报送数据文件，并将文件放入文件监控目录区，本系统文件监控模块通过任务调度程序，实时监听文件，并对文件名的格式按照接口规范要求进行校验。",
      "主要功能有：2)接口实时采集方式：本系统提供web接口与我行柜面系统对接，实时交互需要报送的源数据，同时系统按照人行的接口规范要求，对数据进行校验。",
      "2、影像信息采集 对接行内影像平台，实现账户信息配套影像文件的抓取工作。",
      "第1条 合同目的：本合同旨在明确权利义务。",
      "| 序号 | 项目 | 说明 |",
      "|序号 1 |项目 功能A |说明 实现A功能； |",
      "|序号 2 |项目 功能B |说明 实现B功能。 |",
      "第2条 付款方式：甲方付款。"
      ...
    ]
    请根据以上规则，对以下合同原文进行分句，并返回扁平的 JSON 数组
    '''

    content = await aigc.ai_chat_async(user_request=raw_text, system=system, model="deepseek:deepseek-chat",
                                       host=aigc.AIGC_HOST, time_out=300, get_content=True)

    try:
        parsed = json.loads(content)  # 如果返回的是 JSON 字符串
        # 如果模型已经返回了列表，可以直接返回
        if isinstance(parsed, list):
            return content, [s.strip() for s in parsed if isinstance(s, str) and s.strip()]
        else:
            raise ValueError("Unexpected response format.")
    except json.JSONDecodeError:
        print("Error: Response is not a valid JSON.")
        return content, []
    except Exception as e:
        print(f"Error processing response: {e}")
        return content, []


async def contracts_df(fmt_data):
    df = pd.DataFrame(fmt_data)
    df['需求字数'] = df.需求内容.str.len()
    df['摘要字数'] = df.需求摘要.str.len()
    df.to_excel('data/合同需求提取.xlsx')
    df['chunks'] = df['需求内容'].map(
        lambda txt: cross_sentence_chunk(build_ordered_sentences(txt), 8, 2, 700, tokenizer=None))

    async def process_row(text):
        async with sema:
            return await contracts_sentences_split(text)

    tasks = [process_row(txt) for txt in df['需求内容']]
    results = await tqdm_asyncio.gather(*tasks)
    contents, chunk_lists = zip(*results)

    with open("data/合同分句结果.pkl", "wb") as f:
        pickle.dump(contents, f)

    df["chunks"] = chunk_lists
    flat = df.reset_index().explode('chunks').rename(
        columns={'index': '原始行索引', 'chunks': 'chunk_text'}).reset_index(drop=True)
    flat['chunk_len'] = flat['chunk_text'].str.len()
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import normalize
    # from sentence_transformers import SentenceTransformer
    # from transformers import AutoTokenizer
    # text_encoder = SentenceTransformer('BAAI/bge-large-zh-v1.5')
    # tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5')
    # embeddings = text_encoder.encode(flat['chunk_text'].tolist(), show_progress_bar=True, normalize_embeddings=True)
    return df, flat


if __name__ == "__main__":
    import nest_asyncio

    nest_asyncio.apply()


    async def main():
        client: AsyncOpenAI = AsyncOpenAI(api_key=Config.DashScope_Service_Key,
                                          base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")  # OpenAI

        client = client.with_options(timeout=500, max_retries=3)

        all_msg, fmt_data = await get_contracts(client, folder_path=r'E:\Documents\Jupyter\data\source\合同')

        await contracts_df(fmt_data)


    asyncio.run(main())
