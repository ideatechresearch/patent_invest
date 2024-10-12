import httpx
import asyncio
from typing import List, Any, Callable
import random, time
from openai import OpenAI, Completion
import dashscope
from dashscope.audio.tts import ResultCallback, SpeechSynthesizer
from dashscope.audio.asr import Recognition, Transcription
from config import *
from utils import *

# 模型编码:0默认，1小，-1最大
AI_Models = [
    # https://platform.moonshot.cn/console/api-keys
    {'name': 'moonshot', 'type': 'default', 'api_key': '',
     "model": ["moonshot-v1-32k", "moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"],
     'url': "https://api.moonshot.cn/v1/chat/completions", 'base_url': "https://api.moonshot.cn/v1"},
    # https://open.bigmodel.cn/console/overview
    {'name': 'glm', 'type': 'default', 'api_key': '',
     "model": ["glm-4-air", "glm-4-flash", "glm-4-air", "glm-4", "glm-4v", "glm-4-0520"],
     'url': 'https://open.bigmodel.cn/api/paas/v4/chat/completions',
     'base_url': "https://open.bigmodel.cn/api/paas/v4/"},
    # https://dashscope.console.aliyun.com/overview
    {'name': 'qwen', 'type': 'default', 'api_key': '',
     "model": ["qwen-turbo", "qwen1.5-7b-chat", "qwen1.5-32b-chat", "qwen2-7b-instruct", "qwen2.5-32b-instruct",
               'qwen-long', "qwen-turbo", "qwen-plus", "qwen-max"],  # "qwen-vl-plus"
     'embedding': ["text-embedding-v2", "text-embedding-v1", "text-embedding-v2", "text-embedding-v3"],
     'speech': ['paraformer-v1', 'paraformer-8k-v1', 'paraformer-mtl-v1'],
     'url': 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
     'base_url': "https://dashscope.aliyuncs.com/compatible-mode/v1"},
    # https://cloud.baidu.com/doc/WENXINWORKSHOP/s/clntwmv7t#%E8%AF%B7%E6%B1%82%E8%AF%B4%E6%98%8E
    # https://cloud.baidu.com/doc/WENXINWORKSHOP/s/mlm0nonsv#%E5%AF%BC%E5%85%A5hf%E7%B3%BB%E5%88%97%E6%A8%A1%E5%9E%8B
    {'name': 'ernie', 'type': 'baidu', 'api_key': '',
     "model": ["ERNIE-4.0-8K", "ERNIE-3.5-8K", "ERNIE-4.0-8K", "ERNIE-4.0-8K-Preview", "ERNIE-4.0-8K-Latest",
               "ERNIE-3.5-128K"],
     'url': "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro",
     'base_url': "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/"},
    # https://console.bce.baidu.com/ai/#/ai/nlp/overview/index
    {'name': 'baidu', 'type': 'baidu', 'api_key': '',
     "model": ["txt_mone", "address", "simnet", "word_emb_sim", "ecnet", "text_correction", "keyword", "topic"],
     "completions": ['sqlcoder_7b', 'CodeLlama-7b-Instruct', 'Yi-34B'],
     'url': "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/completions",
     'base_url': "https://aip.baidubce.com/rpc/2.0/nlp/v1/"},
    {'name': 'llama', 'type': 'baidu', 'api_key': '',
     'model': ['llama_3_8b', 'Qianfan-Chinese-Llama-2-7B', 'Qianfan-Chinese-Llama-2-7B-32K', 'llama_3_8b',
               'Llama-2-13B-Chat', 'ChatGLM3-6B', 'ChatGLM2-6B-32K', 'ChatGLM3-6B-32K', 'Baichuan2-7B-Chat',
               'Baichuan2-13B-Chat'],
     'url': "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/",
     'base_url': ''},
    # https://console.volcengine.com/ark/region:ark+cn-beijing/endpoint?config=%7B%7D
    # https://console.volcengine.com/ark/region:ark+cn-beijing/model?vendor=Bytedance&view=LIST_VIEW
    {'name': 'doubao', 'type': 'default', 'api_key': '',
     "model": ['ep-20240919160005-chzhb', 'ep-20240919160119-7rbsn', 'ep-20240919160005-chzhb',
               'ep-20240919161410-7k5d8'],
     # ["doubao-pro-32k", "doubao-lite-4k", "doubao-lite-32k", "doubao-pro-4k", "doubao-pro-32k", "doubao-pro-128k"],
     'url': 'https://ark.cn-beijing.volces.com/api/v3/chat/completions',
     'base_url': "https://ark.cn-beijing.volces.com"},  # open.volcengineapi.com
    # https://cloud.tencent.com/document/product/1729
    {'name': 'hunyuan', 'type': 'tencent', 'api_key': '', 'model': ["hunyuan-pro", 'hunyuan-functioncall'],
     'url': 'https://hunyuan.tencentcloudapi.com',
     'base_url': 'hunyuan.ap-shanghai.tencentcloudapi.com'},
    # https://cloud.siliconflow.cn/playground/chat
    {'name': 'silicon', 'type': 'default', 'api_key': '',
     'model': ["Qwen/Qwen2-7B-Instruct", "Qwen/Qwen1.5-7B-Chat", "Qwen/Qwen1.5-32B-Chat",
               "THUDM/chatglm3-6b", "THUDM/glm-4-9b-chat",
               "Pro/THUDM/glm-4-9b-chat", "deepseek-ai/deepseek-llm-67b-chat", "deepseek-ai/DeepSeek-V2-Chat",
               "google/gemma-2-9b-it", "meta-llama/Meta-Llama-3-8B-Instruct"],
     'reranker': ['BAAI/bge-reranker-v2-m3'],
     'url': 'https://api.siliconflow.cn/v1/chat/completions',
     'base_url': 'https://api.siliconflow.cn'},
    {'name': 'gpt', 'type': 'default', 'api_key': '', 'model': ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
     'completion': ["text-davinci-003", "text-davinci-002", "text-davinci-003", "text-davinci-004"],
     'embedding': ["text-embedding-ada-002", "text-search-ada-doc-001", "text-similarity-babbage-001",
                   "code-search-ada-code-001", "search-babbage-text-001"],
     'url': 'https://api.openai.com/v1/chat/completions',
     'embeddings_url': 'https://api.openai.com/v1/embeddings',
     'base_url': "https://api.openai.com/v1",
     },
]
# moonshot,glm,qwen,ernie,hunyuan,doubao,silicon
API_KEYS = {
    'moonshot': Config.Moonshot_Service_Key,
    'glm': Config.GLM_Service_Key,
    'qwen': Config.DashScope_Service_Key,
    'doubao': Config.ARK_Service_Key,
    'silicon': Config.Silicon_Service_Key,
}
AI_Client = {}


def init_ai_clients(api_keys=API_KEYS):
    for model in AI_Models:
        model_name = model['name']
        api_key = api_keys.get(model_name)
        if api_key:
            model['api_key'] = api_key
            if model_name in ('moonshot', 'glm', 'qwen', 'silicon'):  # OpenAI_Client
                AI_Client[model_name] = OpenAI(api_key=api_key, base_url=model['base_url'])


def find_ai_model(name, model_id: int = 0, search_field: str = 'model'):
    """
    在 AI_Models 中查找模型。如果找到名称匹配的模型，返回模型及其类型或具体的子模型名称。

    参数:
    - name: 要查找的模型名称
    - model_id: 可选参数，指定返回的子模型索引，默认为 0
    - search_field: 要在其中查找名称的字段（默认为 'model'）
    """
    model = next(
        (item for item in AI_Models if item['name'] == name or name in item.get(search_field, [])),
        None
    )
    if model:
        if name in model.get(search_field, []):
            return model, name

        model_list = model.get(search_field, [])
        if model_list:
            model_i = model_id if abs(model_id) < len(model_list) else 0
            return model, model_list[model_i]
        return model, None

    raise ValueError(f"Model with name {name} not found.")


System_content = {'0': '你是一个知识广博且乐于助人的助手，擅长分析和解决各种问题。请根据我提供的信息进行帮助。',
                  '1': ('你是一位领域专家，请回答以下问题。\n'
                        '（注意：1、材料可能与问题无关，请忽略无关材料，并基于已有知识回答问题。'
                        '2、尽量避免直接复制材料，将其作为参考来补充背景或启发分析。'
                        '3、请直接提供分析和答案，请准确引用，并结合技术细节与实际应用案例，自然融入回答。'
                        '4、避免使用“作者认为”等主观表达，直接陈述观点，保持分析的清晰和逻辑性。）'),
                  '2': ('你是一位领域内的技术专家，擅长于分析和解构复杂的技术概念。'
                        '我会向你提出一些问题，请你根据相关技术领域的最佳实践和前沿研究，对问题进行深度解析。'
                        '请基于相关技术领域进行扩展，集思广益，并为每个技术点提供简要且精确的描述。'
                        '请将这些技术和其描述性文本整理成JSON格式，具体结构为 `{ "技术点1": "描述1",  ...}`，请确保JSON结构清晰且易于解析。'
                        '我将根据这些描述的语义进一步查找资料，并开展深入研究。'),
                  '3': (
                      '我有一个数据集，可能是JSON数据、表格文件或文本描述。你需要从中提取并处理数据，现已安装以下Python包：plotly.express、pandas、seaborn、matplotlib，以及系统自带的包如os、sys等。'
                      '请根据我的要求生成一个Python脚本，涵盖以下内容：'
                      '1、数据读取和处理：使用pandas读取数据，并对指定字段进行分组统计、聚合或其他处理操作。'
                      '2、数据可视化分析：生成如折线图、条形图、散点图等，用以展示特定字段的数据趋势或关系。'
                      '3、灵活性：脚本应具备灵活性，可以根据不同字段或分析需求快速调整。例如，当我要求生成某个字段的折线图时，可以快速修改代码实现。'
                      '4、注释和可执行性：确保代码能够直接运行，并包含必要的注释以便理解。'
                      '假设数据已被加载到pandas的DataFrame中，变量名为df。脚本应易于扩展，可以根据需要调整不同的可视化或数据处理逻辑。'),
                  '4': ('你是一位信息提取专家，能够从文本中精准提取信息，并将其组织为结构化的JSON格式。你的任务是：'
                        '1、提取文本中的关键信息，确保信息的准确性和完整性。'
                        '2、根据用户的请求，按照指定的类别对信息进行分类（例如：“人名”、“职位”、“时间”、“事件”、“地点”、“目的”、“计划”等）。'
                        '3、默认情况下，如果某个类别信息不完整或缺失时，不做推测或补充，返回空字符串。如果明确要求，可根据上下文进行适度的推测或补全。'
                        '4、如果明确指定了输出类别或返回格式，请严格按照要求输出，不生成子内容或嵌套结构。'
                        '5、将提取的信息以JSON格式输出，确保结构清晰、格式正确、易于理解。'),
                  '5': ('你是一位SQL转换器，精通SQL语言，能够准确地理解和解析用户的日常语言描述，并将其转换为高效、可执行的SQL查询语句,Generate a SQL query。'
                        '1、理解用户的自然语言描述，保持其意图和目标的完整性。'
                        '2、根据描述内容，将其转换为对应的SQL查询语句。'
                        '3、确保生成的SQL查询语句准确、有效，并符合最佳实践。'
                        '4、输出经过优化的SQL查询语句。'),
                  '6': ('你是一位领域专家，我正在编写一本书，请按照以下要求处理并用中文输出：'
                        '1、内容扩展和总结: 根据提供的关键字和描述，扩展和丰富每个章节的内容，确保细节丰富、逻辑连贯，使整章文本流畅自然。'
                        '必要时，总结已有内容和核心观点，形成有机衔接的连贯段落，避免生成分散或独立的句子。'
                        '2、最佳实践和前沿研究: 提供相关技术领域的最佳实践和前沿研究，结合实际应用场景，深入解析关键问题，帮助读者理解复杂概念。'
                        '3、背景知识和技术细节: 扩展背景知识，结合具体技术细节和应用场景进，提供实际案例和应用方法，增强内容的深度和实用性。保持见解鲜明，确保信息全面和确保段落的逻辑一致性。'
                        '4、连贯段落: 组织生成的所有内容成连贯的段落，确保每段文字自然延续上一段，避免使用孤立的标题或关键词，形成完整的章节内容。'
                        '5、适应书籍风格: 确保内容符合书籍的阅读风格，适应中文读者的阅读习惯与文化背景，语言流畅、结构清晰、易于理解并具参考价值。'),
                  '7': ('请根据以下对话内容生成一个清晰且详细的摘要，帮我总结一下，转换成会议纪要\n：'
                        '1、 提炼出会议的核心讨论点和关键信息。'
                        '2、 按照主题或讨论点对内容进行分组和分类。'
                        '3、 列出所有决定事项及后续的待办事项。'),
                  '8': ('你是一位专业的文本润色专家，擅长处理短句和语音口述转换的内容。请根据以下要求对内容进行润色并用中文输出：'
                        '1、语言优化: 对短句进行适当润色，确保句子流畅、自然，避免生硬表达，提升整体可读性。保持统一的语气和风格，确保文本适应场景，易于理解且专业性强。'
                        '2、信息完整: 确保每个句子的核心信息清晰明确，对于过于简短或含糊的句子进行适当扩展，丰富细节。'
                        '3、信息延展: 在不偏离原意的前提下，适当丰富或补充内容，使信息更加明确。'
                        '4、段落整合: 将相关内容整合成连贯的段落，确保各句之间有逻辑关系，避免信息碎片化，避免信息孤立和跳跃。'),
                  '9': ('你是一位名片助手，请根据以下信息生成一张个人名片。生成一张简洁、专业的名片，适合与他人分享，并附带企业介绍，确保信息完善准确并结构化。'
                        '如果某个类别信息不完整或缺失时，不做推测或补充，返回空字符串，不生成子内容或嵌套结构。'
                        '以下是输出格式：'
                        '{"personal_info": {'
                        '"name": 姓名,'
                        '"position": 岗位,'
                        '"phone": 手机号,'
                        '"email": 邮箱,'
                        '"bio": 个人简介,'
                        '"achievements": 个人所获成就'
                        '},'
                        '"company_info": {'
                        '"name": 企业名称,'
                        '"address": 企业地址,'
                        '"about": 企业简介,帮助他人更好了解公司,'
                        '"products": 公司主要产品介绍'
                        '}}'),
                  '10': "将英文单词转换为包括中文翻译、英文释义和一个例句的完整解释。请检查所有信息是否准确，并在回答时保持简洁，不需要任何其他反馈。"
                  }

AI_Tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "当你想知道现在的时间时非常有用。",
            "parameters": {}  # 因为获取当前时间无需输入参数，因此parameters为空字典
        }
    },
]


def get_current_time():
    current_datetime = datetime.now()
    formatted_time = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    return f"当前时间：{formatted_time}。"


def ai_tool_response(messages, model_name='moonshot', model_id=-1):
    model_info, name = find_ai_model(model_name, model_id)
    client = AI_Client.get(model_info['name'], None)
    if client:
        completion = client.chat.completions.create(
            model=name,
            messages=messages,
            tools=AI_Tools
        )
        return completion.model_dump()  # response['choices'][0]['message']


def ai_embeddings(inputs, model_name='qwen', model_id=0):
    model_info, name = find_ai_model(model_name, model_id, 'embedding')
    if not name:
        return []

    client = AI_Client.get(model_info['name'], None)
    if client:
        completion = client.embeddings.create(
            model=name,
            input=inputs,
            encoding_format="float"
        )  # openai.Embedding.create

        return [item.embedding for item in completion.data]
        # data = json.loads(completion.model_dump_json()
    return []


async def get_chat_payload(messages, user_message: str, system: str = '', temperature: float = 0.4, top_p: float = 0.8,
                           max_tokens: int = 1024, model_name='moonshot', model_id=0,
                           generate_calls: List[Callable[[str], Any]] = lambda x: [],
                           keywords: List[str] = None, **kwargs):
    model_info, name = find_ai_model(model_name, model_id, 'model')
    model_type = model_info['type']

    if isinstance(messages, list) and messages:
        if model_type in ('baidu', 'tencent'):
            if messages[0].get('role') != 'user':  # user（tool）
                del messages[0]  # the role of first message must be user

        if model_type != 'baidu' and system:
            messages.insert(0, {"role": "system", "content": system})
            # messages[-1]['content'] = messages[0]['content'] + '\n' + messages[-1]['content']

        if user_message:
            if messages[-1]["role"] != 'user':
                messages.append({'role': 'user', 'content': user_message})
            else:
                pass  # messages[-1]['content'] = user_message
        else:
            if messages[-1]["role"] == 'user':
                user_message = messages[-1]["content"]
    else:
        if model_type != 'baidu':
            messages = [{"role": "system", "content": system}]
        messages.append({'role': 'user', 'content': user_message})

    # Assume this is the document retrieved from RAG
    # function_call = Agent_functions.get(agent, lambda *args, **kwargs: [])
    # refer = function_call(user_message, ...)
    if generate_calls is None:
        generate_calls = []

    if keywords and not any(callable(func) for func in generate_calls if func is not None):  # not in agent_funcalls
        generate_calls.append(web_search_async)

    async def wrap_sync(func, *args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)

    tasks = []
    items_to_process = keywords if keywords else [user_message]  # ','.join(keywords)
    for func in generate_calls:
        if not callable(func):
            print(getattr(func, '__name__', 'Unknown function'))
            continue
        for w in items_to_process:
            if inspect.iscoroutinefunction(func):
                tasks.append(func(w, **kwargs))
            else:
                tasks.append(wrap_sync(func, w, **kwargs))

    refer = await asyncio.gather(*tasks)  # gather 收集所有异步调用的结果

    if refer:
        refer = [item for sublist in refer for item in sublist]
        messages[-1]['content'] = f'参考材料:\n{refer}\n 材料仅供参考,请根据上下文回答下面的问题:{user_message}'

    payload = {
        "model": name,  # 默认选择第一个模型
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if model_type == 'baidu':
        payload['system'] = system

    return model_info, payload, refer


async def ai_chat(model_info, payload=None, **kwargs):
    if not payload:
        model_info, payload, _ = await get_chat_payload(**kwargs)
    else:
        payload.update(kwargs)  # {**payload, **kwargs}

    client = AI_Client.get(model_info['name'], None)
    if client:
        try:
            completion = await asyncio.to_thread(client.chat.completions.create, **payload)
            return completion.choices[0].message.content
        except Exception as e:
            return f"OpenAI error occurred: {e}"

    # 通过 requests 库直接发起 HTTP POST 请求
    model_type = model_info['type']
    url = model_info['url']
    api_key = model_info['api_key']
    # body = payload
    if api_key:
        headers = {'Content-Type': 'application/json',
                   "Authorization": f'Bearer {api_key}'}
    if model_type == 'baidu':
        url = build_url(url, get_baidu_access_token(Config.BAIDU_qianfan_API_Key, Config.BAIDU_qianfan_Secret_Key))
        headers = {
            'Content-Type': 'application/json',
        }
        payload["disable_search"] = False
        # payload['enable_system_memory'] = False
        # payload["enable_citation"]= False
        # payload["user_id"]=
        # payload['system'] = system
    if model_type == 'tencent':
        service = 'hunyuan'
        host = url.split("//")[-1]  # model_info['base_url'].split("//")[-1]
        payload = convert_keys_to_pascal_case(payload)
        payload.pop('MaxTokens', None)
        headers = get_tencent_signature(service, host, payload, action='ChatCompletions',
                                        secret_id=Config.TENCENT_SecretId,
                                        secret_key=Config.TENCENT_Secret_Key)

        # headers["X-TC-Region"] = 'ap-shanghai'

    # print(headers, payload)

    parse_rules = {
        'baidu': lambda d: d.get('result'),
        'tencent': lambda d: d.get('Response', {}).get('Choices', [{}])[0].get('Message', {}).get('Content'),
        # d.get('Choices')[0].get('Message').get('Content')
        'default': lambda d: d.get('choices', [{}])[0].get('message', {}).get('content')
    }
    limits = httpx.Limits(max_connections=100, max_keepalive_connections=10)
    try:
        async with httpx.AsyncClient(limits=limits, timeout=Config.HTTP_TIMEOUT_SEC) as cx:
            response = await cx.post(url, headers=headers, json=payload)
            response.raise_for_status()  # 如果请求失败，则抛出异常
            data = response.json()
            result = parse_rules.get(model_type, parse_rules['default'])(data)
            if result:
                return result
            print(response.text)
    except Exception as e:
        return f"HTTP error occurred: {e}"


async def ai_chat_async(model_info, payload=None, **kwargs):
    if not payload:
        model_info, payload, _ = await get_chat_payload(**kwargs)
    else:
        payload.update(kwargs)

    payload["stream"] = True
    # payload"stream"]= {"include_usage": True}        # 可选，配置以后会在流式输出的最后一行展示token使用信息
    client = AI_Client.get(model_info['name'], None)

    if client:
        try:
            stream = client.chat.completions.create(**payload)
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:  # 以两个换行符 \n\n 结束当前传输的数据块
                    yield delta.content  # completion.append(delta.content)
        except Exception as e:
            yield f"OpenAI error occurred: {e}"
        # yield '[DONE]'
        return

    model_type = model_info['type']
    url = model_info['url']
    api_key = model_info['api_key']
    if api_key:
        headers = {
            'Content-Type': 'text/event-stream',
            "Authorization": f'Bearer {api_key}'
        }

    if model_type == 'baidu':  # 'ernie'
        url = build_url(url, get_baidu_access_token(Config.BAIDU_qianfan_API_Key,
                                                    Config.BAIDU_qianfan_Secret_Key))  # ?access_token=" + get_access_token()
        headers = {'Content-Type': 'application/json', }
    if model_type == 'tencent':
        service = 'hunyuan'
        host = url.split("//")[-1]
        payload = convert_keys_to_pascal_case(payload)
        payload.pop('MaxTokens', None)
        headers = get_tencent_signature(payload, service, host, action='ChatCompletions',
                                        secret_id=Config.TENCENT_SecretId,
                                        secret_key=Config.TENCENT_Secret_Key)
        headers['X-TC-Version'] = '2023-09-01'
    # if model_name == 'silicon':
    #     headers = {
    #         "accept": "application/json",
    #         "content-type": "application/json",
    #         "authorization": "Bearer sk-tokens"
    #     }
    limits = httpx.Limits(max_connections=100, max_keepalive_connections=10)
    try:
        async with httpx.AsyncClient(limits=limits) as cx:
            async with cx.stream("POST", url, headers=headers, json=payload) as response:
                response.raise_for_status()
                async for content in process_line_stream(response, model_type):
                    yield content
    except httpx.RequestError as e:
        yield str(e)

    # yield "[DONE]"


async def process_line_stream(response, model_type='default'):
    data = ""
    async for line in response.aiter_lines():
        line = line.strip()
        if len(line) == 0:  # 开头的行 not line
            if data:
                yield process_data_chunk(data, model_type)  # 一个数据块的结束 yield from + '\n'
                data = ""
            continue
        if line.startswith("data: "):
            line_data = line.lstrip("data: ")
            if line_data == "[DONE]":
                # if data:
                #     yield process_data_chunk(data, model_type)
                # print(data)
                break
            if model_type == 'tencent':
                chunk = json.loads(line_data)
                reason = chunk.get('Choices', [{}])[0].get('FinishReason')
                if reason == "stop":
                    break
            elif model_type == 'baidu':
                chunk = json.loads(line_data)
                if chunk.get('is_end') is True:
                    break

            content = process_data_chunk(line_data, model_type)
            if content:
                yield content
        else:
            data += "\n" + line

    if data:
        yield process_data_chunk(data, model_type)


def process_data_chunk(data, model_type='default'):
    try:
        chunk = json.loads(data)
        if model_type == 'baidu':  # line.decode("UTF-8")
            return chunk.get("result")
        elif model_type == 'tencent':
            choices = chunk.get('Choices', [])
            if choices:
                delta = choices[0].get('Delta', {})
                return delta.get("Content")
        else:
            choices = chunk.get('choices', [])
            if choices:
                delta = choices[0].get('delta', {})
                return delta.get("content")

    except (json.JSONDecodeError, KeyError, IndexError) as e:
        return str(e)
    return None

async def forward_stream(response):
    async for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("data: "):
            line_data = line.lstrip("data: ")
            if line_data == "[DONE]":
                break
            try:
                parsed_content = json.loads(line_data)
                yield {"json": parsed_content}
            except json.JSONDecodeError:
                yield {"text": line_data}


# 生成:conversation or summary
async def generate_completion(prompt, stream=False, temperature=0.7, max_tokens=300, model_name='gpt', model_id=0):
    model_info, name = find_ai_model(model_name, model_id, 'completion')
    if not name:
        return None  # ai_chat(messages, temperature=0.4, top_p=0.8, model_name='moonshot')

    response = Completion.create(
        engine=name,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        n=1,
        stop=None,
        stream=stream
    )
    if stream:
        async def stream_data():
            for chunk in response:
                yield chunk.choices[0].text

        return stream_data()

    generated_text = response.choices[0].text.strip()
    return generated_text


def web_search(text: str, api_key: str = Config.GLM_Service_Key) -> list:
    msg = [{"role": "user", "content": text}]
    url = "https://open.bigmodel.cn/api/paas/v4/tools"
    data = {
        "request_id": str(uuid.uuid4()),
        "tool": "web-search-pro",
        "stream": False,
        "messages": msg
    }

    headers = {'Authorization': api_key}
    try:
        response = requests.post(url, json=data, headers=headers, timeout=Config.HTTP_TIMEOUT_SEC)
        response.raise_for_status()

        data = response.json()
        search_result = data.get('choices', [{}])[0].get('message', {}).get('tool_calls', [{}])[1].get('search_result')
        if search_result:
            return [{
                'title': result.get('title'),
                'content': result.get('content'),
                'link': result.get('link'),
                'media': result.get('media')
            } for result in search_result]
        return [{'content': response.content.decode()}]
    except (requests.exceptions.RequestException, KeyError, IndexError) as e:
        return [{'error': str(e)}]


async def web_search_async(text: str, api_key: str = Config.GLM_Service_Key) -> list:
    msg = [{"role": "user", "content": text}]
    tool = "web-search-pro"
    url = "https://open.bigmodel.cn/api/paas/v4/tools"
    data = {
        "request_id": str(uuid.uuid4()),
        "tool": tool,
        "stream": False,
        "messages": msg
    }

    headers = {'Authorization': api_key}

    async with httpx.AsyncClient() as cx:
        try:
            resp = await cx.post(url, json=data, headers=headers, timeout=Config.HTTP_TIMEOUT_SEC)
            resp.raise_for_status()

            data = resp.json()
            results = data['choices'][0]['message']['tool_calls'][1]['search_result']
            return [{
                'title': result.get('title'),
                'content': result.get('content'),
                'link': result.get('link'),
                'media': result.get('media')
            } for result in results]

            # return resp.text  # resp.content.decode()
        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            return [{'error': str(exc)}]


def bing_search(query, bing_api_key):
    url = f"https://api.bing.microsoft.com/v7.0/search?q={query}"
    headers = {"Ocp-Apim-Subscription-Key": bing_api_key}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    search_results = response.json()

    results = []
    for result in search_results.get('webPages', {}).get('value', []):
        title = result.get('name')
        snippet = result.get('snippet')
        link = result.get('url')
        results.append((title, snippet, link))
    return results  # "\n".join([f"{i+1}. {title}: {snippet} ({link})" for i, (title, snippet, link) in enumerate(search_results[:5])])


# https://ziyuan.baidu.com/fastcrawl/index
def baidu_search(query, baidu_api_key, baidu_secret_key):
    access_token = get_baidu_access_token(baidu_api_key, baidu_secret_key)
    search_url = "https://aip.baidubce.com/rest/2.0/knowledge/v1/search"  # https://aip.baidubce.com/rpc/2.0/unit/service/v3/chat
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    search_params = {
        "access_token": access_token,
        "query": query,
        "scope": 1,  # 搜索范围
        "page_no": 1,
        "page_size": 5
    }
    search_response = requests.post(search_url, headers=headers, data=search_params)

    if search_response.status_code == 401:  # 如果token失效
        global baidu_access_token
        baidu_access_token = None
        access_token = get_baidu_access_token(baidu_api_key, baidu_secret_key)

        search_params["access_token"] = access_token
        search_response = requests.post(search_url, headers=headers, data=search_params)

    search_response.raise_for_status()

    search_results = search_response.json()
    results = []
    for result in search_results.get('result', []):
        title = result.get('title')
        content = result.get('content')
        url = result.get('url')
        results.append((title, content, url))
    return results  # "\n".join([f"{i+1}. {title}: {content} ({url})" for i, (title, content, url) in enumerate(search_results[:5])])


def wikipedia_search(query):
    response = requests.get(f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&format=json")
    search_results = response.json()['query']['search']
    if search_results:
        page_id = search_results[0]['pageid']
        page_response = requests.get(
            f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&pageids={page_id}&format=json")
        page_data = page_response.json()['query']['pages'][str(page_id)]
        return page_data['extract']
    return "No information found."


async def get_bge_embeddings(texts: List[str], access_token: str = '') -> List:
    if not isinstance(texts, list):
        texts = [texts]

    url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/embeddings/bge_large_zh?access_token={access_token}"
    headers = {
        'Content-Type': 'application/json'
    }
    batch_size = 16
    embeddings = []

    async with httpx.AsyncClient() as cx:
        if len(texts) < batch_size:
            payload = json.dumps({"input": texts})
            response = await cx.post(url, headers=headers, data=payload)
            data = response.json().get('data')
            # if len(texts) == 1:
            #     return data[0].get('embedding') if data else None
            embeddings = [d.get('embedding') for d in data]
        else:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                payload = {"input": batch}
                try:
                    response = await cx.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json().get('data')
                    if data and len(data) == len(batch):
                        embeddings += [d.get('embedding') for d in data]
                except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                    print(exc)

    return embeddings


# https://console.bce.baidu.com/ai/#/ai/machinetranslation/overview/index
async def baidu_translate(text: str, from_lang: str = 'zh', to_lang: str = 'en'):
    """百度翻译 API"""
    salt = str(random.randint(32768, 65536))  # str(int(time.time() * 1000))
    sign = md5_sign(text, salt,
                    Config.BAIDU_trans_AppId, Config.BAIDU_trans_Secret_Key)  # 需要计算 sign = MD5(appid+q+salt+密钥)
    url = "https://fanyi-api.baidu.com/api/trans/vip/translate"
    params = {
        "q": text,
        "from": from_lang,
        "to": to_lang,
        "appid": Config.BAIDU_trans_AppId,
        "salt": salt,
        "sign": sign
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)

    data = response.json()
    if "trans_result" in data:
        return data["trans_result"][0]["dst"]

    # print(response.text)
    raise HTTPException(status_code=400, detail=f"Baidu API Error: {data.get('error_msg', 'Unknown error')}")


async def tencent_translate(text: str, source: str, target: str):
    timestamp = int(time.time())
    nonce = 123456
    params = {
        "SourceText": text,
        "Source": source,
        "Target": target,
        "SecretId": Config.TENCENT_SecretId,
        "Timestamp": timestamp,
        "Nonce": nonce,
    }
    signature = generate_tencent_signature(Config.TENCENT_Secret_Key, "POST", params)
    params["Signature"] = signature

    url = "https://cloud.tencent.com/api/translate"
    async with httpx.AsyncClient() as client:
        response = await client.post(url, data=params)

    # 检查响应状态码和内容
    if response.status_code != 200:
        print(f"Error: Received status code {response.status_code}")
        print(f"Response content: {response.text}")
        raise HTTPException(status_code=response.status_code, detail="Request failed")

    try:
        data = response.json()
    except Exception as e:
        print(f"Failed to decode JSON: {e}")
        print(f"Response text: {response.text}")
        raise

    if "TargetText" in data:
        return data["TargetText"]
    else:
        raise HTTPException(status_code=400, detail=f"Tencent API Error: {data.get('Message', 'Unknown error')}")


# https://ai.baidu.com/ai-doc/OCR/Ek3h7y961
# https://console.bce.baidu.com/ai/#/ai/ocr/overview/index
# with open(image_path, 'rb') as f:
#    image_data = f.read()
def baidu_ocr_recognise(image_data, image_url, access_token, ocr_sign='accurate_basic'):
    headers = {
        'Content-Type': "application/x-www-form-urlencoded",
        'charset': "utf-8"
    }
    # accurate,general_basic,webimage
    url = f"https://aip.baidubce.com/rest/2.0/ocr/v1/{ocr_sign}"  # https://aip.baidubce.com/rest/2.0/solution/v1/iocr/recognise"
    try:
        # 将图像数据编码为base64
        # image_b64 = base64.b64encode(image_data).decode().replace("\r", "")
        params = {
            "access_token": access_token,
            "language_type": 'CHN_ENG',
        }
        if image_data:
            params["image"] = base64.b64encode(image_data)  # quote(image_b64.encode("utf8"))
        if url:
            params["url"] = image_url

        # if template_sign:
        #     params["templateSign"] = template_sign
        # if classifier_id:
        #     params["classifierId"] = classifier_id
        # # 请求模板的bodys
        # recognise_bodys = "access_token=" + access_token + "&templateSign=" + template_sign + "&image=" + quote(image_b64.encode("utf8"))
        # # 请求分类器的bodys
        # classifier_bodys = "access_token=" + access_token + "&classifierId=" + classifier_id + "&image=" + quote(image_b64.encode("utf8"))
        # request_body = "&".join(f"{key}={value}" for key, value in params.items())
        response = requests.post(url, data=params, headers=headers)
        response.raise_for_status()

        return response.json()

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# https://nls-portal.console.aliyun.com/overview
async def ali_speech_to_text(audio_data, format='pcm'):
    """阿里云语音转文字"""
    params = {
        "appkey": Config.ALIYUN_nls_AppId,
        "format": format,  # 也可以传入其他格式，如 wav, mp3
        "sample_rate": 16000,  # 音频采样率
        "version": "4.0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "SignatureMethod": "HMAC-SHA1",
        "SignatureVersion": "1.0",
        "SignatureNonce": str(uuid.uuid4())
    }
    signature = generate_hmac_signature(Config.ALIYUN_Secret_Key, "POST", params)
    params["signature"] = signature
    token, _ = get_aliyun_access_token(Config.ALIYUN_AK_ID, Config.ALIYUN_Secret_Key)
    if not token:
        print("No permission!")

    headers = {
        "Authorization": f"Bearer {Config.ALIYUN_AK_ID}",
        # "Content-Type": "audio/pcm",
        "Content-Type": "application/octet-stream",
        "X-NLS-Token": token,
    }

    # host = 'nls-gateway-cn-shanghai.aliyuncs.com'
    # conn = http.client.HTTPSConnection(host)
    # http://nls-meta.cn-shanghai.aliyuncs.com/
    # "wss://nls-gateway-cn-shanghai.aliyuncs.com/ws/v1"
    url = "https://nls-gateway.cn-shanghai.aliyuncs.com/stream/v1/asr"
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, params=params, data=audio_data.getvalue())

    result = response.json()
    if result.get("status") == 20000000:  # "SUCCESS":
        return {"text": result.get("result")}

    return {"error": result.get('message')}


# {
#     "task_id": "cf7b0c5339244ee29cd4e43fb97f****",
#     "result": "北京的天气。",
#     "status":20000000,
#     "message":"SUCCESS"
# }

# 1536: 适用于普通话输入法模型（支持简单的英文）。
# 1537: 适用于普通话输入法模型（纯中文）。
# 1737: 适用于英文。
# 1936: 适用于粤语。
# audio/pcm pcm（不压缩）、wav（不压缩，pcm编码）、amr（压缩格式）、m4a（压缩格式）
# https://console.bce.baidu.com/ai/#/ai/speech/overview/index
async def baidu_speech_to_text(audio_data, format='pcm', dev_pid=1536):  #: io.BytesIO
    url = "https://vop.baidu.com/server_api"  # 'https://vop.baidu.com/pro_api'
    access_token = get_baidu_access_token(Config.BAIDU_speech_API_Key, Config.BAIDU_speech_Secret_Key)
    # Config.BAIDU_speech_AppId
    url = f"{url}?dev_pid={dev_pid}&cuid={Config.DEVICE_ID}&token={access_token}"
    headers = {'Content-Type': f'audio/{format}; rate=16000'}

    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, data=audio_data.getvalue())

    result = response.json()
    if result.get("err_no") == 0:
        return {"text": result.get("result")[0]}

    return {"error": result.get('err_msg')}


# Paraformer语音识别API基于通义实验室新一代非自回归端到端模型，提供基于实时音频流的语音识别以及对输入的各类音视频文件进行语音识别的能力。可被应用于：
# 对语音识别结果返回的即时性有严格要求的实时场景，如实时会议记录、实时直播字幕、电话客服等。
# 对音视频文件中语音内容的识别，从而进行内容理解分析、字幕生成等。
# 对电话客服呼叫中心录音进行识别，从而进行客服质检等
async def dashscope_speech_to_text(audio_path, format='wav', language: List[str] = ['zh', 'en']):
    recognition = Recognition(model='paraformer-realtime-v2', format=format, sample_rate=16000,
                              language_hints=language, callback=None)
    result = await asyncio.to_thread(recognition.call, audio_path)  # recognition.call(audio_path)
    if result.status_code == 200:
        texts = [sentence.get('text', '') for sentence in result.get_sentence()]
        return {"text": texts[0]}

    return {"error": result.message}


# SenseVoice语音识别大模型专注于高精度多语言语音识别、情感辨识和音频事件检测，支持超过50种语言的识别，整体效果优于Whisper模型，中文与粤语识别准确率相对提升在50%以上。
# SenseVoice语音识别提供的文件转写API，能够对常见的音频或音视频文件进行语音识别，并将结果返回给调用者。
# SenseVoice语音识别返回较为丰富的结果供调用者选择使用，包括全文级文字、句子级文字、词、时间戳、语音情绪和音频事件等。模型默认进行标点符号预测和逆文本正则化。
async def dashscope_speech_to_text_url(file_urls, model='paraformer-v1', language: List[str] = ['zh', 'en']):
    task_response = Transcription.async_call(
        model=model,  # paraformer-8k-v1, paraformer-mtl-v1
        file_urls=file_urls, language_hints=language)

    transcribe_response = Transcription.wait(task=task_response.output.task_id)
    transcription_texts = []
    for r in transcribe_response.output["results"]:
        if r["subtask_status"] == "SUCCEEDED":
            async with httpx.AsyncClient() as client:
                response = await client.get(r["transcription_url"])
                if response.status_code == 200:
                    transcription_data = response.json()
                    if len(transcription_data["transcripts"]) > 0:
                        transcription_texts.append({"file_url": transcription_data["file_url"],
                                                    "transcripts": transcription_data["transcripts"][0]['text']
                                                    })  # transcription_data["transcripts"][0]["sentences"][0]["text"]
                    else:
                        print(f"No transcription text found in the response.Transcription Result: {response.text}")
                else:
                    print(f"Failed to fetch transcription. Status code: {response.status_code}")
        else:
            print(f"Subtask status: {r['subtask_status']}")

    if len(file_urls) != len(transcription_texts):
        print(json.dumps(transcribe_response.output, indent=4, ensure_ascii=False))

    return transcription_texts, task_response.output.task_id


# 非流式合成
async def dashscope_text_to_speech(sentences, model="cosyvoice-v1", voice="longxiaochun"):
    synthesizer = dashscope.audio.tts_v2.SpeechSynthesizer(model=model, voice=voice)
    audio = synthesizer.call(sentences)  # ,sample_rate=48000
    audio_io = io.BytesIO(audio)
    audio_io.seek(0)
    return audio_io, synthesizer.get_last_request_id()

    # SpeechSynthesizer.call(model='sambert-zhichu-v1',
    #                        text='今天天气怎么样',
    #                        sample_rate=48000,
    #                        format='pcm',
    #                        callback=callback)
    # if result.get_audio_data() is not None:


def dashscope_file_response(messages, file_path='.pdf', client=None, api_key=''):
    from pathlib import Path
    if client:
        file_object = client.files.create(file=Path(file_path), purpose="file-extract")  # .is_file()
        messages.append({"role": "system", "content": f"fileid://{file_object.id}"})
        completion = client.chat.completions.create(model="qwen-long", messages=messages, )
        return completion.model_dump_json(), file_object.id

    url = 'https://dashscope.aliyuncs.com/compatible-mode/v1/files'
    headers = {
        'Content-Type': 'application/json',
        "Authorization": f'Bearer {api_key}',
    }

    try:
        files = {
            'file': open(file_path, 'rb'),
            'purpose': (None, 'file-extract'),
        }
        file_response = requests.post(url, headers=headers, files=files)
        file_response.raise_for_status()  # 检查请求是否成功
        file_object = file_response.json()
        file_id = file_object.get('id', 'unknown_id')  # 从响应中获取文件ID

        messages.append({"role": "system", "content": f"fileid://{file_id}"})
        return file_object, file_id

    except Exception as e:
        return {"error": str(e)}, None

    finally:
        files['file'][1].close()


Agent_functions = {
    'default': lambda *args, **kwargs: [],
    '2': web_search,
    '9': web_search_async,
}

if __name__ == "__main__":
    AccessToken = 'd04149e455d44ac09432f0f89c3e0a41'
    # https://nls-portal-service.aliyun.com/ptts?p=eyJleHBpcmF0aW9uIjoiMjAyNC0wOS0wNlQwOToxNjoyNy42MDRaIiwiY29uZGl0aW9ucyI6W1sic3RhcnRzLXdpdGgiLCIka2V5IiwidHRwLzEzODE0NTkxNjIwMDc4MjIiXV19&s=k4sDIZ4lCmUiQ%2BV%2FcTEnFteey54%3D&e=1725614187&d=ttp%2F1381459162007822&a=LTAIiIg37IN8xeMa&h=https%3A%2F%2Ftuatara-cn-shanghai.oss-cn-shanghai.aliyuncs.com&u=qnKV1N8muiAIFiL22JTrgdYExxHS%2BPSxccg9VPiL0Nc%3D
    fileLink = "https://gw.alipayobjects.com/os/bmw-prod/0574ee2e-f494-45a5-820f-63aee583045a.wav"
    import asyncio
    import io

    dashscope.api_key = Config.DashScope_Service_Key
    file_urls = ['https://dashscope.oss-cn-beijing.aliyuncs.com/samples/audio/paraformer/hello_world_female.wav',
                 'https://dashscope.oss-cn-beijing.aliyuncs.com/samples/audio/paraformer/hello_world_male.wav'
                 ]
    # r = requests.get(
    #     'https://dashscope.oss-cn-beijing.aliyuncs.com/samples/audio/paraformer/hello_world_female2.wav'
    # )
    # with open('asr_example.wav', 'wb') as f:
    #     f.write(r.content)

    # asyncio.run(dashscope_speech_to_text_url(file_urls))

    task_id = "8d47c5d9-06bb-47aa-8986-1df91b6c8dd2"
    audio_file = 'data/nls-sample-16k.wav'


    # with open(audio_file, 'rb') as f:
    #     audio_data = io.BytesIO(f.read())

    # print(dashscope_speech_to_text('data/nls-sample-16k.wav'))

    # fetch()调用不会阻塞，将立即返回所查询任务的状态和结果
    # transcribe_response = dashscope.audio.asr.Transcription.fetch(task=task_id)
    # print(json.dumps(transcribe_response.output, indent=4, ensure_ascii=False))
    # for r in transcribe_response.output["results"]:
    #     if r["subtask_status"] == "SUCCEEDED":
    #         url = r["transcription_url"]
    #         response = requests.get(url)
    #         if response.status_code == 200:
    #             transcription_data = response.text  # 可以使用 response.json() 来处理 JSON 响应
    #             print(f"Transcription Result: {transcription_data}")
    #             data = response.json()
    #             print(data["transcripts"][0]['text'])
    #         else:
    #             print(f"Failed to fetch transcription. Status code: {response.status_code}")
    #     else:
    #         print(f"Subtask status: {r['subtask_status']}")

    async def test():
        # audio_file = 'data/nls-sample-16k.wav'
        # with open(audio_file, 'rb') as f:
        #     audio_data = io.BytesIO(f.read())
        #
        # result = await  baidu_speech_to_text(audio_data, 'wav')  # ali_speech_to_text(audio_data,'wav')  #
        result = await web_search_async('易得融信是什么公司')
        print(result)


    asyncio.run(test())

    # asyncio.run(tencent_translate('tencent translate is ok', 'en', 'cn'))
