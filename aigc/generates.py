import httpx, aiohttp, aiofiles, asyncio
from http import HTTPStatus
import oss2
from pathlib import PurePosixPath
from typing import List, Dict, Tuple, Any, Union, Callable, Awaitable, Generator, Optional
import random
from PIL import Image
from openai import OpenAI, AsyncOpenAI
# import qianfan
import dashscope
from dashscope.audio.tts import ResultCallback
from dashscope.audio.asr import Recognition, Transcription
# from lagent import tool_api
from functools import partial  # cache, lru_cache, partial, wraps, wraps
from config import *
from utils import *
from agents.ai_tools import *
from agents.ai_prompt import *

AI_Client = {}  # Dict[str, OpenAI]


async def get_data_for_model(model: dict):
    """获取每个模型的数据"""
    model_name = model.get('name')
    client = AI_Client.get(model_name)
    if client:
        try:
            models = await client.models.list()
            model['data'] = [m.model_dump() for m in models.data]
        except Exception as e:
            print(f"OpenAI {model_name} error occurred: {e}")


async def init_ai_clients(ai_models=AI_Models, api_keys=API_KEYS, get_data=False):
    tasks = []
    for model in ai_models:
        model_name = model.get('name')
        api_key = api_keys.get(model_name)
        if api_key:
            model['api_key'] = api_key
            if model_name not in AI_Client and model.get('supported_openai'):  # model_name in SUPPORTED_OPENAI_MODELS
                AI_Client[model_name]: AsyncOpenAI = AsyncOpenAI(api_key=api_key, base_url=model['base_url'])  # OpenAI
                if get_data:
                    tasks.append(get_data_for_model(model))

    if get_data:
        await asyncio.gather(*tasks)

    ModelListExtract.set()
    # print(len(ModelListExtract.get()))


# client = AI_Client['deepseek']
# print(dir(client.chat.completions))# 'create', 'with_raw_response', 'with_streaming_response'
# print(dir(client.completions))
# print(dir(client.embeddings))
# print(dir(client.files)) #'content', 'create', 'delete', 'list', 'retrieve', 'retrieve_content', 'wait_for_processing'


def find_ai_model(name, model_id: int = 0, search_field: str = 'model'):
    """
    在 AI_Models 中查找模型。如果找到名称匹配的模型，返回模型及其类型或具体的子模型名称。

    参数:
    - name: 要查找的模型名称
    - model_id: 可选参数，指定返回的子模型索引，默认为 0
    - search_field: 要在其中查找名称的字段（默认为 'model'）
     返回:
    - Tuple[Dict[str, Any], Union[str, None]]: 模型及其对应的子模型名称（或 None）

    异常:
    - ValueError: 如果未找到模型
    """
    model = next(
        (item for item in AI_Models if item['name'] == name or name in item.get(search_field, [])),
        None
    )
    if model:
        model_items = model.get(search_field, [])
        if isinstance(model_items, list):
            if name in model_items:
                return model, name
            if model_items:
                model_i = model_id if abs(model_id) < len(model_items) else 0
                return model, model_items[model_i]
        elif isinstance(model_items, dict):
            if name in model_items:
                return model, model_items[name]
            # 如果提供了序号，返回序号对应的值
            keys = list(model_items.keys())
            model_i = model_id if abs(model_id) < len(keys) else 0
            return model, model_items[keys[model_i]]

        return model, None

    raise ValueError(f"Model with name {name} not found.")
    # HTTPException(status_code=400, detail=f"Model with name {name} not found.")


async def ai_tool_response(messages, tools=None, model_name=Config.DEFAULT_MODEL, model_id=1,
                           top_p=0.95, temperature=0.01, **kwargs):
    """
      调用 AI 模型接口，使用提供的工具集和对话消息，返回模型的响应。qwen
        :return: 模型响应的消息对象
    """
    model_info, name = find_ai_model(model_name, model_id)
    client = AI_Client.get(model_info['name'], None)
    # tools = [{"type": "web_search",}]
    if client:
        try:
            completion = await client.chat.completions.create(
                model=name,
                messages=messages,
                tools=tools,
                # tool_choice="auto",
                temperature=temperature,
                top_p=top_p,
                **kwargs,
            )
            return completion.choices[0].message
            # return completion.model_dump()
            # response['choices'][0]['message']
        except Exception as e:
            print(f"OpenAI error occurred: {e}")

    return None  # await ai_chat_post(model_info, payload)


def functions_registry(functions_list: list):
    # 创建全局函数注册表
    return {name: globals().get(name) for name in functions_list}


Function_Registry = {}


async def ai_tools_messages(response_message):
    """
    解析模型响应，动态调用工具并生成后续消息列表。

    :param response_message: 模型响应的消息对象
    :return: 包含原始响应和工具调用结果的消息列表
    """
    messages = [response_message.to_dict()]  # ChatCompletionMessage
    tool_calls = response_message.tool_calls
    if not tool_calls:
        results = execute_code_blocks(response_message.content)
        # print(results)
        for res in results:
            messages.append({
                'role': 'tool',
                'content': res['error'] if res['error'] else res['output'],
                'tool_call_id': 'output.getvalue',
                'name': 'exec'
            })
        return messages

    global Function_Registry
    if not Function_Registry:  # 动态性延迟加载,全局注册表初始化
        Function_Registry = functions_registry(functions_list=[
            "get_times_shift", "get_weather", "web_search", "date_range_calculator",
            "get_day_range", "get_week_range", "get_month_range",
            "get_quarter_range", "get_year_range", "get_half_year_range",
            "search_bmap_location", "auto_translate"
            # 添加更多可调用函数
        ])

    for tool in tool_calls:
        func_name = tool.function.name  # function_name
        func_args = tool.function.arguments  # function_args = json.loads(tool_call.function.arguments)
        func_reg = Function_Registry.get(func_name, None)  # 从注册表中获取函数
        # print(func_args)
        # 检查并解析 func_args 确保是字典
        if isinstance(func_args, str):
            try:
                func_args = json.loads(func_args)  # 尝试将字符串解析为字典
            except json.JSONDecodeError as e:
                messages.append({
                    "role": "tool",
                    "content": f"Error in {func_name}: Invalid arguments format ({str(e)}).",
                    "tool_call_id": tool.id,
                    'name': func_name
                })
                continue

        if not isinstance(func_args, dict):
            messages.append({
                "role": "tool",
                "content": f"Error in {func_name}: Arguments must be a mapping type, got {type(func_args).__name__}.",
                "tool_call_id": tool.id,
                'name': func_name
            })
            continue

        try:
            if func_reg:
                if inspect.iscoroutinefunction(func_reg):
                    func_out = await func_reg(**func_args)
                else:
                    func_out = func_reg(**func_args)
            else:
                func_name = extract_method_calls(func_name)
                # compile(code, '<string>', 'eval')
                # eval(expression, globals=None, locals=None)执行单个表达式,动态计算简单的表达式或从字符串中解析值
                func_out = eval(f'{func_name}(**{func_args})')

            messages.append({
                'role': 'tool',
                'content': f'{func_out}',
                'tool_call_id': tool.id,
                'name': func_name
            })

        except Exception as e:
            error = f"Error in {func_name}: {str(e)}" if func_reg else f"Error: Function '{func_name}' not found."
            messages.append({
                'role': 'tool',
                'content': error,
                'tool_call_id': tool.id,
                'name': func_name
            })

            # print( f"Error in {func_name}: {str(e)}")
    return messages  # [*tool_mmessages,]


async def ai_auto_calls(question, **kwargs):
    messages = [{"role": "system", "content": System_content.get('31')},
                {"role": "user", "content": question}]
    response = await ai_tool_response(messages=messages, tools=AI_Tools, **kwargs)
    if response:
        final_messages = await ai_tools_messages(response)
        return [{msg['name']: msg['content']} for msg in final_messages if msg['role'] == "tool"]
    return []


async def ai_files_messages(files: List[str], question: str = None, model_name: str = 'qwen-long', model_id=-1,
                            **kwargs):
    """
    处理文件并生成 AI 模型的对话结果。

    :param files: 文件路径列表
    :param model_name: 模型名称
    :param model_id: 模型 ID
    :return: 模型生成的对话结果和文件对象列表
    """
    model_info, name = find_ai_model(model_name, model_id)
    client = AI_Client.get(model_info['name'], None)
    messages = []
    for file_path in files:
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():  # .is_file()
            continue

        if client:  # client.files.create
            file_object = await client.files.create(file=file_path_obj, purpose="file-extract")
            if model_info['name'] == 'qwen':
                messages.append({"role": "system", "content": f"fileid://{file_object.id}", })
                # client.files.list()
                # 文件信息client.files.retrieve(file_object.id)
                # 文件内容client.files.content(file_object.id)
            elif model_info['name'] == 'moonshot':
                file_content = await client.files.content(file_id=file_object.id)
                messages.append({"role": "system", "content": file_content.text, })
        else:
            dashscope_file_upload(messages, file_path=str(file_path_obj))

    if question:
        messages.append({"role": "user", "content": question})
        # print(messages)
        completion = await client.chat.completions.create(model=name, messages=messages, **kwargs)
        bot_response = completion.choices[0].message.content
        messages.append({"role": "assistant", "content": bot_response})
        return messages

    return messages


Embedding_Cache = {}


# https://www.openaidoc.com.cn/docs/guides/embeddings
async def ai_embeddings(inputs, model_name: str = 'qwen', model_id: int = 0, **kwargs) -> List[List[float]]:
    """
    text = text.replace("\n", " ")
    从远程服务获取嵌入，支持批量处理和缓存和多模型处理。
    :param inputs: 输入文本或文本列表
    :param model_name: 模型名称
    :return: 嵌入列表
    （1）文本数量不超过 16。
    （2）每个文本长度不超过 512 个 token，超出自动截断，token 统计信息，token 数 = 汉字数+单词数*1.3 （仅为估算逻辑，以实际返回为准)。
    （3） 批量最多 16 个，超过 16 后默认截断。
    """
    if not model_name:
        return []
    if not inputs:
        return []

    global Embedding_Cache
    cache_key = generate_hash_key(inputs, model_name, model_id)
    if cache_key in Embedding_Cache:
        # print(cache_key)
        return Embedding_Cache[cache_key]

    try:
        model_info, name = find_ai_model(model_name, model_id, 'embedding')
    except:
        return []

    batch_size = 16  # DASHSCOPE_MAX_BATCH_SIZE = 25
    has_error = False
    client = AI_Client.get(model_info['name'], None)
    if client:  # openai.Embedding.create
        if isinstance(inputs, list) and len(inputs) > batch_size:
            tasks = [client.embeddings.create(
                model=name, input=inputs[i:i + batch_size],
                encoding_format="float",
                **kwargs
            ) for i in range(0, len(inputs), batch_size)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            embeddings = [None] * len(inputs)
            input_idx = 0
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"Error encountered: {result}")
                    has_error = True
                    continue
                for idx, item in enumerate(result.data):
                    embeddings[input_idx] = item.embedding
                    input_idx += 1

            if not has_error:  # not  any(embedding is None for embedding in embeddings):
                Embedding_Cache[cache_key] = embeddings

        else:
            # await asyncio.to_thread(client.embeddings.create
            completion = await client.embeddings.create(
                model=name, input=inputs, encoding_format="float")

            embeddings = [item.embedding for item in completion.data]
            # data = json.loads(completion.model_dump_json()

        return embeddings

    # dashscope.TextEmbedding.call
    url = model_info['embedding_url']
    api_key = model_info['api_key']
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    payload = {
        "input": inputs,
        "model": name,
        "encoding_format": "float"
    }
    payload.update(kwargs)
    embeddings = []
    try:
        async with httpx.AsyncClient(timeout=Config.HTTP_TIMEOUT_SEC) as cx:
            if isinstance(inputs, list) and len(inputs) > batch_size:
                for i in range(0, len(inputs), batch_size):
                    batch = inputs[i:i + batch_size]
                    payload["input"] = batch  # {"texts":batch}
                    response = await cx.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json().get('data')  # "output"
                    if data and len(data) == len(batch):
                        embeddings += [emb.get('embedding') for emb in data]
                    else:
                        print(f"Error: Batch {i // batch_size + 1} response size mismatch.")
                        has_error = True
            else:
                response = await cx.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json().get('data')
                embeddings = [emb.get('embedding') for emb in data]

    except (httpx.HTTPStatusError, httpx.RequestError) as exc:
        print(exc)

    if not has_error:
        Embedding_Cache[cache_key] = embeddings

    return embeddings


async def ai_reranker(query: str, documents: List[str], top_n: int, model_name="BAAI/bge-reranker-v2-m3", model_id=0,
                      **kwargs):
    if not model_name:
        return []
    try:
        model_info, name = find_ai_model(model_name, model_id, 'reranker')
    except:
        return []
    url = model_info['reranker_url']
    api_key = model_info['api_key']
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    payload = {
        "query": query,
        "model": name,
        "documents": documents,
        "top_n": top_n,
        "return_documents": False,
        # 'max_chunks_per_doc': 1024,  # 最大块数
        # 'overlap_tokens': 80,  # 重叠数量
    }
    payload.update(kwargs)
    async with httpx.AsyncClient(timeout=Config.HTTP_TIMEOUT_SEC) as cx:
        response = await cx.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            results = response.json().get('results')
            matches = [(match["document"]["text"] if match.get("document") else documents[match["index"]],
                        match["relevance_score"], match["index"]) for match in results]
            return matches
        else:
            print(response.text)
    return []


# 生成:conversation or summary,Fill-In-the-Middle
async def ai_generate(prompt: str, user_request: str = '', suffix: str = None, stream=False, temperature=0.7,
                      max_tokens=4096, model_name='silicon', model_id=0, get_content=True, **kwargs):
    '''
    Completions足以解决几乎任何语言处理任务，包括内容生成、摘要、语义搜索、主题标记、情感分析等等。
    需要注意的一点限制是，对于大多数模型，单个API请求只能在提示和完成之间处理最多4096个标记。
    '''
    model_info, name = find_ai_model(model_name, model_id, "generation")
    if not name:
        return await ai_chat(model_info=None, messages=None, user_request=user_request, system=prompt,
                             temperature=temperature, max_tokens=max_tokens, top_p=0.8, model_name=model_name,
                             model_id=model_id, get_content=get_content, **kwargs)

    if user_request:
        prompt += '\n\n' + user_request

    if model_info['name'] == 'qwen':
        response = dashscope.Generation.call(model=name, prompt=prompt)
        return response.output.text

    client = AI_Client.get(model_info['name'], None)
    # client.completions.create
    response = await client.completions.create(
        # engine=name,
        model=name,
        prompt=prompt,
        suffix=suffix,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=stream,
        stop=None,
        **kwargs,
        # n=1,
    )
    if stream:
        async def stream_data():
            async for chunk in response:
                yield chunk.choices[0].text if get_content else chunk.model_dump_json()

        return stream_data()

    return response.choices[0].text.strip() if get_content else response.model_dump()


async def request_ollama(prompt, model_name="mistral", stream=False):
    url = "http://localhost:11434/api/generate"
    payload = {"model": model_name, "prompt": prompt, "stream": stream}

    async with aiohttp.ClientSession() as session:
        response = await session.post(url, json=payload)
        return response  # 直接返回响应对象


async def call_ollama(prompt, model_name="mistral", stream=True):
    response = await request_ollama(prompt, model_name=model_name, stream=stream)

    async def stream_data():
        async for line in response.content:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line.decode('utf-8'))
                    yield data
                    # print(data.get("response", ""), end="", flush=True)  # 实时打印 AI 生成的文本
                except json.JSONDecodeError:
                    pass  # 忽略 JSON 解析错误

    try:
        return stream_data() if stream else await response.json()
    finally:
        await response.release()


def messages_to_prompt(messages):
    """ 将 OpenAI-style messages 转换为 Ollama 的 prompt """
    prompt = ""
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        prompt += f"{role}: {msg['content']}\n"
    prompt += "Assistant:"
    return prompt


async def openai_llama_chat_completions(messages, stream=True, **kwargs):
    """
      模拟 OpenAI 的返回结构
      - 如果 stream=True，模拟流式返回
      - 如果 stream=False，返回完整 JSON 响应
    """
    prompt = messages_to_prompt(messages)
    response = await call_ollama(prompt, model_name="mistral", stream=stream)
    if stream:
        async def stream_data():
            async for chunk in response:
                content = chunk.get("response", "")
                data = {"id": "chatcmpl-xyz", "object": "chat.completion.chunk", "created": int(time.time()),
                        "model": "gpt-4o",
                        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]},
                yield f"data: {json.dumps(data)}\n\n"  # (f"data: {data}\n\n").encode("utf-8")

        return stream_data()

    completion = response.get("response", "")
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": completion,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt),
            "completion_tokens": len(completion),  # .split()
            "total_tokens": len(prompt) + len(completion),
        },
    }


#
# async def create_completion(
#         messages: List[Dict[str, str]],
#         refresh_token: str,
#         model: str = Config.DEFAULT_MODEL,
#         retry_count: int = 0
# ) -> Dict:
#     """同步对话补全
#
#     Args:
#         messages: 消息列表
#         refresh_token: 刷新token
#         model: 模型名称
#         retry_count: 重试次数
#
#     Returns:
#         Dict: 补全结果
#
#     Raises:
#         API_REQUEST_PARAMS_INVALID: 参数无效
#     """
#     try:
#         if not messages:
#             raise  Exception("消息不能为空")
#
#         # 解析模型参数
#         model_info = parse_model(model)
#
#         # 生成图像
#         image_urls = generate_images(
#             model=model_info['model'],
#             prompt=messages[-1]['content'],
#             width=model_info['width'],
#             height=model_info['height'],
#             refresh_token=refresh_token
#         )
#
#         # 构造返回结果
#         return {
#             'id': generate_uuid(),
#             'model': model or model_info['model'],
#             'object': 'chat.completion',
#             'choices': [{
#                 'index': 0,
#                 'message': {
#                     'role': 'assistant',
#                     'content': ''.join(f'![image_{i}]({url})\n' for i, url in enumerate(image_urls))
#                 },
#                 'finish_reason': 'stop'
#             }],
#             'usage': {
#                 'prompt_tokens': 1,
#                 'completion_tokens': 1,
#                 'total_tokens': 2
#             },
#             'created':  int(time.time())
#         }
#     except Exception as e:
#         if retry_count < Config.MAX_RETRY_COUNT:
#             print(f"Response error: {str(e)}")
#             print(f"Try again after {Config.RETRY_DELAY}s...")
#             await asyncio.sleep(Config.RETRY_DELAY)
#             return await create_completion(messages, refresh_token, model, retry_count + 1)
#         raise e

#
# async def create_completion_stream(
#         messages: List[Dict[str, str]],
#         refresh_token: str,
#         model: str = Config.DEFAULT_MODEL,
#         retry_count: int = 0
# ) -> Generator[Dict, None, None]:
#     """流式对话补全
#
#     Args:
#         messages: 消息列表
#         refresh_token: 刷新token
#         model: 模型名称
#         retry_count: 重试次数
#
#     Yields:
#         Dict: 补全结果片段
#     """
#     try:
#         if not messages:
#             yield {
#                 'id': generate_uuid(),
#                 'model': model,
#                 'object': 'chat.completion.chunk',
#                 'choices': [{
#                     'index': 0,
#                     'delta': {'role': 'assistant', 'content': '消息为空'},
#                     'finish_reason': 'stop'
#                 }]
#             }
#             return
#
#         # 解析模型参数
#         model_info = parse_model(model)
#
#         # 发送开始生成消息
#         yield {
#             'id': generate_uuid(),
#             'model': model or model_info['model'],
#             'object': 'chat.completion.chunk',
#             'choices': [{
#                 'index': 0,
#                 'delta': {'role': 'assistant', 'content': '🎨 图像生成中，请稍候...'},
#                 'finish_reason': None
#             }]
#         }
#
#         try:
#             # 生成图像
#             image_urls = generate_images(
#                 model=model_info['model'],
#                 prompt=messages[-1]['content'],
#                 width=model_info['width'],
#                 height=model_info['height'],
#                 refresh_token=refresh_token
#             )
#
#             # 发送图像URL
#             for i, url in enumerate(image_urls):
#                 yield {
#                     'id': generate_uuid(),
#                     'model': model or model_info['model'],
#                     'object': 'chat.completion.chunk',
#                     'choices': [{
#                         'index': i + 1,
#                         'delta': {
#                             'role': 'assistant',
#                             'content': f'![image_{i}]({url})\n'
#                         },
#                         'finish_reason': None if i < len(image_urls) - 1 else 'stop'
#                     }]
#                 }
#
#             # 发送完成消息
#             yield {
#                 'id': generate_uuid(),
#                 'model': model or model_info['model'],
#                 'object': 'chat.completion.chunk',
#                 'choices': [{
#                     'index': len(image_urls) + 1,
#                     'delta': {
#                         'role': 'assistant',
#                         'content': '图像生成完成！'
#                     },
#                     'finish_reason': 'stop'
#                 }]
#             }
#
#         except Exception as e:
#             # 发送错误消息
#             yield {
#                 'id': generate_uuid(),
#                 'model': model or model_info['model'],
#                 'object': 'chat.completion.chunk',
#                 'choices': [{
#                     'index': 1,
#                     'delta': {
#                         'role': 'assistant',
#                         'content': f'生成图片失败: {str(e)}'
#                     },
#                     'finish_reason': 'stop'
#                 }]
#             }
#     except Exception as e:
#         if retry_count < Config.MAX_RETRY_COUNT:
#             print(f"Response error: {str(e)}")
#             print(f"Try again after {Config.RETRY_DELAY}s...")
#             await asyncio.sleep(Config.RETRY_DELAY)
#             async for chunk in create_completion_stream(messages, refresh_token, model, retry_count + 1):
#                 yield chunk
#             return
#         raise e


def agent_func_calls(agent):
    from knowledge import ideatech_knowledge
    function_agent = {
        'default': lambda *args, **kwargs: [],
        '2': web_search,  # web_search_async
        '30': ideatech_knowledge,
        '32': ai_auto_calls
    }
    return function_agent.get(agent, lambda *args, **kwargs: [])


def async_to_sync(func, *args, **kwargs):
    return asyncio.run(func(*args, **kwargs))


async def wrap_sync(func, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)


# retriever
async def retrieved_reference(user_request: str, keywords: List[Union[str, Tuple[str, Any]]] = None,
                              tool_calls: List[Callable[[...], Any]] = None, **kwargs):
    # docs = retriever(question)
    # Assume this is the document retrieved from RAG
    # function_call = Agent_Functions.get(agent, lambda *args, **kwargs: [])
    # refer = function_call(user_message, ...)
    tool_calls = tool_calls or []
    items_to_process = []
    tasks = []  # asyncio.create_task
    if not keywords:
        items_to_process = [user_request]  # ','.join(keywords)
    else:
        if all(not (callable(func) and func.__name__ == '<lambda>' and func()) for func in tool_calls):
            tool_calls.append(ai_auto_calls)  # not in agent_funcalls web_search_async

        function_registry = {'map_search': search_amap_location,
                             'web_search': web_search_async,
                             'translate': baidu_translate,
                             'auto_calls': ai_auto_calls}

        for item in keywords:
            if isinstance(item, tuple) and len(item) > 1:
                func = function_registry.get(item[0])  # user_calls 函数
                if func:
                    func_args = item[1:]  # 剩下的参数
                    if inspect.iscoroutinefunction(func):
                        tasks.append(func(*func_args))
                    else:
                        tasks.append(wrap_sync(func, *func_args))
            else:  # isinstance(keyword, str)
                items_to_process.append(item)  # keyword

    for func in filter(callable, tool_calls):
        if func.__name__ == '<lambda>' and func() == []:  # empty_lambda
            continue
        for item in items_to_process:
            if inspect.iscoroutinefunction(func):
                tasks.append(func(item, **kwargs))
            else:
                tasks.append(wrap_sync(func, item, **kwargs))

    refer = await asyncio.gather(*tasks, return_exceptions=True)  # gather 收集所有异步调用的结果

    for f, r in zip(tasks, refer):
        if not r:
            print(f.__name__, r)
    return [item for result in refer if not isinstance(result, Exception)
            for item in (result.items() if isinstance(result, dict) else result)]  # 展平嵌套结果


# Callable[[参数类型], 返回类型]
async def get_chat_payload(messages, user_request: str, system: str = '', temperature: float = 0.4, top_p: float = 0.8,
                           max_tokens: int = 1024, model_name=Config.DEFAULT_MODEL, model_id=0,
                           tool_calls: List[Callable[[...], Any]] = None,
                           keywords: List[Union[str, Tuple[str, Any]]] = None, images: List[str] = None, **kwargs):
    model_info, name = find_ai_model(model_name, model_id, 'model')
    model_type = model_info['type']

    if isinstance(messages, list) and messages:
        if model_type in ('baidu', 'tencent'):
            if messages[0].get('role') == 'system':  # ['system,assistant,user,tool,function']
                system = messages[0].get('content')
                del messages[0]

            # the role of first message must be user
            if messages[0].get('role') != 'user':  # user（tool）
                messages.insert(0, {'role': 'user', 'content': user_request or '请问您有什么问题？'})

            # 确保 user 和 assistant 消息交替出现
            for i, message in enumerate(messages[:-1]):
                # if (
                #     isinstance(message, dict) and
                #     message.get("role") in ["user", "assistant"] and
                #     isinstance(message.get("content"), str) and
                #     message["content"].strip()  # 确保 content 非空
                # ):
                next_message = messages[i + 1]
                if message['role'] == next_message['role']:  # messages.insert(0, messages.pop(i))
                    if i % 2 == 0:
                        if message['role'] == 'user':
                            messages.insert(i + 1, {'role': 'assistant', 'content': '这是一个默认的回答。'})
                        else:
                            messages.insert(i + 1, {'role': 'user', 'content': '请问您有什么问题？'})
                    else:
                        del messages[i + 1]

        if model_type != 'baidu' and system:
            if messages[0].get('role') != 'system':
                messages.insert(0, {"role": "system", "content": system})
            # messages[-1]['content'] = messages[0]['content'] + '\n' + messages[-1]['content']

        if user_request:
            if messages[-1]["role"] != 'user':
                messages.append({'role': 'user', 'content': user_request})
            else:
                pass
                # if messages[-1]["role"] == 'user':
                #     messages[-1]['content'] = user_request
        else:
            if messages[-1]["role"] == 'user':
                user_request = messages[-1]["content"]
    else:
        if messages is None:
            messages = []
        if model_type != 'baidu' and system:  # system_message
            messages = [{"role": "system", "content": system}]
        messages.append({'role': 'user', 'content': user_request})

    refer = await retrieved_reference(user_request, keywords, tool_calls, **kwargs)
    if refer:
        formatted_refer = '\n'.join(map(str, refer))
        # """Answer the users question using only the provided information below:{docs}""".format(docs=formatted_refer)
        messages[-1]['content'] = (f'以下是相关参考资料:\n{formatted_refer}\n'
                                   f'请结合以上内容或根据上下文，针对下面的问题进行解答：\n{user_request}')

    if images:  # 图片内容理解
        messages[-1]['content'] = [{"type": "text", "text": user_request}]  # text-prompt 请详细描述一下这几张图片。这是哪里？
        messages[-1]['content'] += [{"type": "image_url", "image_url": {"url": image}} for image in images]

    payload = dict(
        model=name,  # 默认选择第一个模型
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        # top_k=50,
        max_tokens=max_tokens,
        # extra_body = {"prefix": "```python\n", "suffix":"后缀内容"} 希望的前缀内容,基于用户提供的前缀信息来补全其余的内容
        # response_format={"type": "json_object"}
        # "tools":retrieval、web_search、function
    )
    if model_type == 'baidu':
        payload['system'] = system
    # if model_info['name']=='baichuan':
    #     extra_body = {
    #         "tools": [{
    #             "type": "retrieval",
    #             "retrieval": {
    #                 "kb_ids": [
    #                     "kb-123",
    #                      "kb-xxx"
    #                 ],
    #           "answer_mode": "knowledge-base-only"
    #             }
    #         }]
    #     }
    #     [{
    #         "type": "web_search",
    #         "web_search": {
    #             "enable": True,
    #             "search_mode": "performance_first"#"quality_first"
    #         }
    #     }]

    # print(payload)
    return model_info, payload, refer


def get_chat_payload_post(model_info, payload):
    # 通过 requests 库直接发起 HTTP POST 请求
    url = model_info['url']
    api_key = model_info['api_key']
    headers = {'Content-Type': 'application/json', }
    payload = payload.deepcopy()
    if api_key:
        if isinstance(api_key, list):
            idx = model_info['model'].index(payload["model"])
            api_key = model_info['api_key'][idx]

        headers = {'Content-Type': 'application/json',
                   "Authorization": f'Bearer {api_key}'
                   }
    if model_info['type'] == 'baidu':  # 'ernie'
        url = build_url(f'{url}{payload["model"]}',
                        get_baidu_access_token(Config.BAIDU_qianfan_API_Key,
                                               Config.BAIDU_qianfan_Secret_Key))  # ?access_token=" + get_access_token()
        # print(url)
        # payload.pop('model', None)
        payload['max_output_tokens'] = payload.pop('max_tokens', 1024)
        # payload['enable_system_memory'] = False
        # payload["enable_citation"]= False
        # payload["disable_search"] = False
        # payload["user_id"]=
        # payload["user_ip"]=
        # payload['system'] = system
    if model_info['type'] == 'tencent':
        service = 'hunyuan'
        host = url.split("//")[-1]
        payload = convert_keys_to_pascal_case(payload)
        payload.pop('MaxTokens', None)
        headers = get_tencent_signature(service, host, payload, action='ChatCompletions',
                                        secret_id=Config.TENCENT_SecretId,
                                        secret_key=Config.TENCENT_Secret_Key)
        headers['X-TC-Version'] = '2023-09-01'
        # headers["X-TC-Region"] = 'ap-shanghai'

    # if model_info['name'] == 'silicon':
    #     headers = {
    #         "accept": "application/json",
    #         "content-type": "application/json",
    #         "authorization": "Bearer sk-tokens"
    #     }
    return url, headers, payload


async def ai_chat(model_info, payload=None, get_content=True, **kwargs):
    if not payload:
        model_info, payload, _ = await get_chat_payload(**kwargs)
    else:
        payload.update(kwargs)  # {**payload, **kwargs}

    client = AI_Client.get(model_info['name'], None)
    if client:
        try:
            # await asyncio.to_thread(client.chat.completions.create, **payload)
            completion = await client.chat.completions.create(**payload)
            return completion.choices[0].message.content if get_content else completion.model_dump()  # 自动序列化为 JSON
            # json.loads(completion.model_dump_json())
        except Exception as e:
            return f"OpenAI error occurred: {e}"

    url, headers, payload = get_chat_payload_post(model_info, payload)
    # print(headers, payload)
    limits = httpx.Limits(max_connections=100, max_keepalive_connections=10)
    parse_rules = {
        'baidu': lambda d: d.get('result'),
        'tencent': lambda d: d.get('Response', {}).get('Choices', [{}])[0].get('Message', {}).get('Content'),
        # d.get('Choices')[0].get('Message').get('Content')
        'default': lambda d: d.get('choices', [{}])[0].get('message', {}).get('content')
    }

    try:
        async with httpx.AsyncClient(limits=limits, timeout=Config.HTTP_TIMEOUT_SEC) as cx:
            response = await cx.post(url, headers=headers, json=payload)
            response.raise_for_status()  # 如果请求失败，则抛出异常
            data = response.json()
            if get_content:
                result = parse_rules.get(model_info['type'], parse_rules['default'])(data)
                if result:
                    return result
                print(response.text)
            return data

    except Exception as e:
        # print(response.text)
        return f"HTTP error occurred: {e}"


async def ai_chat_async(model_info, payload=None, get_content=True, **kwargs):
    if not payload:
        model_info, payload, _ = await get_chat_payload(**kwargs)
    else:
        # payload=copy(payload)
        payload.update(kwargs)

    payload["stream"] = True
    # payload["stream_options"]= {"include_usage": True}        # 可选，配置以后会在流式输出的最后一行展示token使用信息
    client = AI_Client.get(model_info['name'], None)
    # print(payload)
    if client:
        try:
            stream = await client.chat.completions.create(**payload)
            async for chunk in stream:  # for chunk in stream
                if get_content:
                    delta = chunk.choices[0].delta
                    if delta.content:  # 以两个换行符 \n\n 结束当前传输的数据块
                        yield delta.content  # completion.append(delta.content)
                else:
                    yield chunk.model_dump_json()  # 获取字节流数据
        except Exception as e:
            yield f"OpenAI error occurred: {e}"
        # yield '[DONE]'
        return  # 异步生成器的结束无需返回值

    url, headers, payload = get_chat_payload_post(model_info, payload)
    limits = httpx.Limits(max_connections=100, max_keepalive_connections=10)
    try:
        async with httpx.AsyncClient(limits=limits) as cx:
            async with cx.stream("POST", url, headers=headers, json=payload) as response:
                response.raise_for_status()
                async for content in process_line_stream(response, model_info['type']):
                    # print(chunk["choices"][0]["delta"].get("content", ""), end="", flush=True)
                    yield content
    except httpx.RequestError as e:
        yield str(e)

    # yield "[DONE]"


async def process_line_stream(response, model_type='default'):
    data = ""
    async for line in response.aiter_lines():
        # print(line)
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
                    # raise StopIteration(2)
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
            return chunk.get("result") or chunk.get("error_msg")
        else:
            if model_type == 'tencent':
                chunk = convert_keys_to_lower_case(chunk)

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


def web_search(text: str, api_key: str = Config.GLM_Service_Key) -> list:
    """
    通过提供的查询文本执行网络搜索。

    :param text: 查询文本。
    :param api_key: 用于访问网络搜索工具的API密钥（默认为配置中的密钥）。
    :return: 搜索结果列表或错误信息。
    """
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

    async with httpx.AsyncClient(timeout=Config.HTTP_TIMEOUT_SEC) as cx:
        try:
            response = await cx.post(url, json=data, headers=headers)
            response.raise_for_status()

            data = response.json()
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
        # finally:
        #     await response.close()

        # https://portal.azure.com/#home


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
    response = requests.get(f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&format=json",
                            timeout=Config.HTTP_TIMEOUT_SEC, proxies=Config.HTTP_Proxies)
    search_results = response.json().get('query', {}).get('search', [])
    if search_results:
        page_id = search_results[0]['pageid']
        page_response = requests.get(
            f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&pageids={page_id}&format=json")
        page_data = page_response.json()['query']['pages'][str(page_id)]
        return page_data.get('extract', 'No extract found.')
    return "No information found."


Assistant_Cache = {}


# https://platform.baichuan-ai.com/docs/assistants
async def ai_assistant_create(instructions: str, user_name: str, tools_type="code_interpreter", model_id=4):
    # 如果您判断是一次连续的对话，则无需自己拼接上下文，只需将最新的 message 添加到对应的 thread id 即可得到包含了 thread id 历史上下文的回复，历史上下文超过我们会帮您自动截断。
    global Assistant_Cache
    cache_key = generate_hash_key(instructions, user_name, tools_type, model_id)
    if cache_key in Assistant_Cache:
        return Assistant_Cache[cache_key]

    model_info, model_name = find_ai_model('baichuan', model_id, 'model')
    assistants_url = f"{model_info['base_url']}assistants"
    threads_url = f"{model_info['base_url']}threads"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {model_info['api_key']}"
    }
    payload = {
        "instructions": instructions,
        "name": user_name,
        "tools": [{"type": tools_type}],  # web_search,code_interpreter,function
        "model": model_name
    }
    try:
        async with httpx.AsyncClient(timeout=Config.HTTP_TIMEOUT_SEC) as cx:
            assistant_response = await cx.post(assistants_url, headers=headers, json=payload)
            assistant_response.raise_for_status()
            threads_response = await cx.post(threads_url, headers=headers, json={})
            threads_response.raise_for_status()
            results = {'headers': headers,
                       'assistant_data': assistant_response.json(),
                       'threads_url': threads_url,
                       'threads_id': threads_response.json()['id']}
            Assistant_Cache[cache_key] = results  # assistant_response.json()["id"],assistant_id
            return results
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


async def ai_assistant_run(user_request: str, instructions: str, user_name: str, tools_type="code_interpreter",
                           model_id=4, max_retries=20, interval=5, backoff_factor=1.5):
    assistant = await ai_assistant_create(instructions, user_name, tools_type, model_id)
    if "error" in assistant:
        return assistant

    assistant_data = assistant['assistant_data']
    headers = assistant['headers']
    messages_url = f'{assistant["threads_url"]}/{assistant["threads_id"]}/messages'
    threads_url = f'{assistant["threads_url"]}/{assistant["threads_id"]}/runs'

    async with httpx.AsyncClient(timeout=Config.HTTP_TIMEOUT_SEC) as cx:
        messages_response = await cx.post(messages_url, headers=headers, json={"role": "user", "content": user_request})
        messages_response.raise_for_status()
        run_response = await cx.post(threads_url, headers=headers, json={"assistant_id": assistant_data['id']})
        run_response.raise_for_status()
        run_id = run_response.json().get('id')
        run_url = f'{threads_url}/{run_id}'
        retries = 0
        while retries < max_retries:
            try:
                run_status_response = await cx.get(run_url, headers=headers)  # 定期检索 run id 的运行状态
                # print(retries,run_status_response.status_code, run_status_response.headers, run_status_response.text)

                run_status_response.raise_for_status()
                status_data = run_status_response.json()

                if status_data.get('status') == 'completed':
                    messages_status_response = await cx.get(messages_url, headers=headers)
                    run_status_response.raise_for_status()
                    status_data = messages_status_response.json()
                    # print(status_data)
                    return status_data.get('data')

                await asyncio.sleep(interval)
                retries += 1

            except httpx.HTTPStatusError as http_error:
                if run_status_response.status_code == 429:
                    await asyncio.sleep(interval)
                    retries += 1
                    interval *= backoff_factor
                    continue

                return {"error": f"HTTP error: {str(http_error)}"}  # status_data['error']

            except Exception as e:
                return {"error": f"Unexpected error: {str(e)}"}

        return {
            "error": f"Timeout: The task did not complete within the expected time. status:{status_data},retries:{retries}"}


# （1）文本数量不超过 16。 （2）每个文本长度不超过 512 个 token，超出自动截断，token 统计信息，token 数 = 汉字数+单词数*1.3 （仅为估算逻辑，以实际返回为准)。
async def get_baidu_embeddings(texts: List[str], access_token: str, model_name='bge_large_zh') -> List:
    if not isinstance(texts, list):
        texts = [texts]

    url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/embeddings/{model_name}?access_token={access_token}"
    headers = {
        'Content-Type': 'application/json',
        # "Authorization: Bearer $BAICHUAN_API_KEY"
    }
    batch_size = 16
    embeddings = []
    async with httpx.AsyncClient(timeout=Config.HTTP_TIMEOUT_SEC) as cx:
        if len(texts) < batch_size:
            payload = json.dumps({"input": texts})  # "model":
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


def get_hf_embeddings(texts, model_name='BAAI/bge-large-zh-v1.5', access_token=Config.HF_Service_Key):
    # "https://api-inference.huggingface.co/models/BAAI/bge-reranker-large"
    url = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": f'Bearer {access_token}'}
    payload = {"inputs": texts}
    response = requests.post(url, headers=headers, json=payload, proxies=Config.HTTP_Proxies,
                             timeout=Config.HTTP_TIMEOUT_SEC)
    data = response.json().get('data')
    return [emb.get('embedding') for emb in data]


async def similarity_embeddings(query, tokens: List[str], filter_idx: List[int] = None, tokens_vector=None,
                                embeddings_calls: Callable[[...], Any] = ai_embeddings, **kwargs):
    """
    计算查询与一组标记之间的相似度。
    参数:
        query (str): 查询字符串。
        tokens (List[str]): 要比较的标记列表。
        filter_idx (List[int], optional): 要过滤出的标记索引。默认为 `None`，表示不进行过滤。
        tokens_vector (np.ndarray, optional): 预计算的标记向量数组。默认为 `None`。
        embeddings_calls (Callable, optional): 用于生成嵌入的异步函数，默认为 `None`。
        **kwargs: 传递给 `embeddings_calls` 的额外参数。

    返回:
        np.ndarray: 相似度得分数组（与 `tokens` 长度一致）。
    """
    if filter_idx is None:
        filter_idx = list(range(len(tokens)))
    else:
        filter_idx = [i for i in filter_idx if i < len(tokens)]
    similarity = np.full(len(tokens), np.nan)
    if not query:
        return similarity

    tokens_idx = [(i, token) for i, token in enumerate(tokens) if token]
    query_vector = await embeddings_calls(query, **kwargs)
    if tokens_vector is None:
        # list(np.array(idx_tokens)[:, 1])
        tokens_vector = await embeddings_calls([token for _, token in tokens_idx], **kwargs)

    matching_indices = np.isin([j[0] for j in tokens_idx], filter_idx)
    filter_embeddings = np.array(tokens_vector)[matching_indices]

    if len(filter_embeddings):
        # cosine_similarity_np(np.array(query_vector).reshape(1, -1), filter_embeddings).T
        sim_2d = np.array(query_vector).reshape(1, -1) @ filter_embeddings.T
        # matching_indices = np.isin(filter_idx, [j[0] for j in tokens_idx])
        # similarity[matching_indices] = sim_2d.reshape(-1)
        for idx, score in zip(filter_idx, sim_2d.flatten()):
            similarity[idx] = score

    return similarity


def get_similar_vectors(querys, data, exclude: List[str] = None, topn: int = 10, cutoff: float = 0.0):
    '''
    data={
      "name": ["word1", "word2"],
      "vectors": [[0.1, 0.2, ...], [0.3, 0.4, ...]]
    }
    返回：
        List[Tuple[str, List[Tuple[str, float]]]]: 查询词与相似标记及其分数的映射。
    '''
    index_to_key = data['name']
    vectors = np.array(data['vectors'])

    # 获取索引
    query_mask = np.array([w in index_to_key for w in querys])
    exclude_mask = np.array([w in querys + exclude for w in index_to_key])
    # np.delete(vectors, exclude_indices, axis=0)

    # 计算余弦相似度
    sim_matrix = cosine_similarity_np(vectors[query_mask], vectors[~exclude_mask].T)

    results = []
    for i, w in enumerate(querys):
        if not query_mask[i]:
            continue
        sim_scores = sim_matrix[i]
        if topn > 0:
            top_indices = np.argsort(sim_scores)[::-1][:topn]  # 获取前 topn 个索引
        else:
            top_indices = np.arange(sim_scores.shape[0])  # 获取全部索引，不排序
        top_scores = sim_scores[top_indices]

        valid_indices = top_scores > cutoff  # 保留大于 cutoff 的相似度
        top_words = [index_to_key[j] for j in np.where(~exclude_mask)[0][top_indices[valid_indices]]]
        top_scores = top_scores[valid_indices]
        results.append((w, list(zip(top_words, top_scores))))

    return results  # [(w, list(sim[w].sort_values(ascending=False)[:topn].items())) for w in querys]


async def get_similar_embeddings(querys: List[str], tokens: List[str],
                                 embeddings_calls: Callable[[...], Any] = ai_embeddings, topn=10, **kwargs):
    """
    使用嵌入计算查询与标记之间的相似度。
    :param querys: 查询词列表
    :param tokens: 比较词列表
    :param embeddings_calls: 嵌入生成函数
    :param topn: 返回相似结果的数量
    :param kwargs: 其他参数
    返回：
        List[Tuple[str, List[Tuple[str, float]]]]: 查询词与相似标记及其分数的映射。
    """
    query_vector, tokens_vector = await asyncio.gather(
        embeddings_calls(querys, **kwargs),
        embeddings_calls(tokens, **kwargs))

    if not query_vector or not tokens_vector:
        return []

    sim_matrix = np.array(query_vector) @ np.array(tokens_vector).T
    results = []
    for i, w in enumerate(querys):
        sim_scores = sim_matrix[i]
        top_indices = np.argsort(sim_scores)[::-1][:topn] if topn > 0 else np.arange(sim_scores.shape[0])
        top_scores = sim_scores[top_indices]
        top_match = [tokens[j] for j in top_indices]
        results.append((w, list(zip(top_match, top_scores, top_indices))))

    return results  # [(q,[(match,score,index),])]


async def find_closest_matches_embeddings(querys, tokens,
                                          embeddings_calls: Callable[[...], Any] = ai_embeddings, **kwargs):
    """
    使用嵌入计算查询与标记之间的最近匹配,找到每个查询的最佳匹配标记。
    返回：
        Dict[str, Tuple[str, float]]: 查询与最近匹配标记的映射字典。
    """
    matchs = {x: (x, 1.0) for x in querys if x in tokens}
    unmatched_queries = list(set(querys) - matchs.keys())
    if not unmatched_queries:
        return matchs
    query_vector, tokens_vector = await asyncio.gather(
        embeddings_calls(unmatched_queries, **kwargs),
        embeddings_calls(tokens, **kwargs))

    sim_matrix = np.array(query_vector) @ np.array(tokens_vector).T
    closest_matches = tokens[sim_matrix.argmax(axis=1)]  # idxmax
    closest_scores = sim_matrix.max(axis=1)
    matchs.update(zip(unmatched_queries, zip(closest_matches, closest_scores)))
    return matchs


def is_city(city, region='全国'):
    # https://restapi.amap.com/v3/geocode/geo?parameters
    response = requests.get(url="http://api.map.baidu.com/place/v2/suggestion",
                            params={'query': city, 'region': region,
                                    "output": "json", "ak": Config.BMAP_API_Key, })
    data = response.json()

    # 判断返回结果中是否有城市匹配
    for result in data.get('result', []):
        if result.get('city') == city:
            return True
    return False


def get_bmap_location(address, city=''):
    response = requests.get(url="https://api.map.baidu.com/geocoding/v3",
                            params={"address": address,
                                    "city": city,
                                    "output": "json",
                                    "ak": Config.BMAP_API_Key, })
    if response.status_code == 200:
        locat = response.json()['result']['location']
        return round(locat['lng'], 6), round(locat['lat'], 6)
    else:
        print(response.text)
    return None, None


# https://lbsyun.baidu.com/faq/api?title=webapi/place-suggestion-api
async def search_bmap_location(query, region='', limit=True):
    url = "http://api.map.baidu.com/place/v2/suggestion"  # 100
    params = {
        "query": query,
        "region": region,
        "city_limit": 'true' if (region and limit) else 'false',
        "output": "json",
        "ak": Config.BMAP_API_Key,
    }

    async with httpx.AsyncClient(timeout=Config.HTTP_TIMEOUT_SEC) as client:
        response = await client.get(url, params=params)
        res = []
        if response.status_code == 200:
            js = response.json()
            for result in js.get('result', []):
                res.append({'lng_lat': (round(result['location']['lng'], 6), round(result['location']['lat'], 6)),
                            'name': result["name"], 'address': result['address']})
        else:
            print(response.text)
        return res  # baidu_nlp(nlp_type='address', text=region+query+result["name"]+ result['address'])


def get_amap_location(address, city=''):
    response = requests.get(url="https://restapi.amap.com/v3/geocode/geo?parameters",
                            params={"address": address,
                                    "city": city,
                                    "output": "json",
                                    "key": Config.AMAP_API_Key, })

    if response.status_code == 200:
        js = response.json()
        if js['status'] == '1':
            s1, s2 = js['geocodes'][0]['location'].split(',')
            return float(s1), float(s2)  # js['geocodes'][0]['formatted_address']
    else:
        print(response.text)

    return None, None


# https://lbs.amap.com/api/webservice/guide/api-advanced/search
async def search_amap_location(query, region='', limit=True):
    url = "https://restapi.amap.com/v5/place/text?parameters"  # 100
    params = {
        "keywords": query,
        "region": region,
        "city_limit": 'true' if (region and limit) else 'false',
        "output": "json",
        "key": Config.AMAP_API_Key,
    }

    async with httpx.AsyncClient(timeout=Config.HTTP_TIMEOUT_SEC) as client:
        response = await client.get(url, params=params)
        res = []
        if response.status_code == 200:
            js = response.json()
            if js['status'] == '1' and int(js['count']) > 0:
                for result in js.get('pois', []):
                    s1, s2 = result['location'].split(',')
                    res.append({'lng_lat': (float(s1), float(s2)),
                                'name': result["name"], 'address': result['address']})
            else:
                print(response.text)
        return res


# https://console.bce.baidu.com/ai/#/ai/machinetranslation/overview/index
async def baidu_translate(text: str, from_lang: str = 'zh', to_lang: str = 'en', trans_type='texttrans'):
    """百度翻译 API"""
    salt = str(random.randint(32768, 65536))  # str(int(time.time() * 1000))
    sign_str = Config.BAIDU_trans_AppId + text + salt + Config.BAIDU_trans_Secret_Key
    sign = md5_sign(sign_str)  # 需要计算 sign = MD5(appid+q+salt+密钥)
    url = "https://fanyi-api.baidu.com/api/trans/vip/translate"
    lang_map = {'fa': 'fra', 'ja': 'jp', 'ar': 'ara', 'ko': 'kor', 'es': 'spa', 'zh-TW': 'cht', 'vi': 'vie'}

    if from_lang in lang_map.keys():
        from_lang = lang_map[from_lang]
    if to_lang in lang_map.keys():
        to_lang = lang_map[to_lang]

    if to_lang == 'auto':
        to_lang = 'zh'

    params = {
        "q": text,
        "from": from_lang,
        "to": to_lang,
        "appid": Config.BAIDU_trans_AppId,
        "salt": salt,
        "sign": sign
    }

    async with httpx.AsyncClient(timeout=Config.HTTP_TIMEOUT_SEC) as client:
        response = await client.get(url, params=params)

    data = response.json()
    if "trans_result" in data:
        return data["trans_result"][0]["dst"]

    print(response.text)
    # texttrans-with-dict
    url = f"https://aip.baidubce.com/rpc/2.0/mt/{trans_type}/v1?access_token=" + get_baidu_access_token(
        Config.BAIDU_translate_API_Key, Config.BAIDU_translate_Secret_Key)
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    payload = json.dumps({
        "from": from_lang,
        "to": to_lang
    })
    async with httpx.AsyncClient(timeout=Config.HTTP_TIMEOUT_SEC) as client:
        response = await client.post(url, headers=headers, json=payload)

    data = response.json()
    if "trans_result" in data:
        return data["trans_result"][0]["dst"]

    # print(response.text)
    return {'error': data.get('error_msg', 'Unknown error')}


# https://cloud.tencent.com/document/product/551/15619
async def tencent_translate(text: str, source: str, target: str):
    payload = {
        "SourceText": text,
        "Source": source,
        "Target": target,
        "ProjectId": 0
    }
    url = "https://tmt.tencentcloudapi.com"
    headers = get_tencent_signature(service="tmt", host="tmt.tencentcloudapi.com", body=payload,
                                    action='TextTranslate',
                                    secret_id=Config.TENCENT_SecretId, secret_key=Config.TENCENT_Secret_Key,
                                    timestamp=int(time.time()), version='2018-03-21')

    async with httpx.AsyncClient(timeout=Config.HTTP_TIMEOUT_SEC) as client:
        response = await client.post(url, headers=headers, json=payload)

    # 检查响应状态码和内容
    if response.status_code != 200:
        print(f"Error: Received status code {response.status_code},Response content: {response.text}")
        return {'error': f'{response.status_code},Request failed'}

    try:
        data = response.json()
    except Exception as e:
        print(f"Failed to decode JSON: {e},Response text: {response.text}")
        return {'error': f"Failed to decode JSON: {e},Response text: {response.text}"}

    if "Response" in data and "TargetText" in data["Response"]:
        return data["Response"]["TargetText"]
    else:
        print(f"Unexpected response: {data}")
        return {'error': f"Tencent API Error: {data.get('Response', 'Unknown error')}"}


# https://www.xfyun.cn/doc/nlp/xftrans/API.html
async def xunfei_translate(text: str, source: str = 'en', target: str = 'cn'):
    # 将文本进行base64编码
    encoded_text = base64.b64encode(text.encode('utf-8')).decode('utf-8')

    # 构造请求数据
    request_data = {
        "header": {
            "app_id": Config.XF_AppID,  # 你在平台申请的appid
            "status": 3,
            # "res_id": "your_res_id"  # 可选：自定义术语资源id
        },
        "parameter": {
            "its": {
                "from": source,
                "to": target,
                "result": {}
            }
        },
        "payload": {
            "input_data": {
                "encoding": "utf8",
                "status": 3,
                "text": encoded_text
            }
        }
    }

    headers, url = get_xfyun_authorization(api_key=Config.XF_API_Key, api_secret=Config.XF_Secret_Key,
                                           host="itrans.xf-yun.com", path="/v1/its", method='POST')
    url = 'https://itrans.xf-yun.com/v1/its'

    # 异步发送请求
    async with httpx.AsyncClient(timeout=Config.HTTP_TIMEOUT_SEC) as client:
        response = await client.post(url, json=request_data, headers=headers)
        if response.status_code == 200:
            response_data = response.json()

            # 解码返回结果中的text字段
            if "payload" in response_data and "result" in response_data["payload"]:
                base64_text = response_data["payload"]["result"]["text"]
                decoded_result = base64.b64decode(base64_text).decode('utf-8')
                data = json.loads(decoded_result)
                if "trans_result" in data:
                    return data["trans_result"]["dst"]
            else:
                return {"error": "Unexpected response format"}
        else:
            return {"error": f"HTTP Error: {response.status_code}"}


# https://docs.caiyunapp.com/lingocloud-api/
def caiyun_translate(source, direction="auto2zh"):
    url = "http://api.interpreter.caiyunai.com/v1/translator"

    # WARNING, this token is a test token for new developers,
    token = Config.CaiYun_Token

    payload = {
        "source": source,
        "trans_type": direction,
        "request_id": "demo",
        "detect": True,
    }

    headers = {
        "content-type": "application/json",
        "x-authorization": "token " + token,
    }

    response = requests.request("POST", url, data=json.dumps(payload), headers=headers)
    return json.loads(response.text)["target"]


# https://ai.youdao.com/
# https://hcfy.ai/docs/services/youdao-api
async def auto_translate(text: str, model_name='baidu', source: str = 'auto', target: str = 'auto'):
    """
       自动翻译函数，根据输入的文本和指定的翻译模型自动完成语言检测和翻译。
        功能描述:
    1. 自动检测源语言，如果 `source` 为 "auto"：
        - 使用 `detect` 方法检测语言。
        - 如果检测结果是中文，标准化为 "zh"。
        - 如果检测结果不明确，默认根据内容是否包含中文设定语言。
    2. 自动设定目标语言，如果 `target` 为 "auto"：
        - 如果源语言是中文，则目标语言为英文 ("en")。
        - 如果源语言是其他语言，则目标语言为中文 ("zh")。
    3. 根据指定的 `model_name` 调用对应的翻译模型处理文本。
        - 如果未找到对应模型，则调用 `ai_generate` 生成翻译。
    4. 返回翻译结果及相关信息。
    """
    if source == 'auto':
        source = detect(text)
        if source == 'zh-cn':
            source = 'zh'
        if source == 'no':
            source = 'zh' if contains_chinese(text) else 'auto'
    if target == 'auto':
        target = 'en' if source == 'zh' else 'zh'

    translate_map: dict = {
        "baidu": baidu_translate,
        "tencent": tencent_translate,
        "xunfei": xunfei_translate
    }
    error = ''
    handler = translate_map.get(model_name)
    if handler:
        translated_text = await handler(text, source, target)
        if isinstance(translated_text, str):
            return {"translated": translated_text, 'from': source, 'to': target, "model": model_name}

        error = translated_text.get('error')

    system_prompt = System_content.get('9').format(source_language=source, target_language=target)

    model_map = {"baidu": 'ernie', "tencent": "hunyuan", "xunfei": 'spark'}
    if model_name in model_map.keys():
        model_name = model_map[model_name]
    if not model_name:
        model_name = 'qwen'

    translated_text = await ai_generate(
        prompt=system_prompt,
        user_request=text,
        model_name=model_name,
        model_id=0,
        stream=False,
    )
    if translated_text:
        return {"translated": translated_text, 'from': source, 'to': target, "model": model_name}

    return {"translated": error, 'from': source, 'to': target, "model": model_name}


def xunfei_ppt_theme(industry, style="简约", color="蓝色", appid: str = Config.XF_AppID,
                     api_secret: str = Config.XF_Secret_Key):
    url = "https://zwapi.xfyun.cn/api/ppt/v2/template/list"
    timestamp = int(time.time())
    signature = get_xfyun_signature(appid, api_secret, timestamp)
    headers = {
        "appId": appid,
        "timestamp": str(timestamp),
        "signature": signature,
        "Content-Type": "application/json; charset=utf-8"
    }
    # body ={
    #     "query": text,
    #     "templateId": templateId  # 模板ID举例，具体使用 /template/list 查询
    # }
    body = {
        "payType": "not_free",
        "style": style,  # 支持按照类型查询PPT 模板,风格类型： "简约","卡通","商务","创意","国风","清新","扁平","插画","节日"
        "color": color,  # 支持按照颜色查询PPT 模板,颜色类型： "蓝色","绿色","红色","紫色","黑色","灰色","黄色","粉色","橙色"
        "industry": industry,
        # 支持按照颜色查询PPT 模板,行业类型： "科技互联网","教育培训","政务","学院","电子商务","金融战略","法律","医疗健康","文旅体育","艺术广告","人力资源","游戏娱乐"
        "pageNum": 2,
        "pageSize": 10
    }

    response = requests.request("GET", url=url, headers=headers, params=body).text
    return response


# https://www.xfyun.cn/doc/spark/PPTv2.html
async def xunfei_ppt_create(text: str, templateid: str = "20240718489569D", appid: str = Config.XF_AppID,
                            api_secret: str = Config.XF_Secret_Key, max_retries=20):
    from requests_toolbelt.multipart.encoder import MultipartEncoder

    url = 'https://zwapi.xfyun.cn/api/ppt/v2/create'
    timestamp = int(time.time())
    signature = get_xfyun_signature(appid, api_secret, timestamp)
    form_data = MultipartEncoder(
        fields={
            # "file": (path, open(path, 'rb'), 'text/plain'),  # 如果需要上传文件，可以将文件路径通过path 传入
            # "fileUrl":"",   #文件地址（file、fileUrl、query必填其一）
            # "fileName":"",   # 文件名(带文件名后缀；如果传file或者fileUrl，fileName必填)
            "query": text,
            "templateId": templateid,  # 模板的ID,从PPT主题列表查询中获取
            "author": "XXXX",  # PPT作者名：用户自行选择是否设置作者名
            "isCardNote": str(True),  # 是否生成PPT演讲备注, True or False
            "search": str(True),  # 是否联网搜索,True or False
            "isFigure": str(True),  # 是否自动配图, True or False
            "aiImage": "normal"  # ai配图类型： normal、advanced （isFigure为true的话生效）；
            # normal-普通配图，20%正文配图；advanced-高级配图，50%正文配图
        }
    )

    print(form_data)
    headers = {
        "appId": appid,
        "timestamp": str(timestamp),
        "signature": signature,
        "Content-Type": form_data.content_type
    }

    response = requests.request(method="POST", url=url, data=form_data, headers=headers).text
    resp = json.loads(response)
    if resp.get('code') != 0:
        print('创建PPT任务失败,生成PPT返回结果：', response)
        return None

    task_id = resp['data']['sid']
    ppt_url = ''
    retries = 0
    # 轮询任务进度
    await asyncio.sleep(5)

    async with httpx.AsyncClient(timeout=Config.HTTP_TIMEOUT_SEC) as cx:
        while task_id is not None and retries < max_retries:
            task_url = f"https://zwapi.xfyun.cn/api/ppt/v2/progress?sid={task_id}"
            response = await cx.get(url=task_url, headers=headers)
            response.raise_for_status()
            resp = json.loads(response)
            task_status = resp['data']['pptStatus']
            aiImageStatus = resp['data']['aiImageStatus']
            cardNoteStatus = resp['data']['cardNoteStatus']

            if ('done' == task_status and 'done' == aiImageStatus and 'done' == cardNoteStatus):
                ppt_url = resp['data']['pptUrl']
                break

            await asyncio.sleep(3)
            retries += 1

    return ppt_url


# https://www.xfyun.cn/doc/spark/ImageGeneration.html#%E9%89%B4%E6%9D%83%E8%AF%B4%E6%98%8E
async def xunfei_picture(text: str, data_folder=None):
    headers, url = get_xfyun_authorization(api_key=Config.XF_API_Key, api_secret=Config.XF_Secret_Key,
                                           host="spark-api.cn-huabei-1.xf-yun.com", path="/v2.1/tti", method='POST')
    url = 'http://spark-api.cn-huabei-1.xf-yun.com/v2.1/tti' + "?" + urlencode(headers)
    # 构造请求数据
    request_body = {
        "header": {
            "app_id": Config.XF_AppID,  # 你在平台申请的appid
            # 'uid'
            # "res_id": "your_res_id"  # 可选：自定义术语资源id
        },
        "parameter": {
            "chat": {
                "domain": "general",
                "temperature": 0.5,
                # "max_tokens": 4096,
                "width": 640,  # 默认大小 512*512
                "height": 480
            }
        },
        "payload": {
            "message": {
                "text": [
                    {
                        "role": "user",
                        "content": text
                    }
                ]
            }
        }
    }
    # 异步发送请求
    async with httpx.AsyncClient(timeout=Config.HTTP_TIMEOUT_SEC) as client:
        response = await client.post(url, json=request_body, headers={'content-type': "application/json"})
        if response.status_code != 200:
            return None, {"error": f"HTTP Error: {response.status_code}"}

        data = response.json()  # json.loads(response.text)
        code = data['header']['code']
        if code != 0:
            return None, {"error": f'请求错误: {code}, {data}'}

        text = data["payload"]["choices"]["text"]
        image_base = text[0]["content"]  # base64图片结果,base64_string_data
        image_id = data['header']['sid']
        # 解码 Base64 图像数据
        file_data = base64.b64decode(image_base)
        if data_folder:
            # 将解码后的数据转换为图片
            file_path = f"{data_folder}/{image_id}.jpg"  # .png
            img = Image.open(io.BytesIO(file_data))
            # r, g, b, a = img.split()
            # img = img.convert('RGB') 转换为 RGB 格式
            img.save(file_path)

            # buffer = io.BytesIO()
            # img.save(buffer, format="JPEG")  # 保存为 JPEG 格式
            # buffer.seek(0)
            # async with aiofiles.open(file_path, 'wb') as file:
            #     await file.write(buffer.read())
            return file_path, {"urls": '', 'id': image_id}

        return file_data, {"urls": '', 'id': image_id}


# https://www.volcengine.com/docs/6791/1361423
async def ark_visual_picture(image_data, image_urls: List[str], prompt: str = None, logo_info=None, style_name='3D风',
                             return_url=False, data_folder=None):
    style_mapping = {
        "3D风": ("img2img_disney_3d_style", ""),
        "写实风": ("img2img_real_mix_style", ""),
        "天使风": ("img2img_pastel_boys_style", ""),
        "动漫风": ("img2img_cartoon_style", ""),
        "日漫风": ("img2img_makoto_style", ""),
        "公主风": ("img2img_rev_animated_style", ""),
        "梦幻风": ("img2img_blueline_style", ""),
        "水墨风": ("img2img_water_ink_style", ""),
        "新莫奈花园": ("i2i_ai_create_monet", ""),
        "水彩风": ("img2img_water_paint_style", ""),
        "莫奈花园": ("img2img_comic_style", "img2img_comic_style_monet"),
        "精致美漫": ("img2img_comic_style", "img2img_comic_style_marvel"),
        "赛博机械": ("img2img_comic_style", "img2img_comic_style_future"),
        "精致韩漫": ("img2img_exquisite_style", ""),
        "国风-水墨": ("img2img_pretty_style", "img2img_pretty_style_ink"),
        "浪漫光影": ("img2img_pretty_style", "img2img_pretty_style_light"),
        "陶瓷娃娃": ("img2img_ceramics_style", ""),
        "中国红": ("img2img_chinese_style", ""),
        "丑萌粘土": ("img2img_clay_style", "img2img_clay_style_3d"),
        "可爱玩偶": ("img2img_clay_style", "img2img_clay_style_bubble"),
        "3D-游戏_Z时代": ("img2img_3d_style", "img2img_3d_style_era"),
        "动画电影": ("img2img_3d_style", "img2img_3d_style_movie"),
        "玩偶": ("img2img_3d_style", "img2img_3d_style_doll"),
        # "文生图-2.0": ("high_aes_general_v20", ''),
        "文生图-2.0Pro": ("high_aes_general_v20_L", ''),
        "文生图-2.1": ("high_aes_general_v21_L", ''),
        "角色特征保持": ("high_aes_ip_v20", ''),
        "人像融合": ('face_swap3_6', ''),  # 换脸图在前（最多三张），模板图在后（最多一张）
    }
    # inpainting涂抹消除,inpainting涂抹编辑,outpainting智能扩图
    request_body = {'req_key': style_mapping.get(style_name)[0],
                    'sub_req_key': style_mapping.get(style_name)[1],
                    'return_url': return_url  # 链接有效期为24小时
                    }
    if 'general' in request_body['req_key'] or prompt:
        request_body["prompt"] = prompt
        request_body["use_sr"] = True  # AIGC超分
        request_body["scale"] = 3.6  # 影响文本描述的程度
        request_body["seed"] = -1  # -1为不随机种子
        # request_body["use_pre_llm"] = True  #use_rephraser, prompt扩写, 对输入prompt进行扩写优化,辅助生成图片的场景下传True
    if image_urls and all(image_urls):
        request_body["image_urls"] = image_urls
    if image_data:  # 目标图片需小于 5 MB,小于4096*4096,支持JPG、JPEG、PNG格式,仅支持一张图,优先生效
        request_body["binary_data_base64"] = [base64.b64encode(image_data).decode("utf-8")]  # 输入图片base64数组
    if logo_info:
        request_body["logo_info"] = logo_info
        # {
        #     "add_logo": True,
        #     "position": 0,
        #     "language": 0,
        #     "opacity": 0.3,
        #     "logo_text_content": "这里是明水印内容"
        # }
    # 'CVSync2AsyncSubmitTask',JPCartoon
    headers, url = get_ark_signature(action='CVProcess', service='cv', host='visual.volcengineapi.com',
                                     region="cn-north-1", version="2022-08-31", http_method="POST", body=request_body,
                                     access_key_id=Config.VOLC_AK_ID_admin,
                                     secret_access_key=Config.VOLC_Secret_Key_admin,
                                     timenow=None)

    # print(headers,request_body)
    async with httpx.AsyncClient(timeout=Config.HTTP_TIMEOUT_SEC) as cx:
        response = await cx.post(url, json=request_body, headers=headers)
        if response.status_code != 200:
            return None, {"error": f"HTTP Error: {response.status_code},\n{response.text}"}
        response.raise_for_status()
        response_data = response.json()
        # print(response_data.keys())
        # {'code': 10000, 'data': {'1905703073': 1905703073, 'algorithm_base_resp': {'status_code': 0, 'status_message': 'Success'}, 'animeoutlineV4_16_strength_clip': 0.2, 'animeoutlineV4_16_strength_model': 0.2, 'apply_id_layer': '0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15', 'binary_data_base64': [], 'clip_skip': -2, 'cn_mode': 2, 'comfyui_cost': 4, 'controlnet_weight': 1, 'ddim_steps': 20, 'i2t_tag_text': '', 'id_weight': 0, 'image_urls': ['https://p26-aiop-sign.byteimg.com/tos-cn-i-vuqhorh59i/20241225114813D0211FF38607A612BD65-0~tplv-vuqhorh59i-image.image?rk3s=7f9e702d&x-expires=1735184899&x-signature=A1Jay2TWwSsmtwRjDGzK71gzXpg%3D'], 'long_resolution': 832, 'lora_map': {'animeoutlineV4_16': {'strength_clip': 0.2, 'strength_model': 0.2}, 'null': {'strength_clip': 0.7000000000000001, 'strength_model': 0.7000000000000001}}, 'null_strength_clip': 0.7000000000000001, 'null_strength_model': 0.7000000000000001, 'prompt': '(masterpiece), (((best quality))),light tone, sunny day, shinne,tyndall effect light， landscape in the movie of Suzume no Tojimari, daytime, meteor, aurora,', 'return_url': True, 'scale': 5, 'seed': -1, 'strength': 0.58, 'sub_prompts': ['(masterpiece), (((best quality))),light tone, sunny day, shinne,tyndall effect light， landscape in the movie of Suzume no Tojimari, daytime, meteor, aurora,'], 'sub_req_key': ''}, 'message': 'Success', 'request_id': '20241225114813D0211FF38607A612BD65', 'status': 10000, 'time_elapsed': '6.527672506s'}
        if response_data["status"] == 10000:
            image_base = response_data["data"].get("binary_data_base64", [])
            image_urls = response_data["data"].get("image_urls", [''])
            request_id = response_data["request_id"]
            if len(image_base) == 1:
                image_decode = base64.b64decode(image_base[0])
                if data_folder:
                    # 将解码后的数据转换为图片
                    file_path = f"{data_folder}/{request_id}.jpg"
                    img = Image.open(io.BytesIO(image_decode))
                    img.save(file_path)
                    return file_path, {"urls": image_urls, 'id': request_id}
                return image_decode, {"urls": image_urls, 'id': request_id}
            return None, {"urls": image_urls, 'id': request_id}
        return None, response_data


async def ark_drawing_picture(image_data, image_urls: List[str], whitening: float = 1.0, dermabrasion: float = 1.2,
                              logo_info=None, style_name='3d人偶', return_url=False):
    style_mapping = {
        # 头像风格（单人、男女均支持)
        "美漫风格": "img2img_photoverse_american_comics",
        "商务证件照": "img2img_photoverse_executive_ID_photo",
        "3d人偶": "img2img_photoverse_3d_weird",
        "赛博朋克": "img2img_photoverse_cyberpunk",
        # 胸像写真风格(单人、只支持女生)
        "古堡": "img2img_xiezhen_gubao",
        "芭比牛仔": "img2img_xiezhen_babi_niuzai",
        "浴袍风格": "img2img_xiezhen_bathrobe",
        "蝴蝶机械": "img2img_xiezhen_butterfly_machine",
        "职场证件照": "img2img_xiezhen_zhichangzhengjianzhao",
        "圣诞": "img2img_xiezhen_christmas",
        "美式甜点师": "img2img_xiezhen_dessert",
        "old_money": "img2img_xiezhen_old_money",
        "最美校园": "img2img_xiezhen_school"
    }

    request_body = {'req_key': style_mapping.get(style_name),
                    'return_url': return_url,  # 链接有效期为24小时
                    "beautify_info": {"whitening": whitening,  # 自定义美白参数，float类型，数值越大，效果越明显，未做参数范围校验，建议[0, 2]
                                      "dermabrasion": dermabrasion  # 自定义磨皮参数，float类型, 数值越大，效果越明显，未做参数范围校验，建议[0, 2]
                                      }
                    }
    if image_urls and all(image_urls):
        request_body["image_urls"] = image_urls
    if image_data:  # 输入图片base64数组,仅支持一张图,优先生效
        request_body["binary_data_base64"] = [base64.b64encode(image_data).decode("utf-8")]
    if logo_info:
        request_body["logo_info"] = logo_info

    headers, url = get_ark_signature(action='HighAesSmartDrawing', service='cv', host='visual.volcengineapi.com',
                                     region="cn-north-1", version="2022-08-31", http_method="POST", body=request_body,
                                     access_key_id=Config.VOLC_AK_ID_admin,
                                     secret_access_key=Config.VOLC_Secret_Key_admin,
                                     timenow=None)

    # print(headers,request_body)
    async with httpx.AsyncClient(timeout=Config.HTTP_TIMEOUT_SEC) as cx:
        response = await cx.post(url, json=request_body, headers=headers)
        if response.status_code != 200:
            return None, {"error": f"HTTP Error: {response.status_code},\n{response.text}"}
        response.raise_for_status()
        response_data = response.json()
        if response_data["status"] == 10000:
            image_base = response_data["data"].get("binary_data_base64", [])
            image_urls = response_data["data"].get("image_urls", [''])
            if len(image_base) == 1:
                image_decode = base64.b64decode(image_base[0])
                return image_decode, {"urls": image_urls, 'id': response_data["request_id"]}
            return None, {"urls": image_urls, 'id': response_data["request_id"]}
        return None, response_data


# https://help.aliyun.com/zh/viapi/developer-reference/api-overview?spm=a2c4g.11186623.help-menu-142958.d_4_3_1.13e65733U2m63s
# https://help.aliyun.com/zh/viapi/developer-reference/api-version?spm=a2c4g.11186623.help-menu-142958.d_4_3_0.290f6593LRs5Lt&scm=20140722.H_464194._.OR_help-T_cn~zh-V_1
async def ali_cartoon_picture(image_url, style_name='复古漫画'):
    style_mapping = {
        "复古漫画": '0',
        "3D童话": '1',
        "二次元": '2',
        "小清新": '3',
        "未来科技": '4',
        "国画古风": '5',
        "将军百战": '6',
        "炫彩卡通": '7',
        "清雅国风": '8'
    }
    # 图片大小不超过10MB。支持的图片类型：JPEG、PNG、JPG、BMP、WEBP。
    request_body = {'Index': style_mapping.get(style_name, '0'),
                    'ImageUrl': image_url, }
    # 视觉智能开放平台各服务支持的区域为华东2（上海）
    parameters, url = get_aliyun_access_token(service="imageenhan", region="cn-shanghai",
                                              action='GenerateCartoonizedImage', http_method="POST",
                                              body=request_body, version='2019-09-30')

    print(request_body)
    async with httpx.AsyncClient(timeout=Config.HTTP_TIMEOUT_SEC) as cx:
        try:
            response = await cx.post(url, json=request_body)  # , headers=headers
            print(response.text)
            if response.status_code != 200:
                return {"error": f"HTTP Error: {response.status_code},\n{response.text}"}

            response.raise_for_status()
            response_data = response.json()
            request_body = {'JobId': response_data["RequestId"]}

            parameters, url = get_aliyun_access_token(service="imageenhan", region="cn-shanghai",
                                                      action='GetAsyncJobResult', http_method="POST",
                                                      body=request_body, version='2019-09-30')
            response = await cx.post(url, json=request_body)  # , headers=headers
            # response_data["Data"].get("ResultUrl")
            print(response.text)
            # 图片链接非法，请检查图片链接是否可访问 ,图片链接地域不对，请参考https://help.aliyun.com/document_detail/155645.html - imageUrl is invalid region oss url
            if response.status_code != 200:
                return {"error": f"HTTP Error: {response.status_code},\n{response.text}"}
            response_data = response.json()
            if response_data["Data"]["Status"] == "PROCESS_SUCCESS":
                image_url = response_data["Data"]["Result"].get("ImageUrls")

                return {"urls": image_url, 'id': response_data["RequestId"]}
        except httpx.ConnectError as e:
            print(f"Connection error: {e},Request URL: {url},Request body: {request_body}")


# https://cloud.tencent.com/document/product/1668/88066
# https://cloud.tencent.com/document/product/1668/107799
async def tencent_drawing_picture(image_data, image_url: str = '', prompt: str = '', negative_prompt: str = '',
                                  style_name='日系动漫', return_url=False):
    # 单边分辨率小于5000且大于50，转成 Base64 字符串后小于 8MB，格式支持 jpg、jpeg、png、bmp、tiff、webp。
    style_mapping = {
        "水彩画": '104',
        "卡通插画": '107',
        "3D 卡通": '116',
        "日系动漫": '201',
        "唯美古风": '203',
        "2.5D 动画": '210',
        "木雕": '120',
        "黏土": '121',
        "清新日漫": '123',
        "小人书插画": '124',
        "国风工笔": '125',
        "玉石": '126',
        "瓷器": '127',
        "毛毡（亚洲版）": '135',
        "毛毡（欧美版）": '128',
        "美式复古": '129',
        "蒸汽朋克": '130',
        "赛博朋克": '131',
        "素描": '132',
        "莫奈花园": '133',
        "厚涂手绘": '134',
        "复古繁花": "flower",
        "芭比": "babi",
        "白领精英": "commerce",
        "婚纱日记": "wedding",
        "醉梦红尘": "gufeng",
        "暴富": "coin",
        "夏日水镜": "water",
        "复古港漫": "retro",
        "游乐场": "amusement",
        "宇航员": "astronaut",
        "休闲时刻": "cartoon",
        "回到童年": "star",
        "多巴胺": "dopamine",
        "心动初夏": "comic",
        "夏日沙滩": "beach"
    }

    style_type = style_mapping.get(style_name, '201')
    if style_type.isdigit():
        action = 'ImageToImage'  # 图像风格化
        payload = {'Strength': 0.6,  # 生成自由度(0, 1]
                   'EnhanceImage': 1,  # 画质增强开关
                   'RestoreFace': 1,  # 细节优化的面部数量上限，支持0 ~ 6，默认为0。
                   'RspImgType': 'url' if return_url else 'base64',
                   'Styles': [style_type],
                   'LogoAdd': 0
                   # 'ResultConfig': {"Resolution": "768:768"},  # origin
                   }
        if prompt:
            payload["Prompt"] = prompt
        if negative_prompt:
            payload["NegativePrompt"] = negative_prompt
    else:
        action = 'GenerateAvatar'  # 百变头像
        payload = {'RspImgType': 'url' if return_url else 'base64',
                   'Style': style_type,
                   'Type': 'human',  # pet,图像类型
                   'Filter': 1,  # 人像图的质量检测开关，默认开启，仅在人像模式下生效。
                   'LogoAdd': 0
                   }
    if image_data:
        payload["InputImage"] = base64.b64encode(image_data).decode("utf-8")
    if image_url:
        payload["InputUrl"] = image_url

    url = "https://aiart.tencentcloudapi.com"
    headers = get_tencent_signature(service="aiart", host="aiart.tencentcloudapi.com", body=payload,
                                    action=action, timestamp=int(time.time()), region="ap-shanghai",
                                    version='2022-12-29')

    async with httpx.AsyncClient(timeout=Config.HTTP_TIMEOUT_SEC) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        if response.status_code != 200:
            return None, {'error': f'{response.status_code},Request failed,Response content: {response.text}'}

        response_data = response.json()["Response"]
        if return_url:
            return None, {"urls": [response_data["ResultImage"]], 'id': response_data["RequestId"]}

        image_decode = base64.b64decode(response_data["ResultImage"])
        return image_decode, {"urls": [''], 'id': response_data["RequestId"]}


# https://cloud.tencent.com/document/product/1729/108738
async def tencent_generate_image(prompt: str = '', negative_prompt: str = '', style_name='不限定风格',
                                 return_url=False):
    style_mapping = {
        "默认": "000",
        "不限定风格": "000",
        "水墨画": "101",
        "概念艺术": "102",
        "油画1": "103",
        "油画2（梵高）": "118",
        "水彩画": "104",
        "像素画": "105",
        "厚涂风格": "106",
        "插图": "107",
        "剪纸风格": "108",
        "印象派1（莫奈）": "109",
        "印象派2": "119",
        "2.5D": "110",
        "古典肖像画": "111",
        "黑白素描画": "112",
        "赛博朋克": "113",
        "科幻风格": "114",
        "暗黑风格": "115",
        "3D": "116",
        "蒸汽波": "117",
        "日系动漫": "201",
        "怪兽风格": "202",
        "唯美古风": "203",
        "复古动漫": "204",
        "游戏卡通手绘": "301",
        "通用写实风格": "401"
    }
    payload = {'Style': style_mapping.get(style_name, '000'),
               'Prompt': prompt,
               'RspImgType': 'url' if return_url else 'base64',
               'LogoAdd': 0,
               "Resolution": "1024:1024",  # origin
               }

    if negative_prompt:
        payload["NegativePrompt"] = negative_prompt

    url = "https://hunyuan.tencentcloudapi.com"
    headers = get_tencent_signature(service="hunyuan", host="hunyuan.tencentcloudapi.com", body=payload,
                                    action='TextToImageLite', timestamp=int(time.time()), region="ap-guangzhou",
                                    version='2023-09-01')

    async with httpx.AsyncClient(timeout=Config.HTTP_TIMEOUT_SEC) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        if response.status_code != 200:
            return None, {'error': f'{response.status_code},Request failed,Response content: {response.text}'}

        response_data = response.json()["Response"]
        if return_url:
            return None, {"urls": [response_data["ResultImage"]], 'id': response_data["RequestId"]}

        image_decode = base64.b64decode(response_data["ResultImage"])
        return image_decode, {"urls": [''], 'id': response_data["RequestId"]}


async def siliconflow_generate_image(prompt: str = '', negative_prompt: str = '', model_name='siliconflow', model_id=0):
    model_info, name = find_ai_model(model_name, model_id, 'image')

    url = "https://api.siliconflow.cn/v1/images/generations"

    payload = {
        "model": name,
        "prompt": prompt,
        "seed": random.randint(1, 9999999998),  # Required seed range:0 < x < 9999999999
        'prompt_enhancement': True,
    }
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt

    headers = {
        "Authorization": f"Bearer {Config.Silicon_Service_Key}",
        "Content-Type": "application/json"
    }
    async with httpx.AsyncClient(timeout=Config.HTTP_TIMEOUT_SEC) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        if response.status_code != 200:
            return None, {'error': f'{response.status_code},Request failed,Response content: {response.text}'}

        response_data = response.json()

        return None, {"urls": [i['url'] for i in response_data["images"]], 'id': response_data["seed"]}


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


async def baidu_nlp(nlp_type='ecnet', **kwargs):  # text': text
    '''
    address:地址识别
    sentiment_classify:情感倾向分析
    emotion:对话情绪识别
    entity_analysis:实体分析,text,mention
    simnet:短文本相似度,text_1,text_2,
    ecnet,text_correction:文本纠错,搜索引擎、语音识别、内容审查等功能
    txt_keywords_extraction:关键词提取,text[],num
    txt_monet:文本信息提取,content_list[{content,query_lis[{query}]}]
    sentiment_classify:
    depparser:依存句法分析,利用句子中词与词之间的依存关系来表示词语的句法结构信息
    lexer:词法分析,提供分词、词性标注、专名识别三大功能
    keyword:文章标签,title,content
    topic:文章分类,title,content
    news_summary:新闻摘要,title,content,max_summary_len
    titlepredictor,文章标题生成,doc
    '''
    # https://aip.baidubce.com/rpc/2.0/ernievilg/v1/txt2imgv2
    url = f'https://aip.baidubce.com/rpc/2.0/nlp/v1/{nlp_type}'
    access_token = get_baidu_access_token(Config.BAIDU_nlp_API_Key, Config.BAIDU_nlp_Secret_Key)
    url = build_url(url, access_token, charset='UTF-8')  # f"{url}&access_token={access_token}"
    headers = {
        'Content-Type': 'application/json',  # 'application/x-www-form-urlencoded'
        # 'host': 'aip.baidubce.com',
        # 'authorization': 'bce-auth',
        # 'x-bce-date': '2015-03-24T13: 02:00Z',
        # 'x-bce-request-id':'',
    }
    body = {**kwargs}

    async with httpx.AsyncClient(timeout=Config.HTTP_TIMEOUT_SEC) as cx:
        response = await cx.post(url, headers=headers, json=body)
        response.raise_for_status()
        data = response.json()  # .get('items')
        # 遍历键列表，返回第一个找到的键对应的值
        for key in ['items', 'item', 'results', 'results_list', 'error_msg']:
            if key in data:
                if key == 'error_msg':
                    print(response.text)
                return data[key]
        return data


# https://ai.baidu.com/ai-doc/OCR/Ek3h7y961,  https://aip.baidubce.com/rest/2.0/solution/v1/iocr/recognise"
# https://console.bce.baidu.com/ai/#/ai/ocr/overview/index
async def baidu_ocr_recognise(image_data, image_url, ocr_type='accurate_basic'):
    '''
    general:通用文字识别(含位置)
    accurate:通用文字识别(高进度含位置)
    accurate_basic:通用文字识别（高进度）
    general_basic:通用文字识别
    doc_analysis_office:办公文档识别
    idcard:身份证识别
    table:表格文字识别
    numbers:数字识别
    qrcode:二维码识别
    account_opening:开户许可证识别
    handwriting:手写文字识别
    webimage:
    '''
    headers = {
        'Content-Type': "application/x-www-form-urlencoded",
        'Accept': 'application/json',
        # 'charset': "utf-8",
    }
    url = f"https://aip.baidubce.com/rest/2.0/ocr/v1/{ocr_type}"
    access_token = get_baidu_access_token(Config.BAIDU_ocr_API_Key, Config.BAIDU_ocr_Secret_Key)
    params = {
        "access_token": access_token,
        "language_type": 'CHN_ENG',
    }
    if image_url:
        params["url"] = image_url
    if image_data:
        params["image"] = base64.b64encode(image_data).decode("utf-8")
    # 将图像数据编码为base64
    # image_b64 = base64.b64encode(image_data).decode().replace("\r", "")
    # if template_sign:
    #     params["templateSign"] = template_sign
    # if classifier_id:
    #     params["classifierId"] = classifier_id
    # # 请求模板的bodys
    # recognise_bodys = "access_token=" + access_token + "&templateSign=" + template_sign + "&image=" + quote(image_b64.encode("utf8"))
    # # 请求分类器的bodys
    # classifier_bodys = "access_token=" + access_token + "&classifierId=" + classifier_id + "&image=" + quote(image_b64.encode("utf8"))
    # request_body = "&".join(f"{key}={value}" for key, value in params.items())
    async with httpx.AsyncClient(timeout=Config.HTTP_TIMEOUT_SEC) as cx:
        response = await cx.post(url, headers=headers, data=params)
        response.raise_for_status()
        data = response.json()
        for key in ['result', 'results', 'error_msg']:
            if key in data:
                return data[key]
        return data


# https://cloud.tencent.com/document/product/866/36210
async def tencent_ocr_recognise(image_data, image_url, ocr_type='GeneralBasicOCR'):
    '''
    GeneralBasicOCR:通用印刷体识别,TextDetections
    RecognizeTableDDSNOCR: 表格识别,TableDetections
    RecognizeGeneralTextImageWarn:证件有效性检测告警
    GeneralAccurateOCR:通用印刷体识别（高精度版）
    VatInvoiceOCR:增值税发票识别
    VatInvoiceVerifyNew:增值税发票核验
    ImageEnhancement:文本图像增强,包括切边增强、图像矫正、阴影去除、摩尔纹去除等；
    QrcodeOCR:条形码和二维码的识别
    SmartStructuralOCRV2:智能结构化识别,智能提取各类证照、票据、表单、合同等结构化场景的key:value字段信息
    '''
    url = 'https://ocr.tencentcloudapi.com'
    host = url.split("//")[-1]
    payload = {
        # 'Action': ocr_type,
        # 'Version': '2018-11-19'
        # 'Region': 'ap-shanghai',
        # 'ImageBase64': '',
        # 'ImageUrl': image_url,
    }
    if image_url:
        payload['ImageUrl'] = image_url
    else:
        if isinstance(image_data, bytes):
            payload['ImageBase64'] = base64.b64encode(image_data).decode("utf-8")
        else:
            payload['ImageBase64'] = base64.b64encode(image_data)

    headers = get_tencent_signature('ocr', host, body=payload, action=ocr_type,
                                    secret_id=Config.TENCENT_SecretId, secret_key=Config.TENCENT_Secret_Key,
                                    version='2018-11-19')

    # payload = convert_keys_to_pascal_case(params)

    async with httpx.AsyncClient(timeout=Config.HTTP_TIMEOUT_SEC) as cx:
        response = await cx.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["Response"]  # "TextDetections"


# https://help.aliyun.com/zh/ocr/developer-reference/api-ocr-api-2021-07-07-dir/?spm=a2c4g.11186623.help-menu-252763.d_2_2_4.3aba47bauq0U2j
async def ali_ocr_recognise(image_data, image_url, ocr_type='Advanced'):
    # accurate,general_basic,webimage
    # RecognizeAllText
    url = 'https://ocr-api.cn-hangzhou.aliyuncs.com'
    token, _ = get_aliyun_access_token(service="ocr-api", region="cn-hangzhou", access_key_id=Config.ALIYUN_AK_ID,
                                       access_key_secret=Config.ALIYUN_Secret_Key, version='2021-07-07')
    headers = {
        'Content-Type': "application/x-www-form-urlencoded",
        'charset': "utf-8"
    }
    try:
        # 将图像数据编码为base64
        # image_b64 = base64.b64encode(image_data).decode().replace("\r", "")
        params = {
            "access_token": token,
            "language_type": 'CHN_ENG',

            'Type': ocr_type,  # Advanced,HandWriting,General,Table,GeneralStructure
            'PageNo': 1,
            'OutputRow': False,
            'OutputParagraph': False,
            'OutputKVExcel': False,
            'OutputTableHtml': False,
        }
        if image_data:
            params["body"] = base64.b64encode(image_data)  # quote(image_b64.encode("utf8"))
        if url:
            params["Url"] = image_url

        # if template_sign:
        #     params["templateSign"] = template_sign
        # if classifier_id:
        #     params["classifierId"] = classifier_id
        # # 请求模板的bodys
        # recognise_bodys = "access_token=" + access_token + "&templateSign=" + template_sign + "&image=" + quote(image_b64.encode("utf8"))
        # # 请求分类器的bodys
        # classifier_bodys = "access_token=" + access_token + "&classifierId=" + classifier_id + "&image=" + quote(image_b64.encode("utf8"))
        # request_body = "&".join(f"{key}={value}" for key, value in params.items())
        response = requests.post(url, data=params, headers=headers, timeout=Config.HTTP_TIMEOUT_SEC)
        response.raise_for_status()

        return response.json()
    except:
        pass


async def ali_nlp(text):
    url = 'alinlp.cn-hangzhou.aliyuncs.com'
    token, _ = get_aliyun_access_token(service="alinlp", region="cn-hangzhou", access_key_id=Config.ALIYUN_AK_ID,
                                       access_key_secret=Config.ALIYUN_Secret_Key)
    if not token:
        print("No permission!")

    headers = {
        "Authorization": f"Bearer {Config.ALIYUN_AK_ID}",
        'Content-Type': "application/x-www-form-urlencoded",
        "X-NLS-Token": token,

    }
    params = {
        "appkey": Config.ALIYUN_nls_AppId,
        'ServiceCode': 'alinlp',
        'Action': 'GetWeChGeneral',
        'Text': text,
        'TokenizerId': 'GENERAL_CHN',
    }


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
    token, _ = get_aliyun_access_token(service="nls-meta", region="cn-shanghai", action='CreateToken',
                                       http_method="GET",
                                       access_key_id=Config.ALIYUN_AK_ID, access_key_secret=Config.ALIYUN_Secret_Key)
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
    async with httpx.AsyncClient(timeout=Config.HTTP_TIMEOUT_SEC) as client:
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

    async with httpx.AsyncClient(timeout=Config.HTTP_TIMEOUT_SEC) as client:
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
            async with httpx.AsyncClient(timeout=Config.HTTP_TIMEOUT_SEC) as client:
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
    audio_data = synthesizer.call(sentences)  # sample_rate=48000
    return audio_data, synthesizer.get_last_request_id()

    # SpeechSynthesizer.call(model='sambert-zhichu-v1',
    #                        text='今天天气怎么样',
    #                        sample_rate=48000,
    #                        format='pcm',
    #                        callback=callback)
    # if result.get_audio_data() is not None:


# https://bailian.console.aliyun.com/?spm=5176.28197581.0.0.2e2d29a4n0Mukq#/model-market/detail/wanx-v1?tabKey=sdk
def dashscope_image_call(prompt: str, negative_prompt: str = '', image_url: str = '', style_name="默认",
                         model_name="wanx-v1", data_folder=None):
    style_mapping = {
        "默认": "<auto>",
        "摄影": "<photography>",
        "人像写真": "<portrait>",
        "3D卡通": "<3d cartoon>",
        "动画": "<anime>",
        "油画": "<oil painting>",
        "水彩": "<watercolor>",
        "素描": "<sketch>",
        "中国画": "<chinese painting>",
        "扁平插画": "<flat illustration>"
    }
    style = style_mapping.get(style_name, "<auto>")
    rsp = dashscope.ImageSynthesis.call(model=model_name,  # "stable-diffusion-3.5-large"
                                        api_key=Config.Bailian_Service_Key,
                                        prompt=prompt, negative_prompt=negative_prompt, ref_img=image_url,
                                        n=1, size='1024*1024', style=style)

    # ref_strength：控制输出图像与参考图（垫图）的相似度。取值范围为[0.0, 1.0]。取值越大，代表生成的图像与参考图越相似。
    # ref_mode：基于参考图（垫图）生成图像的方式。取值有：repaint代表参考内容，为默认值；refonly代表参考风格。
    if rsp.status_code == HTTPStatus.OK:
        print(rsp)
        # 保存图片到当前文件夹
        image_urls = [result.url for result in rsp.output.results]
        if data_folder:
            image_path = []
            for result in rsp.output.results:
                file_name = PurePosixPath(unquote(urlparse(result.url).path)).parts[-1]
                file_path = f'{data_folder}/%s' % file_name
                with open(file_path, 'wb+') as f:
                    f.write(requests.get(result.url).content)
                image_path.append(file_path)

            return image_path, {"urls": image_urls, 'id': rsp.request_id}
        return requests.get(image_urls[0]).content, {"urls": image_urls, 'id': rsp.request_id}
    return None, {"error": 'Failed, status_code: %s, code: %s, message: %s' % (rsp.status_code, rsp.code, rsp.message)}


# https://help.aliyun.com/zh/model-studio/user-guide/cosplay-anime-character-generation?spm=0.0.0.i1
# https://help.aliyun.com/zh/model-studio/developer-reference/portrait-style-redraw-api-reference?spm=a2c4g.11186623.help-menu-2400256.d_3_3_2_1.3e2f56e5BtF0ok
async def wanx_image_generation(image_urls, style_name="复古漫画",
                                api_key=Config.DashScope_Service_Key, max_retries=20):
    # JPEG，PNG，JPG，BMP，WEB,不超过10M,不小于256*256，不超过5760*3240, 长宽比不超过2:1
    style_mapping = {
        "参考上传图像风格": -1,
        "复古漫画": 0,
        "3D童话": 1,
        "二次元": 2,
        "小清新": 3,
        "未来科技": 4,
        "国画古风": 5,
        "将军百战": 6,
        "炫彩卡通": 7,
        "清雅国风": 8,
        "喜迎新年": 9
    }
    if style_name == 'Cosplay动漫人物':
        model_name = "wanx-style-cosplay-v1"
        input_params = {
            "model_index": 1,
            "face_image_url": image_urls[0],
            "template_image_url": image_urls[1],
        }
    elif len(image_urls) > 1:
        model_name = "wanx-style-repaint-v1"
        input_params = {
            "style_index": -1,
            "image_url": image_urls[0],
            'style_ref_url': image_urls[1]
        }
    else:  # '人像风格重绘'
        model_name = "wanx-style-repaint-v1"
        input_params = {
            "style_index": style_mapping.get(style_name, 0),
            "image_url": image_urls[0],
        }

    url = 'https://dashscope.aliyuncs.com/api/v1/services/aigc/image-generation/generation'
    headers = {
        'Content-Type': 'application/json',
        "Authorization": f'Bearer {api_key}',
        'X-DashScope-Async': 'enable'  # 使用异步方式提交作业
    }
    task_headers = {"Authorization": f'Bearer {api_key}'}
    body = {
        "model": model_name,
        "input": input_params,
        # "parameters": {
        #     "style": "<auto>",
        #     "size": "1024*1024",
        #     "n": 1
        # }
    }
    async with httpx.AsyncClient(timeout=Config.HTTP_TIMEOUT_SEC) as cx:
        response = await cx.post(url, headers=headers, json=body)
        response.raise_for_status()
        data = response.json()
        task_id = data["output"]["task_id"]
        task_status = data["output"]["task_status"]
        retries = 0
        # 轮询任务进度
        await asyncio.sleep(3)
        while task_id is not None and retries < max_retries:
            task_url = f'https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}'
            task_response = await cx.get(task_url, headers=task_headers)
            resp = task_response.json()
            task_status = resp["output"]["task_status"]
            # "task_status":"PENDING""RUNNING","SUCCEEDED"->"results", "FAILED"->"message"
            if task_status == 'SUCCEEDED':
                urls = [item['url'] for item in resp['output'].get('results', []) if 'url' in item]
                result = {"urls": urls or [resp['output'].get('result_url')], 'id': task_id}
                if urls:
                    image_response = await cx.get(urls[0])
                    return image_response.content, result

                return None, result

            if task_status == "FAILED":
                print(resp['output']['message'])
                break

            await asyncio.sleep(3)
            retries += 1

        return None, {"urls": [], 'id': task_id, 'status': task_status,
                      'error': "Task did not succeed within the maximum retry limit."}


def dashscope_file_upload(messages, file_path='.pdf', api_key=Config.DashScope_Service_Key):
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
        file_response = requests.post(url, headers=headers, files=files, timeout=Config.HTTP_TIMEOUT_SEC)
        file_response.raise_for_status()
        file_object = file_response.json()
        file_id = file_object.get('id', 'unknown_id')  # 从响应中获取文件ID

        messages.append({"role": "system", "content": f"fileid://{file_id}"})
        return file_object, file_id

    except Exception as e:
        return {"error": str(e)}, None

    finally:
        files['file'][1].close()


def upload_file_to_oss(bucket, file_obj, object_name, expires: int = 604800):
    """
      上传文件到 OSS 支持 `io` 对象。
      :param bucket: OSS bucket 实例
      :param file_obj: 文件对象，可以是 `io.BytesIO` 或 `io.BufferedReader`
      :param object_name: OSS 中的对象名
      :param expires: 签名有效期，默认一周（秒）
    """
    file_obj.seek(0, os.SEEK_END)
    total_size = file_obj.tell()  # os.path.getsize(file_path)
    file_obj.seek(0)
    if total_size > 1024 * 1024 * 16:
        part_size = oss2.determine_part_size(total_size, preferred_size=128 * 1024)
        upload_id = bucket.init_multipart_upload(object_name).upload_id
        parts = []
        part_number = 1
        offset = 0
        while offset < total_size:
            size_to_upload = min(part_size, total_size - offset)
            result = bucket.upload_part(object_name, upload_id, part_number,
                                        oss2.SizedFileAdapter(file_obj, size_to_upload))
            parts.append(oss2.models.PartInfo(part_number, result.etag, size=size_to_upload, part_crc=result.crc))
            offset += size_to_upload
            part_number += 1

        # 完成分片上传
        bucket.complete_multipart_upload(object_name, upload_id, parts)
    else:
        # OSS 上的存储路径, 本地图片路径
        bucket.put_object(object_name, file_obj)
        # bucket.put_object_from_file(object_name, str(file_path))

    if 0 < expires <= 604800:  # 如果签名signed_URL
        url = bucket.sign_url("GET", object_name, expires=expires)
    else:  # 使用加速域名
        url = f"{Config.ALIYUN_Bucket_Domain}/{object_name}"
        # bucket.bucket_name
    # 获取文件对象
    # result = bucket.get_object(object_name)
    # result.read()获取文件的二进制内容,result.headers元数据（头部信息）
    return url


# 获取文件列表
def list_files(bucket, prefix='upload/', max_keys=100, max_pages=1):
    """
    列出 OSS 中的文件。
    :param bucket: oss2.Bucket 实例
    :param prefix: 文件名前缀，用于筛选
    :param max_keys: 每次返回的最大数量
    :return: 文件名列表
    """
    file_list = []
    if max_pages <= 1:
        for obj in oss2.ObjectIterator(bucket, prefix=prefix, max_keys=max_keys):
            file_list.append(obj.key)
    else:
        i = 0
        next_marker = ''
        while i < max_pages:
            result = bucket.list_objects(prefix=prefix, max_keys=max_keys, marker=next_marker)
            for obj in result.object_list:
                file_list.append(obj.key)
            if not result.is_truncated:  # 如果没有更多数据，退出循环
                break
            next_marker = result.next_marker
            i += 1

    return file_list


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
    '''
    同步下载的流式方法
    如果目标是保存到文件，直接使用 content（无需解码）。（如图片、音频、视频、PDF 等）
    如果目标是处理和解析文本数据，且确定编码正确，使用 text。（如 HTML、JSON）
    '''
    with requests.get(url, stream=True, timeout=Config.HTTP_TIMEOUT_SEC) as response:
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


def upload_by_requests(url: str, file_path, file_key='snapshot'):
    with open(file_path, "rb") as f:
        files = {file_key: f}
        response = requests.post(url, files=files)
    return response.json()


async def download_file(url: str, dest_folder: Path = None, chunk_size=4096,
                        in_decode=False, in_session=False, retries=3, delay=3):
    """
    下载URL中的文件到目标文件夹
    :param url: 下载链接
    :param dest_folder: 保存文件的路径
    :param chunk_size: 每次读取的字节大小（默认4096字节）
    :param in_decode: 是否解码为字符串
    :param in_session: 是否使用 session（长连接优化）
    :param retries: 下载失败后的重试次数
    :param delay: 重试之间的等待时间（秒）
    """
    # filename = url.split("/")[-1]  # 提取文件名
    file_name = unquote(url.split("/")[-1].split("?")[0])
    save_path = None
    # print(file_name)
    if dest_folder:
        save_path = dest_folder / file_name  # 提取文件名

    attempt = 0
    while attempt < retries:
        try:
            if in_session:  # aiohttp长连接优化，适合发送多个请求或需要更好的连接复用,维护多个请求之间的连接
                return await download_by_aiohttp(url, save_path, chunk_size, in_decode), file_name
            else:  # httpx少量请求场景,适合简单的、单个请求场景
                return await download_by_httpx(url, save_path, chunk_size, in_decode), file_name

        except (httpx.RequestError, aiohttp.ClientError, asyncio.TimeoutError) as e:
            print(f"Download attempt {attempt + 1} failed: {e}")
            attempt += 1
            await asyncio.sleep(delay)  # 等待重试
        except httpx.HTTPStatusError as exc:
            print(f"Failed to download {url},HTTP error occurred: {exc.response.status_code} - {exc.response.text}")
        except TypeError:
            return download_by_requests(url, save_path, chunk_size, in_decode), file_name
        except Exception as e:
            print(f"Error: {e}, downloading url: {url} file: {file_name}")
            break

    return None, None


# https://www.weatherapi.com/api-explorer.aspx#forecast
def get_weather(city: str, days: int = 0, date: str = None):
    # 使用 WeatherAPI 的 API 来获取天气信息
    api_key = Config.Weather_Service_Key
    base_url = "http://api.weatherapi.com/v1/current.json"
    city = convert_to_pinyin(city)
    params = {
        'key': api_key,
        'q': city,
        # Pass US Zipcode, UK Postcode, Canada Postalcode, IP address, Latitude/Longitude (decimal degree) or city name.
        'aqi': 'no'  # Get air quality data 空气质量数据
    }
    # Number of days of weather forecast. Value ranges from 1 to 10
    if days > 0:
        params['days'] = days
        params['alerts'] = 'no'
    elif date:
        # Date on or after 1st Jan, 2010 in yyyy-MM-dd format
        # Date between 14 days and 300 days from today in the future in yyyy-MM-dd format
        params['dt'] = date

    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        weather = data['current']['condition']['text']
        temperature = data['current']['temp_c']
        return f"The weather in {city} is {weather} with a temperature of {temperature}°C."
    else:
        return f"Could not retrieve weather information for {city}."


def send_to_wechat(user_name: str, context: str = None, link: str = None, object_name: str = None):
    url = f"{Config.WECHAT_URL}/sendToChat"
    headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
    body = {'user': user_name, 'context': context, 'url': link,
            'object_name': object_name, 'file_type': get_file_type_wx(object_name)}

    try:
        with httpx.Client(timeout=(10, Config.HTTP_TIMEOUT_SEC)) as client:
            response = client.post(url, json=body, headers=headers)
            response.raise_for_status()
        return response.json()

    except Exception as e:
        print(datetime.now(), body)
        print(f"Error occurred while sending message: {e}")

    return None


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


    # asyncio.run(test())
    asyncio.run(baidu_nlp("ecnet", text="百度是一家人工只能公司"))

    # asyncio.run(tencent_translate('tencent translate is ok', 'en', 'cn'))
    # from lagent import list_tools, get_tool
    #
    # list_tools()
