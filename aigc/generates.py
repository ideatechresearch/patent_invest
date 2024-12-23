import httpx
import asyncio
import oss2
from pathlib import Path
from typing import List, Dict, Any, Union, Tuple, Callable, Optional
import random, time, io, os, pickle
from PIL import Image
from openai import OpenAI, Completion
# import qianfan
import dashscope
from dashscope.audio.tts import ResultCallback, SpeechSynthesizer
from dashscope.audio.asr import Recognition, Transcription
from qdrant_client.models import Filter, FieldCondition, IsEmptyCondition, HasIdCondition, MatchValue
from fastapi import HTTPException
import numpy as np
from config import *
from utils import *
from ai_tools import *
from ai_agents import *

AI_Client = {}


def init_ai_clients(api_keys=API_KEYS):
    SUPPORTED_MODELS = {'moonshot', 'glm', 'qwen', 'hunyuan', 'silicon', 'doubao', 'baichuan'}
    for model in AI_Models:
        model_name = model.get('name')
        api_key = api_keys.get(model_name)
        if api_key:
            model['api_key'] = api_key
            if model_name not in AI_Client and model_name in SUPPORTED_MODELS:
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


async def ai_tool_response(messages, tools=None, model_name='moonshot', model_id=1,
                           top_p=0.95, temperature=0.01, **kwargs):
    """
      调用 AI 模型接口，使用提供的工具集和对话消息，返回模型的响应。qwen
        :return: 模型响应的消息对象
    """
    model_info, name = find_ai_model(model_name, model_id)
    client = AI_Client.get(model_info['name'], None)

    if client:
        try:
            completion = await asyncio.to_thread(
                client.chat.completions.create,
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

    function_registry = {
        "get_times_shift": get_times_shift,
        "get_weather": get_weather,
        "web_search": web_search,
        "date_range_calculator": date_range_calculator,
        'get_day_range': get_day_range,
        "get_week_range": get_week_range,
        "get_month_range": get_month_range,
        "get_quarter_range": get_quarter_range,
        "get_year_range": get_year_range,
        "get_half_year_range": get_half_year_range,
        "search_bmap_location": search_bmap_location,
        "auto_translate": auto_translate,
        # 添加更多可调用函数
    }
    for tool in tool_calls:
        func_name = tool.function.name  # function_name
        func_args = tool.function.arguments  # function_args = json.loads(tool_call.function.arguments)
        func_reg = function_registry.get(func_name)  # 从注册表中获取函数
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

        if client:
            file_object = client.files.create(file=file_path_obj, purpose="file-extract")
            if model_info['name'] == 'qwen':
                messages.append({"role": "system", "content": f"fileid://{file_object.id}", })
            elif model_info['name'] == 'moonshot':
                file_content = client.files.content(file_id=file_object.id).text
                messages.append({"role": "system", "content": file_content, })
        else:
            dashscope_file_upload(messages, file_path=str(file_path_obj))

    if question:
        messages.append({"role": "user", "content": question})
        # print(messages)
        completion = client.chat.completions.create(model=name, messages=messages, **kwargs)
        bot_response = completion.choices[0].message.content  # completion.model_dump_json()
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
    """
    if not model_name:
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
            tasks = [asyncio.to_thread(
                client.embeddings.create,
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
            completion = await asyncio.to_thread(client.embeddings.create,
                                                 model=name, input=inputs,
                                                 encoding_format="float")

            embeddings = [item.embedding for item in completion.data]

        return embeddings
        # data = json.loads(completion.model_dump_json()

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
        async with httpx.AsyncClient() as cx:
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
        "return_documents": True,
    }
    payload.update(kwargs)
    async with httpx.AsyncClient() as cx:
        response = await cx.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            results = response.json().get('results')
            matches = [(match.get("document")["text"], match["relevance_score"], match["index"]) for match in results]
            return matches
        else:
            print(response.text)
    return []


# 生成:conversation or summary
async def ai_generate(prompt: str, user_request: str = '', suffix: str = None, stream=False, temperature=0.7,
                      max_tokens=4096, model_name='silicon', model_id=0, **kwargs):
    '''
    Completions足以解决几乎任何语言处理任务，包括内容生成、摘要、语义搜索、主题标记、情感分析等等。
    需要注意的一点限制是，对于大多数模型，单个API请求只能在提示和完成之间处理最多4096个标记。
    '''
    model_info, name = find_ai_model(model_name, model_id, "generation")
    if not name:
        return await ai_chat(model_info=None, messages=None, user_request=user_request, system=prompt,
                             temperature=temperature, max_tokens=max_tokens, top_p=0.8, model_name=model_name,
                             model_id=model_id, **kwargs)

    if user_request:
        prompt += '\n\n' + user_request

    if model_info['name'] == 'qwen':
        response = dashscope.Generation.call(model=name, prompt=prompt)
        return response.output.text

    client = AI_Client.get(model_info['name'], None)
    response = client.completions.create(
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
            for chunk in response:
                yield chunk.choices[0].text

        return stream_data()

    return response.choices[0].text.strip()


def agent_func_calls(agent):
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


async def retrieved_reference(user_request: str, keywords: List[Union[str, Tuple[str, Any]]] = None,
                              tool_calls: List[Callable[[...], Any]] = None, **kwargs):
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
                           max_tokens: int = 1024, model_name='moonshot', model_id=0,
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
        if model_type != 'baidu' and system:
            messages = [{"role": "system", "content": system}]
        messages.append({'role': 'user', 'content': user_request})

    refer = await retrieved_reference(user_request, keywords, tool_calls, **kwargs)
    if refer:
        formatted_refer = '\n'.join(map(str, refer))
        messages[-1]['content'] = (f'以下是相关参考资料:\n{formatted_refer}\n'
                                   f'请结合以上内容或根据上下文，针对下面的问题进行解答：\n{user_request}')

    if images:  # 图片内容理解
        messages[-1]['content'] = [{"type": "text", "text": user_request}]  # text-prompt 请详细描述一下这几张图片。这是哪里？
        messages[-1]['content'] += [{"type": "image_url", "image_url": {"url": image}} for image in images]

    payload = {
        "model": name,  # 默认选择第一个模型
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        # "top_k": 50,
        "max_tokens": max_tokens,
        # extra_body = {"prefix": "```python\n", "suffix":"后缀内容"} 希望的前缀内容,基于用户提供的前缀信息来补全其余的内容
        # response_format={"type": "json_object"}
        # "tools":retrieval、web_search、function
    }
    if model_type == 'baidu':
        payload['system'] = system

    # print(payload)
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

    return await ai_chat_post(model_info, payload)


async def ai_chat_post(model_info, payload):
    # 通过 requests 库直接发起 HTTP POST 请求
    model_type = model_info['type']
    url = model_info['url']
    api_key = model_info['api_key']
    headers = {'Content-Type': 'application/json', }
    if api_key:
        if isinstance(api_key, list):
            idx = model_info['model'].index(payload["model"])
            api_key = model_info['api_key'][idx]

        headers = {'Content-Type': 'application/json',
                   "Authorization": f'Bearer {api_key}'}
    if model_type == 'baidu':
        url = build_url(url, get_baidu_access_token(Config.BAIDU_qianfan_API_Key, Config.BAIDU_qianfan_Secret_Key))
        payload["disable_search"] = False
        # payload['enable_system_memory'] = False
        # payload["enable_citation"]= False
        # payload["user_id"]=
        # payload['system'] = system
    if model_type == 'tencent':
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
    # payload["stream"]= {"include_usage": True}        # 可选，配置以后会在流式输出的最后一行展示token使用信息
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
        if isinstance(api_key, list):
            idx = model_info['model'].index(payload["model"])
            api_key = model_info['api_key'][idx]
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
        headers = get_tencent_signature(service, host, payload, action='ChatCompletions',
                                        secret_id=Config.TENCENT_SecretId,
                                        secret_key=Config.TENCENT_Secret_Key)
        headers['X-TC-Version'] = '2023-09-01'

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
    response = requests.get(f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&format=json",
                            timeout=Config.HTTP_TIMEOUT_SEC)  # proxies=
    search_results = response.json().get('query', {}).get('search', [])
    if search_results:
        page_id = search_results[0]['pageid']
        page_response = requests.get(
            f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&pageids={page_id}&format=json")
        page_data = page_response.json()['query']['pages'][str(page_id)]
        return page_data.get('extract', 'No extract found.')
    return "No information found."


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
    async with httpx.AsyncClient() as cx:
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
    response = requests.post(url, headers=headers, json=payload)
    data = response.json().get('data')
    return [emb.get('embedding') for emb in data]


async def most_similar_embeddings(query, collection_name, client, topn=10, score_threshold=0.0,
                                  match: List[Any] = [], not_match: List[Any] = [], query_vector=[],
                                  embeddings_calls: Callable[[...], Any] = lambda x: [], **kwargs):
    try:
        if not query_vector:
            query_vector = await embeddings_calls(query, **kwargs)
            if not query_vector:
                return []

        query_filter = Filter(must=match, must_not=not_match)
        search_hit = await client.search(collection_name=collection_name,
                                         query_vector=query_vector,  # tolist()
                                         query_filter=query_filter,
                                         limit=topn,
                                         score_threshold=score_threshold, )
        return [(p.payload, p.score) for p in search_hit]
    except Exception as e:
        print('Error:', e)
        return []


def cosine_sim(A, B):
    dot_product = np.dot(A, B)
    similarity = dot_product / (np.linalg.norm(A) * np.linalg.norm(B))
    return similarity


def cosine_similarity_np(ndarr1, ndarr2):
    denominator = np.outer(np.linalg.norm(ndarr1, axis=1), np.linalg.norm(ndarr2, axis=1))
    dot_product = np.dot(ndarr1, ndarr2.T)
    with np.errstate(divide='ignore', invalid='ignore'):
        similarity = np.where(denominator != 0, dot_product / denominator, 0)
    return similarity


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

    async with httpx.AsyncClient() as client:
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

    async with httpx.AsyncClient() as client:
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

    async with httpx.AsyncClient() as client:
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
    async with httpx.AsyncClient() as client:
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
    headers = get_tencent_signature(service="tmt", host="tmt.tencentcloudapi.com", params=payload,
                                    action='TextTranslate',
                                    secret_id=Config.TENCENT_SecretId, secret_key=Config.TENCENT_Secret_Key,
                                    timestamp=int(time.time()), version='2018-03-21')

    async with httpx.AsyncClient() as client:
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

    headers = get_xfyun_authorization(api_key=Config.XF_API_Key, api_secret=Config.XF_Secret_Key,
                                      host="itrans.xf-yun.com", path="/v1/its", method='POST')
    url = 'https://itrans.xf-yun.com/v1/its'  # f"https://{host}{path}?"+ urlencode(headers)

    # 异步发送请求
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=request_data, headers=headers)
        if response.status_code == 200:
            response_data = await response.json()

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
def xunfei_ppt_create(text: str, templateid: str = "20240718489569D", appid: str = Config.XF_AppID,
                      api_secret: str = Config.XF_Secret_Key):
    from requests_toolbelt.multipart.encoder import MultipartEncoder

    url = 'https://zwapi.xfyun.cn/api/ppt/v2/create'
    timestamp = int(time.time())
    signature = get_xfyun_signature(appid, api_secret, timestamp)
    formData = MultipartEncoder(
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

    print(formData)
    headers = {
        "appId": appid,
        "timestamp": str(timestamp),
        "signature": signature,
        "Content-Type": formData.content_type
    }

    response = requests.request(method="POST", url=url, data=formData, headers=headers).text
    resp = json.loads(response)
    if (0 != resp['code']):
        print('创建PPT任务失败,生成PPT返回结果：', response)
        return None

    task_id = resp['data']['sid']
    PPTurl = ''
    # 轮询任务进度
    time.sleep(5)
    while (None != task_id):
        url = f"https://zwapi.xfyun.cn/api/ppt/v2/progress?sid={task_id}"
        response = requests.request("GET", url=url, headers=headers).text
        resp = json.loads(response)
        pptStatus = resp['data']['pptStatus']
        aiImageStatus = resp['data']['aiImageStatus']
        cardNoteStatus = resp['data']['cardNoteStatus']

        if ('done' == pptStatus and 'done' == aiImageStatus and 'done' == cardNoteStatus):
            PPTurl = resp['data']['pptUrl']
            break
        else:
            time.sleep(3)

    return PPTurl


# https://www.xfyun.cn/doc/nlp/xftrans/API.html
async def xunfei_picture(text: str, data_folder=None):
    headers = get_xfyun_authorization(api_key=Config.XF_API_Key, api_secret=Config.XF_Secret_Key,
                                      host="spark-api.cn-huabei-1.xf-yun.com", path="/v2.1/tti", method='POST')
    url = 'http://spark-api.cn-huabei-1.xf-yun.com/v2.1/tti' + "?" + urlencode(headers)
    # f"https://{host}{path}?"+ urlencode(headers)
    # 构造请求数据
    request_body = {
        "header": {
            "app_id": Config.XF_AppID,  # 你在平台申请的appid
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
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=request_body, headers={'content-type': "application/json"})
        if response.status_code != 200:
            return None, {"error": f"HTTP Error: {response.status_code}"}

        data = response.json()  # json.loads(response.text)
        code = data['header']['code']
        if code != 0:
            return None, {"error": f'请求错误: {code}, {data}'}

        text = data["payload"]["choices"]["text"]
        imageContent = text[0]
        image_base = imageContent["content"]  # base64_string_data
        image_name = data['header']['sid']
        # 解码 Base64 图像数据
        file_data = base64.b64decode(image_base)
        if data_folder:
            # 将解码后的数据转换为图片
            file_path = f"{data_folder}/{image_name}.jpg"
            img = Image.open(io.BytesIO(file_data))
            img.save(file_path)
            # with open(file_path, 'wb') as file:
            #     file.write(file_data)
            return file_path, image_name

        return file_data, image_name


# https://cloud.baidu.com/doc/NLP/s/al631z295
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

    async with httpx.AsyncClient() as cx:
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
# with open(image_path, 'rb') as f:
#    image_data = f.read()
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
    service = 'ocr'
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

    headers = get_tencent_signature(service, host, params=payload, action=ocr_type,
                                    secret_id=Config.TENCENT_SecretId, secret_key=Config.TENCENT_Secret_Key,
                                    version='2018-11-19')

    # payload = convert_keys_to_pascal_case(params)

    async with httpx.AsyncClient(timeout=Config.HTTP_TIMEOUT_SEC) as cx:
        response = await cx.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["Response"]  # "TextDetections"


# https://help.aliyun.com/zh/ocr/developer-reference/api-ocr-api-2021-07-07-dir/?spm=a2c4g.11186623.help-menu-252763.d_2_2_4.3aba47bauq0U2j
async def ali_ocr_recognise(image_data, image_url, access_token, ocr_type='accurate_basic'):
    headers = {
        'Content-Type': "application/x-www-form-urlencoded",
        'charset': "utf-8"
    }
    # accurate,general_basic,webimage
    url = 'ocr-api.cn-hangzhou.aliyuncs.com'
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
    token, _ = get_aliyun_access_token(service="nls-meta", region="cn-shanghai", access_key_id=Config.ALIYUN_AK_ID,
                                       access_key_secret=Config.ALIYUN_Secret_Key)
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
    audio_data = synthesizer.call(sentences)  # sample_rate=48000
    return audio_data, synthesizer.get_last_request_id()

    # SpeechSynthesizer.call(model='sambert-zhichu-v1',
    #                        text='今天天气怎么样',
    #                        sample_rate=48000,
    #                        format='pcm',
    #                        callback=callback)
    # if result.get_audio_data() is not None:


def dashscope_file_upload(messages, file_path='.pdf', api_key=''):
    url = 'https://dashscope.aliyuncs.com/compatible-mode/v1/files'
    headers = {
        'Content-Type': 'application/json',
        "Authorization": f'Bearer {api_key or Config.DashScope_Service_Key}',
    }

    try:
        files = {
            'file': open(file_path, 'rb'),
            'purpose': (None, 'file-extract'),
        }
        file_response = requests.post(url, headers=headers, files=files)
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

    # bucket.get_object(object_name)
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


async def download_file(url: str, dest_folder: Path = None) -> Path:
    """下载URL中的文件到目标文件夹"""
    try:
        async with httpx.AsyncClient() as client:
            # requests.get(url, stream=True, timeout=10)
            response = await client.get(url, stream=True, timeout=10)
            # filename = url.split("/")[-1]  # 提取文件名
            file_name = unquote(url.split("/")[-1].split("?")[0])
            # print(file_name)
            if response.status_code == 200:
                if dest_folder:
                    # 提取文件名
                    save_path = dest_folder / file_name
                    save_path.parent.mkdir(parents=True, exist_ok=True)

                    with open(save_path, "wb") as f:
                        async for chunk in response.aiter_bytes(chunk_size=8192):  # iter_content
                            f.write(chunk)

                    return save_path, file_name
                else:
                    file_data = b""
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        file_data += chunk

                    return file_data, file_name
            else:
                print(f"Failed to download file: {response.status_code},{file_name}")
                return None, file_name
    except Exception as e:
        print(f"Error downloading file: {e}")
        return None, None


def find_similar_paragraphs(node, target_embedding, top_n=3):
    """
    找到节点中与目标嵌入最相似的段落。
    :param node: 节点对象
    :param target_embedding: 目标嵌入
    :param top_n: 返回前N个相似的段落
    :return: [(段落文本, 相似度), ...] 按相似度降序排列
    """
    if "paragraph_embeddings" not in node.attributes() or "paragraph" not in node.attributes():
        return []

    paragraphs = node["paragraph"]
    paragraph_embeddings = np.array(node["paragraph_embeddings"])
    similarities = cosine_similarity_np(target_embedding, paragraph_embeddings).flatten()
    # sim_matrix = np.array(query_vector) @ np.array(tokens_vector).T

    # 按相似度降序排序
    similarities = sorted(zip(paragraphs, similarities), key=lambda x: x[1], reverse=True)
    return similarities[:top_n]


def query_similar_paragraphs(graph, target_embedding, top_n=7):
    all_paragraphs = []
    all_embeddings = []

    for v in graph.vs:
        if "paragraph" in v.attributes() and "paragraph_embeddings" in v.attributes():
            all_paragraphs.extend(v["paragraph"])
            all_embeddings.extend(v["paragraph_embeddings"])

    similarities = cosine_similarity_np(target_embedding, np.array(all_embeddings)).flatten()

    similarities = sorted(zip(all_paragraphs, similarities), key=lambda x: x[1], reverse=True)
    return similarities[:top_n]


def query_similar_content(graph, target_embedding, top_n_nodes=3, top_n_paragraphs=3):
    """
    查询图中与目标嵌入最相似的节点和段落。
    :param graph: 图对象
    :param target_embedding: 查询嵌入
    :param top_n_nodes: 返回前 N 个最相似的节点
    :param top_n_paragraphs: 在每个节点中返回前 N 个最相似段落
    :return: {节点: [(段落, 相似度), ...]}
    """
    # 计算目标嵌入与所有节点的相似度

    node_embeddings = np.array(graph.vs["combined_paragraph_embedding"])
    similarities = cosine_similarity_np(target_embedding, node_embeddings).flatten()

    # 按相似度降序排序
    similar_nodes = sorted(zip(graph.vs, similarities), key=lambda x: x[1], reverse=True)[:top_n_nodes]
    # 在每个节点中查找最相似段落
    results = {}
    for node, node_similarity in similar_nodes:
        similar_paragraphs = find_similar_paragraphs(node, target_embedding, top_n=top_n_paragraphs)
        results[node["name"]] = {
            "path": node["path"],
            "node_similarity": float(node_similarity),
            "similar_paragraphs": similar_paragraphs
        }

    return results


Ideatech_Graph = None


async def ideatech_knowledge(query, rerank_model="BAAI/bge-reranker-v2-m3", version=0):
    global Ideatech_Graph
    if not Ideatech_Graph:
        with open(f"{Config.DATA_FOLDER}/ideatech_pdf_graph.pkl", "rb") as f:
            Ideatech_Graph = pickle.load(f)
        print("load graph.")

    query_embedding = np.array(await ai_embeddings(query, model_name='qwen', model_id=0))
    if version:  # nodes and paragraphs
        similar_content = query_similar_content(Ideatech_Graph, query_embedding, top_n_nodes=3, top_n_paragraphs=3)
        paragraphs = [j[0] for i in similar_content.values() for j in i.get("similar_paragraphs")]
    else:
        similar_content = query_similar_paragraphs(Ideatech_Graph, query_embedding, top_n=9)
        paragraphs = [i[0] for i in similar_content]

    if rerank_model and paragraphs:
        similar_content = await ai_reranker(query, documents=paragraphs, top_n=4, model_name=rerank_model)

    return similar_content


def get_weather(city: str):
    # 使用 WeatherAPI 的 API 来获取天气信息
    api_key = Config.Weather_Service_Key
    base_url = "http://api.weatherapi.com/v1/current.json"
    city = convert_to_pinyin(city)
    params = {
        'key': api_key,
        'q': city,
        'aqi': 'no'  # 不需要空气质量数据
    }
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
            'object_name': object_name, 'file_type': get_file_type(object_name)}

    try:
        with httpx.Client(timeout=(10, 60)) as client:
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
