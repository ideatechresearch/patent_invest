import httpx
import asyncio
from pathlib import Path
from typing import List, Any, Union, Tuple, Callable
import random, time
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

AI_Client = {}


def init_ai_clients(api_keys=API_KEYS):
    for model in AI_Models:
        model_name = model['name']
        api_key = api_keys.get(model_name)
        if api_key:
            model['api_key'] = api_key
            if model_name in ('moonshot', 'glm', 'qwen', 'hunyuan', 'silicon', 'doubao', 'baichuan'):  # OpenAI_Client
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


def ai_tool_response(messages, tools=[], model_name='moonshot', model_id=-1, top_p=0.95, temperature=0.01):
    model_info, name = find_ai_model(model_name, model_id)
    client = AI_Client.get(model_info['name'], None)
    if not tools:
        tools = AI_Tools
    if client:
        completion = client.chat.completions.create(
            model=name,
            messages=messages,
            tools=tools,
            # tool_choice="auto",
            temperature=temperature,
            top_p=top_p,
        )
        return completion.choices[0].message  # response['choices'][0]['message']
        # return completion.model_dump()


def ai_tools_messages(response_message):
    tool_calls = response_message.tool_calls
    messages = [response_message]
    for tool_func in tool_calls:
        func_name = tool_func.function.name  # function_name
        func_args = tool_func.function.arguments  # function_args = json.loads(tool_call.function.arguments)
        try:
            func_out = eval(f'{func_name}(**{func_args})')
            messages.append({
                'role': 'tool',
                'content': f'{func_out}',
                'tool_call_id': tool_func.id
            })
        except:
            # exec(code)
            pass
    return messages  # [*tool_mmessages,]


def ai_files_messages(files: List[str], model_name='moonshot'):
    client = AI_Client.get(model_name, None)
    messages = []
    for file in files:
        file_object = client.files.create(file=Path(file), purpose="file-extract")
        file_content = client.files.content(file_id=file_object.id).text
        messages.append({
            "role": "system",
            "content": file_content,
        })
    return messages


async def ai_embeddings(inputs, model_name='qwen', model_id=0):
    model_info, name = find_ai_model(model_name, model_id, 'embedding')
    if not name:
        return []

    client = AI_Client.get(model_info['name'], None)
    if client:  # openai.Embedding.create
        completion = await asyncio.to_thread(client.embeddings.create,
                                             model=name, input=inputs,
                                             encoding_format="float")

        return [item.embedding for item in completion.data]
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
    max_batch_size = 16  # DASHSCOPE_MAX_BATCH_SIZE = 25
    embeddings = []
    async with httpx.AsyncClient() as cx:
        try:
            if isinstance(inputs, str) or (isinstance(inputs, list) and len(inputs) < max_batch_size):
                response = await cx.post(url, headers=headers, json=payload)
                data = response.json().get('data')
                embeddings = [emb.get('embedding') for emb in data]
            elif isinstance(inputs, list):
                for i in range(0, len(inputs), max_batch_size):
                    batch = inputs[i:i + max_batch_size]
                    payload["input"] = batch  # {"texts":batch}
                    response = await cx.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json().get('data')  # "output"
                    if data and len(data) == len(batch):
                        embeddings += [emb.get('embedding') for emb in data]
        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            print(exc)

    return embeddings


async def ai_reranker(query: str, documents: List[str], top_n: int, model_name="BAAI/bge-reranker-v2-m3", model_id=0):
    model_info, name = find_ai_model(model_name, model_id, 'reranker')
    if not name:
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
                      max_tokens=4096, model_name='silicon', model_id=0):
    model_info, name = find_ai_model(model_name, model_id, "generation")
    if not name:
        return ai_chat(messages=None, user_request=user_request, system=prompt, temperature=temperature,
                       max_tokens=max_tokens, top_p=0.8, model_name=model_name, model_id=model_id)

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
        # n=1,
    )
    if stream:
        async def stream_data():
            for chunk in response:
                yield chunk.choices[0].text

        return stream_data()

    return response.choices[0].text.strip()


async def retrieved_reference(user_request: str, keywords: List[Union[str, Tuple[str, Any]]] = None,
                              tool_calls: List[Callable[[...], Any]] = None, **kwargs):
    # Assume this is the document retrieved from RAG
    # function_call = Agent_Functions.get(agent, lambda *args, **kwargs: [])
    # refer = function_call(user_message, ...)

    async def wrap_sync(func, *args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)

    items_to_process = []
    user_calls = {'map_search': search_amap_location,
                  'web_search': web_search_async, }

    tool_calls = tool_calls or []
    if keywords:
        if all(not (callable(func) and func.__name__ == '<lambda>' and func()) for func in tool_calls):
            tool_calls.append(web_search_async)  # not in agent_funcalls
    else:
        items_to_process = [user_request]  # ','.join(keywords)

    tasks = []
    for item in keywords:
        if isinstance(item, tuple) and len(item) > 1:
            func = user_calls.get(item[0])  # 函数
            if func:
                func_args = item[1:]  # 剩下的参数
                if inspect.iscoroutinefunction(func):
                    tasks.append(func(*func_args, **kwargs))
                else:
                    tasks.append(wrap_sync(func, *func_args, **kwargs))
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
    # for f, r in zip(tasks, refer):
    #     print(f.__name__, r)
    return [item for result in refer if not isinstance(result, Exception) for item in result]  # 展平嵌套结果


# Callable[[参数类型], 返回类型]
async def get_chat_payload(messages, user_request: str, system: str = '', temperature: float = 0.4, top_p: float = 0.8,
                           max_tokens: int = 1024, model_name='moonshot', model_id=0,
                           tool_calls: List[Callable[[...], Any]] = None,
                           keywords: List[Union[str, Tuple[str, Any]]] = None, images: List[str] = None, **kwargs):
    model_info, name = find_ai_model(model_name, model_id, 'model')
    model_type = model_info['type']

    if isinstance(messages, list) and messages:
        if model_type in ('baidu', 'tencent'):
            if messages[0].get('role') == 'system':
                system = messages[0].get('content')
                del messages[0]

            # the role of first message must be user
            if messages[0].get('role') != 'user':  # user（tool）
                messages.insert(0, {'role': 'user', 'content': user_request or '请问您有什么问题？'})

            # 确保 user 和 assistant 消息交替出现
            for i, message in enumerate(messages[:-1]):
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
        messages[-1][
            'content'] = f'参考材料:\n{formatted_refer}\n 材料仅供参考,请根据上下文回答下面的问题:{user_request}'

    if images:
        messages[-1]['content'] = [{"type": "text", "text": user_request}]  # text-prompt 请详细描述一下这几张图片。
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

    # 通过 requests 库直接发起 HTTP POST 请求
    model_type = model_info['type']
    url = model_info['url']
    api_key = model_info['api_key']
    # body = payload
    if api_key:
        if isinstance(api_key, list):
            idx = model_info['model'].index(payload["model"])
            api_key = model_info['api_key'][idx]

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
        host = url.split("//")[-1]
        payload = convert_keys_to_pascal_case(payload)
        payload.pop('MaxTokens', None)
        headers = get_tencent_signature(service, host, payload, action='ChatCompletions',
                                        secret_id=Config.TENCENT_SecretId,
                                        secret_key=Config.TENCENT_Secret_Key)

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
        headers = get_tencent_signature(payload, service, host, action='ChatCompletions',
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
                                  match=[], not_match=[], query_vector=[],
                                  embeddings_calls: List[Callable[[str], Any]] = lambda x: [], **kwargs):
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
    div1 = np.linalg.norm(ndarr1, axis=1)
    div2 = np.linalg.norm(ndarr2, axis=1)
    denominator = np.outer(div1, div2)
    dot_product = np.dot(ndarr1, ndarr2.T)
    with np.errstate(divide='ignore', invalid='ignore'):
        similarity = np.where(denominator != 0, dot_product / denominator, 0)
    return similarity


async def similarity_embeddings(query, tokens, filter_idx=[], tokens_vector=None,
                                embeddings_calls: List[Callable[[str], Any]] = ai_embeddings, **kwargs):
    """
    计算查询与一组标记之间的相似度。
        filter_idx (List[int], optional): 要过滤的标记索引。默认为 []。
    返回：
        np.ndarray: 一个相似度得分的数组。
    """
    if filter_idx is None:
        filter_idx = list(range(len(tokens)))
    similarity = np.full(len(filter_idx), np.nan)
    if not query:
        return similarity

    idx_tokens = [(i, x) for i, x in enumerate(tokens) if x]
    query_vector = await embeddings_calls(query, **kwargs)
    if tokens_vector is None:
        # list(np.array(idx_tokens)[:, 1])
        tokens_vector = await embeddings_calls([token for _, token in idx_tokens], **kwargs)

    matching_indices = np.isin([j[0] for j in tokens], filter_idx)
    filter_embeddings = np.array(tokens_vector)[matching_indices]

    if len(filter_embeddings):
        # cosine_similarity_np(np.array(query_vector).reshape(1, -1), filter_embeddings).T
        sim_2d = np.array(query_vector).reshape(1, -1) @ filter_embeddings.T
        matching_indices = np.isin(filter_idx, [j[0] for j in tokens])
        similarity[matching_indices] = sim_2d.reshape(-1)

    return similarity


def get_similar_vectors(data, querys, exclude, topn=10, cutoff=0.0):
    '''
    {
      "name": ["word1", "word2"],
      "vectors": [[0.1, 0.2, ...], [0.3, 0.4, ...]]
    }
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
        top_indices = np.argsort(sim_scores)[::-1][:topn]  # 获取前 topn 个索引
        top_scores = sim_scores[top_indices]

        valid_indices = top_scores > cutoff  # 保留大于 cutoff 的相似度
        top_words = [index_to_key[j] for j in np.where(~exclude_mask)[0][top_indices[valid_indices]]]
        top_scores = top_scores[valid_indices]
        results.append((w, list(zip(top_words, top_scores))))

    return results  # [(w, list(sim[w].sort_values(ascending=False)[:topn].items())) for w in querys]


async def get_similar_embeddings(querys, tokens, embeddings_calls: List[Callable[[str], Any]] = ai_embeddings,
                                 topn=10, **kwargs):
    """
    使用嵌入计算查询与标记之间的相似度。
    返回：
        List[Tuple[str, List[Tuple[str, float]]]]: 查询词与相似标记及其分数的映射。
    """
    query_vector, tokens_vector = await asyncio.gather(
        embeddings_calls(querys, **kwargs),
        embeddings_calls(tokens, **kwargs))

    sim_matrix = np.array(query_vector) @ np.array(tokens_vector).T
    results = []
    for i, w in enumerate(querys):
        sim_scores = sim_matrix[i]
        top_indices = np.argsort(sim_scores)[::-1][:topn]
        top_scores = sim_scores[top_indices]
        top_words = [tokens[j] for j in top_indices]
        results.append((w, list(zip(top_words, top_scores))))

    return results


async def find_closest_matches_embeddings(querys, tokens,
                                          embeddings_calls: List[Callable[[str], Any]] = ai_embeddings, **kwargs):
    """
    使用嵌入计算查询与标记之间的最近匹配。
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
    closest_matches = tokens[sim_matrix.argmax(axis=1)]  # 找到每个查询的最佳匹配标记 idxmax
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
        return res


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
                result_text = response_data["payload"]["result"]["text"]
                decoded_result = base64.b64decode(result_text).decode('utf-8')
                data = json.loads(decoded_result)
                if "trans_result" in data:
                    return data["trans_result"]["dst"]
            else:
                return {"error": "Unexpected response format"}
        else:
            return {"error": f"HTTP Error: {response.status_code}"}


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


# https://help.aliyun.com/zh/ocr/developer-reference/api-ocr-api-2021-07-07-dir/?spm=a2c4g.11186623.help-menu-252763.d_2_2_4.3aba47bauq0U2j
async def ali_ocr_recognise(image_data, image_url, access_token, ocr_sign='accurate_basic'):
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
    token, _ = get_aliyun_access_token(Config.ALIYUN_AK_ID, Config.ALIYUN_Secret_Key, service="nls-meta",
                                       region="cn-shanghai")
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


Agent_Functions = {
    'default': lambda *args, **kwargs: [],
    '2': web_search,  # web_search_async
}

AI_Tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "当你想知道现在的时间时非常有用。",
            "parameters": {}  # 此处是函数参数相关描述, 因为获取当前时间无需输入参数，因此parameters为空字典
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_weather',
            'description': 'Get the current weather for a given city.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'city': {
                        'type': 'string',
                        'description': 'The name of the city to query weather for.',
                    },
                },
                'required': ['city'],
            },
        }
    }
]


def get_current_time():
    current_datetime = datetime.now()
    formatted_time = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    return f"当前时间：{formatted_time}。"


def get_weather(city: str):
    # 使用 WeatherAPI 的 API 来获取天气信息
    api_key = Config.Weather_Service_Key
    base_url = "http://api.weatherapi.com/v1/current.json"
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
