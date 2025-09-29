import logging
from typing import Literal, AsyncIterable
from openai import AsyncOpenAI, BadRequestError

from utils import *
from service import *
from agents import *
from database import BaseReBot

Function_Registry_Global: Optional[FunctionManager] = None


def get_func_manager() -> FunctionManager:
    global Function_Registry_Global
    if Function_Registry_Global is None:
        Function_Registry_Global = FunctionManager(keywords_registry=default_keywords_registry(), copy=True,
                                                   mcp_server=mcp)

    return Function_Registry_Global


def default_keywords_registry():
    # 初始化 keywords_registry 的工厂函数
    return {
        'no_func': lambda *args, **kwargs: f"[❌] Function not loaded",
        'map_search': search_amap_location,
        'web_search': web_search_async,
        'intent_search': web_search_intent,
        'tavily_search': web_search_tavily,
        'tavily_extract': web_extract_tavily,
        'jina_extract': web_extract_jina,
        'news_search': named_partial('news_search', web_search_tavily, topic='news', time_range='w'),
        'baidu_search': named_partial('baidu_search', search_by_api, engine='baidu'),
        'wiki_search': wikipedia_search,
        'arxiv_search': arxiv_search,
        'translate': auto_translate,
        "execute_code": execute_code_results,
        'auto_calls': ai_auto_calls,  # 名称不同，需指定调用，防止重复 ai_auto_calls
        'baidu_nlp': baidu_nlp,
        'ai_chat': ai_client_completions,

        'visitor_records': ideatech_visitor_records,
    }


def global_function_registry(func_name: str = None) -> Union[Callable[..., Any], Dict[str, Callable[..., Any]]]:
    """
    获取全局函数注册表中的函数或整个注册表。
    """
    func_manager = get_func_manager()
    if not func_manager.keywords_registry:
        func_manager.update_registry(default_keywords_registry())

    return func_manager.get_function_registered(func_name)


async def ai_generate_metadata(function_code: str, metadata: dict = None, model_name=Config.DEFAULT_MODEL_METADATA,
                               description: str = None, code_type: str = "Python", dbpool=None, **kwargs) -> dict:
    if not model_name:
        model_name = Config.DEFAULT_MODEL_METADATA

    prompt = System_content.get('84').format(code_type=code_type.lower(), function_code=function_code)
    lines = [f"帮我根据函数代码生成提取函数元数据（JSON格式）。"]
    if metadata:
        lines.append(f"当前已有初始元数据如下，请在此基础上补全或修正:{json.dumps(metadata, ensure_ascii=False)}")
    if description:
        lines.append(f"工具描述为:{description}，请结合此信息完善函数用途和说明")

    prompt_user = "\n".join(lines)
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": prompt_user}]

    content, row_id = await ai_client_completions(messages, client=None, model=model_name, get_content=True,
                                                  max_tokens=1000, temperature=0.3, dbpool=dbpool, **kwargs)
    if content:
        parsed = extract_json_struct(content)
        if parsed:
            metadata = parsed
            await BaseReBot.async_save(
                data={"reference": BaseMysql.format_value(metadata), "transform": BaseMysql.format_value(parsed)},
                row_id=row_id, dbpool=dbpool or DB_Client)
        else:
            logging.warning(f'metadata extract failed:{content}')

    return metadata or {}


def agent_func_calls(agent: str = 'default', messages: list = None,
                     tools: list[dict] = None, keywords: list[str | tuple[str, ...]] = None, prompt: str = None,
                     model: str = Config.DEFAULT_MODEL_FUNCTION, **kwargs) -> List[Callable[..., Any]]:
    from script.knowledge import ideatech_knowledge
    """
    根据 agent 类型和传入的 tools 配置，返回对应的函数调用列表
    :param agent: 代理编号
    :param messages: 历史对话消息
    :param tools: 工具配置列表，每个元素为 dict，包含 'type' 和对应参数
    :param keywords: 输入关键词列表，用于组合调用参数
    :param model: 模型信息，可用于部分工具函数
    :param prompt: 用户请求内容 prompt，用于某些 agent, 单独文本作为 fallback keyword
    :return: 函数列表，每个函数为 partial 结构，可直接调用
    """
    items_to_process = []
    if keywords:
        if isinstance(keywords, list):
            items_to_process = [item for item in keywords if isinstance(item, str)]
    else:
        if prompt:
            items_to_process = [prompt]  # ','.join(keywords)

    callable_map_agent = {
        'default': [lambda *args, **kwargs: [], ],  # 默认返回一个空函数列表
        '2': [named_partial('baidu_search', search_by_api, engine='baidu'), web_search_async],
        '9': [named_partial('auto_translate_baidu', auto_translate, model_name='baidu')],
        '29': [ideatech_knowledge],
        '31': [named_partial('auto_calls_any', ai_auto_calls, user_messages=messages,
                             system_prompt=System_content.get('31'), model_name=model, get_messages=False)],
        '32': [ai_auto_calls],
        '37': [web_search_async]
        # 扩展更多的 agent 类型，映射到多个函数
    }

    func_calls: list[Callable[..., Any]] = callable_map_agent.get(agent, [])
    # 如果有关键词且 func_calls 全部非空 lambda，添加一个默认函数（如意图识别）append auto func to run. ai_auto_calls
    if keywords and all(not is_empty_lambda(_func) for _func in func_calls):  # and _func()
        func_calls.append(web_search_intent)

    tool_calls: list[Callable[..., Any]] = []  # callable_list
    # 对每个关键词，绑定代理函数,多个 tool_calls to process items,kwargs func参数,agent控制
    for _func in filter(callable, func_calls):
        if is_empty_lambda(_func):
            continue
        tool_calls.extend([partial(_func, item, **kwargs) for item in items_to_process])
        # print(bound_func.func.__name__, bound_func.args, bound_func.keywords,bound_func.arguments)

    # 解析 tools 并绑定, 函数绑定特定的配置参数,无参数可调用函数
    if tools:
        # retrieval、web_search、function, https://docs.bigmodel.cn/cn/guide/tools/web-search
        registry = default_keywords_registry()  # tool_calls.extend(convert_to_callable_list())
        tools_params: list = get_tools_params(tools)
        for tool_name, params in tools_params:
            if tool_name not in registry:
                continue
            _func = registry[tool_name]
            if params and isinstance(params, dict):
                tool_calls.append(partial(_func, **params))  # 有参数的情况，直接 partial
            else:
                tool_calls.extend([partial(_func, item) for item in items_to_process])  # 无参数的情况，遍历每个 item

    return tool_calls


async def ideatech_visitor_records(prompt: str, customer_name: str | list | tuple, **kwargs):
    print(prompt, customer_name, kwargs)
    match = field_match('客户名称', customer_name)
    payload = dict(querys=[prompt], collection_name='拜访记录', client=QD_Client,
                   payload_key='record_text', match=match, not_match=[],
                   topn=20, score_threshold=0.5, exact=False,
                   embeddings_calls=ai_embeddings, model_name='BAAI/bge-large-zh-v1.5')
    payload.update(kwargs)
    return await search_by_embeddings(**payload)


async def ai_tools_results(tool_messages, func_manager: FunctionManager) -> list[dict]:
    """
    解析模型响应，动态调用工具并生成后续消息列表。

    :param tool_messages: 模型响应的消息对象
    :param func_manager:
    :return: 包含原始响应和工具调用结果的消息列表
    """

    messages = [tool_messages.to_dict()]  # ChatCompletionMessage,ChatMessage
    tool_calls = tool_messages.tool_calls
    if not tool_calls:
        results = execute_code_results(tool_messages.content)
        # print(results)
        for res in results:
            messages.append({
                'role': 'tool',
                'content': res['error'] if res['error'] else res['output'],
                'tool_call_id': 'output.getvalue',
                'name': 'exec'
            })
        return messages

    for tool in tool_calls:
        func_name = tool.function.name  # function_name
        if not func_name:
            print(f'Error in tool_message:{tool.to_dict()}')
            continue

        func_args = tool.function.arguments  # function_args = json.loads(tool_call.function.arguments)
        # func_reg = global_function_registry(func_name)  # 从注册表中获取函数 tools_map[function_name](**function_args)
        # print(func_args)
        func_out = await func_manager.run_function(func_name, func_args, tool.id)
        messages.append(func_out)

    return messages  # [*tool_mmessages,]


async def ai_auto_calls(question, user_messages: list = None, system_prompt: str = None,
                        model_name=Config.DEFAULT_MODEL_FUNCTION,
                        get_messages=False, **kwargs) -> list:
    """
    call_tool
    自动推理并调用工具，调用 AI 模型接口，使用提供的工具集和对话消息，返回模型的响应,返回调用结果或消息列表
    :param question: 从问题里面获取意图选择工具调用
    :param user_messages: 从对话上下文里面获取意图选择工具调用，两者选一项
    :param system_prompt:获取系统提示词
    :param model_name: 默认使用qwen-max
    :param get_messages: 是否组装成方法：结果列表
    :param kwargs:
    :return: 结果列表或者 模型messages回复+本地执行方法结果
    """
    # 构造对话上下文
    if not system_prompt:
        system_prompt = System_content.get('31')
    if user_messages and isinstance(user_messages, list):
        user_messages = user_messages[-10:].copy()
        if not any(msg.get('role') == 'system' for msg in user_messages):
            user_messages.insert(0, {"role": "system", "content": system_prompt})
    else:
        user_messages = [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": question}]
    # 获取工具列表
    func_manager = get_func_manager()
    tools_metadata = AI_Tools + await func_manager.get_tools_metadata(func_list=[])
    # 模型决定使用哪些工具
    # tools = [{"type": "web_search",
    #          "web_search": {
    #         "enable": True  # 启用网络搜索
    #         "search_query": "自定义搜索的关键词",
    #          "search_result": True,#默认为禁用,允许用户获取详细的网页搜索来源信息
    #     }}]
    tool_messages, row_id = await ai_client_completions(user_messages, client=None, model=model_name, get_content=False,
                                                        tools=tools_metadata, top_p=0.95, temperature=0.01, **kwargs)
    if not tool_messages:
        return []

    # 执行工具调用
    final_messages = await ai_tools_results(tool_messages, func_manager)  # 组合了模型回复后的结果
    tool_results = [{'function': msg['name'], 'result': msg['content']} for msg in final_messages if
                    msg['role'] == "tool"]

    await BaseReBot.async_save(
        data={"user_content": question, "reference": json.dumps(tool_results, ensure_ascii=False)},
        row_id=row_id, dbpool=DB_Client)
    if not get_messages:
        return [tuple(item.values()) for item in tool_results]  # [{func_name:func_result}

    result_messages = user_messages
    for msg in final_messages:
        if msg["role"] == "assistant" and "tool_calls" in msg:
            # 处理 tool_calls 为 JSON 格式
            msg["content"] = json.dumps(msg["tool_calls"], indent=2, ensure_ascii=False)
            result_messages.append(msg)
    # 构造消息列表，适合继续对话
    formatted_results = "\n".join(f"(function:{item['function']},result:{item['result']})" for item in tool_results)
    result_messages.append({
        "role": "system",  # "user"
        "content": f"question:{question},\nresults:\n{formatted_results}"
    })  # {"role": "system", "content": f"已调用函数 {func_name}，结果如下：{func_result}"}]
    return result_messages  # [-1]


async def ai_files_messages(files: List[str], question: str = None, model_name: str = 'qwen-long', max_tokens=4096,
                            dbpool=None, **kwargs):
    """
    处理文件并生成 AI 模型的对话结果。

    :param files: 文件路径列表
    :param question:问题提取
    :param model_name: 模型名称
    :param max_tokens
    :param dbpool
    :return: 模型生成的对话结果和文件对象列表
    """
    model_info, name = find_ai_model(model_name, -1)
    client = AI_Client.get(model_info['name'], None)
    messages = []
    for file_path in files:
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():  # .is_file()
            continue

        if client:  # purpose=Literal["assistants", "batch", "fine-tune", "vision","batch_output"]
            file_object = await client.files.create(file=file_path_obj, purpose="file-extract")
            if model_info['name'] in ('qwen', 'openai'):  # fileid 引用协议
                messages.append({"role": "system", "content": f"fileid://{file_object.id}", })
                # client.files.list()
                # 文件信息 file_info = await client.files.retrieve(file_object.id)
            else:  # 直接读取内容注入到 prompt 中,内容大小有限制,通用性最强,Claude、Mistral、Gemini、moonshot
                file_content = await client.files.content(file_id=file_object.id)  # 文件内容
                messages.append({"role": "system", "content": file_content.text, })
        else:
            dashscope_file_upload(messages, file_path=str(file_path_obj))

    if question:
        messages.append({"role": "user", "content": question})
        bot_response, _ = await ai_client_completions(messages, client, model=name, get_content=True,
                                                      max_tokens=max_tokens, dbpool=dbpool, **kwargs)
        messages.append({"role": "assistant", "content": bot_response})
        return messages

    return messages


async def ai_batch(inputs_list: List[list[dict] | tuple[str, str]], task_id: str, model: str = 'qwen-long',
                   search_field: str = 'model', **kwargs):
    model_info, name = find_ai_model(model, -1, search_field)
    endpoint_map = {  # 选择Embedding文本向量模型进行调用时，url的值需填写"/v1/embeddings",其他模型填写/v1/chat/completions
        "model": "/v1/chat/completions",
        "embedding": "/v1/embeddings",
        "generation": "/v1/completions"
    }
    endpoint = endpoint_map.get(search_field, "/v1/chat/completions")
    input_filename = f'{task_id}_input.jsonl'
    file_path = Path(Config.DATA_FOLDER) / input_filename
    async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
        for i, msg in enumerate(inputs_list):
            messages = [{"role": "system", "content": msg[0]},
                        {"role": "user", "content": msg[1]}] if isinstance(msg, tuple) else msg
            body = {"model": name, "messages": messages, **kwargs}
            request = {"custom_id": i, "method": "POST", "url": endpoint, "body": body}
            await f.write(json.dumps(request, separators=(',', ':'), ensure_ascii=False) + "\n")

    client = AI_Client.get(model_info['name'], None)
    # Step 1: 上传包含请求信息的JSONL文件,得到输入文件ID,如果您需要输入OSS文件,可将下行替换为：input_file_id = "实际的OSS文件URL或资源标识符"
    file_object = await client.files.create(file=file_path, purpose="batch")
    # Step 2: 基于输入文件ID,创建Batch任务
    batch = await client.batches.create(input_file_id=file_object.id, endpoint=endpoint, completion_window="24h")
    return {
        "input_file_id": file_object.id,
        "batch_id": batch.id,
        "input_file": input_filename,
        "client_name": model_info['name']
    }


async def ai_batch_result(batch_id: str, task_id: str, client_name: str,
                          interval: int = 10,  # 每次查询间隔秒数
                          timeout: int = 3600,  # 最长等待时间（秒），默认 1 小时
                          oss_expires: int = 86400
                          ):
    """
    轮询批处理任务状态，等待完成后下载结果文件。
    - batch_id: 批处理任务ID
    - client_name: AI_Client 的 key
    - task_id: 用来命名输出文件，避免混淆
    - interval: 轮询间隔秒数，<=0 不轮询
    - timeout: 超时时间，防止永远等待，<=0 永远等待
    """
    client = AI_Client.get(client_name, None)
    loop = asyncio.get_running_loop()
    start_time = loop.time()
    while True:
        batch = await client.batches.retrieve(batch_id=batch_id)
        status = batch.status

        if status in ["completed", "failed", "expired", "cancelled"]:
            # 结束状态，停止轮询
            print(f"[Batch状态] {batch_id}: {status}")
            break

        if loop.time() - start_time > timeout > 0:
            logger.warning(f"[Batch超时] {batch_id} 等待超时({timeout}s)")
            return {"status": "timeout"}

        if interval > 0:
            await asyncio.sleep(interval)
        else:
            print(batch.model_dump())  # 'in_progress'
            break  # 不轮询，立即查看状态

    # ✅ 任务完成，下载结果文件
    result = {"status": status, "request_counts": batch.request_counts.model_dump()}

    if status == "failed":
        result["errors"] = str(batch.errors)
        return result

    output_filename = f'{task_id}_{batch_id}_result.jsonl'
    error_filename = f'{task_id}_{batch_id}_error.jsonl'
    output_file_path = Path(Config.DATA_FOLDER) / output_filename
    error_file_path = Path(Config.DATA_FOLDER) / error_filename
    if batch.error_file_id and not error_file_path.exists():
        content = await client.files.content(batch.error_file_id)
        logger.error(f"[Batch error] {batch_id} 前1000字符:\n{content.text[:1000]}...\n")
        content.write_to_file(error_file_path)
        result["error_file"] = error_filename
        result["error_file_id"] = batch.error_file_id
        result["batch_id"] = batch.id or batch_id

    if batch.output_file_id and not output_file_path.exists():
        content = await client.files.content(batch.output_file_id)
        print(f"[Batch result] {batch_id} 前1000字符:\n{content.text[:1000]}...\n")
        content.write_to_file(output_file_path)
        result["result_file"] = output_filename
        result["output_file_id"] = batch.output_file_id
        result["batch_id"] = batch.id or batch_id
        if oss_expires != 0:
            object_name = f"upload/{output_filename}"
            with open(output_file_path, 'rb') as file_obj:
                result["output_file_url"] = await asyncio.to_thread(upload_file_to_oss, AliyunBucket, file_obj,
                                                                    object_name, expires=oss_expires)
            # os.remove(output_file_path)

    return result


Embedding_Cache = {}


async def get_embedding_from_cache(inputs: Union[str, List[str], Tuple[str]], model_name=Config.DEFAULT_MODEL_EMBEDDING,
                                   arg_list: list = None, redis=None):
    """检查缓存，如果没有就计算嵌入"""
    global Embedding_Cache
    cache_key = generate_hash_key(inputs, model_name, arg_list)

    try:
        if redis:
            cached_embedding = await redis.get(f"embedding:{cache_key}")
            await redis.expire(f"embedding:{cache_key}", Config.REDIS_CACHE_SEC)
        else:
            raise Exception
    except:
        cached_embedding = Embedding_Cache.get(cache_key, [])

    if cached_embedding:
        embedding = json.loads(cached_embedding) if isinstance(cached_embedding, str) else cached_embedding
        if isinstance(embedding, list) and all(isinstance(vec, list) for vec in embedding):
            return cache_key, embedding
    return cache_key, []

    # if cache_key in Embedding_Cache:
    #     print(cache_key)
    #     return Embedding_Cache[cache_key]
    # embedding = ai_embeddings(inputs, model_name, model_id, **kwargs)
    # Embedding_Cache[cache_key] = embedding
    # return embedding


# https://www.openaidoc.com.cn/docs/guides/embeddings
async def ai_embeddings(inputs: Union[str, List[str], Tuple[str], List[Dict[str, str]]],
                        model_name: str = Config.DEFAULT_MODEL_EMBEDDING,
                        model_id: int = 0, get_embedding: bool = True, normalize: bool = False,
                        **kwargs) -> Union[List[List[float]], Dict]:
    """
    text = text.replace("\n", " ")
    从远程服务获取嵌入，支持批量处理和缓存和多模型处理。
    :param inputs: 输入文本或文本列表
    :param model_name: 模型名称
    :param model_id: 模型序号
    :param get_embedding: 返回响应数据类型
    :param normalize: 归一化
    :return: 嵌入列表
    （1）文本数量不超过 16。
    （2）每个文本长度不超过 512 个 token，超出自动截断，token 统计信息，token 数 = 汉字数+单词数*1.3 （仅为估算逻辑，以实际返回为准)。
    （3） 批量最多 16 个，超过 16 后默认截断。
    """
    if not model_name:
        return []
    if not inputs:
        return []

    redis = get_redis()
    cache_key, embedding = await get_embedding_from_cache(inputs, model_name, [model_id, get_embedding, normalize],
                                                          redis=redis)
    if embedding:
        # print(f"Embedding already cached for key: {cache_key}")
        return embedding

    try:
        model_info, name = find_ai_model(model_name, model_id, 'embedding')
    except Exception as e:
        print(e)
        return []

    batch_size = 10  # DASHSCOPE_MAX_BATCH_SIZE = 16,25
    has_error = False
    payload = dict(
        model=name,
        encoding_format="float",
        # "dimensions":# 指定向量维度（text-embedding-v3及 text-embedding-v4）
    )
    payload.update(kwargs)

    client = AI_Client.get(model_info['name'], None)
    if client:  # openai.Embedding.create
        if isinstance(inputs, (list, tuple)) and len(inputs) > batch_size:
            create_embeddings = run_togather(max_concurrent=Config.MAX_CONCURRENT, batch_size=batch_size,
                                             input_key="input")(client.embeddings.create)
            results = await create_embeddings(inputs=inputs, **payload)
            if not get_embedding:
                results_data = results[0]
                all_data = []
                for i, result in enumerate(results):
                    for idx, item in enumerate(result.data):
                        item.index = len(all_data)
                        all_data.append(item)

                    if i > 0:
                        results_data.usage.prompt_tokens += result.usage.prompt_tokens
                        results_data.usage.total_tokens += result.usage.total_tokens

                results_data.data = all_data
                return results_data.model_dump()

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
        else:
            # await asyncio.to_thread(client.embeddings.create
            results = await client.embeddings.create(input=inputs, **payload)

            if not get_embedding:
                return results.model_dump()

            embeddings = [item.embedding for item in results.data]

        if normalize and not any(embedding is None for embedding in embeddings):
            embeddings = normalize_embeddings(embeddings, to_list=True)

        if not has_error:  # and len(embeddings) > batch_size
            if redis:
                try:
                    await redis.setex(f"embedding:{cache_key}", Config.REDIS_CACHE_SEC, json.dumps(embeddings))
                except Exception as e:
                    print(e)
            Embedding_Cache[cache_key] = embeddings

        return embeddings

    # dashscope.TextEmbedding.call
    url = model_info['embedding_url'] if model_info.get('embedding_url') else model_info['base_url'] + '/embeddings'
    api_key = model_info['api_key']
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    payload["input"]: list | str = inputs
    embeddings = []
    cx = get_httpx_client()
    try:
        if isinstance(inputs, (list, tuple)) and len(inputs) > batch_size:
            for i in range(0, len(inputs), batch_size):
                batch = inputs[i:i + batch_size]
                payload["input"] = batch  # {"texts":batch}
                response = await cx.post(url, headers=headers, json=payload, timeout=Config.LLM_TIMEOUT_SEC)
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

    if normalize:
        embeddings = normalize_embeddings(embeddings, to_list=True)

    if not has_error:
        if redis:
            try:
                await redis.setex(f"embedding:{cache_key}", Config.REDIS_CACHE_SEC, json.dumps(embeddings))
            except Exception as e:
                print(e)
        Embedding_Cache[cache_key] = embeddings

    return embeddings


@async_error_logger(1)
async def ai_reranker(query: str, documents: List[str], top_n: int, model_name="BAAI/bge-reranker-v2-m3", model_id=0,
                      **kwargs):
    if not model_name:
        return []
    model_info, name = find_ai_model(model_name, model_id, 'reranker')

    url = model_info['reranker_url']
    api_key = model_info.get('api_key')
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
            matches = [(match["document"].get("text", match["document"])
                        if match.get("document") else documents[match["index"]],
                        match["relevance_score"], match["index"])
                       for match in results]
            return matches
        else:
            print(response.text)
    return []


@async_error_logger(1)
async def ai_client_completions(messages: list, client: Optional[AsyncOpenAI] = None, model: str = 'deepseek-chat',
                                get_content=True, tools: list[dict] = None, max_tokens=4096, top_p: float = 0.95,
                                temperature: float = 0.1, dbpool=None, **kwargs):
    """
    return: str or 模型响应的消息对象, lastrowid
    """
    chat = True
    if not client:
        try:
            model_info, name = find_ai_model(model, 0, search_field='model')
        except:
            chat = False
            model_info, name = find_ai_model(model, search_field="generation")

        client = AI_Client.get(model_info['name'], None)
        if not client:
            raise ValueError(f"Client for model {model_info['name']}:{model} not found")
        model = name

    payload = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stream=False,
        **kwargs
    )
    try:
        if chat:
            if tools:
                filter_tools: list[dict] = deduplicate_functions_by_name(tools)
                if any(filter_tools):  # 一旦你传了 tools 字段，它 必须 至少包含一个合法的 tool
                    payload['tools'] = filter_tools
                    # tool_choice="auto",

            completion = await client.chat.completions.create(**payload)
            row_id = await BaseReBot.async_save(model=model, messages=messages, model_response=completion.model_dump(),
                                                dbpool=dbpool or DB_Client, user='local', agent='chat')
            return (completion.choices[0].message.content if get_content else completion.choices[0].message), row_id
        else:
            payload['prompt'] = build_prompt(messages, use_role=False)
            payload.pop('messages', None)
            completion = await client.completions.create(**payload)
            row_id = await BaseReBot.async_save(model=model, messages=messages, model_response=completion.model_dump(),
                                                dbpool=dbpool or DB_Client, user='local', agent='generate')
            return completion.choices[0].text, row_id
    except Exception as e:
        print(f"OpenAI error occurred: {e}")
        raise


async def ai_analyze(results: dict | list | str, system_prompt: str = None, desc: str = None,
                     model='deepseek-reasoner', max_tokens=4096, temperature: float = 0.2, dbpool=None, **kwargs):
    user_request = results if isinstance(results, str) else json.dumps(results, ensure_ascii=False)
    if desc:
        user_request = f"{desc}: {user_request}"

    messages = [{"role": "system", "content": system_prompt}, {'role': 'user', 'content': user_request}]
    content, _ = await ai_client_completions(messages, client=None, model=model, get_content=True,
                                             max_tokens=max_tokens, temperature=temperature, dbpool=dbpool, **kwargs)
    logging.info(f'</{desc}>: {content}')
    return content.split("</think>")[-1]


async def ai_analyze_together(variable_name: str, variable_values: list,
                              system_prompt: str = None, user_request: str = None,
                              model='deepseek-reasoner', max_tokens=4096, temperature: float = 0.2,
                              max_retries: int = 2, max_concurrent: int = Config.MAX_CONCURRENT, dbpool=None, **kwargs):
    '''
    多变量并发分析，对同一数据使用不同变量值进行多角度分析,返回分析结果列表
    顺序与variable_values对应,variable_name: 要变化的变量名(覆盖)
    '''
    if variable_name == 'model':
        client = None
    else:
        model_info, name = find_ai_model(model, 0, search_field='model')
        client = AI_Client.get(model_info['name'], None)

    @run_togather(max_concurrent=max_concurrent, batch_size=-1, input_key=variable_name)
    async def _run(item):
        payload = {"system_prompt": system_prompt, "user_request": user_request, "model": model,
                   "max_tokens": max_tokens, "temperature": temperature, **kwargs, variable_name: item}
        messages = [{"role": "system", "content": payload.pop('system_prompt')},
                    {'role': 'user', 'content': payload.pop('user_request')}]
        content, row_id = await ai_client_completions(messages, client=client, get_content=True, dbpool=dbpool,
                                                      **payload, max_retries=max_retries)
        return {"content": content, "variable_item": item}  # , "row_id": row_id

    results = await _run(inputs=variable_values)
    return [{'id': i, **r} for i, r in enumerate(results) if
            not isinstance(r, Exception)], [{'id': i, 'error': str(r)} for i, r in enumerate(results) if
                                            isinstance(r, Exception)]


async def ai_summary(long_text: str | list[str | dict],
                     extract_prompt: str = None, summary_prompt: str = None,
                     extract_model='qwen:qwen-plus', summary_model='qwen:qwen-plus',
                     encoding_model=Config.DEFAULT_MODEL_ENCODING,
                     extract_max_tokens=4096, summary_max_tokens=8182, segment_max_length=10000, dbpool=None):
    tokenizer = get_tokenizer(encoding_model)
    segments_chunk: list = []
    if isinstance(long_text, str):
        lang = lang_detect_to_trans(long_text)
        if lang == 'zh':
            # sentences = split_sentences_clean(long_text, h_symbols=True, h_tables=True)
            segments_chunk: list[str] = structure_aware_chunk(long_text, max_size=segment_max_length)
        else:
            tokens = tokenizer.encode(long_text)
            small_chunks, large_chunks = organize_segments(tokens,
                                                           small_chunk_size=max(175, int(segment_max_length * 0.34)),
                                                           large_chunk_size=segment_max_length,
                                                           overlap=max(20, int(segment_max_length * 0.04)))
            segments_chunk = [tokenizer.decode(chunk) for chunk in large_chunks if chunk]

    elif isinstance(long_text, list):
        if all(isinstance(x, str) for x in long_text):
            segments_chunk: list[list[str]] = organize_segments_chunk(long_text, chunk_size=7, overlap_size=2,
                                                                      max_length=segment_max_length,
                                                                      tokenizer=tokenizer)
        if all(isinstance(x, dict) for x in long_text):
            segments_chunk: list[list[dict]] = list(
                df2doc_split(long_text, max_tokens=segment_max_length, tokenizer=tokenizer))

    if not segments_chunk:
        raise ValueError("long_text 必须是 str 或 list[str|dict]")

    if not summary_prompt:
        summary_prompt = '你是总结助手，请根据以下多个信息片段进行归纳总结，提炼主旨、关键事件或共同规律。如片段内容存在重复或上下文重叠，可适当整合。'
    else:
        if not extract_prompt:
            extract_prompt = '你是信息摘要助手，请提炼以下关键信息:\n' + summary_prompt

    @run_generator(max_concurrent=Config.MAX_CONCURRENT, return_exceptions=True)
    async def extract_item(segment: list | str):
        if not segment:
            raise ValueError("bad input")
        content = format_content_str(segment, exclude_null=True)

        if lang_token_size(content, tokenizer=tokenizer) > extract_max_tokens:
            messages = [{"role": "system",
                         "content": extract_prompt or '你是信息摘要助手，请提炼以下文本中的关键信息、数据或事件。'},
                        {'role': 'user', 'content': content}]
            content, _ = await ai_client_completions(messages, client=None, model=extract_model, get_content=True,
                                                     max_tokens=extract_max_tokens, temperature=0.3, dbpool=dbpool)
        print(segment, '>', content)
        return content

    raw_results = []
    async for idx, extract in extract_item(inputs=segments_chunk):
        if isinstance(extract, Exception):
            yield {'seq': idx, 'content': None, 'type': 'extract', 'error': str(extract), 'item': segments_chunk[idx]}
            continue
        res = {'seq': idx, 'content': extract, 'type': 'extract'}
        yield res
        raw_results.append(res)
    results = sorted(raw_results, key=lambda x: x['seq'], reverse=True)
    summary = await ai_analyze(results, system_prompt=summary_prompt, desc='信息片段', model=summary_model,
                               max_tokens=summary_max_tokens, dbpool=dbpool)
    yield {'seq': -1, 'content': summary, 'type': 'summary'}  # 'extract': results,


async def call_ollama(prompt: str | list[str] = None, messages: list[dict] = None, model_name="mistral",
                      host='localhost', time_out: float = 100, stream=True, tools: list[dict] = None,
                      embed: bool = False, **kwargs):
    """
    https://github.com/ollama/ollama/blob/main/docs/api.md
    返回两种模式的结果：async generator（stream）或 JSON dict（非 stream）
    """
    if not prompt and not messages:
        raise ValueError("必须提供 prompt 或 messages 之一")
    # api/tags:列出本地模型,/api/ps:列出当前加载到内存中的模型

    if embed:
        url = f"http://{host}:11434/api/embed"
        payload = {"model": model_name, "input": prompt}
    elif messages is not None:
        url = f'http://{host}:11434/api/chat'
        payload = {
            "model": model_name,
            "messages": messages,
            # 如果 messages 数组为空，则模型将被加载到内存中, [{"role": "user","content": "Hello!","images": ["..."]}]
            "stream": stream
        }
        if tools:
            payload['tools'] = tools
    else:
        if isinstance(prompt, list):
            prompt = "\n".join(prompt)

        url = f"http://{host}:11434/api/generate"
        payload = {"model": model_name, "prompt": prompt, "stream": stream
                   # "suffix": "    return result",模型响应后的文本
                   # "images"：要包含在消息中的图像列表（对于多模态模型，例如llava)
                   # "keep_alive": 控制模型在请求后加载到内存中的时间（默认值：5m)
                   # "keep_alive": 0，如果提供了空提示，将从内存中卸载模型
                   # "format": "json",
                   # "options": {
                   #     "temperature": 0,"seed": 123，"top_k": 20,"top_p": 0.9, "stop": ["\n", "user:"],
                   #  "num_batch": 2,"num_gpu": 1,"main_gpu": 0,"use_mmap": true,"num_thread": 8
                   # },
                   }

    payload.update(**kwargs)

    if stream:
        return post_aiohttp_stream(url, payload, time_out)
        # content=''
        # async for chunk in post_aiohttp_stream(url,payload):
        #     content += chunk.get("response", "")
        # return content
    timeout = aiohttp.ClientTimeout(total=time_out, sock_read=time_out, sock_connect=10.0)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                raise Exception(f"Ollama Error: {response.status}: {await response.text()}")
            return await response.json()


async def openai_ollama_chat_completions(messages: list, model_name="qwen3:14b", host='localhost', stream=False,
                                         time_out=300, **kwargs):
    """
      模拟 OpenAI 的返回结构
      - 如果 stream=True，模拟流式返回
      - 如果 stream=False，返回完整 JSON 响应
      ['model', 'created_at', 'response', 'done', 'done_reason', 'context', 'total_duration', 'load_duration', 'prompt_eval_count', 'prompt_eval_duration', 'eval_count', 'eval_duration']
    """

    prompt = build_prompt(messages, use_role=True) + "\nAssistant:"
    response = await call_ollama(prompt, messages=messages, model_name=model_name, host=host, time_out=time_out,
                                 stream=stream, **kwargs)
    tokenizer = get_tokenizer(Config.DEFAULT_MODEL_ENCODING)
    if stream:
        async def stream_data():
            async for chunk in response:
                content = chunk.get("response", chunk.get('message', {}).get('content', ''))
                data = {"id": "chatcmpl-xyz", "object": "chat.completion.chunk",
                        "created": int(time.time()), "model": chunk.get('model', model_name),
                        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]},
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"  # (f"data: {data}\n\n").encode("utf-8")

        return stream_data()

    # print(json.dumps(response, indent=4))  # 打印响应数据
    content = response.get("response", response.get('message', {}).get('content', ''))
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),  # unix_timestamp(response["created_at"])
        "model": response.get('model', model_name),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": response.get('done_reason', "stop"),
            }
        ],
        "usage": {
            "prompt_tokens": lang_token_size(prompt, tokenizer),
            "completion_tokens": lang_token_size(content, tokenizer),  # .split()
            "total_tokens": lang_token_size(prompt, tokenizer) + lang_token_size(content, tokenizer),
        },
    }


# retriever
async def retrieved_reference(keywords: List[Union[str, Tuple[str, ...]]] = None,
                              tool_calls: List[Callable[[...], Any]] = None):
    """
      根据用户请求和关键字调用多个工具函数，并返回处理结果。

      :param keywords: 关键字列表，可以是字符串或元组（函数名, 参数）。
      :param tool_calls: 工具函数列表。
      :return: 处理结果的扁平化列表。
    """

    # docs = retriever(question)
    # Assume this is the document retrieved from RAG
    # function_call = Agent_Functions.get(agent, lambda *args, **kwargs: [])
    # refer = function_call(user_message, ...)
    async def wrap_sync(func, *args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)

    tool_calls = tool_calls or []
    keywords = keywords or []
    callables = {}
    # 多个keywords,用在registry list 的 user_calls,自定义func参数
    for item in keywords:
        if isinstance(item, tuple):  # (函数名,无参,单个参数,多个位置参数列表,关键字参数字典)
            try:
                tool_name = item[0]
                _func = global_function_registry(tool_name)  # user_calls 函数
                if _func:
                    func_args = item[1] if len(item) > 1 else []
                    func_kwargs = item[2] if len(item) > 2 else {}
                    callable_key = generate_hash_key(id(_func), func_args, **func_kwargs)
                    bound_func = partial(_func, *func_args, **func_kwargs)
                    callables[callable_key] = bound_func
                    # callables.append((_func, func_args, func_kwargs)))
            except TypeError as e:
                logging.error(f"类型错误: {e},{item}")
            except Exception as e:
                logging.error(f"其他错误: {e},{item}")

    for bound_func in filter(callable, tool_calls):
        if isinstance(bound_func, partial):
            callable_key = generate_hash_key(id(bound_func), bound_func.args, **bound_func.keywords)
            callables[callable_key] = bound_func
            logging.info(f"{bound_func.func.__name__} with args: {bound_func.args}, kwargs: {bound_func.keywords}")
        else:
            func_name = getattr(bound_func, "__name__", type(bound_func).__name__)
            logging.warning(f"{func_name}(not partial)")

    # print(callables)
    tasks = []  # asyncio.create_task(),创建任务对象并将其加入任务列表
    for _key, bound_func in callables.items():
        if inspect.iscoroutinefunction(bound_func.func):
            tasks.append(bound_func())  # 添加异步函数任务，同时传递kwargs
        else:
            tasks.append(wrap_sync(bound_func))  # (*bound_func.args, **bound_func.keywords)

    refer = await asyncio.gather(*tasks, return_exceptions=True)  # gather 收集所有异步调用的结果

    err_count = 0
    for t, r in zip(tasks, refer):
        if isinstance(r, Exception):
            logging.error(f"Task {t.__name__} failed with error: {r}")
            err_count += 1
        elif not r:
            logging.warning(f"Task returned empty result: {t}")

    if err_count:
        logging.error(callables, refer)
    # 展平嵌套结果,(result.items() if isinstance(result, dict) else result)
    return [item for result in refer if not isinstance(result, Exception)
            for item in (result if isinstance(result, list) else [result])]


# Callable[[参数类型], 返回类型]
async def get_chat_payload(messages: list[dict] = None, user_request: str = '', system: str = '',
                           temperature: float = 0.4, top_p: float = 0.8, max_tokens: int = 1024,
                           model_name=Config.DEFAULT_MODEL, model_id=0,
                           agent: str = None, tools: List[dict] = None,
                           keywords: List[Union[str, Tuple[str, ...]]] = None, images: List[str] = None,
                           thinking: int = 0, **kwargs):
    model_info, name = find_ai_model(model_name, model_id, 'model')
    model_type = model_info['type']
    if messages is None:
        messages = []

    if isinstance(messages, list) and messages:
        if model_type in ('baidu', 'tencent'):
            # system_message = next((msg for msg in messages if msg["role"] == "system"), None)
            if messages[0].get('role') == 'system':  # ['system,assistant,user,tool,function']
                system = messages[0].get('content')
                del messages[0]

            # the role of first message must be user
            if messages[0].get('role') != 'user':  # user（tool）
                messages.insert(0, {'role': 'user', 'content': user_request or '请问您有什么问题？'})

            # 确保 user 和 assistant 消息交替出现
            messages = alternate_chat_history(messages)

        for message in messages:
            if 'name' in message:
                if (model_info['name'] == 'mistral' or
                        not isinstance(message["name"], str) or
                        not message["name"].strip()):
                    del message['name']

        if model_type != 'baidu' and system:
            if not any(msg.get('role') == 'system' for msg in messages):
                messages.insert(0, {"role": "system", "content": system})
            # messages[0].get('role') != 'system':
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
        if model_type != 'baidu' and system:  # system_message
            messages = [{"role": "system", "content": system}]
        messages.append({'role': 'user', 'content': user_request})

    tool_callable: List[Callable[[...], Any]] = agent_func_calls(agent, messages, tools, keywords, user_request)

    refer = await retrieved_reference(keywords, tool_callable)
    if refer:
        # """Answer the users question using only the provided information below:{docs}""".format(docs=formatted_refer)
        if model_type != 'baidu':
            formatted_refer = [{"type": "text", "text": str(text)} for text in refer]
            messages.append({"role": "system", "content": formatted_refer})
        else:
            formatted_refer = '\n'.join(map(str, refer))  # 百度模型对结构化 role 不敏感
            messages[-1]['content'] = (f'以下是相关检索结果或函数调用信息:\n{formatted_refer}\n'
                                       f'请结合以上参考内容或根据上下文，针对下面的问题进行解答：\n{user_request}')

    if images:  # 图片内容理解,(str, list[dict[str, str | Any]]))
        image_data = [{"type": "image_url", "image_url": {"url": image}} for image in images]
        if isinstance(messages[-1]['content'], list):  # text-prompt 请详细描述一下这几张图片。这是哪里？
            messages[-1]['content'].extend(image_data)
        else:
            messages[-1]['content'] = [{"type": "text", "text": user_request}] + image_data

    # print(messages)
    payload = dict(
        model=name,  # 默认选择第一个模型
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        # top_k=50,
        max_tokens=max_tokens,
        stream=False,
        # extra_body = {"prefix": "```python\n", "suffix":"后缀内容"} 希望的前缀内容,基于用户提供的前缀信息来补全其余的内容
        # response_format={"type": "json_object"}
    )
    payload.update(kwargs)

    if tools:
        filter_tools: list[dict] = deduplicate_functions_by_name(tools)
        if any(filter_tools):  # 一旦你传了 tools 字段，它 必须 至少包含一个合法的 tool
            payload['tools'] = filter_tools

    if model_type == 'baidu':
        payload['system'] = system

    if name in ('o3-mini', 'o4-mini', 'openai/o3', "openai/o1", 'openai/o1-mini', 'openai/o1-pro',
                'openai/o3-mini', 'openai/o3-pro', 'openai/o4-mini'):
        payload['max_completion_tokens'] = payload.pop('max_tokens', max_tokens)
        payload.pop('top_p', None)  # https://www.cnblogs.com/xiaoxi666/p/18827733
    if name in ('gpt-5', 'gpt-5-mini', 'gpt-5-nano',
                'openai/gpt-5', 'openai/gpt-5-chat', 'openai/gpt-5-mini', 'openai/gpt-5-nano'):
        for param in ['top_p', 'presence_penalty', 'frequency_penalty']:
            payload.pop(param, None)

    extra_body = payload.get("extra_body")
    if not isinstance(extra_body, dict):
        extra_body = {}
    enable_thinking = extra_body.get('enable_thinking', thinking > 0)
    thinking_budget = max(0, thinking) or extra_body.get('thinking_budget', 0)
    # https://help.aliyun.com/zh/model-studio/deep-thinking?spm=a2c4g.11186623.0.0.31a44823XJPxi9#e7c0002fe4meu
    if name in ('qwen3-32b', "qwen3-14b", "qwen3-8b", "qwen3-235b-a22b", "qwq-32b", "qwen-turbo", "qwen-plus",
                "qwen-plus-2025-04-28", "deepseek-v3.1"):
        # 开启深度思考,参数开启思考过程，该参数对 qwen3-30b-a3b-thinking-2507、qwen3-235b-a22b-thinking-2507、QwQ 模型无效
        extra_body["enable_thinking"] = enable_thinking
        if thinking_budget > 0:
            extra_body["thinking_budget"] = thinking_budget
    if name in ('deepseek-v3-1-terminus', "deepseek-v3-1-250821", "doubao-seed-1-6-250615"):
        extra_body["thinking"] = {"type": "enabled" if enable_thinking else "disabled"}  # /"auto"
        if 'max_completion_tokens' in payload:
            payload.pop('max_tokens', None)  # 不可与 max_tokens 字段同时设置，会直接报错
        else:
            payload['max_completion_tokens'] = payload.pop('max_tokens', max_tokens) + thinking_budget  # 最大输出长度
    if name in ("deepseek-reason",):
        extra_body["thinking_budget"] = thinking_budget or 2048
    if name in ("qwen-mt-plus", "qwen-mt-turbo"):
        extra_body["translation_options"] = {"source_lang": "auto", "target_lang": "English"}

    if extra_body:
        if isinstance(payload.get("extra_body"), dict):
            payload["extra_body"].update(extra_body)
        else:
            payload["extra_body"] = extra_body

    # payload['response_format'] = {"type": "json_object"}

    # 1. 系统设定（system）
    # 2. 原始提问（user）
    # 3. 检索结果 or 函数调用信息（system / user）
    # 4. 最终 assistant 回复

    return model_info, payload, refer


def get_chat_payload_post(model_info: Dict[str, Any], payload: dict):
    # 通过 requests 库直接发起 HTTP POST 请求
    url = model_info['url'] if model_info.get('url') else model_info['base_url'] + '/chat/completions'
    api_key = model_info['api_key']
    headers = {'Content-Type': 'application/json', }
    payload = payload.copy()
    if api_key:
        if isinstance(api_key, list):
            idx = next((i for i, m in enumerate(model_info['model']) if m == payload["model"]), 0)
            # model_info['model'].index(payload["model"])
            if api_key[idx]:
                headers["Authorization"] = f'Bearer {api_key[idx]}'
        else:
            headers["Authorization"] = f'Bearer {api_key}'

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

    if model_info['type'] == 'anthropic':
        headers["x-api-key"] = api_key
        headers.pop("Authorization", None)
        headers["anthropic-version"] = '2023-06-01'

    # if model_info['name'] == 'silicon':
    #     headers = {
    #         "accept": "application/json",
    #         "content-type": "application/json",
    #         "authorization": "Bearer sk-tokens"
    #     }
    return url, headers, payload


async def get_generate_payload(prompt: str, user_request: str = '', suffix: str = None, stream=False,
                               temperature: float = 0.7, top_p: float = 1, max_tokens: int = 4096,
                               model_name=Config.DEFAULT_MODEL, model_id=0, agent: str = None,
                               keywords: List[Union[str, Tuple[str, ...]]] = None,
                               **kwargs):
    chat = False
    try:
        model_info, name = find_ai_model(model_name, model_id, search_field="generation")
    except:
        model_info, name = find_ai_model(model_name, model_id)
        chat = True

    if chat or not name:
        model_info, payload, refer = await get_chat_payload(messages=None, user_request=user_request, system=prompt,
                                                            temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                                                            model_name=model_name, model_id=model_id,
                                                            agent=agent, tools=None, keywords=keywords, thinking=0,
                                                            **kwargs)

    else:
        tool_callable: List[Callable[[...], Any]] = agent_func_calls(agent, keywords=keywords, prompt=user_request)
        # 检索参考资料
        refer = await retrieved_reference(keywords, tool_callable)
        # 如果检索到内容，就把它拼到 user_request 里
        if refer:
            formatted_refer = "\n".join(map(str, refer))
            user_request = f"参考材料:\n{formatted_refer}\n材料仅供参考,请回答下面的问题: {user_request}"

        if user_request:
            prompt += '\n\n' + user_request

        payload = dict(model=name,
                       prompt=prompt,
                       suffix=suffix,
                       max_tokens=max_tokens,
                       temperature=temperature,
                       top_p=top_p,
                       stream=stream,
                       stop=kwargs.get("stop", None),
                       )

    return model_info, payload, refer


# 生成:conversation or summary,Fill-In-the-Middle
async def ai_generate(model_info: Optional[Dict[str, Any]], payload: dict = None, get_content=True,
                      **kwargs):
    '''
    Completions足以解决几乎任何语言处理任务，包括内容生成、摘要、语义搜索、主题标记、情感分析等等。
    需要注意的一点限制是，对于大多数模型，单个API请求只能在提示和完成之间处理最多4096个标记。
    '''
    if not payload:
        model_info, payload, _ = await get_generate_payload(**kwargs)
    else:
        payload.update(kwargs)  # 修改更新 payload

    stream = payload.get("stream", False)
    if 'messages' in payload:
        if stream:
            async def stream_data():
                async for content, data in ai_chat_stream(model_info, payload):
                    yield content if get_content else data

            return stream_data()
        else:
            return await ai_chat(model_info, payload, get_content)

    if model_info['name'] == 'qwen' and not stream:
        response = dashscope.Generation.call(model=payload['model'], prompt=payload['prompt'])
        return response.output.text if get_content else response.model_dump()

    client = AI_Client.get(model_info['name'], None)
    if not client:
        raise ValueError(f"Client for model {model_info['name']} not found")

    # <|fim_prefix|>{prefix_content}<|fim_suffix|>
    # <|fim_prefix|>{prefix_content}<|fim_suffix|>{suffix_content}<|fim_middle|>
    response = await client.completions.create(**payload)

    if stream:
        async def stream_data():
            async for chunk in response:
                yield chunk.choices[0].text if get_content else chunk.model_dump_json()

        return stream_data()  # async generator 对象

    return response.choices[0].text.strip() if get_content else response.model_dump()


async def ai_try(client, payload: dict, e: Exception, get_content: bool = True):
    """
        封装 BadRequestError 的错误信息解析和修正处理逻辑。
        # 尝试解析结构化 JSON 错误信息
        # This model only supports streaming responses. Please set stream=True.
        # Rate limit reached for TPM
        # Model disabled
        # Unsupported model
        # Invalid URL
        # does not exist or you do not have access to it
        # Authenticaton failed, please make sure that a valid ModelScope token is supplied.
        # You exceeded your current quota, please check your plan and billing details.
        # parameter.enable_thinking must be set to false for non-streaming calls
        # openai.BadRequestError: Error code: 400 - {'error': {'code': 'invalid_parameter_error', 'param': None, 'message': 'This model only support stream mode, please enable the stream parameter to access the model. ', 'type': 'invalid_request_error'}, 'id': 'chatcmpl-4c7a91c5-c35c-9e94-8ff8-6e1b09a8a151', 'request_id': '4c7a91c5-c35c-9e94-8ff8-6e1b09a8a151'}
        # openai.BadRequestError: Error code: 400 - {'error': {'code': 'invalid_parameter_error', 'param': None, 'message': '<400> InternalError.Algo.InvalidParameter: Range of input length should be [1, 3072]', 'type': 'invalid_request_error'},
        # openai.BadRequestError: Error code: 400 - {'error': {'message': 'Invalid max_tokens value, the valid range of max_tokens is [1, 8192]', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_request_error'}}

       Args:
           e: 捕获的 BadRequestError 异常对象
           client:
           payload:模型参数
           get_content: 是否仅返回内容（否则返回完整结果）

       Returns:
           fallback_callable 的执行结果（内容或完整响应）

       Raises:
           原始异常或 ValueError（当输入长度过大）
       """
    try:
        if hasattr(e, "response") and hasattr(e.response, "json"):
            error_json = e.response.json()
        elif hasattr(e, "response") and hasattr(e.response, "text"):
            error_json = json.loads(e.response.text)
        else:
            error_json = {}

        error_message = error_json.get("error", {}).get("message", "") or str(e)
    except Exception:
        error_message = str(e)

    logging.error(f"[BadRequestError] 捕获错误消息：{error_message}")

    stream_required_phrases = [
        "only support stream mode",
        "only supports streaming",
        "non-streaming calls",  # +/no_think
        "stream=true",
        "模型不支持sync调用"
    ]
    length_required_phrases = [
        "range of input length",
        "input length should be",
        "max input characters",
        "invalid max_tokens value",
        "the max_tokens must be less than"
    ]
    not_exists_phrases = ['model not exists', 'invalid model id']

    error_msg = error_message.lower()

    if any(p in error_msg for p in stream_required_phrases):
        payload["extra_body"] = {"enable_thinking": True}
        try:
            last_value = None
            async for chunk in stream_chat_completion(client, payload):
                last_value = chunk
            if last_value is None:
                raise RuntimeError("[stream_chat_completion] No data received from stream.")
            _, completion = last_value
            completion = json.loads(completion)
            content = completion.get('choices', [{}])[0].get('message', {}).get('content')
            await BaseReBot.async_save(model=payload.get('model'), messages=payload.get('messages', []),
                                       assistant_content=content, model_response=completion, dbpool=DB_Client,
                                       user='local', agent='chat', robot_id='stream_chat')
        except Exception as e:
            raise RuntimeError(f"[stream_chat_completion] Failed to complete streaming call: {e}") from e
        return content if get_content else completion
    elif any(p in error_msg for p in length_required_phrases):
        raise ValueError(f"[InputTokenError] 输入超过模型限制或无效：{error_message}")
        # payload['message']=cut_chat_history(payload['message'], max_size=33000)
    elif any(p in error_msg for p in not_exists_phrases):
        raise ValueError(f"[InputModelError] 模型无效<{payload.get('model')}>：{error_message}")
    else:
        # 内部 raise 会继续被下一个 except 捕获
        raise e


async def ai_chat(model_info: Optional[Dict[str, Any]], payload: dict = None, get_content: bool = True,
                  **kwargs) -> Union[str, Dict]:
    """
    模拟发送请求给AI模型并接收响应。
    :param model_info: 模型信息（如名称、ID、配置等）
    :param payload: 请求的负载，通常是对话或文本输入
    :param get_content: 返回模型响应类型
    :param kwargs: 其他额外参数
    :return: 返回模型的响应
    """
    if not payload:
        model_info, payload, _ = await get_chat_payload(**kwargs)
    elif not model_info:
        model_info, new_payload, _ = await get_chat_payload(**kwargs)
        payload = {**(payload or {}), **new_payload}  # 合并： {**payload, **kwargs}
    else:
        payload.update(kwargs)  # 修改更新 payload

    fake_response = {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": payload.get('model', 'unknown'),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": 'null',
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }
    client = AI_Client.get(model_info['name'], None)
    if client:
        try:
            # await asyncio.to_thread(client.chat.completions.create, **payload)
            # await asyncio.wait_for(client.chat.completions.create, **payload)
            completion = await client.chat.completions.create(**payload)
            if completion is None:
                raise ValueError("OpenAI API returned None instead of a valid response")
            if not hasattr(completion, "choices") or not completion.choices:
                raise ValueError(f"Unexpected API response, missing choices: {completion}")
            first_choice = completion.choices[0]
            if not hasattr(first_choice, "message") or not hasattr(first_choice.message, "content"):
                raise ValueError(f"Unexpected API response, missing message: {completion}")
            return first_choice.message.content if get_content else completion.model_dump()  # 自动序列化为 JSON
            # content = getattr(first_choice.message, "content", None)
            # json.loads(completion.model_dump_json())

        except BadRequestError as e:
            try:
                return await ai_try(client, payload, e, get_content)
            except:
                raise
        except Exception as e:
            logging.error(f"OpenAI error:{e}, {payload}")
            error_message = f"OpenAI error occurred: {e}"
            if get_content:
                return error_message
            fake_response["choices"][0]["message"]["content"] = error_message
            return fake_response

    url, headers, payload = get_chat_payload_post(model_info, payload)
    # print(headers, payload)
    parse_rules = {
        'baidu': lambda d: d.get('result'),
        'tencent': lambda d: d.get('Response', {}).get('Choices', [{}])[0].get('Message', {}).get('Content'),
        # d.get('Choices')[0].get('Message').get('Content')
        'default': lambda d: d.get('choices', [{}])[0].get('message', {}).get('content')
    }
    cx = get_httpx_client(proxy=Config.HTTP_Proxy if model_info.get('proxy') else None)
    try:
        response = await cx.post(url, headers=headers, json=payload, timeout=Config.LLM_TIMEOUT_SEC)
        response.raise_for_status()  # 如果请求失败，则抛出异常
        data = response.json()
        if get_content:
            result = parse_rules.get(model_info['type'], parse_rules['default'])(data)
            if result:
                return result
            print(response.text)

        if model_info['type'] == 'tencent':
            return convert_keys_to_lower_case(data)

        if model_info['type'] == 'baidu':
            content = data.pop('result', data.get("error_msg", ""))
            data.update({
                "model": payload.get('model', 'unknown'),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content, },
                        "finish_reason": "stop",
                    }
                ],
            })
            return data

        return data

    except Exception as e:
        # print(response.text)
        error_message = f"HTTP error occurred: {e}"
        if get_content:
            return error_message
        fake_response["choices"][0]["message"]["content"] = error_message
        return fake_response


async def stream_chat_completion(client, payload: dict):
    payload["stream"] = True
    set_usage = payload.get("stream_options", {}).get("include_usage", False)
    if not set_usage:
        if not isinstance(payload.get("stream_options"), dict):
            payload["stream_options"] = {}
        payload["stream_options"]["include_usage"] = True  # 可选，配置以后会在流式输出的最后一行展示token使用信息

    # print(payload, client)
    try:
        stream = await client.chat.completions.create(**payload)
        if not stream:
            raise ValueError("OpenAI API returned an empty response")
        if not hasattr(stream, "__aiter__"):
            raise TypeError("OpenAI API returned a non-streaming response")

        reasoning_content = ''
        assistant_content = []  # answer_content
        completion_chunk = None
        async for chunk in stream:
            if not chunk:
                continue

            completion_chunk = chunk
            # 若 choices 为空但包含 usage，说明是 stream 末尾数据,qwen
            if not chunk.choices:
                if set_usage:
                    yield None, chunk.model_dump_json()  # '[DONE]' usage
                if assistant_content:
                    break
                else:
                    continue

            delta = chunk.choices[0].delta
            if delta.content:  # 以两个换行符 \n\n 结束当前传输的数据块
                assistant_content.append(delta.content)
                yield delta.content, chunk.model_dump_json()  # 获取字节流数据

            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                reasoning_content += delta.reasoning_content
                yield delta.reasoning_content, chunk.model_dump_json()

        if not assistant_content and not reasoning_content:
            raise ValueError("OpenAI API returned an empty stream response")
        if completion_chunk:
            completion = completion_chunk.model_dump()
            completion.update({
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": ''.join(assistant_content),
                                    "reasoning_content": reasoning_content},
                        "finish_reason": "stop",
                        # "metadata": {"reasoning": reasoning}  # .split("</think>")[-1]
                    }
                ],
            })
            yield None, json.dumps(completion)

    except Exception as e:
        logging.error(f"OpenAI error:{e}, {payload}")
        error_message = f"OpenAI error occurred: {e}"
        fake_response = {
            "id": f"chatcmpl-{int(time.time())}", "object": "chat.completion.chunk",
            "created": int(time.time()), "model": payload.get("model", "unknown"),
            "choices": [{"index": 0, "delta": {"content": error_message}, "finish_reason": "stop"}],
        }
        yield error_message, json.dumps(fake_response)


async def ai_chat_stream(model_info: Optional[Dict[str, Any]], payload=None, **kwargs):
    if not payload:
        model_info, payload, _ = await get_chat_payload(**kwargs)
    else:
        # payload=copy(payload)
        payload.update(kwargs)

    client = AI_Client.get(model_info['name'], None)
    if client:
        async for item in stream_chat_completion(client, payload):
            yield item

        # with client.chat.completions.with_streaming_response.create(**payload) as response:
        #     print(response.headers.get("X-My-Header"))
        #     for line in response.iter_lines():
        #         yield line

        return  # 异步生成器的结束无需返回值

    fake_response = {
        "id": f"chatcmpl-{int(time.time())}", "object": "chat.completion.chunk",
        "created": int(time.time()), "model": payload.get("model", "unknown"),
        "choices": [{"index": 0, "delta": {"content": 'null'}, "finish_reason": "stop"}],
    }
    url, headers, payload = get_chat_payload_post(model_info, payload)
    payload["stream"] = True
    cx = get_httpx_client(proxy=Config.HTTP_Proxy if model_info.get('proxy') else None)
    model_type = model_info['type']
    async for item in post_httpx_sse(url, payload, headers, Config.LLM_TIMEOUT_SEC, cx):
        data_type = item.get("type")
        if data_type == 'done':
            break

        data = item.get("data", None)
        if not data:
            continue

        if data_type != "data":
            # text / error 直接返回原始字符串 yield
            content = f"OpenAI error occurred: {data}" if data_type == "error" else data
            fake_response["choices"][0]["delta"]["content"] = content
            yield data, json.dumps(fake_response)
            print(item)
            continue

        if model_type == 'baidu':
            if data.get('is_end') is True:
                break
            content = data.get("result") or data.get("error_msg")
            fake_response["choices"][0]["delta"]["content"] = content
            yield content, json.dumps(fake_response)
        elif model_type == 'tencent':
            reason = data.get('Choices', [{}])[0].get('FinishReason')
            if reason == "stop":
                # raise StopIteration(2)
                break
            data = convert_keys_to_lower_case(data)

        choices = data.get('choices', [])  # 通用逻辑：处理 choices -> delta -> content
        if choices:
            delta = choices[0].get('delta', {})
            yield delta.get("content", ""), json.dumps(data)
            # print(delta.get("content", ""), end="", flush=True)

    # yield "[DONE]"


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


Assistant_Cache = {}


# https://platform.baichuan-ai.com/docs/assistants
async def ai_assistant_create(instructions: str, user_name: str,
                              tools_type: Literal['web_search', 'code_interpreter', 'function'] = "code_interpreter",
                              model_id=4):
    # 如果您判断是一次连续的对话，则无需自己拼接上下文，只需将最新的 message 添加到对应的 thread id 即可得到包含了 thread id 历史上下文的回复，历史上下文超过我们会帮您自动截断。
    global Assistant_Cache
    cache_key = generate_hash_key(instructions, user_name, tools_type, model_id)
    if cache_key in Assistant_Cache:
        return Assistant_Cache[cache_key]

    model_info, model_name = find_ai_model('baichuan', model_id, 'model')
    assistants_url = model_info.get('assistants_url', f"{model_info['base_url']}assistants")
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
    cx = get_httpx_client(time_out=Config.HTTP_TIMEOUT_SEC)
    try:
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
    headers: dict = assistant.get('headers', {})
    messages_url = f'{assistant["threads_url"]}/{assistant["threads_id"]}/messages'
    threads_url = f'{assistant["threads_url"]}/{assistant["threads_id"]}/runs'

    cx = get_httpx_client(time_out=Config.HTTP_TIMEOUT_SEC)
    messages_response = await cx.post(messages_url, headers=headers, json={"role": "user", "content": user_request})
    messages_response.raise_for_status()
    run_response = await cx.post(threads_url, headers=headers, json={"assistant_id": assistant_data['id']})
    run_response.raise_for_status()
    run_id = run_response.json().get('id')
    run_url = f'{threads_url}/{run_id}'
    retries = 0
    current_interval = interval
    while retries < max_retries:
        try:
            run_status_response = await cx.get(run_url, headers=headers)  # 定期检索 run id 的运行状态
            # print(retries,run_status_response.status_code, run_status_response.headers, run_status_response.text)
            if run_status_response.status_code == 429:
                await asyncio.sleep(current_interval)
                retries += 1
                current_interval *= backoff_factor
                continue

            run_status_response.raise_for_status()
            status_data = run_status_response.json()

            if status_data.get('status') == 'completed':
                messages_status_response = await cx.get(messages_url, headers=headers)
                run_status_response.raise_for_status()
                status_data = messages_status_response.json()
                # print(status_data)
                return status_data.get('data')

            await asyncio.sleep(current_interval)
            retries += 1

        except httpx.HTTPStatusError as http_error:
            return {"error": f"HTTP error: {str(http_error)}"}  # status_data['error']

        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}

    return {"error": "Max retries reached, task not completed"}


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


async def similarity_score_embeddings(query, tokens: List[str], filter_idx: List[int] = None,
                                      tokens_vector: List[float] = None,
                                      embeddings_calls: Callable[[...], Any] = ai_embeddings, **kwargs):
    """
    计算查询与一组标记之间的相似度分数。
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


async def get_similar_embeddings(querys: Union[str, List[str]], tokens: List[str], topn: int = 10,
                                 embeddings_calls: Callable[[...], Any] = ai_embeddings,
                                 **kwargs) -> List[Tuple[str, List[Tuple[str, float, int]]]]:
    """
    使用嵌入计算查询与标记之间的相似度。
    :param querys: 查询词列表
    :param tokens: 比较词列表
    :param embeddings_calls: 嵌入生成函数
    :param topn: 返回相似结果的数量
    :param kwargs: 其他参数
    返回：
        List[Tuple[str, List[Tuple[str, float,int]]]]: 查询词与相似标记及其分数的映射。
    """
    if isinstance(querys, str):
        querys = [querys]

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


def get_similar_words(querys: List[str], data: dict, exclude: List[str] = None, topn: int = 10,
                      cutoff: float = 0.0) -> List[Tuple[str, List[Tuple[str, float]]]]:
    '''
   计算查询词与数据中的词之间的相似度，并返回每个查询词的最相似词及其相似度。

    :param querys: 查询词列表
    :param data: 包含词汇和对应向量的字典，格式为：
                 {
                     "name": ["word1", "word2", ...],
                     "vectors": [[0.1, 0.2, ...], [0.3, 0.4, ...], ...]
                 }
    :param exclude: 要排除的词列表，这些词不会出现在结果中
    :param topn: 返回最相似的前 n 个词
    :param cutoff: 只有相似度大于该值的词才会被返回
    :return: 每个查询词的相似词和相似度的元组列表
    返回：
        List[Tuple[str, List[Tuple[str, float]]]]: 查询词与相似标记及其分数的映射。
    '''
    index_to_key = data['name']
    vectors = np.array(data['vectors'])

    # 获取索引
    query_mask = np.array([w in index_to_key for w in querys])
    exclude_mask = np.array([w in querys + (exclude or []) for w in index_to_key])
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


async def find_closest_matches_embeddings(querys: List[str], tokens: List[str],
                                          embeddings_calls: Callable[[...], Any] = ai_embeddings,
                                          **kwargs) -> Dict[str, Tuple[str, float]]:
    """
    使用嵌入计算查询与标记之间的最近匹配，找到每个查询的最佳匹配标记,topk=1。

    :param querys: 查询字符串列表。
    :param tokens: 标记列表，用于匹配查询。
    :param embeddings_calls: 嵌入生成函数，接收查询和标记返回嵌入向量（默认为 ai_embeddings）。
    :param kwargs: 其他参数，传递给 embeddings_calls 函数。
    :return: 返回一个字典，其中每个查询的最佳匹配标记及其相似度分数被映射到查询上。
    返回：
        Dict[str, Tuple[str, float]]: 查询与最近匹配标记的映射字典。
    """
    matches = {x: (x, 1.0) for x in querys if x in tokens}
    unmatched_queries = list(set(querys) - matches.keys())
    # 所有查询都已匹配，直接返回完全匹配项
    if not unmatched_queries:
        return matches
    query_vector, tokens_vector = await asyncio.gather(
        embeddings_calls(unmatched_queries, **kwargs),
        embeddings_calls(tokens, **kwargs))
    sim_matrix = np.array(query_vector) @ np.array(tokens_vector).T

    closest_matches = tokens[sim_matrix.argmax(axis=1)]  # idxmax
    closest_scores = sim_matrix.max(axis=1)
    matches.update(zip(unmatched_queries, zip(closest_matches, closest_scores)))
    return matches


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
        source = lang_detect_to_trans(text)
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

    # model_map = {"baidu": 'ernie', "tencent": "hunyuan", "xunfei": 'spark'}
    # if model_name in model_map.keys():
    #     model_name = model_map[model_name]
    if not model_name:
        model_name = 'qwen'

    translated_text = await ai_generate(
        model_info=None,
        prompt=system_prompt,
        user_request=text,
        model_name=model_name,
        stream=False,
    )
    if translated_text:
        return {"translated": translated_text, 'from': source, 'to': target, "model": model_name}

    return {"translated": error, 'from': source, 'to': target, "model": model_name}


async def siliconflow_generate_image(prompt: str = '', negative_prompt: str = '', model_name='siliconflow', model_id=0):
    model_info, name = find_ai_model(model_name, model_id, 'image')

    url = model_info.get('image_url') or "https://api.siliconflow.cn/v1/images/generations"
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
    cx = get_httpx_client()
    response = await cx.post(url, headers=headers, json=payload, timeout=Config.LLM_TIMEOUT_SEC)
    response.raise_for_status()
    if response.status_code != 200:
        return None, {'error': f'{response.status_code},Request failed,Response content: {response.text}'}

    response_data = response.json()
    return None, {"urls": [i['url'] for i in response_data["images"]], 'id': response_data["seed"]}


@mcp.tool
async def generate_summary(content: str, ctx: MCPContext) -> str:
    """Generate a summary of the provided content."""
    prompt = f"Please provide a concise summary of the following content:\n\n{content}"

    response = await ctx.sample(prompt)
    return response.text


@mcp.tool
async def generate_code_example(concept: str, ctx: MCPContext) -> str:
    """Generate a Python code example for a given concept."""
    response = await ctx.sample(
        messages=f"Write a simple Python code example demonstrating '{concept}'.",
        system_prompt="You are an expert Python programmer. Provide concise, working code examples without explanations.",
        temperature=0.7,
        max_tokens=300
    )

    code_example = response.text
    return f"```python\n{code_example}\n```"


@mcp.tool
async def technical_analysis(data: str, ctx: MCPContext) -> str:
    """Perform technical analysis with a reasoning-focused model."""
    response = await ctx.sample(
        messages=f"Analyze this technical data and provide insights: {data}",
        model_preferences=["claude-3-opus", "gpt-4"],  # Prefer reasoning models
        temperature=0.2,  # Low randomness for consistency
        max_tokens=800
    )

    return response.text


async def baidu_nlp(text: str | list[str] = None, nlp_type='ecnet', **kwargs):  # text': text
    '''
    nlp_type:
        address:地址识别,kw:text
        sentiment_classify:情感倾向分析
        emotion:对话情绪识别
        entity_analysis:实体分析,kw:text,mention
        simnet:短文本相似度,kw: text_1,text_2,
        ecnet,text_correction:文本纠错,搜索引擎、语音识别、内容审查等功能
        txt_keywords_extraction:关键词提取,kw:text[],num
        txt_monet:文本信息提取,content_list[{content,query_lis[{query}]}]
        sentiment_classify:
        depparser:依存句法分析,利用句子中词与词之间的依存关系来表示词语的句法结构信息
        lexer:词法分析,提供分词、词性标注、专名识别三大功能
        keyword:文章标签,kw:title,content
        topic:文章分类,kw:title,content
        news_summary:新闻摘要,title,content,max_summary_len
        titlepredictor,文章标题生成,kw:doc
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
    if text:
        body['text'] = text

    cx = get_httpx_client(time_out=Config.HTTP_TIMEOUT_SEC)
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


async def ali_nlp(text):
    # https://help.aliyun.com/zh/sdk/product-overview/v3-request-structure-and-signature?spm=a2c4g.11186623.0.0.38ee5703p01VbZ#section-mqj-l8f-ak0
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


def dashscope_file_upload(messages, file_path='.pdf', api_key=Config.DashScope_Service_Key):
    url = 'https://dashscope.aliyuncs.com/compatible-mode/v1/files'
    headers = {
        # 'Content-Type': 'application/json',
        "Authorization": f'Bearer {api_key}',
    }

    try:
        with open(file_path, 'rb') as f:
            files = {
                'file': f,
                'purpose': (None, 'file-extract'),
            }
            response = requests.post(url, headers=headers, files=files, timeout=Config.HTTP_TIMEOUT_SEC)
            response.raise_for_status()

            file_object = response.json()
            file_id = file_object.get('id', 'unknown_id')

            messages.append({"role": "system", "content": f"fileid://{file_id}"})
            return file_object, file_id

    except Exception as e:
        return {"error": str(e)}, None


async def download_file(url: str, dest_folder: Path = None, chunk_size=4096,
                        in_decode=False, in_session=False, retries=3, delay=3, time_out=Config.HTTP_TIMEOUT_SEC):
    """
    下载URL中的文件到目标文件夹
    :param url: 下载链接
    :param dest_folder: 保存文件的路径
    :param chunk_size: 每次读取的字节大小（默认4096字节）
    :param in_decode: 是否解码为字符串
    :param in_session: 是否使用 session（长连接优化）
    :param retries: 下载失败后的重试次数
    :param delay: 重试之间的等待时间（秒）
    :param time_out: 超时时间（秒）
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
                timeout = aiohttp.ClientTimeout(total=time_out) if time_out > 0 else None
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    return await download_by_aiohttp(url, session, save_path, chunk_size, in_decode), file_name
            else:  # httpx少量请求场景,适合简单的、单个请求场景
                timeout = httpx.Timeout(time_out) if time_out > 0 else None
                async with httpx.AsyncClient(timeout=timeout) as client:
                    return await download_by_httpx(url, client, save_path, chunk_size, in_decode), file_name

        except (httpx.RequestError, aiohttp.ClientError, asyncio.TimeoutError, requests.Timeout) as e:
            print(f"Download attempt {attempt + 1} failed: {e}")
            attempt += 1
            await asyncio.sleep(delay)  # 等待重试
        except httpx.HTTPStatusError as exc:
            print(f"Failed to download {url},HTTP error occurred: {exc.response.status_code} - {exc.response.text}")
        except TypeError:
            return download_by_requests(url, save_path, chunk_size, in_decode, time_out), file_name
        except Exception as e:
            print(f"Error: {e}, downloading url: {url} file: {file_name}")
            break

    return None, None


async def send_callback(callback_data: dict, result, **kwargs):
    url = callback_data.get("url")
    fallback_url = "http://127.0.0.1:7000/callback"
    if not url:
        print(f"[Missing callback URL]:{callback_data}")
        url = fallback_url

    payload = callback_data.get("payload", {})
    if not isinstance(payload, dict):
        payload = {}

    mapping = callback_data.get("mapping", {})

    def apply_mapping(data: dict):
        return {mapping.get(k, k): v for k, v in data.items()} if mapping else data

    def filter_payload(data: dict):
        return {mapping[k]: data[k] for k in mapping if k in data} if mapping else data

    if isinstance(result, dict):
        payload.update(result)
    elif isinstance(result, list) and len(result) == 1 and isinstance(result[0], dict):
        payload.update(result[0])
    else:
        payload = {**payload, "result": result}

    payload = apply_mapping(payload)
    headers = callback_data.get("headers") or {}
    params = callback_data.get("params")
    timeout = Config.HTTP_TIMEOUT_SEC
    if params and isinstance(params, dict):
        timeout = params.pop("timeout", timeout)
        kwargs.update(params)

    res = None
    format_type = callback_data.get("format", 'json').lower()  # "query" or "json" form"

    async with aiohttp.ClientSession() as session:
        if format_type == "json":  # post_http_json
            res = await aiohttp_request("POST", url, session, json=payload, headers=headers, timeout=timeout, **kwargs)
        elif format_type == "query":  # get_http_query query 参数或表单参数
            query_payload = filter_payload(payload)
            res = await aiohttp_request("GET", url, session, params=query_payload, headers=headers, timeout=timeout)
        elif format_type == "form":  # post_http_form 支持 query 或 form
            headers["Content-Type"] = "application/x-www-form-urlencoded"
            res = await aiohttp_request("POST", url, session, data=payload, headers=headers, timeout=timeout)

        if res and res.get('error'):
            return await aiohttp_request("POST", fallback_url, session, json=payload, headers=headers, **kwargs)
        return res


if __name__ == "__main__":
    AccessToken = 'd04149e455d44ac09432f0f89c3e0a41'

    import nest_asyncio

    nest_asyncio.apply()


    async def test():
        result = await web_search_async('易得融信是什么公司')
        print(result)
        result = await baidu_nlp(text="百度是一家人工只能公司", nlp_type="ecnet")
        print(result)

        result = await call_ollama('你好', model_name="qwen3:14b", host='10.168.1.10', stream=False)

        print(result)
        print(result.keys())
        result = await call_ollama('你好啊', model_name="qwen3:14b", host='10.168.1.10', stream=True)
        # print(result)
        async for chunk in result:
            content = chunk.get("response")
            print(content, end="", flush=True)

        result = await call_ollama('你好啊', messages=[{"role": "user", "content": "Hello!"}], model_name="qwen3:14b",
                                   host='10.168.1.10', stream=True)
        print(result)
        async for chunk in result:
            content = chunk.get("message", [])
            print(content["content"], end="", flush=True)

        result = await openai_ollama_chat_completions(messages=[{"role": "user", "content": "你好啊"}],
                                                      model_name="qwen3:14b",
                                                      host='10.168.1.10', stream=False)
        print(result)

    # asyncio.run(test())
