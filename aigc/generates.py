import httpx, aiohttp, aiofiles, asyncio
import oss2
from typing import List, Dict, Tuple, Any, Union, Callable, Literal, Iterator, Sequence, Awaitable, Generator, Optional, \
    get_origin, get_args

from openai import OpenAI, AsyncOpenAI, AsyncClient, DefaultHttpxClient, BadRequestError

from utils import *
from agents.ai_tools import *
from agents.ai_prompt import *
from agents.ai_vectors import *
from agents.ai_tasks import *
from agents.ai_search import *
from agents.ai_multi import *

AI_Client: Dict[str, Optional[AsyncOpenAI]] = {}  # OpenAI
QD_Client = AsyncQdrantClient(host=Config.QDRANT_HOST, grpc_port=Config.QDRANT_GRPC_PORT,
                              prefer_grpc=True) if Config.QDRANT_GRPC_PORT else AsyncQdrantClient(url=Config.QDRANT_URL)


async def get_data_for_model(model: dict):
    """获取每个模型的数据"""
    model_name = model.get('name')
    client = AI_Client.get(model_name)

    if client:
        try:
            models = await client.models.list()
            return [m.model_dump() for m in models.data]
        except Exception as e:
            print(f"OpenAI error occurred:{e},name:{model_name}")
    else:
        url = model['base_url'] + '/models'
        models = await call_http_request(url)
        if models:
            return models.get('data')

    return None


async def set_data_for_model(model: dict, redis=None):
    key = f"model_data_list:{model.get('name')}"
    data = await redis.get(key) if redis else None
    if data:
        model['data'] = json.loads(data)
        return

    data = await get_data_for_model(model)
    if data:
        model['data'] = data
        if redis:
            await redis.set(key, json.dumps(data, ensure_ascii=False))
        print('model:', model.get('name'), 'data:', data)


class ModelListExtract:
    models = []
    _redis = None

    @classmethod
    def extract(cls):
        """
        提取 AI_Models 中的 name 以及 search_field 中的所有值，并存入一个大列表。

        返回：
        - List[str]: 包含所有模型名称及其子模型的列表
        """
        # list(itertools.chain(*[sublist[1] for sublist in extract_ai_model("model")]))
        extracted_data = extract_ai_model("model")
        return [i for item in extracted_data for i in [item[0]] + item[1]]  # flattened_list

    @classmethod
    async def set(cls, redis=None):
        """更新 MODEL_LIST,并保存到 Redis"""
        cls.models = cls.extract()
        if cls._redis is None:
            cls._redis = redis or get_redis()
        if cls._redis:
            await cls._redis.set("model_list", json.dumps(cls.models, ensure_ascii=False))

    @classmethod
    async def get(cls):
        if cls._redis:
            data = await cls._redis.get("model_list")
            if data:
                return json.loads(data)
        if not cls.models:
            await cls.set()
            print("model_list updated:", cls.models)

        return cls.models

    @classmethod
    def contains(cls, value):
        models = cls.models or async_to_sync(cls.get)
        if ':' in value:
            owner, name = value.split(':')
            return owner in models or name in models

        return value in models


async def init_ai_clients(ai_models=AI_Models, get_data=False, redis=None):
    api_keys = model_api_keys()
    limits = httpx.Limits(max_connections=100, max_keepalive_connections=50)
    transport = httpx.AsyncHTTPTransport(proxy=Config.HTTP_Proxy)
    # proxies = {"http://": Config.HTTP_Proxy, "https://": Config.HTTP_Proxy}
    # http_client = DefaultHttpxClient(proxy="http://my.test.proxy.example.com", transport=httpx.HTTPTransport(local_address="0.0.0.0"))
    for model in ai_models:
        model_name = model.get('name')
        api_key = api_keys.get(model_name)
        if api_key:
            model['api_key'] = api_key
            if model_name not in AI_Client and model.get('supported_openai'):  # model_name in SUPPORTED_OPENAI_MODELS
                http_client = None
                time_out = model.get('timeout', Config.HTTP_TIMEOUT_SEC * 2)
                if model.get('proxy'):  # proxies=proxies
                    timeout = httpx.Timeout(time_out, read=time_out, write=60.0, connect=10.0)
                    http_client = httpx.AsyncClient(transport=transport, limits=limits, timeout=timeout)

                AI_Client[model_name]: AsyncOpenAI = AsyncOpenAI(api_key=api_key, base_url=model['base_url'],
                                                                 http_client=http_client)  # OpenAI
                if http_client is None:
                    AI_Client[model_name] = AI_Client[model_name].with_options(timeout=time_out, max_retries=3)

    if get_data:
        tasks = [set_data_for_model(model, redis=redis) for model in ai_models if
                 model.get('supported_list') and model.get('api_key')]
        await asyncio.gather(*tasks)

    await ModelListExtract.set(redis)
    # print(len(ModelListExtract.models))


# client = AI_Client['deepseek']
# print(dir(client.chat.completions))# 'create', 'with_raw_response', 'with_streaming_response'
# print(dir(client.completions))
# print(dir(client.embeddings))
# print(dir(client.files)) #'content', 'create', 'delete', 'list', 'retrieve', 'retrieve_content', 'wait_for_processing'


def find_ai_model(name, model_id: int = 0, search_field: str = 'model') -> Tuple[dict, str]:
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
    if ':' in name:
        owner, model_name = name.split(':')
        model = next((item for item in AI_Models if item['name'] == owner), None)
        if model and model_name in model.get(search_field, []):
            return model, model_name

    model = next(
        (item for item in AI_Models if item['name'] == name or name in item.get(search_field, [])),
        None
    )
    if model:
        model_items = model.get(search_field, [])

        if isinstance(model_items, (list, tuple)):
            if name in model_items:
                return model, name
            if model_items:
                model_id %= len(model_items)
                return model, model_items[model_id]
        elif isinstance(model_items, dict):
            if name in model_items:
                return model, model_items[name]
            # 如果提供了序号，返回序号对应的值
            keys = list(model_items.keys())
            model_id = model_id if abs(model_id) < len(keys) else 0
            return model, model_items[keys[model_id]]

        return model, name if model_items == name else ''

    raise ValueError(f"Model with name {name} not found.")
    # HTTPException(status_code=400, detail=f"Model with name {name} not found.")


async def ai_generate_metadata(function_code: str, metadata: dict = None, model_name=Config.DEFAULT_MODEL_METADATA,
                               description: str = None, code_type: str = "Python", **kwargs) -> dict:
    if not model_name:
        model_name = Config.DEFAULT_MODEL_METADATA

    chat = True
    try:
        model_info, name = find_ai_model(model_name)
    except:
        chat = False
        model_info, name = find_ai_model(model_name, search_field="generation")

    client = AI_Client.get(model_info['name'], None)
    if client:
        prompt = System_content.get('84').format(function_code=function_code, code_type=code_type)
        lines = [f"帮我根据函数代码生成提取函数元数据（JSON格式）。"]
        if metadata:
            lines.append(f"初始元数据结构如下:{json.dumps(metadata, ensure_ascii=False)}")
        if description:
            lines.append(f"工具描述为:{description}")
        prompt_user = "\n".join(lines)

        if chat:
            messages = [{"role": "system", "content": prompt},
                        {"role": "user", "content": prompt_user}]
            payload = dict(model=name,
                           messages=messages,
                           max_tokens=1000,
                           temperature=0.3,
                           stream=False)
            full_payload = {**payload, **kwargs}
            response = await client.chat.completions.create(**full_payload)
            content = response.choices[0].message.content.strip()
        else:
            response = await client.completions.create(
                model=name,
                prompt=prompt + '\n' + prompt_user,
                max_tokens=1000,
                temperature=0.3,
                stream=False,
                **kwargs,
            )
            content = response.choices[0].text.strip()

        match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
        if match:
            json_str = match.group(1)
            try:
                metadata = json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e},{content}")
        else:
            print(response)

    return metadata or {}


Function_MetaData_Store = {}  # cache_func_key:cached_metadata


async def get_metadata_from_cache(func: Callable, redis=None, key: str = "funcmeta"):
    # 获取函数的名称、参数以及docstring
    func_name = func.__name__
    function_code = remove_function_decorators(func)
    # extract_function_metadata(function_code)
    cache_key = generate_hash_key(func_name, function_code)  # 来生成缓存键

    try:
        if redis:
            cached_metadata = await redis.get(f"{key}:{cache_key}")
            await redis.expire(f"{key}:{cache_key}", Config.REDIS_CACHE_SEC)
        else:
            raise Exception
    except:
        cached_metadata = Function_MetaData_Store.get(cache_key, {})

    if cached_metadata:
        # print(f"Metadata already cached for function: {func_name}")
        metadata = json.loads(cached_metadata) if isinstance(cached_metadata, str) else cached_metadata
        return cache_key, metadata
    return cache_key, {}


async def generate_function_metadata(func: Callable, model_name=None, redis=None, key="funcmeta", **kwargs) -> Optional[
    Dict]:
    cache_key, metadata = await get_metadata_from_cache(func, redis, key)
    if metadata:
        return metadata

    function_code = remove_function_decorators(func)
    metadata = extract_function_metadata(func)
    # print(f"""
    # Extract the metadata for this Python function:
    # {function_code}
    # and output it in the following JSON format:
    # {json.dumps(metadata, indent=4, ensure_ascii=False)}
    #     """)
    metadata = await ai_generate_metadata(function_code, metadata, model_name=model_name, **kwargs)
    # 获取并存储生成的元数据

    if redis:
        try:
            await redis.setex(f"{key}:{cache_key}", Config.REDIS_CACHE_SEC, json.dumps(metadata, ensure_ascii=False))
            # await redis.set(f‘function:{ metadata["function"]["name"]}', str(metadata))  # 用函数名作为Redis的键,存储为JSON格式
        except Exception as e:
            print(e)
    Function_MetaData_Store[cache_key] = metadata

    return metadata


async def get_cached_tools_metadata(func_list: list[Callable] = None, model_name=None, **kwargs) -> list[dict]:
    redis = get_redis()
    if not func_list:
        tools_metadata = await scan_from_redis(redis, "funcmeta", batch_count=30) or list(
            Function_MetaData_Store.values())
        return tools_metadata

    # global_function_registry
    tasks = [generate_function_metadata(_f, model_name, redis=redis, key="funcmeta", **kwargs) for _f in func_list]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    tools_metadata: list[dict] = [metadata for metadata in results if isinstance(metadata, dict)]
    return tools_metadata


def metadata_decorator(func: Callable) -> Callable:
    """ 仅在后台生成元数据，不影响函数执行 """
    # print(func.__name__, type(func))
    if inspect.isfunction(func):  # 普通函数
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(generate_function_metadata(func, redis=get_redis()))
        else:
            asyncio.run(generate_function_metadata(func, redis=get_redis()))

    # @wraps(func)
    # def wrapper(*args, **kwargs):
    #
    #     return func(*args, **kwargs)# 执行

    return func  # wrapper


async def generate_tools_metadata(model_name=None, **kwargs) -> list[dict]:
    global_function: list[Callable] = [v for k, v in global_function_registry(func_name=None).items()]
    return AI_Tools + await get_cached_tools_metadata(func_list=global_function, model_name=model_name, **kwargs)


# @mcp.tool()
async def ai_tools_response(messages: list, tools: list[dict] = None, model_name=Config.DEFAULT_MODEL_FUNCTION,
                            model_id=1, top_p=0.95, temperature=0.01, **kwargs):
    """
      调用 AI 模型接口，使用提供的工具集和对话消息，返回模型的响应。qwen
        :return: 模型响应的消息对象
    """
    model_info, name = find_ai_model(model_name, model_id, 'model')
    client = AI_Client.get(model_info['name'], None)
    if not tools:
        tools = await get_cached_tools_metadata(func_list=[])
        # tools = [{"type": "web_search",
        #          "web_search": {
        #         "enable": True  # 启用网络搜索
        #         "search_query": "自定义搜索的关键词",
        #          "search_result": True,#默认为禁用,允许用户获取详细的网页搜索来源信息
        #     }}]

    tools = deduplicate_tools_by_name(tools)
    payload = dict(model=name,
                   messages=messages,
                   tools=tools,
                   # tool_choice="auto",
                   temperature=temperature,
                   top_p=top_p,
                   **kwargs)
    if client:
        try:
            completion = await client.chat.completions.create(**payload)
            return completion.choices[0].message

        except Exception as e:
            print(f"OpenAI error occurred: {e}")

    return None  # await ai_chat(model_info, payload)


Function_Registry_Global: Optional[Dict[str, Callable[..., Any]]] = {}


def global_function_registry(func_name: str = None, use_keywords: bool = False) -> Union[
    Callable[..., Any], Dict[str, Callable[..., Any]]]:
    """
    获取全局本地函数注册表中的函数或整个注册表。
    - use_keywords: 使用带关键字映射的注册（如自定义 lambda,用户网络请求运行函数,伪接口数据插入）
        - 用在registry list 的 user_calls,自定义func参数
    - 非 use_keywords: 使用标准全局函数名注册,用于自运行

    :param func_name: 函数名称。如果为 None，则返回整个注册表。
    :param use_keywords: 是否使用 keywords 模式
    :return: 如果 func_name 为 None，返回整个注册表；否则返回指定名称的函数。
    """
    keywords_registry: Dict[str, Callable[..., Any]] = {
        'no_func': lambda *args, **kwargs: f"[❌] Function not loaded",
        'map_search': search_amap_location,
        'web_search': web_search_async,
        'tavily_search': web_search_tavily,
        'tavily_extract': web_extract_tavily,
        'jina_extract': web_extract_jina,
        'news_search': lambda x: web_search_tavily(x, topic='news', time_range='w'),
        'baidu_search': lambda x: search_by_api(x, engine='baidu'),
        'wiki_search': wikipedia_search,
        'translate': auto_translate,
        "execute_code": execute_code_results,
        'auto_calls': ai_auto_calls,  # 名称不同，需指定调用，防止重复 ai_auto_calls
        'baidu_nlp': baidu_nlp,

        'visitor_records': ideatech_visitor_records,
    }

    global Function_Registry_Global
    # 动态性延迟加载,全局注册表初始化,调用可能需要确定某些参数
    if not Function_Registry_Global or len(Function_Registry_Global) < len(keywords_registry):
        Function_Registry_Global = functions_registry(functions_list=[
            "get_times_shift", "date_range_calculator",
            "get_day_range", "get_week_range", "get_month_range", "lang_token_size",
            "get_quarter_range", "get_year_range", "get_half_year_range", 'call_http_request', "math_solver",
            "extract_links", "extract_web_content", "remove_markdown", "extract_table_segments",
            "generates:auto_translate", 'database:patent_search', 'database:company_search',
            'generates:siliconflow_generate_image',
            'knowledge:ideatech_knowledge', 'generates:ideatech_visitor_records',
            # 添加更多可调用函数
        ])
        Function_Registry_Global.update(functions_registry(
            functions_list=['xunfei_ppt_create', 'tencent_generate_image'],
            module_name="agents.ai_multi"))

        Function_Registry_Global.update(functions_registry(
            functions_list=["web_search_async", "tokenize_with_zhipu", "get_weather", "duckduckgo_search",
                            "web_search_tavily", "web_extract_tavily",
                            "web_search_jina", "web_extract_jina", "segment_with_jina",
                            "search_by_api", "serper_search", "brave_search",
                            "exa_search", "web_extract_exa", "exa_retrieved",
                            "firecrawl_search", "firecrawl_scrape", "web_extract_firecrawl",
                            'wikipedia_search', 'arxiv_search', "get_amap_location",  # "search_bmap_location",
                            "baidu_translate", "tencent_translate", "xunfei_translate"],
            module_name="agents.ai_search"))

        Function_Registry_Global.update(functions_registry(
            functions_list=['annual_report_info', 'base_account_record', 'case_filing', 'company_black_list',
                            'company_exception_list', 'company_out_investment', 'company_personnel_risk',
                            'company_stock_relation', 'court_announcement', 'court_notice_info', 'dishonesty_info',
                            'equity_share_list',
                            'exact_saic_info', 'final_beneficiary', 'implements_info', 'judgment_doc',
                            'real_time_saic_info',
                            'saic_basic_info', 'shell_company', 'simple_cancellation', 'stock_freeze'],
            module_name="agents.ai_company"))

        Function_Registry_Global.update(keywords_registry)  # 合并 keywords 映射
        print(Function_Registry_Global)

    if use_keywords:
        return keywords_registry.get(func_name)

    return Function_Registry_Global.get(func_name, None) if func_name else Function_Registry_Global  # 从注册表中获取函数


def agent_func_calls(agent: str, messages: list = None, model: str = None, prompt: str = None) -> List[
    Callable[..., Any]]:  #
    from knowledge import ideatech_knowledge
    callable_map_agent = {
        'default': [lambda *args, **kwargs: [], ],  # 默认返回一个空函数列表
        '2': [named_partial('search_by_baidu', search_by_api, engine='baidu'), web_search_async],
        '9': [named_partial('auto_translate_baidu', auto_translate, model_name='baidu')],
        '29': [ideatech_knowledge],
        '31': [named_partial('auto_calls_any', ai_auto_calls, user_messages=messages,
                             system_prompt=System_content.get('31'), get_messages=False)],
        '32': [ai_auto_calls],
        '37': [web_search_async]
        # 扩展更多的 agent 类型，映射到多个函数
    }
    return callable_map_agent.get(agent, callable_map_agent['default'])


async def ideatech_visitor_records(prompt: str, customer_name: str | list | tuple, **kwargs):
    print(prompt, customer_name, kwargs)
    match = field_match('客户名称', customer_name)
    payload = dict(querys=[prompt], collection_name='拜访记录', client=QD_Client,
                   payload_key='record_text', match=match, not_match=[],
                   topn=20, score_threshold=0.5, exact=False,
                   embeddings_calls=ai_embeddings, model_name='BAAI/bge-large-zh-v1.5')
    payload.update(kwargs)
    return await search_by_embeddings(**payload)


def convert_to_callable_list(tool_list: List[Tuple[str, Any]],
                             callable_map: dict[str, Callable[..., Any]] = None) -> List[Callable[[], Any]]:
    """
        将工具列表转换为可调用函数列表。List[Tuple[str, Any]]->List[Callable[[], Any]]

        :param tool_list: 工具列表，每个工具是一个元组 (tool_name, config)。
        :param callable_map: 工具名称到可调用函数的映射。如果未提供，则使用全局注册表。
        :return: 可调用函数列表。
    """
    callable_list = []
    for tool_name, config in tool_list:
        tool_func = None
        if callable_map:
            if tool_name in callable_map:
                tool_func = callable_map[tool_name]  # 将配置绑定到 Callable
        else:
            tool_func = global_function_registry(tool_name)

        if tool_func:
            callable_list.append(partial(tool_func, **config))  # 函数绑定特定的配置参数,无参数可调用函数
        # else:
        #     print(f"Tool '{tool_name}' not found in callable_map.")
    return callable_list


async def execute_function_call(func_name: str, arguments: dict):
    redis = get_redis()
    key = f"registry:{func_name}"
    data = await redis.get(key)
    if not data:
        return None

    registry = json.loads(data)

    # 判断是否是远程调用
    if "x-url" in registry:
        return await send_callback(registry["x-url"], arguments)
    registry["parameters"] = registry.get("parameters", {})
    registry["parameters"]["arguments"] = arguments
    return registry


async def ai_tools_results(tool_messages) -> list[dict]:
    """
    解析模型响应，动态调用工具并生成后续消息列表。

    :param tool_messages: 模型响应的消息对象
    :return: 包含原始响应和工具调用结果的消息列表
    """
    messages = [tool_messages.to_dict()]  # ChatCompletionMessage
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
        func_reg = global_function_registry(func_name)  # 从注册表中获取函数 tools_map[function_name](**function_args)
        func_out = None
        error = None
        # print(func_args)
        # 检查并解析 func_args 确保是字典
        if isinstance(func_args, str):
            try:
                func_args = json.loads(func_args)  # 尝试将字符串解析为字典
            except json.JSONDecodeError as e:
                error = f"Error in {func_name}: Invalid arguments format ({str(e)})."

        if not isinstance(func_args, dict):
            error = f"Error in {func_name}: Arguments must be a mapping type, got {type(func_args).__name__}."

        if error:
            messages.append({
                "role": "tool",
                "content": error,
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
                func_out = await execute_function_call(func_name, arguments=func_args)
                if not func_out:
                    func_names = extract_method_calls(func_name)
                    if func_names:
                        func_name = func_names[-1]
                        func_out = safe_eval_call(func_name, func_args)
                    else:
                        error = f'Error in {func_name}: Function {func_names} extract method,not found.'

            messages.append({
                'role': 'tool',
                'content': f'{func_out or error}',  # json.dumps(result)
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


async def ai_auto_calls(question, user_messages: list = None, system_prompt: str = None,
                        model_name=Config.DEFAULT_MODEL_FUNCTION,
                        get_messages=False, **kwargs) -> list:
    """
    call_tool
    自动推理并调用工具，返回调用结果或消息列表
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
        user_messages = user_messages[-8:].copy()
        if not any(msg.get('role') == 'system' for msg in user_messages):
            user_messages.insert(0, {"role": "system", "content": system_prompt})
    else:
        user_messages = [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": question}]
        # 获取工具列表
    tools = AI_Tools + await get_cached_tools_metadata([])
    # 模型决定使用哪些工具
    tool_messages = await ai_tools_response(messages=user_messages, tools=tools, model_name=model_name, **kwargs)
    if not tool_messages:
        return []

    # 执行工具调用
    final_messages = await ai_tools_results(tool_messages)  # 组合了模型回复后的结果
    if not get_messages:
        return [(msg['name'], msg['content']) for msg in final_messages if msg['role'] == "tool"]
        # [{func_name:func_result}

    # 构造消息列表，适合继续对话
    tool_results = [f"(function:{msg['name']},result:{msg['content']})" for msg in final_messages if
                    msg['role'] == "tool"]

    result_messages = user_messages
    for msg in final_messages:
        if msg["role"] == "assistant" and "tool_calls" in msg:
            # 处理 tool_calls 为 JSON 格式
            msg["content"] = json.dumps(msg["tool_calls"], indent=2, ensure_ascii=False)
            result_messages.append(msg)

    result_messages.append({
        "role": "system",  # "user"
        "content": f"question:{question},\nresults:\n" + "\n".join(tool_results)
    })  # {"role": "system", "content": f"已调用函数 {func_name}，结果如下：{func_result}"}]
    return result_messages


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
        # print(messages)
        completion = await client.chat.completions.create(model=name, messages=messages, **kwargs)
        bot_response = completion.choices[0].message.content
        messages.append({"role": "assistant", "content": bot_response})
        return messages

    return messages


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

    batch_size = 16  # DASHSCOPE_MAX_BATCH_SIZE = 25
    has_error = False
    client = AI_Client.get(model_info['name'], None)
    if client:  # openai.Embedding.create
        if isinstance(inputs, (list, tuple)) and len(inputs) > batch_size:
            tasks = [client.embeddings.create(
                model=name, input=inputs[i:i + batch_size],
                encoding_format="float",
                **kwargs
            ) for i in range(0, len(inputs), batch_size)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

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
            results = await client.embeddings.create(
                model=name, input=inputs, encoding_format="float", **kwargs)

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
    payload = {
        "input": inputs,
        "model": name,
        "encoding_format": "float"
    }
    payload.update(kwargs)
    embeddings = []
    cx = get_httpx_client(time_out=Config.HTTP_TIMEOUT_SEC)
    try:
        if isinstance(inputs, (list, tuple)) and len(inputs) > batch_size:
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


def async_to_sync(func, *args, **kwargs):
    # 异步代码转换为同步代码
    return asyncio.run(func(*args, **kwargs))


async def wrap_sync(func, *args, **kwargs):
    # 同步代码转换为异步执行,在后台独立线程中运行,以避免阻塞主事件循环,使用 await 来执行
    return await asyncio.to_thread(func, *args, **kwargs)


# retriever
async def retrieved_reference(user_request: str, keywords: List[Union[str, Tuple[str, ...]]] = None,
                              tool_calls: List[Callable[[...], Any]] = None, **kwargs):
    """
      根据用户请求和关键字调用多个工具函数，并返回处理结果。

      :param user_request: 用户请求内容。
      :param keywords: 关键字列表，可以是字符串或元组（函数名, 参数）。
      :param tool_calls: 工具函数列表。
      :param kwargs: 其他关键字参数。
      :return: 处理结果的扁平化列表。
    """
    # docs = retriever(question)
    # Assume this is the document retrieved from RAG
    # function_call = Agent_Functions.get(agent, lambda *args, **kwargs: [])
    # refer = function_call(user_message, ...)
    tool_calls = tool_calls or []  # callable_list
    items_to_process = []
    callables = {}  # []
    tasks = []  # asyncio.create_task(),创建任务对象并将其加入任务列表

    if not keywords:
        if user_request:
            items_to_process = [user_request]  # ','.join(keywords)
    else:
        if all(not is_empty_lambda(_func) for _func in tool_calls):  # and _func()
            tool_calls.append(web_search_async)  # append auto func to run. ai_auto_calls
        # 多个keywords,用在registry list 的 user_calls,自定义func参数
        for item in keywords:
            if isinstance(item, tuple):  # (函数名,无参,单个参数,多个位置参数列表,关键字参数字典)
                try:
                    tool_name = item[0]
                    _func = global_function_registry(tool_name, use_keywords=True)  # user_calls 函数
                    if _func:
                        # if len(item) == 1:
                        #     callables.append((_func, [], {}))
                        func_args = []
                        func_kwargs = {}
                        for _arg in item[1:]:
                            if isinstance(_arg, dict):  # 处理关键字参数
                                if "params" in _arg:
                                    func_kwargs.update(_arg["params"])
                                else:
                                    func_kwargs.update(_arg)  # 将 dict 转换为可哈希类型frozenset(_arg.items())
                            elif isinstance(_arg, (list, tuple)):  # 处理多个位置参数
                                func_args.extend(_arg)
                            else:  # (函数名, 单个参数)
                                func_args.append(_arg)  # 剩下的参数[_arg]

                        callable_key = generate_hash_key(id(_func), func_args, **func_kwargs)
                        bound_func = partial(_func, *func_args, **func_kwargs)
                        callables[callable_key] = bound_func
                        # callables.append((_func, func_args, func_kwargs)))
                    # else:
                    #     tool_calls.extend(convert_to_callable_list(tools_list))
                    #     _func = global_function_registry(tool_name)
                    #     config=item[1]global_function_registry
                    #     if _func:
                    #         tool_calls.append(partial(_func, **config))  # 函数绑定特定的配置参数,无参数可调用函数
                except TypeError as e:
                    print(f"类型错误: {e},{item}")
                except Exception as e:
                    print(f"其他错误: {e},{item}")

            else:  # isinstance(keyword, str)
                items_to_process.append(item)  # keyword

    # 多个tool_calls to process items,kwargs func参数,agent控制
    for _func in filter(callable, tool_calls):
        if _func.__name__ == '<lambda>' and _func() == []:  # empty_lambda
            continue
        for item in items_to_process:
            callable_key = generate_hash_key(id(_func), item, **kwargs)
            bound_func = partial(_func, item, **kwargs)
            callables[callable_key] = bound_func
            # print(bound_func.func.__name__, bound_func.args, bound_func.keywords)
            # if inspect.iscoroutinefunction(_func):
            #     tasks.append(_func(item, **kwargs))
            # else:
            #     tasks.append(wrap_sync(_func, item, **kwargs))

    # print(callables)
    for _key, bound_func in callables.items():
        if inspect.iscoroutinefunction(bound_func.func):
            tasks.append(bound_func())  # 添加异步函数任务，同时传递kwargs
        else:
            tasks.append(wrap_sync(bound_func))  # (*bound_func.args, **bound_func.keywords)

    refer = await asyncio.gather(*tasks, return_exceptions=True)  # gather 收集所有异步调用的结果

    err_count = 0
    for t, r in zip(tasks, refer):
        if isinstance(r, Exception):
            print(f"Task {t.__name__} failed with error: {r}")
            err_count += 1
        elif not r:
            print(f"Task returned empty result: {t}")

    if err_count:
        print(callables, refer)
    # 展平嵌套结果,(result.items() if isinstance(result, dict) else result)
    return [item for result in refer if not isinstance(result, Exception)
            for item in (result if isinstance(result, list) else [result])]


# Callable[[参数类型], 返回类型]
async def get_chat_payload(messages: list[dict] = None, user_request: str = '', system: str = '',
                           temperature: float = 0.4, top_p: float = 0.8, max_tokens: int = 1024,
                           model_name=Config.DEFAULT_MODEL, model_id=0,
                           agent: str = None, tools: List[dict] = None,
                           keywords: List[Union[str, Tuple[str, ...]]] = None, images: List[str] = None, **kwargs):
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

        if model_info['name'] == 'mistral':
            for message in messages:
                if 'name' in message:
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

    tool_callable: List[Callable[[...], Any]] = agent_func_calls(agent, messages)  # .extend

    refer = await retrieved_reference(user_request, keywords, tool_callable, **kwargs)
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

    if tools and any(tools):  # 一旦你传了 tools 字段，它 必须 至少包含一个合法的 tool
        payload['tools'] = tools  # retrieval、web_search、function
    if model_type == 'baidu':
        payload['system'] = system
    if model_info['name'] == 'qwen':
        payload['extra_body'] = {"enable_thinking": False}  # 开启深度思考
    if name == 'o3-mini':
        payload['max_completion_tokens'] = payload.pop('max_tokens', max_tokens)

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
                                                            agent=agent, tools=None, keywords=keywords, **kwargs)

    else:
        tool_callable: List[Callable[[...], Any]] = agent_func_calls(agent)
        # 检索参考资料
        refer = await retrieved_reference(user_request, keywords, tool_callable, **kwargs)
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
                async for chunk in ai_chat_stream(model_info, payload, get_content):
                    yield chunk

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


async def stream_chat_completion(client, payload: dict) -> tuple[str, dict]:
    payload["stream"] = True
    payload["stream_options"] = {"include_usage": True}  # 可选，配置以后会在流式输出的最后一行展示token使用信息
    payload["extra_body"] = {"enable_thinking": True}  # enable_thinking 参数开启思考过程，该参数对 QwQ 模型无效
    try:
        stream = await client.chat.completions.create(**payload)

        if not stream or not hasattr(stream, "__aiter__"):
            raise TypeError("Returned stream is not async iterable")

        thinking = ''
        content = ''
        results_data = None

        async for chunk in stream:
            if not chunk:
                continue

            # 若 choices 为空但包含 usage，说明是 stream 末尾数据
            if not hasattr(chunk, "choices") or not chunk.choices:
                results_data = chunk
                # {"id": "chatcmpl-xxx", "choices": [], "created": 1719286190, "model": "qwen-plus",
                #  "object": "chat.completion.chunk", "system_fingerprint": null,
                #  "usage": {"completion_tokens": 16, "prompt_tokens": 22, "total_tokens": 38}}
                # {'id': 'chatcmpl-eba4e423-a1bf-90d6-b296-3e526727a0f0', 'choices': [], 'created': 1744611825,
                #  'model': 'qwq-plus', 'object': 'chat.completion.chunk', 'service_tier': None,
                #  'system_fingerprint': None,'usage': {'completion_tokens': 196, 'prompt_tokens': 14, 'total_tokens': 210,
                #            'completion_tokens_details': None, 'prompt_tokens_details': None}}

                break

            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                content += delta.content

            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                thinking += delta.reasoning_content

        if not content:
            raise ValueError("OpenAI API returned an empty stream response.")
        if thinking:
            print('thinking:', thinking)

        # 构建最终 completion 格式
        completion = results_data.model_dump() if results_data else {}
        completion.update({
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},  # .split("</think>")[-1]
                    "finish_reason": "stop",
                    "metadata": {
                        "reasoning": thinking
                    }
                }
            ],
        })
        return content, completion

    except Exception as e:
        raise RuntimeError(f"[stream_chat_completion] Failed to complete streaming call: {e}") from e


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

    client = AI_Client.get(model_info['name'], None)
    if client:
        try:
            # await asyncio.to_thread(client.chat.completions.create, **payload)
            # await asyncio.wait_for(client.chat.completions.create
            completion = await client.chat.completions.create(**payload)
            if completion is None:
                raise ValueError("OpenAI API returned None instead of a valid response")
            if not hasattr(completion, "choices") or not hasattr(completion.choices[0], "message"):
                raise ValueError(f"Unexpected API response: {completion}")

            return completion.choices[0].message.content if get_content else completion.model_dump()  # 自动序列化为 JSON
            # json.loads(completion.model_dump_json())

        except BadRequestError as e:
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

            try:
                error_json = e.response.json() if hasattr(e.response, "json") else json.loads(e.response.text)
                error_message = error_json.get("error", {}).get("message", "")
            except:
                error_message = str(e)

            print('BadRequest error message:', error_message)
            stream_required_phrases = [
                "only support stream mode",
                "only supports streaming",
                "non-streaming calls",  # +/no_think
                "stream=true",
                "模型不支持sync调用"
            ]

            if any(p in error_message.lower() for p in stream_required_phrases):
                content, completion = await stream_chat_completion(client, payload)
                return content if get_content else completion
            else:
                # length_required_phrases = [
                #     "range of input length",
                #     "input length should be",
                #     "max input characters"
                #     "invalid max_tokens value"
                # ]
                # if any(p in error_message.lower() for p in length_required_phrases):
                #     raise ValueError(f"Input token length exceeds model limit or is empty. Raw message: {error_message}")
                #     # payload['message']=cut_chat_history(payload['message'], max_size_limit_count=33000)
                raise

        except Exception as e:
            print("OpenAI error:", e, payload)
            error_message = f"OpenAI error occurred: {e}"
            if get_content:
                return error_message
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
                            "content": error_message,
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
            return fake_response

    url, headers, payload = get_chat_payload_post(model_info, payload)
    # print(headers, payload)
    parse_rules = {
        'baidu': lambda d: d.get('result'),
        'tencent': lambda d: d.get('Response', {}).get('Choices', [{}])[0].get('Message', {}).get('Content'),
        # d.get('Choices')[0].get('Message').get('Content')
        'default': lambda d: d.get('choices', [{}])[0].get('message', {}).get('content')
    }
    cx = get_httpx_client(time_out=Config.HTTP_TIMEOUT_SEC)
    try:
        response = await cx.post(url, headers=headers, json=payload)
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
        return f"HTTP error occurred: {e},{response.text}"


async def ai_chat_stream(model_info: Optional[Dict[str, Any]], payload=None, get_content=True, **kwargs):
    if not payload:
        model_info, payload, _ = await get_chat_payload(**kwargs)
    else:
        # payload=copy(payload)
        payload.update(kwargs)

    payload["stream"] = True
    payload["extra_body"] = {"enable_thinking": True}
    client = AI_Client.get(model_info['name'], None)
    # print(payload, client)
    if client:
        try:
            stream = await client.chat.completions.create(**payload)
            if not stream:
                raise ValueError("OpenAI API returned an empty response")
            if not hasattr(stream, "__aiter__"):
                raise TypeError("OpenAI API returned a non-streaming response")
            has_content = False
            async for chunk in stream:  # for chunk in stream
                if not chunk:
                    continue
                has_content = True
                if get_content:
                    delta = chunk.choices[0].delta
                    if delta.content:  # 以两个换行符 \n\n 结束当前传输的数据块
                        yield delta.content
                    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                        yield delta.reasoning_content
                else:
                    yield chunk.model_dump_json()  # 获取字节流数据

            if not has_content:
                raise ValueError("OpenAI API returned an empty stream")
        except Exception as e:
            print("OpenAI error:", e, payload)
            error_message = f"OpenAI error occurred: {e}"
            fake_response = {
                "id": f"chatcmpl-{int(time.time())}", "object": "chat.completion.chunk",
                "created": int(time.time()), "model": payload.get("model", "unknown"),
                "choices": [
                    {"index": 0, "delta": {"content": error_message}, "finish_reason": "stop"}],
            }
            yield error_message if get_content else json.dumps(fake_response)

            # yield '[DONE]'
            # with client.chat.completions.with_streaming_response.create(**payload) as response:
            #     print(response.headers.get("X-My-Header"))
            #     for line in response.iter_lines():
            #         yield line

        return  # 异步生成器的结束无需返回值

    url, headers, payload = get_chat_payload_post(model_info, payload)
    cx = get_httpx_client(time_out=Config.HTTP_TIMEOUT_SEC)
    try:
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
    cx = get_httpx_client(time_out=Config.HTTP_TIMEOUT_SEC)
    response = await cx.post(url, headers=headers, json=payload)
    response.raise_for_status()
    if response.status_code != 200:
        return None, {'error': f'{response.status_code},Request failed,Response content: {response.text}'}

    response_data = response.json()
    return None, {"urls": [i['url'] for i in response_data["images"]], 'id': response_data["seed"]}


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
def oss_list_files(bucket, prefix='upload/', max_keys=100, max_pages=1):
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


async def send_to_wechat(user_name: str, context: str = None, link: str = None, object_name: str = None):
    url = f"{Config.WECHAT_URL}/sendToChat"
    headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
    body = {'user': user_name, 'context': context, 'url': link,
            'object_name': object_name, 'file_type': get_file_type_wx(object_name)}

    try:
        cx = get_httpx_client(time_out=Config.HTTP_TIMEOUT_SEC)
        response = await cx.post(url, json=body, headers=headers)
        response.raise_for_status()
        return response.json()
        # with httpx.Client(timeout=Config.HTTP_TIMEOUT_SEC) as cx:
        #     response = cx.post(url, json=body, headers=headers)
        #     response.raise_for_status()
        # return response.json()

    except Exception as e:
        print('send_to_wechat', datetime.now(), body)
        print(f"Error occurred while sending message: {e}")

    return None


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
    if params and isinstance(params, dict):
        kwargs.update(params)

    res = None
    format_type = callback_data.get("format", 'json').lower()  # "query" or "json" form"
    if format_type == "json":
        res = await post_http_json(url, json=payload, headers=headers, time_out=Config.HTTP_TIMEOUT_SEC, **kwargs)
    if format_type == "query":  # query 参数或表单参数
        query_payload = filter_payload(payload)
        print(query_payload)
        res = await get_http_query(url, params=query_payload, headers=headers, time_out=Config.HTTP_TIMEOUT_SEC)
    if format_type == "form":  # 支持 query 或 form
        res = await post_http_form(url, data=payload, headers=headers, time_out=Config.HTTP_TIMEOUT_SEC)

    if res and res.get('error'):
        return await post_http_json(fallback_url, json=payload, headers=headers, **kwargs)
    return res


if __name__ == "__main__":
    print(Function_MetaData_Store)
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
