from typing import Callable, Dict, Any, Optional, Union, List, Tuple
from functools import partial, wraps
import inspect
import json

from agents.ai_generates import ai_generate_metadata
from utils import fix_indentation, remove_function_decorators, generate_hash_key, extract_function_metadata, \
    functions_registry, extract_method_calls, safe_eval_call, run_with_async, run_bound_tasks, run_by_future, \
    run_togather, ClassMethodRegistry, strip_kwargs_wrapper
from service import get_redis, get_redis_retry, scan_from_redis, run_with_lock, send_callback, logger, FastMCP
from config import Config


class FunctionManager:
    _meta_store: Dict[str, dict] = {}  # Function_MetaData_Store {cache_func_key:cached_metadata}
    global_registry: Dict[str, Callable[..., Any]] = {}  # 全局标准函数注册表（自动加载模块）Function_Registry_Global
    class_registry = ClassMethodRegistry
    mcp_server = FastMCP(name="DEF MCP Server")
    key_meta = "funcmeta"
    key_reg_name = "registry"  # {cache_key:func_name}

    def __init__(self, keywords_registry: Dict[str, Callable] = None, copy: bool = True,
                 mcp_server: Optional[FastMCP] = None, redis=None):
        """外部手动注册函数表"""
        self.redis = redis or get_redis()
        self.registry_meta: Dict[str, dict] = {}  # 缓存函数元数据 {function_name:metadata}
        self.keywords_registry: Dict[str, Callable[..., Any]] = keywords_registry or {}  # handlers

        if mcp_server:
            self.__class__.mcp_server = mcp_server
            print('Mcp_Server:', self.mcp_server)

        # 动态性延迟加载,全局注册表初始化,调用可能需要确定某些参数
        if not self.__class__.global_registry:
            for m, f in Registry_Module.items():
                self.set_registry_global(functions_list=f, module_name=m)
            print('Function_Registry:', len(self.__class__.global_registry))

        if copy:
            self.update_registry(self.__class__.global_registry)

    def __str__(self):
        return (f"Global_Registry: {list(self.__class__.global_registry.keys())}"
                f"Keywords_Registry: {list(self.keywords_registry.keys())}")

    def update_registry(self, keywords_registry: Dict[str, Callable] = None):
        """外部手动更新函数表,外部手动注册函数表,用户自定义关键词映射（支持 lambda、闭包等）"""
        self.keywords_registry.update(keywords_registry or {})  # 合并 keywords 映射

    @classmethod
    def set_registry_global(cls, functions_list: list, module_name: str = None):
        cls.global_registry.update(functions_registry(functions_list=functions_list,
                                                      module_name=None if module_name == 'default' else module_name))

    @classmethod
    async def get_mcp_tools(cls) -> dict:
        return await cls.mcp_server.get_tools() if cls.mcp_server else {}

    @classmethod
    def register(cls, func: Callable):
        name = func.__name__
        cls.global_registry[name] = func
        if cls.mcp_server:
            cls.mcp_server.tool(func, name=name)
        return func

    @classmethod
    async def register_mcp_tools(cls):
        tools = await cls.get_mcp_tools()
        for name, func in cls.global_registry.items():
            if name in tools:
                continue  # Tool already exists
            func = strip_kwargs_wrapper(func)
            try:
                cls.mcp_server.tool(func, name=name)  # from_function
            except Exception as e:
                logger.error(f"[register_mcp_tools] {name}: {e}")

    @property
    def registered_method(self):
        """返回当前实例可用的注册方法表（优先实例级，否则用全局级）"""
        return self.keywords_registry or self.__class__.global_registry

    @property
    def class_method(self):
        """返回类级的 class_registry"""
        return self.__class__.class_registry

    @property
    def mcp_handler(self):
        """返回类级的 mcp_server"""
        return self.__class__.mcp_server

    # global_function_registry
    def get_function_registered(self, func_name: str | list = None) -> Union[Callable, Dict[str, Callable]]:
        """
        获取全局本地函数注册表中的函数或整个注册表。
        :param func_name: 函数名称。如果为 None，则返回整个注册表。
        :return: 如果 func_name 为 None，返回整个注册表；否则返回指定名称的函数。
        """
        if func_name is None:
            return self.registered_method
        if isinstance(func_name, list):
            return {f: self.keywords_registry.get(f) for f in func_name if f in self.keywords_registry}
        return self.keywords_registry.get(func_name, None)

    @classmethod
    async def get_function_registered_callback(cls, func_name: str, arguments: dict, redis=None, **kwargs):
        redis = redis or get_redis()
        registered = await redis.hgetall(cls.key_reg_name)  # await redis.hget(cls.key_reg_name, cache_key)
        cache_key = next((k for k, v in registered.items() if v == func_name), None)
        if not cache_key:
            return None
        registry = await cls.get_metadata_from_cache(cache_key, redis=redis, retry=1)
        if "callback" in registry:  # 判断是否是远程调用
            return await send_callback(registry["callback"], arguments, **kwargs)
        registry["parameters"] = registry.get("parameters", {})
        registry["parameters"]["arguments"] = arguments
        return registry

    async def get_metadata_registered(self, func_name: str, model_name: str = None):
        '''优先获取已注册的元数据，没有则生成；支持本地缓存和 Redis 缓存。'''
        func: Callable = self.keywords_registry.get(func_name, None)
        if not callable(func):
            return {}, None
        metadata = self.registry_meta.get(func_name, {})
        if metadata:
            return metadata, func
        # 尝试从缓存获取 或者生成元数据
        metadata = await self.generate_metadata(func, model_name, redis=self.redis)
        if metadata and isinstance(metadata, dict):
            self.registry_meta[func_name] = metadata
        return metadata, func

    @classmethod
    def convert_to_callable_list(cls, func_list: List[tuple[str, Any]],
                                 callable_map: dict[str, Callable[..., Any]] = None) -> List[Callable[[], Any]]:
        """
        将工具列表转换为可调用函数列表。List[Tuple[str, Any]]->List[Callable[[], Any]]

        :param func_list: 工具列表，每个工具是一个元组 (tool_name, config)。
        :param callable_map: 工具名称到可调用函数的映射。如果未提供，则使用全局注册表。
        :return: 可调用函数列表。
        """
        callable_list = []
        for tool_name, config in func_list:
            tool_func = None
            if callable_map:
                if tool_name in callable_map:
                    tool_func = callable_map[tool_name]  # 将配置绑定到 Callable
            else:
                tool_func = cls.global_registry.get(tool_name)

            if tool_func:
                callable_list.append(partial(tool_func, **config))  # 函数绑定特定的配置参数,无参数可调用函数
            # else:
            #     print(f"Tool '{tool_name}' not found.")
        return callable_list

    async def run_function(self, func_name: str, func_args: dict | str, tool_id: str):
        error = None
        # 检查并解析 func_args 确保是字典
        if isinstance(func_args, str):
            try:
                func_args = json.loads(func_args)  # 尝试将字符串解析为字典:tool_call.function.arguments
            except json.JSONDecodeError as e:
                error = f"Error in {func_name}: Invalid arguments format ({str(e)})."

        if not isinstance(func_args, dict):
            error = f"Error in {func_name}: Arguments must be a mapping type, got {type(func_args).__name__}."

        if error:
            return {"role": "tool", "content": error, "tool_call_id": tool_id, 'name': func_name}
        # 从注册表中获取函数 tools_map[function_name](**function_args)
        func_reg = self.get_function_registered(func_name)
        try:
            if func_reg:
                func_out = await run_with_async(func_reg, **func_args)
            else:
                func_out = await self.get_function_registered_callback(func_name, arguments=func_args, redis=self.redis)
                if not func_out:
                    func_names = extract_method_calls(func_name)
                    if func_names:
                        func_name = func_names[-1]
                        func_out = safe_eval_call(func_name, func_args)
                    else:
                        error = f'Error in {func_name}: Function {func_names} extract method,not found.'

            return {
                'role': 'tool',
                'content': f'{func_out or error}',  # json.dumps(result)
                'tool_call_id': tool_id,
                'name': func_name
            }

        except Exception as e:
            error = f"Error in {func_name}: {str(e)}" if func_reg else f"Error: Function '{func_name}' not found."
            logger.error(error)
            return {'role': 'tool', 'content': error, 'tool_call_id': tool_id, 'name': func_name}

    # retriever
    async def retrieved_reference(self, keywords: List[Union[str, Tuple[str, ...]]] = None,
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
        tool_calls = tool_calls or []
        keywords = keywords or []
        callables = {}
        # 多个keywords,用在registry list 的 user_calls,自定义func参数
        for item in keywords:
            if isinstance(item, tuple):  # (函数名,无参,单个参数,多个位置参数列表,关键字参数字典)
                try:
                    tool_name = item[0]
                    _func = self.get_function_registered(tool_name)  # user_calls 函数
                    if _func:
                        func_args = item[1] if len(item) > 1 else []
                        func_kwargs = item[2] if len(item) > 2 else {}
                        callable_key = generate_hash_key(id(_func), func_args, **func_kwargs)
                        bound_func = partial(_func, *func_args, **func_kwargs)
                        callables[callable_key] = bound_func
                        # callables.append((_func, func_args, func_kwargs)))
                except TypeError as e:
                    logger.error(f"类型错误: {e},{item}")
                except Exception as e:
                    logger.error(f"其他错误: {e},{item}")

        for bound_func in filter(callable, tool_calls):
            if isinstance(bound_func, partial):
                callable_key = generate_hash_key(id(bound_func), bound_func.args, **bound_func.keywords)
                callables[callable_key] = bound_func
                logger.info(f"{bound_func.func.__name__} with args: {bound_func.args}, kwargs: {bound_func.keywords}")
            else:
                func_name = getattr(bound_func, "__name__", type(bound_func).__name__)
                logger.warning(f"{func_name}(not partial)")

        refer = await run_bound_tasks(bound_funcs=[f for _, f in callables.items()])
        # 展平嵌套结果,(result.items() if isinstance(result, dict) else result)
        return [item for result in refer if not isinstance(result, Exception)
                for item in (result if isinstance(result, list) else [result])]

    @classmethod
    async def get_metadata_from_cache(cls, cache_key: str, redis=None, retry: int = 2, ex: int = 0):
        try:
            if not redis:
                raise RuntimeError("Redis not available")
            cache_full_key = f"{cls.key_meta}:{cache_key}"
            cached_metadata = await get_redis_retry(redis, cache_full_key, retry=retry)
            await redis.expire(cache_full_key, ex or Config.REDIS_CACHE_SEC)
            await redis.expire(cls.key_reg_name, ex or Config.REDIS_CACHE_SEC)
            if cached_metadata:
                return json.loads(cached_metadata)
        except Exception:
            # Too many connections
            return cls._meta_store.get(cache_key, {})

    @classmethod
    async def set_metadata_to_cache(cls, metadata: dict, cache_key: str, func_name: str, redis=None, ex: int = 0):
        """存储生成的元数据"""
        if redis:
            try:
                await redis.setex(f"{cls.key_meta}:{cache_key}", ex or Config.REDIS_CACHE_SEC,
                                  json.dumps(metadata, ensure_ascii=False))
                await redis.hset(cls.key_reg_name, cache_key, func_name)  # mapping={cache_key:func_name}
                await redis.expire(cls.key_reg_name, ex or Config.REDIS_CACHE_SEC)
            except Exception as e:
                logger.error(f'[set_metadata_to_cache]:{e}')
        cls._meta_store[cache_key] = metadata

    @classmethod
    async def get_tools_metadata_by_redis(cls, func_name: str = None, redis=None) -> list[dict]:
        redis = redis or get_redis()
        if not func_name:
            return await scan_from_redis(redis, key=cls.key_meta, batch_count=50)

        registered = await redis.hgetall(cls.key_reg_name)  # await redis.hget(cls.key_reg_name, cache_key)
        cache_keys = [f'{cls.key_meta}:{k}' for k, v in registered.items() if v == func_name]
        if not cache_keys:
            return []
        cached_values = await redis.mget(*cache_keys)
        return [json.loads(v) for v in cached_values if v]

    @classmethod
    async def generate_metadata(cls, func: Callable, model_name: str = None, redis=None, retry=2, **kwargs) -> dict:
        '''
        为单个函数生成元数据（可从 Redis/本地缓存中恢复）
        '''
        # 获取函数的名称、参数以及docstring
        func_name = func.__name__
        function_code = fix_indentation(remove_function_decorators(func))
        cache_key = generate_hash_key(func_name, function_code)
        metadata = await cls.get_metadata_from_cache(cache_key, redis=redis, retry=retry)
        if metadata:
            # print(f"Metadata already cached for function: {func_name},{metadata}")
            return metadata
        # print(f"Metadata no cached for function: {func_name}\n{function_code}")

        metadata = extract_function_metadata(func)
        metadata = await ai_generate_metadata(function_code, metadata, model_name=model_name, **kwargs)
        func_name = metadata.get("function", {}).get("name", func.__name__)
        await cls.set_metadata_to_cache(metadata, cache_key, func_name, redis=redis)
        return metadata

    @classmethod
    async def get_tools_metadata(cls, func_list: list[Callable] = None, model_name=None, redis=None,
                                 **kwargs) -> list[dict]:
        """
        为给定函数列表获取或生成元数据（批量）
        如果没有 func_list []，直接redis里面获取，否则逐个检查 func_list 注册的缓存或者生成；返回元数据列表
        :return tools_metadata: list[dict]
        """
        if not func_list:
            tools_metadata = None
            if redis:
                tools_metadata = await cls.get_tools_metadata_by_redis(func_name=None, redis=redis)
            return tools_metadata or list(cls._meta_store.values())

        process_func = run_togather(max_concurrent=Config.REDIS_MAX_CONCURRENT)(cls.generate_metadata)
        results = await process_func(inputs=func_list, model_name=model_name, redis=redis, **kwargs)
        return [metadata for metadata in results if isinstance(metadata, dict)]

    async def tools_metadata(self, model_name: str = Config.DEFAULT_MODEL_METADATA, registered: bool = True,
                             **kwargs) -> list[dict]:
        '''
        registered:获取所有注册表中的函数元数据（keywords + global）
        否则 cache获取（redis or local）
        '''
        function_list: list[Callable] = [v for k, v in self.registered_method.items()] if registered else None
        return await self.get_tools_metadata(func_list=function_list, model_name=model_name, redis=self.redis, **kwargs)

    async def generate_tools_metadata(self, model_name: str = Config.DEFAULT_MODEL_METADATA, lock_timeout: int = 600):
        '''
        加锁后生成所有注册表函数元数据，用于定时任务等场景
        '''
        await run_with_lock(self.tools_metadata, lock_timeout=lock_timeout, lock_key="lock:generate_tools_metadata",
                            model_name=model_name, registered=True)

    @classmethod
    def generate_metadata_future(cls, func: Callable):
        """
        异步生成函数元数据，不影响函数执行，自动检测事件循环状态，选择合适的执行方式
        """
        # print(func.__name__, type(func))
        if inspect.isfunction(func):
            run_by_future(cls.generate_metadata, func, redis=get_redis())

    @classmethod
    def metadata_decorator(cls, func: Callable) -> Callable:
        """
        类方法装饰器，用于在后台生成函数元数据
        @FunctionManager.metadata_decorator
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            # 在函数执行前启动元数据生成（不阻塞）
            cls.generate_metadata_future(func)
            return func(*args, **kwargs)  # 执行原函数

        return cls.register(wrapper)  # 返回注册过的包装函数


Registry_Module = {
    "default": [
        "generates:auto_translate", 'database:patent_search', 'database:company_search',
        'generates:siliconflow_generate_image',
        'script.knowledge:ideatech_knowledge', 'generates:ideatech_visitor_records',
        # 添加更多可调用函数
    ],
    "utils": ["get_times_shift", "date_range_calculator", 'extract_json_struct', 'extract_json_array',
              "get_day_range", "get_week_range", "get_month_range", "lang_token_size",
              "get_quarter_range", "get_year_range", "get_half_year_range", "math_solver",
              "extract_links", "extract_web_content", "remove_markdown", "extract_table_segments",
              "save_markdown", "read_markdown"],
    "agents.ai_multi": ['xunfei_ppt_create', 'tencent_generate_image'],
    "agents.ai_company": ['annual_report_info', 'base_account_record', 'case_filing', 'company_black_list',
                          'company_exception_list', 'company_out_investment', 'company_personnel_risk',
                          'company_stock_relation', 'company_stock_deep_relation',
                          'court_announcement', 'court_notice_info', 'dishonesty_info', 'equity_share_list',
                          'exact_saic_info', 'final_beneficiary', 'implements_info', 'judgment_doc',
                          'real_time_saic_info',
                          'saic_basic_info', 'shell_company', 'simple_cancellation', 'stock_freeze'],
    "agents.ai_search": ["web_search_async", "web_search_intent", "tokenize_with_zhipu",
                         "get_weather", "duckduckgo_search",
                         "web_search_tavily", "web_extract_tavily",
                         "web_search_jina", "web_extract_jina", "segment_with_jina",
                         "search_by_api", "serper_search", "brave_search",
                         "exa_search", "web_extract_exa", "exa_retrieved",
                         "firecrawl_search", "firecrawl_scrape", "web_extract_firecrawl",
                         'wikipedia_search', 'arxiv_search', "get_amap_location",  # "search_bmap_location",
                         "baidu_translate", "tencent_translate", "xunfei_translate"],
}

AI_Tools = [
    {
        "type": "function",
        "function": {
            "name": "get_times_shift",
            "description": "当你想知道时间时非常有用。获取当前时间，并根据偏移的天数和小时数调整时间。",
            "parameters": {
                "type": "object",
                "properties": {
                    "days_shift": {
                        "type": "integer",
                        "description": "偏移的天数，>0 表示未来，<0 表示过去，0 表示当前日期。",
                        "default": 0
                    },
                    "hours_shift": {
                        "type": "integer",
                        "description": "偏移的小时数，>0 表示未来，<0 表示过去，0 表示当前时间。",
                        "default": 0
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "date_range_calculator",
            "description": "计算基于参考日期的时间范围，支持按天、周、月、季度、年或半年计算日期范围，支持偏移周期数和返回多个周期范围。",
            "parameters": {
                "type": "object",
                "properties": {
                    "period_type": {
                        "type": "string",
                        "enum": ["days", "weeks", "months", "quarters", "year", "half_year"],
                        "description": "时间周期类型，可以是 'days'、'weeks'、'months'、'quarters'、'year' 或 'half_year'。"
                    },
                    "date": {
                        "type": "string",
                        "format": "date",
                        "description": "基准日期，格式为 'YYYY-MM-DD'，默认为当前日期。"
                    },
                    "shift": {
                        "type": "integer",
                        "default": 0,
                        "description": "偏移量，表示时间周期的偏移，0 表示当前周期，负值表示过去，正值表示未来。"
                    },
                    "count": {
                        "type": "integer",
                        "default": 1,
                        "description": "时间周期数量，表示返回多少个周期的日期范围,默认为 1。"
                    }
                },
                "required": ["period_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search_async",
            "description": "通过提供的查询文本执行网络搜索，外部知识实时动态信息检索。",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "需要在网络上搜索的查询文本。",
                    },
                    "api_key": {
                        "type": "string",
                        "description": "用于访问网络搜索工具的API密钥，默认不需要提供。",
                        "default": "Config.GLM_Service_Key"
                    }
                },
                "required": ["text"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_bmap_location",
            "description": "使用百度地图API搜索指定地点。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索地点的关键词。",
                    },
                    "region": {
                        "type": "string",
                        "description": "限制搜索的区域，例如城市名。",
                        "default": ""
                    },
                    "limit": {
                        "type": "boolean",
                        "description": "是否仅限指定区域内搜索，默认为 True。",
                        "default": True
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "auto_translate",
            "description": "自动翻译工具，根据输入的文本和指定的翻译模型完成语言检测与翻译。",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "需要翻译的文本内容。"
                    },
                    "model_name": {
                        "type": "string",
                        "description": "指定使用的翻译模型，支持 'baidu','tencent','xunfei'。其他模型(如:'qwen','moonshot','baichuan',doubao','hunyuan','spark')将自动调用 AI 翻译,将提供目标语言释义和例句.",
                        "default": "baidu"
                    },
                    "source": {
                        "type": "string",
                        "description": "原文本的语言代码，支持具体语言代码（如 'en', 'zh','zh-TW','ja','ru','it','fr','pt','th','ko','es','vi','id'）或 'auto'（自动检测）。",
                        "default": "auto"
                    },
                    "target": {
                        "type": "string",
                        "description": "目标翻译语言代码，支持具体语言代码（如 'en', 'zh','zh-TW','ja','ru','it','fr','pt','th','ko','es','vi','id'）或 'auto'（根据源语言自动设定）。",
                        "default": "auto"
                    }
                },
                "required": ["text"]
            }
        }
    }
]

'''
功能型 (function): 核心逻辑的实现。
工具型 (tool): 通过现成工具提供结果。
API型 (api): 封装外部接口调用。
服务型 (service): 长时间运行或后台任务。
插件型 (plugin): 为现有系统提供扩展。
数据型 (data): 处理和生成数据的任务。
查询型 (query): 面向数据检索的操作。
'''
if __name__ == 'main':
    pass
