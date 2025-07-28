import httpx
import asyncio
import logging
import json, uuid
from logging.handlers import RotatingFileHandler
from utils import call_http_request, async_to_sync
from config import Config, AI_Models, model_api_keys, extract_ai_model

# Config.load('../config.yaml')
# if os.getenv('AIGC_DEBUG', '0').lower() in ('1', 'true', 'yes'):
#  Config.debug()
from neo4j import GraphDatabase, AsyncGraphDatabase
from dask.distributed import Client as DaskClient, LocalCluster
from qdrant_client import AsyncQdrantClient, QdrantClient
from openai import AsyncOpenAI, OpenAI

from redis.asyncio import Redis, StrictRedis, ConnectionPool
# from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
# from sqlalchemy.pool import NullPool, QueuePool
from database import MysqlData, BaseReBot
from abc import ABC, abstractmethod
from functools import partial, wraps
from typing import Callable, Optional, Type, Awaitable, Dict, Tuple

_httpx_clients: Dict[str, httpx.AsyncClient] = {}
_graph_driver: Optional[GraphDatabase] = None
# _graph_driver_lock = asyncio.Lock()  # 防止并发初始化
_redis_client: Optional[Redis] = None  # StrictRedis(host='localhost', port=6379, db=0)
_redis_pool: Optional[ConnectionPool] = None
_dask_client: Optional[DaskClient] = None

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.WARNING,
                    handlers=[
                        logging.StreamHandler(),  # 输出到终端,控制台输出
                        RotatingFileHandler("app.log", maxBytes=1_000_000, backupCount=3)
                        # 文件日志logging.FileHandler('errors.log')
                    ])
AI_Client: Dict[str, Optional[AsyncOpenAI]] = {}  # OpenAI
QD_Client = AsyncQdrantClient(host=Config.QDRANT_HOST, grpc_port=Config.QDRANT_GRPC_PORT,
                              prefer_grpc=True) if Config.QDRANT_GRPC_PORT else AsyncQdrantClient(url=Config.QDRANT_URL)

DB_Client = MysqlData(persistent=True, async_mode=True)


# echo=True 仅用于调试 poolclass=NullPool,每次请求都新建连接，用完就断，不缓存
# async_engine = create_async_engine(Config.ASYNC_SQLALCHEMY_DATABASE_URI)
# AsyncSessionLocal = sessionmaker(autocommit=False, autoflush=False, expire_on_commit=False, bind=async_engine,
#                                  class_=AsyncSession)
# poolclass=QueuePool,多线程安全的连接池，复用连接


class DataProcessor(ABC):
    @abstractmethod
    def process(self, intermediate: dict) -> dict:
        raise NotImplementedError("必须实现此方法")


def get_httpx_client(time_out: float = Config.HTTP_TIMEOUT_SEC, proxy: str = None) -> httpx.AsyncClient:
    # @asynccontextmanager
    key = proxy or "default"
    global _httpx_clients
    if key not in _httpx_clients or _httpx_clients[key].is_closed:
        transport = httpx.AsyncHTTPTransport(proxy=proxy or None)
        limits = httpx.Limits(max_connections=100, max_keepalive_connections=50)
        timeout = httpx.Timeout(timeout=time_out, read=60.0, write=30.0, connect=5.0)
        _httpx_clients[key] = httpx.AsyncClient(transport=transport, limits=limits, timeout=timeout)
    # try:
    #     yield _httpx_clients[key] #Depends(get_httpx_client)
    # finally:
    #     # 注意：不要在这里关闭客户端，因为它是单例，全局用的
    #     pass

    return _httpx_clients[key]


async def shutdown_httpx():
    for key, _client in _httpx_clients.items():
        if _client and not _client.is_closed:
            await _client.aclose()


# async def get_async_session() -> AsyncSession:
#     # 异步依赖
#     async with AsyncSessionLocal() as session:
#         yield session  # 自动 await 生成器
#     # finally:
#     #   await session.close()

def get_redis() -> Optional[Redis]:
    global _redis_client, _redis_pool
    if _redis_client is None:
        _redis_pool = ConnectionPool(host=Config.REDIS_HOST, port=Config.REDIS_PORT, db=0,
                                     decode_responses=True,  # 自动解码为字符串
                                     max_connections=Config.REDIS_MAX_CONCURRENT
                                     # connection_pool=pool
                                     )
        _redis_client = Redis(connection_pool=_redis_pool)

    return _redis_client


async def shutdown_redis():
    global _redis_client, _redis_pool
    if _redis_client:
        await _redis_client.close()
        _redis_client = None

    if _redis_pool:
        await _redis_pool.disconnect()
        _redis_pool = None


async def check_redis_connection(redis):
    try:
        await redis.ping()
        print("Redis connected.")
        return True
    except ConnectionError as e:
        print(f"❌ Redis connection failed: {e}")
    return False


async def get_redis_connection():
    redis = get_redis()
    if not await check_redis_connection(redis):
        return None
    return redis


async def get_redis_retry(redis, key, retry: int = 3, delay: float = 0.1):
    for attempt in range(retry):
        try:
            return await redis.get(key)
        except Exception as e:
            print(f"[Redis GET] attempt {attempt + 1} failed: {e}")
            await asyncio.sleep(delay)
    raise Exception(f"Redis GET failed after {retry} retries.")


async def scan_from_redis(redis, key: str = "funcmeta", user: str = None, batch_count: int = 0):
    """
    从 Redis 中获取匹配的所有元数据记录，支持 scan 或 keys 方式。

    Args:
        redis: Redis 实例。
        key: Redis key 前缀（如 "funcmeta"）。
        user: 用户 ID（可选），若指定则拼接为 funcmeta:{user}:*
        batch_count: 每批 scan 的数量（大于 0 使用 scan，否则用 keys）。

    Returns:
        List[dict]: 匹配到的 JSON 数据列表。
    """
    if user:
        match_pattern = f"{key}:{user}:*"
    else:
        match_pattern = f"{key}:*"

    data = []
    if batch_count > 0:
        cursor = b'0'
        while cursor:
            cursor, keys = await redis.scan(cursor=cursor, match=match_pattern, count=batch_count)
            if keys:
                values = await redis.mget(*keys)
                data.extend(json.loads(v) for v in values if v)
    else:
        keys = await redis.keys(match_pattern)
        if keys:
            cached_values = await redis.mget(*keys)
            data = [json.loads(v) for v in cached_values if v]  # set(cached_values
    return data


async def do_job_by_lock(func_call: Callable, redis_key: str = None, lock_timeout: int = 600,
                         logger_name: Optional[str] = None, **kwargs):
    redis = get_redis()
    if not redis:
        return await func_call(**kwargs)
    logger = logging.getLogger(logger_name)
    func_name = getattr(func_call, "__qualname__", getattr(func_call, "__name__", repr(func_call)))
    if not redis_key:
        redis_key = f'lock:{func_name}'
    lock_value = str(uuid.uuid4())  # str(time.time())，每个worker使用唯一的lock_value
    lock_acquired = await redis.set(redis_key, lock_value, nx=True, ex=lock_timeout)
    if not lock_acquired:
        logger.info(f"⚠️ 分布式锁已被占用，跳过任务: {func_name}")
        return None

    res = None
    try:
        logger.info(f"🔒 获取锁成功，开始执行任务: {func_name}")
        res = await func_call(**kwargs)
    except Exception as e:
        logger.error(f"⚠️ 任务执行出错: {func_name} -> {e}")
    finally:
        # current_lock_value = await redis.get(redis_key)
        # if current_lock_value and current_lock_value == lock_value:
        #     await redis.delete(redis_key)
        # 使用 Lua 脚本保证原子性，确保只有锁持有者能释放，只有最初获取锁的那个worker才能成功删除锁
        lua_script = """
           if redis.call("get", KEYS[1]) == ARGV[1] then
               return redis.call("del", KEYS[1])
           else
               return 0
           end
           """
        await redis.eval(lua_script, 1, redis_key, lock_value)

    return res


def get_dask_client(cluster=None):
    global _dask_client
    if not _dask_client:
        try:
            if not cluster:
                # 本机上启动若干个 worker 进程（和一个 scheduler) http://127.0.0.1:8787
                cluster = LocalCluster(scheduler_port=8786, n_workers=4, threads_per_worker=1)
            _dask_client = DaskClient(cluster)  # 创建Dask客户端 'tcp://127.0.0.1:8786
            # print(_dask_client.get_versions(check=True))
        except OSError:
            print("Scheduler端口被占用，连接已有集群")
            _dask_client = DaskClient("tcp://127.0.0.1:8786", compression=None)
    return _dask_client


def get_neo_driver():
    global _graph_driver
    if _graph_driver is None:
        _graph_driver = AsyncGraphDatabase.driver(uri=Config.NEO_URI, auth=(Config.NEO_Username, Config.NEO_Password),
                                                  max_connection_lifetime=3600,  # 单连接生命周期
                                                  max_connection_pool_size=30,  # 最大连接池数量
                                                  connection_timeout=30  # 超时
                                                  )
    return _graph_driver


def get_w3():
    try:
        from web3 import Web3

        w3 = Web3(
            Web3.HTTPProvider(f'https://mainnet.infura.io/v3/{Config.INFURA_PROJECT_ID}'))  # ("http://127.0.0.1:8545")
        return w3
    except ImportError:
        print("[Web3 Init] Web3 not installed.")
    except Exception as e:
        print(f"[Web3 Init] Failed to get web3: {e}")

    return None


def error_logger(extra_msg=None, logger_name=None):
    """
    错误日志装饰器 @error_logger()
    参数:
        logger_name (str): 日志器名称
    """
    logger = logging.getLogger(logger_name)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # logger.debug(f"Entering {func.__name__}")
                return func(*args, **kwargs)
            except Exception as e:
                msg = f"Error in {func.__name__}: {e}"
                if extra_msg:
                    msg += f" | Extra: {extra_msg}"
                logger.error(msg, exc_info=True)
                raise  # 重新抛出异常

        return wrapper

    return decorator


def async_error_logger(max_retries: int = 0, delay: int | float = 1, backoff: int | float = 2,
                       exceptions: Type[Exception] | tuple[Type[Exception], ...] = Exception,
                       extra_msg: str = None, logger_name: Optional[str] = None, log_level: int = logging.ERROR):
    """
    异步函数的错误重试和日志记录装饰器

    参数:
        max_retries (int): 最大重试次数（不含首次尝试），默认为 0，表示不重试；设为 1 表示失败后重试一次（共尝试 2 次）。
        delay (int/float): 初始延迟时间(秒)，默认为1
        backoff (int/float): 延迟时间倍增系数，默认为2
        exceptions (Exception/tuple): 要捕获的异常类型，默认为所有异常
        logger_name (Logger): 自定义logger，默认使用模块logger
        log_level (int): 日志级别
    """
    logger = logging.getLogger(logger_name)

    def decorator(func: Callable[..., Awaitable]):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 优先使用调用时的参数，如果没有就用装饰器默认值
            _max_retries = kwargs.pop("max_retries", max_retries)
            _delay = kwargs.pop("delay", delay)
            _backoff = kwargs.pop("backoff", backoff)
            _extra_msg = kwargs.pop("extra_msg", extra_msg)

            retry_count = 0
            current_delay = _delay

            while True:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    retry_count += 1
                    msg = f"Async function {func.__name__} failed with error: {str(e)}."
                    if _extra_msg:
                        msg += f" | Extra: {_extra_msg}"
                    if retry_count > _max_retries:
                        logger.log(log_level, f"{msg} After {_max_retries} retries", exc_info=True)
                        raise  # 重试次数用尽后重新抛出异常

                    logger.log(log_level, f"{msg} Retrying {retry_count}/{_max_retries} in {current_delay} seconds...")
                    await asyncio.sleep(current_delay)
                    current_delay *= _backoff  # 指数退避

        return wrapper

    return decorator


@async_error_logger(1)
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
    limits = httpx.Limits(max_connections=100, max_keepalive_connections=50)
    transport = httpx.AsyncHTTPTransport(proxy=Config.HTTP_Proxy)
    # proxies = {"http://": Config.HTTP_Proxy, "https://": Config.HTTP_Proxy}
    # http_client = DefaultHttpxClient(proxy="http://my.test.proxy.example.com", transport=httpx.HTTPTransport(local_address="0.0.0.0"))
    for model in ai_models:
        model_name = model.get('name')
        api_key = model_api_keys(model_name)
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
        parts = name.split(':', 1)
        owner, model_name = parts[0], parts[1]
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
