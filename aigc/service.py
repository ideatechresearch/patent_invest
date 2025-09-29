import httpx, aiohttp, aiofiles, requests
import asyncio
import logging
import json, uuid, os, time
from typing import Callable, Optional, Type, Dict, List, Tuple, Any, Union, Awaitable, AsyncIterator, AsyncGenerator, \
    Coroutine

from contextlib import asynccontextmanager
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
# from starlette.routing import Route, Mount
from logging.handlers import RotatingFileHandler
from utils import async_to_sync, generate_hash_key, parse_database_uri, is_port_open, chunks_iterable, get_file_type_wx
from config import Config, AI_Models, model_api_keys

# Config.load('config.yaml')
# if os.getenv('AIGC_DEBUG', '0').lower() in ('1', 'true', 'yes'):
#     Config.debug()
import pymysql, aiomysql
from neo4j import GraphDatabase, AsyncGraphDatabase
from dask.distributed import Client as DaskClient, LocalCluster
from qdrant_client import AsyncQdrantClient, QdrantClient
from openai import AsyncOpenAI, OpenAI
from fastmcp import FastMCP, Context as MCPContext, Client as MCPClient, settings
import oss2
# https://gofastmcp.com/servers/context
from redis.asyncio import Redis, StrictRedis, ConnectionPool
from functools import partial, wraps
import inspect

_httpx_clients: Dict[str, httpx.AsyncClient] = {}
_graph_driver: Optional[GraphDatabase] = None
# _graph_driver_lock = asyncio.Lock()  # 防止并发初始化
_redis_client: Optional[Redis] = None  # StrictRedis(host='localhost', port=6379, db=0)
_redis_pool: Optional[ConnectionPool] = None
_dask_cluster: Optional[LocalCluster | str] = None
_dask_client: Optional[DaskClient] = None
logger = logging.getLogger(__name__)
AI_Client: Dict[str, Optional[AsyncOpenAI]] = {}  # OpenAI
QD_Client = AsyncQdrantClient(host=Config.QDRANT_HOST, grpc_port=Config.QDRANT_GRPC_PORT,
                              prefer_grpc=True) if Config.QDRANT_GRPC_PORT else AsyncQdrantClient(url=Config.QDRANT_URL)

AliyunBucket = oss2.Bucket(oss2.Auth(Config.ALIYUN_oss_AK_ID, Config.ALIYUN_oss_Secret_Key), Config.ALIYUN_oss_endpoint,
                           Config.ALIYUN_Bucket_Name)
mcp = FastMCP(name="FastMCP Server")  # Create a server instance,main_mcp


# dependencies=["pandas", "matplotlib", "requests"]


def setup_logging(file_name="app.log", level=logging.WARNING):
    log_handler = RotatingFileHandler(file_name, maxBytes=1_000_000, backupCount=3, encoding='utf-8')
    # 文件日志logging.FileHandler('errors.log')
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=level, encoding='utf-8',
                        handlers=[
                            logging.StreamHandler(),  # 输出到终端,控制台输出
                            log_handler
                        ])


def get_httpx_client(time_out: float = None, proxy: str = None) -> httpx.AsyncClient:
    # @asynccontextmanager
    key = proxy or "default"
    global _httpx_clients
    if key not in _httpx_clients or _httpx_clients[key].is_closed:
        transport = httpx.AsyncHTTPTransport(proxy=proxy or None)
        limits = httpx.Limits(max_connections=Config.MAX_HTTP_CONNECTIONS,
                              max_keepalive_connections=Config.MAX_KEEPALIVE_CONNECTIONS)
        timeout = httpx.Timeout(timeout=time_out or Config.HTTP_TIMEOUT_SEC, read=60.0, write=30.0, connect=5.0)
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


def get_redis(db: int = 0) -> Optional[Redis]:
    global _redis_client, _redis_pool
    if _redis_client is None:
        _redis_pool = ConnectionPool(host=Config.REDIS_HOST, port=Config.REDIS_PORT, db=db,
                                     decode_responses=True,  # 自动解码为字符串
                                     max_connections=Config.REDIS_MAX_CONCURRENT
                                     )
        _redis_client = Redis(connection_pool=_redis_pool)

    return _redis_client


async def shutdown_redis():
    global _redis_client, _redis_pool
    if _redis_client:
        await _redis_client.aclose()
        _redis_client = None

    if _redis_pool:
        await _redis_pool.disconnect()
        _redis_pool = None


async def check_redis_connection(redis):
    try:
        await redis.ping()
        print("✅ Redis connected.")
        return True
    except ConnectionError as e:
        print(f"❌ Redis connection failed: {e}")
    return False


async def get_redis_connection():
    redis = get_redis()
    if not await check_redis_connection(redis):
        return None
    return redis


async def get_redis_retry(redis, key: str, retry: int = 3, delay: float = 0.1):
    for attempt in range(retry):
        try:
            return await redis.get(key)
        except Exception as e:
            print(f"[Redis GET] attempt {attempt + 1} failed: {e}")
            await asyncio.sleep(delay)
    raise Exception(f"Redis GET failed after {retry} retries.")


async def get_redis_value(redis, key: str) -> Union[Dict, list, str, int, float, None]:
    """
    Redis 值获取方法，支持通配符查询和 JSON 解析

    Args:
        key: 要查询的键名（支持通配符）
        redis: Redis 客户端实例

    Returns:
        根据内容返回解析后的数据
    """

    def parse_json(value: str) -> Any:
        if not isinstance(value, str):
            return value

        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value

    try:
        # 检查是否使用通配符
        if any(x in key for x in ("*", "?", "[")):
            keys = await redis.keys(key)
            if not keys:
                return None

            values = await redis.mget(*keys)
            result = {}
            for k, v in zip(keys, values):
                # 键名解码
                key_str = k.decode("utf-8") if isinstance(k, bytes) else k
                if v is None:
                    result[key_str] = None
                else:  # 值处理和 JSON 解析尝试
                    result[key_str] = parse_json(v.decode("utf-8") if isinstance(v, bytes) else v)

            return result

        else:
            # 单个键查询
            value = await redis.get(key)
            if value is None:
                return None
            if isinstance(value, bytes):
                value = value.decode('utf-8')

            return parse_json(value)

    except (ConnectionError, TimeoutError) as e:
        # Connect call failed,redis.exceptions.ConnectionError
        raise Exception(f"Redis 连接失败,detail:{e}")
    except Exception as e:
        raise Exception(f"Redis 查询错误,detail:{e}")


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


async def stream_to_redis(redis, batch: list, key: str = 'streams'):
    pipe = redis.pipeline()
    for stream_id, chunk in batch:
        stream_name = f"{key}:{stream_id % 3}"  # 选择分片流名
        pipe.xadd(stream_name, fields=chunk, id="*", maxlen=10000, approximate=True)  # {"data": json.dumps(chunk)}
    try:
        await pipe.execute()
    except Exception as e:
        raise


async def run_with_lock(func_call: Callable, *args, lock_timeout: int = 600, lock_key: str = None, redis=None,
                        **kwargs):
    redis = redis or get_redis()
    if not redis:
        return await func_call(*args, **kwargs)
    func_name = getattr(func_call, "__qualname__", getattr(func_call, "__name__", repr(func_call)))
    lock_key = lock_key or f'lock:{func_name}'
    lock_value = str(uuid.uuid4())  # str(time.time())，每个worker使用唯一的lock_value
    lock_acquired = await redis.set(lock_key, lock_value, nx=True, ex=lock_timeout)
    if not lock_acquired:
        logger.info(f"⚠️ 分布式锁已被占用，跳过任务: {func_name}")
        return None

    result = None
    try:
        logger.info(f"🔒 获取锁成功，开始执行任务: {func_name}")
        result = await func_call(*args, **kwargs)
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
        await redis.eval(lua_script, 1, lock_key, lock_value)

    return result


def distributed_lock(lock_timeout: int = 600, redis_key: Optional[str] = None):
    '''
    locked_operation = distributed_lock(lock_timeout=300)(my_task)    手动应用装饰器,临时需要加锁的函数
    await locked_operation(123, {"name": "John"})
    @distributed_lock(300) 长期使用的任务函数
    :param lock_timeout:
    :param redis_key:
    :return:
    '''

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            redis = get_redis()
            if not redis:
                return await func(*args, **kwargs)

            lock_key = redis_key or f"lock:{func.__qualname__}"

            # 尝试获取锁
            async with with_distributed_lock(lock_key, None, lock_timeout * 1000, redis) as lock_acquired:
                if not lock_acquired:
                    logger.info(f"⚠️ 分布式锁已被占用，跳过任务: {func.__qualname__}")
                    return None

                logger.info(f"🔒 获取锁成功，开始执行任务: {func.__qualname__}")
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"⚠️ 任务执行出错: {func.__qualname__} -> {e}")
                    raise

        return wrapper

    return decorator


@asynccontextmanager
async def with_distributed_lock(lock_key: str, lock_value: str = None, lock_timeout=10000, redis=None,
                                release: bool = True):
    """
    分布式锁上下文管理器
    lock_timeout # 毫秒
    用法：
    async with with_distributed_lock( "my_lock",None,10000,redis_conn) as lock_acquired:
        if lock_acquired:
            # 执行受保护的操作
    """
    redis_conn = redis or get_redis()
    lock_identifier = lock_value or str(uuid.uuid4())
    acquired = await redis_conn.set(lock_key, lock_identifier, nx=True, px=lock_timeout)

    try:
        yield acquired
    finally:
        if acquired and release:
            # 原子性释放锁
            script = """
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("del", KEYS[1])
            else
                return 0
            end
            """
            await redis_conn.eval(script, 1, lock_key, lock_identifier)


async def is_main_worker(worker_id: str = None, redis=None):
    async with with_distributed_lock("lock:main_worker", worker_id, lock_timeout=60 * 1000, redis=redis,
                                     release=False) as acquired:
        return acquired


def get_dask_client(cluster=None, n_workers: int = 3):
    global _dask_client, _dask_cluster
    if _dask_client:
        return _dask_client

    if cluster is None:
        if not _dask_cluster:
            if is_port_open("127.0.0.1", 8786):
                _dask_cluster = "tcp://127.0.0.1:8786"
                print("Dask Scheduler 端口被占用，连接已有集群")
            else:
                # 启动本地 Dask 集群,本机上启动若干个 worker 进程,使用线程而不是进程（和一个 scheduler) http://127.0.0.1:8787
                _dask_cluster = LocalCluster(scheduler_port=8786, dashboard_address=":8787",
                                             n_workers=n_workers, threads_per_worker=1, processes=True)

        cluster = _dask_cluster

    try:
        _dask_client = DaskClient(cluster, timeout=3)  # 创建Dask客户端, compression=None
        print(_dask_client.ncores())  # _dask_client.get_versions(check=True)
    except Exception as e:
        print(f"❌ 无法创建 Dask Client: {e}")
        # raise RuntimeError(f"❌ 无法创建 Dask Client: {e}")

    return _dask_client


def close_dask_client():
    global _dask_client, _dask_cluster
    # print("Closing Dask client or cluster...")
    if _dask_client:
        _dask_client.close()
        _dask_client = None
    if _dask_cluster:
        if isinstance(_dask_cluster, LocalCluster):
            _dask_cluster.close()
        _dask_cluster = None


def get_neo_driver():
    global _graph_driver
    if _graph_driver is None:
        _graph_driver = AsyncGraphDatabase.driver(uri=Config.NEO_URI,  # uri="bolt://localhost:7687"
                                                  auth=(Config.NEO_Username, Config.NEO_Password),
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


def error_logger(extra_msg=None):
    """
    错误日志装饰器 @error_logger()
    """

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
                       extra_msg: str = None, log_level: int = logging.ERROR):
    """
    异步函数的错误重试和日志记录装饰器

    参数:
        max_retries (int): 最大重试次数（不含首次尝试），默认为 0，表示不重试；设为 1 表示失败后重试一次（共尝试 2 次）。
        delay (int/float): 初始延迟时间(秒)，默认为1
        backoff (int/float): 延迟时间倍增系数，默认为2
        exceptions (Exception/tuple): 要捕获的异常类型，默认为所有异常
        log_level (int): 日志级别
    """

    def decorator(func: Callable[..., Awaitable]):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 优先使用调用时的参数，如果没有就用装饰器默认值
            _max_retries = kwargs.pop("max_retries", max_retries)
            _delay = kwargs.pop("delay", delay)
            _backoff = kwargs.pop("backoff", backoff)
            _extra_msg = kwargs.pop("extra_msg", extra_msg)

            attempt = 0
            current_delay = _delay

            while True:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    msg = f"Async function {func.__name__} failed with error: {str(e)}."
                    if _extra_msg:
                        msg += f" | Extra: {_extra_msg}"
                    if attempt > _max_retries:
                        logger.log(log_level, f"{msg} After {_max_retries} retries", exc_info=True)
                        raise  # 重试次数用尽后重新抛出异常

                    logger.log(log_level, f"{msg} Retrying {attempt}/{_max_retries} in {current_delay} seconds...")
                    await asyncio.sleep(current_delay)
                    current_delay *= _backoff  # 指数退避

        return wrapper

    return decorator


def task_processor_worker(max_retries: int = 0, delay: float = 1.0, backoff: int | float = 2,
                          timeout: int | float = -1):
    """
    任务处理装饰器，封装任务执行、重试和异常处理逻辑
    @task_processor_worker(max_retries=3, delay=0.5,timeout=10)
    async def handle_task(task: tuple[Any, int], queue: asyncio.Queue[tuple[Any, int]], **kwargs):
        print(f"处理任务: {task}")
        ...

        task_data,X,Y= task
        result = await async_processing(task_data, **kwargs)
        if task == "error":
            queue.put_nowait(task_data)
            #raise ValueError("异常")
        return True

    @task_processor_worker(max_retries=2)
    async def handle_task(task: tuple, **kwargs):
        print(f"任务数据: {task}")
        if result == "error":
            raise ValueError("异常")

    queue = asyncio.Queue()  asyncio.Queue 是内存对象，只存在于当前进程的事件循环中
    worker_task = asyncio.create_task(handle_task(queue=queue))
    启动 worker:
        workers_task_background = [asyncio.create_task(handle_task(queue)) for _ in range(4)],
        #await asyncio.gather(*tasks)
        #await queue.put((f"task-{i}", 0))
        #await queue.put(None)

    :param queue:asyncio.Queue()
    :param func:注入处理函数 async def f(task: tuple, **kwargs),返回值 true,false..
    :param max_retries,0不重试
    :param delay: 重试前延迟时间（秒）
    :param backoff:
    :param timeout: 超时时间（秒）
    :return:
    """

    def decorator(func: Callable):
        async def wrapper(queue: asyncio.Queue, **kwargs):
            # 判断 func 是否接收 queue 参数
            accepts_queue = 'queue' in inspect.signature(func).parameters

            while True:
                task_data = await queue.get()
                if task_data is None:
                    logger.info("[Info] Received shutdown signal")
                    queue.task_done()
                    break

                try:
                    process_func = func
                    if max_retries > 0:
                        process_func = async_error_logger(max_retries=max_retries, delay=delay, backoff=backoff)(
                            func)  # 处理重试逻辑
                    if timeout > 0:
                        if accepts_queue:
                            success = await asyncio.wait_for(process_func(task_data, queue, **kwargs), timeout=timeout)
                        else:
                            success = await asyncio.wait_for(process_func(task_data, **kwargs), timeout=timeout)
                    else:
                        if accepts_queue:
                            success = await process_func(task_data, queue, **kwargs)
                        else:
                            success = await process_func(task_data, **kwargs)  # 执行实际的任务处理

                    if not success:
                        logger.error(f"[Task Failed] {task_data}")

                except asyncio.TimeoutError:
                    logger.error(f"[Timeout] Task {task_data} timed out")
                    # await queue.put(task),.put_nowait(x)
                except asyncio.CancelledError:
                    logger.info("[Cancel] Worker shutting down...")
                    break
                except Exception as e:
                    logger.error(f"[Error] Unexpected error processing task: {e}")
                finally:
                    queue.task_done()  # 必须调用，标记任务完成

        return wrapper

    return decorator


def start_consumer_workers(queue: asyncio.Queue, worker_func: Callable, num_workers: int = 4, **kwargs) -> list:
    # worker 启动器，workers_task_background
    return [asyncio.create_task(worker_func(queue=queue, **kwargs)) for _ in range(num_workers)]


async def stop_worker(queue: asyncio.Queue, worker_tasks: list):
    '''优雅停止所有 worker'''
    try:
        await queue.join()  # 等待队列清空

        logger.info("All tasks processed. Stopping consumers...")
        for _ in worker_tasks:
            await queue.put(None)  # 发送停止信号
    except Exception as e:
        logger.error(f"[Tasks Error] {e}, attempting to cancel workers...")
        for c in worker_tasks:
            c.cancel()

    finally:
        # 统一回收所有任务
        await asyncio.gather(*worker_tasks, return_exceptions=True)


def async_timer_cron(interval: float = 60) -> Callable[
    [Callable[..., Coroutine[Any, Any, None]]], Callable[..., Coroutine[Any, Any, None]]]:
    """
    定时执行异步任务的装饰器

    :param interval: 执行间隔时间（秒）
    """

    def decorator(func: Callable[..., Coroutine[Any, Any, None]]) -> Callable[..., Coroutine[Any, Any, None]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> None:
            while True:
                try:
                    await func(*args, **kwargs)
                except Exception as e:
                    logging.error(f"Error in timed task {func.__name__}: {e}")

                await asyncio.sleep(interval)

        return wrapper

    return decorator


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
def oss_list_files(bucket, prefix='upload/', max_keys: int = 100, max_pages: int = 1):
    """
    列出 OSS 中的文件。
    :param bucket: oss2.Bucket 实例
    :param prefix: 文件名前缀，用于筛选
    :param max_keys: 每次返回的最大数量
    :param max_pages:
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


class BaseMysql:
    def __init__(self, host: str, user: str, password: str, db_name: str,
                 port: int = 3306, charset: str = "utf8mb4"):
        self.host = host
        self.user = user
        self.password = password
        self.db_name = db_name
        self.port = port
        self.charset = charset

    def close(self):
        """同步实现中关闭连接；子类实现"""
        raise NotImplementedError

    async def close_pool(self):
        """异步实现中关闭连接池；子类实现"""
        raise NotImplementedError

    def table_schema(self, table_name: str) -> tuple[str, tuple]:
        if not table_name:
            raise ValueError("[Schema] 表名不能为空")
        sql = """
                 SELECT column_name, data_type, is_nullable, column_type, column_comment
                 FROM information_schema.columns
                 WHERE table_schema = %s AND table_name = %s
                 ORDER BY ordinal_position
             """
        params = (self.db_name, table_name)
        return sql, params

    @staticmethod
    def format_value(value) -> Union[str, int, float, bool, None]:
        """
        保留你原来的 format_value 行为（dict->json, list/tuple->\n\n if all str else json, set->; or json）
        """
        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)  # 处理字典类型 → JSON序列化, indent=2

        if isinstance(value, (tuple, list)):
            if all(isinstance(item, str) for item in value):
                return "\n\n---\n\n".join(value)  # "\n\n"
            return json.dumps(value, ensure_ascii=False)  # 非全字符串元素则JSON序列化

        if isinstance(value, set):
            if all(isinstance(item, str) for item in value):
                return ";".join(sorted(value))
            return json.dumps(list(value), ensure_ascii=False)

        return str(value)  # 其他类型保持原样 (None等)

    @staticmethod
    def build_insert(table_name: str, params_data: dict, update_fields: list[str] | None = None) -> tuple[str, tuple]:
        """
        生成插入 SQL（可选 ON DUPLICATE KEY UPDATE）

        Args:
            table_name: 表名
            params_data: 数据字典
            update_fields: 冲突时更新的字段，None 表示默认更新非主键字段

        Returns:
            tuple: (sql, values)
        """
        if not params_data:
            raise ValueError("params_data 不能为空")

        fields = list(params_data.keys())
        values = tuple(params_data.values())  # [tuple(row[f] for f in fields) for row in params_data]
        field_str = ', '.join(f"`{field}`" for field in fields)  # columns_str
        placeholder_str = ', '.join(['%s'] * len(fields))
        sql = f"INSERT INTO `{table_name}` ({field_str}) VALUES ({placeholder_str})"

        if update_fields is None:
            update_fields = [f for f in fields if f.lower() not in ("id", "created_at", "created_time")]

        if update_fields:
            sql += " AS new"
            update_str = ', '.join(f"`{field}` = new.`{field}`" for field in update_fields)
            if "updated_at" in fields and "updated_at" not in update_fields:
                update_str += ", `updated_at` = CURRENT_TIMESTAMP"
            sql += f" ON DUPLICATE KEY UPDATE {update_str}"

        return sql, values

    @staticmethod
    def get_dataframe(db_config: dict, sql: str, params: tuple | dict = None):
        import pandas as pd
        # with create_engine(SQLALCHEMY_DATABASE_URI).connect() as conn:
        conn = pymysql.connect(**db_config)
        try:
            with conn.cursor() as cursor:
                cursor.execute(sql, params or ())
                rows = cursor.fetchall()
                cols = [desc[0] for desc in cursor.description]
                return pd.DataFrame(rows, columns=cols)
        except Exception as e:
            return pd.read_sql(sql, conn, params=params)
        finally:
            conn.close()

    @staticmethod
    def query_dataframe_process(conn, sql: str, process_chunk: Callable = None, params: tuple | dict = None,
                                chunk_size: int = 100000, aggregate: bool = False):
        """
        通用大表分页查询并处理每块数据。

        Args:
            conn: 数据库连接对象（需支持 conn.cursor()）
            sql (str): 原始 SQL 查询语句（不含 LIMIT 和 OFFSET）
            process_chunk (Callable): 对每个批次的进行处理的函数
            chunk_size (int): 每批次读取的记录数
            params(tuple|dict|None): SQL 查询参数
            aggregate (bool): 是否把所有 DataFrame 合并成一个大 DataFrame
        Returns:
            list | pd.DataFrame: 返回所有处理结果的列表
        """
        import pandas as pd

        offset = 0
        results = []
        chunk_query = f"{sql} LIMIT %s OFFSET %s"

        with conn.cursor() as cur:
            while True:
                if isinstance(params, tuple):
                    query_params = (*params, chunk_size, offset)
                elif isinstance(params, dict):
                    query_params = {**params, "limit": chunk_size, "offset": offset}
                else:
                    query_params = (chunk_size, offset)

                cur.execute(chunk_query, query_params)
                rows = cur.fetchall()
                if not rows:
                    break

                df = pd.DataFrame(rows)
                if not df.empty:
                    results.append(process_chunk(df) if process_chunk else df)

                offset += chunk_size

        if aggregate and not process_chunk:
            return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

        return results


class SyncMysql(BaseMysql):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conn: Optional[pymysql.connections.Connection] = None
        self.cur = None

    def __del__(self):
        if self.conn or self.cur:
            print("[OperationMysql] Warning:正在自动清理,关闭数据库连接。")
            self.close()

    def __enter__(self):
        # 打开连接
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 关闭游标和连接
        self.close()

    def connect(self):
        self.close()
        try:
            self.conn = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                db=self.db_name,
                charset=self.charset,
                cursorclass=pymysql.cursors.DictCursor  # 这个定义使数据库里查出来的值为字典类型
            )
            self.cur = self.conn.cursor()  # 原生数据库连接方式
        except Exception as e:
            print(f"[Sync] 连接数据库失败: {e}")
            self.conn = None
            self.cur = None

    def get_conn(self):
        if not self.conn:
            self.connect()
        return self.conn

    def close(self):
        if self.cur:
            self.cur.close()
            self.cur = None
        if self.conn:
            self.conn.close()
            self.conn = None

    def ensure_connection(self):
        try:
            if self.conn:
                self.conn.ping(reconnect=True)  # 已连接且健康
            else:
                self.connect()
        except Exception as e:
            print(f"[Sync] 自动重连失败: {e}")
            self.connect()

    def run(self, sql: str, params: tuple | dict | list = None):
        sql_type = (sql or "").strip().split()[0].lower()
        self.ensure_connection()
        try:
            if isinstance(params, list):
                self.cur.executemany(sql, params)  # 批量执行
            else:
                self.cur.execute(sql, params or ())  # 单条执行

            if sql_type == "select":
                return self.cur.fetchall()

            elif sql_type in {"insert", "update", "delete", "replace"}:
                self.conn.commit()
                if sql_type == "insert":
                    return self.cur.lastrowid
                return True

        except Exception as e:
            if self.conn:
                self.conn.rollback()
            print(f"[Sync] 数据库执行出错: {e}")
        return None

    def search(self, sql: str, params: tuple | dict = None):
        if not sql.lower().startswith("select"):
            raise ValueError("search 方法只能执行 SELECT 语句")
        if params is None:
            params = ()
        self.cur.execute(sql, params)
        result = self.cur.fetchall()
        return result

    def execute(self, sql: str, params: tuple | dict | list = None):
        # INSERT,UPDATE,DELETE
        try:
            if isinstance(params, list):
                self.cur.executemany(sql, params)  # 批量执行
            else:
                self.cur.execute(sql, params or ())  # 单条执行
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            print(f"执行 SQL 出错: {e}")

    def insert(self, sql: str = None, params: tuple | dict = None, table_name: str = None):
        # 单条 INSERT 语句，且目标表有 AUTO_INCREMENT 字段
        if isinstance(params, dict) and table_name:
            fields = tuple(params.keys())
            values = tuple(params.values())
            field_str = ', '.join(f"`{field}`" for field in fields)
            placeholder_str = ', '.join(['%s'] * len(fields))
            sql = f"INSERT INTO `{table_name}` ({field_str}) VALUES ({placeholder_str})"
            params = values
        try:
            self.cur.execute(sql, params or ())
            self.conn.commit()
            return self.cur.lastrowid or int(self.conn.insert_id())
        except Exception as e:
            if self.conn:
                self.conn.rollback()
            print(f"执行 SQL 出错: {e}")
            print(f"SQL: {repr(sql)} \n 参数: {params}")
        # finally:
        #     self.cur.close()
        #     self.conn.close()
        return -1

    def query_batches(self, ids: list | tuple, index_key: str, table_name: str, fields: list = None, chunk_size=10000):
        """
        await asyncio.to_thread
        大批量 IN 查询，分批执行，避免 SQL 参数溢出。
        Args:
            ids (list | tuple): 要查找的 ID 列表
            index_key (str): 作为筛选条件的字段
            table_name (str): 表名
            fields (list): 返回的字段
            chunk_size (int): 每次 IN 的最大数量<65535
        Returns:
            list[dict]: 查询结果列表
        """
        if not ids:
            return []
        field_str = ', '.join(f"`{field}`" for field in fields) if fields else '*'

        result_rows = []
        for batch in chunks_iterable(ids, chunk_size):
            placeholders = ', '.join(['%s'] * len(batch))
            sql = f"SELECT {field_str} FROM `{table_name}` WHERE `{index_key}` IN ({placeholders})"
            self.cur.execute(sql, tuple(batch))
            result_rows.extend(self.cur.fetchall())
            # filtered_chunk = pd.read_sql_query(text(sql+'`{index_key}` in :ids'), conn, params={'ids': batch})
            # df_chunk.to_sql(table_name, con=engine,chunksize=chunk_size, if_exists='append', index=False)
        return result_rows

    def get_table_schema(self, table_name: str) -> list:
        """
        获取指定表的结构信息（原始 implementation），用于自然语言描述
        参数:
            table_name: 表名（不含数据库名）
        返回:
            表结构列表，每列包含 column_name, data_type, is_nullable, column_type, column_comment
        """
        try:
            self.ensure_connection()
            sql, params = self.table_schema(table_name)
            return self.search(sql, params)
        except Exception as e:
            print(f"[Schema] 获取表结构失败: {str(e)}")
        return []


class AsyncMysql(BaseMysql):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pool: Optional[aiomysql.Pool] = None
        self.conn = None  # used only when temporarily acquiring

    async def __aenter__(self):
        if self.pool is None:
            await self.init_pool()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close_pool()

    async def init_pool(self, minsize: int = 1, maxsize: int = 30, autocommit: bool = True):
        if self.pool is not None:
            return
        try:
            self.pool = await aiomysql.create_pool(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                db=self.db_name,
                charset=self.charset,
                autocommit=autocommit,
                minsize=minsize,
                maxsize=maxsize
            )
        except Exception as e:
            print(f"[Async] 创建连接池失败: {e}")
            self.pool = None

    async def close_pool(self):
        if self.pool is not None:
            self.pool.close()
            await self.pool.wait_closed()
            self.pool = None

    async def get_conn(self):
        if self.pool is None:
            await self.init_pool()
        return await self.pool.acquire()

    def release(self, conn):
        if self.pool is not None:
            self.pool.release(conn)

    @asynccontextmanager
    async def get_cursor(self, conn=None) -> AsyncIterator[aiomysql.Cursor]:
        """获取游标（支持自动释放）
        释放 await cursor.close()
        Args:
            conn: 外部传入的连接对象。如果为None，则自动创建新连接

        Yields:
            aiomysql.Cursor: 数据库游标

        注意：
            - 当conn为None时，会自动创建并最终释放连接
            - 当conn由外部传入时，不会自动释放连接
        """
        should_release = False
        if conn is None:
            conn = await self.get_conn()
            should_release = True

        try:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                yield cursor
        finally:
            if should_release:
                self.release(conn)

    async def async_run(self, sql: str, params: tuple | dict | list = None, conn=None):
        async def _run(c):
            sql_type = (sql or "").strip().split()[0].lower()
            async with c.cursor(aiomysql.DictCursor) as cur:
                if isinstance(params, list):
                    await cur.executemany(sql, params)
                else:
                    await cur.execute(sql, params or ())

                if sql_type == "select":
                    return await cur.fetchall()
                elif sql_type in {"insert", "update", "delete", "replace"}:
                    await c.commit()  # 显式保险,autocommit=True
                    if sql_type == "insert":
                        return cur.lastrowid or int(c.insert_id())
                    else:
                        return True

                return None

        try:
            if conn:
                return await _run(conn)
            if self.pool is None:
                await self.init_pool()
            async with self.pool.acquire() as conn:
                return await _run(conn)

        except Exception as e:
            print(f"[Async] SQL执行错误: {e}, SQL={sql}\nVALUE={params}")
        return None

    async def async_execute(self, sql_list: list[tuple[str, tuple | dict | list | None]], conn=None):
        """
        批量执行多条 SQL 并自动提交或回滚（同一个事务）
        :param sql_list: 形如 [(sql1, params1), (sql2, params2), ...]
        :param conn
        """
        if conn:
            should_release = False
        else:
            if self.pool is None:
                await self.init_pool()
            conn = await self.pool.acquire()
            should_release = True

        try:
            async with conn.cursor() as cur:
                for sql, params in sql_list:
                    if not sql.strip():
                        continue
                    if isinstance(params, list):
                        await cur.executemany(sql, params)
                    else:
                        await cur.execute(sql, params or ())
                await conn.commit()
                return True

        except Exception as e:
            print(f"[Async] 批量 SQL 执行失败: {e}")
            try:
                await conn.rollback()
            except Exception as rollback_err:
                print(f"[Async] 回滚失败: {rollback_err}")
            return False

        finally:
            if should_release:
                self.release(conn)

    async def async_query(self, query_list: list[tuple[str, tuple | dict]], fetch_all: bool = True,
                          cursor=None) -> list:
        """
        执行多个查询，分别返回结果列表
        :param query_list: [(sql1, params1), (sql2, params2), ...]
        :param fetch_all: True 表示 fetchall，False 表示 fetchone
        :param cursor: 可选外部 cursor
        :return: [result1, result2, ...]
        """

        async def _run(c) -> list:
            results = []
            for sql, params in query_list:
                await c.execute(sql, params or ())
                result = await c.fetchall() if fetch_all else await c.fetchone()
                results.append(result)
            return results

        if cursor:
            return await _run(cursor)

        if self.pool is None:
            await self.init_pool()

        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                return await _run(cur)

    async def query_one(self, sql: str, params: tuple | dict = (), cursor=None) -> dict | None:
        results = await self.async_query([(sql, params)], fetch_all=False, cursor=cursor)
        return results[0] if results else None

    async def async_insert(self, table_name: str, params_data: dict, conn=None):
        """
        插入数据
        Args:
            table_name (str): 表名
            params_data (dict): 要插入的数据（更新必须包含主键/唯一索引字段）
            conn:可选外部传入连接
        """
        sql, values = self.build_insert(table_name, params_data, update_fields=[])
        return await self.async_run(sql, values, conn=conn)

    async def async_merge(self, table_name: str, params_list: list[dict], update_fields: list[str] = None, conn=None):
        """
        批量插入/更新数据,返回true（根据主键或唯一键自动合并）
        update_fields (list): 需要更新的字段列表，默认为除了主键以外的字段,在发生冲突时被更新的字段列表,[]为插入
        """
        if not params_list:
            raise ValueError("参数列表不能为空")
        sql_list = [self.build_insert(table_name, row, update_fields=update_fields or []) for row in params_list]
        return await self.async_execute(sql_list, conn=conn)  # list[tuple[str, tuple]]

    async def async_update(self, table_name: str, params_data: dict, row_id: int, primary_key: str = "id", conn=None):
        """
        根据主键字段更新指定行数据。

        Args:
            table_name (str): 表名
            row_id: 主键值（通常是 id）
            params_data (dict): 要更新的字段及新值
            primary_key (str): 主键字段名，默认是 'id'
            conn:可选外部传入连接
        """
        if not params_data:
            raise ValueError("更新数据不能为空")
        if not row_id:
            raise ValueError("row_id 不能为空")

        update_fields = ', '.join(f"`{k}` = %s" for k in params_data.keys())  # 构建更新字段列表
        values = tuple(params_data.values()) + (row_id,)
        sql = f"UPDATE `{table_name}` SET {update_fields} WHERE `{primary_key}` = %s"
        return await self.async_run(sql, values, conn=conn)

    async def get_offset(self, table_name: str, page: int = None, per_page: int = 10, cursor=None,
                         use_estimate: bool = True):
        total = 0
        if use_estimate:
            row = await self.query_one(
                "SELECT TABLE_ROWS AS estimate "
                "FROM information_schema.TABLES "
                "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s",
                params=(table_name,), cursor=cursor)
            total = int(row["estimate"]) if row and row.get("estimate") is not None else 0
        if (not use_estimate) or total <= 0:
            count_res = await self.query_one(f"SELECT COUNT(*) as count FROM {table_name}", cursor=cursor)
            total = int(count_res["count"]) if count_res else 0

        total_pages = (total + per_page - 1) // per_page if total > 0 else 1
        if page is None:  # default to last page when page not provided
            page = total_pages
        if page < 1:
            page = 1
        if page > total_pages:
            page = total_pages
        offset = (page - 1) * per_page
        return offset, page, total_pages, total

    async def get_table_columns(self, table_name: str, cursor=None) -> list[dict]:
        async def _run(c) -> list:
            await c.execute(f"DESCRIBE {table_name}")
            columns = await c.fetchall()
            # 转换列信息为更友好的格式
            formatted_columns = [{
                "name": col["Field"],
                "type": col["Type"],
                "nullable": col["Null"] == "YES",
                "key": col["Key"],
                "default": col["Default"],
                "extra": col["Extra"]
            } for col in columns]
            return formatted_columns

        if cursor:
            return await _run(cursor)

        if self.pool is None:
            await self.init_pool()

        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                return await _run(cur)

    async def get_primary_key(self, table_name: str, cursor=None) -> Optional[str]:
        columns = await self.get_table_columns(table_name, cursor=cursor)
        for col in columns:
            if col["key"] == "PRI":  # Primary Key,"UNI","MUL"
                return col["name"]
        return None

    async def get_table_schema(self, table_name: str, conn=None) -> list:
        """异步版本获取表结构信息"""
        sql, params = self.table_schema(table_name)
        return await self.async_run(sql, params, conn) or []


class OperationMysql(AsyncMysql, SyncMysql, BaseMysql):
    db_config = parse_database_uri(Config.SQLALCHEMY_DATABASE_URI)

    """
    dbop = OperationMysql(persistent=True,async_mode=True)
    await dbop.init_pool()
    result = await dbop.async_run("SELECT * FROM users WHERE id=%s", (...,))
    await dbop.close_pool()

    async with OperationMysql(async_mode=True) as dbop:
        result = await dbop.async_run("SELECT ...")
        
    with SyncMysql(host, user, password, db) as db:
        result = db.run("SELECT * FROM users WHERE id=%s", (...,))

    async with AsyncMysql(...) as dbop:
        await dbop.async_run(...)
    """

    def __init__(self, persistent: bool = False, async_mode: bool = False, db_config: dict = None):
        """
        persistent: 是否立即连接数据库以便长期持久化使用（不需要用 with），...close()，连接不线程安全，不同线程应用不同实例
        """

        self.persistent = persistent
        self.async_mode = async_mode
        self.config = db_config or type(self).db_config
        # super().__init__(**self.config)

        if async_mode:
            AsyncMysql.__init__(self, **self.config)
        else:
            SyncMysql.__init__(self, **self.config)

        if persistent and not async_mode:
            self.ensure_connection()
            print('[MysqlData] Sync connected.')

    async def __aenter__(self):
        if self.async_mode:
            return await AsyncMysql.__aenter__(self)
        raise RuntimeError("Use 'with' for sync mode")

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.async_mode and not self.persistent:
            await AsyncMysql.__aexit__(self, exc_type, exc_val, exc_tb)

    def __enter__(self):
        if not self.async_mode:
            return SyncMysql.__enter__(self)
        raise RuntimeError("Use 'async with' for async mode")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.async_mode:
            SyncMysql.__exit__(self, exc_type, exc_val, exc_tb)

    @classmethod
    async def get_async_conn(cls, **kwargs):
        async with cls(async_mode=True, **kwargs) as dbop:
            yield dbop


DB_Client = OperationMysql(persistent=True, async_mode=True)


class CollectorMysql(AsyncMysql):
    def __init__(self, *,
                 batch_size: int = 1000,
                 queue_maxsize: int = 10000,
                 max_wait_seconds: float = 1.0,
                 worker_count: int = 1,
                 retry_times: int = 2, retry_backoff: float = 1.0,
                 instance: AsyncMysql = None):
        '''
        collector = CollectorMysql(instance=DB_Client)
        await collector.start()
        await collector.enqueue(
        ok = collector.enqueue_nowait(
        wait collector.stop(flush=True)# 停机前清理
        '''
        if instance is not None:
            super().__init__(host=instance.host, user=instance.user, password=instance.password,
                             db_name=instance.db_name, port=instance.port, charset=instance.charset)
            self.pool = instance.pool
        else:
            config = parse_database_uri(Config.SQLALCHEMY_DATABASE_URI)
            super().__init__(**config)

        self.batch_size = batch_size
        self.batch_interval = max_wait_seconds
        self.worker_count = worker_count
        self.retry_times = retry_times
        self.retry_backoff = retry_backoff

        self._queue: asyncio.Queue = asyncio.Queue(maxsize=queue_maxsize)
        self._workers: list[asyncio.Task] = []
        self._stopped = asyncio.Event()
        self._started = False

    async def start(self):
        if self._started:
            return
        self._stopped.clear()
        for _ in range(self.worker_count):
            self._workers.append(asyncio.create_task(self._worker_loop()))
        self._started = True

    async def stop(self, flush: bool = True, timeout: float = 10.0):
        """
        优雅停机：可选先 flush 所有队列再退出
        """
        if not self._started:
            return

        if flush:
            if timeout > 0:  # 轮询等待队列为空
                start = time.time()
                while not self._queue.empty() and (time.time() - start) < timeout:
                    await asyncio.sleep(0.05)

        self._stopped.set()
        # cancel workers if they block on queue.get
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        self._started = False

        if flush and self._queue.qsize() > 0:  # flush 队列剩余数据
            try:
                await asyncio.wait_for(self.flush_all(), timeout=timeout)
            except asyncio.TimeoutError:
                print("[Collector] flush timeout, some data may not be written")

    @asynccontextmanager
    async def context(self):
        """上下文管理器，用于安全地初始化和关闭处理器"""
        await self.start()
        try:
            yield self
        finally:
            await self.stop()

    async def enqueue(self, table_name: str, params_data: dict, update_fields: list[str] = None):
        """
        将一条记录加入队列（默认阻塞直到放入，产生回压），直到队列有空位可以放入
        """
        sql, values = self.build_insert(table_name, params_data, update_fields=update_fields or [])
        await self._queue.put((sql, values))

    def enqueue_nowait(self, table_name: str, params_data: dict, update_fields: list[str] = None) -> bool:
        """
        尝试非阻塞入队，失败返回 False（可用于采样/舍弃策略），队列满不等待，直接抛出 QueueFull
        """
        try:
            sql, values = BaseMysql.build_insert(table_name, params_data, update_fields=update_fields or [])
            self._queue.put_nowait((sql, values))
            return True
        except asyncio.QueueFull:
            return False  # 选择：丢弃、降采样、写入备份文件、或写到 Redis 等持久队列

    async def _worker_loop(self):
        """
        消费者主循环：按 batch_size 或 max_wait_seconds 刷盘
        """
        while not self._stopped.is_set():
            try:
                # 用 batch_interval 等待第一个元素，避免 busy loop
                item = await asyncio.wait_for(self._queue.get(), timeout=self.batch_interval)
                # if item is None:
                #     break # propagate sentinel to stop
            except asyncio.TimeoutError:
                continue  # 没有新数据，检查 _stopped 后继续
            batch = [item]
            deadline = time.time() + self.batch_interval

            while len(batch) < self.batch_size:  # 尝试在剩余时间内凑满 batch_size
                timeout = deadline - time.time()
                if timeout <= 0:
                    break
                try:
                    item = self._queue.get_nowait()
                    batch.append(item)
                except asyncio.QueueEmpty:
                    try:  # 等待直到剩余时间结束或新数据到来
                        item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break

            if not batch:
                continue

            conn = await self.get_conn()
            try:
                await self._flush_batch(batch, conn)  # 处理 batch（异步执行 DB 写）
            except Exception as e:
                # 此处应该记录日志/告警；为防止数据丢失，可考虑把失败的 batch 放到后端持久化或重试队列
                logger.error(f"[Collector] flush batch failed: {e}")
                # 这里不 raise，让 loop 继续消费后续 batch
            finally:
                for _ in batch:  # 标记任务完成，确保释放连接
                    self._queue.task_done()
                self.release(conn)

    async def _flush_batch(self, batch: list, conn=None):
        # 分组：将字段一致的行合并到一起以便用 executemany
        groups: dict[str, list] = {}
        for sql, params in batch:
            groups.setdefault(sql, []).append(params)

        # prepare sql_list for async_execute: if a group has same fields -> executemany
        sql_list = [(sql, params_list) for sql, params_list in groups.items()]

        # 执行并带重试逻辑
        process_execute = async_error_logger(max_retries=self.retry_times, delay=1, backoff=self.retry_backoff,
                                             extra_msg="InsertCollector flush batch")(self.async_execute)
        ok = await process_execute(sql_list, conn=conn)
        if not ok:
            raise RuntimeError("flush batch failed after retries")

    async def flush_all(self):
        items = []
        while not self._queue.empty():
            items.append(await self._queue.get())
        if not items:
            return
        async with self.pool.acquire() as conn:
            await self._flush_batch(items, conn)


class AsyncBatchAdd:
    """
    通用的异步批量处理器，可用于任何 SQLAlchemy ORM 类
    """

    def __init__(
            self,
            model_class: Type,
            batch_size: int = 100,
            batch_timeout: float = 5.0,
            queue_maxsize: int = 10000,
            get_session_func: Optional[Callable] = None
    ):
        """
        初始化批量处理器

        Args:
            model_class: SQLAlchemy ORM 类
            batch_size: 每批处理的记录数量
            batch_timeout: 批处理超时时间（秒）
            get_session_func: 获取数据库会话的函数,AsyncSessionLocal
        """
        self.model_class = model_class
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.get_session_func = get_session_func  # coro

        self._queue = asyncio.Queue(maxsize=queue_maxsize)
        self._task = None
        self._is_running = False

    async def initialize(self):
        """初始化处理器，启动后台任务"""
        if not self._is_running:
            self._is_running = True
            self._task = asyncio.create_task(self._worker())
            logger.info(f"Initialized batch processor for {self.model_class.__name__}")

    async def shutdown(self):
        """关闭处理器，停止后台任务并处理剩余数据"""
        if self._is_running:
            self._is_running = False
            if self._task:
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
                self._task = None
            logger.info(f"Shutdown batch processor for {self.model_class.__name__}")

    async def put_nowait(self, data: Dict[str, Any]):
        """将数据放入队列（非阻塞）"""
        if not self._is_running:
            await self.initialize()
        await self._queue.put(data)

    def put_many_nowait(self, data_list: List[Dict]) -> int:
        """将多条数据放入队列（非阻塞）"""
        count = 0
        for data in data_list:
            try:
                self._queue.put_nowait(data)
                count += 1
            except asyncio.QueueFull:
                break
        return count

    async def _worker(self):
        """后台批量插入工作器"""
        batch = []
        last_insert_time = time.time()

        while self._is_running:
            try:
                # 等待新消息或超时
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=self.batch_timeout)
                    batch.append(item)
                    self._queue.task_done()
                except asyncio.TimeoutError:
                    pass  # 超时，继续处理当前批次

                current_time = time.time()
                # 如果批次达到指定大小或超时，执行插入
                if (len(batch) >= self.batch_size or
                        (batch and current_time - last_insert_time >= self.batch_timeout)):
                    await self.process_batch(batch)
                    batch.clear()
                    last_insert_time = current_time

            except asyncio.CancelledError:
                if batch:  # 任务被取消，处理剩余批次
                    await self.process_batch(batch)
                break
            except Exception as e:
                logger.error(f"Unexpected error in batch insert worker: {e}")

    async def process_batch(self, batch: List[Dict[str, Any]]):
        """处理一批数据，插入到数据库"""
        if not batch:
            return

        if not self.get_session_func:
            logger.error("No session function provided for batch processor")
            return

        try:
            async with self.get_session_func() as session:
                session.add_all([self.model_class(**data) for data in batch])
                await session.commit()
                logger.info(f"Successfully inserted {len(batch)} records for {self.model_class.__name__}")
        except Exception as e:
            logger.error(f"Error inserting batch for {self.model_class.__name__}: {e}")
            # await session.rollback()
            # 可以根据需要添加重试逻辑或错误处理

    async def process_one(self, data):
        """直接插入（不使用队列）"""
        async with self.get_session_func() as session:
            try:
                session.add(self.model_class(**data))
                await session.commit()
                return True
            except Exception as e:
                await session.rollback()
                logger.error(f"Error insert for {self.model_class.__name__}: {e}")
        return False

    async def execute_batch(self, stmt, values_list: List[tuple | dict]):
        """
           批量插入或更新历史记录（支持 executemany），使用 MySQL 的 ON DUPLICATE KEY UPDATE。
        """
        total = 0
        try:
            async with self.get_session_func() as session:
                for chunk in chunks_iterable(values_list, self.batch_size):
                    await session.execute(stmt, chunk)
                    total += len(chunk)
                await session.commit()
            logger.info(f"Inserted/Updated {total} records")
            return total

        except Exception as e:
            logger.error(f"Error during history upsert: {e}\n{stmt}")
        return total

    @asynccontextmanager
    async def context(self):
        """上下文管理器，用于安全地初始化和关闭处理器"""
        await self.initialize()
        try:
            yield self
        finally:
            await self.shutdown()


async def aiohttp_request(method: str, url: str, session: aiohttp.ClientSession = None, json=None, data=None,
                          params=None, headers=None, timeout=30, **kwargs):
    headers = headers or {}

    # 拼接 GET query
    if method.upper() == "GET" and params:
        from urllib.parse import urlencode
        query_string = urlencode(params)
        url = f"{url}?{query_string}"

    async def fetch_url(session: aiohttp.ClientSession):
        async with session.request(method, url, json=json, data=data, headers=headers, timeout=timeout,
                                   **kwargs) as resp:
            try:
                resp.raise_for_status()
                return await resp.json()
            except aiohttp.ContentTypeError:
                return {"status": resp.status, "error": "Non-JSON response", "body": await resp.text()}

    try:
        if session:
            return await fetch_url(session)

        async with aiohttp.ClientSession(headers=headers, timeout=aiohttp.ClientTimeout(total=timeout)) as ts:
            return await fetch_url(ts)

    except aiohttp.ClientResponseError as e:
        logging.error(f"[HTTP Error] {method} {url} | Status: {e.status}, Message: {e.message}, Body: {json or data}")
        return {"status": e.status, "error": e.message, "url": url}
    except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as e:
        logging.error(f"[Connection Error] {method} {url} | Message: {e}")
        return {"status": 503, "error": f"Connection failed: {str(e)}", "url": url}
    except Exception as e:
        logging.error(f"[Unknown Error] {method} {url} | Message: {e}")
        return {"status": 500, "error": str(e), "url": url}


async def post_aiohttp_stream(url, payload: dict = None, time_out=60, headers: dict = None, **kwargs):
    timeout = aiohttp.ClientTimeout(total=time_out, sock_read=max(60, time_out), sock_connect=5.0)
    async with aiohttp.ClientSession(timeout=timeout, headers=headers or {}) as session:
        async with session.post(url, json=payload, **kwargs) as response:
            response.raise_for_status()
            buffer = bytearray()  # b""
            async for chunk in response.content.iter_any():  # .iter_chunked(1024)
                if not chunk:
                    continue
                buffer.extend(chunk.tobytes() if isinstance(chunk, memoryview) else chunk)
                while b"\n" in buffer:  # 处理缓冲区中的所有完整行
                    line, buffer = buffer.split(b"\n", 1)
                    if line.strip():
                        try:
                            yield json.loads(line.decode("utf-8").strip())
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            continue

            if buffer:
                tail_bytes = bytes(buffer).rstrip(b"\r\n")
                if tail_bytes and tail_bytes.strip():
                    try:
                        yield json.loads(tail_bytes.decode("utf-8").strip())  # if decoded_line.startswith(('{', '[')):
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        pass


async def post_httpx_sse(url, payload: dict = None, headers: dict = None, time_out=60,
                         client: httpx.AsyncClient = None, **kwargs):
    def parse_line(data):
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return None

    async def _run(cx: httpx.AsyncClient):
        try:
            async with cx.stream('POST', url, json=payload, headers=headers, **kwargs) as response:
                response.raise_for_status()
                buffer = ""
                async for line in response.aiter_lines():  # 使用 aiter_bytes() 处理原始字节流
                    line = line.strip()
                    if not line:  # 空行 -> 事件结束,开头的行 not line
                        if buffer:
                            if buffer == "[DONE]":
                                yield {"type": "done"}
                                return
                            parse = parse_line(buffer)  # 一个数据块的结束
                            if parse:
                                yield {"type": "data", "data": parse}
                            else:
                                yield {"type": "text", "data": buffer}
                            buffer = ""  # 重置清空
                        continue

                    if line.startswith("data: "):
                        content = line[6:]  # line.lstrip("data: ")
                        if content in ("[DONE]", '"[DONE]"', "DONE"):
                            yield {"type": "done"}
                            return
                        parse = parse_line(content)  # 单行 JSON
                        if parse:
                            yield {"type": "data", "data": parse}
                    else:  # 处理非 data: 行或 JSON 解析失败时
                        buffer += ("\n" + line) if buffer else line

                if buffer:  # 处理最后遗留的 buffer
                    parse = parse_line(buffer)
                    if parse:
                        yield {"type": "data", "data": parse}
                    else:
                        yield {"type": "text", "data": buffer}

        except Exception as e:
            # yield error event and return
            yield {"type": "error", "data": f"HTTP error: {e}"}

    if client:
        async for item in _run(client):
            yield item
        return

    timeout = httpx.Timeout(time_out, read=60.0, connect=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async for item in _run(client):
            yield item


async def call_http_request(url: str, headers=None, timeout=100.0, httpx_client: httpx.AsyncClient = None, **kwargs):
    """
    异步调用HTTP请求并返回JSON响应，如果响应不是JSON格式则返回文本内容
    国内，无代理，可能内容无解析
    :param url: 请求地址
    :param headers: 请求头
    :param timeout: 请求超时时间
    :param httpx_client: 外部传入的 AsyncClient 实例（可复用连接）
    :param kwargs: 其他传递给 httpx 的参数
    :return: dict 或 None
    """

    async def fetch_url(cx: httpx.AsyncClient):
        response = await cx.get(url, headers=headers, timeout=timeout, **kwargs)  # 请求级别的超时优先级更高
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type or not content_type:
            try:
                return response.json()
            except json.JSONDecodeError:
                pass

        return {'text': response.text}

    if httpx_client:
        return await fetch_url(httpx_client)

    async with httpx.AsyncClient() as cx:
        return await fetch_url(cx)


async def download_by_aiohttp(url: str, session: aiohttp.ClientSession, save_path, chunk_size=4096, in_decode=False):
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


async def download_by_httpx(url: str, client: httpx.AsyncClient, save_path, chunk_size=4096, in_decode=False):
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


async def upload_by_httpx(url: str, client: httpx.AsyncClient = None, files_path=('example.txt', b'Hello World')):
    '''
    with open("local.txt", "rb") as f:
        await upload_by_httpx("http://127.0.0.1:8000/upload", files_path=("local.txt", f))
    '''
    files = {'file': files_path}  # (filename, bytes/文件对象)

    if client:
        resp = await client.post(url, files=files)
    else:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, files=files)

    resp.raise_for_status()
    return resp.json()


def download_by_requests(url: str, save_path, chunk_size=4096, in_decode=False, timeout=30):
    """
    同步下载的流式方法
    如果目标是保存到文件，直接使用 content（无需解码）。（如图片、音频、视频、PDF 等）
    如果目标是处理和解析文本数据，且确定编码正确，使用 text。（如 HTML、JSON）
    """
    with requests.get(url, stream=True, timeout=timeout) as response:
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

    except Exception as e:
        logger.error(f'send_to_wechat{body}')
        logger.error(f"Error occurred while sending message: {e}")
        # with httpx.Client(timeout=Config.HTTP_TIMEOUT_SEC) as cx:
        #     response = cx.post(url, json=body, headers=headers)
        #     response.raise_for_status()
        # return response.json()

    return None


class OperationHttp:
    def __init__(self, use_sync=False, time_out: int | float = 100.0, proxy: str = None):
        self.client = None  # httpx.AsyncClient | aiohttp.ClientSession | requests.Session
        self._is_httpx: bool = False
        self._is_sync: bool = use_sync
        self._timeout: int = time_out
        self._proxy: str = proxy  # 支持 None / "http://host:port" / "socks5://host:port"
        # self._mode: str = mode.lower()

        try:
            import httpx
            self._is_httpx = True
        except ImportError:
            try:
                import aiohttp
                self._is_httpx = False
            except ImportError:
                import requests
                self._is_sync = True

    async def __aenter__(self):
        self.init_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close_async()

    def __enter__(self):
        self.init_client()
        return self

    def __exit__(self, *args):
        self.close_sync()

    def init_client(self):
        if self.client is not None:
            return

        if self._is_sync:
            if self._is_httpx:
                transport = httpx.HTTPTransport(proxy=self._proxy or None)
                self.client = httpx.Client(timeout=self._timeout, transport=transport)  # 底层为 httpcore
            else:
                self.client = requests.Session()  # 基于 urllib3
                if self._proxy:
                    self.client.proxies.update({"http": self._proxy, "https": self._proxy})
        else:
            if self._is_httpx:
                transport = httpx.AsyncHTTPTransport(proxy=self._proxy or None)
                limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
                timeout = httpx.Timeout(self._timeout, read=self._timeout, write=30.0, connect=5.0)
                self.client = httpx.AsyncClient(limits=limits, timeout=timeout, transport=transport)
            else:
                connector = aiohttp.TCPConnector(limit=100, limit_per_host=20)
                timeout = aiohttp.ClientTimeout(total=self._timeout, sock_read=self._timeout, sock_connect=5.0)
                self.client = aiohttp.ClientSession(connector=connector, timeout=timeout)  # headers=headers or {}

            # self.semaphore = asyncio.Semaphore(30)

    async def close_client(self):
        if self._is_sync:
            self.close_sync()
        else:
            await self.close_async()

    def close_sync(self):
        if self.client:
            self.client.close()
            self.client = None

    async def close_async(self):
        if self.client:
            if self._is_httpx:
                if not self.client.is_closed:
                    await self.client.aclose()
            else:
                if not self.client.closed:
                    await self.client.close()
            self.client = None

    async def get(self, url, headers=None, **kwargs):
        if self._is_sync:
            if self._is_httpx:
                resp = self.client.get(url, headers=headers, **kwargs)  # params=data
            else:
                resp = self.client.get(url, headers=headers, timeout=(self._timeout, self._timeout), **kwargs)
            resp.raise_for_status()
            return resp.json()
        else:
            if self._is_httpx:
                resp = await self.client.get(url, headers=headers, **kwargs)
                resp.raise_for_status()  # 如果请求失败，则抛出异常
                return resp.json()
            else:
                async with self.client.get(url, headers=headers or {}, **kwargs) as resp:
                    resp.raise_for_status()
                    return await resp.json()  # await resp.text()

    async def post(self, url, json=None, headers=None, **kwargs):
        if self._is_sync:
            if self._is_httpx:
                resp = self.client.post(url, json=json, headers=headers, **kwargs)
            else:
                resp = self.client.post(url, json=json, headers=headers, timeout=(self._timeout, self._timeout),
                                        **kwargs)
            resp.raise_for_status()
            return resp.json()
        else:
            if self._is_httpx:
                resp = await self.client.post(url, json=json, headers=headers, **kwargs)
                resp.raise_for_status()  # 如果请求失败，则抛出异常
                return resp.json()
            else:
                async with self.client.post(url, json=json, headers=headers or {}, **kwargs) as resp:
                    resp.raise_for_status()  # 抛出 4xx/5xx 错误
                    return await resp.json()

    def fallback_post(self, url, json_payload, headers=None, stream=False):
        try:
            resp = requests.post(url, headers=headers, json=json_payload, timeout=(5, self._timeout), stream=stream,
                                 proxies={"http": self._proxy, "https": self._proxy} if self._proxy else None)
            if resp.status_code == 200:
                return resp.json()  # json.loads(resp.content)
            else:
                raise RuntimeError(f"[requests fallback] 返回异常: {resp.status_code}, 内容: {resp.text}")
        except Exception as e:
            print(f"[requests fallback] 请求失败: {e}")
            return None


@async_error_logger(max_retries=1, delay=3, exceptions=(Exception, httpx.HTTPError))
async def follow_http_html(url, time_out: float = 100.0, **kwargs):
    async with httpx.AsyncClient(timeout=time_out, follow_redirects=True) as cx:
        response = await cx.get(url, **kwargs)
        response.raise_for_status()
        return response.text


async def proxy_http_html(base_url: str, full_path: str, request: Request):
    url = f"{base_url}/{full_path}"
    async with httpx.AsyncClient(follow_redirects=True) as cx:
        method = request.method
        headers = dict(request.headers)
        body = await request.body()

        proxy_req = cx.build_request(method, url, headers=headers, content=body)
        resp = await cx.send(proxy_req)

        return Response(content=resp.content, status_code=resp.status_code, headers=dict(resp.headers))


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
        url = model.get('model_url') or model['base_url'] + '/models'
        headers = {}
        api_key = model.get('api_key')
        if api_key:
            headers["Authorization"] = f'Bearer {api_key}'
            if model['type'] == 'anthropic':
                headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}

        cx = get_httpx_client(proxy=Config.HTTP_Proxy if model.get('proxy') else None)
        models = await call_http_request(url, headers, httpx_client=cx)
        if models:
            return models.get('data')

    return None


class ModelList:
    models = []
    _redis = None
    _worker_id: str = None
    _list_key = "model_list"
    _data_key = "model_data_list"
    _hash_key = "model_data_hash"

    @classmethod
    def extract(cls, ai_models: list):
        """
        提取 AI_Models 中的 name 以及 search_field 中的所有值，并存入一个大列表。

        返回：
        - List[str]: 包含所有模型名称及其子模型的列表
        """
        # list(itertools.chain(*[sublist[1] for sublist in extract_ai_model("model")]))
        extracted_data = extract_ai_model("model", ai_models)
        return [i for item in extracted_data for i in [item[0]] + item[1]]  # flattened_list

    @classmethod
    async def set(cls, redis=None, worker_id: str = None, ai_models: list = AI_Models):
        """更新 MODEL_LIST,并保存到 Redis"""
        cls.models = cls.extract(ai_models)
        if cls._redis is None:
            cls._redis = redis or get_redis()
        if worker_id:
            cls._worker_id = worker_id

        await cls.to_redis(cls._list_key, cls.models)

        await cls.set_datas(ai_models)

    @classmethod
    async def to_redis(cls, key: str, value, **kwargs):
        """
        分布式写入：谁先抢到锁谁写入，失败自动释放锁，成功不释放
        kwargs: 传递给 redis.set 的其他参数，如 ex/px/nx/xx 等
        """
        if not cls._redis:
            return False

        lock_key = f"lock:{key}"
        async with with_distributed_lock(lock_key, cls._worker_id, 60000, redis=cls._redis, release=False) as acquired:
            if not acquired:  # 锁已被占用，直接返回失败
                return False

            try:
                await cls._redis.set(key, json.dumps(value, ensure_ascii=False), **kwargs)
                return True
            except Exception as e:
                logging.error(f"[Redis SET Error] key={key}, error={e}")
                if cls._worker_id:  # 有暗号不释放，写入失败才释放锁
                    current_lock_value = await cls._redis.get(lock_key)
                    if current_lock_value and current_lock_value == cls._worker_id:
                        await cls._redis.delete(lock_key)

        # await run_with_lock(cls._redis.set, cls._list_key, json.dumps(cls.models, ensure_ascii=False),
        #                     lock_key=f"lock:{cls._list_key}", lock_timeout=60)
        return False

    @classmethod
    async def get(cls, updated=True):
        if cls._redis:
            data = await cls._redis.get(cls._list_key)
            if data:
                return json.loads(data)
        if not cls.models and updated:
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

    @classmethod
    async def set_model_data(cls, model: dict, hash_data: dict = None) -> tuple:
        name = model.get('name')
        key = f"{cls._data_key}:{name}"
        hash_key = generate_hash_key(model.get('model', []))
        old_hash_key = (hash_data or {}).get(name)

        if old_hash_key == hash_key:  # 如果旧的 hash 相同，则无需更新
            data_raw = await cls._redis.get(key) if cls._redis else None
            if data_raw:
                model['data'] = json.loads(data_raw)
                return name, hash_key

        data = await get_data_for_model(model)  # 否则重新拉取数据并缓存
        if data:
            model['data'] = data
            await cls.to_redis(key, data)
            print('model:', model.get('name'), 'data:', data)
        return name, hash_key

    @classmethod
    async def set_datas(cls, ai_models=AI_Models):
        hash_data_raw = await cls._redis.get(cls._hash_key) if cls._redis else None
        hash_data = json.loads(hash_data_raw) if hash_data_raw else {}

        tasks = [cls.set_model_data(model, hash_data) for model in ai_models
                 if model.get('supported_list') and model.get('api_key')]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:  # 过滤出成功的结果，并更新 hash_data
            if isinstance(r, Exception):
                print(f"[set_model_datas] error {r}")
                continue

            name, hash_key = r
            if name and hash_key:
                hash_data[name] = hash_key

        await cls.to_redis(cls._hash_key, hash_data)

    @staticmethod
    def get_model_data(model: str):
        try:
            model_info, model_id = find_ai_model(model, 0, 'model')
            model_data = next((item for item in model_info.get('data', []) if item['id'] == model_id), {})
            data = {
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": model_info['name'],
                "type": "chat",
                "permission": [
                    {
                        "id": f"modelperm-{model_info['name']}:{model_id}",
                    }
                ],
                "supported_parameters": ["max_tokens", "stop", "temperature", "tool_choice", "tools",
                                         "top_k", "top_p"],
            }
            for k, v in model_data.items():
                if k not in {"id", "owned_by"}:
                    data[k] = v
        except ValueError as e:
            data = {'error': str(e)}
        return data

    @staticmethod
    def get_models(ai_models: list = AI_Models):
        extracted_data = extract_ai_model("model", ai_models)
        models_data = []
        for owner, models in extracted_data:
            owner_data = next((item.get('data', []) for item in ai_models if item['name'] == owner), [])
            for i, model_id in enumerate(models):
                data = {
                    "id": f"{owner}:{model_id}",  # 唯一模型ID,  用于指定模型进行请求 fine-tuned-model
                    "object": "model",
                    "type": "chat",
                    "created": 1740386673,
                    "owned_by": owner,  # 拥有该模型的组织
                    "root": model_id,  # 根版本，与 ID 相同
                    "parent": None,  # 如果没有父模型，则为 None
                    # "max_model_len": 4096,#GPU内存限制而需要调整模型的最大序列长度
                    "permission": [
                        {
                            "id": f"modelperm-{owner}:{model_id}",
                            "object": "model_permission",
                        }
                    ],
                    "supported_parameters": ["max_tokens", "stop", "temperature", "tool_choice", "tools",
                                             "top_k", "top_p"],
                }  # 基础结构
                model_data = next((item for item in owner_data if item['id'] == model_id), {})
                for k, v in model_data.items():  # 覆盖模型信息
                    if k not in {"id", "owned_by"}:
                        data[k] = v
                models_data.append(data)

        return {"object": "list", "data": models_data}


async def init_ai_clients(ai_models: list = AI_Models):
    limits = httpx.Limits(max_connections=max(Config.MAX_HTTP_CONNECTIONS, Config.MAX_KEEPALIVE_CONNECTIONS),
                          max_keepalive_connections=Config.MAX_KEEPALIVE_CONNECTIONS)
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
                time_out = model.get('timeout', Config.LLM_TIMEOUT_SEC)
                if model.get('proxy'):  # proxies=proxies
                    timeout = httpx.Timeout(time_out, read=time_out, write=100.0, connect=10.0)
                    http_client = httpx.AsyncClient(transport=transport, limits=limits, timeout=timeout)

                AI_Client[model_name]: AsyncOpenAI = AsyncOpenAI(api_key=api_key, base_url=model['base_url'],
                                                                 http_client=http_client,
                                                                 max_retries=Config.MAX_RETRY_COUNT)
                if http_client is None:
                    AI_Client[model_name] = AI_Client[model_name].with_options(timeout=time_out,
                                                                               max_retries=Config.MAX_RETRY_COUNT)


# client = AI_Client['deepseek']
# print(dir(client.chat.completions))# 'create', 'with_raw_response', 'with_streaming_response'
# print(dir(client.completions))
# print(dir(client.embeddings))
# print(dir(client.files)) #'content', 'create', 'delete', 'list', 'retrieve', 'retrieve_content', 'wait_for_processing'


def find_ai_model(name: str, model_id: int = 0, search_field: str = 'model') -> Tuple[dict, str]:
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
        if model:
            model_items = model.get(search_field, [])
            if model_name in model_items:
                return model, model_items[model_name] if isinstance(model_items, dict) else model_name

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


def extract_ai_model(search_field: str = "model", ai_models: list = AI_Models):
    """
    提取 AI_Models 中的 name 以及 search_field 中的所有值（列表或字典 key）。

    返回：
    - List[Tuple[str, List[str]]]: 每个模型的名称及其对应的模型列表
    """
    extracted_data = []

    for model in ai_models:
        name = model["name"]
        field_value = model.get(search_field, [])
        if model.get('supported_openai', True) and not model.get('api_key'):
            continue

        if isinstance(field_value, list):
            extracted_data.append((name, list(dict.fromkeys(field_value))))
        elif isinstance(field_value, dict):
            extracted_data.append((name, list(field_value.keys())))
        else:
            extracted_data.append((name, [field_value]))

    return extracted_data


async def run_mcp_task(transport="streamable-http", port=7007, **kwargs):
    # subprocess.Popen(["python", "-m", "mcp", "--port", "7007"])
    shutdown_event = asyncio.Event()

    async def _run():
        await mcp.run_async(transport=transport, port=port, host="127.0.0.1", path="/mcp", **kwargs)
        await shutdown_event.wait()  # 等待关闭信号

    task = asyncio.create_task(_run())
    return task, shutdown_event


def create_openai_mcp(base_url="http://127.0.0.1:7000", timeout=Config.HTTP_TIMEOUT_SEC,
                      instructions: str | None = None) -> FastMCP:
    from fastmcp.server.openapi import RouteMap, MCPType
    api_client = httpx.AsyncClient(base_url=base_url, headers={"Authorization": "Bearer YOUR_TOKEN"})
    # import requests
    # resp = requests.get(f"{base_url}/openapi.json")
    # print("TEXT:", resp.text)
    # openapi_spec = resp.json()
    from utils import load_dictjson

    # openapi_spec = httpx.get(f"{base_url}/openapi.json").json()  # Load your OpenAPI spec
    openapi_spec = load_dictjson('openapi.json', encoding='utf-8')
    print(openapi_spec)
    DEFAULT_ROUTE_MAPPINGS = [
        # custom mapping logic goes here
        # ... your specific route maps ...
        RouteMap(methods=["GET"], pattern=r".*\{.*\}.*", mcp_type=MCPType.RESOURCE_TEMPLATE),
        RouteMap(methods=["GET"], pattern=r".*", mcp_type=MCPType.RESOURCE),
        RouteMap(pattern=r"^/admin/.*", mcp_type=MCPType.EXCLUDE),
        # exclude all remaining routes
        RouteMap(mcp_type=MCPType.EXCLUDE),
    ]
    api_mcp = FastMCP.from_openapi(
        openapi_spec=openapi_spec,
        client=api_client,
        timeout=timeout,  # 30 second timeout for all requests
        route_maps=DEFAULT_ROUTE_MAPPINGS,
        instructions=instructions,
    )
    return api_mcp


# mcp.mount(openai_mcp("http://47.110.156.41:7000"), prefix="openapi")

async def call_mcp_tool(config: dict[str, Any] | str, name: str, **kwargs):
    async with MCPClient(config) as client:
        # Access tools and resources with server prefixes
        # answer = await client.call_tool("assistant_answer_question", {"query": "What is MCP?"})
        return await client.call_tool(name, **kwargs)


@mcp.custom_route("/", methods=["GET"])
async def health_check(request: Request) -> Response:
    return JSONResponse({"status": "ok"})


@mcp.resource("config://version")
def get_version():
    return Config.Version


@mcp.tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


if __name__ == "__main__":
    import nest_asyncio

    # nest_asyncio.apply()
    import threading


    # mcp.run(transport="sse", log_level="debug")
    # from fastmcp.server.proxy import FastMCPProxy
    # mcp.mount(FastMCPProxy("http://other-host:8001/mcp"), prefix="remote")

    # mcp.run(transport="streamable-http", host="127.0.0.1", port=8000, path="/mcp")
    # _redis_client = Redis(host='47.110.156.41', port=7007, db=0, decode_responses=True)
    # import uvicorn
    # uvicorn.run(mcp, host="0.0.0.0", port=7007)

    async def main():
        # 进程间通信，适用于命令行或脚本工具执行
        # await asyncio.to_thread(mcp.run, transport="stdio", **kwargs)
        async with MCPClient("http://127.0.0.1:7000/mcp/mcp") as client:
            # ... use the client
            tools = await client.list_tools()
            print(f"Available tools: {tools}")
            result = await client.call_tool("add", {"a": 5, "b": 3})
            print(f"Result: {result}")
            resources = await client.list_resources()
            # Read a resource from the server
            data = await client.read_resource(resources[0].uri)
            print(f"Result: {data[0].text}")

        # map_task = await run_mcp_task("stdio")
        mcp_task, exit_event = await run_mcp_task(port=7007)
        # kk = await _redis_client.get('dd')
        # print(kk)
        # async with MCPClient("utils.py") as client:
        #     tools = await client.list_tools()
        #     print(f"Available tools: {tools}")
        #     result = await client.call_tool("add", {"a": 5, "b": 3})
        #     print(f"Result: {result.text}")

        # Connect via in-memory transport
        async with MCPClient(mcp) as client:
            tools = await client.list_tools()
            print(f"Available tools: {tools}")
            result = await client.call_tool("add", {"a": 5, "b": 3})
            print(f"Result: {result}")
            resources = await client.list_resources()
            results = await client.read_resource(resources[0].uri)
            print(f"Result: {results[0].text}")

        # Connect via SSE
        # async with MCPClient("http://localhost:8000/sse") as client:
        #     # ... use the client
        #     tools = await client.list_tools()
        #     print(f"Available tools: {tools}")

        # 发出退出信号并取消任务
        exit_event.set()
        # mcp_task.cancel()
        # await mcp_task await asyncio.shield(mcp_task)
        try:
            await mcp_task
        except asyncio.CancelledError:
            print("后台任务已取消")


    Config.debug()


    async def test_r():
        redis = get_redis()
        result = await get_redis_value(redis, 'model_data_list:zzz')  # tokenflux,aihubmix
        print([item.get('id') for item in result])
        await shutdown_redis()


    # asyncio.run(main())

    asyncio.run(test_r())
