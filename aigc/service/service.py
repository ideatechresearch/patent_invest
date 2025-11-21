import httpx
import json, os, io, time, uuid
from datetime import datetime
from typing import Optional, Type, Dict, List, Tuple, Any, Union, Literal
from contextlib import asynccontextmanager, contextmanager
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from redis.asyncio import Redis, StrictRedis, ConnectionPool
from neo4j import GraphDatabase, AsyncGraphDatabase
from dask.distributed import Client as DaskClient, LocalCluster
from qdrant_client import AsyncQdrantClient, QdrantClient
from openai import AsyncOpenAI, OpenAI
from fastmcp import FastMCP, Context as MCPContext, Client as MCPClient, settings
import oss2

# https://gofastmcp.com/servers/context

from .base import *
from .mysql_ops import OperationMysql
from .task_ops import HierarchicalTimeWheel
from utils import async_to_sync, generate_hash_key, is_port_open, chunks_iterable, get_file_type_wx
from config import Config, AI_Models, model_api_keys

# Config.load('config.yaml')
# if os.getenv('AIGC_DEBUG', '0').lower() in ('1', 'true', 'yes'):
#     Config.debug()

_httpx_clients: Dict[str, httpx.AsyncClient] = {}
_graph_driver: Optional[GraphDatabase] = None
# _graph_driver_lock = asyncio.Lock()  # 防止并发初始化
_redis_clients: Dict[int, Optional[Redis]] = {}  # StrictRedis(host='localhost', port=6379, db=0)
_redis_pools: Dict[int, Optional[ConnectionPool]] = {}
_dask_cluster: Optional[LocalCluster | str] = None
_dask_client: Optional[DaskClient] = None
AI_Client: Dict[str, Optional[AsyncOpenAI]] = {}  # OpenAI
QD_Client = AsyncQdrantClient(host=Config.QDRANT_HOST, grpc_port=Config.QDRANT_GRPC_PORT,
                              prefer_grpc=True) if Config.QDRANT_GRPC_PORT else AsyncQdrantClient(url=Config.QDRANT_URL)
DB_Client = OperationMysql(async_mode=True, minsize=2)

AliyunBucket = oss2.Bucket(oss2.Auth(Config.ALIYUN_oss_AK_ID, Config.ALIYUN_oss_Secret_Key), Config.ALIYUN_oss_endpoint,
                           Config.ALIYUN_Bucket_Name)
logger = get_root_logging(file_name="app.log")  # logging.getLogger(__name__)
mcp = FastMCP(name="FastMCP Server")  # Create a server instance,main_mcp


# mcp_app = mcp.http_app(transport="streamable-http", path="/mcp")

# dependencies=["pandas", "matplotlib", "requests"]

def get_scheduler(redis_host: str = Config.REDIS_HOST, redis_port: int = Config.REDIS_PORT,
                  timezone: str = "Asia/Shanghai"):
    """
    创建并返回一个全局 AsyncIOScheduler 调度器实例。
    支持 Redis JobStore + 内存 JobStore，默认异步执行器。
    """
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.executors.asyncio import AsyncIOExecutor
    from apscheduler.jobstores.memory import MemoryJobStore
    from apscheduler.jobstores.redis import RedisJobStore
    # from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
    # from apscheduler.schedulers.background import BackgroundScheduler
    # from apscheduler.executors.pool import ThreadPoolExecutor

    # executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
    # scheduler = BackgroundScheduler(jobstores={'default': SQLAlchemyJobStore(engine=engine), 'memory': MemoryJobStore()},
    #                                 executors={'default': ThreadPoolExecutor(4)}, timezone='Asia/Shanghai')  # 设置线程池大小
    scheduler = AsyncIOScheduler(
        executors={"default": AsyncIOExecutor()},
        jobstores={
            "memory": MemoryJobStore(),
            "redis": RedisJobStore(
                jobs_key="apscheduler.jobs",
                run_times_key="apscheduler.run_times",
                host=redis_host, port=redis_port, db=0,
            ),
        },
        timezone=timezone,
    )  # 异步调度器

    if not scheduler.running:
        scheduler.start()
        print(f"[scheduler] Started with timezone={timezone}")
    # scheduler.shutdown()
    return scheduler


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
    global _redis_clients, _redis_pools
    if db not in _redis_clients or _redis_clients[db] is None:
        pool = ConnectionPool(host=Config.REDIS_HOST, port=Config.REDIS_PORT, db=db,
                              decode_responses=True,  # 自动解码为字符串
                              max_connections=Config.REDIS_MAX_CONCURRENT
                              )
        _redis_clients[db] = Redis(connection_pool=pool)
        _redis_pools[db] = pool

    return _redis_clients[db]


async def shutdown_redis():
    global _redis_clients, _redis_pools
    for key, _client in _redis_clients.items():
        if _client:
            await _client.aclose()
        _redis_clients[key] = None
    for key, _pool in _redis_pools.items():
        if _pool:
            await _pool.disconnect()
            _redis_pools[key] = None


async def check_redis_connection(redis: Redis):
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


async def get_redis_retry(redis: Redis, key: str, retry: int = 3, delay: float = 0.1):
    for attempt in range(retry):
        try:
            return await redis.get(key)
        except Exception as e:
            print(f"[Redis GET] attempt {attempt + 1} failed: {e}")
            await asyncio.sleep(delay)
    raise Exception(f"Redis GET failed after {retry} retries.")


async def get_redis_value(redis: Redis, key: str) -> Union[dict, set, list, str, int, float, None]:
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

        else:  # 单个键查询
            t = await redis.type(key)
            if t == "none":
                return None
            if t == "string":
                value = await redis.get(key)
                if value is None:
                    return None
                if isinstance(value, bytes):
                    value = value.decode('utf-8')
                return parse_json(value)
            if t == "hash":
                value = await redis.hgetall(key)
                return {k: parse_json(v) for k, v in value.items()}
            if t == "set":
                return await redis.smembers(key)
            if t == "list":
                items = await redis.lrange(key, 0, -1)
                return [parse_json(i) for i in items]
            print(key, t)
            # 其他类型，如 zset, stream
            return None

    except (ConnectionError, TimeoutError) as e:
        # Connect call failed,redis.exceptions.ConnectionError
        raise Exception(f"Redis 连接失败,detail:{e}")
    except Exception as e:
        raise Exception(f"Redis 查询错误,detail:{e}")


async def scan_from_redis(redis: Redis, key: str = "funcmeta", batch_count: int = 0) -> list[dict]:
    """
    从 Redis 中获取匹配的所有元数据记录，支持 scan 或 keys 方式。

    Args:
        redis: Redis 实例。
        key: Redis key 前缀（如 "funcmeta"）。
        batch_count: 每批 scan 的数量（大于 0 使用 scan，否则用 keys）。

    Returns:
        List[dict]: 匹配到的 JSON 数据列表。
    """
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


async def stream_to_redis(redis: Redis, batch: list, key: str = 'streams'):
    pipe = redis.pipeline()
    for stream_id, chunk in batch:
        # if not isinstance(chunk, dict):
        #     chunk = {"data": json.dumps(chunk, ensure_ascii=False)}
        stream_name = f"{key}:{stream_id % 3}"  # 选择分片流名
        await pipe.xadd(stream_name, fields=chunk, id="*", maxlen=10000, approximate=True)
    try:
        results = await pipe.execute()
        return len(results)
    except Exception as e:
        print(f"[Redis Stream Error] key={key}, batch_size={len(batch)}, error={e}")
        raise


async def sadd_to_redis(redis: Redis, key: str, values: list | set | tuple | str, ex: int = 3600) -> int:
    if not values:
        return 0
    if isinstance(values, str):
        values = [values]
    pipe = redis.pipeline()
    await pipe.sadd(key, *values)
    if ex > 0:
        await pipe.expire(key, ex)  # 每次添加都重置TTL
    results = await pipe.execute()
    return results[0] if results else 0  # r.smembers,scard/sismember


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


def upload_file_to_oss(bucket, file_obj, object_name=None, expires: int = 604800, total_size: int = 0):
    """
      上传文件到 OSS 支持 `io` 对象。
      :param bucket: OSS bucket 实例
      :param file_obj: 文件对象，可以是 `io.BytesIO` 或 `io.BufferedReader`
      :param object_name: OSS 中的对象名
      :param expires: 签名有效期，默认一周（秒）
      :param total_size
    """
    if isinstance(file_obj, bytes):
        file_obj = io.BytesIO(file_obj)
    if not total_size:
        file_obj.seek(0, os.SEEK_END)
        total_size = file_obj.tell()  # os.path.getsize(file_path)
        file_obj.seek(0)
    if not object_name:
        if not hasattr(file_obj, "name"):
            file_obj.name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.bin"
        object_name = f"upload/{file_obj.name}"

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
    return url, object_name


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


class AsyncBatchAdd:
    """
    通用的异步批量处理器，可用于任何 SQLAlchemy ORM 类
    """

    def __init__(
            self,
            model_class: Type,
            batch_size: int = 100,
            batch_timeout: float = 3.0,
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
        self._last_insert_time = None

    @property
    def insert_time(self):
        return self._last_insert_time

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

    async def enqueue(self, data: Dict[str, Any]):
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
        self._last_insert_time = time.time()
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
                        (batch and current_time - self._last_insert_time >= self.batch_timeout)):
                    await self.process_batch(batch)
                    batch.clear()
                    self._last_insert_time = current_time

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
        response = await cx.get(url, headers=headers, timeout=model.get('timeout', Config.LLM_TIMEOUT_SEC))
        response.raise_for_status()
        models = response.json()
        if models:
            return models.get('data')

    return None


class ModelList:
    ai_models: list[dict] = []
    models = []
    owners = []
    _redis = None
    _worker_id: str = None
    _list_key = "model_list"
    _data_key = "model_data_list"
    _hash_key = "model_data_hash"

    @classmethod
    def extract(cls) -> tuple[list, list]:
        """
        提取 AI_Models 中的 name 以及 search_field 中的所有值，并存入一个大列表。

        返回：
        - List[str]: 包含所有模型名称及其子模型的列表
        """
        extracted_data = extract_ai_model("model", cls.ai_models)
        owners = [item[0] for item in extracted_data]
        flattened_list = [i for item in extracted_data for i in [item[0]] + item[1]]
        # duplicates = {item: count for item, count in Counter(flattened_list).items() if count > 1}
        # print("模型数量:", len(flattened_list), "重复模型:", duplicates)
        # list(itertools.chain(*[sublist[1] for sublist in extracted_data])) #去重并保持顺序
        return list(dict.fromkeys(flattened_list)), owners

    @classmethod
    async def set(cls, redis=None, worker_id: str = None, ai_models: list = AI_Models):
        """更新 MODEL_LIST,并保存到 Redis"""
        cls.ai_models = ai_models or AI_Models
        cls.models, cls.owners = cls.extract()
        if cls._redis is None:
            cls._redis = redis or get_redis()
        if worker_id:
            cls._worker_id = worker_id

        await cls.to_redis(cls._list_key, cls.models)

        await cls.set_datas()

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
                logger.error(f"[Redis SET Error] key={key}, error={e}")
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
                if not model.get('model'):
                    model['model'] = [d['id'] for d in model['data']]
                return name, hash_key

        data = await get_data_for_model(model)  # 否则重新拉取数据并缓存
        if data:
            model['data'] = data
            if not model.get('model'):
                model['model'] = [d['id'] for d in data]
            await cls.to_redis(key, data)
            print('model:', model.get('name'), 'data:', data)
        return name, hash_key

    @classmethod
    async def set_datas(cls):
        # key_type = await cls._redis.type(cls._hash_key)
        # if key_type != "hash":
        #     await cls._redis.delete(cls._hash_key)
        hash_data = await cls._redis.hgetall(cls._hash_key) if cls._redis else {}
        tasks = [cls.set_model_data(model, hash_data) for model in cls.ai_models
                 if model.get('supported_list') and model.get('api_key')]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        pipe = cls._redis.pipeline()
        for r in results:  # 过滤出成功的结果，并更新 hash_data
            if isinstance(r, Exception):
                print(f"[set_model_datas] error {r}")
                continue
            name, hash_key = r
            if not (name and hash_key):
                continue
            if hash_data.get(name) != hash_key:
                pipe.hset(cls._hash_key, name, hash_key)
        await pipe.execute()

    @classmethod
    def save(cls, file_path='models.json'):
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(cls.ai_models, file, ensure_ascii=False)
        print(json.dumps(cls.ai_models, indent=4))

    @staticmethod
    def extract_models(model_type: str = "model", ai_models: list = AI_Models) -> list:
        extracted_data = extract_ai_model(model_type, ai_models)
        models = [f"{owner}:{val}" if val else owner for owner, values in extracted_data for val in values]
        return list(dict.fromkeys(models))

    @staticmethod
    def get_model_data(model: str, ai_models: list = AI_Models):
        try:
            model_info, model_id = find_ai_model(model, 0, 'model', ai_models)
            model_data = next((item for item in model_info.get('data', []) if item['id'] == model_id), {})
            data = {
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": model_info['name'],
                "type": "chat",
                "context_length": 32768,
                "permission": [
                    {
                        "id": f"modelperm-{model_info['name']}:{model_id}",
                        "object": "model_permission",
                    }
                ],
                "supported_parameters": ["max_tokens", "stop", "temperature", "tool_choice", "tools",
                                         "top_k", "top_p"],
            }
            for k, v in model_data.items():
                if not v:  # null
                    continue
                if k == 'id':
                    continue
                if k == "owned_by":
                    if v == "system":
                        continue
                    data[k] += f'-{v}'
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
                    "context_length": 32768,  # max_context_length: 65536,131072,163840,256000
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
                    if k not in {"id", "owned_by"} and v:
                        data[k] = v
                models_data.append(data)

        return {"object": "list", "data": models_data}


async def init_ai_clients(ai_models: list = AI_Models) -> dict:
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
    return AI_Client


def find_ai_model(name: str, model_id: int = 0, search_field: str = 'model',
                  ai_models: list = AI_Models) -> Tuple[dict, str]:
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
        model = next((item for item in ai_models if item['name'] == owner), None)
        if model:
            if model_name in model.get(search_field, []):
                return model, model_name
            if model_name in model.get('model_map', {}):
                return model, model['model_map'][model_name]

    model = next(
        (item for item in ai_models if item['name'] == name or name in item.get(search_field, [])),
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
        elif name in model.get('model_map', {}):
            return model, model['model_map'][name]

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

        model_map = model.get('model_map', {})
        if model_map:
            extracted_data.append((name, list(model_map.keys())))

    return extracted_data


async def run_mcp_task(transport: Literal["stdio", "streamable-http", "sse"] = "streamable-http", port=7007, **kwargs):
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
    openapi_spec = load_dictjson('../openapi.json', encoding='utf-8')
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


if __name__ == "__main__":
    import nest_asyncio

    # nest_asyncio.apply()
    import threading


    # client = AI_Client['deepseek']
    # print(dir(client.chat.completions))# 'create', 'with_raw_response', 'with_streaming_response'
    # print(dir(client.completions))
    # print(dir(client.embeddings))
    # print(dir(client.files)) #'content', 'create', 'delete', 'list', 'retrieve', 'retrieve_content', 'wait_for_processing'

    # mcp.run(transport="sse", log_level="debug")
    # from fastmcp.server.proxy import FastMCPProxy
    # mcp.mount(FastMCPProxy("http://other-host:8001/mcp"), prefix="remote")

    # mcp.run(transport="streamable-http", host="127.0.0.1", port=8000, path="/mcp")
    # redis_client = Redis(host='47.110.156.41', port=7007, db=0, decode_responses=True)
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
        # kk = await redis_client.get('dd')
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


    Config.load('../config.yaml')
    aliyun_bucket = oss2.Bucket(oss2.Auth(Config.ALIYUN_oss_AK_ID, Config.ALIYUN_oss_Secret_Key),
                                Config.ALIYUN_oss_endpoint,
                                Config.ALIYUN_Bucket_Name)
    files = oss_list_files(aliyun_bucket, prefix='upload/', max_keys=100, max_pages=1)
    print(files)
    Config.debug()


    async def test_r():
        redis = get_redis()
        result = await get_redis_value(redis, 'model_data_list:zzz')  # tokenflux,aihubmix
        print([item.get('id') for item in result])
        await shutdown_redis()

    # asyncio.run(main())

    # asyncio.run(test_r())
