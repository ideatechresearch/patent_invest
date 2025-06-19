import igraph as ig
import json, time, random
import asyncio
from redis.asyncio import Redis, StrictRedis, ConnectionPool
from typing import Dict, List, Tuple, Union, Iterable, Callable, Optional, Any
from enum import IntEnum, Enum
from collections import defaultdict
from dataclasses import asdict, dataclass, is_dataclass, field
from threading import Thread
import uuid
import zmq, zmq.asyncio
from neo4j import GraphDatabase, AsyncGraphDatabase

from config import Config
from structs import dataclass2dict

# Config.debug()

_graph_driver: Optional[GraphDatabase] = None
# _graph_driver_lock = asyncio.Lock()  # 防止并发初始化
_redis_client: Optional[Redis] = None  # StrictRedis(host='localhost', port=6379, db=0)
_redis_pool: Optional[ConnectionPool] = None


def get_redis() -> Optional[Redis]:
    global _redis_client, _redis_pool
    if _redis_client is None:
        _redis_pool = ConnectionPool(host=Config.REDIS_HOST, port=Config.REDIS_PORT, db=0,
                                     decode_responses=True,  # 自动解码为字符串
                                     max_connections=50
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


def get_neo_driver():
    global _graph_driver
    if _graph_driver is None:
        _graph_driver = AsyncGraphDatabase.driver(uri=Config.NEO_URI, auth=(Config.NEO_Username, Config.NEO_Password),
                                                  max_connection_lifetime=3600,  # 单连接生命周期
                                                  max_connection_pool_size=50,  # 最大连接池数量
                                                  connection_timeout=30  # 超时
                                                  )
    return _graph_driver


async def get_neo_nodes(cypher):
    driver = get_neo_driver()
    async with driver.session() as session:
        result = await session.run(cypher)  # "CREATE (n:Person {name: 'Alice'})"
        return [record async for record in result]


async def merge_neo_node(tx, node_id: str, props: dict, label: str = 'Node', id_field: str = "id"):
    """
    节点 MERGE 和属性更新函数
    Args:
        tx: Neo4j 事务
        node_id: 主键值
        props: 其他属性字典
        label: 节点标签（GraphQL里的 Node 类型）
        id_field: 属性字段用于识别节点，默认为 id
    """

    def serialize(v):
        if isinstance(v, (dict, list, tuple)):
            return json.dumps(v)
        return v

    safe_props = {k: serialize(v) for k, v in props.items()}
    props_str = ", ".join(f"{k}: ${k}" for k in safe_props)
    query = (
        f"MERGE (n:{label} {{{id_field}: $node_id}}) "
        f"SET n += {{{props_str}}}"
    )
    await tx.run(query, node_id=node_id, **safe_props)


async def merge_neo_relationship(tx, src_id, tgt_id, props: dict = None, rel_type: str = 'RELATED',
                                 label_src: str = 'Node', label_tgt: str = 'Node', id_field: str = "id"):
    '''
    rel_type:关系类型名 "RELATED"任意泛节点,"DEPENDS_ON"依赖
    MATCH (a:Task {{id: $source_id}}), (b:Task {{id: $target_id}})
    MERGE (a)-[r:DEPENDS_ON]->(b)
    SET r += {{{props_str}}}
    '''
    if props:
        def serialize(v):
            if isinstance(v, (dict, list, tuple)):
                return json.dumps(v)
            return v

        safe_props = {k: serialize(v) for k, v in props.items()}
        props_str = ", ".join(f"{k}: ${k}" for k in safe_props)
        query = (
            f"MATCH (a:{label_src} {{{id_field}: $src_id}}), (b:{label_tgt} {{{id_field}: $tgt_id}})"
            f"MERGE (a)-[r:{rel_type}]->(b) "
            f"SET r += {{{props_str}}}"
        )
        await tx.run(query, src_id=src_id, tgt_id=tgt_id, **safe_props)
    else:
        # 无属性关系
        query = (
            f"MATCH (a:{label_src} {{{id_field}: $src_id}}),(b:{label_tgt} {{{id_field}: $tgt_id}}) "
            f"MERGE (a)-[r:{rel_type}]->(b)"
        )
        await tx.run(query, src_id=src_id, tgt_id=tgt_id)


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


async def do_job_by_lock(func_call: Callable, redis_key: str = None, lock_timeout: int = 600, **kwargs):
    redis = get_redis()
    if not redis:
        await func_call(**kwargs)
        return

    if not redis_key:
        redis_key = f'lock:{func_call.__name__}'
    lock_value = str(uuid.uuid4())  # str(time.time())，每个worker使用唯一的lock_value
    lock_acquired = await redis.set(redis_key, lock_value, nx=True, ex=lock_timeout)
    if not lock_acquired:
        print(f"⚠️ 分布式锁已被占用，跳过任务: {func_call.__name__}")
        return

    try:
        print(f"🔒 获取锁成功，开始执行任务: {func_call.__name__}")
        await func_call(**kwargs)
    except Exception as e:
        print(f"⚠️ 任务执行出错: {func_call.__name__} -> {e}")
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


async def consumer_worker(queue: asyncio.Queue[tuple[Any, int]], process_task: Callable, max_retries: int = 0,
                          delay: float = 1.0, **kwargs):
    """
    启动 worker:
        workers_task_background = [asyncio.create_task(consumer_worker(queue, process_task)) for _ in range(4)],
        #await asyncio.gather(*tasks)

    asyncio.Queue 是内存对象，只存在于当前进程的事件循环中
    :param queue:asyncio.Queue()
    :param process_task:注入处理函数 async def f(task: tuple, **kwargs),返回值 true,false,失败可以选择 await queue.put(...retry+1):会等待队列有空间再放入 /.put_nowait(x)
    :param max_retries,0不重试
    :param delay: 重试前延迟时间（秒）
    :return:
    """
    while True:
        task = await queue.get()
        if task is None:
            print("[Info] Received shutdown signal")
            queue.task_done()
            break
        try:
            success = await process_task(task, **kwargs)
            if not success and max_retries > 0:
                if isinstance(task, (list, tuple)) and isinstance(task[-1], int):
                    retry_count = task[-1]
                    task_data = task[:-1]
                    if retry_count < max_retries:
                        await asyncio.sleep(delay)
                        new_task = (*task_data, retry_count + 1)  # 重建任务，重试次数+1
                        await queue.put(new_task)
                        print(f"[Task Retry] {task_data} (attempt {retry_count + 1})")
                    else:
                        print(f"[Task Failed] {task_data}")

        except asyncio.CancelledError:
            print("[Cancel] Worker shutting down...")
            break
        except Exception as e:
            print(f"[Error] Unexpected error processing task: {e}")
        finally:
            queue.task_done()  # 必须调用，标记任务完成


async def stop_worker(queue: asyncio.Queue, worker_tasks: list):
    '''优雅停止所有 worker'''
    try:
        await queue.join()  # 等待队列清空

        print("All tasks processed. Stopping consumers...")
        for _ in worker_tasks:
            await queue.put(None)  # 发送停止信号
    except Exception as e:
        print(f"[Tasks Error] {e}, attempting to cancel workers...")
        for c in worker_tasks:
            c.cancel()

    finally:
        # 统一回收所有任务
        await asyncio.gather(*worker_tasks, return_exceptions=True)


# 异步生产者
async def producer_push(messages: list, queue_key: str = "message_queue", status_key: str = "task_status", redis=None,
                        sleep: float = 0):
    """
    异步任务生产者：将任务写入 Redis 队列和状态记录哈希表。

    Args:
        messages (list): 待写入的消息（字符串或 dict 列表）。
        queue_key (str): Redis 队列名称（用于 BLPOP 消费）。
        status_key (str): Redis 哈希表名称，用于记录任务状态。
        redis: Redis 实例（若为空则自动创建）。
        sleep (float): 每条消息写入间隔时间，单位秒。
    """
    # rpush, hset, 等写操作
    if redis is None:
        redis = get_redis()

    # await redis.rpush(queue_key, *messages)  # 异步放入队列
    for message in messages:
        if isinstance(message, (dict, list, tuple)):
            msg_str = json.dumps(message, ensure_ascii=False)
        else:
            msg_str = str(message)
        await redis.rpush(queue_key, msg_str)
        await redis.hset(status_key, msg_str, "pending")  # 如果多个任务有重复文本，状态会被覆盖
        print(f"Produced: {message}")
        if sleep > 0:
            await asyncio.sleep(sleep)


# 异步消费者
async def consumer_redis(process_task: Callable, queue_key: str | list[str] = "message_queue",
                         status_key: str = "task_status", redis=None, max_errors: int = 0, timeout: Optional[int] = 0,
                         **kwargs):
    """
    Redis 异步消费者
    异步处理任务
        task = asyncio.create_task(consumer_redis(process_task,queue_key,args..., **kwargs))
        task.cancel()
        await task  # 捕获异常后自然退出
    流式获取任务
        async for task in consumer_redis(None,queue_key):
            await handler(task)
            await asyncio.sleep(1)

    Args:
        process_task (Callable): 处理函数，接收 task_data, 返回 bool 表示是否成功
        queue_key (str): Redis 列表队列键名（用于 BLPOP）
        status_key (str): Redis 哈希，用于记录每条任务状态
        redis (Redis): Redis 实例
        max_errors (int): 连续失败最大次数，超过后退出（0 表示不限）
        timeout (int): 阻塞超时秒数
    """
    if redis is None:
        redis = get_redis()
    error_count = 0
    while True:
        try:
            # 异步阻塞消费,支持多队列监听 Left Pop, timeout fallback
            message = await redis.blpop([queue_key] if isinstance(queue_key, str) else queue_key, timeout=timeout)
            if not message:
                print("[Info] Queue empty, exiting...")
                break

            q, item = message  # 第一个元素 channel_name,监听的队列名,'item' 是 bytes 类型
            try:
                task_data = json.loads(item.decode('utf-8'))
                print(f"[Task] Consumed: {task_data}")  # message[1].decode()
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                print(f"[Error] Invalid task format: {item}, error: {e}")
                await redis.hset(status_key, item, f"invalid_format: {str(e)}")
                continue

            if process_task is None:
                yield task_data
            else:
                try:
                    success = await process_task(task_data, **kwargs)
                    status = "completed" if success else "failed"
                    # task_id = task_data.get("id", hash(item))
                    await redis.hset(status_key, item, status)
                    error_count = 0
                except asyncio.CancelledError:
                    print("[Info] Consumer received shutdown signal")
                    break
                except Exception as e:
                    error_count += 1
                    print(f"[Error] Task processing error: {e}")
                    await redis.hset(status_key, item, f"error: {str(e)}")
                    await asyncio.sleep(1)  # 防止崩溃循环

                    if error_count >= max_errors > 0:
                        print(f"[Fatal] Exceeded max_errors ({max_errors}), exiting consumer.")
                        break

        except Exception as e:
            print(f"[Critical] Redis error in consumer loop: {e}")
            await asyncio.sleep(2)

        # finally:
        #     await redis.close()
    print("[Exit] Consumer loop ended.")


# asyncio.create_task(

class AsyncAbortController:
    def __init__(self):
        self._abort_event = asyncio.Event()  # 线程安全 threading.Event()

    def should_abort(self) -> bool:
        """查询是否已触发终止信号,实时检查是否终止"""
        return self._abort_event.is_set()

    async def wait_abort(self):
        """等待终止信号（可用于并发 await）,可 await 等待中断"""
        await self._abort_event.wait()

    def abort(self):
        """外部触发终止信号,触发终止,是否提前终止,可供外部触发"""
        self._abort_event.set()

    def reset(self):
        """清除终止信号，为下一轮任务做准备,重新启动前复位,可多轮复用"""
        self._abort_event.clear()


class TaskStatus(Enum):
    # "pending" | "ready" | "running" | "done" | "failed"
    PENDING = "pending"  # 等待条件满足
    READY = "ready"  # 条件满足，可以执行
    IN_PROGRESS = "running"  # processing

    COMPLETED = "done"
    FAILED = "failed"
    RECEIVED = "received"


@dataclass
class TaskNode:
    name: str  # task_id 任务名或别名
    description: Optional[str] = None
    # type: Optional[str] = None  # 'api', 'llm', 'script'
    action: Optional[str] = None  # execute 任务的执行逻辑（可调用对象函数、脚本或引用的操作类型),可执行动作（如脚本、注册名、函数名）
    event: Any = None  # 触发标志（不处理依赖逻辑）事件是标识符，用于任务之间的触发,指示触发的事件类型和附加数据

    status: TaskStatus = TaskStatus.PENDING
    progress: int = 0  # 执行进度，适合长任务、流任务场景
    priority: int = 10  # 执行顺序控制
    command: Any = field(default_factory=dict)  # 节点执行函数动态跳转,goto,静态边走 TaskEdge
    tags: List[str] = field(default_factory=list)  # 分类/搜索，索引、分组、过滤

    start_time: float = field(default_factory=time.time)
    end_time: float = 0

    data: Any = field(default_factory=dict)  # 执行输入
    result: Any = field(default_factory=list)
    count: int = 0  # 结果数量

    @staticmethod
    def default_node_attrs(attributes: dict):
        default_attrs = {
            "status": "pending",
            "start_time": lambda: time.time(),
            "event": None,
            "action": None,
            **attributes
        }
        return {k: v() if callable(v) else v for k, v in default_attrs.items()}

    @property
    def duration(self) -> float:
        return (self.end_time - self.start_time) if self.end_time > self.start_time else 0.0

    async def to_redis(self, redis_client, ex=3600, key_prefix='task'):
        key = f"{key_prefix}:{self.name}"
        payload = dataclass2dict(self)
        payload.pop("data", None)
        await redis_client.setex(key, ex, json.dumps(payload, ensure_ascii=False))

    @classmethod
    async def from_redis(cls, redis_client, task_id: str, ex: int = 0, key_prefix='task'):
        key = f"{key_prefix}:{task_id}"
        value = await redis_client.get(key)
        if value:
            if ex > 0:
                await redis_client.expire(key, ex)
            data = json.loads(value)
            if "status" in data and data["status"] is not None:
                data["status"] = TaskStatus[data["status"]]
            return cls(**data)
        return None


class TaskManager:
    Task_queue: dict[str, TaskNode] = {}  # queue.Queue(maxsize=Config.MAX_TASKS)
    Task_lock = asyncio.Lock()
    key_prefix = 'task'

    @classmethod
    async def add_task(cls, task_id: str, task: TaskNode, redis_client=None, ex=3600):
        if redis_client:
            await task.to_redis(redis_client, ex=ex, key_prefix=cls.key_prefix)

        async with cls.Task_lock:
            cls.Task_queue[task_id or task.name] = task

    @classmethod
    async def remove_task(cls, task_id: str):
        async with cls.Task_lock:
            cls.Task_queue.pop(task_id, None)

    @classmethod
    async def get_task(cls, task_id: str, redis_client=None, ex: int = 0) -> TaskNode | None:
        async with cls.Task_lock:
            if task_id in cls.Task_queue:
                return cls.Task_queue[task_id]

        if redis_client:
            task = await TaskNode.from_redis(redis_client, task_id, ex=ex, key_prefix=cls.key_prefix)
            if task:
                async with cls.Task_lock:
                    cls.Task_queue[task_id] = task
            return task

        return None

    @classmethod
    async def get_task_status(cls, task_id: str, redis_client=None) -> TaskStatus | None:
        task = await cls.get_task(task_id, redis_client)
        if task:
            return task.status
        return None

    @classmethod
    async def set_task_status(cls, task: TaskNode, status: TaskStatus, progress: int = 10, redis_client=None):
        async with cls.Task_lock:
            task.status = status
            task.progress = progress
            cls.Task_queue[task.name] = task

        if redis_client:
            await task.to_redis(redis_client, key_prefix=cls.key_prefix)

    @classmethod
    async def update_task_result(cls, task_id: str, result, status: TaskStatus = TaskStatus.COMPLETED,
                                 redis_client=None):
        task: TaskNode = await cls.get_task(task_id, redis_client)
        if not task:
            print(f"[update_task_result] Task {task_id} not found.")
            return None

        async with cls.Task_lock:
            task.result = result
            task.count = len(result) if isinstance(result, (list, set, tuple)) else 1 if result else 0
            task.status = status
            task.end_time = time.time()
            if task.status in (TaskStatus.COMPLETED, TaskStatus.RECEIVED):
                task.data = None
                task.progress = 100
            elif task.status == TaskStatus.FAILED:
                print(f"Task failed: {dataclass2dict(task)}")

        if redis_client:
            await task.to_redis(redis_client, key_prefix=cls.key_prefix)

        return task.result

    @classmethod
    async def clean_old_tasks(cls, timeout_received=3600, timeout=86400):
        current_time = time.time()
        task_ids_to_delete = []

        for _id, task in cls.Task_queue.items():
            if task.end_time and (current_time - task.end_time) > timeout_received:
                if task.status == TaskStatus.RECEIVED:
                    task_ids_to_delete.append(_id)
                    print(f"Task {_id} has been marked for cleanup. Status: RECEIVED")
            elif (current_time - task.start_time) > timeout:
                task_ids_to_delete.append(_id)
                print(f"Task {_id} has been marked for cleanup. Timeout exceeded")

        if task_ids_to_delete:
            async with cls.Task_lock:
                for _id in task_ids_to_delete:
                    cls.Task_queue.pop(_id, None)


@dataclass
class TaskEdge:
    """定义任务依赖关系的有向边"""
    source: str  # 依赖的起始任务ID
    target: str  # 被触发的任务ID
    # 边上的触发条件（与源任务的状态相关),["done",{"deadline": time.time() + 60}]
    condition: Union[str, Dict[str, Any]]  # 触发条件，如 "done" 或 {"deadline": timestamp}

    # 使用field提供默认值以避免可变默认值问题,absolute,relative,[None, {"relative": 5}]
    trigger_time: Optional[Dict[str, Union[int, float]]] = field(
        default=None,
        metadata={
            "description": "时间触发配置，如 {'relative': 5}(秒) 或 {'absolute': 1680000000}(时间戳)"
        }
    )
    # 任务触发事件,None无依赖
    trigger_event: Optional[str] = field(
        default=None,
        metadata={"description": "事件名称，如 'file_uploaded'"}
    )
    # 复杂条件,函数或复杂的逻辑判断
    rule: Optional[Callable[..., bool]] = field(
        default=None,
        metadata={"description": "自定义条件函数，接受上下文返回布尔值"}
    )

    def __post_init__(self):
        """数据校验和转换"""
        self._validate_condition()
        self._normalize_trigger_time()

    def as_dict(self) -> dict:
        return dataclass2dict(self)

    @classmethod
    def from_dict(cls, data: dict, rule_function: Callable = None):
        if rule_function:
            if "rule" in data and isinstance(data["rule"], str):
                data["rule"] = rule_function(data["rule"])  # 从注册表或动态导入恢复函数
        return cls(**data)

    @classmethod
    def get(cls, key):
        return getattr(cls, key, None)

    def _validate_condition(self):
        """验证condition字段格式"""
        if isinstance(self.condition, str):
            if self.condition not in ("done", "failed", "running"):
                raise ValueError(f"Invalid condition string: {self.condition}")
        elif isinstance(self.condition, dict):
            deadline = self.condition.get("deadline")
            if deadline is not None and not isinstance(deadline, (int, float)):
                raise TypeError("Deadline must be numeric")

    def _normalize_trigger_time(self):
        """将相对时间转换为绝对时间戳"""
        if self.trigger_time and "relative" in self.trigger_time:
            self.trigger_time = {
                "absolute": time.time() + self.trigger_time["relative"]
            }

    def should_trigger(self, context: Dict[str, Any]) -> bool:
        """判断是否满足触发条件"""
        # 1. 检查时间条件
        if self.trigger_time and "absolute" in self.trigger_time:
            if time.time() < self.trigger_time["absolute"]:
                return False

        # 2. 检查基础条件
        if isinstance(self.condition, str):
            if context.get("status") != self.condition:
                return False
        elif isinstance(self.condition, dict):
            if "deadline" in self.condition:
                if time.time() < self.condition["deadline"]:
                    return False

        # 3. 检查自定义规则
        if self.rule and not self.rule(**context):
            return False

        return True


class TaskScheduler:
    Task_graph = ig.Graph(directed=True)  # 创建有向图

    def __init__(self, graph: Optional[ig.Graph] = None, driver: Optional[GraphDatabase] = None):
        self.graph = self.__class__.Task_graph.copy() if graph is None else graph  # 局部副本
        for attr in ["name", "status", "action", "start_time", "event", "priority"]:
            if attr not in self.graph.vs.attributes():  # "name" in Task_graph.vs.attributes()
                self.graph.vs[attr] = [None] * self.graph.vcount()

        self.driver = driver or get_neo_driver()

    def set_task_node(self, task_id: str, attributes: Dict[str, Any]) -> None:
        """添加或更新任务节点,添加节点到图中"""
        # nodes = self.graph.vs["name"]
        try:
            node = self.graph.vs.find(name=task_id)
            node.update_attributes(**attributes)
            # if task_id in nodes:
            # # self.graph.vs[node.index].update_attributes(**attributes)
            # for k, v in attributes.items():
            #     node[k] = v
        except ValueError:
            default_attrs = TaskNode.default_node_attrs(attributes)
            self.graph.add_vertex(name=task_id, **default_attrs)

    async def set_node(self, task_node: TaskNode, **kwargs):
        """添加或更新任务节点，使用 TaskNode 对象,向 Neo4j 添加或更新一个任务节点"""
        task_id = task_node.name
        attributes = dataclass2dict(task_node)  # asdict(task_node)
        attributes.update(kwargs)
        attributes.pop("data", None)
        try:
            node = self.graph.vs.find(name=task_id)
            node.update_attributes(**attributes)
        except ValueError:
            self.graph.add_vertex(name=task_id, **attributes)

        async with self.driver.session() as session:
            await session.execute_write(merge_neo_node, task_id, attributes, 'Task')
        return task_id

    def update_task_status(self, task_id: str, new_status: TaskStatus | str) -> None:
        """更新任务状态"""
        if isinstance(new_status, str):
            new_status = TaskStatus(new_status)
        node = self.graph.vs.find(name=task_id)
        node["status"] = new_status  # self.graph.vs[node.index]["status"] = new_status
        if new_status == TaskStatus.IN_PROGRESS:  # "running"
            node["start_time"] = time.time()

    @staticmethod
    async def update_task_status_neo(driver, task_id: str, new_status: TaskStatus | str):
        """
        将任务状态更新同步到 Neo4j 节点（Node 标签）

        Args:
            driver: Neo4j 异步驱动
            task_id (str): 节点 ID（即 name）
            new_status (TaskStatus | str): 新状态
        """
        if isinstance(new_status, TaskStatus):
            new_status = new_status.value  # 如果是 Enum，取出字符串

        start_time = time.time() if new_status == TaskStatus.IN_PROGRESS else None

        async def _update(tx, task_id, new_status, start_time):
            query = """
            MATCH (n:Task {id: $task_id})
            SET n.status = $new_status
            """ + (", n.start_time = $start_time" if start_time is not None else "")
            await tx.run(query, task_id=task_id, new_status=new_status, start_time=start_time)

        async with driver.session() as session:
            await session.execute_write(_update, task_id, new_status, start_time)

    def set_task_dependency(self, edge: TaskEdge | dict, **kwargs):
        """添加任务依赖关系"""
        if isinstance(edge, dict):
            edge = TaskEdge.from_dict(edge)  # TaskEdge(**edge)
        if edge.source not in self.graph.vs["name"] or edge.target not in self.graph.vs["name"]:
            raise ValueError("Source or target task does not exist")

        rel_props = {**edge.__dict__, **kwargs}
        rel_props.pop("source", None)
        rel_props.pop("target", None)

        if self.graph.are_adjacent(edge.source, edge.target):  # 是否直接相连（有无一条边）
            eid = self.graph.get_eid(edge.source, edge.target)
            self.graph.es[eid].update_attributes(**rel_props)
        else:
            self.graph.add_edge(edge.source, edge.target, **rel_props)
        return rel_props

    async def set_edge(self, task_edge: TaskEdge, **kwargs):
        """
        向 Neo4j 添加或更新一条任务依赖边（任务之间的依赖关系）

        Args:
            task_edge (TaskEdge): 包含 source、target 和其他边属性
            kwargs: 可选附加属性（将合并进 edge 属性中）
        """
        rel_props = self.set_task_dependency(task_edge, **kwargs)
        async with self.driver.session() as session:
            await session.execute_write(merge_neo_relationship, task_edge.source, task_edge.target, rel_props,
                                        'DEPENDS_ON', 'Task', 'Task')

    def graph_edge_to_task(self, edge: ig.Edge) -> TaskEdge:
        edge_dict = edge.attributes()
        # edge_dict.pop("source", None)
        # edge_dict.pop("target", None)
        return TaskEdge(source=self.graph.vs[edge.source]["name"], target=self.graph.vs[edge.target]["name"],
                        **edge_dict)

    def set_task_edges(self, edges: List[tuple], attributes: List[Dict[str, Any]]):
        # 批量设置边结构（源+目标+属性）
        for (source, target), attr in zip(edges, attributes):
            attrs = {k: v for k, v in attr.items() if k in TaskEdge.__annotations__}
            other_attrs = {k: v for k, v in attr.items() if k not in TaskEdge.__annotations__}
            edge = TaskEdge(source=source, target=target, **attrs)  # from_dict()
            self.set_task_dependency(edge, **other_attrs)

    def export_adjacency_list(self) -> dict:
        adjacency_list = defaultdict(list)

        for edge in self.graph.es:
            source = self.graph.vs[edge.source]["name"]
            target = self.graph.vs[edge.target]["name"]

            adjacency_list[source].append({
                "name": target,
                "attrs": edge.attributes()
            })

        return dict(adjacency_list)

    def export_nodes(self) -> dict:
        return {v["name"]: v.attributes() for v in self.graph.vs}
        # [self.graph.vs[n]["name"] for n in self.graph.successors(v.index)]

    def _check_condition(self, edge: TaskEdge | ig.Edge | Dict[str, Any]) -> int:
        """检查边触发条件是否满足"""
        if isinstance(edge, ig.Edge):
            edge = self.graph_edge_to_task(edge)
        elif isinstance(edge, dict):
            edge = TaskEdge.from_dict(edge)  # TaskEdge(**edge)

        source = self.graph.vs.find(name=edge.source)
        target = self.graph.vs.find(name=edge.target)

        # 基础状态检查
        if target["status"] != TaskStatus.PENDING:
            return 0

        # 条件类型处理,状态条件
        condition = edge.condition
        if condition is None:
            return 0

        if isinstance(condition, str):
            try:
                condition = TaskStatus(condition)
            except:
                print(condition)

        event_ready = 0
        if isinstance(condition, TaskStatus):
            if source["status"] != condition:
                return 0
            # 任务状态变化后触发事件驱动的任务边或者任务A完成时触发多个事件
            if edge.trigger_event and source.get("event") != edge.trigger_event:
                return 0
            # 检查时间条件>
            if edge.trigger_time:
                if "absolute" in edge.trigger_time and time.time() < edge.trigger_time["absolute"]:
                    return 0
                if "relative" in edge.trigger_time and time.time() < source["start_time"] + edge.trigger_time[
                    "relative"]:
                    return 0
            event_ready = 1

        elif isinstance(condition, dict) and "deadline" in condition:
            if time.time() > condition["deadline"]:
                return 0
            event_ready = 1 << 1  # 时间条件<
        elif callable(condition):
            if not condition():
                return 0
            event_ready = 1 << 2  # 自定义条件
        # 自定义规则
        if edge.rule:
            if not edge.rule():
                return 0

        return event_ready

    def check_and_trigger_tasks(self) -> List[str]:
        """检查并触发符合条件的任务"""
        triggered = []
        for edge in self.graph.es:  # 遍历全部边
            if self._check_condition(edge) > 0:
                target = self.graph.vs.find(name=edge.target)
                target["status"] = TaskStatus.READY
                triggered.append(target["name"])
                print(f"Task {target['name']} triggered by {edge.source}->{edge.target}")
        return triggered

    @staticmethod
    def _simulate_task_execution(task):
        # 模拟任务执行
        attrs = task.attributes()
        action = attrs.get("action")
        if isinstance(action, str):
            print(f"Simulating execution for {task['name']}:{action}")
        elif callable(action):
            return action()
        time.sleep(1)
        return True

    def execute_ready_tasks(self, executor: Callable[[Any], bool] = None) -> bool:
        """执行所有就绪任务"""
        # 查找状态为 ready 的任务并执行
        ready_tasks = [v for v in self.graph.vs if v["status"] == TaskStatus.READY]
        if not ready_tasks:
            return False  # 无可执行任务时退出

        for task in ready_tasks:
            print(f"Executing Ready task {task['name']}...")
            task["status"] = TaskStatus.IN_PROGRESS  # "running"
            try:
                exec_func = executor or self._simulate_task_execution
                result = exec_func(task)
                # 任务执行成功
                task["status"] = TaskStatus.COMPLETED if result else TaskStatus.FAILED  # "done"'completed'
            except Exception as e:
                task["status"] = TaskStatus.FAILED
                print(f"Task {task['name']} failed,error: {e}")

        return True

    def step(self, executor: Callable[[Any], bool] = None) -> bool:
        triggered = self.check_and_trigger_tasks()  # 检查并触发新的任务
        executed = self.execute_ready_tasks(executor)
        return triggered or executed

    def run_scheduler(self, max_cycles: int = 100) -> None:
        if not self.graph.is_dag():
            raise ValueError("任务图存在循环依赖，无法调度")
        """运行任务调度主循环"""
        if max_cycles:
            for _ in range(max_cycles):
                if not self.step():
                    break
                time.sleep(1)
        else:
            while True:
                if not self.step():
                    break
                time.sleep(1)

        print("Scheduler finished")

        # 检查依赖并触发任务


def get_children(g, node):
    # g.successors(node.index)
    return [g.vs[neighbor]["name"] for neighbor in g.neighbors(node, mode="OUT")]


def get_parent(g, node):
    # g.predecessors(node.index)
    neighbors = g.neighbors(node, mode="IN")
    return g.vs[neighbors[0]]["name"] if neighbors else None


def load_graph_from_dict(nodes: dict, adjacency_list: dict) -> ig.Graph:
    """
    从节点信息和邻接表构建 igraph.Graph 对象，并赋值给 self.graph

    Args:
        nodes (dict): 节点字典，如 {"n1": {"content": "Q1"}, "n2": {"content": "Q2"}}
        adjacency_list (dict): 邻接表，如 {"node1": [{"name": "node2", "attrs": {...}}, ...], ...}

    Returns:
        ig.Graph: 构建完成的图对象
    """
    g = ig.Graph(directed=True)

    node_names = list(nodes.keys())
    name_to_index = {name: i for i, name in enumerate(node_names)}

    g.add_vertices(len(node_names))
    g.vs["name"] = node_names
    # 设置节点属性
    for attr in set(k for v in nodes.values() for k in v):
        g.vs[attr] = [nodes[name].get(attr) for name in node_names]

    # 添加边及其属性
    edge_list = []
    edge_attrs = defaultdict(list)

    for src, neighbors in adjacency_list.items():
        for neighbor in neighbors:
            tgt = neighbor["name"]
            edge_list.append((name_to_index[src], name_to_index[tgt]))

            # 提取所有边属性（如果没有则为 None）
            attrs = neighbor.get("attrs", {})
            for k in attrs:
                edge_attrs[k].append(attrs.get(k))
            # 对没有 attrs 的键也保持齐整性
            for k in edge_attrs:
                if k not in attrs:
                    edge_attrs[k].append(None)

    g.add_edges(edge_list)
    # 设置边属性
    for attr, values in edge_attrs.items():
        g.es[attr] = values

    return g


async def export_graph_to_neo4j(nodes: dict, adjacency_list: dict, driver, g: ig.Graph = None, label: str = 'Node',
                                rel_type: str = 'RELATED'):
    """
    将图导出到 Neo4j

    label: 节点标签名
    rel_type: 关系类型名
    nodes: {
        "n1": {"content": "Q1", "type": "root"},
        "n2": {"content": "Q2", "type": "search"}
    }
    adjacency_list: {
        "n1": [{"name": "n2", "attrs": {"relation": "connects"}}]
    }
    """
    async with driver.session() as session:
        if g:
            for v in g.vs:
                node_id = v["name"]
                props = {k: v[k] for k in g.vs.attributes() if k != "name"}
                await session.execute_write(merge_neo_node, node_id, props, label)

            for e in g.es:
                src_id = g.vs[e.source]["name"]
                tgt_id = g.vs[e.target]["name"]
                props = {k: e[k] for k in g.es.attributes()}
                await session.execute_write(merge_neo_relationship, src_id, tgt_id, props, rel_type, label, label)

        else:
            for node_id, props in nodes.items():
                await session.execute_write(merge_neo_node, node_id, props, label)

            for src, neighbors in adjacency_list.items():
                for neighbor in neighbors:
                    tgt = neighbor["name"]
                    rel_props = neighbor.get("attrs", {})
                    await session.execute_write(merge_neo_relationship, src, tgt, rel_props, rel_type, label, label)


def export_to_json_from_graph(graph, filename):
    data = {"nodes": [{"id": v.index, **v.attributes()} for v in graph.vs],
            "edges": [{"source": e.source, "target": e.target, **e.attributes()} for e in graph.es], }
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

    # graph.write_pickle("graph.pkl")


def import_to_graph_from_json(filename):
    with open(filename, "r") as f:
        data = json.load(f)

    g = ig.Graph(directed=True)  # ig.Graph.TupleList(
    g.add_vertices(len(data["nodes"]))
    # g.vs["name"] = [node["id"] for node in data["nodes"]]
    # 为节点设置属性
    for idx, node in enumerate(data["nodes"]):
        g.vs[idx].update_attributes(node)  # node.items()

    g.add_edges([(edge["source"], edge["target"]) for edge in data["edges"]])  # [("task1", "task2")]

    # 为边设置属性
    for idx, edge in enumerate(data["edges"]):
        # 跳过 source 和 target 属性
        g.es[idx].update_attributes({k: v for k, v in edge.items() if k not in ["source", "target"]})
        # g.es[idx][key] = value

    return g


class WebSearchGraph:
    def __init__(self):
        import queue

        # 初始化节点内容字典
        self.nodes: Dict[str, Dict[str, str]] = {}
        # 初始化邻接表
        self.adjacency_list: Dict[str, List[dict]] = defaultdict(list)
        self.task_queue = queue.Queue()
        self.n_active_tasks = 0

    async def add_root_node(self, node_content: str, node_name: str = 'root'):
        # 添加根节点
        self.nodes[node_name] = dict(content=node_content, type="root")
        # 在图中添加节点

        self.adjacency_list[node_name] = []
        return node_name

    async def add_node(self, node_name: str, node_content: str):
        # 添加子问题节点
        self.nodes[node_name] = dict(content=node_content, type="search")

        self.adjacency_list[node_name] = []

        # 处理父节点，查找相关的历史上下文
        parent_response = []
        for start_node, adj in self.adjacency_list.items():
            for neighbor in adj:
                if (node_name == neighbor["name"]  # 判断是否有连接,是否是当前节点的父节点，并且该父节点包含 response 信息
                        and start_node in self.nodes and "response" in self.nodes[start_node]):
                    parent_response.append(
                        dict(question=self.nodes[start_node]["content"], answer=self.nodes[start_node]["response"]))

        await self._async_node_stream(node_name, node_content)

        self.n_active_tasks += 1  # f"{node_name}-{node_content}"
        return self.n_active_tasks

    async def _async_node_stream(self, node_name: str, node_content: str, parent_response: List[dict]):
        """执行异步搜索"""
        cfg = {"search_config": "value"}  # 配置搜索
        session_id = random.randint(0, 999999)  # 会话ID
        agent = None

        try:
            # 模拟搜索过程
            searcher_message = "mock_search_message"  # 假设的搜索消息
            self.nodes[node_name]["response"] = searcher_message  # 更新节点响应
            self.nodes[node_name]["session_id"] = session_id
            self.task_queue.put((node_name, self.nodes[node_name]))  # 将结果放入队列
        except Exception as exc:
            self.task_queue.put((exc, None))

    async def add_response_node(self, node_name: str = 'response'):
        # 添加回复节点
        self.nodes[node_name] = dict(content='Search completed, thought response added.', type="response")
        # self.adjacency_list[node_name] = []
        self.task_queue.put((node_name, self.nodes[node_name], []))

    async def add_edge(self, start_node: str, end_node: str):
        self.adjacency_list[start_node].append(dict(id=str(uuid.uuid4()), name=end_node, state=2))

        self.task_queue.put((start_node, self.nodes[start_node], self.adjacency_list[start_node]))

    async def reset(self):
        # 重置图和节点
        self.nodes.clear()
        self.adjacency_list.clear()

    def node(self, node_name: str):
        # 获取节点信息
        if node_name in self.nodes:
            return self.nodes[node_name].copy()

        return None

    def graph(self):
        """根据节点信息和邻接表生成图
        nodes = {
            "root": {"content": "What is AI?", "type": "root"},
            "node1": {"content": "What is machine learning?", "type": "search"},
            "node2": {"content": "What is deep learning?", "type": "search"}
        }

        adjacency_list = {
            "root": [{"name": "node1"}, {"name": "node2"}],
            "node1": [{"name": "node2"}],
            "node2": []
        }

        Returns:
            ig.Graph: 返回生成的图对象
        """

        return load_graph_from_dict(self.nodes, self.adjacency_list)


# class Task(Base):
#     __tablename__ = 'tasks'
#
#     task_id = Column(String, primary_key=True)
#     status = Column(Enum(TaskStatus), default=TaskStatus.PENDING)
#
#     # 创建任务的方法
#     @classmethod
#     def create_task(cls, session: Session):
#         task_id = str(uuid.uuid4())  # 生成唯一 task_id
#         new_task = cls(task_id=task_id)  # 创建任务实例
#         session.add(new_task)
#         session.commit()
#         return task_id
#
#     # 更新任务状态的方法
#     @classmethod
#     def update_task_status(cls, session: Session, task_id: str, new_status: TaskStatus):
#         task = session.query(cls).filter_by(task_id=task_id).first()
#         if task:
#             task.status = new_status
#             session.commit()
#
#     # 获取任务状态的方法
#     @classmethod
#     def get_task_status(cls, session: Session, task_id: str):
#         task = session.query(cls).filter_by(task_id=task_id).first()
#         if task:
#             return task.status
#         return None
#
#     # 模拟异步任务执行
#     @classmethod
#     def async_task(cls, session: Session, task_id: str):
#         import time
#         try:
#             cls.update_task_status(session, task_id, TaskStatus.IN_PROGRESS)
#             time.sleep(1)
#             cls.update_task_status(session, task_id, TaskStatus.COMPLETED)
#         except Exception as e:
#             cls.update_task_status(session, task_id, TaskStatus.FAILED)

_StringLikeT = Union[bytes, str, memoryview]


def list_or_args_keys(keys: Union[_StringLikeT, Iterable[_StringLikeT]],
                      args: Tuple[_StringLikeT, ...] = None) -> List[_StringLikeT]:
    # 将 keys 和 args 合并成一个新的列表
    # returns a single new list combining keys and args
    try:
        iter(keys)
        # a string or bytes instance can be iterated, but indicates
        # keys wasn't passed as a list
        if isinstance(keys, (bytes, str)):
            keys = [keys]
        else:
            keys = list(keys)  # itertools.chain.from_iterable(keys)
    except TypeError:
        keys = [keys]
    if args:
        keys.extend(args)
    return keys


# async def setr(key, value, ex=None):
#     redis = get_redis()
#     await redis.set(name=key, value=value, ex=ex)
#     # await redis.delete(key) redis.get(key, encoding='utf-8')

# pip install celery[redis]
# celery = Celery('tasks', broker='redis://localhost:6379/0') #Celery 任务
# message_queue = asyncio.Queue()


class MessageZeroMQ:

    def __init__(self, pull_port="7556", push_port="7557", req_port="7555", process_callback=None):
        self.context = zmq.asyncio.Context(io_threads=2)  # zmq.Context()

        # 设置接收消息的 socket
        self.pull_socket = self.context.socket(zmq.PULL)
        self.pull_socket.bind(f"tcp://*:{pull_port}")  # 绑定接收端口

        # 设置发送消息的 socket
        self.push_socket = self.context.socket(zmq.PUSH)
        self.push_socket.connect(f"tcp://localhost:{push_port}")  # 连接到 Java 的接收端口

        # 设置 REQ socket 用于请求-响应模式
        self.req_socket = self.context.socket(zmq.REQ)  # zmq.DEALER
        if req_port:
            self.req_socket.connect(f"tcp://localhost:{req_port}")  # 连接到服务端

        self.process_callback = process_callback or self.default_process_message
        # self.push_socket.send_string('topic1 Hello, world!')

    def __del__(self):
        self.context.destroy(linger=0)

    @staticmethod
    def default_process_message(message):
        # 处理逻辑
        print(f"Processing message: {message}")
        # 对消息进行某些处理后需要将其转发回 Java
        return f"Processed: {message}"

    async def send_request(self, message: str = "Hello, server!", topic: str = 'Request'):
        """
        使用 REQ socket 主动发送消息并接收响应,主动请求-响应
        """
        await self.req_socket.send_string(f'{topic} {message}')
        print(f"Sent request: {message} under topic: {topic}")
        response = await self.req_socket.recv_string()
        print(f"Received response: {response}")
        return response

    async def call_service(self, data):
        """
        使用 REQ socket 发送 JSON 数据并接收 JSON 响应 zmq.Context()
        """
        self.req_socket.send_json(data)
        response = self.req_socket.recv_json()
        # message = await self.req_socket.recv()
        # response = json.loads(message.decode())
        print(f"Received response: {response}")
        return response

    # 主动发送消息到 ZeroMQ
    async def send_message(self, message: str, topic: str = "Default"):
        await self.push_socket.send_string(f'{topic} {message}', flags=zmq.DONTWAIT, encoding='utf-8')
        print(f"Sent message: {message}")

    async def send_data(self, data=b'Hello in binary'):
        await self.push_socket.send(data, flags=zmq.DONTWAIT, copy=True, track=False)

    async def recv_messages(self):
        while True:
            # 将接收到的消息 yield 出去
            message = await self.pull_socket.recv_string()
            yield message

    async def stream_start(self):
        async for message in self.recv_messages():
            print(f"Received from ZeroMQ: {message}")
            processed_msg = self.process_callback(message)
            await self.send_message(processed_msg, topic="Processed")

    async def start(self):
        # 使用 asyncio 和 run_in_executor 进行阻塞操作
        # loop = asyncio.get_event_loop()

        while True:
            # 接收消息
            message = await self.pull_socket.recv_string()
            print(f"Received from ZeroMQ: {message}")

            # 处理消息
            processed_msg = self.process_callback(message)

            # 将处理后的消息发送
            await self.push_socket.send_string(processed_msg)
            print(f"Sent processed message back.")


#
# import pika
# https://www.rabbitmq.com/tutorials
#
# # Pika is a RabbitMQ,发送消息到 RabbitMQ
# def send_event(event_data):
#     connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
#     channel = connection.channel()
#     channel.queue_declare(queue='event_queue')
#
#     channel.basic_publish(exchange='', routing_key='event_queue',
#                           body=event_data)
#     connection.close()
#
#
# # 监听消息并调用服务
# def listen_for_events(callback):
#     connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
#     channel = connection.channel()
#     channel.queue_declare(queue='event_queue')
#
#     channel.basic_consume(queue='event_queue', on_message_callback=callback, auto_ack=True)
#
#     print('Waiting for events...')
#     channel.start_consuming()


if __name__ == "__main__":
    from utils import configure_event_loop

    configure_event_loop()


    # message_zero_mq = MessageZeroMQ()  # 创建消息处理器实例
    # asyncio.run(message_zero_mq.stream_start())
    # asyncio.run(message_zero_mq.start())

    # 运行异步任务
    # async def main():
    #     await asyncio.gather(producer_push([f"Message {i}" for i in range(5)]), consumer())
    # redis = Redis(host="localhost", port=6379)
    # await redis.set("key", "value")
    # value = await redis.get("key")
    # print(value.decode())  # 输出: value
    # await redis.close()

    #
    # asyncio.run(main())

    def test():
        import pprint
        scheduler = TaskScheduler()

        # 添加任务节点
        scheduler.set_task_node("task1", {"priority": 1})
        scheduler.set_task_node("task2", {"priority": 2})
        scheduler.set_task_node("task3", {"priority": 3})

        # 添加依赖关系
        scheduler.set_task_dependency(TaskEdge(
            source="task1",
            target="task2",
            condition="done",
            trigger_time={"relative": 1}
        ))

        scheduler.set_task_dependency(TaskEdge(
            source="task2",
            target="task3",
            condition="done",  # 自定义条件
            rule=lambda: time.localtime().tm_hour < 23  # 晚上11点前才触发
        ))

        # 启动任务
        scheduler.update_task_status("task1", TaskStatus.COMPLETED)  # "done"

        abs_time = time.time() + 5  # taskY 会延迟 5 秒后变为 ready 并执行
        scheduler.set_task_node("taskX", {"action": 'X'})
        scheduler.set_task_node("taskY", {"action": 'y'})

        scheduler.set_task_dependency(TaskEdge("taskX", "taskY", condition="done", trigger_time={"absolute": abs_time}))
        scheduler.update_task_status("taskX", TaskStatus.READY)

        # scheduler.set_task_node("A", {})
        # scheduler.set_task_node("B", {})
        # scheduler.set_task_node("C", {})
        #
        # scheduler.add_task_dependency(TaskEdge("A", "B", condition="done"))
        # scheduler.add_task_dependency(TaskEdge("B", "C", condition="done"))
        # scheduler.add_task_dependency(TaskEdge("C", "A", condition="done"))  # 闭环

        print(scheduler.graph)
        pprint.pprint([(v["name"], v["status"]) for v in scheduler.graph.vs])

        scheduler.run_scheduler()

        for v in scheduler.graph.vs:
            print(f"Task {v['name']}: {v['status']}")

        print(scheduler.export_nodes())
        print(scheduler.export_adjacency_list())

        async def test_save():
            await scheduler.set_node(task_node=TaskNode(name='taskX', action='X'))
            await scheduler.set_node(task_node=TaskNode(name='taskY', action='Y'))
            await scheduler.set_edge(task_edge=TaskEdge(source='taskX', target='taskY', condition='done',
                                                        trigger_time={"absolute": abs_time}))

        asyncio.run(test_save())


    test()
