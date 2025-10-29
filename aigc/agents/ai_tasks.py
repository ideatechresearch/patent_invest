import igraph as ig
import json, time, random, os, pickle
from datetime import datetime, timedelta
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue
from functools import partial, wraps
from redis.asyncio import Redis, StrictRedis, ConnectionPool
from typing import Dict, List, Tuple, Union, Iterable, Callable, Optional, Any, Type, Awaitable
from enum import IntEnum, Enum, IntFlag
from collections import defaultdict
from dataclasses import asdict, dataclass, is_dataclass, field, fields, replace
import uuid
import logging
import zmq, zmq.asyncio
from neo4j import GraphDatabase, AsyncGraphDatabase
from dask.distributed import Client as DaskClient
from dask import delayed

from structs import dataclass2dict
from config import Config
from service import get_redis, get_neo_driver, get_dask_client
from utils import pickle_deserialize


def store_in_dask(client, obj, key):
    future = client.scatter(obj)
    client.futures[key] = future


def get_from_dask(client, key):
    return client.futures[key].result()  # 获取复杂对象


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


class TaskStatus(Enum):
    # "pending" | "ready" | "running" | "done" | "failed"
    PENDING = "pending"  # 等待条件满足，（包含 created、inited、waiting）

    READY = "ready"  # 条件满足，可以执行
    IN_PROGRESS = "running"  # processing，retrying

    COMPLETED = "done"  # completed
    FAILED = "failed"  # error，timeout，这里不做细分

    RECEIVED = "received"  # 客户接收数据，等待延时释放
    CANCELLED = "cancelled"  # 手动取消,expired


class TaskCommand(Enum):
    GOTO = "goto"  # 跳转到指定节点
    CALL = "call"  # 调用子工作流
    RETURN = "return"  # 返回上级工作流
    BREAK = "break"  # 跳出循环
    CONTINUE = "continue"  # 继续下一轮循环
    EXIT = "exit"  # 终止工作流


@dataclass
class TaskNode:
    name: str  # task_id 任务名或别名
    description: Optional[str] = None  # 任务描述 内容 content
    # type: Optional[str] = None  # 'api', 'llm', 'script'
    action: Optional[str] = None  # 可执行动作,任务用途(如脚本、注册名、函数名、脚本或引用的操作类型) strategies
    function: Optional[Callable] = field(default=None)  # execute 激活函数任务的执行逻辑（可调用对象函数,绑定后的 partial(func, args)）
    event: Optional[Dict[str, Any]] = None  # 触发标志（不处理依赖逻辑）事件是标识符，用于任务之间的触发,指示触发的事件类型和附加数据,外部触发信号
    command: Dict[str, Any] = field(default_factory=dict)  # cmd 任务控制逻辑,节点执行函数动态跳转,goto,静态边走 TaskEdge，内部控制

    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0  # 执行进度，适合长任务、流任务场景（0-100）
    priority: int = 10  # bias 执行顺序控制，优先级
    tags: List[str] = field(default_factory=list)  # channel,label 辅助标签，分类/搜索，索引、分组、过滤

    start_time: float = field(default_factory=time.time)  # created_at
    end_time: float = 0  # updated_at

    data: Any = field(default_factory=dict)  # metadata,动态更新参数，自由定义不用明确结构，此处不限制，保持灵活性，Dict[str, Any]
    params: Any = field(default_factory=dict)  # input payload,context,message 执行输入参数,dict/list[dict]
    result: Any = field(default_factory=list)  # output 输出结果，允许 reason，error，此处不限制，不约束类型保持灵活性
    count: int = 0  # 结果条目数量，自定义 result_count,event_count

    @staticmethod
    def default_node_attrs(task_id: str, **attributes):
        auto_call = {"start_time", "end_time", "create_time"}
        default_attrs = {
            "name": task_id,
            "status": TaskStatus.PENDING,
            "start_time": time.time,
            "event": None,
            "action": None,
            "function": None,
            **attributes
        }
        return {k: v() if callable(v) and k in auto_call else v
                for k, v in default_attrs.items()}

    @property
    def duration(self) -> float:
        return (self.end_time - self.start_time) if self.end_time > self.start_time else 0.0

    async def to_redis(self, redis_client, ex: int = 3600, key_prefix='task'):
        key = f"{key_prefix}:{self.name}"
        payload = dataclass2dict(self)
        if self.function and (asyncio.iscoroutinefunction(self.function) or asyncio.iscoroutine(self.function)):
            payload.pop("function", None)
        # payload.pop("data", None) # pickle.dumps(payload["data"])
        await redis_client.setex(key, ex, json.dumps(payload, ensure_ascii=False))

    @classmethod
    async def from_redis(cls, redis_client, task_id: str, ex: int = 0, key_prefix='task'):
        key = f"{key_prefix}:{task_id}"
        value = await redis_client.get(key)
        if not value:
            return None

        if ex > 0:
            await redis_client.expire(key, ex)
        data = json.loads(value)
        return cls.set_data(data)

    @classmethod
    def set_data(cls, data: dict):
        if "status" in data and data["status"] is not None:
            data["status"] = TaskStatus(data["status"])
        if "params" in data:
            data["params"] = pickle_deserialize(data["params"])
        if "data" in data:
            data["data"] = pickle_deserialize(data["data"])  # pickle.loads(data["data"])
        if "function" in data:
            data["function"] = pickle_deserialize(data["function"])

        return cls(**data)

    def copy_and_update(self, *args, **kwargs):
        """
        Copy the object and update it with the given Info
        instance and/or other keyword arguments.
        """
        new_info = asdict(self)

        for si in args:
            if isinstance(si, TaskNode):
                new_info.update({k: v for k, v in asdict(si).items() if v is not None})
            elif isinstance(si, dict):
                new_info.update({k: v for k, v in si.items() if v is not None})

        if kwargs:
            new_info.update(kwargs)

        # for k, v in new_info.items():
        #     setattr(self, k, v)
        return replace(self, **new_info)  # 返回一个新的 TaskNode 实例


class TaskManager:
    Task_queue: dict[str, TaskNode] = {}  # task_map queue.Queue(maxsize=Config.MAX_TASKS)
    Task_lock = asyncio.Lock()
    key_prefix = 'task'

    @classmethod
    async def add_task(cls, task_id: str, task: TaskNode, redis_client=None, ex: int = 3600):
        if redis_client:
            await task.to_redis(redis_client, ex=ex, key_prefix=cls.key_prefix)

        async with cls.Task_lock:
            cls.Task_queue[task_id or task.name] = task

    @classmethod
    async def add(cls, redis=None, ex: int = 3600, **kwargs) -> tuple[str, TaskNode]:
        task_id = kwargs.pop('name', str(uuid.uuid4()))
        redis = redis or get_redis()
        task_fields = TaskNode.default_node_attrs(task_id, **kwargs)
        task = TaskNode(**task_fields)
        await cls.add_task(task_id, task, redis_client=redis, ex=ex)
        return task_id, task

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
    async def set_task_status(cls, task: TaskNode, status: TaskStatus, progress: float = 0, redis_client=None):
        async with cls.Task_lock:
            task.status = status
            if progress > 0:
                task.progress = progress
            cls.Task_queue[task.name] = task

        if redis_client:
            await task.to_redis(redis_client, key_prefix=cls.key_prefix)

    @classmethod
    async def put_task_result(cls, task: TaskNode, result, total: int = -1, status: TaskStatus = None,
                              params: dict = None, redis_client=None) -> TaskNode:
        async with cls.Task_lock:
            task.result.append(result)
            task.count = len(task.result)
            if params is not None:
                task.params = params

            if total > 0:
                task.progress = round(task.count * 100 / total, 2)
                if task.count >= total:
                    task.progress = 100.0
            if status is not None:
                task.status = status
            else:
                if task.count >= total > 0:
                    task.status = TaskStatus.COMPLETED  # 标记完成,最终状态
                else:
                    task.status = TaskStatus.IN_PROGRESS  # 中间更新,-1 任务还没结束

            task.end_time = time.time()
            cls.Task_queue[task.name] = task

        if redis_client:  # 把IO操作放到锁外
            await task.to_redis(redis_client, key_prefix=cls.key_prefix)

        return task

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


class ConditionFlag(IntFlag):
    NONE = 0
    STATUS_MATCH = 1 << 0
    TIME_OK = 1 << 1
    CUSTOM_OK = 1 << 2


@dataclass
class TaskEdge:
    """定义任务依赖关系的有向边"""
    relation: tuple[str, str]  # (source,target),依赖的起始任务ID,被触发的任务ID
    # 边上的触发条件（与源任务的状态相关),["done",{"deadline": time.time() + 60}]
    condition: Union[str, Dict[str, Any]]  # 触发条件，如 "done" 或 {"deadline": timestamp}，dependencies,lambda->rule

    # 使用field提供默认值以避免可变默认值问题,absolute,relative,[None, {"relative": 5}]
    trigger_time: Optional[Dict[str, Union[int, float]]] = field(
        default=None,
        metadata={"description": "时间触发配置，如 {'relative': 5}(秒) 或 {'absolute': 1680000000}(时间戳)"}
    )
    # 任务触发事件,None无依赖
    trigger_event: Optional[str] = field(default=None, metadata={"description": "事件名称，如 'file_uploaded'"})
    # 复杂条件,函数或复杂的逻辑判断
    rule: Optional[Callable[..., bool]] = field(default=None,
                                                metadata={"description": "自定义条件函数，接受上下文返回布尔值"})
    map: Optional[dict] = field(default_factory=dict,
                                metadata={"description": "挂载函数，对上游结果处理映射,参数变换逻辑"})

    weight: float = 1.0  # 适用于图遍历/调度优先级；-1,0,1

    def __post_init__(self):
        """数据校验和转换"""
        self._validate_condition()
        self._normalize_trigger_time()

    def as_dict(self) -> dict:
        return dataclass2dict(self)

    @classmethod
    def set_data(cls, data: dict, rule_function: Optional[Callable] = None):
        if "rule" in data:
            rule_obj = pickle_deserialize(data["rule"])  # pickle.loads()
            if rule_function and isinstance(rule_obj, str):
                rule_obj = rule_function(rule_obj)  # 从注册表或动态导入恢复函数,用于从函数名等恢复为函数对象
            data["rule"] = rule_obj

        field_names = {f.name for f in fields(cls) if f.init}
        attrs = {k: v for k, v in data.items() if k in field_names}  # cls.__annotations__
        return cls(**attrs)

    @classmethod
    def get(cls, key):
        return getattr(cls, key, None)

    def _validate_condition(self):
        """验证condition字段格式"""
        condition = self.condition  # STR
        if isinstance(condition, TaskStatus):
            condition = condition.value
        if isinstance(condition, str):
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

    def _check_status_condition(self, source: dict) -> bool:
        if self.condition is None:
            return False

        condition = self.condition  # STR
        if isinstance(condition, TaskStatus):
            condition = condition.value
        source_status = source.get("status")
        if isinstance(source_status, TaskStatus):
            source_status = source_status.value

        return source_status == condition

    def should_trigger(self, context: Dict[str, Any]) -> bool:
        """判断是否满足触发条件"""
        # 1. 检查时间条件
        if self.trigger_time and "absolute" in self.trigger_time:
            if time.time() < self.trigger_time["absolute"]:
                return False

        # 2. 检查基础条件
        if isinstance(self.condition, str):
            if not self._check_status_condition(context):
                return False
        elif isinstance(self.condition, dict):
            if "deadline" in self.condition:
                if time.time() < self.condition["deadline"]:
                    return False

        # 3. 检查自定义规则
        if self.rule and not self.rule(**context):
            return False

        return True

    @staticmethod
    def _evaluate_rule(rule, context: Dict[str, Any], conditions=[]):
        """评估复杂规则对象"""
        rule_type = rule.get("type", "and")

        if rule_type == "and":
            return all(c.should_trigger(context) for c in conditions)
        elif rule_type == "or":
            return any(c.should_trigger(context) for c in conditions)
        elif rule_type == "not":
            return not conditions[0].should_trigger(context)
        elif rule_type == "compare":
            left = context.get("left")
            right = context.get("right")
            op = rule["operator"]

            if op == "==": return left == right
            if op == "!=": return left != right
            if op == ">": return left > right
            if op == "<": return left < right
            if op == ">=": return left >= right
            if op == "<=": return left <= right
            if op == "in": return left in right
            if op == "not_in": return left not in right

        return False


class TaskGraphManager:
    Task_graph = ig.Graph(directed=True)  # 创建有向图
    Dask_client = None

    def __init__(self, graph: Optional[ig.Graph] = None, driver: Optional[GraphDatabase] = None,
                 dask_client: Optional[DaskClient] = None):
        self.graph = self.__class__.Task_graph.copy() if graph is None else graph  # 局部副本
        for attr in ["name", "status", "action", "function", "start_time", "event", "priority"]:
            if attr not in self.graph.vs.attributes():  # "name" in Task_graph.vs.attributes()
                self.graph.vs[attr] = [None] * self.graph.vcount()

        self.driver = driver  # get_neo_driver()
        if not self.__class__.Dask_client and dask_client:
            self.__class__.Dask_client = dask_client  # get_dask_client(cluster)

        self.client = self.__class__.Dask_client
        self.context = None  # history,global,update,relevant,extract,reflect,optimize,analyze,generate,decision

    def set_task_node(self, task_id: str, attributes: Dict[str, Any]) -> ig.Vertex:
        """添加或更新任务节点,添加节点到图中,对name去重"""
        # nodes = self.graph.vs["name"]
        if not task_id:
            task_id = str(uuid.uuid4())
        try:
            node = self.graph.vs.find(name=task_id)
            node.update_attributes(**attributes)
            # if task_id in nodes:
            # # self.graph.vs[node.index].update_attributes(**attributes)
            # for k, v in attributes.items():
            #     node[k] = v
        except ValueError:
            default_attrs = TaskNode.default_node_attrs(task_id, **attributes)
            node = self.graph.add_vertex(**default_attrs)
        return node

    def task_to_vertex(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(self.graph.vs["name"])}

    async def set_node(self, task_node: TaskNode, **kwargs) -> ig.Vertex:
        """添加或更新任务节点，使用 TaskNode 对象,向 Neo4j 添加或更新一个任务节点"""
        task_id = task_node.name
        attributes = dataclass2dict(task_node)  # asdict(task_node)
        attributes.update(kwargs)
        attributes.pop("data", None)
        try:
            node = self.graph.vs.find(name=task_id)
            node.update_attributes(**attributes)
        except ValueError:
            node = self.graph.add_vertex(**attributes)
        if self.driver:
            async with self.driver.session() as session:
                await session.execute_write(merge_neo_node, task_id, attributes, 'Task')
        return node

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
        """添加任务依赖关系，对连接去重，source，target对应 name:task_id"""
        if isinstance(edge, dict):
            edge = TaskEdge.set_data(edge)  # TaskEdge(**edge)
        if edge.relation[0] not in self.graph.vs["name"] or edge.relation[1] not in self.graph.vs["name"]:
            raise ValueError("Source or target task does not exist")

        source_id = self.graph.vs.find(name=edge.relation[0]).index
        target_id = self.graph.vs.find(name=edge.relation[1]).index
        rel_props = {**edge.__dict__, **kwargs}

        rel_props.pop("source", None)
        rel_props.pop("target", None)

        if self.graph.are_adjacent(source_id, target_id):  # 是否直接相连（有无一条边）
            eid = self.graph.get_eid(source_id, target_id)
            self.graph.es[eid].update_attributes(**rel_props)
        else:
            self.graph.add_edge(source_id, target_id, **rel_props)
        return rel_props

    async def set_edge(self, task_edge: TaskEdge, **kwargs):
        """
        向 Neo4j 添加或更新一条任务依赖边（任务之间的依赖关系）

        Args:
            task_edge (TaskEdge): 包含 source、target 和其他边属性
            kwargs: 可选附加属性（将合并进 edge 属性中）
        """
        rel_props = self.set_task_dependency(task_edge, **kwargs)
        if self.driver:
            async with self.driver.session() as session:
                await session.execute_write(merge_neo_relationship, task_edge.relation[0], task_edge.relation[1],
                                            rel_props, 'DEPENDS_ON', 'Task', 'Task')

    async def build_subgraph(self, graph_def: dict) -> None:
        """
        批量添加任务节点与依赖边
        """
        nodes: list[TaskNode] = [TaskNode(**nd) for nd in graph_def.get("nodes", [])]
        edges: list[TaskEdge] = [TaskEdge(**ed) for ed in graph_def.get("edges", [])]

        for node in nodes:
            await self.set_node(task_node=node)

        for edge in edges:
            await self.set_edge(task_edge=edge)

    @staticmethod
    def detect_cycles(g: ig.Graph) -> List[tuple[str]]:
        """检测图中的环"""
        cycles = g.feedback_arc_set(method="exact")
        if not cycles:
            return []
        # 将边索引转换为任务ID
        cycle_tasks = []
        for eid in cycles:
            edge: ig.Edge = g.es[eid]
            source = g.vs[edge.source]["name"]
            target = g.vs[edge.target]["name"]  # edge.target_vertex["name"]
            cycle_tasks.append((source, target))
        return cycle_tasks

    @staticmethod
    def graph_edge_to_task(edge: ig.Edge) -> TaskEdge:
        attrs = edge.attributes()
        if not attrs.get('relation'):
            attrs['relation'] = (edge.source_vertex["name"], edge.target_vertex["name"])
        return TaskEdge.set_data(attrs)

    @staticmethod
    def reroute_edges(g: ig.Graph, source_name: str, new_target_name: str):
        """
        将 source_node 的所有出边重新定向到 new_target。
        注意：只处理 name 属性标识的顶点。
        """
        source_id = g.vs.find(name=source_name).index
        new_target_id = g.vs.find(name=new_target_name).index

        # 收集所有出边
        out_edges = g.incident(source_id, mode="OUT")

        for eid in out_edges:
            edge = g.es[eid]
            old_target = edge.target
            g.delete_edges([(source_id, old_target)])  # 移除旧边
            g.add_edge(source_id, new_target_id, **edge.attributes())  # 添加新边（保留 edge 属性）

    @staticmethod
    def insert_between(g: ig.Graph, source_name: str, target_name: str, new_name: str, new_node_attrs=None):
        """
        在 source 和 target 之间插入一个新节点。
        - new_node_attrs: 可选，用于设置新节点的属性（字典）
        """
        source_id = g.vs.find(name=source_name).index
        target_id = g.vs.find(name=target_name).index

        g.delete_edges([(source_id, target_id)])  # 删除原边

        new_node_id = g.vcount()  # 添加新节点
        g.add_vertex(name=new_name, **(new_node_attrs or {}))

        # 添加两条新边
        g.add_edge(source_id, new_node_id)
        g.add_edge(new_node_id, target_id)

    def set_task_edges(self, edges: List[tuple], attributes: List[Dict[str, Any]]):
        # 批量设置边结构（源+目标+属性）
        assert len(edges) == len(attributes), "Edges and attributes must have the same length."
        for (source, target), attr in zip(edges, attributes):
            attrs = {k: v for k, v in attr.items() if k in TaskEdge.__annotations__}
            other_attrs = {k: v for k, v in attr.items() if k not in TaskEdge.__annotations__}
            edge = TaskEdge(relation=(source, target), **attrs)
            self.set_task_dependency(edge, **other_attrs)

    def export_adjacency_list(self, field_alignment=False) -> dict:
        adjacency_list = defaultdict(list)
        edge_attrs = self.graph.es.attributes()
        for edge in self.graph.es:
            source = self.graph.vs[edge.source]["name"]
            target = self.graph.vs[edge.target]["name"]

            adjacency_list[source].append({
                "name": target,
                "attrs": {attr: edge[attr] for attr in edge_attrs} if field_alignment
                else edge.attributes()  # 所有节点字段齐全
            })

        return dict(adjacency_list)

    def export_nodes(self, field_alignment=False) -> dict:
        node_attrs = self.graph.vs.attributes()
        return {v["name"]: {attr: v[attr] for attr in node_attrs} if field_alignment else v.attributes()
                for v in self.graph.vs}
        # [self.graph.vs[n]["name"] for n in self.graph.successors(v.index)]

    def _check_condition(self, edge: TaskEdge | ig.Edge | Dict[str, Any]) -> int:
        """检查边触发条件是否满足"""
        if isinstance(edge, ig.Edge):
            edge = self.graph_edge_to_task(edge)
        elif isinstance(edge, dict):
            edge = TaskEdge.set_data(edge)  # TaskEdge(**edge)

        source = self.graph.vs.find(name=edge.relation[0])
        target = self.graph.vs.find(name=edge.relation[1])

        # 基础状态检查
        if target["status"] != TaskStatus.PENDING:
            return 0

        # 条件类型处理,状态条件
        condition = edge.condition
        if condition is None:
            return 0  # return True

        if isinstance(condition, TaskStatus):
            condition = condition.value

        event_ready = ConditionFlag.NONE  # 0
        if isinstance(condition, str):
            source_status = source["status"]
            if isinstance(source_status, TaskStatus):
                source_status = source_status.value
            if source_status != condition:
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
            event_ready = ConditionFlag.STATUS_MATCH

        elif isinstance(condition, dict) and "deadline" in condition:
            if time.time() > condition["deadline"]:
                return 0
            event_ready = ConditionFlag.TIME_OK  # 时间条件<
        elif callable(condition):
            if not condition():
                return 0
            event_ready = ConditionFlag.CUSTOM_OK  # 自定义条件
        # 自定义规则
        if edge.rule:
            if not edge.rule():
                return 0

        return event_ready

    def check_trigger_tasks(self) -> List[str]:
        """检查并触发符合条件的任务"""
        triggered = []
        for edge in self.graph.es:  # 遍历全部边
            if self._check_condition(edge) <= 0:
                continue
            target = edge.target_vertex  # self.graph.vs[edge.target]
            target["status"] = TaskStatus.READY
            triggered.append(target["name"])
            print(f"Task {target['name']} triggered by {edge.source_vertex['name']} -> {target['name']}")

        return triggered

    @staticmethod
    def execute_task(task: ig.Vertex, action_registry: Optional[Dict[str, Callable]] = None):
        """执行单个任务（Dask延迟函数）"""
        attrs = task.attributes()
        exec_func = attrs.get('function', None)  # execute 预定义函数
        params = attrs.get("params", {}) or attrs.get("data", {})
        if callable(exec_func):
            return exec_func(**params)

        name = task["name"]
        action = attrs.get("action")
        if isinstance(action, str):
            if action_registry:
                exec_func = action_registry.get(action)
                if callable(exec_func):
                    return exec_func(**params)

            # 使用AI解析任务输入输出类型 tool_call
            parameters_prompt = """
                  根据任务描述确定输入和输出数据类型,根据以下信息生成方法参数
                  方法: {action} 任务: "{description}"

                  输出格式: 输入类型|输出类型
                  可用类型: {input_type} {output_type}
                  可用上下文: {parent_results}
                  """

            # ai_generate/generate_code Smart...learning strategy 动态调整 context global action expression
            decision_prompt = """
                    根据以下信息做出决策：
                    目标: {global}，当前状态: {state}
                    action:{action}，task desc:{description}
                    父节点结果: {parent_results}
                    决策选项: {options}

                    请分析并选择最佳选项，说明理由。
                    """
            decision_prompt = decision_prompt.format(action=action, description=attrs.get('description'))
            print(f"Simulating execution for {name}:{decision_prompt}")
            # context=...
            time.sleep(1)
        else:
            # reflect,optimize,analyze 反思优化任务，隐藏节点 or human interaction,调整工作流结构,知识整合
            reflection_prompt = """
                   你是一个高级AI工作流优化专家。请分析以下工作流执行情况：

                   工作流目标: {global}
                   执行历史摘要: {summarize_history} 
                   分析报告: {analyze}

                   请指出：
                   1. 工作流设计中的潜在问题
                   2. 可以优化的执行策略
                   3. 未来类似工作流的改进建议
                   4. 需要添加到知识库的关键洞察
                   """
            print(f"[SKIP] Task {name} has no executable logic.")
            return {"status": "skipped", "task": name}

        return True

    def build_dask_graph(self, action_registry: Optional[Dict[str, Callable]] = None) -> Dict[str, delayed]:
        """将 igraph 中的任务节点转换为 Dask 延迟任务图"""
        delayed_tasks = {}

        for v in self.graph.vs:
            task_id = v["name"]
            func = v["function"] or v["action"]
            if isinstance(func, str) and action_registry:
                func = action_registry.get(func)  # 从注册表中解析函数

            if not callable(func):
                continue  # 或 raise Error

            # 找到前置依赖任务名
            predecessors = self.graph.predecessors(v.index)
            dep_names = [self.graph.vs[p]["name"] for p in predecessors if self.graph.vs[p]["name"] in delayed_tasks]
            params = v.attributes().get("params", {}) or v.attributes().get("data", {})  # 静态+动态参数
            # 构建 delayed task，注入上游参数
            if dep_names:
                delayed_tasks[task_id] = delayed(func)(*[delayed_tasks[dep] for dep in dep_names], **params)
            else:
                delayed_tasks[task_id] = delayed(func)(**params)

        return delayed_tasks

    def execute_ready_tasks(self, executor: Callable[[Any], bool] = None) -> bool:
        """执行所有就绪任务"""
        # 查找状态为 ready 的任务并执行
        ready_tasks = [v for v in self.graph.vs if v["status"] == TaskStatus.READY]
        if not ready_tasks:
            return False  # 无可执行任务时退出

        exec_func = executor or self.execute_task
        for task in ready_tasks:
            print(f"Executing Ready task {task['name']}...")
            task["status"] = TaskStatus.IN_PROGRESS  # "running"
            try:
                result = exec_func(task)
                # 任务执行成功
                task['result'] = result  # 将结果映射回任务
                task["status"] = TaskStatus.COMPLETED if result else TaskStatus.FAILED  # "done"'completed'
            except Exception as e:
                task["status"] = TaskStatus.FAILED
                print(f"Task {task['name']} failed,error: {e}")

        return True

    def run_scheduler(self, max_cycles: int = 100, executor: Callable[[Any], bool] = None) -> None:
        if not self.graph.is_dag():
            cycles = self.detect_cycles(self.graph)
            raise ValueError(f"任务图存在循环依赖，无法调度，图中存在环 {cycles}")

        def step() -> bool:
            triggered = self.check_trigger_tasks()  # 检查并触发新的任务
            executed = self.execute_ready_tasks(executor)
            return triggered or executed

        """运行任务调度主循环"""
        if max_cycles:
            for cycle_num in range(max_cycles):
                if not step():
                    print(f"Scheduler finished after {cycle_num} cycles with no more tasks.")
                    break
                time.sleep(1)
        else:
            while True:
                if not step():
                    break
                time.sleep(1)

        print("Scheduler finished")

        # 检查依赖并触发任务

    def run_dask_schedule(self):
        """执行任务图"""
        graph_delayed_tasks = self.build_dask_graph()
        if not graph_delayed_tasks:
            return []
        results = self.client.compute(*graph_delayed_tasks.values(), sync=True)  # 同步等待执行完毕

        for task_id, result in zip(graph_delayed_tasks.keys(), results):
            try:
                vertex = self.graph.vs.find(name=task_id)
                vertex["result"] = result
                vertex["status"] = TaskStatus.COMPLETED  # 可选：设置任务状态
            except Exception as e:
                print(f"[Error] Failed to update result for task {task_id}: {e}")

        return results


def get_neighbors(g: ig.Graph, node: int | str | ig.Vertex, mode="ALL"):
    '''
    mode:
    - "OUT": 获取子任务（后继）g.successors(node.index),OUT:children
    - "IN": 获取父任务（前驱） g.predecessors(node.index),IN:parent
    - "ALL": 所有邻居
    '''
    if isinstance(node, str):
        node = g.vs.find(name=node)
    elif isinstance(node, int):
        node = g.vs[node]

    neighbors = g.neighbors(node.index, mode=mode)
    return [g.vs[n]["name"] for n in neighbors]


def find_root_nodes(g: ig.Graph) -> list[str]:
    """找到所有根节点（入度为0的节点）"""
    in_degrees = g.indegree()
    return [g.vs[i]["name"] for i, deg in enumerate(in_degrees) if deg == 0]


def calculate_depths(g: ig.Graph) -> Dict[str, int]:
    """计算每个节点的深度（拓扑深度），使用 BFS"""
    in_degrees = g.indegree()
    root_nodes = [(i, g.vs[i]["name"]) for i, deg in enumerate(in_degrees) if deg == 0]  # 返回[(index, name)]
    queue = [(v, 0) for v, root in root_nodes]  # deque
    depths = {}
    for v, depth in queue:
        task_id = g.vs[v]["name"]
        if task_id in depths and depths[task_id] >= depth:
            continue  # 已处理，且更深或一样深，不更新
        depths[task_id] = depth

        for succ in g.successors(v):  # 获取后继节点，略去局部判断
            queue.append((succ, depth + 1))

    return depths


def calculate_dependencies(g: ig.Graph) -> Dict[str, List[str]]:
    """计算每个节点的所有依赖（反向遍历所有节点祖先），使用 DFS"""
    dependencies = {}

    for v in range(g.vcount()):
        task_id = g.vs[v]["name"]
        visited = set()
        stack = list(g.predecessors(v))

        while stack:
            pred = stack.pop()
            if pred not in visited:
                visited.add(pred)
                stack.extend(g.predecessors(pred))

        dependencies[task_id] = [g.vs[u]["name"] for u in visited]

    return dependencies


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


def export_to_json_from_graph(graph: ig.Graph, filename):
    data = {"nodes": [{"id": v.index, **v.attributes()} for v in graph.vs],
            "edges": [{"source": e.source, "target": e.target, **e.attributes()} for e in graph.es],
            }
    with open(filename, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    # graph.write_pickle("graph.pkl")


def import_to_graph_from_json(filename) -> ig.Graph:
    with open(filename, "r") as f:
        data = json.load(f)
    if "nodes" not in data or "edges" not in data:
        raise ValueError("Invalid graph JSON format")

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
        # 初始化节点内容字典
        self.nodes: Dict[str, Dict[str, str]] = {}
        # 初始化邻接表
        self.adjacency_list: Dict[str, List[dict]] = defaultdict(list)
        self.task_queue = Queue()
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


# async def setr(key, value, ex=None):
#     redis = get_redis()
#     await redis.set(name=key, value=value, ex=ex)
#     # await redis.delete(key) redis.get(key, encoding='utf-8')

# pip install celery[redis]
# celery = Celery('tasks', broker='redis://localhost:6379/0') #Celery 任务
# message_queue = asyncio.Queue()


class MessageZeroMQ:

    def __init__(self, pull_port="7556", push_port="7557", req_port="7555", process_callback: Callable = None):
        self.context = zmq.asyncio.Context(io_threads=2)  # zmq.Context()

        # 设置接收消息的 socket
        self.pull_socket = self.context.socket(zmq.PULL)
        if pull_port:
            self.pull_socket.bind(f"tcp://*:{pull_port}")  # 绑定接收端口

        # 设置发送消息的 socket
        self.push_socket = self.context.socket(zmq.PUSH)
        if push_port:
            self.push_socket.connect(f"tcp://localhost:{push_port}")  # 连接到 Java 的接收端口

        # 设置 REQ socket 用于请求-响应模式
        self.req_socket = self.context.socket(zmq.REQ)  # zmq.DEALER
        if req_port:
            self.req_socket.connect(f"tcp://localhost:{req_port}")  # 连接到服务端

        self.process_callback = process_callback or self.default_process_message
        # self.push_socket.send_string('topic1 Hello, world!')

    def __del__(self):
        self.close()

    def close(self):
        try:
            self.pull_socket.close(linger=0)
            self.push_socket.close(linger=0)
            self.req_socket.close(linger=0)
            self.context.destroy(linger=0)
            # self.context.term()  # or destroy()
        except Exception as e:
            print(f"Error closing sockets: {e}")

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
        print(f"Received reply: {response}")
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


class TimeWheel:
    def __init__(self, slots: int, tick_duration: int | float, name: str):
        self.slots = slots
        self.tick_duration = tick_duration
        self.name = name
        self.current_pos: int = 0
        self.wheel = [[] for _ in range(slots)]
        self.next_wheel = None

        self._running = False
        self._task: asyncio.Task | None = None
        # self._timer = None

    def set_next_wheel(self, next_wheel: "TimeWheel"):
        self.next_wheel = next_wheel

    def add_task(self, delay: float | int, task, *args, **kwargs):
        ticks = int(delay // self.tick_duration)
        rounds = ticks // self.slots
        pos = int((self.current_pos + ticks) % self.slots)

        wrapped_task = partial(task, *args, **kwargs)
        self.wheel[pos].append((rounds, wrapped_task))

    @property
    def elapsed_time(self) -> float:
        """当前时间轮已推进的秒数"""
        return self.current_pos * self.tick_duration

    async def tick_once(self) -> bool:
        """
        执行一次 tick（供外部手动控制）
        如果触发了上层时间轮 tick，则返回 True，否则 False
        """
        tasks = self.wheel[self.current_pos]
        remaining_tasks = []

        for rounds, task in tasks:
            if rounds > 0:
                remaining_tasks.append((rounds - 1, task))
            else:  # 执行到期任务
                try:
                    if asyncio.iscoroutinefunction(task):
                        await task()
                    else:
                        await asyncio.to_thread(task)  # task() 支持普通函数
                except Exception as e:
                    print(f"[{self.name}] Task execution failed: {e}")

        self.wheel[self.current_pos] = remaining_tasks  # 更新轮槽
        self.current_pos = (self.current_pos + 1) % self.slots

        # 级联上层任务,如果转完一圈，通知上层时间轮
        if self.current_pos == 0 and self.next_wheel:
            self.cascade_tasks()  # 将上层时间轮的任务降级到本层
            return True  # 通知外部“我转完一圈”

        return False

    def cascade_tasks(self):
        """把上层时间轮当前槽的任务下放到本层"""
        if not self.next_wheel:
            return

        # 获取上层时间轮下一槽的所有任务
        next_pos = int((self.next_wheel.current_pos + 1) % self.next_wheel.slots)
        tasks = self.next_wheel.wheel[next_pos]

        # 将这些任务重新添加到本层时间轮
        for rounds, task in tasks:
            new_rounds = int(rounds * (self.next_wheel.slots // self.slots))  # 计算在新层的rounds
            delay = new_rounds * self.tick_duration
            self.add_task(delay, task)

        self.next_wheel.wheel[next_pos] = []  # 清空上层时间轮的这些任务

    async def run(self):
        """自动运行 tick 循环"""
        self._running = True
        print(f"[{self.name}] started (tick={self.tick_duration}s)")
        try:
            while self._running:
                await self.tick_once()
                await asyncio.sleep(self.tick_duration)
        except asyncio.CancelledError:
            pass
        finally:
            self._running = False
            print(f"[{self.name}] stopped")

    def start(self, loop=None):
        """启动自动循环"""
        if not self._running:
            loop = loop or asyncio.get_event_loop()
            self._task = loop.create_task(self.run())

        # def tick():
        #     self.tick_once()
        #
        #     # 继续下一次tick
        #     self._timer = threading.Timer(self.tick_duration, tick)
        #     self._timer.start()
        #
        # tick()

    def stop(self):
        """停止自动循环"""
        if self._task:
            self._running = False
            self._task.cancel()

        # if self._timer:
        #     self._timer.cancel()


class HierarchicalTimeWheel:
    def __init__(self):
        # 初始化各层时间轮
        self.second_wheel = TimeWheel(60, 1, "second")  # 60 slots, 1s per tick
        self.minute_wheel = TimeWheel(60, 60, "minute")  # 60 slots, 60s per tick
        self.hour_wheel = TimeWheel(24, 3600, "hour")  # 24 slots, 3600s per tick
        self.day_wheel = TimeWheel(30, 86400, "day")  # 30 slots, 86400s per tick

        # 连接各层时间轮
        self.second_wheel.set_next_wheel(self.minute_wheel)
        self.minute_wheel.set_next_wheel(self.hour_wheel)
        self.hour_wheel.set_next_wheel(self.day_wheel)

        self._loop = None

    @property
    def elapsed_time(self) -> float:
        # get_estimated_time
        total_seconds = (
                self.second_wheel.elapsed_time +
                self.minute_wheel.elapsed_time +
                self.hour_wheel.elapsed_time +
                self.day_wheel.elapsed_time
        )
        return total_seconds

    def add_task(self, execute_time: datetime | float, task, *args, **kwargs):
        """
        添加定时任务，可传入 datetime 或时间戳
        :param execute_time: 执行时间(datetime对象或时间戳)
        :param task: 要执行的任务(函数)
        """
        if isinstance(execute_time, datetime):
            execute_time = execute_time.timestamp()

        delay = max(0.0, execute_time - time.time())
        if delay <= 60:
            self.second_wheel.add_task(delay, task, *args, **kwargs)
        elif delay <= 3600:
            self.minute_wheel.add_task(delay, task, *args, **kwargs)
        elif delay <= 86400:
            self.hour_wheel.add_task(delay, task, *args, **kwargs)
        else:
            self.day_wheel.add_task(delay, task, *args, **kwargs)

    def add_daily_task(self, hour: int, minute: int, task, *args, **kwargs):
        """每天固定时刻执行任务"""
        now = datetime.now()
        run_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if run_time < now:
            run_time += timedelta(days=1)  # 今天的时间已过 → 明天执行

        def wrapper():
            if asyncio.iscoroutinefunction(task):  # 执行任务
                asyncio.run_coroutine_threadsafe(task(*args, **kwargs), self._loop)
            else:
                task(*args, **kwargs)
            self.add_daily_task(hour, minute, task, *args, **kwargs)  # 计算下一次运行时间并注册，重新注册下一次任务

        self.add_task(run_time, wrapper)

    async def tick(self, level: str = "second"):
        mapping = {
            "second": self.second_wheel,
            "minute": self.minute_wheel,
            "hour": self.hour_wheel,
            "day": self.day_wheel,
            "month": self.day_wheel,  # 暂时复用 day_wheel
        }
        wheel = mapping.get(level)
        if not wheel:
            print(f"[TimeWheel] Invalid level: {level}")
            return False
        return await wheel.tick_once()

    def start(self):
        if self._loop is not None:
            return
        try:
            loop = asyncio.get_running_loop()
            self._loop = loop
            self.second_wheel.start(loop)  # 已经在异步上下文中
        except RuntimeError:
            # 没有运行中的事件循环 → 创建一个新的 独立线程方式
            def loop_thread_background():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self._loop = loop
                self.second_wheel.start(loop)  # 创建任务必须在 loop 中
                loop.run_forever()

            threading.Thread(target=loop_thread_background, daemon=True).start()  # 在独立线程中运行事件循环
            while self._loop is None:
                time.sleep(0.1)

    def stop(self):
        self.second_wheel.stop()
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop = None


class AsyncCaller:
    """
    通过后台线程和队列将同步函数调用（如日志记录、网络请求）转换为异步执行,适合不关心返回值的场景
    通过将耗时操作转移到后台线程，主线程可以继续执行核心逻辑，提高程序响应速度。
    This AsyncCaller tries to make it easier to async call

    Currently, it is used in MLflowRecorder to make functions like `log_params` async

    NOTE:
    - This caller didn't consider the return value
    """

    STOP_MARK = "__STOP"

    def __init__(self) -> None:
        self._q = Queue()
        self._stop = False
        self._t = threading.Thread(target=self.run)
        self._t.start()

    def close(self):
        self._q.put(self.STOP_MARK)

    def run(self):
        while True:
            # NOTE:
            # atexit will only trigger when all the threads ended. So it may results in deadlock.
            # So the child-threading should actively watch the status of main threading to stop itself.
            main_thread = threading.main_thread()
            if not main_thread.is_alive():  # 检测主线程状态，避免程序退出时死锁
                break
            try:
                data = self._q.get(timeout=1)
            except Empty:
                # NOTE: avoid deadlock. make checking main thread possible
                continue
            if data == self.STOP_MARK:
                break
            data()

    def __call__(self, func, *args, **kwargs):
        self._q.put(partial(func, *args, **kwargs))

    def wait(self, close=True):
        if close:  # 资源释放,等待线程结束（可选是否自动关闭）必须显式调用 wait() 或 close()，否则线程可能无法退出
            self.close()
        self._t.join()

    @staticmethod
    def async_dec(ac_attr):
        """允许类通过 ac_attr 指定的钩子函数拦截方法调用。通过属性（ac_attr）动态决定是否拦截并包装某个方法的调用逻辑"""

        def decorator_func(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                hook = getattr(self, ac_attr, None)
                if isinstance(hook, Callable):
                    return hook(func, self, *args, **kwargs)

                return func(self, *args, **kwargs)

            return wrapper

        return decorator_func


class ThreadPoolTask:
    def __init__(self, max_thread_num, **pool_kwargs):
        self.max_thread_num = max_thread_num
        self.pool_kwargs = pool_kwargs  # thread_name_prefix

    def run(self, function, args_list, **kwargs):
        """执行线程池任务
       :param function: 被调用的函数
       :param args_list: iterable 可迭代的参数列表（如 [1, 2, 3]）
       :param kwargs: 传递给每个任务的 func_kwargs
        """
        bound_func = partial(function, **kwargs) if kwargs else function
        with ThreadPoolExecutor(max_workers=self.max_thread_num, **self.pool_kwargs) as executor:
            return list(executor.map(bound_func, args_list))


def run_thread_pool(max_thread_num=10, **pool_kwargs):
    # 自动用线程池处理每个item
    def wrapper(func):
        def inner(args_list, **kwargs):
            thread_pool = ThreadPoolTask(max_thread_num, **pool_kwargs)
            return thread_pool.run(func, args_list, **kwargs)  # 返回结果

        return inner

    return wrapper


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
        scheduler = TaskGraphManager()

        # 添加任务节点
        scheduler.set_task_node("task1", {"priority": 1})
        scheduler.set_task_node("task2", {"priority": 2})
        scheduler.set_task_node("task3", {"priority": 3})

        # 添加依赖关系
        scheduler.set_task_dependency(TaskEdge(relation=('task1', 'task2'),
                                               condition="done",
                                               trigger_time={"relative": 1}
                                               ))

        scheduler.set_task_dependency(TaskEdge(relation=('task2', 'task3'),
                                               condition="done",  # 自定义条件
                                               rule=lambda: time.localtime().tm_hour < 23  # 晚上11点前才触发
                                               ))

        # 启动任务
        scheduler.update_task_status("task1", TaskStatus.COMPLETED)  # "done"

        abs_time = time.time() + 5  # taskY 会延迟 5 秒后变为 ready 并执行
        scheduler.set_task_node("taskX", {"action": 'X'})
        scheduler.set_task_node("taskY", {"action": 'y'})

        scheduler.set_task_dependency(
            TaskEdge(relation=('taskX', 'taskY'), condition="done", trigger_time={"absolute": abs_time}))
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
        print(find_root_nodes(scheduler.graph), calculate_depths(scheduler.graph))

        print(get_neighbors(scheduler.graph, 0))

        print(scheduler.run_dask_schedule())

        async def test_save():
            await scheduler.set_node(task_node=TaskNode(name='taskX', action='X'))
            await scheduler.set_node(task_node=TaskNode(name='taskY', action='Y'))
            await scheduler.set_edge(task_edge=TaskEdge(relation=('taskX', 'taskY'), condition='done',
                                                        trigger_time={"absolute": abs_time}))

        asyncio.run(test_save())


    test()


    def print_time(msg):
        time.sleep(1)  # 模拟耗时操作
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


    caller = AsyncCaller()

    # 异步调用（立即返回）
    caller(print_time, "test1")
    caller(print_time, "test2")

    # 等待所有任务完成
    caller.wait()


    # async def print_time(msg):
    #     await asyncio.sleep(1)  # 模拟耗时操作
    #     print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

    async def main():
        tw = TimeWheel(60, 1, "second")
        tw.add_task(5, lambda: print_time("5秒后执行"))  # 5秒后执行
        tw.add_task(65, lambda: print_time("65秒后执行"))
        tw.start()

        await asyncio.sleep(75)  # 等待任务执行
        tw.stop()
        print("时间轮停止", tw.elapsed_time)


    asyncio.run(main())

    ht = HierarchicalTimeWheel()
    now = datetime.now()

    ht.add_task(now + timedelta(seconds=10), lambda: print_time("10秒后执行"))
    ht.add_task(now + timedelta(minutes=2), lambda: print_time("2分钟后执行"))
    ht.start()

    ht.add_task(time.time() + 5, lambda: print_time("5秒后"))
    ht.add_daily_task(18, 0, lambda: print_time("每天18:00"))

    try:
        input("按回车退出...\n")
        print(ht.elapsed_time)
    except KeyboardInterrupt:
        print('KeyboardInterrupt')
    finally:
        ht.stop()
