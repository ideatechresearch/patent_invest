from enum import Enum, IntFlag
import json, time, uuid
from typing import Dict, List, Union, Callable, Optional
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime, timedelta
import threading

from .base import *
from utils import pickle_deserialize, dataclass2dict, run_with_async, run_bound_tasks, merge_partial


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

    # completion_callbacks: Optional[ List[Callable]] = field(default_factory=list) # 回调函数

    def __post_init__(self):
        if not self.name:
            self.name = str(uuid.uuid4())
        if self.status is None:
            self.status = TaskStatus.PENDING
        if self.start_time is None:
            self.start_time = time.time()

    @staticmethod
    def default_node_attrs(task_id: str, **attributes) -> dict:
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
        info = asdict(self)
        info.pop("function", None)  # 移除函数
        payload = dataclass2dict(info)
        # payload.pop("data", None)
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

    def update_result(self, result, status: TaskStatus = TaskStatus.COMPLETED):
        self.result = result
        self.count = len(result) if isinstance(result, (list, tuple, set)) else 1 if result else 0
        self.end_time = time.time()
        self.status = status
        if self.status in (TaskStatus.COMPLETED, TaskStatus.RECEIVED):
            self.data = None
            self.progress = 100
        elif self.status == TaskStatus.FAILED:
            logging.error(f"Task {self.name} failed: {asdict(self)}")
        return self

    async def execute(self, *args, registry: dict = None, **kwargs):
        """执行单个任务（支持 function 或 registry action）"""
        exec_func = None
        if callable(self.function):
            exec_func = self.function
        elif isinstance(self.action, str) and registry is not None:
            candidate = registry.get(self.action)
            if callable(candidate):
                exec_func = candidate

        if exec_func is None:
            return None

        try:
            self.status = TaskStatus.IN_PROGRESS
            kwargs["_context"] = {"name": self.name, "description": self.description, "params": self.params,
                                  "data": self.data, "start_time": self.start_time}
            result = await run_with_async(exec_func, *args, **kwargs)
            self.update_result(result, TaskStatus.COMPLETED)
        except Exception as e:
            logging.error(f'function:{self.function.__name__} execute,params:{self.params},error:{e}')
            self.update_result(result={"error": str(e)}, status=TaskStatus.FAILED)

        return self


@dataclass
class TaskInfo(TaskNode):
    last_check: float = 0
    next_check: float = 0
    checked_count: int = 0
    error: Optional[str] = None
    # task_type: str = None  # 'api', 'llm', 'script'
    # created_at: float = 0
    # updated_at: float = 0


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


class PollingTask:
    def __init__(self, func: Callable, args=(), kwargs=None, interval=5, timeout=300):
        self.func = func
        self.args = args
        self.kwargs = kwargs or {}
        self.interval = interval
        self.timeout = timeout

        self.start_time = time.time()
        self.future = asyncio.get_event_loop().create_future()
        self.cancelled = False

    def is_timeout(self) -> bool:
        return 0 < self.timeout < (time.time() - self.start_time)

    async def check(self):
        """
          返回:
              True  -> 任务完成
              False -> 继续轮询
              None  -> 异常或取消
        """
        if self.cancelled:
            if not self.future.done():
                self.future.set_exception(Exception("Polling cancelled"))
            return None

        if self.is_timeout():
            if not self.future.done():
                self.future.set_exception(TimeoutError("Polling timeout"))
            return None

        try:
            result = self.func(*self.args, **self.kwargs)
            if asyncio.iscoroutine(result):
                result = await result

            if result is not False:
                if not self.future.done():
                    self.future.set_result(result)  # 任务完成
                return True

            return False  # 继续轮询

        except Exception as e:
            if not self.future.done():
                self.future.set_exception(e)
        return None

    async def worker(self):
        """执行单次任务并决定是否继续"""
        if self.cancelled or self.future.done():
            return None

        result = await self.check()
        if result is False and self.interval > 0:
            await asyncio.sleep(self.interval)  # continue
            return self  # next_task 延迟轮询，返回给队列
        return None


class TimeWheel:
    def __init__(self, slots: int, tick_duration: int | float, name: str = "tw", next_wheel: "TimeWheel" = None):
        assert slots > 0 and tick_duration > 0
        self.slots = slots
        self.tick_duration = tick_duration
        self.name = name
        self.current_pos: int = 0
        self.wheel = [[] for _ in range(slots)]
        self.next_wheel: Optional["TimeWheel"] = next_wheel

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._lock = threading.RLock()
        # self._timer = None

    def set_next_wheel(self, next_wheel: "TimeWheel"):
        self.next_wheel = next_wheel

    def add_task(self, delay: float | int, task: Callable, *args, **kwargs) -> int:
        if delay < 0:
            delay = 0.0

        remaining_delay = delay - self.wheel_span
        if remaining_delay > 0 and self.next_wheel:  # 超过本层 span，上放到上层轮
            return self.next_wheel.add_task(remaining_delay, task, *args, **kwargs)

        ticks = int(delay // self.tick_duration)
        rounds = ticks // self.slots  # 当前层内轮数
        pos = int((self.current_pos + ticks) % self.slots)

        delay_sec = self.slots_delay(rounds, pos)
        wait = delay - delay_sec  # extra_delay,精度补偿值

        wrapped_task = merge_partial(task, *args, **kwargs)  # bound_func
        with self._lock:
            self.wheel[pos].append((rounds, wait, wrapped_task))
        # print(f'{self.__class__.__name__}.{self.name}[{wrapped_task.func.__name__}]:'
        #       f'{delay:.1f}->{rounds},{pos},{wait:.1f}s')
        return delay_sec

    @property
    def elapsed_time(self) -> float:
        """当前时间轮已推进的秒数（相对本层）"""
        return self.current_pos * self.tick_duration

    @property
    def wheel_span(self) -> float:
        """当前时间轮总秒数（相对本层）一圈总秒数 circle_seconds"""
        return self.slots * self.tick_duration

    @property
    def task_count(self) -> int:
        return sum(len(slot) for slot in self.wheel)

    def slots_delay(self, rounds: int, pos: int) -> int:
        """
        延迟 = 当前轮数 * 本轮整圈时间 + 当前槽偏移
        pos: 目标槽的索引（绝对槽号 0..slots-1）
        rounds = ticks // self.slots
        offset_slots = ticks % self.slots
        """
        offset_slots = (pos - self.current_pos) % self.slots
        return rounds * self.slots * self.tick_duration + offset_slots * self.tick_duration

    def clear(self) -> list[tuple]:
        # 清空所有槽
        tasks = []
        for slot in self.wheel:
            tasks.extend(slot)
            slot.clear()
        self.current_pos = 0
        return tasks

    async def tick_once(self) -> bool:
        """
        执行一次 tick（供外部手动控制）
        如果触发了上层时间轮 tick，则返回 True，否则 False
        """
        tasks = self.wheel[self.current_pos]
        remaining_tasks = []
        exec_tasks = []

        for rounds, wait, task in tasks:
            if rounds > 0:
                remaining_tasks.append((rounds - 1, wait, task))
            else:
                exec_tasks.append(task)

        with self._lock:
            self.wheel[self.current_pos] = remaining_tasks  # 更新轮槽
            self.current_pos = (self.current_pos + 1) % self.slots

        if exec_tasks:  # 到期任务
            asyncio.create_task(run_bound_tasks(exec_tasks))

        # 级联上层任务,如果转完一圈，通知上层时间轮
        if self.current_pos == 0 and self.next_wheel:
            await self.next_wheel.tick_once()
            return True  # 通知外部“我转完一圈”

        return False

    def cascade_tasks(self, pos: int = None):
        """把上层时间轮当前槽的任务下放到本层"""
        if not self.next_wheel:
            return
        # next_pos = int((self.next_wheel.current_pos+1) % self.next_wheel.slots) #tick_once 已经做过 +1
        pos = int(pos % self.next_wheel.slots) if pos else self.next_wheel.current_pos
        with self.next_wheel._lock:
            tasks = self.next_wheel.wheel[pos]  # 获取上层时间轮下一槽的所有任务
            self.next_wheel.wheel[pos] = []  # 清空上层时间轮的这些任务
        # 将这些任务重新添加到本层时间轮
        span = self.next_wheel.wheel_span
        for rounds, wait, task in tasks:
            remaining_delay = rounds * span + wait
            self.add_task(remaining_delay, task)

    def move_to_upper(self):
        if not self.next_wheel:
            return
        if not self.task_count:
            return
        # upper.wheel[upper.current_pos].extend(self.clear())  # 清空所有槽
        for slot_index, slot in enumerate(self.wheel):
            for rounds, wait, task in slot:
                # 重新计算在上层轮子中的 rounds 和位置 剩余延迟 = 当前轮数 * 本轮整圈时间 + 当前槽偏移
                remaining_delay = self.slots_delay(rounds, slot_index) + wait
                self.next_wheel.add_task(remaining_delay, task)

            slot.clear()  # 清空本层

    async def run(self):
        """自动运行 tick 循环"""
        if self._running:
            return
        self._running = True
        print(f"[{self.name}] started (tick={self.tick_duration}s)")

        loop = asyncio.get_running_loop()
        next_tick = loop.time()
        try:
            while self._running:
                missed = max(1, int((loop.time() - next_tick) / self.tick_duration) + 1)
                for _ in range(missed):
                    finished_circle = await self.tick_once()
                    if finished_circle:
                        self.cascade_tasks()  # 将上层时间轮的任务降级到本层,当前槽（current_pos）tick 结束时才执行
                next_tick += missed * self.tick_duration
                sleep_time = max(0.0001, next_tick - loop.time())
                await asyncio.sleep(sleep_time)

        except asyncio.CancelledError:
            pass
        finally:
            self._running = False
            print(f"[{self.name}] stopped")

    def start(self, loop=None):
        """启动自动循环"""
        if self._running:
            return
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
            self._task = None

        # if self._timer:
        #     self._timer.cancel()


class HierarchicalTimeWheel:
    def __init__(self):
        # 初始化各层时间轮
        self.second_wheel = TimeWheel(60, 1, "second")  # 60 slots, 1s per tick
        self.minute_wheel = TimeWheel(60, 60, "minute")  # 60 slots, 60s per tick
        self.hour_wheel = TimeWheel(24, 3600, "hour")  # 24 slots, 3600s per tick
        self.day_wheel = TimeWheel(365, 86400, "day")  # 30 slots, 86400s per tick

        # 连接各层时间轮
        self.second_wheel.set_next_wheel(self.minute_wheel)
        self.minute_wheel.set_next_wheel(self.hour_wheel)
        self.hour_wheel.set_next_wheel(self.day_wheel)

        self.tick_time: datetime = datetime.now()
        self._loop = None

    @property
    def tick_time_str(self) -> str:
        return self.tick_time.strftime('%Y-%m-%d %H:%M:%S')

    def delay_sec(self, execute_time: datetime) -> float:
        return (execute_time - self.tick_time).total_seconds()

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

    @property
    def task_count(self) -> int:
        """统计所有层的任务总数"""
        total = (
                self.second_wheel.task_count +
                self.minute_wheel.task_count +
                self.hour_wheel.task_count +
                self.day_wheel.task_count
        )
        return total

    def add_task(self, delay: float | int, task: Callable, *args, **kwargs) -> int:
        """
        添加定时任务，可传入 datetime 或时间戳,有精度丢失问题
        :param delay: 执行时间(datetime对象或时间戳)
        :param task: 要执行的任务(函数)
        """
        if delay <= self.second_wheel.wheel_span:  # 60
            new_delay = self.second_wheel.add_task(delay, task, *args, **kwargs)
        elif delay <= self.minute_wheel.wheel_span:  # 3600
            new_delay = self.minute_wheel.add_task(delay, task, *args, **kwargs)
        elif delay <= self.hour_wheel.wheel_span:  # 86400
            new_delay = self.hour_wheel.add_task(delay, task, *args, **kwargs)
        else:
            new_delay = self.day_wheel.add_task(delay, task, *args, **kwargs)
        return new_delay

    def add_task_absolute(self, execute_time: datetime | float, task, *args, **kwargs):
        """
        添加定时任务，可传入 datetime 或时间戳
        :param execute_time: 执行时间(datetime对象或时间戳)
        :param task: 要执行的任务(函数)
        """
        if isinstance(execute_time, datetime):
            execute_time = execute_time.timestamp()

        delay = max(0.0, execute_time - time.time())
        self.add_task(delay, task, *args, **kwargs)

    def add_daily_task(self, hour: int, minute: int, task, *args, **kwargs):
        """每天固定时刻执行任务"""
        handle = {"cancel": False}

        async def wrapper():
            if handle["cancel"]:
                return

            await run_with_async(task, *args, **kwargs)
            # 注册下一次执行
            next_run = wrapper.next_run
            next_run += timedelta(days=1)
            print(f'daily task:{task.__name__} run_time={wrapper.next_run} next_run={next_run}')
            wrapper.next_run = next_run
            self.add_task_absolute(next_run, wrapper)

        now = datetime.now()
        run_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if run_time < now:
            run_time += timedelta(days=1)  # 今天的时间已过 → 明天执行
        wrapper.next_run = run_time
        self.add_task_absolute(run_time, wrapper)
        print(f'daily task:{task.__name__} run_time={run_time}')
        return handle

    def add_periodic_task(self, interval: int, task, *args, **kwargs):
        """周期任务"""
        handle = {"cancel": False}  # 用于外部取消 h = add_periodic_task,h["cancel"] = True 停止循环

        async def wrapper():
            if handle["cancel"]:
                return

            await run_with_async(task, *args, **kwargs)
            self.add_task(interval, wrapper)  # 重新注册

        self.add_task(interval, wrapper)
        return handle

    def add_polling_task(self, task, *args, interval: int, timeout: int, **kwargs):
        """
        轮询任务，支持超时和取消,
        async def check_status(...):
        future, polling = add_polling_task(task=check_status,...
        await tick
        try:
            result = await future
        except TimeoutError:
           ...
        finally:
            polling.cancelled = True
        """
        polling = PollingTask(task, args, kwargs, interval, timeout)

        async def wrapper():
            result = await polling.check()  # 任务内部负责完成 future
            if result is False:
                if not polling.future.done():  # 如果既不是完成也不是失败，wrapper 会在 interval 后重新调用任务
                    self.add_task(interval, wrapper)

        self.add_task(0, wrapper)  # 第一次注册轮询任务，interval<=0 直接执行一次
        return polling.future, polling

    async def tick(self, level: str = "second", tick_time=None) -> bool:
        mapping = {
            self.second_wheel.name: self.second_wheel,
            self.minute_wheel.name: self.minute_wheel,
            self.hour_wheel.name: self.hour_wheel,
            self.day_wheel.name: self.day_wheel
        }
        wheel = mapping.get(level)
        if not wheel:
            print(f"[TimeWheel] Invalid level: {level}")
            return False

        self.tick_time = tick_time or datetime.now()
        for name, w in mapping.items():
            if w is wheel:
                break  # 到达当前级别停止
            w.move_to_upper()

        finished = await wheel.tick_once()
        if finished and wheel.next_wheel:
            wheel.cascade_tasks()
        return finished

    @async_timer_cron(interval=1)
    async def tick_start(self):
        tick_time = datetime.now().replace(microsecond=0)
        await self.tick(level="second", tick_time=tick_time)  # 推动时间轮每秒执行一次

    def start(self, daemon=True):
        if self._loop is not None:
            return self._loop
        try:
            loop = asyncio.get_running_loop()
            self._loop = loop
            self.second_wheel.start(loop)  # 已经在异步上下文中
        except RuntimeError:
            # 没有运行中的事件循环 → 创建一个新的 独立线程方式
            ready = threading.Event()

            def loop_thread_background():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self._loop = loop
                self.second_wheel.start(loop)  # 创建任务必须在 loop 中
                # loop.create_task(self.second_wheel.run())
                ready.set()
                loop.run_forever()

            t = threading.Thread(target=loop_thread_background, daemon=daemon)  # 在独立线程中运行事件循环
            t.start()
            ready.wait()

        return self._loop

    def stop(self):
        self.second_wheel.stop()
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            print(f'[TimeWheel] stopped (elapsed={self.elapsed_time}s)')
        self._loop = None


class TaskManager:
    Task_map: dict[str, TaskNode] = {}  # queue.Queue(maxsize=Config.MAX_TASKS)
    htw = HierarchicalTimeWheel()  # htw.start()
    key_prefix = 'task'
    map_key = 'task_map'

    _lock = asyncio.Lock()

    def __len__(self):
        return len(type(self).Task_map)

    @classmethod
    def size(cls):
        return len(cls.Task_map)

    @classmethod
    async def commit(cls, task: TaskNode, redis_client=None) -> bool:
        if task is None:
            return False

        async with cls._lock:
            cls.Task_map[task.name] = task

        if redis_client:  # 把IO操作放到锁外
            await task.to_redis(redis_client, key_prefix=cls.key_prefix)
        return True

    @classmethod
    async def add_task(cls, task_id: str, task: TaskNode, redis_client=None, ex: int = 3600):
        if redis_client:
            await task.to_redis(redis_client, ex=ex, key_prefix=cls.key_prefix)
            # await add_set_to_redis(redis_client, f'{cls.key_prefix}_ids', task_id, ex=ex)
            await redis_client.hset(cls.map_key, task_id, int(task.start_time))

        async with cls._lock:
            cls.Task_map[task_id or task.name] = task

    @classmethod
    async def add(cls, redis=None, ex: int = 3600, **kwargs) -> tuple[str, TaskNode]:
        task_id = kwargs.pop('name', str(uuid.uuid4()))
        # task_fields = TaskNode.default_node_attrs(task_id, **kwargs)
        task = TaskNode(name=task_id, **kwargs)
        await cls.add_task(task_id, task, redis_client=redis, ex=ex)
        return task_id, task

    @classmethod
    def task_node(cls, name: Optional[str] = None, action: Optional[str] = None, params: Optional[dict] = None,
                  redis=None, ex: int = 3600, execute_time: datetime | float = 0,
                  description: Optional[str] = None, tags: Optional[list] = None, **extra_fields):
        """
        装饰器：将一个函数注册为 TaskNode,每次调用函数时生成独立 TaskNode 并注册
        返回 task_id, 结果由 run 后续收集
        """

        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                task_id = name or str(uuid.uuid4())
                node_params = {**(params or {}), **kwargs}
                task = TaskNode(
                    name=task_id,
                    description=description or func.__doc__,
                    function=func,
                    action=action or func.__name__,
                    tags=tags or [],
                    params=node_params,
                    **extra_fields
                )
                await cls.add_task(task_id, task, redis_client=redis, ex=ex)

                async def _run():
                    await cls.set_task_status(task, TaskStatus.IN_PROGRESS, redis_client=redis)
                    await task.execute(*args, **kwargs)
                    await cls.commit(task, redis_client=redis)
                    return task.result

                if execute_time:
                    cls.htw.add_task_absolute(execute_time, _run)
                else:
                    asyncio.create_task(_run())
                return task_id

            return wrapper

        return decorator

    @classmethod
    async def remove_task(cls, task_id: str, redis_client=None):
        async with cls._lock:
            cls.Task_map.pop(task_id, None)

        if redis_client:
            await redis_client.delete(f"{cls.key_prefix}:{task_id}")
            await redis_client.hdel(cls.map_key, task_id)

    @classmethod
    async def delete_tasks(cls, task_ids: list[str], redis_client=None) -> List[TaskNode]:
        """批量删除任务（包括Redis和内存）"""
        if not task_ids:
            return []

        deleted_tasks: List[TaskNode] = []
        async with cls._lock:
            for _id in task_ids:
                task = cls.Task_map.pop(_id, None)
                if task:
                    deleted_tasks.append(task)

        if redis_client:
            await redis_client.hdel(cls.map_key, *task_ids)
            # redis_keys = [f"{cls.key_prefix}:{_id}" for _id in task_ids]
            # if redis_keys:
            #     await redis_client.delete(*redis_keys)
        return deleted_tasks

    @classmethod
    async def get_task(cls, task_id: str, redis_client=None, ex: int = 0) -> TaskNode | None:
        async with cls._lock:
            task = cls.Task_map.get(task_id)
            if task is not None:
                return task

        if redis_client:
            task = await TaskNode.from_redis(redis_client, task_id, ex=ex, key_prefix=cls.key_prefix)
            if task:
                async with cls._lock:
                    cls.Task_map[task_id] = task
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
        task.status = status
        if progress > 0:
            task.progress = progress
        await cls.commit(task, redis_client=redis_client)

    @classmethod
    async def put_task_result(cls, task: TaskNode, result, total: int = -1, status: TaskStatus = None,
                              params: dict = None, redis_client=None) -> TaskNode:

        task.result.append(result)
        task.end_time = time.time()
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

        await cls.commit(task, redis_client=redis_client)
        return task

    @classmethod
    async def update_task_result(cls, task_id: str, result, status: TaskStatus = TaskStatus.COMPLETED,
                                 redis_client=None):
        task: TaskNode = await cls.get_task(task_id, redis_client)
        if not task:
            print(f"[update_task_result] Task {task_id} not found.")
            return None

        task = task.update_result(result, status)
        await cls.commit(task, redis_client=redis_client)
        return task.result

    @classmethod
    async def clean_old_tasks(cls, timeout_received=3600, timeout: int = 86400, redis_client=None):
        if not len(cls.Task_map):
            return

        current_time = time.time()
        task_ids_to_delete = []

        for _id, task in cls.Task_map.items():
            if task.end_time and (current_time - task.end_time) > timeout_received:
                if task.status == TaskStatus.RECEIVED:
                    task_ids_to_delete.append(_id)
                    print(f"Task {_id} has been marked for cleanup. Status: RECEIVED")
            elif (current_time - task.start_time) > timeout:
                task_ids_to_delete.append(_id)
                print(f"Task {_id} has been marked for cleanup. Timeout exceeded")

        if task_ids_to_delete:
            await cls.delete_tasks(task_ids_to_delete, redis_client=redis_client)
            print(f"[cleanup] Deleted {len(task_ids_to_delete)} tasks.")


class AsyncTaskQueue:
    def __init__(self, num_workers: int = 2):
        self.queue = asyncio.Queue()
        self.num_workers = num_workers
        self.worker_tasks: list[asyncio.Task] = []

        self._lock = asyncio.Lock()
        self._is_running = False

    async def _worker(self, worker_func: Callable, max_retries: int = 0, delay: float = 1.0, backoff: int | float = 2,
                      timeout: int | float = -1, **kwargs):
        """
        通用 worker
        :param worker_func:注入处理函数 async def f(task: tuple, **kwargs),返回值 true,false..
        :param max_retries,0不重试
        :param delay: 重试前延迟时间（秒）
        :param backoff:
        :param timeout: 超时时间（秒）
        :return:
        """
        import inspect  # 判断 func 是否接收 queue 参数
        accepts_queue = 'queue' in inspect.signature(worker_func).parameters

        while True:
            task_data = await self.queue.get()
            if task_data is None:  # 停止信号
                logging.info("[Info] Received shutdown signal")
                self.queue.task_done()
                break

            try:
                if isinstance(task_data, PollingTask):
                    next_task = await task_data.worker()
                    if next_task is not None:
                        await self.queue.put(next_task)
                elif worker_func is not None:
                    process_func = worker_func
                    # 如果需要重试，可用 retry 装饰
                    if max_retries > 0:
                        process_func = async_error_logger(max_retries=max_retries, delay=delay, backoff=backoff)(
                            worker_func)

                    if timeout > 0:
                        if accepts_queue:
                            success = await asyncio.wait_for(process_func(task_data, self.queue, **kwargs),
                                                             timeout=timeout)
                        else:
                            success = await asyncio.wait_for(process_func(task_data, **kwargs), timeout=timeout)
                    else:
                        if accepts_queue:
                            success = await process_func(task_data, self.queue, **kwargs)
                        else:
                            success = await process_func(task_data, **kwargs)

                    if not success:
                        logging.error(f"[Task Failed] {task_data}")

            except asyncio.TimeoutError:
                logging.error(f"[Timeout] Task {task_data} timed out")  # .put_nowait(x)
            except asyncio.CancelledError:
                logging.info("[Cancel] Worker shutting down...")
                break
            except Exception as e:
                logging.error(f"[Error] Unexpected error processing task: {e}")
            finally:
                self.queue.task_done()  # 确保每个任务只调用一次

    async def start(self, worker_func: Callable | None = None, **kwargs):
        """启动 worker"""
        if self._is_running:
            return self.worker_tasks
        async with self._lock:
            self._is_running = True
            logging.info(f"Starting {self.num_workers} workers")
            self.worker_tasks = [asyncio.create_task(self._worker(worker_func, **kwargs)) for _ in
                                 range(self.num_workers)]
            return self.worker_tasks

    async def stop(self):
        """优雅停止 worker"""
        if not self._is_running:
            return
        async with self._lock:
            self._is_running = False
            try:
                await self.queue.join()  # 等待任务处理完
                logging.info("All tasks processed. Stopping consumers...")
                for _ in self.worker_tasks:
                    await self.queue.put(None)  # 发送停止信号
            except Exception as e:
                logging.error(f"[Tasks Error] {e}, attempting to cancel workers...")
                for c in self.worker_tasks:
                    c.cancel()
            finally:
                await asyncio.gather(*self.worker_tasks, return_exceptions=True)
                self.worker_tasks.clear()

    async def resize_workers(self, new_num: int, **kwargs):
        """动态调整 worker 数量"""
        async with self._lock:
            delta = new_num - self.num_workers
            if delta == 0:
                return

            if delta > 0:
                for _ in range(delta):  # 扩容：新增 worker
                    task = asyncio.create_task(self._worker(**kwargs))
                    self.worker_tasks.append(task)
                logging.info(f"Added {delta} new workers (total={len(self.worker_tasks)})")

            else:
                for _ in range(abs(delta)):  # 缩容：发送停止信号给多余的 worker
                    await self.queue.put(None)
                logging.info(f"Removed {abs(delta)} workers (total={len(self.worker_tasks) + delta})")

            self.num_workers = new_num

    async def put_task(self, task: Any):
        """添加任务到队列"""
        await self.queue.put(task)

    async def add_polling_task(self, func: Callable, *args, interval: int | float = 3,
                               timeout: int | float = 300, **kwargs):
        """
        queue = AsyncTaskQueue(num_workers=2)
        await queue.start()  # 不需要 worker_func

        future, polling = await queue.add_polling_task(poll_http_task, "job-007", interval=2)
        result = await future
        """
        task = PollingTask(func, args, kwargs, interval, timeout)
        await self.queue.put(task)
        return task.future, task

    @classmethod
    def processor_worker(cls, max_retries: int = 0, delay: float = 1.0, backoff: int | float = 2,
                         timeout: int | float = -1):
        """
        无限流式任务（比如消息消费）
        任务处理装饰器，用于外部注入 worker 函数, 封装任务执行、重试和异常处理逻辑
        @AsyncTaskQueue.processor_worker(max_retries=3, timeout=10)
        async def handle_task(task: tuple):
            print(f"处理任务: {task}")
            if task == "error":
                queue.put_nowait(task_data)
                #raise ValueError("异常")
            return True
        queue = await handle_task(num_workers=2)
        await queue.put_task("task-2")
        await asyncio.sleep(2)
        await queue.stop()
        """

        def decorator(func: Callable):
            async def wrapper(*args, num_workers: int = 1, **kwargs):
                queue = cls(num_workers=num_workers)
                await queue.start(func, max_retries=max_retries, delay=delay, backoff=backoff, timeout=timeout,
                                  **kwargs)
                return queue

            return wrapper

        return decorator
