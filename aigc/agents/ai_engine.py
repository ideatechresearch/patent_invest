import asyncio
from typing import Dict, Any, Callable, TypeVar, ParamSpec
from enum import Enum
from functools import wraps
import inspect
import logging

P = ParamSpec('P')
T = TypeVar('T')


class EventType(Enum):
    TASK_START = "task_start"
    TASK_STOP = "task_stop"
    SYSTEM_ALERT = "system_alert"

    BEFORE_REQUEST = "before_request"
    AFTER_REQUEST = "after_request"
    TASK_COMPLETED = "task_completed"


class EventEngine:
    def __init__(self, log_name=None):
        self.event_queue = asyncio.Queue()
        self.handlers = {
            EventType.TASK_START: self._handle_task_start,
            EventType.TASK_STOP: self._handle_task_stop,
            EventType.SYSTEM_ALERT: self._handle_system_alert
        }
        self.running = False
        self.logger = logging.getLogger(log_name or __name__)

    async def start(self):
        """启动引擎"""
        self.running = True
        self.logger.info("Event engine started")
        while self.running:
            try:
                event = await self.event_queue.get()
                await self._process_event(event)
            except Exception as e:
                self.logger.error(f"Error processing event: {e}")

    async def stop(self):
        """停止引擎"""
        self.running = False
        self.logger.info("Event engine stopping")

    async def post_event(self, event_type: EventType, data: Dict[str, Any] = None):
        """发布事件"""
        event = {"type": event_type, "data": data or {}}
        await self.event_queue.put(event)
        self.logger.info(f"Event posted: {event_type}")

    async def _process_event(self, event: Dict):
        """处理事件"""
        event_type = event["type"]
        handler = self.handlers.get(event_type)
        if handler:
            try:
                await handler(event["data"])
            except Exception as e:
                self.logger.error(f"Error in handler for {event_type}: {e}")
        else:
            self.logger.warning(f"No handler for event type: {event_type}")

    async def _handle_task_start(self, data: Dict):
        self.logger.info(f"Handling TASK_START with data: {data}")
        # 实现任务启动逻辑

    async def _handle_task_stop(self, data: Dict):
        self.logger.info(f"Handling TASK_STOP with data: {data}")
        # 实现任务停止逻辑

    async def _handle_system_alert(self, data: Dict):
        self.logger.info(f"Handling SYSTEM_ALERT with data: {data}")
        # 实现系统警报处理逻辑


class EventDispatcher:
    def __init__(self):
        self._handlers = {}

    def register(self, event_type: EventType):
        """注册事件处理器的装饰器"""

        def decorator(handler: Callable):
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(handler)
            return handler

        return decorator

    async def dispatch(self, event_type: EventType, data: Dict[str, Any] = None):
        """分发事件"""
        data = data or {}
        if event_type in self._handlers:
            for handler in self._handlers[event_type]:
                try:
                    await handler(data)
                except Exception as e:
                    print(f"Error in event handler {handler.__name__}: {e}")


dispatcher = EventDispatcher()


class EventAware:
    """增强版事件感知装饰器"""

    def __init__(self, *event_types: EventType, data_extractor: Callable[..., Dict] = None):
        self.event_types = event_types
        self.data_extractor = data_extractor or (lambda *args, **kwargs: {})

    def __call__(self, func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # 触发前置事件
            for event_type in self.event_types:
                await dispatcher.dispatch(
                    event_type,
                    {"phase": "before", "data": self.data_extractor(*args, **kwargs)}
                )

            # 执行函数
            result = await func(*args, **kwargs)

            # 触发后置事件
            for event_type in self.event_types:
                await dispatcher.dispatch(
                    event_type,
                    {"phase": "after", "data": self.data_extractor(result)}
                )

            return result

        if inspect.iscoroutinefunction(func):
            return async_wrapper

        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # 同步函数的处理（略）
            pass

        return sync_wrapper
