import os
import logging
from logging.handlers import RotatingFileHandler
from functools import partial, wraps
from typing import Callable, Type, Any, Awaitable, Coroutine
import asyncio


def get_root_logging(file_name="app.log", level: int = logging.WARNING,
                     _format: str = "%(asctime)s - %(levelname)s - %(message)s"):
    root_logger = logging.getLogger()  # logging.getLogger(__name__)
    root_logger.setLevel(level)

    # 如果已有 handler，就不重复添加
    if root_logger.handlers:
        return root_logger

    formatter = logging.Formatter(_format)
    _dir = os.path.dirname(file_name)
    if _dir:
        os.makedirs(_dir, exist_ok=True)
    # 文件日志logging.FileHandler('errors.log')
    file_handler = RotatingFileHandler(file_name, maxBytes=1_000_000, backupCount=3, encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    console_handler = logging.StreamHandler()  # 输出到终端,控制台输出
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    # logging.basicConfig(format=_format, level=level, encoding='utf-8',
    #                     handlers=[console_handler, file_handler])
    return root_logger


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
                logging.error(msg, exc_info=True)
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
                        logging.log(log_level, f"{msg} After {_max_retries} retries", exc_info=True)
                        raise  # 重试次数用尽后重新抛出异常

                    logging.log(log_level, f"{msg} Retrying {attempt}/{_max_retries} in {current_delay} seconds...")
                    await asyncio.sleep(current_delay)
                    current_delay *= _backoff  # 指数退避

        return wrapper

    return decorator


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


def async_polling_check(interval: float = 3, timeout: float = 300):
    """
    通用异步轮询任务装饰器, 如果返回 True 或非 False 非 None 的值，则表示完成

    参数:
        interval: 每次任务之间的间隔秒数
        timeout: 超时时间（秒）
    用法:
        @async_polling_check(interval=3, timeout=60)
        async def check_status(...):
            ...
            return True / False

        @async_polling_check(interval=2, timeout=10)
        async def poll_http_task(future,...)
            ...
            future.set_result
            return False

    handle, future = await poll_http_task()
        try:
            result = await future
        except TimeoutError:
           ...
        handle['cancel'] = True
    """

    def decorator(func: Callable[..., Coroutine[Any, Any, bool]]):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            handle = {"cancelled": False}  # 用于外部取消
            start = loop.time()  # 使用 loop.time 保障时间单调

            kwargs["_future"] = future
            kwargs["_handle"] = handle

            async def runner():
                nonlocal start
                while True:
                    if handle["cancelled"]:
                        if not future.done():
                            future.set_exception(Exception("Polling cancelled"))
                        break
                    if loop.time() - start > timeout > 0:
                        if not future.done():
                            future.set_exception(TimeoutError("Polling timeout"))
                        logging.warning(f"Polling task {func.__name__} timeout({timeout}s)")
                        break

                    try:
                        result = func(*args, **kwargs)  # , future=future 任务内部负责完成 future
                        if asyncio.iscoroutine(result):
                            result = await result

                        if interval <= 0 or result not in (False, None):  # 完成
                            if not future.done():  # 其他值直接返回
                                future.set_result(result)
                            break

                        await asyncio.sleep(interval)  # None/False:'in_progress'

                    except Exception as e:
                        logging.exception(f"Polling task {func.__name__} failed: {e}")
                        if not future.done():
                            future.set_exception(e)
                        break

            asyncio.create_task(runner())
            return future, handle

        return wrapper

    return decorator
