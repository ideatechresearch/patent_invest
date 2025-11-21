import io, os, sys, socket
import re, json
import joblib
import subprocess
import platform
import inspect, importlib, ast
from functools import partial, wraps  # cache, lru_cache, partial, wraps
import gc  # 添加垃圾回收模块
import psutil  # 添加系统监控模块
import tracemalloc
from pathlib import Path
from collections import deque
from typing import Union, Iterable, Callable, AsyncGenerator, Any, get_origin, get_args
import aiofiles, asyncio
from concurrent.futures import ThreadPoolExecutor


def execute_code_results(text):
    """
    执行给定文本中的所有Python代码块，并返回每个代码块的输出、命名空间和错误信息。需要包含Python代码块的文本。
    仅用于已知安全的本地代码块执行。不要用于下载、安装或系统级操作。
    仅限本地已经有的或明确指出的，仅限本地环境、只读、无外部依赖的代码块。不支持 pip 安装、系统命令（如 cd、git、curl 等）。仅当已知代码可信且执行环境支持时调用。
    :param text:包含一个或多个 Python 代码块的纯文本内容。支持 Markdown 代码块风格的格式。
    :return:运行结果
    """
    from contextlib import redirect_stdout
    from .convert_ops import extract_python_code
    code_blocks = extract_python_code(text)
    results = []
    global_namespace = globals()  # 引用全局命名空间 {"__builtins__": dict(__builtins__)}
    for code in code_blocks:
        local_namespace = {}  # 用于存储代码的局部变量
        captured_output = io.StringIO()  # 用于捕获 `print` 输出
        try:
            with redirect_stdout(captured_output):  # 重定向 `print` 输出
                exec(code, global_namespace, local_namespace)
                # exec(code, globals=None, locals=None)用于动态执行较复杂的代码块,不返回结果,需要通过全局或局部变量获取结果
            output = captured_output.getvalue()  # 获取 `print` 的内容
            results.append({
                "output": output.strip(),
                "namespace": local_namespace,
                "error": None
            })
        except Exception as e:
            results.append({
                "output": captured_output.getvalue().strip(),
                "namespace": local_namespace,
                "error": f"Error executing code block: {e},Code:{code}"
            })
        finally:
            captured_output.close()

    return results


def safe_eval_call(func_name: str, func_args: dict):
    allowed_names = {
        **globals(), **locals(),
        "__builtins__": {"str": str, "int": int, "float": float, "list": list, "dict": dict}
    }
    if func_name not in allowed_names:
        raise ValueError(f"不允许动态调用函数：{func_name}")
    # compile(code, '<string>', 'eval')
    # eval(expression, globals=None, locals=None) 执行单个表达式,动态计算简单的表达式或从字符串中解析值
    return eval(f'{func_name}(**{func_args})', allowed_names)


def pip_installed_list():
    # from importlib.metadata import distributions
    # return sorted(
    #     [{"name": d.metadata["Name"], "version": d.version} for d in distributions()],
    #     key=lambda x: x["name"].lower()
    # )
    result = subprocess.run(
        [sys.executable, "-m", "pip", "list", "--format=json"],
        capture_output=True, text=True, check=True
    )
    return json.loads(result.stdout)


def pip_install(*packages, index_url="https://pypi.tuna.tsinghua.edu.cn/simple"):
    """
    安装 pip 包（支持空格/逗号分隔的多个包），并尝试 import。

    示例：
    pip_install("pandas")
    pip_install("pandas scikit-learn")
    pip_install("numpy", "matplotlib==3.7.1")
    """
    pkg_list = []
    for p in packages:
        # 拆分支持空格、逗号等格式
        if isinstance(p, str):
            parts = p.replace(",", " ").split()
            pkg_list.extend(parts)
        else:
            pkg_list.append(str(p))

    try:
        print(f"正在安装: {pkg_list}")
        subprocess.run([sys.executable, "-m", "pip", "install", "-i", index_url, *pkg_list],
                       # shell=(platform.system() == "Windows"),
                       check=True)
    except subprocess.CalledProcessError as e:
        print("❌ 安装失败:", str(e))
        return {"error": str(e), "installed": pkg_list}

    return pkg_list


def import_packages(pkg_list):
    success_imports = []
    for pkg in pkg_list:
        try:
            mod_name = pkg.split("==")[0].split(">")[0].split("<")[0]
            importlib.import_module(mod_name)
            success_imports.append(mod_name)
        except Exception as e:
            print(f"⚠️ 无法自动导入 {pkg}: {e}")
    return success_imports


def git_repo_clone(repo_url: str, revision: str = None, add_to_sys_path: bool = True) -> Path:
    # 自动化环境部署、动态加载 Git 仓库代码
    repo_path = Path(repo_url.split("/")[-1].replace(".git", ""))
    if not repo_path.exists():  # 本地没有该仓库
        try:
            subprocess.run(["git", "clone", repo_url], check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except Exception as exc:
            print(f"Failed to clone the repository: {exc.stderr}")
            raise

        if revision:  # 切换分支或提交
            subprocess.Popen(["git", "checkout", revision], cwd=str(repo_path))
    if add_to_sys_path and str(repo_path.resolve()) not in sys.path:
        sys.path.insert(0, str(repo_path.resolve()))
        # 加入 Python 的 sys.path，以便 import该仓库的模块

    return repo_path


def start_app_instance(port=7000, workers=4):
    return subprocess.Popen([
        "uvicorn", "main:app",  # 启动 main.py 中的 app 实例
        "--host", "127.0.0.1",
        "--port", str(port),
        "--workers", str(workers)
    ])


def configure_event_loop():
    if platform.system() != "Windows":
        try:
            import uvloop
            uvloop.install()
            print("🚀 uvloop activated")
        except ImportError:
            print("⚠️ uvloop not available, using default event loop")
    else:
        # 启用IOCP
        if sys.platform.startswith('win'):
            if sys.version_info >= (3, 8):
                # Python 3.8+ 使用更高效的 Proactor
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            else:
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # 使用 Selector
        print("🖥️ Windows detected, using optimized event loop policy")


def kill_process_tree(pid: int):
    import signal, contextlib
    """
    Kills all descendant processes of the given pid by sending SIGKILL.

    Args:
        pid (int): Process ID of the parent process
    """
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    # Get all children recursively
    children = parent.children(recursive=True)

    # Send SIGKILL to all children first
    for child in children:
        with contextlib.suppress(ProcessLookupError):
            os.kill(child.pid, signal.SIGKILL)

    # Finally kill the parent
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGKILL)


def chunks_iterable(lst, n: int):
    """将大数据分成小块"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def is_iterable(obj):
    try:
        iter(obj)
        return not isinstance(obj, (str, bytes))
    except TypeError:
        return False


def named_partial(name, func, *args, **kwargs):
    p = partial(func, *args, **kwargs)
    p.__name__ = name
    return p


def merge_partial(task, *args, **kwargs):
    if isinstance(task, partial):
        return partial(task.func, *(task.args + args), **{**task.keywords, **kwargs})
    return partial(task, *args, **kwargs)


def is_empty_lambda(func):
    try:
        if callable(func) and getattr(func, "__name__", "") == "<lambda>" and len(
                inspect.signature(func).parameters) == 0:
            return func() == []
    except:
        return False
    return False


def async_to_sync(func: Callable, *args, **kwargs):
    '''同步环境中调用异步函数，并且等待结果返回'''
    coro = func(*args, **kwargs)
    if not inspect.iscoroutine(coro):
        raise TypeError("func must be an async function")
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # 没在事件循环里 → 创建新循环执行

    if loop.is_running():
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()  # 提交到已有循环，阻塞等待结果

    return loop.run_until_complete(coro)


def run_by_future(func: Callable, *args, **kwargs):
    """
    不等待结果，不阻塞调用者 fire-and-forget
    """
    coro = func(*args, **kwargs)
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(coro)
        # if loop.is_running():
        #    task = asyncio.ensure_future(coro)  # 已有事件循环.run_coroutine_threadsafe
        # while not task.done():
        #     time.sleep(0.05)  # 阻塞直到完成
        # return task.result()
    except RuntimeError:
        asyncio.run(coro)  # 没有事件循环 → 创建新循环执行


async def run_with_async(func: Callable, *args, **kwargs):
    """
    通用方法：根据函数是否为协程自动选择 await 或直接调用
    支持在异步上下文中统一处理同步/异步函数调用
    # 同步代码转换为异步执行,在后台独立线程中运行,以避免阻塞主事件循环,使用 await 来执行
    # await asyncio.get_event_loop().run_in_executor(None, func, *args)
    :param func: 待执行函数（同步或异步）
    :param args: 位置参数
    :param kwargs: 关键字参数
    :return: 函数返回结果
    """
    if inspect.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    else:
        return await asyncio.to_thread(func, *args, **kwargs)  # 用 asyncio.to_thread 以避免阻塞, 等价于传统的 run_in_executor


async def run_bound_tasks(bound_funcs: list):
    """
    并发执行任务（支持协程函数与普通函数）
    """
    if not bound_funcs:
        return []

    coro_tasks = []  # 生成 coroutine
    for t in bound_funcs:
        if asyncio.iscoroutinefunction(t.func):
            coro_tasks.append(t())
        else:  # task() 支持普通函数
            coro_tasks.append(asyncio.to_thread(t))  # loop.run_in_executor(None, t))

    results = await asyncio.gather(*coro_tasks, return_exceptions=True)
    for t, res in zip(bound_funcs, results):
        if isinstance(res, Exception):
            print(f"Task {t.func.__name__} execution failed with error: {res}")
    return results


async def run_with_executor(func, args_list: list[tuple], max_workers: int = 10):
    """
    异步线程池执行多个任务
    :param func: 要执行的同步函数
    :param args_list: 参数列表，每项是一个 tuple（传给 func）
    :param max_workers: 最大线程数
    """
    loop = asyncio.get_running_loop()
    if len(args_list) == 1:
        return await loop.run_in_executor(None, func, *args_list[0])  # to_thread
    # 多任务并发执行
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        tasks = [loop.run_in_executor(pool, func, *args) for args in args_list]
        return await asyncio.gather(*tasks, return_exceptions=True)


async def run_with_semaphore(func, *args, semaphore=None, **kwargs):
    if semaphore:
        async with semaphore:
            return await func(*args, **kwargs)
    else:
        return await func(*args, **kwargs)


def run_togather(max_concurrent: int = 100, batch_size: int = -1, input_key: str = None, return_exceptions=True):
    """
    concurrency 并发限制装饰器，支持批量调用控制并发数量
    参数：
        max_concurrent: 最大并发数，默认 100；若 <=0 则不限制。
        batch_size: 批量大小，默认 -1 表示不分批；若 >0 则按批调用函数。
        - 第一个参数为单个输入（如 str|list[str],list[tuple]），或用 inputs=... 关键字传入。
        input_key: 如果不为 None，则以关键字参数方式传递输入值，如 func(**{input_key: x})
    用法：
        @run_togather(max_concurrent=10, batch_size=16)
        async def create_embeddings(input: List[str]|str, ..., model='xxx'):
            ...
        process_func = run_togather(max_concurrent=100, input_key="input")(cls.generate_metadata)
        results = await process_func(inputs=func_list,**kwargs)
    """
    semaphore = asyncio.Semaphore(max_concurrent) if max_concurrent > 0 else None

    def decorator(func: Callable[..., Any]):
        @wraps(func)
        async def wrapper(*args, inputs: list = None, **kwargs):
            _args = list(args)
            if inputs is None and _args:
                inputs = _args.pop(0)  # 允许 positional 方式传 list
            if not isinstance(inputs, list):
                inputs = [inputs]

            async def run_semaphore(x):
                if input_key:
                    coro = func(*_args, **{input_key: x}, **kwargs)
                else:
                    coro = func(x, *_args, **kwargs)
                if semaphore:
                    async with semaphore:
                        return await coro
                return await coro

            if batch_size <= 0:
                # 所有 input 独立请求
                tasks = [run_semaphore(item) for item in inputs]
            else:
                tasks = [run_semaphore(inputs[i:i + batch_size]) for i in range(0, len(inputs), batch_size)]

            return await asyncio.gather(*tasks, return_exceptions=return_exceptions)  # 确保任务取消时不会引发异常，并发执行多个异步任务

        return wrapper

    return decorator


def run_repeat(max_concurrent: int = 100, repeat: int = 1, return_exceptions: bool = True):
    # 创建信号量控制并发,按完成顺序返回(index, result),并发生成器装饰器，AI多次生成
    semaphore = asyncio.Semaphore(max_concurrent) if max_concurrent > 0 else None

    def decorator(func: Callable[..., Any]):
        @wraps(func)
        async def async_generator(*args, **kwargs) -> AsyncGenerator[tuple[int, Any], None]:
            async def worker(index):
                try:
                    if semaphore:
                        async with semaphore:
                            return index, await func(*args, **kwargs)
                    return index, await func(*args, **kwargs)
                except Exception as e:
                    if return_exceptions:
                        return index, e  # isinstance(r, Exception)
                    raise

            tasks = [asyncio.create_task(worker(i)) for i in range(repeat)]
            # 使用 as_completed 按完成顺序处理结果
            for coro in asyncio.as_completed(tasks):
                yield await coro  # (index, result or exception)/ task.get_coro().cr_frame.f_locals['index']

        return async_generator

    return decorator


def run_generator(max_concurrent: int = 100, input_key: str = None, return_exceptions: bool = True):
    """
    并发、异常处理、索引映射都继承自 run_repeat
    并发生成器装饰器，将异步函数转换为按完成顺序生成结果的异步生成器
    async for idx, res in ai_generates(inputs=[...])
    参数:
        max_concurrent: 最大并发数
        input_key: 如果不为 None，则以关键字参数方式传递输入值
        return_exceptions: 是否将异常作为结果返回，而不是抛出

    返回:
        异步生成器，按完成顺序生成 (index, result)
    """

    def decorator(func: Callable[..., Any]):
        @wraps(func)
        async def async_generator(*args, inputs: list = None, **kwargs):
            _args = list(args)
            if inputs is None and _args:  # 获取输入列表
                inputs = _args.pop(0)
            if not isinstance(inputs, list):
                inputs = [inputs]

            semaphore = asyncio.Semaphore(max_concurrent) if max_concurrent > 0 else None

            async def wrapper(index: int):
                try:
                    if input_key:
                        coro = func(*_args, **{input_key: inputs[index]}, **kwargs)
                    else:
                        coro = func(inputs[index], *_args, **kwargs)
                    if semaphore:
                        async with semaphore:
                            return index, await coro
                    return index, await coro
                except Exception as e:
                    if return_exceptions:
                        return index, e
                    raise

            tasks = [asyncio.create_task(wrapper(i)) for i in range(len(inputs))]
            for coro in asyncio.as_completed(tasks):
                yield await coro  # (index, result 或 exception)

        return async_generator

    return decorator


def make_runner(func: Callable, max_concurrent: int = 10, stream: bool = True, input_key: str = None,
                return_exceptions=True, **kwargs):
    """
    根据 stream 参数，返回 run_generator 或 run_togather 包装后的函数
    """
    if stream:
        @run_generator(max_concurrent=max_concurrent, input_key=input_key, return_exceptions=return_exceptions)
        async def wrapped(item):
            return await func(item, **kwargs)
    else:
        @run_togather(max_concurrent=max_concurrent, batch_size=-1, input_key=input_key,
                      return_exceptions=return_exceptions)
        async def wrapped(data: tuple[int, Any]):
            idx, item = data
            return idx, await func(item, **kwargs)
    return wrapped


async def runner_togather_sample(params: list, func: Callable, max_concurrent: int = 10, stream=True,
                                 input_key: str = None, **kwargs):
    """
     通用并发执行入口
     :param params: 输入列表
     :param func: 单个元素处理函数 async def func(item, **kwargs)
     :param stream: True -> run_generator (流式输出), False -> run_togather (批量输出)
     :param max_concurrent: 最大并发数
     :param input_key
     """
    runner = make_runner(func, max_concurrent=max_concurrent, stream=stream, input_key=input_key, **kwargs)

    def wrap_result(i, r):
        """统一结果格式，区分 dict / Exception / 普通值"""
        if isinstance(r, dict):
            return {'id': i, **r}
        elif isinstance(r, Exception):
            return {'id': i, 'error': str(r)}
        else:
            return {'id': i, 'value': r}

    if stream:
        results = []
        async for i, r in runner(inputs=params):
            print(f"[stream] idx={i} res={r}")
            # yield wrap_result(i, r)
            results.append(wrap_result(i, r))
        return sorted(results, key=lambda x: x['id'])
    else:
        results = await runner(inputs=[(i, p) for i, p in enumerate(params)])  # 已经是 [(idx, result), ...] 保持输入顺
        return [wrap_result(i, r) for i, r in results]


def run_by_threads(max_concurrency: int = 10, repeat: int = 1):
    """
    通用多线程批处理装饰器
    :param max_concurrency: 并发线程数
    :param repeat: 每个任务重复次数
    """
    import threading
    from queue import Queue
    def decorator(func):
        @wraps(func)
        def wrapper(inputs: list, **kwargs):
            results = [None] * (len(inputs) * repeat)
            lock = threading.Lock()
            q = Queue()

            # 填充队列
            for i, x in enumerate(inputs):
                for j in range(repeat):
                    q.put((i, j, x))

            def worker():
                while True:
                    item = q.get()
                    if item is None:
                        break
                    _i, _j, _x = item
                    idx: int = _i * repeat + _j
                    try:
                        res = func(_x, **kwargs)
                        with lock:
                            results[idx] = res
                    except Exception as e:
                        with lock:
                            results[idx] = {"error": str(e), "task": item}
                    finally:
                        q.task_done()

            # 启动线程
            threads = []
            for _ in range(max_concurrency):
                t = threading.Thread(target=worker)
                t.start()
                threads.append(t)

            # 等待所有任务完成
            q.join()

            # 停止线程
            for _ in range(max_concurrency):
                q.put(None)
            for t in threads:
                t.join()

            return results

        return wrapper

    return decorator


def class_property(attr_name: str):
    """
    缓存装饰器，支持首次调用生成值并缓存到类属性。
    自定义缓存属性名，适合无参或固定参数的懒加载。
    用于类级懒加载（lazy load）型类属性。仅检查当前类。
    :param attr_name: 缓存属性名
    @class_property("cached_value")
    """

    def decorator(func) -> classmethod:
        def wrapper(cls, *args):
            if not hasattr(cls, attr_name):
                print(f"{attr_name} -> {cls.__name__}.{func.__name__}")
                setattr(cls, attr_name, func(cls, *args))
            return getattr(cls, attr_name)

        return classmethod(wrapper)

    return decorator


def chainable_method(func):
    """装饰器，使方法支持链式调用，保留显式返回值"""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        return self if result is None else result

    return wrapper


_StringLikeT = Union[bytes, str, memoryview]


def list_or_args_keys(keys: Union[_StringLikeT, Iterable[_StringLikeT]],
                      args: tuple[_StringLikeT, ...] = None) -> list[_StringLikeT]:
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


def get_function_parameters(func) -> list:
    """返回参数名列表"""
    return list(inspect.signature(func).parameters.keys())


def get_function_params(func) -> dict:
    """只取可传递的参数 （排除 *args）"""
    signature = inspect.signature(func)
    return {
        k: (v.default if v.default is not inspect.Parameter.empty else None)
        for k, v in signature.parameters.items()
        if v.kind in (v.POSITIONAL_OR_KEYWORD, v.KEYWORD_ONLY)
    }


def remove_function_decorators(func) -> str:
    """
    获取函数源码，去掉所有装饰器
    """
    source = inspect.getsource(func)  # 获取原始代码 function_code
    tree = ast.parse(source)  # 解析 AST

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):  # 找到函数定义
            node.decorator_list = []  # 清空装饰器列表

    return ast.unparse(tree)  # 重新转换成代码
    # import textwrap
    # textwrap.dedent(source)  # 处理缩进


def strip_kwargs_wrapper(func):
    """返回一个去掉 **kwargs 的包装函数"""
    sig = inspect.signature(func)
    allowed_params = []
    # any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    for p in sig.parameters.values():
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            continue
        if p.annotation == inspect._empty:
            allowed_params.append(p)
        elif isinstance(p.annotation, type) and p.annotation.__module__.startswith("httpx"):
            continue  # 跳过 httpx 等复杂类型
        else:
            allowed_params.append(p)
    # 创建新的函数签名
    new_sig = inspect.Signature(allowed_params)

    @wraps(func)
    def wrapper(*args, **kwargs):
        bound_args = new_sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        return func(*bound_args.args, **bound_args.kwargs)

    wrapper.__signature__ = new_sig  # 替换签名（仅用于工具注册，不影响实际调用）
    return wrapper


def python_type_to_json_schema_type(python_type):
    origin = get_origin(python_type) or python_type

    # 处理 Union/Optional
    if origin is Union:
        args = [a for a in get_args(python_type) if a is not type(None)]
        if len(args) == 1:
            return python_type_to_json_schema_type(args[0])
        else:
            return "string"  # fallback for complex Unions

    mapping = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }
    return mapping.get(origin, "string")


def extract_function_metadatas(code):
    # 用于解析Python函数并提取元数据,从源代码级别提取详细的代码信息
    # 使用AST来解析Python代码,将 Python 源代码（字符串形式）转换为抽象语法树（AST）
    tree = ast.parse(code)
    functions = []

    # 遍历AST节点，找到函数定义
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            function_name = node.name
            arguments = [arg.arg for arg in node.args.args]
            docstring = ast.get_docstring(node) or ""
            function_code = ast.unparse(node) if hasattr(ast, 'unparse') else ast.dump(node)
            # if generate_docstring and not docstring:
            #     docstring = generate_docstring(function_name, function_code)

            metadata = {
                'name': function_name,
                'args': arguments,
                'docstring': docstring,
                'code': function_code
            }
            functions.append(metadata)

    return functions


def extract_function_metadata(func) -> dict:
    # 获取函数的名称、参数以及docstring
    func_name = func.__name__
    # print(f"[tools] 未找到缓存: {func_name},生成 metadata")
    # 获取函数参数列表
    signature = inspect.signature(func)
    docstring = func.__doc__

    parameters = {}
    required_params = []

    for param_name, param in signature.parameters.items():
        param_type = param.annotation if param.annotation != inspect.Parameter.empty else str
        json_type = python_type_to_json_schema_type(param_type)
        # 如果有类型提示，推断参数类型str(param.annotation).replace("<class '", "").replace("'>", "")
        param_info = {
            "type": json_type,
            "description": f"Description of the {param_name} parameter."
        }

        if param.default == inspect.Parameter.empty:
            required_params.append(param_name)
        else:
            param_info["default"] = param.default
        parameters[param_name] = param_info

    # 初始metadata结构
    metadata = {
        "type": "function",
        "function": {
            "name": func_name,
            "description": docstring or "No description available.",
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required_params
            },
            # "x-url":工具的实际 API 地址
        },
        # "func": func,
    }
    return metadata


class ClassMethodRegistry:
    """类方法注册表 - 使用类变量"""
    _registry: dict[str, dict] = {}
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, target_class, exclude: list[str] = None):
        """注册一个类及其所有类方法"""
        if exclude is None:
            exclude = []

        class_name = target_class.__name__
        class_info = {
            "class": target_class,
            "doc": target_class.__doc__,
            "methods": {}
        }

        # 收集所有类方法
        for name in dir(target_class):
            if name.startswith('_') or name in exclude:
                continue
            attr = getattr(target_class, name)  #
            if callable(attr):
                func = attr.__func__ if isinstance(attr, classmethod) else attr
                sig = inspect.signature(attr)
                class_info["methods"][name] = {
                    "function": func,
                    "doc": func.__doc__,
                    "signature": sig,
                    "type": cls.determine_method_type(sig, attr, target_class.__dict__.get(name, None))
                }

        cls._registry[class_name] = class_info
        print(f"Registered {class_name} with [{len(class_info['methods'])}] methods.")
        # print(f"Registered {class_name} with methods: {list(cls._registry[class_name]['methods'].keys())}")
        return target_class

    @classmethod
    def call_method(cls, class_name: str, method_name: str, *args, **kwargs):
        """调用注册的类方法"""
        if class_name not in cls._registry:
            raise ValueError(f"Class '{class_name}' not registered")

        class_info = cls._registry[class_name]
        if method_name not in class_info["methods"]:
            raise ValueError(
                f"Method '{method_name}' not found in class '{class_name}'. Available: {list(class_info['methods'].keys())}")

        method_info = class_info["methods"][method_name]
        func = getattr(class_info["class"], method_name, method_info["function"])
        try:
            if method_info.get("type") == "instance":
                instance = class_info["class"]()  # 对于实例方法，创建实例后调用
                return func(instance, *args, **kwargs)
            return func(*args, **kwargs)  # 对于类方法和静态方法，直接调用
        except TypeError as e:
            raise TypeError(f"参数不匹配: {e}")

    @classmethod
    def list_classes(cls) -> dict[str, Any] | None:
        """列出所有注册的类"""
        return {
            name: {
                "doc": info["doc"],
                "methods": list(info["methods"].keys())
            }
            for name, info in cls._registry.items()
        }

    @classmethod
    def get_class_info(cls, class_name: str) -> dict[str, Any] | None:
        """获取类的详细信息"""
        if class_name not in cls._registry:
            raise ValueError(f"Class '{class_name}' not registered")

        info = cls._registry[class_name]
        method_details = {}

        for method_name, method_info in info["methods"].items():
            method_details[method_name] = {
                "doc": method_info["doc"],
                "type": method_info.get("type", cls.determine_method_type(method_info["signature"],
                                                                          method_info["function"])),
                "parameters": cls.get_parameters_info(method_info["signature"]),
            }

        return {
            "class": class_name,
            "doc": info["doc"],
            "methods": method_details
        }

    @staticmethod
    def get_parameters_info(signature) -> dict[str, Any]:
        """获取参数信息"""
        params = {}
        for param_name, param in signature.parameters.items():
            if param_name in ['cls', 'self']:  # 跳过cls,self参数
                continue
            params[param_name] = {
                "kind": str(param.kind),
                "default": param.default if param.default != param.empty else None,
                "annotation": str(param.annotation) if param.annotation != param.empty else "any"
            }
        return params

    @staticmethod
    def determine_method_type(signature, method_obj=None, raw_obj=None) -> str:
        """确定方法的类型"""
        # 优先从类定义本身（未绑定）中取
        if isinstance(raw_obj, classmethod):
            return "classmethod"
        if isinstance(raw_obj, staticmethod):
            return "staticmethod"

        params = list(signature.parameters.values())
        # 如果有参数，检查第一个参数名
        if params:
            first_param = params[0].name
            if first_param == 'cls':
                return "classmethod"
            elif first_param == 'self':
                return "instance"

        # 判断绑定对象（适用于 bound method）
        if method_obj is not None and hasattr(method_obj, "__self__"):
            bound_self = getattr(method_obj, "__self__")
            if isinstance(bound_self, type):
                return "classmethod"
            elif bound_self is not None:
                return "instance"

        # 默认认为是类方法（对于使用自定义@class_property装饰器的方法）
        return "classmethod"

    @classmethod
    def clear_registry(cls):
        """清空注册表（主要用于测试）"""
        cls._registry.clear()


def register_class(cls, exclude: list[str] = None):
    """装饰器，用于自动注册类"""
    return ClassMethodRegistry.register(cls, exclude)


def load_class_from_string(class_path: str, path=None):
    """按字符串路径动态加载类 path="/plugins" """
    path_in_sys = False
    if path:
        if path not in sys.path:
            path_in_sys = True
            sys.path.insert(0, path)  # 临时允许从自定义目录加载模块

    try:
        module_name, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return cls
    finally:
        if path and path_in_sys:
            sys.path.remove(path)  # 恢复原始导入环境


def get_module_functions(module_name: str = None):
    module = importlib.import_module(module_name) if module_name else inspect.getmodule(inspect.currentframe())
    module_name = module.__name__
    funcs = inspect.getmembers(module, inspect.isfunction)
    local_funcs = [(name, func) for name, func in funcs if func.__module__ == module_name]
    # for name, _func in funcs:
    #     print(f"Function: {name}")
    return local_funcs


def functions_registry(functions_list: list, safe_path=True, module_name: str = None) -> dict:
    """
    根据函数名称列表,创建全局函数注册表,或者指定模块中动态加载
    1. 从当前全局作用域查找函数名；
    2. 指定 module_name，批量从该模块加载；
    3. 使用 'module.path:func' 格式，单个动态加载。

    :param functions_list: 需要注册的函数名列表
    :param safe_path: 取消不检查是否可调用。
    :param module_name: 模块名称（字符串形式），适合从一个模块中匹配加载多个函数。
    :return: Dict[str, Callable[..., Any]]
    """
    if not safe_path:
        module = importlib.import_module(module_name) if module_name else globals()
        return {name: getattr(module, name)
                for name in functions_list
                if hasattr(module, name) and callable(getattr(module, name))}

    return functions_registry_safe(functions_list, module_name)


def functions_registry_safe(functions_list: list, module_name: str = None) -> dict:
    """
    根据函数名称列表,创建全局函数注册表,或者指定模块中动态加载,检查是否可调用
    1. 从当前全局作用域查找函数名；
    2. 指定 module_name，批量从该模块加载；
    3. 使用 'module.path:func' 格式，单个动态加载。优先匹配

    :param functions_list: 需要注册的函数名列表
    :param module_name: 模块名称（字符串形式），适合从一个模块中加载多个函数。
    :return: Dict[str, Callable[..., Any]]
    """
    # 如果没有模块名，回退到全局作用域
    try:
        module = importlib.import_module(module_name) if module_name else globals()
    except ModuleNotFoundError:
        print(f"Module '{module_name}' not found.")
        module = globals()

    registry = {}
    for name in functions_list:
        if name in registry:  # 避免重复覆盖，只注册第一个找到的
            continue
        try:
            if ":" in name:
                module_path, func_name = name.rsplit(":", 1)
                _module = importlib.import_module(module_path)
                func_obj = getattr(_module, func_name, None)
                name = func_name
            else:
                func_obj = getattr(module, name, None)

            if not callable(func_obj):
                raise ValueError(f"函数 {name} 不存在或不是可调用对象,未在当前作用域中找到,可能未导入或模块未指定。")

            registry[name] = func_obj

        except ModuleNotFoundError:
            print(f"Module '{module_name}','{name}' not found.")
            registry[name] = None
            # importlib.reload(module_path)

        except Exception as e:
            print(f"[⚠️] 加载函数失败: {name} → {type(e).__name__}: {e}")
            registry[name] = None

    return registry


def parse_tool_text(text):
    # 定义正则表达式模式来匹配 <tags>, <tool_call>, <content> 及其内容
    tags_pattern = r'<tags>(.*?)</tags>'
    tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
    content_pattern = r'<content>(.*?)</content>'
    # 使用正则表达式查找匹配的内容
    tags_match = re.search(tags_pattern, text, re.DOTALL)
    tool_call_match = re.search(tool_call_pattern, text, re.DOTALL)
    content_match = re.search(content_pattern, text, re.DOTALL)
    # 提取匹配的内容，如果没有匹配到则返回空字符串
    tags = tags_match.group(1).strip() if tags_match else ""
    tool_call = tool_call_match.group(1).strip() if tool_call_match else ""
    content = content_match.group(1).strip() if content_match else ""
    # 将提取的内容存储在字典中
    result = {
        "tags": tags,
        "tool_call": tool_call,
        "content": content
    }
    return result


def openai_tools_to_mcp(tools: list[dict]):
    def decorator(mcp_server):
        for tool in tools:
            if tool["type"] != "function":
                continue

            func_info = tool["function"]

            def make_handler(func_name, func_desc):
                async def handler(params):
                    # 这里可以添加通用处理逻辑
                    print(f"Handling {func_name} with params: {params}")
                    # 实际处理函数应该在别处定义
                    return await globals()[f"handle_{func_name}"](params)

                handler.__name__ = func_name
                handler.__doc__ = func_desc
                return handler

            mcp_server.add_handler(
                action=func_info["name"],
                handler=make_handler(func_info["name"], func_info["description"])
            )
        return mcp_server

    return decorator


def deduplicate_functions_by_name(tools: list[dict]) -> list[dict]:
    seen = set()
    tools_metadata = []
    for tool in tools:
        name = tool.get("function", {}).get("name") or tool.get("name")
        if name and name not in seen:
            seen.add(name)
            tools_metadata.append(tool)
    return tools_metadata


def get_tools_params(tools: list[dict]) -> list[tuple[str, dict]]:
    return [(tool["type"], tool.get(tool["type"], {}))
            for tool in tools
            if isinstance(tool, dict) and tool.get("type") and tool["type"] != "function"]


def description_tools(tools: list[dict]) -> list[dict]:
    return [{'name': tool.get("function", {}).get("name"),
             'description': tool.get("function", {}).get("description")}
            for tool in tools if isinstance(tool, dict)]


def filter_directory(base_path: str, config: dict) -> list:
    """
    根据配置过滤目录，返回匹配的文件路径列表（相对于 base_path）

    config 字段说明：
      - include: list of glob patterns to include at root level
      - recursive_include: dict of {dir_name: [patterns]}，表示在指定子目录下递归匹配模式
      - prune: list of directory names to完全跳过
      - exclude: list of glob patterns to排除的文件或路径（相对于 base）
      - global_exclude: list of glob patterns to 排除的文件或目录（相对于 base, 作用于任意层级）
    """
    import fnmatch
    base = Path(base_path)
    matched = set()

    include = config.get("include", [])
    recursive_include = config.get("recursive_include", {})
    prune = set(config.get("prune", []))
    exclude = config.get("exclude", [])
    global_exclude = config.get("global_exclude", [])

    # Helper to check global exclude
    def is_glob_excluded(rel_path: Path) -> bool:
        for pattern in global_exclude:
            if fnmatch.fnmatch(str(rel_path), pattern):
                return True
        return False

    # 1. Root include: only files in base directory matching include patterns
    for pattern in include:
        for p in base.glob(pattern):
            rel = p.relative_to(base)
            if p.is_file() and not is_glob_excluded(rel):
                matched.add(str(rel))

    # 2. Recursive include: for each specified directory, walk under it
    for dir_name, patterns in recursive_include.items():
        dir_path = base / dir_name
        if not dir_path.exists() or not dir_path.is_dir():
            continue
        for root, dirs, files in os.walk(dir_path):
            # Prune dirs in-place
            dirs[:] = [d for d in dirs if d not in prune]
            for fname in files:
                rel = Path(root).relative_to(base) / fname
                # Check global exclude
                if is_glob_excluded(rel):
                    continue
                # Check exclude list
                skip = False
                for ex in exclude:
                    if fnmatch.fnmatch(str(rel), ex):
                        skip = True
                        break
                if skip:
                    continue
                # Check patterns
                for pat in patterns:
                    if fnmatch.fnmatch(fname, pat):
                        matched.add(str(rel))
                        break

    # Convert to sorted list
    return sorted(matched)


def get_file_info(directory):
    """
    获取指定目录下的文件信息，包括名称、类型、大小、创建时间、修改时间，并计算大小（以MB为单位）。
    get_file_info("E://Documents//Jupyter")
    """
    import pandas as pd
    file_info_list = []
    for root, dirs, files in os.walk(directory):
        # 先获取文件夹的信息
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            # 检查是否为符号链接
            if os.path.islink(dir_path):
                link_target = os.readlink(dir_path)  # 获取符号链接指向的目标
                if os.path.exists(link_target):  # 检查目标路径是否存在
                    # 遍历符号链接目标的内容
                    for sub_root, sub_dirs, sub_files in os.walk(link_target):
                        for sub_file in sub_files:
                            sub_file_path = os.path.join(sub_root, sub_file)
                            sub_file_stat = os.stat(sub_file_path)
                            file_info_list.append({
                                "Name": sub_file,
                                "Type": os.path.splitext(sub_file)[1],  # 文件扩展名
                                "Size (Bytes)": sub_file_stat.st_size,  # 文件大小（字节）
                                "Size (MB)": round(sub_file_stat.st_size / (1024 * 1024), 2),  # 文件大小（MB）
                                "Creation Time": pd.to_datetime(sub_file_stat.st_ctime, unit='s'),
                                "Modification Time": pd.to_datetime(sub_file_stat.st_mtime, unit='s'),
                                "Path": sub_file_path,
                                "Link Target": link_target  # 记录符号链接目标路径
                            })
            else:
                # 如果是普通文件夹
                dir_stat = os.stat(dir_path)
                file_info_list.append({
                    "Name": dir_name,
                    "Type": "Directory",  # 标记为文件夹
                    "Size (Bytes)": 0,  # 文件夹大小设为0
                    "Size (MB)": 0,  # 文件夹大小设为0
                    "Creation Time": pd.to_datetime(dir_stat.st_ctime, unit='s'),
                    "Modification Time": pd.to_datetime(dir_stat.st_mtime, unit='s'),
                    "Path": dir_path,
                    "Link Target": None  # 没有符号链接目标
                })

        # 获取文件的信息
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.islink(file_path):  # 如果是符号链接
                link_target = os.readlink(file_path)  # 获取符号链接目标
                if os.path.exists(link_target):  # 检查符号链接目标是否存在
                    file_stat = os.stat(link_target)  # 获取目标文件的状态信息
                    file_info_list.append({
                        "Name": file,
                        "Type": os.path.splitext(file)[1],
                        "Size (Bytes)": file_stat.st_size,
                        "Size (MB)": round(file_stat.st_size / (1024 * 1024), 2),
                        "Creation Time": pd.to_datetime(file_stat.st_ctime, unit='s'),
                        "Modification Time": pd.to_datetime(file_stat.st_mtime, unit='s'),
                        "Path": file_path,
                        "Link Target": link_target  # 记录符号链接目标路径
                    })
            else:  # 普通文件
                file_stat = os.stat(file_path)
                file_info_list.append({
                    "Name": file,
                    "Type": os.path.splitext(file)[1],
                    "Size (Bytes)": file_stat.st_size,
                    "Size (MB)": round(file_stat.st_size / (1024 * 1024), 2),
                    "Creation Time": pd.to_datetime(file_stat.st_ctime, unit='s'),
                    "Modification Time": pd.to_datetime(file_stat.st_mtime, unit='s'),
                    "Path": file_path,
                    "Link Target": None
                })

    return pd.DataFrame(file_info_list)


def batch_convert_encoding(input_dir, output_dir, from_encoding='gb2312', to_encoding='utf-8'):
    """
    批量将指定目录中的文件从一种编码转换为另一种编码。

    :param input_dir: 输入文件夹路径
    :param output_dir: 输出文件夹路径
    :param from_encoding: 原始文件编码（默认 gb2312）
    :param to_encoding: 转换后的文件编码（默认 utf-8）
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            input_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(input_file_path, input_dir)
            output_file_path = os.path.join(output_dir, relative_path)

            try:
                # 创建输出目录
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

                # 转换文件编码
                with open(input_file_path, 'r', encoding=from_encoding) as f_in:
                    content = f_in.read()
                with open(output_file_path, 'w', encoding=to_encoding) as f_out:
                    f_out.write(content)

                print(f"转换成功: {input_file_path} -> {output_file_path}")
            except Exception as e:
                print(f"转换失败: {input_file_path} 错误信息: {e}")


def get_memory_info():
    """从 /proc/self/status 获取内存使用（Linux Only）"""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    mem_kb = int(line.split()[1])
                    break
            return mem_kb / 1024  # 转为 MB
    except Exception:
        # psutil.virtual_memory().total / (1024 ** 3)
        return -1


def get_cpu_time():
    """从 /proc/self/stat 获取 CPU 时间（单位：秒）"""
    try:
        with open("/proc/self/stat") as f:
            values = f.read().split()
            utime = int(values[13])
            stime = int(values[14])
            ticks_per_sec = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
            return (utime + stime) / ticks_per_sec
    except Exception:
        return -1


def get_open_fds_count() -> int:
    try:
        base = f"/proc/{os.getpid()}/fd"
        return len(os.listdir(base))
    except Exception:
        return -1


def count_http_connections(port=8000):
    if platform.system() != "Linux":
        # print("Warning: count_http_connections is only supported on Linux.")
        return -1
    count = 0
    with open("/proc/net/tcp", "r") as f:
        lines = f.readlines()[1:]
        for line in lines:
            parts = line.strip().split()
            local_address = parts[1]
            local_port_hex = local_address.split(":")[1]
            if int(local_port_hex, 16) == port:
                count += 1
    return count


def is_port_open(host, port):
    try:
        with socket.create_connection((host, port), timeout=3):
            return True
    except (OSError, ConnectionRefusedError):
        return False


def get_worker_identity():
    """获取当前 Worker 的唯一标识,使用进程 ID + 主机名 + 启动时间"""
    worker_id = os.environ.get("GUNICORN_WORKER_ID", None)  # 使用 Gunicorn 环境变量（如果使用 Gunicorn）
    if worker_id is not None:
        return worker_id
    pid = os.getpid()
    hostname = socket.gethostname()
    # start_time = os.times().elapsed  # 进程启动后的时间
    worker_info = f"{pid}-{hostname}"
    # unique_id = hashlib.sha256(worker_info.encode()).hexdigest()[:16]  # 使用唯一哈希标识
    return f"worker-{worker_info}"


def memory_monitor(threshold_percent: float = 60, desc: bool = False):
    """监控进程内存使用率，超出阈值则触发垃圾回收"""
    try:
        # 获取内存使用情况
        process = psutil.Process()
        memory_percent = process.memory_percent()
        memory_mb = process.memory_info().rss / 1024 / 1024
        if desc:
            print(f"[MemoryMonitor] CPU 使用率: {psutil.cpu_percent(interval=1)}%")
            print(f"[MemoryMonitor] 系统内存: {psutil.virtual_memory()}")
            if tracemalloc.is_tracing():
                snapshot = tracemalloc.take_snapshot()  # 获取内存分配统计
                top_stats = snapshot.statistics('lineno')[:10]
                for stat in top_stats[:10]:
                    print(stat)

        # 当内存使用超过阈值时触发垃圾回收
        if memory_percent > threshold_percent:  # 70% 内存使用率阈值
            print(f"[MemoryMonitor] 当前内存使用率: {memory_percent:.2f}%, RSS: {memory_mb:.2f}MB - 触发垃圾回收")
            gc.collect()  # 强制垃圾回收
            after = process.memory_info().rss / 1024 / 1024
            print(f"[MemoryMonitor] 垃圾回收后内存: {after:.2f}MB，释放：{memory_mb - after:.2f}MB")

    except Exception as e:
        print(f"[MemoryMonitor] error: {e}")


async def get_log_lines(log_path: str | Path = "app.log", lines: int = 100):
    log_file = Path(log_path)
    if not log_file.exists():
        return {"error": f"日志文件不存在: {log_file}"}

    try:
        lines_list = deque(maxlen=lines)
        async with aiofiles.open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            # f.readlines(),await f.read()
            async for line in f:
                lines_list.append(line.strip())

        return {"logs": list(lines_list), "count": len(lines_list)}

    except Exception as e:
        return {"error": f"读取日志失败: {str(e)}"}


def get_job_list(scheduler):  # AsyncIOScheduler
    job_list = [
        {'id': job.id,
         'name': job.name,
         'func': str(job.func),
         # 'args': job.args,
         # 'kwargs': job.kwargs,
         'next_run_time': job.next_run_time.isoformat() if job.next_run_time else None,
         'trigger': str(job.trigger)
         } for job in scheduler.get_jobs()
    ]
    return job_list


# 保存所有模型到文件
def save_models(models: dict, model_dir='data/models/'):
    """
    保存训练好的模型到指定路径
    :param models: 一个字典，包含所有需要保存的模型
    :param model_dir: 保存模型的路径
    """
    os.makedirs(model_dir, exist_ok=True)
    for model_name, model in models.items():
        if model is not None:
            joblib.dump(model, f'{model_dir}/{model_name}.pkl')
    print(f"模型已保存至 {model_dir}")


# 加载模型
def load_models(model_names: list, model_dir='data/models/'):
    """
    加载保存的模型
    :param model_names: 模型名称列表
    :param model_dir: 保存模型的路径
    :return: 加载的模型字典
    """
    models = {}
    for model_name in model_names:
        model_path = f'{model_dir}/{model_name}.pkl'
        if os.path.exists(model_path):
            models[model_name] = joblib.load(model_path)
    print(f"模型已从 {model_dir} 加载")
    return models


def save_dictjson(dictionary, file_path, encoding='utf-8', ascii=False):
    with open(file_path, 'w', encoding=encoding) as file:
        json.dump(dictionary, file, ensure_ascii=ascii)


def load_dictjson(file_path, encoding='utf-8'):
    with open(file_path, 'r', encoding=encoding) as file:
        dictionary = json.load(file)
    return dictionary


def load_datasets(path):
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            samples.append(json.loads(line.strip()))
    return samples


async def save_markdown(content: str, filename: str, folder="data/output"):
    """异步保存单个Markdown文件"""
    if not os.path.exists(folder):
        os.makedirs(folder)

    if not filename.endswith('.md'):
        filename += '.md'

    filepath = os.path.join(folder, filename)
    async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
        await f.write(content)

    return filepath


async def read_markdown(filename: str, folder="data/output") -> str | None:
    """异步读取Markdown文件并返回内容"""
    if not filename.endswith('.md'):
        filename += '.md'

    filepath = os.path.join(folder, filename)
    if not os.path.exists(filepath):
        return None

    async with aiofiles.open(filepath, 'r', encoding='utf-8') as f:
        content = await f.read()

    return content


if __name__ == "__main__":
    def main():
        with open("utils.py", 'r', encoding='utf-8') as file:
            code = file.read()
        functions = extract_function_metadatas(code)
        for func in functions:
            print(func)

        # funcs = get_module_functions('utils.utils')
        # print([i[0] for i in funcs])

        filtered_files = filter_directory(r"/", config={
            "include": ["*.py", "deploy.bat", "requirements.txt", "CI"],
            "recursive_include": {
                "agents": ["*"],
                "script": ["*"],
                "docker": ["*"],
                "data": ["*.yaml", "*.pkl"]
            },
            "prune": [".venv", ".git", "docker_build", "script/data"],
            "exclude": ["script/nlp.py", "script/data/*"],
            "global_exclude": ["__pycache__/*", "*.py[cod]", "*.log"]
        })
        print(json.dumps(filtered_files, indent=2, ensure_ascii=False))


    import random


    async def test(params: list, stream=True):
        @run_togather(max_concurrent=10, batch_size=-1)
        async def single(data: tuple[int, dict]):
            i, param = data
            await asyncio.sleep(10)
            print(i, param)
            result = {**param, 'id': i}
            return result

        @run_generator(max_concurrent=10, return_exceptions=True)
        async def single_item(param):
            await asyncio.sleep(random.randint(3, 7))

            return param

        if stream:
            raw_results = []
            async for idx, r in single_item(inputs=params):
                print(idx, r)
                r = {'id': idx, **r}
                # yield res
                raw_results.append(r)
            results = sorted(raw_results, key=lambda x: x['id'], reverse=True)
        else:
            results = await single(inputs=[(i, p) for i, p in enumerate(params)])

        return results


    async def single_item(param):
        await asyncio.sleep(random.randint(3, 7))

        return param


    res = asyncio.run(runner_togather_sample([{"agents": ["*"],
                                               "script": ["*"],
                                               "docker": ["*"],
                                               "data": ["*.yaml", "*.pkl"]},
                                              {"global_exclude": ["__pycache__/*", "*.py[cod]", "*.log"]}], single_item,
                                             stream=True))
    print(res)

    # main()

    p = partial(single_item)
    print(asyncio.iscoroutinefunction(p.func), asyncio.iscoroutinefunction(p))
    p2 = partial(main)
    print(asyncio.iscoroutinefunction(p2.func), asyncio.iscoroutinefunction(p2), asyncio.iscoroutine(p2))
