import io, os, sys, socket
import re, json
import pickle, joblib
import subprocess
import platform
import inspect, importlib, ast
from functools import partial, wraps  # cache, lru_cache, partial, wraps
import gc  # 添加垃圾回收模块
import psutil  # 添加系统监控模块
import tracemalloc
from pathlib import Path
from collections import deque
import base64, hashlib
from typing import Union, Iterable, get_origin, get_args
import aiofiles, asyncio


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


def named_partial(name, func, *args, **kwargs):
    p = partial(func, *args, **kwargs)
    p.__name__ = name
    return p


def is_empty_lambda(func):
    try:
        if callable(func) and getattr(func, "__name__", "") == "<lambda>" and len(
                inspect.signature(func).parameters) == 0:
            return func() == []
    except:
        return False
    return False


def async_to_sync(func, *args, **kwargs):
    # 异步代码转换为同步代码
    return asyncio.run(func(*args, **kwargs))


async def wrap_sync(func, *args, **kwargs):
    # 同步代码转换为异步执行,在后台独立线程中运行,以避免阻塞主事件循环,使用 await 来执行
    return await asyncio.to_thread(func, *args, **kwargs)


async def run_with_async(func, *args, **kwargs):
    """
    通用方法：根据函数是否为协程自动选择 await 或直接调用
    支持在异步上下文中统一处理同步/异步函数调用

    :param func: 待执行函数（同步或异步）
    :param args: 位置参数
    :param kwargs: 关键字参数
    :return: 函数返回结果
    """
    if inspect.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    else:
        return await asyncio.to_thread(func, *args, **kwargs)  # 用 asyncio.to_thread 以避免阻塞


async def run_with_semaphore(func, *args, semaphore=None, **kwargs):
    if semaphore:
        async with semaphore:
            return await func(*args, **kwargs)
    else:
        return await func(*args, **kwargs)


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


def get_function_parameters(func):
    signature = inspect.signature(func)
    parameters = signature.parameters
    return [param for param in parameters]


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


def get_module_functions(module_name: str = None):
    module = importlib.import_module(module_name) if module_name else inspect.getmodule(inspect.currentframe())
    module_name = module.__name__
    funcs = inspect.getmembers(module, inspect.isfunction)
    local_funcs = [(name, func) for name, func in funcs if func.__module__ == module_name]
    # for name, _func in funcs:
    #     print(f"Function: {name}")
    return local_funcs


def functions_registry(functions_list: list, safe_path=True, module_name: str | list = None) -> dict:
    """
    根据函数名称列表,创建全局函数注册表,或者指定模块中动态加载
    1. 从当前全局作用域查找函数名；
    2. 指定 module_name，批量从该模块加载；
    3. 使用 'module.path:func' 格式，单个动态加载。

    :param functions_list: 需要注册的函数名列表
    :param safe_path: 取消不检查是否可调用。
    :param module_name: 模块名称或者模块名称列表（字符串形式），适合从一个模块或多个模块中匹配加载多个函数。
    :return: Dict[str, Callable[..., Any]]
    """
    if not safe_path:
        module = importlib.import_module(module_name) if module_name else None
        return {name: getattr(module, name) if module else globals().get(name) for name in functions_list}

    if isinstance(module_name, list):
        return function_registry_dynamic(functions_list, module_name)
    return functions_registry_safe(functions_list, module_name)
    # get_function_parameters


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
    module = importlib.import_module(module_name) if module_name else globals()
    registry = {}
    for name in functions_list:
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
            registry[name] = None
            print(f"Module '{module_name}','{name}' not found.")
            # importlib.reload(module_path)

        except Exception as e:
            registry[name] = None
            print(f"[⚠️] 加载函数失败: {name} → {type(e).__name__}: {e}")

    return registry


def function_registry_dynamic(functions_list: list, module_names: list):
    """
    动态加载模块并注册函数
    :param functions_list: 需要注册的函数名列表
    :param module_names: 模块名称列表（字符串形式）
    :return: 函数注册表
    """
    registry = {}
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)  # 动态加载模块
            for name in functions_list:
                if name in registry:  # 避免重复覆盖，只注册第一个找到的
                    continue
                func = getattr(module, name, None)
                if func is not None and callable(func):
                    registry[name] = func
        except ModuleNotFoundError:
            print(f"Module '{module_name}' not found.")
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
    """获取当前 Worker 的唯一标识"""
    #  使用进程 ID + 主机名 + 启动时间
    pid = os.getpid()
    hostname = socket.gethostname()
    # start_time = os.times().elapsed  # 进程启动后的时间
    worker_info = f"{pid}-{hostname}"

    worker_id = os.environ.get("GUNICORN_WORKER_ID", None)  # 使用 Gunicorn 环境变量（如果使用 Gunicorn）
    if worker_id is not None:
        # is_main_worker = worker_id == "0"
        return worker_id, worker_info
    unique_id = hashlib.sha256(worker_info.encode()).hexdigest()[:16]  # 使用唯一哈希标识
    return f"worker-{unique_id}", worker_info


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


def pickle_serialize(obj):
    try:
        json.dumps(obj)  # 测试是否可JSON序列化
        return obj
    except (TypeError, ValueError):  # 将不可JSON序列化的部分转为pickle的base64字符串
        try:
            data = pickle.dumps(obj)  # 返回 bytes 类型
            return {'__pickle__': base64.b64encode(data).decode('utf-8')}  # 转为 ASCII-safe 字节串
        except Exception as e:
            print(f"Object cannot be pickled: {e}")
            return obj


def pickle_deserialize(obj):
    if isinstance(obj, dict):
        if '__pickle__' in obj:
            try:
                return pickle.loads(base64.b64decode(obj['__pickle__'].encode('utf-8')))  # encoding='bytes' 避免自动导入模块
            except Exception as e:
                raise ValueError(f"Failed to decode pickle: {e}")
    return obj


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


    main()
