from collections.abc import Mapping
from functools import wraps
import os, joblib


class IndexProxy(Mapping):
    """
    兼容：
      proxy[key]
      proxy.get(key)
      proxy()
    """

    def __init__(self, data: dict | list | tuple):
        self._data = data

    def __getitem__(self, key):
        if isinstance(self._data, dict) and key not in self._data:
            raise KeyError(f"Invalid key: {key}")
        return self._data[key]

    def get(self, key, default=None):
        return self._data.get(key, default)

    def __call__(self):
        return self

    def __contains__(self, key):
        return key in self._data

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f"{self.__class__.__name__}({self._data})"

    def items(self):
        return self._data.items()

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def index(self, key):
        return self._data.index(key)

    def clear(self):
        self._data.clear()


class class_property:
    """
    带缓存的 class-level property
    第一次访问计算，之后缓存到类
    自定义缓存属性名，适合无参或固定参数的懒加载。
    用于懒加载（lazy load）型类属性。仅检查当前类。
    """

    def __init__(self, attr_name: str = None):
        self.attr_name = attr_name
        self.func = None

    def __call__(self, func):
        self.func = func
        self.attr_name = self.attr_name or func.__name__.upper()
        return self

    def __get__(self, obj, cls):
        if not hasattr(cls, self.attr_name):
            raw = self.func(cls)
            if isinstance(raw, (dict, list, tuple)):
                raw = IndexProxy(raw)
            setattr(cls, self.attr_name, raw)
        return getattr(cls, self.attr_name)

    def __set_name__(self, owner, name):
        if not hasattr(owner, '_class_prop_names'):
            owner._class_prop_names = []
        owner._class_prop_names.append(name)

        if not hasattr(owner, '_class_cache_names'):
            owner._class_cache_names = []
        owner._class_cache_names.append(self.attr_name)

    @staticmethod
    def class_property(attr_name: str):
        """
        缓存装饰器，支持首次调用生成值并缓存到类属性。
        """

        def decorator(func) -> classmethod:
            def wrapper(cls, *args):
                if not hasattr(cls, attr_name):
                    setattr(cls, attr_name, func(cls, *args))
                return getattr(cls, attr_name)

            return classmethod(wrapper)

        return decorator

    @staticmethod
    def save(cls, prop_dir='data/props/'):
        """
        保存类的所有 class_property 缓存值到文件。
        自动触发计算并保存，无需指定属性名。
        """
        os.makedirs(prop_dir, exist_ok=True)

        # 触发所有懒加载计算
        for prop_name in getattr(cls, '_class_prop_names', []):
            _ = getattr(cls, prop_name)

        for cache_name in getattr(cls, '_class_cache_names', []):
            if hasattr(cls, cache_name):
                value = getattr(cls, cache_name)
                joblib.dump(value, f'{prop_dir}/{cls.__name__}_{cache_name}.pkl')
        print(f"类 {cls.__name__} 的属性已保存至 {prop_dir}")

    @staticmethod
    def load(cls, prop_dir='data/props/'):
        """
        加载保存的类属性值，并设置回类。
        自动处理所有已注册的缓存属性，无需指定属性名。
        """
        for cache_name in getattr(cls, '_class_cache_names', []):
            prop_path = f'{prop_dir}/{cls.__name__}_{cache_name}.pkl'
            if os.path.exists(prop_path):
                value = joblib.load(prop_path)
                setattr(cls, cache_name, value)
        print(f"类 {cls.__name__} 的属性已从 {prop_dir} 加载")


class class_cache:
    def __init__(self, cache_name: str = None, key=None):
        self.cache_name = cache_name
        self.key_func = key
        self.func = None

    def __call__(self, func):
        self.func = func
        if self.cache_name is None:
            self.cache_name = f"_{func.__name__}_cache".upper()
        return self

    def __get__(self, obj, cls):
        cache = cls.__dict__.get(self.cache_name)
        if cache is None:
            cache = {}
            setattr(cls, self.cache_name, cache)

        bound = self.func.__get__(obj, cls)  # self.func.__func__/self.func(cls,...

        def wrapper(*args, **kwargs):
            key = self.key_func(*args, **kwargs) if self.key_func else (args, tuple(sorted(kwargs.items())))
            if key not in cache:
                cache[key] = bound(*args, **kwargs)
            return cache[key]

        wrapper.cache = cache
        return wrapper

    def __set_name__(self, owner, name):
        if not hasattr(owner, '_class_cache_names'):
            owner._class_cache_names = []
        owner._class_cache_names.append(self.cache_name)

    @staticmethod
    def save(cls, cache_dir='data/cache/'):
        """
        保存类的所有 class_cache 缓存字典到文件。
        自动保存现有缓存，无需指定名称。
        """
        os.makedirs(cache_dir, exist_ok=True)

        for cache_name in getattr(cls, '_class_cache_names', []):
            if hasattr(cls, cache_name):
                cache = getattr(cls, cache_name)
                joblib.dump(cache, f'{cache_dir}/{cls.__name__}_{cache_name}.pkl')
        print(f"类 {cls.__name__} 的缓存已保存至 {cache_dir}")

    @staticmethod
    def load(cls, cache_dir='data/cache/'):
        """
        加载保存的类缓存字典，并设置回类。
        自动处理所有已注册的缓存名称。
        """
        for cache_name in getattr(cls, '_class_cache_names', []):
            cache_path = f'{cache_dir}/{cls.__name__}_{cache_name}.pkl'
            if os.path.exists(cache_path):
                cache = joblib.load(cache_path)
                setattr(cls, cache_name, cache)
        print(f"类 {cls.__name__} 的缓存已从 {cache_dir} 加载")


def chainable_method(func):
    """装饰器，使方法支持链式调用，保留显式返回值"""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        return self if result is None else result

    return wrapper


def class_status(state: str = 'TODO', notes=None):
    """标记方法状态的装饰器，不改变原方法行为。
    '未实现''已废弃''待完成''实验性''有BUG''参考方法'
    """

    def decorator(func):
        func._method_status = {"state": state, "notes": notes}
        return func

    return decorator


def check_class_status(cls):
    """检查类中所有方法的状态，返回字典。"""
    status_dict = {}
    for name in dir(cls):
        if not name.startswith('_'):
            attr = getattr(cls, name)
            if callable(attr):
                status_info = getattr(attr, '_method_status', None)
                if status_info:
                    status_dict[name] = status_info
    return status_dict
