from collections.abc import Mapping
from functools import wraps


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

        def wrapper(*args, **kwargs):
            key = self.key_func(*args, **kwargs) if self.key_func else (args, tuple(sorted(kwargs.items())))

            if key not in cache:
                cache[key] = self.func(cls, *args, **kwargs)
            return cache[key]

        wrapper.cache = cache
        return wrapper


def chainable_method(func):
    """装饰器，使方法支持链式调用，保留显式返回值"""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        return self if result is None else result

    return wrapper
