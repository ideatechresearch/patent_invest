from collections import defaultdict, OrderedDict
from pydantic import BaseModel
import json
from utils import generate_hash_key
from .base import *


class LRUCache:
    CHANNEL = "cache_invalidate"

    def __init__(self, capacity: int = -1, redis=None, prefix: str = "LRUCache", expire_sec: float = 99999):
        self.stack = OrderedDict()
        self.capacity = capacity

        self.redis = redis
        self.prefix = prefix
        self.expire_sec = expire_sec
        self.subscribed = False

    def get(self, key, default=None):
        if key in self.stack:
            self.stack.move_to_end(key)
            return self.stack[key]
        return default

    async def get_cache(self, args: list | tuple = None, redis=None, default=None, **kwargs):
        """检查缓存"""
        cache_key = generate_hash_key(*(args or ()), **kwargs)
        if cache_key in self.stack:
            return cache_key, self.get(cache_key, default)
        redis = redis or self.redis
        if redis:
            try:
                cached = await redis.get(f"{self.prefix}:{cache_key}")
                if cached:
                    data = json.loads(cached)
                    self.put(cache_key, data)
                    await redis.expire(f"{self.prefix}:{cache_key}", self.expire_sec)
                    return cache_key, data
            except Exception as e:
                logging.warning(f"cache get failed: {e}")
        return cache_key, self.get(cache_key, default)

    def put(self, key, value) -> None:
        if key in self.stack:
            self.stack[key] = value
            self.stack.move_to_end(key)
        else:
            self.stack[key] = value
        if len(self.stack) > self.capacity > 0:
            self.stack.popitem(last=False)

    async def set_cache(self, key, value, redis=None):
        redis = redis or self.redis
        if redis:
            try:
                await redis.setex(f"{self.prefix}:{key}", self.expire_sec, json.dumps(value))
                if self.subscribed:
                    await redis.publish(self.CHANNEL, key)  # 发布失效消息,其他 worker删本地
            except Exception as e:
                logging.warning(f"cache set failed: {e}")
        self.put(key, value)

    async def subscribe_invalidate(self):
        if not self.redis:
            raise RuntimeError("redis not set")
        try:
            pubsub = self.redis.pubsub()
            await pubsub.subscribe(self.CHANNEL)
            self.subscribed = True
            async for message in pubsub.listen():
                if message is None or message['type'] != 'message':
                    continue

                key = message['data']
                if isinstance(key, bytes):
                    key = key.decode()
                self.stack.pop(key, None)  # 失效本地 LRU
        except Exception as e:
            logging.warning(f"cache subscribe failed: {e}")
            self.subscribed = False

    def change_capacity(self, capacity):
        self.capacity = capacity
        for i in range(len(self.stack) - capacity):
            self.stack.popitem(last=False)

    def delete(self, key):
        if key in self.stack:
            del self.stack[key]

    def keys(self):
        return self.stack.keys()

    def __len__(self):
        return len(self.stack)

    def __contains__(self, key):
        return key in self.stack


class IntentMemory:
    def __init__(self, capacity: int = 5, redis=None, prefix: str = "intent_history"):
        # 使用 defaultdict 来存储用户和机器人的意图历史,deque(maxlen=capacity)
        self.history = defaultdict(lambda: defaultdict(list))  # memory {robot_id: {user_id: [intent_history]}}
        self.capacity = capacity
        self.redis = redis
        self.prefix = prefix

    def _generate_key(self, robot_id, user_id):
        return f"{self.prefix}:{robot_id}:{user_id}"

    async def loads_data(self):
        """从Redis加载所有历史记录到内存"""
        if self.redis:
            try:
                # 获取所有匹配的键
                keys = await self.redis.keys(f"{self.prefix}:*")
                if keys:
                    # 批量获取对应的值
                    cached_values = await self.redis.mget(*keys)
                    for key, value in zip(keys, cached_values):
                        if value:
                            key_str = key.decode() if isinstance(key, bytes) else key
                            parts = key_str.split(":")
                            if len(parts) == 3:
                                _, robot_id, user_id = parts
                                intents = json.loads(value)
                                self.history[robot_id][user_id] = intents[-self.capacity:]  # 截断最大数量
            except Exception as e:
                print(f"Redis loads error: {e}")

    async def _load_data(self, robot_id, user_id):
        """从Redis载入某个用户历史"""
        if self.redis:
            key = self._generate_key(robot_id, user_id)
            try:
                data = await self.redis.get(key)
                if data:
                    intents = json.loads(data)
                    self.history[robot_id][user_id] = intents[-self.capacity:]  # 截断
            except Exception as e:
                print(f"Redis load error: {e}")

    async def _save_data(self, robot_id, user_id, day: int = 7):
        """将某个用户的历史保存到Redis"""
        if self.redis:
            his = self.history[robot_id][user_id]
            # index = len(his) - 1  # 当前记录位置
            key = self._generate_key(robot_id, user_id)
            try:
                await self.redis.setex(key, day * 24 * 3600, json.dumps(his))  # 保存7天
            except Exception as e:
                print(f"Redis save error: {e}")

    async def append(self, robot_id, user_id, intent, day: int = 7):
        """添加用户的意图到历史记录"""
        if not robot_id or not user_id or not intent:
            return
        self.history[robot_id][user_id].append(intent)

        # 保持每个用户历史记录最多为 5 条，避免历史记录过长
        if len(self.history[robot_id][user_id]) > self.capacity:
            self.history[robot_id][user_id].pop(0)

        await self._save_data(robot_id, user_id, day)

    def get_last(self, robot_id, user_id):
        """获取指定用户和机器人的最近意图"""
        return self.history[robot_id][user_id][-1] if self.history[robot_id][user_id] else None

    def get_history(self, robot_id, user_id, recent_n: int = None) -> list:
        """获取指定用户和机器人的意图"""
        memory = self.history.get(robot_id, {}).get(user_id, [])
        recent_n = recent_n or self.capacity
        return memory[-recent_n:]

    def delete(self, robot_id, user_id, index: list | int) -> None:
        memory = self.history.get(robot_id, {}).get(user_id, [])
        if not memory:
            return
        if isinstance(index, int):
            del memory[index]
        else:
            for i in index:
                del memory[i]

    def export_all(self) -> list[dict]:
        """导出为列表结构便于写入数据库"""
        records = []
        for robot_id, users in self.history.items():
            for user_id, intents in users.items():
                for idx, intent in enumerate(intents):
                    data = intent
                    if isinstance(intent, str):
                        pass
                    elif isinstance(intent, BaseModel):
                        data = intent.model_dump()
                    else:
                        data = json.dumps(intent)
                    records.append({
                        "robot_id": robot_id,
                        "user_id": user_id,
                        "intent": data,
                        "index": idx
                    })
        return records

        # if self.redis:
        #     results = await self.redis.lrange(self.redis_key, 0, -1)
        #     return [item.decode('utf-8') for item in results]
        # return list(self.history)
