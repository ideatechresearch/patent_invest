from sqlalchemy import create_engine, select, JSON, Column, ForeignKey, String, Integer, BigInteger, Boolean, Float, \
    DateTime, Index, TEXT
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy import func, or_, and_, text
from sqlalchemy.dialects.mysql import MEDIUMTEXT
from sqlalchemy.orm import sessionmaker, declarative_base, mapped_column, Mapped, Session
# from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import NullPool, QueuePool
import pymysql
import secrets
import copy
from typing import List, Dict, Optional, Union
from collections import defaultdict
from config import *

Base = declarative_base()  # ORM 模型继承基类
Chat_History_Cache = []

# echo=True 仅用于调试 poolclass=NullPool,每次请求都新建连接，用完就断，不缓存
# async_engine = create_async_engine(Config.ASYNC_SQLALCHEMY_DATABASE_URI)
# AsyncSessionLocal = sessionmaker(autocommit=False, autoflush=False, expire_on_commit=False, bind=async_engine,
#                                  class_=AsyncSession)
# poolclass=QueuePool,多线程安全的连接池，复用连接
engine = create_engine(Config.SQLALCHEMY_DATABASE_URI, pool_recycle=14400, pool_size=8, max_overflow=20,
                       pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)  # isolation_level='SERIALIZABLE'


# async def get_async_session() -> AsyncSession:
#     # 异步依赖
#     async with AsyncSessionLocal() as session:
#         yield session  # 自动 await 生成器
#     # finally:
#     #   await session.close()


def get_db():
    # 同步依赖
    db = SessionLocal()
    try:
        yield db
    # except Exception:
    #     db.rollback()
    #     raise
    finally:
        db.close()


class OperationMysql:
    def __init__(self, host, user, password, db_name, port=3306, charset="utf8mb4"):
        self.host = host
        self.user = user
        self.password = password
        self.db_name = db_name
        self.charset = charset
        self.port = port
        self.conn = None
        self.cur = None

    def __enter__(self):
        # 打开连接
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 关闭游标和连接
        self.close()

    def close(self):
        if self.cur:
            self.cur.close()
            self.cur = None
        if self.conn:
            self.conn.close()
            self.conn = None

    def connect(self):
        self.close()
        try:
            self.conn = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                db=self.db_name,
                charset=self.charset,
                cursorclass=pymysql.cursors.DictCursor  # 这个定义使数据库里查出来的值为字典类型
            )
            self.cur = self.conn.cursor()  # 原生数据库连接方式
        except Exception as e:
            print(f"连接数据库失败: {e}")
            self.conn = None
            self.cur = None

    def ensure_connection(self):
        try:
            if self.conn:
                self.conn.ping(reconnect=True)
            else:
                self.connect()
        except Exception as e:
            print(f"[自动重连失败] {e}")
            self.connect()

    def run(self, sql, params=None):
        if params is None:
            params = ()

        sql_type = sql.strip().split()[0].lower()
        self.ensure_connection()
        try:
            self.cur.execute(sql, params)

            if sql_type == "select":
                return self.cur.fetchall()

            elif sql_type in {"insert", "update", "delete", "replace"}:
                self.conn.commit()
                if sql_type == "insert":
                    return self.conn.insert_id()
                return True

        except Exception as e:
            self.conn.rollback()
            print(f"[数据库执行出错] {e}")
        return None

    def search(self, sql, params: tuple | dict = None):
        if not sql.lower().startswith("select"):
            pass
        if params is None:
            params = ()
        self.cur.execute(sql, params)
        result = self.cur.fetchall()
        return result

    def execute(self, sql):
        # INSERT,UPDATE,DELETE
        try:
            self.cur.execute(sql)
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            print(f"执行 SQL 出错: {e}")

    def insert(self, sql, params: tuple | dict = None):
        try:
            if params is None:
                params = ()
            self.cur.execute(sql, params)
            new_id = int(self.conn.insert_id())
            self.conn.commit()
            return new_id  # self.cursor.lastrowid
        except Exception as e:
            self.conn.rollback()
            print(f"执行 SQL 出错: {e}")
            print(f"SQL: {sql} \n 参数: {params}")
        # finally:
        #     self.cur.close()
        #     self.conn.close()


class MysqlData(OperationMysql):
    db_config = parse_database_uri(Config.SQLALCHEMY_DATABASE_URI)

    def __init__(self):
        super().__init__(**type(self).db_config)
        # if not self.conn:
        #     self.connect()


def patent_search(patent_id, limit=10):
    table_name = 'patent_all_2408'  # fixed_table '融资公司专利-202406'
    column_name = '申请号'  # '公开（公告）号'
    query = text(f'SELECT table_name FROM `{table_name}` WHERE `{column_name}` = :id')
    with SessionLocal() as session:
        result = session.execute(query, {'id': patent_id})  # session.结合 ORM 或 Core 的方式来执行原生 SQL
        row = result.fetchone()
        if row:
            table_name = row[0]

        detail_query = text(f'SELECT * FROM `{table_name}` WHERE `{column_name}` = :id LIMIT :limit')
        result = session.execute(detail_query, {'id': patent_id, 'limit': limit})

        rows = result.fetchall()  # 获取所有行
        columns = result.keys()  # 获取列名
        results = [dict(zip(columns, row)) for row in rows]
        for item in results:
            for key, value in item.items():
                if isinstance(value, datetime):
                    item[key] = value.isoformat()  # 转换为ISO 8601格式
        return results


def company_search(co_name, search_type='invest', limit=10):
    if search_type == 'invest':
        column_name = '企业全称'
        table_name = 'invest_all_2412'
    elif search_type == 'zjtx':
        column_name = '公司名称'
        table_name = '专精特新企业基本信息表_2408'
    elif search_type == 'gxjs':
        column_name = '企业名称'
        table_name = '高新技术企业基本信息_202408'
    else:
        column_name = 'co_name'
        table_name = 'company_all_2408'

    query = text(f'SELECT * FROM `{table_name}` WHERE `{column_name}` LIKE :name LIMIT :limit')
    with SessionLocal() as session:
        result = session.execute(query, {'name': f"%{co_name}%", 'limit': limit})

        rows = result.fetchall()  # 获取所有行
        columns = result.keys()  # 获取列名
        results = [dict(zip(columns, row)) for row in rows]
        for item in results:
            for key, value in item.items():
                if isinstance(value, datetime):
                    item[key] = value.isoformat()  # 转换为ISO 8601格式
        return results


class IntentMemory:
    def __init__(self, max_his=5, redis=None, prefix="intent_history"):
        # 使用 defaultdict 来存储用户和机器人的意图历史,deque(maxlen=max_his)
        self.history = defaultdict(lambda: defaultdict(list))  # {robot_id: {user_id: [intent_history]}}
        self.max_his = max_his
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
                                self.history[robot_id][user_id] = intents[-self.max_his:]  # 截断最大数量
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
                    self.history[robot_id][user_id] = intents[-self.max_his:]  # 截断
            except Exception as e:
                print(f"Redis load error: {e}")

    async def _save_data(self, robot_id, user_id):
        """将某个用户的历史保存到Redis"""
        if self.redis:
            his = self.history[robot_id][user_id]
            # index = len(his) - 1  # 当前记录位置
            key = self._generate_key(robot_id, user_id)
            try:
                await self.redis.setex(key, 7 * 24 * 3600, json.dumps(his))  # 保存7天
            except Exception as e:
                print(f"Redis save error: {e}")

    async def append(self, robot_id, user_id, intent, db=None):
        """添加用户的意图到历史记录"""
        if not robot_id or not user_id or not intent:
            return
        self.history[robot_id][user_id].append(intent)

        # 保持每个用户历史记录最多为 5 条，避免历史记录过长
        if len(self.history[robot_id][user_id]) > self.max_his:
            self.history[robot_id][user_id].pop(0)

        await self._save_data(robot_id, user_id)

    def get_last(self, robot_id, user_id):
        """获取指定用户和机器人的最近意图"""
        return self.history[robot_id][user_id][-1] if self.history[robot_id][user_id] else None

    def get_history(self, robot_id, user_id):
        """获取指定用户和机器人的意图"""
        return self.history.get(robot_id, {}).get(user_id, [])

    def export_all(self):
        """导出为列表结构便于写入数据库"""
        records = []
        for robot_id, users in self.history.items():
            for user_id, intents in users.items():
                for idx, intent in enumerate(intents):
                    records.append({
                        "robot_id": robot_id,
                        "user_id": user_id,
                        "intent": intent,
                        "index": idx
                    })
        return records

        # if self.redis:
        #     results = await self.redis.lrange(self.redis_key, 0, -1)
        #     return [item.decode('utf-8') for item in results]
        # return list(self.history)


async def search_from_database(session, sql, **kwargs):
    import asyncio
    sql = text(sql)
    params = kwargs or {}
    if asyncio.iscoroutinefunction(session.execute):
        result = await session.execute(sql, params)
    else:
        result = session.execute(sql, params)

    return result.mappings().all()


# with SessionLocal() as session:
#     return await search_from_database(db,...)

class User(Base):
    __tablename__ = 'agent_users'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    username: Mapped[str] = mapped_column(String(99), unique=True, nullable=False, index=True)
    password: Mapped[str] = mapped_column(String(128), nullable=True)
    eth_address: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, unique=True)
    public_key: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, unique=True)

    api_key: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, unique=True, index=True)
    secret_key: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    group: Mapped[Optional[str]] = mapped_column(String(20), nullable=True, default='0')
    role: Mapped[str] = mapped_column(String(50), nullable=True, default='user')
    disabled: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=False, default=False)

    parent_id: Mapped[Optional[int]] = mapped_column(ForeignKey('agent_users.id'), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), onupdate=func.utc_timestamp(),
                                                 default=func.utc_timestamp())
    expires_at: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    chatcut_at: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True, default=0)

    def __repr__(self):
        return f"<User {self.username}>"

    def is_primary_account(self):
        return self.parent_id is None

    def update(self, db: Session, eth_address: Optional[str] = None, public_key: Optional[str] = None,
               expires_day: Optional[int] = 0, **kwargs):
        update_data = {}
        if eth_address is not None:
            update_data['eth_address'] = eth_address
        if public_key is not None:
            update_data['public_key'] = public_key
        if expires_day:
            update_data['expires_at'] = int((datetime.now(timezone.utc) + timedelta(days=expires_day)).timestamp())

        for key, value in kwargs.items():
            if hasattr(self, key):  # in cls.__mapper__.c:
                update_data[key] = value

        if update_data:
            db.query(User).filter_by(id=self.id).update(update_data)
            db.commit()
            db.refresh(self)
            return self

        return None

    @classmethod
    def create_user(cls, db: Session, username: str, password: str, role: str = 'user', group: str = '0',
                    eth_address: Optional[str] = None, public_key: Optional[str] = None):
        if not password and not (eth_address or public_key):
            print("必须提供密码，或者提供 eth_address 或 public_key 之一")  # raise ValueError(
            return None
        new_user = cls(
            username=username,
            password=cls.hash_password(password) if password else None,
            role=role,
            group=group,
            eth_address=eth_address,
            public_key=public_key
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return new_user

    @classmethod
    def get_user(cls, db: Session, username: Optional[str] = None, user_id: Optional[str] = None,
                 public_key: Optional[str] = None, eth_address: Optional[str] = None):
        query = db.query(cls)
        if username:
            return query.filter_by(username=username).first()
        if user_id:
            return query.filter_by(user_id=user_id).first()
        if public_key:
            return query.filter_by(public_key=public_key).first()
        if eth_address:
            return query.filter_by(eth_address=eth_address).first()
        return None

    @classmethod
    def update_user(cls, _id: int, db: Session, eth_address: Optional[str] = None, public_key: Optional[str] = None,
                    expires_day: Optional[int] = 0, **kwargs):
        user = db.query(cls).filter_by(id=_id).first()
        if user:
            return user.update(db=db, eth_address=eth_address, public_key=public_key, expires_day=expires_day, **kwargs)
        return None

    @classmethod
    def create_api_key(cls, _id: int, db: Session):
        while True:
            api_key = str(uuid.uuid4())
            if not db.query(cls).filter_by(api_key=api_key).first():
                break
        secret_key = secrets.token_urlsafe(32)
        cls.update_user(_id, db, api_key=api_key, secret_key=secret_key)
        return api_key, secret_key

    @classmethod
    def get_api_keys(cls, db: Session):
        api_keys = db.query(cls.api_key, cls.secret_key).all()
        return {api_key: secret_key for api_key, secret_key in api_keys}

    @staticmethod
    def hash_password(password: str) -> str:
        """哈希用户密码"""
        return hashlib.sha256(password.encode()).hexdigest()

    @staticmethod
    def verify_password(input_password: str, stored_password: str) -> bool:
        """验证密码是否匹配"""
        return hashlib.sha256(input_password.encode()).hexdigest() == stored_password

    @classmethod
    def validate_credentials(cls, username: str, password: str, db: Session):
        user = db.query(cls).filter_by(username=username).first()
        if user and cls.verify_password(password, str(user.password)):
            return user

        return None


class BaseChatHistory(Base):
    __tablename__ = 'agent_history'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # 'user' 或 'assistant'
    content: Mapped[str] = mapped_column(MEDIUMTEXT, nullable=False)
    name: Mapped[str] = mapped_column(String(99), nullable=True, index=True)

    user: Mapped[str] = mapped_column(String(99), nullable=False, index=True)  # role_id
    robot_id: Mapped[str] = mapped_column(String(99), nullable=True, index=True)
    model: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    agent: Mapped[str] = mapped_column(String(99), nullable=True)

    index: Mapped[int] = mapped_column(Integer, nullable=False)
    # data = Column(JSON, nullable=True)
    reference: Mapped[Optional[str]] = mapped_column(MEDIUMTEXT, nullable=True)
    transform: Mapped[Optional[str]] = mapped_column(TEXT, nullable=True)

    # prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=True, default=0)
    # completion_tokens: Mapped[int] = mapped_column(Integer, nullable=True, default=0)
    timestamp: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_name_time', 'name', 'robot_id', 'user', 'timestamp'),
    )

    def __init__(self, role, content, user, name=None, robot_id=None, model=None, agent=None, index=0,
                 reference=None, transform=None, timestamp: int = 0):
        self.role = role
        self.content = content
        self.name = name

        self.user = user
        self.robot_id = robot_id
        self.model = model
        self.agent = agent
        self.index = index
        self.reference = reference
        self.transform = transform
        self.timestamp = timestamp or int(time.time())
        # datetime.utcfromtimestamp(timestamp) datetime.utcnow().timestamp()

    def asdict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

    def display(self):
        print(f"Role: {self.role}, Content: {self.content}, Name: {self.name},Timestamp: {self.timestamp}")

    @classmethod
    def history_insert(cls, new_history, db: Session):
        try:
            if len(new_history) > 0:
                db.add_all([cls(**msg) for msg in new_history])
                db.commit()
        except Exception as e:
            db.rollback()
            print(f"Error inserting history: {e}")

    @classmethod
    def user_history(cls, db: Session, user: str, name: str = None, robot_id: str = None, agent: str = None,
                     filter_time: float = 0, all_payload: bool = True):
        query = db.query(cls).filter(cls.user == user,
                                     or_(cls.name == name, name is None),
                                     or_(cls.robot_id == robot_id, robot_id is None),
                                     cls.timestamp >= filter_time)
        if agent:  # or_(cls.agent == agent, agent is None)
            query = query.filter(cls.agent == agent)

        if all_payload:
            return [record.asdict() for record in query.all()]
        records = query.with_entities(cls.role, cls.content, cls.name).all()
        return [{'role': record.role, 'content': record.content, 'name': record.name} for record in records]

    @classmethod
    def sequential_insert(cls, db: Session, user=None, robot_id=None, agent=None):
        i = 0
        while i < len(Chat_History_Cache):
            try:
                msg = Chat_History_Cache[i]
                if ((not agent or msg['agent'] == agent) and
                        (not user or msg['user'] == user) and
                        (not robot_id or msg['robot_id'] == robot_id)):

                    record = cls(**msg)
                    db.add(record)
                    db.commit()
                    del Chat_History_Cache[i]
                    if not any(
                            (not agent or m['agent'] == agent) and
                            (not user or msg['user'] == user) and
                            (not robot_id or m['robot_id'] == robot_id)
                            for m in Chat_History_Cache[i:]):
                        break
                else:
                    i += 1
            except Exception as e:
                db.rollback()
                print(f"Error inserting chat history: {e}")
                break


def get_user_history(user: str, name: Optional[str], robot_id: Optional[str], filter_time: float, db: Session,
                     agent: str = None, request_uuid: Optional[str] = None) -> List[dict]:
    user_history = [msg for msg in Chat_History_Cache
                    if msg['user'] == (user or request_uuid)
                    and (not name or msg['name'] == name)
                    and (not robot_id or msg['robot_id'] == robot_id)
                    and (not agent or msg['agent'] == agent)
                    and msg['timestamp'] >= filter_time]

    if user and db:  # 从数据库中补充历史记录
        user_history.extend(
            BaseChatHistory.user_history(db, user, name, robot_id, agent=agent, filter_time=filter_time,
                                         all_payload=True))

    return user_history


def cut_chat_history(user_history: List[dict], max_size_limit_count=33000):
    """
    根据 token 数截断对话历史，保留最近的上下文。

    :param user_history: 完整的消息列表，每项 {'role':..., 'content':...}
    :param max_size_limit_count: 最大允许的 token 数
    :return: 截断后的消息列表
    32K tokens
    64K tokens
    128K token
    """
    last_records = []
    total_size = 0
    if max_size_limit_count > 0:
        for i in range(len(user_history) - 2, -1, -2):
            pair = user_history[i:i + 2]
            pair_len = sum(len(record.get("content", "")) for record in pair)  # 计算这一对消息的总长度,lang_token_size

            if total_size + pair_len > max_size_limit_count:
                break

            last_records = pair + last_records
            total_size += pair_len

    elif max_size_limit_count < 0:
        last_records = user_history[max_size_limit_count * 2:]  # -(filter_limit * 2):
    else:
        last_records = user_history

    return last_records


def build_chat_history(user_request: str, user: str, name: Optional[str], robot_id: Optional[str],
                       db: Session, user_history: List[dict] = None, use_hist=False,
                       filter_limit: int = -500, filter_time: float = 0,
                       agent: Optional[str] = None, request_uuid: Optional[str] = None):
    # 构建用户的聊天历史记录，并生成当前的用户消息。
    history = []
    if not user_history:
        if use_hist:
            user_history = get_user_history(user, name, robot_id, filter_time, db, agent=agent,
                                            request_uuid=request_uuid)
            last_records = cut_chat_history(sorted(user_history, key=lambda x: x['timestamp']),
                                            max_size_limit_count=filter_limit)
            history.extend(
                [{'role': msg['role'], 'content': msg['content'], 'name': msg['name']} for msg in last_records])

        history.append({'role': 'user', 'content': user_request, 'name': name})
    else:
        last_records = cut_chat_history(user_history, max_size_limit_count=filter_limit)
        history.extend([msg.dict() for msg in last_records])
        if not user_request:
            if history[-1]["role"] == 'user':
                user_request = history[-1]["content"]

    return history, user_request, len(user_history)


def save_chat_history(user_request: str, bot_response: str,
                      user: str, name: Optional[str], robot_id: Optional[str],
                      agent: Optional[str], hist_size: int, model_name: str, timestamp: float,
                      db: Session, refer: List[str], transform=None, request_uuid: Optional[str] = None):
    if not user_request or not bot_response:
        return
    uid = user or request_uuid
    if not uid:
        return
    new_history = [
        {'role': 'user', 'content': user_request, 'name': name, 'robot_id': robot_id, 'user': uid,
         'agent': agent, 'index': hist_size + 1, 'timestamp': timestamp},
        {'role': 'assistant', 'content': bot_response, 'name': name, 'robot_id': robot_id, 'user': uid,
         'agent': agent, 'index': hist_size + 2, 'model': model_name,  # 'timestamp': time.time(),
         'reference': json.dumps(refer, ensure_ascii=False) if refer else None,  # '\n'.join(refer)
         'transform': json.dumps(transform, ensure_ascii=False) if transform else None}
    ]
    # 保存聊天记录到数据库，或者保存到内存中当数据库不可用时。
    try:
        if user and db:
            BaseChatHistory.history_insert(new_history, db)
        else:
            raise Exception
    except:
        Chat_History_Cache.extend(new_history)


class ChatHistory(BaseChatHistory):
    def __init__(self, user: str, name: Optional[str], robot_id: Optional[str],
                 agent: Optional[str], model_name: Optional[str],
                 timestamp: float, request_uuid: Optional[str] = None):
        super().__init__(
            role='user',
            content="",
            name=name,
            user=user,
            robot_id=robot_id,
            model=model_name,
            agent=agent,
            index=0,
            timestamp=int(timestamp)
        )
        # self.db = db #SessionLocal()
        self.uid = user or request_uuid
        self.user_request = ''
        self.user_history: List[dict] = []

    def get(self, filter_time: float = 0, db: Session = None):
        """
        user 用于标识整个请求的发起者（如用户 ID 或机器人 ID）,用户/机器人追踪
        name 是可选字段，通常用于多用户对话，仅作为上下文参考,用于区分对话中不同角色（如多个用户、机器人、系统）的名称
        :param filter_time:
        :param db:
        :return:
        """
        if not self.uid:
            return []

        user_history = [msg for msg in Chat_History_Cache
                        if msg['user'] == self.uid
                        and (not self.name or msg['name'] == self.name)
                        and (not self.robot_id or msg['robot_id'] == self.robot_id)
                        and (not self.agent or msg['agent'] == self.agent)
                        and msg['timestamp'] >= filter_time]

        if self.user and db:  # 从数据库中补充历史记录
            user_history.extend(
                BaseChatHistory.user_history(db, self.user, self.name, self.robot_id, agent=self.agent,
                                             filter_time=filter_time, all_payload=True))

        return user_history

    def build(self, user_request: str, user_messages: List[dict] = None, use_hist=False,
              filter_limit: int = -500, filter_time: float = 0, db: Session = None):
        history = []
        if not user_messages:
            if use_hist:  # 如果 use_hist 为真，可以根据 filter_limit 和 filter_time 筛选出历史记录，如果没有消息提供，过滤现有的聊天记录，user_message为问题
                self.user_history = self.get(filter_time, db)
                message_records = cut_chat_history(sorted(self.user_history, key=lambda x: x['timestamp']),
                                                   max_size_limit_count=filter_limit)
                history.extend(
                    [{'role': msg['role'], 'content': msg['content'], 'name': msg['name']} for msg in message_records])

            if user_request:
                self.user_request = user_request
                history.append({'role': 'user', 'content': user_request, 'name': self.name})
        else:
            self.user_history = user_messages.copy()
            message_records = cut_chat_history(self.user_history, max_size_limit_count=filter_limit)
            history.extend([msg.dict() for msg in message_records])

            if not self.name:
                self.name = next((msg.get("name") for msg in reversed(history) if msg.get("role") == "user"), None)
            if not self.uid:
                self.uid = next((msg.get("name") for msg in reversed(history) if msg.get("role") == "assistant"), None)

            if not user_request:
                if history[-1]["role"] == 'user':
                    self.user_request = history[-1]["content"]
                    # 如果提供消息,则使用最后一条user content为问题
            else:
                self.user_request = user_request
                # if history[-1]["role"] != 'user':
                #     history.append({'role': 'user', 'content': user_request, 'name': self.name})
                # 后续发送模型前手动添加

        return history

    @classmethod
    def rebuild(cls, user_request: str, history: List[dict] = None):
        # 创建message副本
        user_history = copy.deepcopy(history)
        if user_history and user_history[-1]["role"] == 'user':
            user_history[-1]['content'] = user_request
        return user_history

    def save(self, bot_response: str, refer: Union[List[str], Dict] = None, transform=None,
             user_request: str = None, model_name: str = None, db: Session = None):
        """
        为了方便成对取出记录，'user','robot_id','name','agent'保持一致并成对保存
        模型可以得到多条回复，为了方便存储，暂时只记录一条
        user 建议用唯一标识,如果未提供,为匿名用户,不保存数据库
        :param bot_response:
        :param refer:
        :param transform:
        :param user_request:
        :param model_name:
        :param db:
        :return:
        """
        user_request = user_request or self.user_request
        if not user_request or not bot_response:
            print('no content to save')
            return
        if not self.uid:
            return

        hist_size = len(self.user_history)
        new_history = [
            {'role': 'user', 'content': user_request, 'name': self.name,
             'user': self.uid, 'robot_id': self.robot_id, 'agent': self.agent,
             'index': hist_size + 1, 'model': self.model or model_name, 'timestamp': self.timestamp
             },
            {'role': 'assistant', 'content': bot_response, 'name': self.name,
             'user': self.uid, 'robot_id': self.robot_id, 'agent': self.agent,
             'index': hist_size + 2, 'model': model_name or self.model,
             'reference': json.dumps(refer, ensure_ascii=False, default=str) if refer else None,  # '\n'.join(refer)
             'transform': json.dumps(transform, ensure_ascii=False, default=str) if transform else None
             # 'timestamp': time.time(),
             }
        ]
        # 保存聊天记录到数据库，或者保存到内存中当数据库不可用时。
        try:
            if self.user and db:
                BaseChatHistory.history_insert(new_history, db)
            else:
                raise Exception
        except:
            Chat_History_Cache.extend(new_history)

    def save_cache(self):
        """保存缓存到数据库"""
        with SessionLocal() as session:
            self.sequential_insert(session, user=self.user, robot_id=self.robot_id)


class BaseReBot(Base):
    __tablename__ = 'agent_robot'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    user_content: Mapped[str] = mapped_column(MEDIUMTEXT, nullable=True)  # "prompt"
    assistant_content: Mapped[str] = mapped_column(MEDIUMTEXT, nullable=True)  # "completion"
    system_content: Mapped[str] = mapped_column(TEXT, nullable=True)

    agent: Mapped[str] = mapped_column(String(50), nullable=True, default='chat')  # chat_type

    model: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    name: Mapped[Optional[str]] = mapped_column(String(99), nullable=True, index=True)
    user: Mapped[str] = mapped_column(String(99), nullable=True, index=True)
    robot_id: Mapped[Optional[str]] = mapped_column(String(99), nullable=True, index=True)
    msg_id: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    summary: Mapped[Optional[str]] = mapped_column(TEXT, nullable=True)
    reference: Mapped[Optional[str]] = mapped_column(MEDIUMTEXT, nullable=True)
    transform: Mapped[Optional[str]] = mapped_column(TEXT, nullable=True)

    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=True, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=True, default=0)

    timestamp: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_name_time', 'name', 'user', 'robot_id', 'timestamp'),
    )

    def __init__(self, user_content=None, assistant_content=None, system_content=None, agent='chat',
                 name: str = None, user: str = None, robot_id: str = None, model: str = None, msg_id=0,
                 summary=None, reference=None, transform=None, timestamp: int = 0):
        self.user_content = user_content
        self.assistant_content = assistant_content
        self.system_content = system_content
        self.agent = agent

        self.name = name
        self.user = user
        self.robot_id = robot_id
        self.model = model
        self.msg_id = msg_id

        self.summary = summary  # 背景信息
        self.reference = reference  # 参考信息,rag,tolls
        self.transform = transform

        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.timestamp = timestamp or int(time.time())
        self.created_at = datetime.now(timezone.utc)
        # datetime.utcfromtimestamp(timestamp) datetime.utcnow().timestamp()

    def asdict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

    def setdata(self, data: dict):
        for k, v in data.items():
            setattr(self, k, v)

    def display(self):
        print(f"User: {self.user_content}, Assistant: {self.assistant_content}, Timestamp: {self.timestamp}")

    def insert(self, db):
        try:
            db.add(self)
            db.commit()
            return True
        except Exception as e:
            db.rollback()
            return False

    @classmethod
    def get(cls, db: Session, name: str, user: str, robot_id: str, limit: int = 10) -> list[dict]:
        """
        获取某用户的历史对话记录
        :param db: SQLAlchemy Session
        :param name: 用户名
        :param user: 用户 ID
        :param robot_id: 机器人 ID
        :param limit: 限制返回条数
        """
        history = []
        try:
            records = db.query(cls).filter(
                cls.name == name,
                cls.user == user,
                cls.robot_id == robot_id
            ).order_by(cls.timestamp.desc()).limit(limit).all()

            for rec in reversed(records):  # 倒序，保持时间先后
                if rec.system_content:
                    history.append({"role": "system", "content": rec.system_content, "name": "system"})
                if rec.user_content:
                    history.append({"role": "user", "content": rec.user_content, "name": rec.name})
                if rec.assistant_content:
                    history.append({"role": "assistant", "content": rec.assistant_content, "name": rec.robot_id})

        except Exception as e:
            print(f"[get_history] 查询失败: {e}")

        return history

    @classmethod
    def save(cls, user, robot_id, model=None, agent='chat', instance=None, messages: list[dict] = None,
             model_response: dict = None, summary=None, reference=None, transform=None):
        """自动从 instance 或 messages/model_response 中提取并保存数据库"""
        data = cls(user=user, robot_id=robot_id, model=model, agent=agent,
                   summary=summary, reference=reference, transform=transform).asdict()

        if isinstance(instance, cls):
            data.update({k: v for k, v in instance.asdict().items() if v})

        if messages and isinstance(messages, list):
            last_user_msg = next((msg for msg in reversed(messages) if msg.get("role") == "user"), None)
            data['msg_id'] = len(messages)
            if last_user_msg:
                data["name"] = data["name"] or last_user_msg.get("name")
                data["user_content"] = data["user_content"] or last_user_msg.get("content")
            if not data["robot_id"]:
                data["robot_id"] = next(
                    (msg.get("name") for msg in reversed(messages) if msg.get("role") == "assistant"),
                    None)
            if not data["system_content"]:
                data["system_content"] = next((msg.get("content") for msg in messages if msg.get("role") == "system"),
                                              None)

        if model_response:
            data["assistant_content"] = model_response.get('choices', [{}])[0].get('message', {}).get('content') or \
                                        data["assistant_content"]
            data["model"] = model_response.get('model', data["model"])
            data["timestamp"] = model_response.get("created", data["timestamp"])

            data["prompt_tokens"] = model_response.get('usage', {}).get('prompt_tokens', 0)
            data["completion_tokens"] = model_response.get('usage', {}).get('completion_tokens', 0)

        final_data = {k: v for k, v in data.items() if v is not None}
        if isinstance(instance, cls):
            instance.setdata(final_data)

        fields = tuple(final_data.keys())
        params = tuple(final_data.values())
        sql = f"INSERT INTO {cls.__tablename__} ({', '.join(fields)}) VALUES ({', '.join(['%s'] * len(fields))})"
        # print(f"SQL: {sql} \n 参数: {params}")
        with MysqlData() as session:
            session.insert(sql, params)

    @classmethod
    def history(cls, user, robot_id, name, system_content=None, user_content=None, agent='chat', filter_day=None,
                limit: int = 10):
        """组装hsistory"""
        if not filter_day:
            filter_day = datetime.today().strftime('%Y-%m-%d')
        sql = f"""
             SELECT user_content, assistant_content, system_content, reference ,transform, summary, robot_id, name
             FROM {cls.__tablename__}
             WHERE created_at >= %s
                AND agent = %s 
                AND user = %s 
                AND robot_id = %s
                AND name = %s 
             ORDER BY timestamp ASC LIMIT {limit}
         """
        params = (filter_day, agent, user, robot_id, name)
        with MysqlData() as session:
            result = session.search(sql, params)
        history = []

        for item in result:
            system = item.get('system_content') or item.get('summary')  # 系统提示
            if system:
                history.append({"role": "system", "content": system, 'name': "system"})
            question = item['user_content'] or item.get('reference')
            if question:
                history.append({"role": "user", "content": question, 'name': item.get('name', name)})
            answer = item.get('assistant_content') or item.get('transform')  # 答复信息
            if answer:
                history.append({"role": "assistant", "content": answer, 'name': item.get('robot_id', robot_id)})

        if system_content:
            history.append({"role": "system", "content": system_content, 'name': "system"})
        if user_content:
            history.append({"role": "user", "content": user_content, 'name': name})
        return history


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


if __name__ == "__main__":
    # with OperationMysql(host="localhost", user="root", password="password", db_name="test_db") as db111:
    #     print(db111.search("SELECT * FROM my_table"))

    with MysqlData() as db:
        print(db.search("SELECT * FROM agent_users"))
