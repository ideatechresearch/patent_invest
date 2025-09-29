from sqlalchemy import create_engine, select, JSON, Column, ForeignKey, String, Integer, BigInteger, Boolean, Float, \
    DateTime, Index, TEXT
from sqlalchemy import func, or_, and_, text
from sqlalchemy.dialects.mysql import MEDIUMTEXT
from sqlalchemy.orm import sessionmaker, declarative_base, mapped_column, Mapped, Session
from sqlalchemy.schema import FetchedValue
# from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
# from sqlalchemy.pool import NullPool, QueuePool
from contextlib import asynccontextmanager
import hashlib, secrets, uuid
import time, json, copy
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Union, ClassVar, AsyncGenerator
from collections import defaultdict
from pydantic import BaseModel

from utils import cut_chat_history
from config import Config
from service import BaseMysql, AsyncMysql, OperationMysql, AsyncBatchAdd

Base = declarative_base()  # ORM 模型继承基类

engine = create_engine(Config.SQLALCHEMY_DATABASE_URI, pool_recycle=14400, pool_size=3, max_overflow=Config.DB_MAX_SIZE,
                       pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)  # isolation_level='SERIALIZABLE'
# echo=True 仅用于调试
# poolclass=QueuePool,多线程安全的连接池，复用连接,poolclass=NullPool,每次请求都新建连接，用完就断，不缓存
async_engine = create_async_engine(Config.ASYNC_SQLALCHEMY_DATABASE_URI,
                                   pool_size=4,  # 最大连接数
                                   max_overflow=Config.DB_MAX_SIZE,  # 额外允许溢出的连接 20
                                   pool_timeout=30,  # 获取连接超时时间（秒）
                                   pool_recycle=3600,  # 连接回收时间，1h 回收，避免空闲断连
                                   future=True)
AsyncSessionLocal = sessionmaker(bind=async_engine, class_=AsyncSession, autocommit=False, autoflush=False,
                                 expire_on_commit=False)


def get_db():
    """同步依赖"""
    db = SessionLocal()
    try:
        yield db
    # except Exception:
    #     db.rollback()
    #     raise
    finally:
        db.close()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """异步依赖"""
    async with AsyncSessionLocal() as session:
        yield session
        # 这里不用手动 close，async with 已经处理了
        # pass


@asynccontextmanager
async def get_session_context():
    """手动控制事务的 async 上下文"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            # await session.commit()  # 自动提交
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def get_db_connection(db_config: dict, dictionary=True):
    """
    获取 MySQL 数据库连接对象。
    示例：
        db_config = parse_database_uri(Config.SQLALCHEMY_DATABASE_URI)
        conn = get_db_connection(db_config)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM table")

    使用完后请务必关闭连接：
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

    :param db_config: 连接配置 dict，若为 None 可默认读取配置文件或环境变量
    :return: mysql.connector.Connection | None
    """
    try:
        import mysql.connector
        return mysql.connector.connect(**db_config, dictionary=dictionary)  # connection
    except Exception as e:
        print(f"Error connecting to MySQL: {e}")
        return None


async def search_from_database(session, sql: str, **kwargs):
    import asyncio
    sql = text(sql)
    params = kwargs or {}
    if asyncio.iscoroutinefunction(session.execute):
        result = await session.execute(sql, params)  # AsyncSession
    else:
        result = session.execute(sql, params)  # Session

    return result.mappings().all()


async def patent_search(patent_id, limit=10):
    table_name = 'patent_all_2408'  # fixed_table '融资公司专利-202406'
    column_name = '申请号'  # '公开（公告）号'
    query = text(f'SELECT table_name FROM `{table_name}` WHERE `{column_name}` = :id')
    async with AsyncSessionLocal() as session:
        result = await session.execute(query, {'id': patent_id})  # session.结合 ORM 或 Core 的方式来执行原生 SQL
        row = result.fetchone()
        if row:
            table_name = row[0]

        detail_query = text(f'SELECT * FROM `{table_name}` WHERE `{column_name}` = :id LIMIT :limit')
        result = await session.execute(detail_query, {'id': patent_id, 'limit': limit})

        rows = result.fetchall()  # 获取所有行
        columns = result.keys()  # 获取列名
        results = [dict(zip(columns, row)) for row in rows]
        for item in results:
            for key, value in item.items():
                if isinstance(value, datetime):
                    item[key] = value.isoformat()  # 转换为ISO 8601格式value.strftime("%Y-%m-%d %H:%M:%S")
                elif isinstance(value, dict):
                    item[key] = json.dumps(value, ensure_ascii=False)
        return results


async def company_search(co_name, search_type='invest', limit=10):
    """
    根据公司名称和搜索类型查询相关信息，如投融资、专精特新、高新技术，并返回限定数量的结果。
    搜索类型:
    invest：企业投融资
    zjtx：专精特新企业基本信息
    gxjs：高新技术企业基本信息
    """
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
    async with AsyncSessionLocal() as session:
        result = await session.execute(query, {'name': f"%{co_name}%", 'limit': limit})

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
        if len(self.history[robot_id][user_id]) > self.max_his:
            self.history[robot_id][user_id].pop(0)

        await self._save_data(robot_id, user_id, day)

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
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, server_default=FetchedValue())

    __table_args__ = (
        Index('uniq_user_robot_agent_role_time_index_msg', 'user', 'name', 'robot_id', 'role', 'agent',
              'index', 'timestamp', 'content_hash', unique=True),
        Index('idx_user_robot_time', 'user', 'name', 'robot_id', 'timestamp'),
    )

    Chat_History_Cache: ClassVar[List[dict]] = []
    _batch_processor: ClassVar[AsyncBatchAdd] = None  # 批量处理器

    # chat_cache_lock: ClassVar[Lock] = Lock()

    def __init__(self, role, content: str, user, name=None, robot_id=None, model=None, agent=None, index: int = 0,
                 reference=None, transform=None, timestamp: int = 0):
        self.role = role or 'user'
        self.content = content or ''
        self.name = name

        self.user = user
        self.robot_id = robot_id
        self.model = model
        self.agent = agent
        self.index = index or 0
        self.reference = reference
        self.transform = transform
        self.timestamp = timestamp or int(time.time())
        # self.content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest() 由数据库自动生成
        # datetime.utcfromtimestamp(timestamp) datetime.utcnow().timestamp()

    @classmethod
    async def initialize_processor(cls, get_session_func):
        """初始化批量处理器"""
        if cls._batch_processor is None:
            cls._batch_processor = AsyncBatchAdd(
                model_class=cls,
                batch_size=1000,
                batch_timeout=3.0,
                get_session_func=get_session_func
            )
            await cls._batch_processor.initialize()

    @classmethod
    async def shutdown_processor(cls):
        """关闭批量处理器"""
        if cls._batch_processor:
            await cls._batch_processor.shutdown()
            cls._batch_processor = None

    @classmethod
    async def put_nowait(cls, history_data: dict | list[dict]) -> int:
        """将历史记录放入队列（非阻塞）"""
        if cls._batch_processor is None:
            raise RuntimeError("Batch processor not initialized. Call initialize_processor first.")
        if isinstance(history_data, dict):
            await cls._batch_processor.put_nowait(history_data)
            return 1
        elif isinstance(history_data, list):
            return cls._batch_processor.put_many_nowait(history_data)
        return 0

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
    async def async_history_insert(cls, new_history: list[dict], session: AsyncSession):
        """直接插入历史记录（不使用队列）"""
        try:
            if len(new_history) > 0:
                session.add_all([cls(**msg) for msg in new_history])
                await session.commit()
        except Exception as e:
            await session.rollback()
            print(f"Error async inserting history: {e},{new_history}")

    @classmethod
    def history_save(cls, new_history: list[dict], user: str, db: Session = None):
        # 保存聊天记录到数据库，或者保存到内存中当数据库不可用时。
        try:
            if user and db:
                cls.history_insert(new_history, db)
            else:
                raise Exception
        except:
            cls.Chat_History_Cache.extend(new_history)

    @classmethod
    def user_history(cls, db: Session, user: str, name: str = None, robot_id: str = None, agent: str = None,
                     filter_time: float = 0, all_payload: bool = True):
        query = db.query(cls).filter(
            cls.user == user,
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
    async def async_user_history(cls, session: AsyncSession, user: str, name: str = None, robot_id: str = None,
                                 agent: str = None, filter_time: float = 0, all_payload: bool = True):
        """查询用户历史记录"""
        stmt = select(cls).where(
            cls.user == user,
            or_(cls.name == name, name is None),
            or_(cls.robot_id == robot_id, robot_id is None)
        )
        if filter_time > 0:
            stmt = stmt.where(cls.timestamp >= filter_time)
        if agent:
            stmt = stmt.where(cls.agent == agent)

        result = await session.execute(stmt)
        records = result.scalars().all()

        if all_payload:
            return [record.asdict() for record in records]
        return [{'role': record.role, 'content': record.content, 'name': record.name} for record in records]

    @classmethod
    async def update(cls, history: List[dict]):
        if not history:
            return 0
        if cls._batch_processor is None:
            raise RuntimeError("Batch processor not initialized. Call initialize_processor first.")
        # UNIQUE(user, name, robot_id, role, agent, index, timestamp, content_hash)
        insert_stmt = text(f"""
            INSERT INTO {cls.__tablename__}
            (user, name, agent, role, created_at, timestamp, model, content, reference, robot_id, `index`)
            VALUES
            (:user, :name, :agent, :role, :created_at, :timestamp, :model, :content, :reference, :robot_id, :index)
            ON DUPLICATE KEY UPDATE
                content   = VALUES(content),
                reference = VALUES(reference),
                `index` = VALUES(`index`)
        """)

        now_ts = int(time.time())
        # 填充默认值，保证 SQL 参数完整
        for i, rec in enumerate(history):
            rec.setdefault("index", i)
            rec.setdefault("robot_id", None)
            rec.setdefault("model", None)
            rec.setdefault("timestamp", now_ts)
            rec.setdefault("created_at", datetime.fromtimestamp(rec["timestamp"]))
            rec.setdefault("reference", None)
            # rec['content_hash'] = hashlib.sha256(rec['content'].encode('utf-8')).hexdigest()
            if isinstance(rec['reference'], (dict, list)):
                rec['reference'] = json.dumps(rec['reference'], ensure_ascii=False, default=str)

        return await cls._batch_processor.execute_batch(insert_stmt, history)

    @classmethod
    def sequential_insert(cls, db: Session, user=None, robot_id=None, agent=None):
        i = 0
        while i < len(cls.Chat_History_Cache):
            try:
                msg = cls.Chat_History_Cache[i]
                if ((not agent or msg['agent'] == agent) and
                        (not user or msg['user'] == user) and
                        (not robot_id or msg['robot_id'] == robot_id)):

                    record = cls(**msg)
                    db.add(record)
                    db.commit()
                    del cls.Chat_History_Cache[i]
                    if not any(
                            (not agent or m['agent'] == agent) and
                            (not user or msg['user'] == user) and
                            (not robot_id or m['robot_id'] == robot_id)
                            for m in cls.Chat_History_Cache[i:]):
                        break
                else:
                    i += 1
            except Exception as e:
                db.rollback()
                print(f"Error inserting chat history: {e}")
                break

    @classmethod
    def flush_cache(cls, user=None):
        """一次性把类级缓存落库"""
        if len(cls.Chat_History_Cache):
            session = SessionLocal()
            try:
                cls.sequential_insert(session, user=user)
            finally:
                session.close()

    def save_cache(self):
        """保存当前实例缓存到数据库"""
        with SessionLocal() as session:
            self.sequential_insert(session, user=self.user, robot_id=self.robot_id)

    def get_cache(self, filter_time: float = 0):
        user_history = [msg for msg in self.Chat_History_Cache
                        if msg['user'] == self.user
                        and (not self.name or msg['name'] == self.name)
                        and (not self.robot_id or msg['robot_id'] == self.robot_id)
                        and (not self.agent or msg['agent'] == self.agent)
                        and msg['timestamp'] >= filter_time]
        return user_history


def get_user_history(user: str, name: Optional[str], robot_id: Optional[str], filter_time: float, db: Session,
                     agent: str = None, request_uid: Optional[str] = None) -> List[dict]:
    user_history = BaseChatHistory(None, None, user=user or request_uid, name=name, robot_id=robot_id,
                                   agent=agent).get_cache(filter_time)
    if user and db:  # 从数据库中补充历史记录
        user_history.extend(
            BaseChatHistory.user_history(
                db, user, name, robot_id, agent=agent, filter_time=filter_time, all_payload=True))

    return user_history


def build_chat_history(user_request: str, user: str, name: Optional[str], robot_id: Optional[str],
                       db: Session, user_history: List[dict] = None, use_hist=False,
                       filter_limit: int = -500, filter_time: float = 0,
                       agent: Optional[str] = None, request_uid: Optional[str] = None):
    # 构建用户的聊天历史记录，并生成当前的用户消息。
    history = []
    if not user_history:
        if use_hist:
            user_history = get_user_history(user, name, robot_id, filter_time, db, agent=agent,
                                            request_uid=request_uid)
            last_records = cut_chat_history(sorted(user_history, key=lambda x: x['timestamp']),
                                            max_size=filter_limit, max_pairs=-filter_limit,
                                            model_name=Config.DEFAULT_MODEL_ENCODING)
            history.extend(
                [{'role': msg['role'], 'content': msg['content'], 'name': msg['name']} for msg in last_records])

        history.append({'role': 'user', 'content': user_request, 'name': name})
    else:
        last_records = cut_chat_history(user_history, max_size=filter_limit, max_pairs=-filter_limit,
                                        model_name=Config.DEFAULT_MODEL_ENCODING)
        history.extend([msg.dict() for msg in last_records])
        if not user_request:
            if history[-1]["role"] == 'user':
                user_request = history[-1]["content"]

    return history, user_request, len(user_history)


def save_chat_history(user_request: str, bot_response: str,
                      user: str, name: Optional[str], robot_id: Optional[str],
                      agent: Optional[str], hist_size: int, model: str, timestamp: float,
                      db: Session, refer: List[str], transform=None, request_uid: Optional[str] = None):
    if not user_request or not bot_response:
        return
    uid = user or request_uid
    if not uid:
        return
    new_history = [
        {'role': 'user', 'content': user_request, 'name': name, 'robot_id': robot_id, 'user': uid,
         'agent': agent, 'index': hist_size + 1, 'timestamp': timestamp},
        {'role': 'assistant', 'content': bot_response, 'name': name, 'robot_id': robot_id, 'user': uid,
         'agent': agent, 'index': hist_size + 2, 'model': model,  # 'timestamp': time.time(),
         'reference': json.dumps(refer, ensure_ascii=False) if refer else None,  # '\n'.join(refer)
         'transform': json.dumps(transform, ensure_ascii=False) if transform else None}
    ]
    # 保存聊天记录到数据库，或者保存到内存中当数据库不可用时。
    BaseChatHistory.history_save(new_history, user, db)


class ChatHistory(BaseChatHistory):
    def __init__(self, user: str, name: Optional[str], robot_id: Optional[str],
                 agent: Optional[str], model: Optional[str], timestamp: float,
                 index: int = 0, request_uid: Optional[str] = None, **kwargs):
        self.user_request = kwargs.get("user_request", '')
        self.user_history: List[dict] = kwargs.get('user_history', [])
        super().__init__(
            role='user',
            content=self.user_request,
            name=name,
            user=user,
            robot_id=robot_id,
            model=model,
            agent=agent,
            index=index,
            timestamp=int(timestamp)
        )
        self.uid = user or request_uid

    async def build(self, user_request: str, user_messages: List[dict | BaseModel] = None, use_hist=False,
                    filter_limit: int = -500, filter_time: float = 0, session: AsyncSession = None):
        history = []
        if not user_messages:
            if use_hist and self.user:
                self.user_history = []  # self.get_cache(filter_time)
                if session:  # 从数据库中补充历史记录
                    self.user_history = await BaseChatHistory.async_user_history(
                        session, self.user, self.name, self.robot_id, agent=self.agent, filter_time=filter_time,
                        all_payload=True)
                self.index = max((msg.get("index", 0) for msg in self.user_history), default=self.index)
                message_records = cut_chat_history(sorted(self.user_history, key=lambda x: x['timestamp']),
                                                   max_size=filter_limit, max_pairs=-filter_limit,
                                                   model_name=Config.DEFAULT_MODEL_ENCODING)
                history.extend(
                    [{'role': msg['role'], 'content': msg['content'], 'name': msg['name']} for msg in message_records])

            if user_request:
                self.user_request = user_request
                history.append({'role': 'user', 'content': user_request, 'name': self.name})
        else:
            if isinstance(user_messages[0], BaseModel):  # from structs import ChatMessage
                self.user_history = [msg.model_dump() for msg in user_messages]
            else:
                self.user_history = user_messages.copy()
            self.index = max(self.index, len(self.user_history))
            message_records = cut_chat_history(self.user_history, max_size=filter_limit, max_pairs=-filter_limit,
                                               model_name=Config.DEFAULT_MODEL_ENCODING)
            history.extend(message_records)

            if not self.name:
                self.name = next((msg.get("name") for msg in reversed(history) if msg.get("role") == "user"), None)
            if not self.uid:
                self.uid = next((msg.get("name") for msg in reversed(history) if msg.get("role") == "assistant"), None)

            if not user_request:
                if history and history[-1]["role"] == 'user':
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

    async def save(self, bot_response: str, refer: Union[List, Dict] = None, transform=None,
                   user_request: str = None, model: str = None, session: AsyncSession = None):
        """
        为了方便成对取出记录，'user','robot_id','name','agent'保持一致并成对保存
        模型可以得到多条回复，为了方便存储，暂时只记录一条
        user 建议用唯一标识,如果未提供,为匿名用户,不保存数据库
        :param bot_response:
        :param refer:
        :param transform:
        :param user_request:
        :param model:
        :param session:
        :return:
        """
        user_request = user_request or self.user_request
        if not user_request or not bot_response:
            print('no content to save')
            return
        if not self.uid:
            return

        current_max = max(self.index, len(self.user_history))
        new_history = [
            {'role': 'user', 'content': user_request, 'name': self.name,
             'user': self.uid, 'robot_id': self.robot_id, 'agent': self.agent,
             'index': current_max + 1, 'model': self.model or model, 'timestamp': self.timestamp
             },
            {'role': 'assistant', 'content': bot_response, 'name': self.name,
             'user': self.uid, 'robot_id': self.robot_id, 'agent': self.agent,
             'index': current_max + 2, 'model': model or self.model,
             'reference': json.dumps(refer, ensure_ascii=False, default=str) if refer else None,  # '\n'.join(refer)
             'transform': json.dumps(transform, ensure_ascii=False, default=str) if transform else None
             # 'timestamp': time.time(),
             }
        ]

        if self.user:
            self.index = current_max + 2  # index 单调递增
            if session:
                await BaseChatHistory.async_history_insert(new_history, session)
            else:
                await self.put_nowait(new_history)


class BaseReBot(Base):
    __tablename__ = 'agent_robot'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    user_content: Mapped[str] = mapped_column(MEDIUMTEXT, nullable=True)  # "prompt"
    assistant_content: Mapped[str] = mapped_column(MEDIUMTEXT, nullable=True)  # "completion"
    system_content: Mapped[str] = mapped_column(TEXT, nullable=True)
    reasoning_content: Mapped[Optional[str]] = mapped_column(TEXT, nullable=True)

    agent: Mapped[str] = mapped_column(String(50), nullable=True, default='chat')  # chat_type

    model: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    name: Mapped[Optional[str]] = mapped_column(String(99), nullable=True, index=True)
    user: Mapped[str] = mapped_column(String(99), nullable=True, index=True)
    robot_id: Mapped[Optional[str]] = mapped_column(String(99), nullable=True, index=True)
    msg_id: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    reference: Mapped[Optional[str]] = mapped_column(MEDIUMTEXT, nullable=True)
    transform: Mapped[Optional[str]] = mapped_column(TEXT, nullable=True)

    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=True, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=True, default=0)

    timestamp: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), onupdate=func.utc_timestamp(),
                                                 default=func.utc_timestamp())

    __table_args__ = (
        Index('idx_name_time', 'name', 'user', 'robot_id', 'timestamp'),
    )

    def __init__(self, user_content=None, assistant_content=None, system_content=None, agent='chat',
                 name: str = None, user: str = None, robot_id: str = None, model: str = None, **kwargs):
        self.user_content = user_content
        self.assistant_content = assistant_content
        self.system_content = system_content
        self.agent = agent

        self.name = name
        if user and ':' in user:
            parts = user.split(':', 1)
            self.user, self.robot_id = parts[0], robot_id or parts[1]
        else:
            self.user = user
            self.robot_id = robot_id

        self.model = model
        self.msg_id = kwargs.get('msg_id', 0)

        self.reference = kwargs.get('reference', None)  # 参考信息,rag,tools,context
        self.transform = kwargs.get('transform', None)
        self.reasoning_content = kwargs.get('reasoning_content', None)  # 背景信息,intent,analysis,summary,title

        self.prompt_tokens = kwargs.get('prompt_tokens', 0)
        self.completion_tokens = kwargs.get('completion_tokens', 0)
        self.timestamp = kwargs.get('timestamp', int(time.time()))
        self.created_at = kwargs.get('created_at', datetime.now(timezone.utc))
        self.updated_at = kwargs.get('updated_at', datetime.now(timezone.utc))
        # datetime.utcfromtimestamp(timestamp) datetime.utcnow().timestamp()

    def asdict(self) -> dict:
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

    def set_data(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    @classmethod
    def copy(cls, data: dict, instance: Optional["BaseReBot"] = None):
        """
           从数据字典和可选的类实例中构造或更新一个类对象。
           data 合并用户提交的字典数据,此处优先覆盖
           instance: 已有实例，用于合并数据或在其上更新
        """
        if isinstance(instance, cls):
            data = {**{k: v for k, v in instance.asdict().items() if v}, **data}
        final_data = {k: v for k, v in data.items() if v is not None}
        if isinstance(instance, cls):
            instance.set_data(**final_data)
            return instance

        return cls(**final_data)  # 否则创建一个新实例返回

    def display(self):
        print(f"User: {self.user_content}, Assistant: {self.assistant_content}, Timestamp: {self.timestamp}")

    async def insert(self, session: AsyncSession):
        try:
            session.add(self)
            await session.commit()
            await session.refresh(self)
            return True
        except Exception as e:
            await session.rollback()
            print(f"[Insert Error]: {e}")
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

    @staticmethod
    def before(messages: list[dict | BaseModel]):
        data = {}
        if messages and isinstance(messages, list):
            if isinstance(messages[0], BaseModel):
                messages = [msg.model_dump() for msg in messages]

            last_user_msg = next((msg for msg in reversed(messages) if msg.get("role") == "user"), None)
            data['msg_id'] = len(messages)
            if last_user_msg:
                data["name"] = last_user_msg.get("name")
                data["user_content"] = BaseMysql.format_value(last_user_msg.get("content"))
            if not data.get("robot_id"):
                last_assistant_msg = next((msg for msg in reversed(messages) if msg.get("role") == "assistant"), None)
                if last_assistant_msg and "name" in last_assistant_msg:
                    data["robot_id"] = last_assistant_msg.get("name")
            if not data.get("system_content"):
                system_content = next((msg.get("content") for msg in messages if msg.get("role") == "system"), None)
                if system_content and isinstance(system_content, str):
                    data["system_content"] = system_content
            if not data.get("reference"):
                reference = next((msg.get("content") for msg in reversed(messages) if msg.get("role") == "system"),
                                 None)  # [{"type": "text", "text": str(text)} for text in refer]
                if reference and isinstance(reference, (list, dict, set, tuple)):
                    data["reference"] = BaseMysql.format_value(reference)
        return data

    @staticmethod
    def after(model_response: dict | str | BaseModel, assistant_content: str = ''):
        if not model_response:
            return {}

        data = {}
        if isinstance(model_response, BaseModel):
            model_response = model_response.model_dump()
        if isinstance(model_response, str):
            try:
                model_response = json.loads(model_response)
            except json.JSONDecodeError:
                print("Warning: invalid JSON chunk:", model_response.encode("utf-8"))

        if isinstance(model_response, dict):
            choice = model_response.get('choices', [{}])
            if choice:
                choice = choice[0]
                data["assistant_content"] = assistant_content or choice.get('message', {}).get('content') or choice.get(
                    'text')
                if not data["assistant_content"]:
                    tool_calls = choice.get('message', {}).get('tool_calls', [])
                    if tool_calls:
                        data["assistant_content"] = BaseMysql.format_value(tool_calls)

                reasoning_content = choice.get('message', {}).get('reasoning_content')
                if reasoning_content:
                    data["reasoning_content"] = reasoning_content

            data["model"] = model_response.get('model') or data.get("model", 'unknown')
            data["timestamp"] = model_response.get("created") or data.get("timestamp", int(time.time()))

            usage = model_response.get('usage', {}) or {}
            data["prompt_tokens"] = usage.get('prompt_tokens', 0)
            data["completion_tokens"] = usage.get('completion_tokens', 0)

        return data

    @classmethod
    def build(cls, user: str = None, messages: list[dict | BaseModel] = None,
              model_response: dict | str | BaseModel = None,
              instance: Optional["BaseReBot"] = None, **kwargs) -> dict:
        """自动从 messages/model_response 中提取,排除空值"""
        if user and ':' in user:
            user, robot_id = user.split(':', 1)
        else:
            robot_id = kwargs.pop('robot_id', None)

        if isinstance(instance, cls):
            instance.set_data(user=user, robot_id=robot_id, **kwargs)
            data = instance.asdict()
        else:
            data = cls(user=user, robot_id=robot_id, **kwargs).asdict()

        if messages:
            data.update(cls.before(messages=messages))

        if model_response:
            data.update(cls.after(model_response=model_response, assistant_content=data.get('assistant_content', '')))

        final_data = {k: v for k, v in data.items() if v is not None}
        return final_data

    @classmethod
    async def save(cls, user=None, instance=None, messages: list[dict] = None, model_response: dict = None,
                   session: AsyncSession = None, **kwargs):
        """自动从 instance 或 messages/model_response 中提取并保存数据库"""

        data = cls.build(user=user, messages=messages, model_response=model_response, **kwargs)
        instance = cls.copy(data, instance)
        if session:
            await instance.insert(session)
        else:
            with AsyncSessionLocal() as session:
                await instance.insert(session)
        return instance

    @classmethod
    async def async_save(cls, data: dict = None, dbpool: AsyncMysql = None, row_id: int = None, **kwargs):
        if not data:
            data = cls.build(**kwargs)
        else:
            data = {**kwargs, **data}  # 如果 kwargs 和 data 有重复字段，以 data 为准

        data["updated_at"] = datetime.now(timezone.utc)
        if dbpool:
            if row_id and row_id > 0:
                return await dbpool.async_update(table_name=cls.__tablename__, params_data=data, row_id=row_id)
            if hasattr(dbpool, "enqueue_nowait"):  # CollectorMysql
                ok = dbpool.enqueue_nowait(table_name=cls.__tablename__, params_data=data, update_fields=[])
                if ok:
                    return True  # 入队成功，不返回 lastrowid
                # 队列满 fallback,默认 insert,lastrowid
            return await dbpool.async_insert(table_name=cls.__tablename__, params_data=data)

        else:
            # async with MysqlData(async_mode=True) as dbop:
            with OperationMysql() as session:
                return session.insert(table_name=cls.__tablename__, params=data)

    @classmethod
    async def history(cls, user, robot_id, name, system_content=None, user_content=None, agent='chat', filter_day=None,
                      limit: int = 10, dbpool: OperationMysql = None):
        """组装hsistory"""
        if not filter_day:
            filter_day = datetime.today().strftime('%Y-%m-%d')
        sql = f"""
             SELECT user_content, assistant_content, system_content, reference, transform, reasoning_content, robot_id, name
             FROM {cls.__tablename__}
             WHERE created_at >= %s
                AND agent = %s 
                AND user = %s 
                AND robot_id = %s
                AND name = %s 
             ORDER BY timestamp ASC LIMIT {limit}
         """
        params = (filter_day, agent, user, robot_id, name)
        result = []
        if dbpool:
            result = await dbpool.async_run(sql, params)
        else:
            with OperationMysql() as session:
                result = session.search(sql, params)

        history = []

        for item in result:
            system = item.get('system_content') or item.get('reasoning_content')  # 系统提示
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


# from agents.ai_tasks import TaskStatus
#
#
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

    with OperationMysql() as db:
        print(db.search("SELECT * FROM agent_users"))
