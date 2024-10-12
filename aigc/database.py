from sqlalchemy import create_engine, Column, ForeignKey, String, Integer, BigInteger, Float, DateTime, Index, TEXT
from sqlalchemy import func, or_, and_, text
from sqlalchemy.dialects.mysql import MEDIUMTEXT
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, mapped_column, Mapped, Session
from datetime import datetime
import pymysql
import hashlib
from typing import List, Optional
import json, time

Base = declarative_base()
Chat_history = []


class User(Base):
    __tablename__ = 'agent_users'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    uuid: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, unique=True)
    username: Mapped[str] = mapped_column(String(99), unique=True, nullable=False, index=True)
    password: Mapped[str] = mapped_column(String(128), nullable=True)
    eth_address: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, unique=True)
    public_key: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, unique=True)

    group: Mapped[Optional[str]] = mapped_column(String(20), nullable=True, default='0')
    role: Mapped[str] = mapped_column(String(50), nullable=True, default='user')

    parent_id: Mapped[Optional[int]] = mapped_column(ForeignKey('agent_users.id'), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), onupdate=func.utc_timestamp(),
                                                 default=func.utc_timestamp())

    chatcut_at: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True, default=0)
    disabled_at: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)

    def __repr__(self):
        return f"<User {self.username}>"

    def is_primary_account(self):
        return self.parent_id is None

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
    def get_user(cls, db: Session, username: Optional[str] = None, uuid: Optional[str] = None,
                 public_key: Optional[str] = None, eth_address: Optional[str] = None):
        query = db.query(cls)
        if username:
            return query.filter_by(username=username).first()
        if uuid:
            return query.filter_by(uuid=uuid).first()
        if public_key:
            return query.filter_by(public_key=public_key).first()
        if eth_address:
            return query.filter_by(eth_address=eth_address).first()
        return None

    @classmethod
    def update_user(cls, user_id: int, eth_address: Optional[str] = None, public_key: Optional[str] = None, **kwargs):
        update_data = {}
        if eth_address is not None:
            update_data['eth_address'] = eth_address
        if public_key is not None:
            update_data['public_key'] = public_key

        for key, value in kwargs.items():
            if hasattr(cls, key):
                update_data[key] = value

        if update_data:
            db.query(cls).filter_by(id=user_id).update(update_data)
            db.commit()

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
        if user and cls.verify_password(password, user.password):
            return user

        return None


class ChatHistory(Base):
    __tablename__ = 'agent_history'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # 'user' 或 'assistant'
    content: Mapped[str] = mapped_column(MEDIUMTEXT, nullable=False)
    username: Mapped[str] = mapped_column(String(99), nullable=False, index=True)

    user_id: Mapped[str] = mapped_column(String(99), nullable=True, index=True)
    model: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    agent: Mapped[str] = mapped_column(String(50), nullable=True, default='0', index=True)

    index: Mapped[int] = mapped_column(Integer, nullable=False)
    reference: Mapped[Optional[str]] = mapped_column(MEDIUMTEXT, nullable=True)
    summary: Mapped[Optional[str]] = mapped_column(TEXT, nullable=True)
    transform: Mapped[Optional[str]] = mapped_column(TEXT, nullable=True)
    timestamp: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_agent_username_time', 'agent', 'username', 'timestamp'),
    )

    def __init__(self, role, content, username, user_id=None, model=None, agent='0', index=0,
                 reference=None, summary=None, transform=None, timestamp=0):
        self.role = role
        self.content = content
        self.username = username
        self.user_id = user_id
        self.model = model
        self.agent = agent
        self.index = index
        self.reference = reference
        self.summary = summary
        self.transform = transform
        self.timestamp = timestamp or int(time.time())
        # datetime.utcfromtimestamp(timestamp) datetime.utcnow().timestamp()

    def asdict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

    @classmethod
    def history_insert(cls, new_history, db: Session):
        if len(new_history) > 0:
            db.add_all([cls(**msg) for msg in new_history])
            db.commit()

    @classmethod
    def user_history(cls, db: Session, username: str, user_id: str = None, agent: str = None, filter_time: float = 0,
                     all_payload: bool = True):
        query = db.query(cls).filter(cls.username == username, or_(cls.user_id == user_id, user_id is None),
                                     cls.timestamp > filter_time)
        if agent:  # or_(cls.agent == agent, agent is None)
            query = query.filter(cls.agent == agent)

        if all_payload:
            return [record.asdict() for record in query.all()]
        records = query.with_entities(cls.role, cls.content).all()
        return [{'role': record.role, 'content': record.content} for record in records]

    @classmethod
    def sequential_insert(cls, db: Session, chat_history=Chat_history, username=None, agent=None):
        i = 0
        while i < len(chat_history):
            try:
                msg = chat_history[i]
                if (not agent or msg['agent'] == agent) and (not username or msg['username'] == username):
                    record = cls(**msg)
                    db.add(record)
                    db.commit()
                    del chat_history[i]
                    if not any((not agent or m['agent'] == agent) and (not username or m['username'] == username)
                               for m in chat_history[i:]):
                        break
                else:
                    i += 1
            except Exception as e:
                db.rollback()
                print(f"Error inserting chat history: {e}")
                break


def get_user_history(user_name: str, user_id: Optional[str], filter_time: float, db: Session, agent: str = None,
                     request_uuid: Optional[str] = None) -> List[dict]:
    user_history = [msg for msg in Chat_history if msg['username'] == (user_name or request_uuid)
                    and (not user_id or msg['user_id'] == user_id) and (not agent or msg['agent'] == agent)
                    and msg['timestamp'] > filter_time]

    if user_name and db:  # 从数据库中补充历史记录
        user_history.extend(
            ChatHistory.user_history(db, user_name, user_id, agent=agent, filter_time=filter_time,
                                     all_payload=True))

    return user_history


def build_chat_history(user_name: str, user_message: str, user_id: Optional[str],
                       filter_time: float, db: Session, user_history: List[str], use_hist: bool = False,
                       request_uuid: Optional[str] = None):
    # 构建用户的聊天历史记录，并生成当前的用户消息。
    history = []
    if not user_history:
        if use_hist:  # 如果没有消息提供，过滤现有的聊天记录，user_message为问题
            user_history = get_user_history(user_name, user_id, filter_time, db, agent=None, request_uuid=request_uuid)
            history.extend([{'role': msg['role'], 'content': msg['content']} for msg in
                            sorted(user_history, key=lambda x: x['timestamp'])])

        history.append({'role': 'user', 'content': user_message})
    else:
        history.extend([msg.dict() for msg in user_history])
        if not user_message:
            if history[-1]["role"] == 'user':
                user_message = history[-1]["content"]
        # 如果提供消息,则使用最后一条user content为问题

    return history, user_message


def save_chat_history(user_name: str, user_message: str, bot_response: str, user_id: Optional[str],
                      agent: str, hist_size: int, model_name: str, timestamp: float,
                      db: Session, refer: List[str], transform=None, request_uuid: Optional[str] = None):
    if not user_message or not bot_response:
        return
    username = user_name or request_uuid
    if not username:
        return
    new_history = [
        {'role': 'user', 'content': user_message, 'username': username, 'user_id': user_id,
         'agent': agent, 'index': hist_size - 1, 'timestamp': timestamp},
        {'role': 'assistant', 'content': bot_response, 'username': username, 'user_id': user_id,
         'agent': agent, 'index': hist_size, 'model': model_name,  # 'timestamp': time.time(),
         'reference': json.dumps(refer, ensure_ascii=False) if refer else None,  # '\n'.join(refer)
         'transform': json.dumps(transform, ensure_ascii=False) if transform else None}
    ]
    # 保存聊天记录到数据库，或者保存到内存中当数据库不可用时。
    try:
        if user_name and db:
            ChatHistory.history_insert(new_history, db)
        else:
            raise Exception
    except:
        Chat_history.extend(new_history)


class OperationMysql:
    def __init__(self, host, user, password, db_name, port=3306):
        self.host = host
        self.user = user
        self.password = password
        self.db_name = db_name
        self.port = port
        self.conn = None
        self.cur = None

    def __enter__(self):
        # 打开连接
        self.conn = pymysql.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            db=self.db_name,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor
        )
        self.cur = self.conn.cursor()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 关闭游标和连接
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()

    def search(self, sql):
        self.cur.execute(sql)
        result = self.cur.fetchall()
        return result

    def execute(self, sql):
        try:
            self.cur.execute(sql)
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            print(f"执行 SQL 出错: {e}")
        # finally:
        #     self.cur.close()
        #     self.conn.close()


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
    with OperationMysql(host="localhost", user="root", password="password", db_name="test_db") as db:
        result = db.search("SELECT * FROM my_table")
        print(result)
