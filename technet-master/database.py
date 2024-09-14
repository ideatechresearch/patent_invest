from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, TIMESTAMP, ForeignKey, func, or_, text, create_engine
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.mysql import MEDIUMTEXT
from datetime import datetime
import time

db = SQLAlchemy()


class Base(db.Model):
    __abstract__ = True
    created = Column(TIMESTAMP, server_default=func.now())
    status = Column(db.SmallInteger, default=1)

    # 动态赋值
    def set_attrs(self, attrs_dict):
        for key, value in attrs_dict.items():
            if hasattr(self, key) and key != 'id':  # 当前对象是否包含名字为k的属性
                setattr(self, key, value)


# class Role(db.Model):
#     # 定义表名
#     __tablename__ = 'roles'
#     # 定义字段
#     id = db.Column(db.Integer, primary_key=True,autoincrement=True)
#     name = db.Column(db.String(64), unique=True)
#     users = db.relationship('User',backref='role') # 反推与role关联的多个User模型对象

class User(db.Model):
    __tablename__ = 'users'
    id: Mapped[int] = db.Column(db.Integer, primary_key=True)
    user_id: Mapped[int] = db.Column(db.Integer, unique=True, nullable=False, index=True)
    username: Mapped[str] = mapped_column(db.String(99), unique=True, nullable=False, index=True)
    password: Mapped[str] = mapped_column(db.String(128), nullable=False)
    # role_id = db.Column(db.Integer, db.ForeignKey('roles.id'))
    created_at: Mapped[datetime] = mapped_column(db.DateTime(timezone=True), default=func.now())
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP, onupdate=func.utc_timestamp(), default=func.utc_timestamp())
    chatcut_at: Mapped[int] = mapped_column(db.BigInteger, nullable=True)
    cross = mapped_column(db.Boolean)

    def __repr__(self):
        return '<User %r>' % self.username


class UserLog(db.Model):
    id = Column(db.Integer, primary_key=True)
    # username: Mapped[str] = db.Column(db.String(99), nullable=False)
    user_id: Mapped[int] = mapped_column(db.Integer, ForeignKey('users.user_id'))
    word: Mapped[str] = mapped_column(db.String(255), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP, onupdate=func.utc_timestamp(), default=func.utc_timestamp())
    stoped = db.Column(db.Boolean, default=False)
    action = db.Column(db.String(255))


class StopWords(db.Model):
    id = Column(db.Integer, primary_key=True)
    word: Mapped[str] = db.Column(db.String(255), unique=True, nullable=False)
    read_flag: Mapped[int] = db.Column(db.BigInteger, default=0)
    stop_flag: Mapped[int] = db.Column(db.BigInteger, default=0)
    updated_at = mapped_column(TIMESTAMP, onupdate=func.utc_timestamp(), default=func.utc_timestamp())

    def __repr__(self):
        return '<Word %r>' % self.word

    # create_table_sql = '''
    # CREATE TABLE IF NOT EXISTS user (
    #     id INTEGER PRIMARY KEY AUTOINCREMENT,
    #     username VARCHAR(40) UNIQUE NOT NULL,
    #     password VARCHAR(120) NOT NULL,
    # );
    # '''
    # insert_user_sql = '''
    #             INSERT INTO user (username, password)
    #             VALUES (?, ?);
    #             ''

    # query_user_sql = '''
    #         SELECT COUNT(*)
    #         FROM user
    #         WHERE username = ?
    #         AND password = ?
    #         '''


class ChatHistory(db.Model):
    __tablename__ = 'chat_history'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    role = db.Column(db.String(20), nullable=False)  # 'user' 或 'assistant'
    content = db.Column(MEDIUMTEXT, nullable=False)
    username = db.Column(db.String(99), nullable=False, index=True)
    model = db.Column(db.String(50), nullable=True, default='moonshot')  # 模型名称
    agent = db.Column(db.String(50), nullable=False, default='0', index=True)  # 代理名称或角色
    index = db.Column(db.Integer, nullable=False)  # 消息索引
    reference = db.Column(MEDIUMTEXT, nullable=True)  # 参考信息，仅对'assistant'有用 db.Text
    timestamp = db.Column(db.BigInteger, nullable=False, index=True)  # 消息 Unix 时间戳
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)  # 消息创建时间

    __table_args__ = (
        db.Index('idx_agent_username_time', 'agent', 'username', 'timestamp'),
    )

    def __init__(self, role, content, username, model='moonshot', agent='0', index=0, reference=None, timestamp=0):
        self.role = role
        self.content = content
        self.username = username
        self.model = model
        self.agent = agent
        self.index = index
        self.reference = reference
        self.timestamp = timestamp or time.time()

    def asdict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}  # vars()

    @classmethod
    def history_insert(cls, new_history, session):
        if len(new_history) > 0:
            session.add_all([cls(**msg) for msg in new_history])
            session.commit()

    @classmethod
    def user_history(cls, user_name, agent=None, filter_time=0, all_payload=True):
        query = cls.query.filter(or_(cls.agent == agent, agent is None),
                                 cls.username == user_name, cls.timestamp > filter_time)
        if all_payload:
            return [record.asdict() for record in query.all()]
        records = query.with_entities(cls.role, cls.content).all()
        return [{'role': record.role, 'content': record.content} for record in records]

    @classmethod
    def sequential_insert(cls, chat_history, session, user_name=None, agent=None):
        i = 0
        while i < len(chat_history):
            try:
                msg = chat_history[i]
                if (not agent or msg['agent'] == agent) and (not user_name or msg['username'] == user_name):
                    record = cls(**msg)
                    session.add(record)
                    session.commit()

                    del chat_history[i]  # pop(0)
                    if not any((not agent or m['agent'] == agent) and (not user_name or m['username'] == user_name)
                               for m in chat_history[i:]):
                        break
                else:
                    i += 1

            except Exception as e:
                session.rollback()
                print(f"Error inserting chat history: {e}")
                break


# def fetch_by_ids(engine,table_name, id_column, ids,columns=None):
# 	column_name = '`'+'`,`'.join(columns)+'`' if isinstance(columns,list) else columns if columns else '*'
# 	placeholders = ', '.join(['%s'] * len(ids))
# 	query = f"SELECT {column_name} FROM {table_name} WHERE {id_column} IN ({placeholders})"
# 	return pd.read_sql(query, engine, params=tuple(ids))#\'in ({});'.format(','.join(.astype(str).to_list()))

import pymysql


class OperationMysql:
    def __init__(self, host, user, password, db_name, port=3306):
        self.conn = pymysql.connect(
            host=host,  # 外网
            port=port,
            user=user,
            password=password,
            db=db_name,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor  # 这个定义使数据库里查出来的值为字典类型
        )
        self.cur = self.conn.cursor()

    def search(self, sql):
        self.cur.execute(sql)
        result = self.cur.fetchall()
        # result=json.dumps(result,cls=DateEncoder) #把字典转成字符串,方便与处理后的返回结果进行对比
        # result=json.dumps(result,cls=DecimalEncoder) #Decimal类型转成json
        self.cur.close()
        self.conn.close()
        return result

    def create_table(self, sql):
        self.cur.execute(sql)

    def update_table(self, sql):
        self.cur.execute(sql)
        self.conn.commit()
        self.cur.close()
        self.conn.close()

    def delete_data(self, sql):
        self.cur.execute(sql)
        self.conn.commit()
        self.cur.close()
        self.conn.close()

    def insert_table(self, sql):
        try:
            self.cur.execute(sql)
            # 提交
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
        finally:
            self.conn.close()


if __name__ == "__main__":
    # # 删除所有表
    # db.drop_all()
    # # 创建所有表
    # db.create_all()
    # with app.open_resource('schema.sql') as f:
    #     db.executescript(f.read().decode('utf8'))
    pass
