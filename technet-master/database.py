from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, TIMESTAMP, ForeignKey, func, text,create_engine
from sqlalchemy.orm import Mapped, mapped_column, relationship
from datetime import datetime

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
    user_id: Mapped[int] = db.Column(db.Integer, unique=True, nullable=False)
    username: Mapped[str] = db.Column(db.String(99), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)
    #role_id = db.Column(db.Integer, db.ForeignKey('roles.id'))
    created_at: Mapped[datetime] = db.Column(db.DateTime(timezone=True), default=func.now())
    updated_at = mapped_column(TIMESTAMP, onupdate=func.utc_timestamp(), default=func.utc_timestamp())
    cross = mapped_column(db.Boolean)

    def __repr__(self):
        return '<User %r>' % self.username


class UserLog(db.Model):
    id = Column(db.Integer, primary_key=True)
    # username: Mapped[str] = db.Column(db.String(99), nullable=False)
    user_id: Mapped[int] = mapped_column(db.Integer, ForeignKey('users.user_id'))
    word: Mapped[str] = db.Column(db.String(255), nullable=False)
    updated_at = db.Column(TIMESTAMP, onupdate=func.utc_timestamp(), default=func.utc_timestamp())
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


# def fetch_by_ids(engine,table_name, id_column, ids,columns=None):
# 	column_name = '`'+'`,`'.join(columns)+'`' if isinstance(columns,list) else columns if columns else '*'
# 	placeholders = ', '.join(['%s'] * len(ids))
# 	query = f"SELECT {column_name} FROM {table_name} WHERE {id_column} IN ({placeholders})"
# 	return pd.read_sql(query, engine, params=tuple(ids))#\'in ({});'.format(','.join(.astype(str).to_list()))


if __name__ == "__main__":
    # # 删除所有表
    # db.drop_all()
    # # 创建所有表
    # db.create_all()
    # with app.open_resource('schema.sql') as f:
    #     db.executescript(f.read().decode('utf8'))
    pass
