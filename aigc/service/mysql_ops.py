from contextlib import asynccontextmanager, contextmanager
from typing import Callable, Optional, Union, AsyncIterator
import pymysql, aiomysql
import json
import pandas as pd

from utils import parse_database_uri, chunks_iterable, records_to_list
from .base import *
from config import Config


class BaseMysql:
    def __init__(self, host: str, user: str, password: str, db_name: str,
                 port: int = 3306, charset: str = "utf8mb4"):
        self.host = host
        self.user = user
        self.password = password
        self.db_name = db_name
        self.port = port
        self.charset = charset

    def close(self):
        """同步实现中关闭连接；子类实现"""
        raise NotImplementedError

    @staticmethod
    @contextmanager
    def get_engine(db_config: dict):
        from sqlalchemy import create_engine
        if db_config:
            url = f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}?charset=utf8mb4"
        else:
            url = Config.SQLALCHEMY_DATABASE_URI
        engine = create_engine(url, pool_pre_ping=True)
        try:
            yield engine
        finally:
            engine.dispose()

    @staticmethod
    @contextmanager
    def get_cursor(db_config: dict):
        conn = pymysql.connect(**db_config)
        try:
            with conn.cursor() as cursor:
                yield cursor
            # conn.commit()
        # except Exception:
        #     conn.rollback()
        #     raise
        finally:
            conn.close()

    def table_schema(self, table_name: str) -> tuple[str, tuple]:
        if not table_name:
            raise ValueError("[Schema] 表名不能为空")
        sql = """
                 SELECT column_name, data_type, is_nullable, column_type, column_comment
                 FROM information_schema.columns
                 WHERE table_schema = %s AND table_name = %s
                 ORDER BY ordinal_position
             """
        params = (self.db_name, table_name)
        return sql, params

    @staticmethod
    def format_value(value) -> Union[str, int, float, bool, None]:
        """
        保留你原来的 format_value 行为（dict->json, list/tuple->\n\n if all str else json, set->; or json）
        """
        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)  # 处理字典类型 → JSON序列化, indent=2

        if isinstance(value, (tuple, list)):
            if all(isinstance(item, str) for item in value):
                return "\n\n---\n\n".join(value)  # "\n\n"
            return json.dumps(value, ensure_ascii=False)  # 非全字符串元素则JSON序列化

        if isinstance(value, set):
            if all(isinstance(item, str) for item in value):
                return ";".join(sorted(value))
            return json.dumps(list(value), ensure_ascii=False)

        return str(value)  # 其他类型保持原样 (None等)

    @staticmethod
    def safe_dataframe(df: pd.DataFrame, dumps: bool = True) -> pd.DataFrame:
        import numpy as np

        df = df.replace({np.nan: None, np.inf: None, -np.inf: None})  # df.where(pd.notnull(df), None)
        if dumps:
            df = df.map(lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v)
        for col in ['created_at', 'updated_at', 'completed_at']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: x.to_pydatetime() if pd.notnull(x) else None)
        return df  # .to_sql(table_name, con=engine, if_exists="append", index=False)

    @staticmethod
    def build_insert(table_name: str, params_data: dict, update_fields: list[str] | None = None,
                     explode: bool = False, with_new: bool = True) -> tuple[str, tuple | list]:
        """
        生成插入 SQL（可选 ON DUPLICATE KEY UPDATE）

        Args:
            table_name: 表名
            params_data: 数据字典 {"col1":[v1,v2], "col2":[v3,v4], ...} .to_dict(orient="list")
            update_fields: 冲突时更新的字段，None 表示默认更新非主键字段
            explode: True -> [(v1,v3), (v2,v4)] 每行 tuple,把字典的每列拆开, False -> ([v1,v2], [v3,v4])
            with_new

        Returns:
            tuple: (sql, values)
        """
        if not params_data:
            raise ValueError("params_data 不能为空")

        fields = list(params_data.keys())
        values = list(zip(*params_data.values())) if explode else tuple(params_data.values())
        # [tuple(row[f] for f in fields) for row in params_data]
        field_str = ', '.join(f"`{field}`" for field in fields)  # columns_str
        placeholder_str = ', '.join(['%s'] * len(fields))
        sql = f"INSERT INTO `{table_name}` ({field_str}) VALUES ({placeholder_str})"

        if update_fields is None:
            update_fields = [f for f in fields if f.lower() not in ("id", "batch_no", "created_at", "created_time")]

        if update_fields:
            if with_new:
                sql += " AS new"
                update_list = [f"`{field}` = new.`{field}`" for field in update_fields if field != "updated_at"]
            else:
                update_list = [f"`{field}` = VALUES(`{field}`)" for field in update_fields if field != "updated_at"]
            update_str = ', '.join(update_list)
            if "updated_at" not in fields and "updated_at" in update_fields:
                update_str += ", `updated_at` = CURRENT_TIMESTAMP"
            sql += f" ON DUPLICATE KEY UPDATE {update_str}"

        return sql, values

    @staticmethod
    def build_insert_dataframe(table_name: str, df: pd.DataFrame, update_fields: list[str] | None = None,
                               with_new: bool = True, dumps: bool = False) -> tuple[str, tuple | list]:
        df = BaseMysql.safe_dataframe(df, dumps)
        params_data = df.to_dict(orient="list")  # 转 dict {列名: [值, 值, 值]} df.values.tolist()
        sql, values = BaseMysql.build_insert(table_name, params_data=params_data, update_fields=update_fields,
                                             explode=True, with_new=with_new)
        return sql, values

    @staticmethod
    def write(db_config: dict, sql: str, params: tuple | dict | list = None, conn=None) -> bool:
        """
        通用 SQL 执行器
        """
        should_release = False
        try:
            if not conn:
                conn = pymysql.connect(**db_config)
                should_release = True
            with conn.cursor() as cursor:
                if isinstance(params, list) and params:
                    cursor.executemany(sql, params)  # 批量执行
                else:
                    cursor.execute(sql, params or ())  # 单条执行
            conn.commit()
            return True
        except Exception as e:
            if conn:
                conn.rollback()
            logging.error(f"[SQL Write ERROR]: {e},{sql}")
            return False
        finally:
            if should_release and conn:
                conn.close()

    @staticmethod
    def read(db_config: dict, sql: str, params: tuple | dict | str = None, conn=None):
        if isinstance(params, str):
            params = (params,)
        should_release = False
        try:
            if not conn:
                conn = pymysql.connect(**db_config)
                should_release = True
            with conn.cursor() as cursor:
                cursor.execute(sql, params or ())
                rows = cursor.fetchall()
                cols = [desc[0] for desc in cursor.description]
                return pd.DataFrame(rows, columns=cols)
        except Exception as e:
            logging.error(f"[SQL Read ERROR]: {e},{sql}")
            with BaseMysql.get_engine(db_config) as engine:
                with engine.connect() as conn:
                    return pd.read_sql(sql, conn, params=params)
        finally:
            if should_release and conn:
                conn.close()

    @staticmethod
    def query_dataframe_process(cursor, sql: str, params: tuple | dict = None, process_chunk: Callable = None,
                                chunk_size: int = 100000):
        """
        通用大表分页查询并处理每块数据。

        Args:
            cursor: 数据库连接对象（需支持 conn.cursor()）
            sql (str): 原始 SQL 查询语句（不含 LIMIT 和 OFFSET）
            process_chunk (Callable): 对每个批次的进行处理的函数
            chunk_size (int): 每批次读取的记录数
            params(tuple|dict|None): SQL 查询参数
        Returns:
            list | pd.DataFrame: 返回所有处理结果的列表
        """
        offset = 0
        results = []
        chunk_query = f"{sql} LIMIT %s OFFSET %s"

        while True:
            if isinstance(params, tuple):
                query_params = (*params, chunk_size, offset)
            elif isinstance(params, dict):
                query_params = {**params, "limit": chunk_size, "offset": offset}
            else:
                query_params = (chunk_size, offset)

            cursor.execute(chunk_query, query_params)
            rows = cursor.fetchall()
            if not rows:
                break

            if process_chunk:
                data = process_chunk(rows)
                if data:  # Only append non-empty results
                    results.append(data)
            else:
                df = pd.DataFrame(rows)
                if not df.empty:
                    results.append(df)

            offset += chunk_size

        if not process_chunk:
            return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

        return results

    @staticmethod
    def query_batches(cursor, ids: list | tuple, index_key: str, table_name: str, fields: list = None,
                      chunk_size: int = 10000, conditions: str = None) -> tuple[list[dict], list]:
        """
        await asyncio.to_thread
        大批量 IN 查询，分批执行，避免 SQL 参数溢出。
        Args:
            cursor:
            ids (list | tuple): 要查找的 ID 列表
            index_key (str): 作为筛选条件的字段
            table_name (str): 表名
            fields (list): 返回的字段
            chunk_size (int): 每次 IN 的最大数量<65535
            conditions
        Returns:
            list[dict]: 查询结果列表
        """
        if not ids:
            raise ValueError("ids 不能为空")

        field_str = ', '.join(f"`{field}`" for field in fields) if fields else '*'

        result_rows = []
        for batch in chunks_iterable(ids, chunk_size):
            placeholders = ', '.join(['%s'] * len(batch))
            sql = f"SELECT {field_str} FROM `{table_name}` WHERE `{index_key}` IN ({placeholders})"
            if conditions:
                sql += f" AND {conditions}"
            cursor.execute(sql, tuple(batch))
            result_rows.extend(cursor.fetchall())
            # filtered_chunk = pd.read_sql_query(text(sql+'`{index_key}` in :ids'), conn, params={'ids': batch})
            # df_chunk.to_sql(table_name, con=engine,chunksize=chunk_size, if_exists='append', index=False)
        cols = [desc[0] for desc in cursor.description]
        return result_rows, cols

    @staticmethod
    def execute_sql(conn, sql, params=None):
        """
        执行给定的SQL语句，返回结果。
        参数：
            - conn： SQLite连接
            - sql：要执行的SQL语句
            - params：SQL语句中的参数
        """
        try:
            # connection.text_factory = bytes
            cursor = conn.cursor()
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            return cursor.fetchall()
        except Exception as e:
            try:
                conn.text_factory = bytes
                cursor = conn.cursor()
                if params:
                    cursor.execute(sql, params)
                else:
                    cursor.execute(sql)
                rdata = cursor.fetchall()
                conn.text_factory = str
                return rdata
            except Exception as e:
                logging.error(f"**********\nSQL: {sql}\nparams: {params}\n{e}\n**********", exc_info=True)
                return None


class SyncMysql(BaseMysql):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.autocommit = bool(kwargs.pop("autocommit", True))
        self.conn: Optional[pymysql.connections.Connection] = None
        self.cur = None

    def __del__(self):
        if self.conn or self.cur:
            print("[OperationMysql] Warning:正在自动清理,关闭数据库连接。")
            self.close()

    def __enter__(self):
        # 打开连接
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 关闭游标和连接
        self.close()

    def connect(self, autocommit: bool = True):
        self.close()
        try:
            self.conn = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                db=self.db_name,
                charset=self.charset,
                cursorclass=pymysql.cursors.DictCursor,  # 这个定义使数据库里查出来的值为字典类型
                autocommit=autocommit,
            )
            self.autocommit = autocommit
            self.cur = self.conn.cursor()  # 原生数据库连接方式
        except Exception as e:
            print(f"[Sync] 连接数据库失败: {e}")
            self.conn = None
            self.cur = None

    def get_conn(self):
        if not self.conn:
            self.connect()
        return self.conn

    def close(self):
        if self.cur:
            self.cur.close()
            self.cur = None
        if self.conn:
            self.conn.close()
            self.conn = None

    def ensure_connection(self):
        try:
            if self.conn:
                self.conn.ping(reconnect=True)  # 已连接且健康
            else:
                self.connect()
        except Exception as e:
            print(f"[Sync] 自动重连失败: {e}")
            self.connect()

    def run(self, sql: str, params: tuple | dict | list = None):
        sql_type = (sql or "").strip().split()[0].lower()
        self.ensure_connection()
        try:
            if isinstance(params, list):
                self.cur.executemany(sql, params)  # 批量执行
            else:
                self.cur.execute(sql, params or ())  # 单条执行

            if sql_type == "select":
                return self.cur.fetchall()

            elif sql_type in {"insert", "update", "delete", "replace"}:
                self.conn.commit()
                if sql_type == "insert":
                    return self.cur.lastrowid
                return True

        except Exception as e:
            if self.conn:
                self.conn.rollback()
            print(f"[Sync] 数据库执行出错: {e}")
        return None

    def search(self, sql: str, params: tuple | dict = None):
        if not sql.lower().startswith("select"):
            raise ValueError("search 方法只能执行 SELECT 语句")
        if params is None:
            params = ()
        self.cur.execute(sql, params)
        result = self.cur.fetchall()
        return result

    def insert(self, sql: str = None, params: tuple | dict = None, table_name: str = None):
        # 单条 INSERT 语句，且目标表有 AUTO_INCREMENT 字段
        if isinstance(params, dict) and table_name:
            fields = tuple(params.keys())
            values = tuple(params.values())
            field_str = ', '.join(f"`{field}`" for field in fields)
            placeholder_str = ', '.join(['%s'] * len(fields))
            sql = f"INSERT INTO `{table_name}` ({field_str}) VALUES ({placeholder_str})"
            params = values
        try:
            self.cur.execute(sql, params or ())
            self.conn.commit()
            return self.cur.lastrowid or int(self.conn.insert_id())
        except Exception as e:
            if self.conn:
                self.conn.rollback()
            print(f"执行 SQL 出错: {e}")
            print(f"SQL: {repr(sql)} \n 参数: {params}")
        # finally:
        #     self.cur.close()
        #     self.conn.close()
        return -1

    def get_table_schema(self, table_name: str) -> list:
        """
        获取指定表的结构信息（原始 implementation），用于自然语言描述
        参数:
            table_name: 表名（不含数据库名）
        返回:
            表结构列表，每列包含 column_name, data_type, is_nullable, column_type, column_comment
        """
        try:
            self.ensure_connection()
            sql, params = self.table_schema(table_name)
            return self.search(sql, params)
        except Exception as e:
            print(f"[Schema] 获取表结构失败: {str(e)}")
        return []


class AsyncMysql(BaseMysql):
    def __init__(self, *args, **kwargs):
        self.minsize = int(kwargs.pop("minsize", 1))
        self.maxsize = int(kwargs.pop("maxsize", 3))
        self.autocommit = bool(kwargs.pop("autocommit", True))
        super().__init__(*args, **kwargs)
        self.pool: Optional[aiomysql.Pool] = None
        self.conn: Optional[aiomysql.Connection] = None  # used only when temporarily acquiring

    async def __aenter__(self):
        await self.init_pool(self.minsize, self.maxsize, self.autocommit)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close_pool()

    async def init_pool(self, minsize: int = 1, maxsize: int = 30, autocommit: bool = True):
        if self.pool is not None:
            if self.minsize == minsize and self.maxsize == maxsize and self.autocommit == autocommit:
                return self.pool
            else:
                await self.close_pool()
        try:
            self.pool = await aiomysql.create_pool(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                db=self.db_name,
                charset=self.charset,
                autocommit=autocommit,
                minsize=minsize,
                maxsize=maxsize
            )
            self.minsize, self.maxsize, self.autocommit = minsize, maxsize, autocommit
            print(f"[Async] 创建连接池: [{self.minsize}-{self.maxsize}]")
        except Exception as e:
            print(f"[Async] 创建连接池失败: {e}")
            self.pool = None
        return self.pool

    async def close_pool(self):
        """关闭连接池和单连接"""
        if self.pool is not None:
            self.pool.close()
            await self.pool.wait_closed()
            self.pool = None
        if self.conn is not None:
            await self.conn.ensure_closed()
            self.conn = None

    async def get_conn(self) -> aiomysql.Connection:
        if self.pool is None and (self.minsize > 1 or self.maxsize > 1):
            await self.init_pool(self.minsize, self.maxsize, self.autocommit)  # 初始化连接池
        if self.pool:
            return await self.pool.acquire()
        if self.conn is None or self.conn.closed:
            self.conn = await aiomysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                db=self.db_name,
                charset=self.charset,
                autocommit=self.autocommit
            )
        return self.conn  # 返回单连接

    def release(self, conn):
        if not conn:
            return

        if self.pool is not None:
            self.pool.release(conn)
        else:
            conn.close()
            if conn == self.conn:
                self.conn = None

    @asynccontextmanager
    async def get_conn_ctx(self) -> AsyncIterator[aiomysql.Connection]:
        if self.pool is None:
            await self.init_pool(self.minsize, self.maxsize, self.autocommit)
        conn = await self.pool.acquire()
        try:
            if not self.autocommit:
                await conn.begin()
            yield conn
            if not self.autocommit:
                await conn.commit()
        except Exception:
            if not self.autocommit:
                await conn.rollback()
            raise
        finally:
            self.pool.release(conn)

    @asynccontextmanager
    async def get_cursor(self, conn=None) -> AsyncIterator[aiomysql.Cursor]:
        """
        获取游标（支持自动释放）自动管理连接
        释放 await cursor.close()
        Args:
            conn: 外部传入的连接对象。如果为None，则自动创建新连接

        Yields:
            aiomysql.Cursor: 数据库游标

        注意：
            - 当conn为None时，会自动创建并最终释放连接
            - 当conn由外部传入时，不会自动释放连接
        """
        should_release = False
        if conn is None:
            conn = await self.get_conn()
            should_release = True
        try:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                yield cursor
        except Exception:
            if not self.autocommit:
                await conn.rollback()
            raise
        finally:
            if should_release:
                if not self.autocommit:
                    await conn.commit()
                self.release(conn)

    async def async_run(self, sql: str, params: tuple | dict | list = None, conn=None):
        async def _run(c):
            sql_type = (sql or "").strip().split()[0].lower()
            async with c.cursor(aiomysql.DictCursor) as cur:
                if isinstance(params, list):
                    await cur.executemany(sql, params)
                else:
                    await cur.execute(sql, params or ())

                if sql_type == "select":
                    return await cur.fetchall()
                elif sql_type in {"insert", "update", "delete", "replace"}:
                    await c.commit()  # 显式保险,autocommit=True
                    if sql_type == "insert":
                        return cur.lastrowid or int(c.insert_id())
                    else:
                        return True

                return None

        try:
            if conn:
                return await _run(conn)
            async with self.get_conn_ctx() as conn:
                return await _run(conn)

        except Exception as e:
            print(f"[Async] SQL执行错误: {e}, SQL={sql}\nVALUE={params}")
        return None

    async def async_execute(self, sql_list: list[tuple[str, tuple | dict | list | None]], conn=None):
        """
        批量执行多条 SQL 并自动提交或回滚（同一个事务）
        :param sql_list: 形如 [(sql1, params1), (sql2, params2), ...]
        :param conn
        """
        if conn:
            should_release = False
        else:
            conn = await self.get_conn()
            should_release = True

        try:
            async with conn.cursor() as cur:
                for sql, params in sql_list:
                    if not sql.strip():
                        continue
                    if isinstance(params, list):
                        await cur.executemany(sql, params)
                    else:
                        await cur.execute(sql, params or ())
                await conn.commit()
                return True

        except Exception as e:
            print(f"[Async] 批量 SQL 执行失败: {e}")
            try:
                await conn.rollback()
            except Exception as rollback_err:
                print(f"[Async] 回滚失败: {rollback_err}")
            return False

        finally:
            if should_release:
                self.release(conn)

    async def async_query(self, query_list: list[tuple[str, tuple | dict]], fetch_all: bool = True,
                          cursor=None) -> list:
        """
        执行多个查询，分别返回结果列表
        :param query_list: [(sql1, params1), (sql2, params2), ...]
        :param fetch_all: True 表示 fetchall，False 表示 fetchone
        :param cursor: 可选外部 cursor
        :return: [result1, result2, ...]
        """

        async def _run(c) -> list:
            results = []
            for sql, params in query_list:
                await c.execute(sql, params or ())
                result = await c.fetchall() if fetch_all else await c.fetchone()
                results.append(result)
            return results

        if cursor:
            return await _run(cursor)
        async with self.get_cursor() as cur:
            return await _run(cur)

    async def query_one(self, sql: str, params: tuple | dict = (), cursor=None) -> dict | None:
        results = await self.async_query([(sql, params)], fetch_all=False, cursor=cursor)
        return results[0] if results else None

    async def async_insert(self, table_name: str, params_data: dict, conn=None):
        """
        插入数据
        Args:
            table_name (str): 表名
            params_data (dict): 要插入的数据（更新必须包含主键/唯一索引字段）
            conn:可选外部传入连接
        """
        sql, values = self.build_insert(table_name, params_data, update_fields=[])
        return await self.async_run(sql, values, conn=conn)

    async def async_merge(self, table_name: str, params_records: list[dict], update_fields: list[str] = None,
                          conn=None):
        """
        批量插入/更新数据,返回true（根据主键或唯一键自动合并）
        update_fields (list): 需要更新的字段列表，默认为除了主键以外的字段,在发生冲突时被更新的字段列表,[]为插入
        """
        if not params_records:
            raise ValueError("参数列表不能为空")
        params_data: dict = records_to_list(params_records)
        sql, values = self.build_insert(table_name, params_data, update_fields=update_fields or [], explode=True)
        return await self.async_run(sql, values, conn=conn)

    async def async_update(self, table_name: str, params_data: dict, row_id: int, primary_key: str = "id", conn=None):
        """
        根据主键字段更新指定行数据。

        Args:
            table_name (str): 表名
            row_id: 主键值（通常是 id）
            params_data (dict): 要更新的字段及新值
            primary_key (str): 主键字段名，默认是 'id'
            conn:可选外部传入连接
        """
        if not params_data:
            raise ValueError("更新数据不能为空")
        if not row_id:
            raise ValueError("row_id 不能为空")

        update_fields = ', '.join(f"`{k}` = %s" for k in params_data.keys())  # 构建更新字段列表
        values = tuple(params_data.values()) + (row_id,)
        sql = f"UPDATE `{table_name}` SET {update_fields} WHERE `{primary_key}` = %s"
        return await self.async_run(sql, values, conn=conn)

    async def get_offset(self, table_name: str, page: int = None, per_page: int = 10, cursor=None,
                         use_estimate: bool = False):
        total = 0
        if use_estimate:
            row = await self.query_one(
                "SELECT TABLE_ROWS AS estimate "
                "FROM information_schema.TABLES "
                "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s",
                params=(table_name,), cursor=cursor)
            total = int(row["estimate"]) if row and row.get("estimate") is not None else 0
        if (not use_estimate) or total <= 0:
            count_res = await self.query_one(f"SELECT COUNT(*) as count FROM {table_name}", cursor=cursor)
            total = int(count_res["count"]) if count_res else 0

        total_pages = (total + per_page - 1) // per_page if total > 0 else 1
        if page is None:  # default to last page when page not provided
            page = total_pages
        if page < 1:
            page = 1
        if page > total_pages:
            page = total_pages
        offset = (page - 1) * per_page
        return offset, page, total_pages, total

    async def get_table_columns(self, table_name: str, cursor=None) -> list[dict]:
        async def _run(c) -> list:
            await c.execute(f"DESCRIBE {table_name}")
            columns = await c.fetchall()
            # 转换列信息为更友好的格式
            formatted_columns = [{
                "name": col["Field"],
                "type": col["Type"],
                "nullable": col["Null"] == "YES",
                "key": col["Key"],
                "default": col["Default"],
                "extra": col["Extra"]
            } for col in columns]
            return formatted_columns

        if cursor:
            return await _run(cursor)
        async with self.get_cursor() as cur:
            return await _run(cur)

    async def get_primary_key(self, table_name: str, cursor=None) -> Optional[str]:
        columns = await self.get_table_columns(table_name, cursor=cursor)
        for col in columns:
            if col["key"] == "PRI":  # Primary Key,"UNI","MUL"
                return col["name"]
        return None

    async def get_table_schema(self, table_name: str, conn=None) -> list:
        """异步版本获取表结构信息"""
        sql, params = self.table_schema(table_name)
        return await self.async_run(sql, params, conn) or []


class OperationMysql(AsyncMysql, SyncMysql, BaseMysql):
    db_config = parse_database_uri(Config.SQLALCHEMY_DATABASE_URI)

    """
    dbop = OperationMysql(async_mode=True)
    await dbop.init_pool()
    result = await dbop.async_run("SELECT * FROM users WHERE id=%s", (...,))
    await dbop.close_pool()

    async with OperationMysql(async_mode=True) as dbop:
        result = await dbop.async_run("SELECT ...")

    async with OperationMysql.context(...,db_config=Config.Risk_DB_CONFIG) as dbop:
        async with dbop.get_conn_ctx() as conn:
            await dbop.async_run(...,conn)

    with SyncMysql(host, user, password, db) as db:
        result = db.run("SELECT * FROM users WHERE id=%s", (...,))

    async with AsyncMysql(...,**db_config,maxsize=1) as dbop:
        conn = await dbop.get_conn()
        await dbop.async_run(...,conn)
        dbop.release(conn)
    """

    def __init__(self, async_mode: bool = False, db_config: dict | None = None, **kwargs):
        self.async_mode = async_mode
        self.config = db_config or type(self).db_config

        if async_mode:
            # super(AsyncMysql, self).__init__(**self.config, **kwargs)
            AsyncMysql.__init__(self, **self.config, **kwargs)
        else:
            SyncMysql.__init__(self, **self.config, **kwargs)

        if not async_mode:
            self.ensure_connection()
            print('[MysqlData] Sync connected.')

    async def __aenter__(self):
        if self.async_mode:
            return await AsyncMysql.__aenter__(self)
        raise RuntimeError("Use 'with' for sync mode")

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.async_mode:
            await AsyncMysql.__aexit__(self, exc_type, exc_val, exc_tb)

    def __enter__(self):
        if not self.async_mode:
            return SyncMysql.__enter__(self)
        raise RuntimeError("Use 'async with' for async mode")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.async_mode:
            SyncMysql.__exit__(self, exc_type, exc_val, exc_tb)

    @classmethod
    async def context(cls, **kwargs):
        """类级别的异步上下文生成器,创建实例并托管,对外暴露类接口,提供外部使用"""
        async with cls(async_mode=True, **kwargs) as dbop:
            yield dbop


class CollectorMysql(AsyncMysql):
    def __init__(self, *,
                 batch_size: int = 1000,
                 queue_maxsize: int = 10000,
                 max_wait_seconds: float = 1.0,
                 worker_count: int = 1,
                 retry_times: int = 2, retry_backoff: float = 1.0,
                 db_config: dict = None):
        '''
        collector = CollectorMysql(db_config)
        await collector.start()
        await collector.enqueue(
        ok = collector.enqueue_nowait(
        wait collector.stop(flush=True)# 停机前清理
        '''
        config = db_config or parse_database_uri(Config.SQLALCHEMY_DATABASE_URI)
        super().__init__(**config)

        self.batch_size = batch_size
        self.batch_interval = max_wait_seconds
        self.worker_count = worker_count
        self.retry_times = retry_times
        self.retry_backoff = retry_backoff

        self._queue: asyncio.Queue = asyncio.Queue(maxsize=queue_maxsize)
        self._workers: list[asyncio.Task] = []
        self._stopped = asyncio.Event()
        self._started = False  # _is_running

    @classmethod
    def from_instance(cls, instance: AsyncMysql, **kwargs):
        """工厂方法，从现有实例创建独立的Collector
        CollectorMysql.from_instance(instance=DB_Client, batch_size=2000, max_wait_seconds=1.0)
        """
        if hasattr(instance, 'config'):
            collector = cls(db_config=instance.config, **kwargs)
        else:
            collector = cls(**kwargs)

        # 复制所有连接配置
        collector.host = instance.host
        collector.user = instance.user
        collector.password = instance.password
        collector.db_name = instance.db_name
        collector.port = instance.port
        collector.charset = instance.charset

        # 复制其他运行时属性
        collector.pool = instance.pool  # 共享池
        collector.minsize = instance.minsize
        collector.maxsize = instance.maxsize
        collector.autocommit = instance.autocommit

        return collector

    async def start(self):
        if self._started:
            return
        self._stopped.clear()
        for _ in range(self.worker_count):
            self._workers.append(asyncio.create_task(self._worker_loop()))
        self._started = True

    async def stop(self, flush: bool = True, timeout: float = 10.0):
        """
        优雅停机：可选先 flush 所有队列再退出
        """
        if not self._started:
            return

        if flush:
            if timeout > 0:  # 轮询等待队列为空
                loop = asyncio.get_running_loop()
                start = loop.time()
                while not self._queue.empty() and (loop.time() - start) < timeout:
                    await asyncio.sleep(0.05)

        self._stopped.set()
        # cancel workers if they block on queue.get
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        self._started = False

        if flush and self._queue.qsize() > 0:  # flush 队列剩余数据
            try:
                await asyncio.wait_for(self.flush_all(), timeout=timeout)
            except asyncio.TimeoutError:
                print("[Collector] flush timeout, some data may not be written")

    @asynccontextmanager
    async def context(self):
        """实例级上下文管理器，用于对象内资源安全地初始化和关闭处理器"""
        await self.start()
        try:
            yield self
        finally:
            await self.stop()

    async def enqueue(self, table_name: str, params_data: dict, update_fields: list[str] = None):
        """
        将一条记录加入队列（默认阻塞直到放入，产生回压），直到队列有空位可以放入
        """
        sql, values = self.build_insert(table_name, params_data, update_fields=update_fields or [])
        await self._queue.put((sql, values))

    def enqueue_nowait(self, table_name: str, params_data: dict, update_fields: list[str] = None) -> bool:
        """
        尝试非阻塞入队，失败返回 False（可用于采样/舍弃策略），队列满不等待，直接抛出 QueueFull
        """
        try:
            sql, values = BaseMysql.build_insert(table_name, params_data, update_fields=update_fields or [])
            self._queue.put_nowait((sql, values))
            return True
        except asyncio.QueueFull:
            return False  # 选择：丢弃、降采样、写入备份文件、或写到 Redis 等持久队列

    async def _worker_loop(self):
        """
        消费者主循环：按 batch_size 或 max_wait_seconds 刷盘
        """
        loop = asyncio.get_running_loop()
        while not self._stopped.is_set():
            try:
                # 用 batch_interval 等待第一个元素，避免 busy loop
                item = await asyncio.wait_for(self._queue.get(), timeout=self.batch_interval)
                # if item is None:
                #     break # propagate sentinel to stop
            except asyncio.TimeoutError:
                continue  # 没有新数据，检查 _stopped 后继续
            batch = [item]
            deadline = loop.time() + self.batch_interval

            while len(batch) < self.batch_size:  # 尝试在剩余时间内凑满 batch_size
                timeout = deadline - loop.time()
                if timeout <= 0:
                    break
                try:
                    item = self._queue.get_nowait()
                    batch.append(item)
                except asyncio.QueueEmpty:
                    try:  # 等待直到剩余时间结束或新数据到来
                        item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break

            if not batch:
                continue

            conn = await self.get_conn()
            try:
                await self._flush_batch(batch, conn)  # 处理 batch（异步执行 DB 写）
            except Exception as e:
                # 此处应该记录日志/告警；为防止数据丢失，可考虑把失败的 batch 放到后端持久化或重试队列
                logging.error(f"[Collector] flush batch failed: {e}")
                # 这里不 raise，让 loop 继续消费后续 batch
            finally:
                for _ in batch:  # 标记任务完成，确保释放连接
                    self._queue.task_done()
                self.release(conn)

    async def _flush_batch(self, batch: list, conn=None):
        # 分组：将字段一致的行合并到一起以便用 executemany
        groups: dict[str, list] = {}
        for sql, params in batch:
            groups.setdefault(sql, []).append(params)

        # prepare sql_list for async_execute: if a group has same fields -> executemany
        sql_list = [(sql, params_list) for sql, params_list in groups.items()]

        # 执行并带重试逻辑
        process_execute = async_error_logger(max_retries=self.retry_times, delay=1, backoff=self.retry_backoff,
                                             extra_msg="InsertCollector flush batch")(self.async_execute)
        ok = await process_execute(sql_list, conn=conn)
        if not ok:
            raise RuntimeError("flush batch failed after retries")

    async def flush_all(self):
        items = []
        while not self._queue.empty():
            items.append(await self._queue.get())
        if not items:
            return
        async with self.pool.acquire() as conn:
            await self._flush_batch(items, conn)
