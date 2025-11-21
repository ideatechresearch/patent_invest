# -*- coding: utf-8 -*-
import asyncio
import datetime
import os, copy
import tempfile
import logging

from typing import AsyncGenerator, Generator
from fastapi import FastAPI, Request, Header, Depends, Query, Body, File, UploadFile, BackgroundTasks, Form, \
    WebSocket, WebSocketDisconnect, HTTPException, status
from fastapi.responses import Response, StreamingResponse, JSONResponse, FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from starlette.middleware.sessions import SessionMiddleware
from sse_starlette.sse import EventSourceResponse

from collections.abc import AsyncIterator
from passlib.context import CryptContext
from config import Config

Config.load()
if os.getenv('AIGC_DEBUG', '0').lower() in ('1', 'true', 'yes'):
    Config.debug()  # 测试环境配置,生产环境注释

from structs import *
from database import *
from generates import *
from router import *
from utils import configure_event_loop
from rime.allele import Allele

configure_event_loop()

mcp_app = mcp.http_app(transport="streamable-http", path="/mcp")


#  定义一个上下文管理器,初始化任务（如初始化数据库、调度器等） @app.lifespa
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        print("Starting up.")
        get_database_manager().create_tables()
        # if not w3.is_connected():
        #     print("Failed to connect to Ethereum node")

        app.state.redis = await get_redis_connection()
        await DB_Client.init_pool(minsize=2, maxsize=Config.DB_MAX_SIZE)
        app.state.collector = CollectorMysql.from_instance(instance=DB_Client, max_wait_seconds=1.0)
        await app.state.collector.start()
        await BaseChatHistory.initialize(get_session_context)
        app.state.httpx_client = get_httpx_client(time_out=Config.HTTP_TIMEOUT_SEC)
        app.state.logger = logger

        app.state.worker_id = get_worker_identity()
        app.state.worker_num = int(os.environ.get("UVICORN_WORKERS") or os.environ.get("GUNICORN_WORKERS", 1))
        app.state.is_debug = Config.get('IS_DEBUG', False)
        if app.state.is_debug:
            print('Config:', Config.get_config_data())
            tracemalloc.start()

        await init_ai_clients(AI_Models)
        await ModelList.set(redis=app.state.redis, worker_id=app.state.worker_id, ai_models=AI_Models)

        app.state.scheduler = get_scheduler(redis_host=Config.REDIS_HOST)
        app.state.ai_scheduler = TaskGraphManager(dask_client=None)
        app.state.is_main = await is_main_worker(worker_id=app.state.worker_id, redis=app.state.redis)
        if app.state.is_main:
            TaskManager.htw.add_daily_task(18, 18, run_adjustment_sync_tasks, concurrent=False)
            print("Main worker started. Do once-only init. Total workers:", app.state.worker_num)
            # app.state.dask_client = get_dask_client()
            print('AI Client:', AI_Client.keys(), ModelList.owners)
        else:
            pass
            # app.state.dask_client = get_dask_client(Config.DASK_Cluster)

        t = asyncio.create_task(handle_minute())
        app.state.cron_tasks = [t]
        if not app.state.scheduler.get_job("hour_job"):
            app.state.scheduler.add_job(handle_hour, 'interval', id="hour_job", seconds=3600, misfire_grace_time=60,
                                        jobstore='memory', executor='default')  # , max_instances=3

        app.state.func_manager = get_func_manager()
        app.state.func_manager.class_method.register(Allele, None)  # ClassMethodRegistry.register(Allele)
        if not app.state.scheduler.get_job("metadata_job"):
            app.state.scheduler.add_job(app.state.func_manager.generate_tools_metadata,
                                        'cron', id="metadata_job", hour=5, minute=20,
                                        misfire_grace_time=300,
                                        kwargs={"model_name": Config.DEFAULT_MODEL_METADATA}, jobstore='memory',
                                        # 'redis',
                                        max_instances=1, replace_existing=True)

        print(app.state.worker_id, "inited.")
        # message_zero_mq = MessageZeroMQ()
        # task1 = asyncio.create_task(message_zero_mq.start())
        # global_function_registry()
        # if Config.get('WORKERS',0) == 1:
        #     mcp_task, exit_event = await run_mcp_task(port=7007)
        # Run both lifespans
        async with mcp_app.lifespan(app):
            yield

    except asyncio.CancelledError:
        logging.warning("Lifespan 被取消（应用关闭中）")
    except Exception as e:
        logging.error(f"生命周期异常: {e}")
        # asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    finally:
        print("Shutting down.")
        for t in app.state.cron_tasks:
            t.cancel()
        await asyncio.gather(*app.state.cron_tasks, return_exceptions=True)
        app.state.scheduler.shutdown()
        get_database_manager().dispose()
        await BaseChatHistory.shutdown()
        await async_engine.dispose()
        await app.state.collector.stop(flush=True)
        await DB_Client.close_pool()

        await shutdown_httpx()
        await shutdown_redis()

        close_dask_client()
        # if app.state.is_main:
        #     ModelList.save()
        if app.state.is_debug:
            tracemalloc.stop()


@asynccontextmanager
async def combined_lifespan(app: FastAPI):
    async with lifespan(app):
        async with mcp_app.lifespan(app):
            yield


app = FastAPI(title="AIGC API", lifespan=lifespan)  # combined_lifespan
app.add_middleware(SessionMiddleware, secret_key=Config.SECRET_KEY)
# mcp = FastMCP.from_fastapi(app=app,name='aigc MCP')
dashscope.api_key = Config.DashScope_Service_Key
# 加密配置,密码哈希上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# OAuth2 密码令牌,设置 tokenUrl
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
os.makedirs(Config.DATA_FOLDER, exist_ok=True)
app.mount("/data", StaticFiles(directory=Config.DATA_FOLDER), name="data")
# directory=os.path.abspath('.') + "/static")
# 配置静态文件和模板
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/.well-known", StaticFiles(directory="static/.well-known"), name="well-known")
app.mount("/mcp", mcp_app)

app.include_router(index_router, prefix="")
app.include_router(ideatech_router, prefix="/ideatech")
app.include_router(table_router, prefix="/table")
app.include_router(chat_router, prefix="/chat")


# model_config['protected_namespaces'] = ()

@async_timer_cron(interval=60)
async def handle_minute():
    tick_now = datetime.now()
    if not TaskManager.htw.tick_time or TaskManager.htw.tick_time.minute != tick_now.minute:  # 每分钟只执行一次（忽略秒和微秒）
        await TaskManager.htw.tick(level="minute", tick_time=tick_now.replace(second=0, microsecond=0))  # 推动分钟轮
        await TaskManager.clean_old_tasks(3600, redis_client=app.state.redis)
        cleanup_old_tempfiles(min_age=600, prefix="tmp_")
        # print(f"Tick new minute! The time is: {TaskManager.htw.tick_time_str}")


async def handle_hour():
    print(f"Tick new bar! The time is: {TaskManager.htw.tick_time_str}")
    if app.state.is_main:
        if TaskManager.htw.tick_time.hour == 23 and TaskManager.htw.tick_time.weekday() == 6:
            res = await run_adjustment_sync_tasks(concurrent=False)
            print(f'[adjustment_sync]:{res}')

    BaseChatHistory.flush_cache()

    await asyncio.to_thread(memory_monitor, threshold_percent=60, desc=app.state.is_debug)


# @app.on_event("startup")
# async def startup_event():
#     print("Starting up...")
#     Base.metadata.create_all(bind=engine)
#     asyncio.create_task(async_cron())
#     sse_transport = SseServerTransport(app)
#     mcp.mount_transport(sse_transport)
#
# @app.on_event("shutdown")
# async def shutdown_event():
#     print("Shutting down...")

# def api_route(path: str):
#     """装饰器：把变量变成 API"""
#     def decorator(func):
#         @functools.wraps(func)
#         @app.get(path)
#         def wrapper():
#             return func()
#         return wrapper
#     return decorator


# @app.middleware("http")
# async def verify_api_key(request: Request, call_next):
#     api_key = request.headers.get("Authorization")
#     if api_key:
#         api_key = api_key.replace("Bearer ", "")  # 如果用的是 Bearer Token
#     if api_key not in Config.VALID_API_KEYS:
#         raise HTTPException(status_code=401, detail="Invalid API key")
#     response = await call_next(request)
#     return response


@app.get("/send_wechat_code")
async def send_verification_code(username: str):
    if not username:
        raise HTTPException(status_code=400, detail="用户名不能为空")
    key_verify = f"verify_code:{username}"

    code = str(random.randint(100000, 999999))
    await app.state.redis.setex(key_verify, int(Config.VERIFY_TIMEOUT_SEC), code)
    response = await send_to_wechat(username, f'验证码：{code}，{Config.VERIFY_TIMEOUT_SEC // 60}分钟有效期') or {}
    return {"status": "success", "message": "验证码已发送", "data": {**response}}


@app.post("/register")
async def register_user(request: Registration, db: Session = Depends(get_db)):
    """
    如果提供了 eth_address 或 public_key，则不强制提供密码。
    如果提供了 username 或 uuid，并且没有提供 eth_address 或 public_key，则需要提供密码进行注册。
    """
    username = request.username
    public_key = request.public_key
    eth_address = request.eth_address
    signed_message = request.signed_message
    original_message = request.original_message

    if not (username or eth_address or public_key):
        raise HTTPException(status_code=400,
                            detail="At least one of username, eth_address, or public_key must be provided")

    db_user = User.get_user(db, username, public_key=public_key, eth_address=eth_address)
    if db_user:
        raise HTTPException(status_code=400, detail="User already registered")
    # 验证签名
    if signed_message and original_message:
        if eth_address and w3:
            recovered_address = w3.eth.account.recover_message(text=original_message, signature=signed_message)
            if recovered_address.lower() != eth_address.lower():
                raise HTTPException(status_code=400, detail="Signature verification failed")

        if public_key:
            is_verified = verify_ecdsa_signature(public_key, original_message, signed_message)
            if not is_verified:
                raise HTTPException(status_code=400, detail="Public key authentication failed")

    if request.code:
        key_verify = f"verify_code:{username}"
        r = app.state.redis or get_redis()
        stored_code = await r.get(key_verify)
        if not stored_code:
            raise HTTPException(status_code=400, detail="验证码已过期或未发送")
        if isinstance(stored_code, bytes):
            stored_code = stored_code.decode()
        if stored_code != request.code:
            raise HTTPException(status_code=400, detail="验证码不正确")
        await r.delete(key_verify)
    # 注册新用户
    db_user = User.create_user(db=db, username=username, password=request.password, role=request.role,
                               group=request.group, eth_address=eth_address, public_key=public_key)

    if not db_user:
        raise HTTPException(status_code=400,
                            detail="Password is required when neither eth_address nor public_key is provided")

    return {"status": "success", "message": "User registered successfully"}
    # User.update_user(user_id=db_user.id, eth_address=eth_address)


@app.post("/authenticate", response_model=Token)
async def authenticate_user(request: AuthRequest, response: Response, session: AsyncSession = Depends(get_db_session)):
    '''
    登录路由，颁发访问令牌和刷新令牌,令牌生成 login_for_access_token,
    如果 eth_address 或 public_key 认证成功，通过公钥验证签名则不需要密码。
    使用 username 或 uuid 和密码登录。
    '''
    public_key = request.public_key  # 用户的公钥
    signed_message = request.signed_message  # 签名的消息，Base64 编码格式（确保已正确编码）
    original_message = request.original_message  # 要验证的原始消息内容
    username = request.username
    is_verified = 0
    db_user = None
    # 验证签名
    if signed_message and original_message:
        if request.eth_address and w3:
            recovered_address = w3.eth.account.recover_message(text=original_message, signature=signed_message)
            if recovered_address.lower() != eth_address.lower():
                raise HTTPException(status_code=400, detail="Authentication failed")

            db_user = await User.async_get_user(session, eth_address=request.eth_address)
            is_verified = 1 << 0

        if public_key:
            if not verify_ecdsa_signature(public_key, original_message, signed_message):
                raise HTTPException(status_code=400, detail="Public key authentication failed")

            db_user = await User.async_get_user(session, public_key=public_key)
            is_verified |= 1 << 1

        if is_verified and db_user:
            return issue_access_token(response, db_user, Config.ACCESS_TOKEN_EXPIRE_MINUTES)

    if username:
        db_user = await User.async_get_user(session, username=username)  # User.validate_credentials(
        if not db_user:
            raise HTTPException(status_code=400, detail="User not found")

        if request.password:
            if not User.verify_password(request.password, db_user.password):
                raise HTTPException(status_code=400, detail="Invalid credentials")
            is_verified |= 1 << 2

        if request.code:
            key_verify = f"verify_code:{username}"
            r = app.state.redis or get_redis()
            stored_code = await r.get(key_verify)
            if not stored_code:
                raise HTTPException(status_code=400, detail="验证码已过期或未发送")
            if stored_code != request.code:
                raise HTTPException(status_code=400, detail="验证码不正确")
            await r.delete(key_verify)
            is_verified |= 1 << 3

        if is_verified:
            return issue_access_token(response, db_user, Config.ACCESS_TOKEN_EXPIRE_MINUTES)

    raise HTTPException(status_code=400, detail="Invalid authentication request")


@app.post("/refresh_token", response_model=Token)  # dict
async def refresh_token(token: str = Form(...)):
    new_access_token = refresh_access_token(token)
    return {"access_token": new_access_token, "token_type": "bearer"}


@app.post("/secure")
async def secure_route(request: Request, session: AsyncSession = Depends(get_db_session)):
    """防止伪造 API Key,防止请求内容被篡改,防止重放攻击（时间戳）,避免签名泄露,HMAC 使用安全算法（SHA256）"""
    # User.update_user( db,1, public_key='83e2c687a44f839f6b3414d63e1a54ad32d8dbe4706cdd58dc6bd4233a592f78367ee1bff0e081ba678a5dfdf068d4b4c72f03085aa6ba5f0678e157fc15d305')
    username = get_access_user(request)
    api_keys = {}
    if username:
        user = await User.async_get_user(session=session, username=username)
        if user and not user.disabled:
            api_keys = await User.get_api_keys(session=session, _id=user.id, redis=app.state.redis,
                                               ex=Config.VERIFY_TIMEOUT_SEC)
    if not api_keys:
        api_keys = await User.get_active_api_keys(session=session, redis=app.state.redis, ex=Config.VERIFY_TIMEOUT_SEC)
    await verify_request_signature(request, api_keys, time_out=Config.VERIFY_TIMEOUT_SEC)
    return {"status": "success", "message": "Request authenticated successfully"}


def issue_access_token(response: Response, user: User,
                       expires_minutes: int = Config.ACCESS_TOKEN_EXPIRE_MINUTES) -> Token:
    """
    签发访问令牌（Access Token），并写入安全 Cookie。
    """
    if user.disabled:
        raise HTTPException(status_code=403, detail="Access forbidden,Inactive user")
    if user.expires_at and user.expires_at <= int(time.time()):  # 防止已过期账户登录
        raise HTTPException(status_code=403, detail="Access forbidden: user expired")

    access_token = create_access_token(data={"sub": user.username, 'user_id': user.user_id, "type": "access"},
                                       expires_minutes=expires_minutes)
    max_age = int(expires_minutes * 60)
    response.set_cookie(key="access_token", value=access_token,
                        httponly=True,
                        secure=True,  # 生产部署时强制 https
                        samesite="lax",  # 或 "strict" / "none"（前后端分离、跨域时应为 "none" 且 secure=True）
                        max_age=max_age,
                        path="/")  # extract_token->cookie_token
    return Token(access_token=access_token, token_type="bearer", expires_in=max_age)


# 检查该用户是否在数据库中有效或是否具备某些标志位, 严格依赖
async def get_current_user(token: str = Depends(oauth2_scheme),
                           session: AsyncSession = Depends(get_db_session)) -> User:
    username = verify_access_token(token)
    user = await User.async_get_user(session, username=username, user_id=username)
    if not user:
        raise HTTPException(status_code=404, detail="Access forbidden,Invalid credentials,User not found")
    if user.disabled:
        raise HTTPException(status_code=403, detail="Access forbidden,Inactive user")
    if user.expires_at and user.expires_at <= int(time.time()):
        raise HTTPException(status_code=403, detail="Access forbidden: user expired")
    return user


@app.post("/protected")
async def protected_route(user: User = Depends(get_current_user)):
    '一般认证需求,访问控制'
    return {"message": "Access granted", "username": user.username}


@app.get("/api/key/")
async def get_apikey(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if user.api_key:
        return {"username": user.username, 'api_key': user.api_key}

    api_key, secret_key = User.create_api_key(db, user.id)
    return {"username": user.username, 'api_key': api_key}  # 标识和授权用户请求


@app.post("/api/login", response_model=Token)
async def login_with_apikey(api_key: str = Form(...), response: Response = None,
                            session: AsyncSession = Depends(get_db_session)):
    """
    通过 API Key 登录，验证后返回 JWT Token 并写入 Cookie。
    """
    user = await User.async_get_user(session, api_key=api_key)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return issue_access_token(response, user)


@app.post("/api/logout")
async def logout(request: Request, response: Response):
    """
    安全登出接口：清除后端 Cookie。
    """
    response.delete_cookie(key="access_token", path="/")
    request.session.pop("username", None)
    # if request.session.get("user_id"):
    #     request.session.clear(), 用于跟踪客户端,不重复初始化用户状态
    return {"status": "success", "message": "Logged out successfully"}


@app.get("/admin/")
async def admin(user: User = Depends(get_current_user)):
    if user.username == 'admin' and user.role == 'admin':
        return JSONResponse(
            content={'history': BaseChatHistory.Chat_History_Cache, 'task': dataclass2dict(TaskManager.Task_map),
                     'task_nodes': app.state.ai_scheduler.export_nodes(),
                     'task_edges': app.state.ai_scheduler.export_adjacency_list(),
                     'job': get_job_list(app.state.scheduler)})
    return {"message": "Access denied: Admin privileges required"}


@app.get('/user/')
async def user(request: Request, session: AsyncSession = Depends(get_db_session)):
    '''尝试使用 optional token（header/cookie/query)'''
    username = get_access_user(request)
    if username:
        # 没 token -> 如果 session 中有 user_id 就认为是匿名用户,get_current_user(token=token, db=db)
        user = await User.async_get_user(session, username=username, user_id=username)
        if not user:
            return {"error": "User not found"}
        if user.disabled:
            return {"error": f"Access forbidden for disabled users {username}"}
        request.session['username'] = username  # 注意多worker缓存用户名 失效，内存不共享
        return {"user": username, "type": "authenticated"}

    session_uid = request.session.get('user_id')
    if not session_uid:
        session_uid = str(uuid.uuid4())
        request.session['user_id'] = session_uid  # 伪用户信息用于 session,匿名用户临时标识

    return {"user": session_uid, "type": "anonymous"}


@app.get("/status/")
async def system_status():
    import threading
    thread_count = threading.active_count()
    task_count = len(asyncio.all_tasks())  # asyncio 任务数

    redis_info = await app.state.redis.info()
    pool = async_engine.pool

    return JSONResponse(content={
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "event_loop_type": str(type(asyncio.get_running_loop())),  # asyncio.get_event_loop()
        "worker_id": app.state.worker_id,
        "worker_num": app.state.worker_num,
        "pid": os.getpid(),
        "threads": thread_count,
        "asyncio_task_count": task_count,
        "job_count": len(app.state.scheduler.get_jobs()),
        "memory_mb": round(get_memory_info(), 2),
        "cpu_time_sec": round(get_cpu_time(), 2),
        "open_fd_count": get_open_fds_count(),  # socket连接数 (打开的文件描述符数量)
        "http_connection_count": count_http_connections(7000),
        "redis_connected_clients": redis_info.get("connected_clients"),
        "redis_keys": redis_info.get("db0", {}).get("keys", 0),
        "db_connections_total": pool.size(),
        "db_connections_in_use": pool.checkedout(),
        "db_connections_idle": pool.checkedin(),
        "task_manager_count": TaskManager.size(),
        'history_cache_count': BaseChatHistory.size(),
        "timewheel_task_count": TaskManager.htw.task_count,
        "timewheel_elapsed_time": TaskManager.htw.elapsed_time,
        "tick_time": TaskManager.htw.tick_time_str,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })


@app.get("/status/logs")
async def get_logs_info(lines: int = 100):
    return await get_log_lines("app.log", lines)


@app.get("/status/routes")
def list_routes():
    return [{"path": r.path, "name": r.name} for r in app.routes]


@app.get("/status/files")
async def list_upload_files(max_pages: int = 1):
    return await asyncio.to_thread(oss_list_files, AliyunBucket, prefix='upload/', max_pages=max_pages)


@require_api_key
@app.post("/data")
async def push_redis_data(payload: dict):
    'VALID_API_KEYS'
    try:
        await app.state.redis.lpush("task_queue", json.dumps(payload))
        return {"status": "queued"}
    except (ConnectionError, TimeoutError) as e:
        return JSONResponse(status_code=503, content={"error": "Redis 连接失败", "detail": str(e)})
    except Exception as e:
        return {"error": str(e)}


@app.get("/get/{key}")
async def read_redis_value(key: str = 'funcmeta:*'):
    try:
        result = await get_redis_value(app.state.redis, key)
        if result is None:
            return {"error": f"Key '{key}' not found"}
        return {"result": result}

    except Exception as e:
        app.state.logger.error(e)
        raise HTTPException(status_code=500, detail={"error": str(e)})


@require_api_key
@app.delete("/delete/{key}")
async def delete_redis_key(key: str):
    try:
        if "*" in key or "?" in key or "[" in key:
            redis = app.state.redis or get_redis()
            keys = await redis.keys(key)
            if not keys:
                return {"error": f"No keys match pattern '{key}'"}
            count = await redis.delete(*keys)
            return {"keys": [k.decode() if isinstance(k, bytes) else k for k in keys],
                    "count": count}
        else:
            count = await redis.delete(key)
            return {"key": key, "count": count}
    except (ConnectionError, TimeoutError) as e:
        return JSONResponse(status_code=503, content={"error": "Redis 连接失败", "detail": str(e)})
    except Exception as e:
        return {"error": str(e)}


@app.get("/pip/install")
async def install_packages_import(q: str):
    """
    /install?q=numpy,matplotlib==3.7.1
    """
    packages = q.replace("，", ",").split(",")  # 支持中文逗号
    pkg_list = pip_install(*packages)
    if isinstance(pkg_list, dict) and pkg_list.get("error"):
        return pkg_list
    imports = import_packages(pkg_list)
    return {"packages": pkg_list, "imports": imports}


@app.get("/pip/list")
def pip_list():
    return {"packages": pip_installed_list()}


@app.get("/health")
async def healthcheck():
    return {"status": True}


# @app.get("/mcp_sse")
# async def mcp_sse(request: Request):
#     """SSE 端点用于 MCP 实时通信"""
#
#     async def event_generator():
#         while True:
#             if await request.is_disconnected():
#                 break
#             # 发送心跳保持连接
#             yield {
#                 "event": "heartbeat",
#                 "data": json.dumps({"timestamp": datetime.now().isoformat()})
#             }
#             await asyncio.sleep(30)
#
#     return EventSourceResponse(event_generator())


# @app.get("/mcp_openai")
# async def mount_mcp_openai():
#     mcp.mount(prefix="openapi", server=create_openai_mcp("http://127.0.0.1:7000"))


# @app.get("/opensearch.xml")
# async def get_opensearch_xml():
#     # OpenSearch是一个开放的搜索标准，允许用户将一个网站的搜索功能集成到浏览器或其他应用程序中。
#     xml_content = rf"""
#     <OpenSearchDescription xmlns="http://a9.com/-/spec/opensearch/1.1/" xmlns:moz="http://www.mozilla.org/2006/browser/search/">
#     <ShortName>{Config.WEBUI_NAME}</ShortName>
#     <Description>Search {Config.WEBUI_NAME}</Description>
#     <InputEncoding>UTF-8</InputEncoding>
#     <Image width="16" height="16" type="image/x-icon">{Config.WEBUI_URL}/static/favicon.png</Image>
#     <Url type="text/html" method="get" template="{Config.WEBUI_URL}/?q={"{searchTerms}"}"/>
#     <moz:SearchForm>{Config.WEBUI_URL}</moz:SearchForm>
#     </OpenSearchDescription>
#     """
#     return Response(content=xml_content, media_type="application/xml")


@app.get('/retrieval/{text}')
async def retrieval(text: str, platform: Literal[
                                             'duckduckgo', 'tavily', 'serper', 'brave', 'firecrawl', 'exa', 'zhipu', 'arxiv', 'wiki', 'patent', 'invest', 'google', 'bing', 'baidu', 'yahoo', 'auto'] | None = None):
    if platform == 'duckduckgo':
        results = await duckduckgo_search(text)
    elif platform == 'tavily':
        results = await web_search_tavily(text)
    elif platform == 'serper':
        results = await serper_search(text)
    elif platform == 'brave':
        results = await brave_search(text)
    elif platform == 'firecrawl':
        results = await firecrawl_search(text)
    elif platform == 'exa':
        results = await exa_search(text)
    elif platform == 'jina':
        results = await web_search_jina(text)
    elif platform in ('google', 'bing', "baidu", "yahoo"):
        results = await search_by_api(text, engine=platform)
    elif platform == 'zhipu':
        results = await web_search_async(text)
    elif platform == 'arxiv':
        results = await arxiv_search(text)
    elif platform == 'wiki':
        results = await wikipedia_search(text)
    elif platform == 'auto':
        results = await search_by_api(text, engine=None, location='中国')
    elif platform == 'patent':
        results = await patent_search(text, limit=3)
    elif platform == 'invest':
        results = await company_search(text, search_type='invest', limit=10)
    else:
        results = await web_search_intent(text)

    return JSONResponse(results)


@app.get("/extract/")
async def extract(text: str = Query(...), extract: str = Query(default='all')):
    text = clean_escaped_string(text)  # 去除外层成对的引号（单引号或双引号）

    if is_url(text):
        if extract == 'jina':
            return await web_extract_jina(text)
        if extract == 'tavily':
            return await web_extract_tavily(text)
        if extract == 'exa':
            return await web_extract_exa(text)
        if extract == 'firecrawl':
            return await firecrawl_scrape(text)

    return JSONResponse(extract_string(text, extract))


@app.post("/embeddings")
async def embeddings(request: EmbeddingRequest):
    inputs = [x.replace("\n", " ") for x in request.texts]
    # print(request.dict(),inputs)
    if request.model_name == 'baidu_bge':
        access_token = get_baidu_access_token()
        embedding = await get_baidu_embeddings(inputs, access_token=access_token)
        return {"embedding": embedding}

    embedding = await ai_embeddings(inputs, model_name=request.model_name, model_id=request.model_id,
                                    normalize=request.normalize)
    if embedding:
        return {"embedding": embedding}

    return {"embedding": get_hf_embeddings(inputs, model_name=request.model_name)}


@app.post("/fuzzy")
async def fuzzy_matches(request: FuzzyMatchRequest):
    querys = [x.replace("\n", " ").strip() for x in request.texts]
    corpus = [x.replace("\n", " ").strip() for x in request.terms]
    results = []

    if request.method == 'levenshtein':  # char_unigr,计算两个词组之间的最小编辑次数,from nltk.metrics.distance import edit_distance
        for token in querys:
            matches = find_best_matches(token, corpus, top_n=request.top_n, cutoff=request.cutoff, best=False)
            matches = [(match[0], round(match[1], 3), match[2]) for match in matches]
            results.append({'query': token, 'matches': matches})
        # results = [{'token': token, 'matches': rapidfuzz.process.extract(token, corpus, limit=top_n, score_cutoff=cutoff)}
        #            for token in querys]
        # rapidfuzz.process.extractOne(token,choices)
    elif request.method == 'bm25':  # 初步检索,将词组转化为向量化表示（TF-IDF）,概率检索模型,适合在短词组或句子相似性上做简单匹配
        from script.bm25 import BM25
        bm25 = BM25(corpus)  # 全库太大（如数亿条），先用关键词快速 filter 再做 embedding,用户输入极短关键词，embedding 向量不稳定
        for token in querys:
            scores = bm25.rank_documents(token, request.top_n, normalize=True)
            matches = [(corpus[match[0]], round(match[1], 3), match[0]) for match in scores if
                       match[1] >= request.cutoff]
            results.append({'query': token, 'matches': matches})
    elif request.method == 'reranker':  # 精细排序,BERT Attention,Cross-encoder / Bi-encoder,通过 Transformer 编码，进行对比分析,可以利用上下文信息
        async def process_token(token):
            scores = await ai_reranker(token, documents=corpus, top_n=request.top_n,
                                       model_name="BAAI/bge-reranker-v2-m3", model_id=0)
            matches = [(match[0], round(match[1], 3), match[2]) for match in scores if
                       match[1] >= request.cutoff]
            return {'query': token, 'matches': matches}

        results = await asyncio.gather(*(process_token(token) for token in querys))  # [(match,score,index)]
    elif request.method == 'embeddings':  # SBERT,MiniLM,将词组或句子嵌入为高维向量，通过余弦相似度衡量相似性
        similars = await get_similar_embeddings(querys, corpus, topn=request.top_n, embeddings_calls=ai_embeddings,
                                                model_name=Config.DEFAULT_MODEL_EMBEDDING)
        results = [{'query': token, 'matches':
            [(match[0], round(match[1], 3), corpus.index(match[0])) for match in matches if match[1] >= request.cutoff]}
                   for token, matches in similars]
    elif request.method == 'wordnet':  # 词典或同义词库，将词组扩展为同义词词组列表进行匹配,PageRank基于链接或交互关系构建的图中节点间的“传递性”来计算排名
        pass

    # corpus = list({match for token in querys for match in get_close_matches(token, corpus)})
    # match, score = process.extractOne(token, corpus)
    # results.append({'token': token, 'match': match, 'score': score})
    return JSONResponse(content=results, media_type="application/json; charset=utf-8")
    # Response(content=list_to_xml('results', results), media_type='application/xml; charset=utf-8')


@app.post("/classify")
async def classify_text(request: ClassifyRequest):
    intents = []
    query = request.query.strip()
    intent_tokens = [(it, x.replace("\n", " ").strip()) for it, keywords in request.class_terms.items() for x in
                     keywords]
    corpus = [token for _, token in intent_tokens]
    last_c = request.class_default

    # 遍历 class_terms 字典，检查文本是否匹配任意关键词
    if lang_token_size(query, model_name=Config.DEFAULT_MODEL_ENCODING) < 32:
        for intent, keywords in request.class_terms.items():
            if any(re.search(r'\b' + re.escape(keyword) + r'\b', query) for keyword in keywords):
                intents.append({"class": intent, 'score': 0, 'type': 'keyword'})

    matches = get_close_matches(query, corpus, n=10, cutoff=0.8)
    edit_scores = [(corpus.index(match), SequenceMatcher(None, query, match).ratio())
                   for match in matches]

    if edit_scores:
        intent = intent_tokens[edit_scores[0][0]][0]
        intents.append({"class": intent, 'score': edit_scores[0][1], 'type': 'edit'})
        if (all(intent_tokens[i[0]][0] == intent for i in edit_scores[:3]) and intent == last_c) or \
                (max(i[1] for i in edit_scores) > request.cutoff and all(i['class'] == last_c for i in intents)):
            return {"class": intent, 'match': intents}

        from script.bm25 import BM25
        best_match = BM25(corpus).best_match(query)
        if best_match >= 0:
            intent = intent_tokens[best_match][0]
            if all(intent_tokens[i[0]][0] == intent for i in edit_scores[:3]) and max(
                    i[1] for i in edit_scores) > request.cutoff:
                intents.append({"class": intent, 'score': None, 'type': 'bm25'})

    # 如果当前意图得分大于0.85并且与历史意图相同，更新历史记录并返回
    if any(i['score'] >= request.cutoff for i in intents if i['score']):
        if last_c and all(i['class'] == last_c for i in intents):
            return {"class": last_c, 'match': intents}

    similar_scores = await get_similar_embeddings([query], corpus, topn=10, embeddings_calls=ai_embeddings,
                                                  model_name=request.emb_model)  # [(q,[(match,score,index),])]

    if similar_scores:
        similar_scores = [(int(match[2]), float(match[1])) for match in similar_scores[0][1] if match[1] >= 0.8]
    if similar_scores:
        intent = intent_tokens[similar_scores[0][0]][0]
        intents.append({"class": intent, 'score': similar_scores[0][1], 'type': request.emb_model})

    if any(i['score'] >= request.cutoff for i in intents if i['score']):
        if all(i['class'] == intents[0]['class'] for i in intents):
            intent = intents[0]['class']
            return {"class": intent, 'match': intents}

    reranker_scores = await ai_reranker(query, documents=corpus, top_n=10, model_name=request.rerank_model,
                                        model_id=0)

    reranker_scores = [(match[2], match[1]) for match in reranker_scores if match[1] >= 0.8]  # [(match,score,index)]
    if reranker_scores:
        intent = intent_tokens[reranker_scores[0][0]][0]
        intents.append({"class": intent, 'score': reranker_scores[0][1], 'type': request.rerank_model})

    # 如果多个意图匹配得分超过0.85，并且意图相同，则返回这些意图
    if any(i['score'] >= 0.8 for i in intents if i['score']):
        if all(i['class'] == intents[0]['class'] for i in intents):
            intent = intents[0]['class']
            return {"class": intent, 'match': intents}

    return {"class": None, 'match': intents}


@app.get("/vec/create_collection/")
async def create_embeddings_collection(collection_name: str, model_name: str = Config.DEFAULT_MODEL_EMBEDDING):
    embeddings = await ai_embeddings(inputs=collection_name, model_name=model_name, model_id=0)
    if embeddings and embeddings[0]:
        size = len(embeddings[0])
        alias_name = collection_name + f"_{model_name}"  # suffix
        res = await create_collection(collection_name, client=QD_Client, size=size, alias_name=alias_name)

        if res:
            res['model_name'] = model_name
            res['model_size'] = size
            return res
    return {}


@app.post("/vec/upsert_points")
async def upsert_embeddings_points(payloads: List[Dict], inputs: Optional[List[str]], collection_name: str,
                                   text_field: str = None):
    collection_aliases = await QD_Client.get_collection_aliases(collection_name)
    model_name = collection_aliases.aliases[0].alias_name.split("_")[-1] if len(
        collection_aliases.aliases) else Config.DEFAULT_MODEL_EMBEDDING
    if text_field and not inputs:
        inputs = [p.get(text_field) for p in payloads]
    embeddings = await ai_embeddings(inputs=inputs, model_name=model_name, model_id=0, get_embedding=True)

    if embeddings:
        operation_id, ids = await upsert_points(payloads, vectors=embeddings, collection_name=collection_name,
                                                client=QD_Client)
        return {'operation_id': operation_id, 'ids': ids,
                'embeddings_size': len(embeddings),
                'embeddings_model': model_name}
    return {}


@app.post('/vec/search_points')
async def search_embeddings_points(querys: Union[str, List[str], Tuple[str]], collection_name: str, topn: int = 10,
                                   score_threshold: float = 0.0, payload_key: str = None, field_key: str = None,
                                   match_values=None):
    collection_aliases = await QD_Client.get_collection_aliases(collection_name)
    model_name = collection_aliases.aliases[0].alias_name.split("_")[-1] if len(
        collection_aliases.aliases) else Config.DEFAULT_MODEL_EMBEDDING
    match = field_match(field_key, match_values) if field_key else []

    if isinstance(querys, str):
        querys = [querys]
    if 'bge' in model_name.lower():
        querys = [f"为这个句子生成表示用于检索：{t}" for t in querys]

    return await search_by_embeddings(querys, collection_name, client=QD_Client,
                                      payload_key=payload_key, match=match, not_match=[],
                                      topn=topn, score_threshold=score_threshold, exact=False, get_dict=True,
                                      embeddings_calls=ai_embeddings, model_name=model_name)


@app.post('/vec/recommend_points')
async def recommend_points(ids: Union[List[int], Tuple[int]], collection_name: str, topn: int = 10,
                           score_threshold: float = 0.0, payload_key: str = None, field_key: str = None,
                           match_values=None):
    match = field_match(field_key, match_values) if field_key else []
    return await recommend_by_id(ids, collection_name, client=QD_Client, payload_key=payload_key,
                                 match=match, not_match=[], topn=topn, score_threshold=score_threshold)


@app.get("/nlp/")
async def nlp(text: str, nlp_type: str = 'ecnet'):
    return await baidu_nlp(nlp_type=nlp_type, text=text)


@app.get("/prompts/")
async def get_prompts(agent: str = None):
    if agent:
        system_prompt = System_content.get(agent, None)
        if system_prompt:
            JSONResponse({'prompt': system_prompt})

    return JSONResponse(System_content)


@app.post("/prompts/generate")
async def generate_prompts(request: PromptRequest):
    async def event_generator():
        depth_iter = iter(request.depth)
        to_do = next(depth_iter, None)
        if not to_do:
            yield json.dumps({"error": "depth 不能为空"}, ensure_ascii=False) + "\n"
            return
        content = await ai_analyze(request.query, System_content.get(to_do), '构造或改进系统提示词',
                                   model=request.model, max_tokens=4096, temperature=0.3)

        reason, prompt = extract_tagged_split(content, tag="reasoning")
        first = {'reason': reason, 'prompt': prompt, 'depth': 0}
        yield json.dumps(first, ensure_ascii=False) + "\n"
        await asyncio.sleep(0.03)
        to_do = next(depth_iter, None)
        if to_do:
            messages = [
                {"role": "system", "content": System_content.get(to_do)},
                {"role": "user", "content": request.query},
                {"role": "assistant", "content": prompt},
                {"role": "user", "content": "请帮我把下面的指令改写为更明确的系统提示词:" + prompt}
            ]
            for i, to_do in enumerate(depth_iter, start=1):
                try:
                    content, _ = await ai_client_completions(messages, client=None, model=request.model,
                                                             get_content=True, max_tokens=4096, temperature=0.3,
                                                             dbpool=app.state.collector)

                    reason, prompt = extract_tagged_split(content, tag="reasoning")
                    item = {'reason': reason, 'prompt': prompt, 'depth': i}
                    yield json.dumps(item, ensure_ascii=False) + "\n"
                    await asyncio.sleep(0.03)

                    messages[0] = {"role": "system", "content": System_content.get(to_do)}
                    messages.extend([{"role": "assistant", "content": content},
                                     {"role": "user", "content": "根据上下文，再帮我调整优化下系统提示词:" + prompt}])
                except Exception as e:
                    yield json.dumps({"error": f"Error at depth {i}: {str(e)}", "depth": i}, ensure_ascii=False) + "\n"
                    print(f"Error at depth {i}: {e}.{messages}")
                    break

        yield json.dumps({"done": True, 'message': messages}, ensure_ascii=False) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


@app.post("/tools")
async def get_tools(request: ToolRequest, session: AsyncSession = Depends(get_db_session)):
    """
    返回 OpenAI 兼容的 tools 定义，并调用模型接口后解析执行结果。
    如果 request.tools 为空且 request.user 存在，则从 Redis 获取用户缓存的 tools。否则调用 AI 自动生成 tool 元数据。
    最终根据 messages 与 tools 调用大模型，解析 tool_call 执行结果并返回。
    """
    if not request.messages:
        request.messages = create_analyze_messages(System_content.get('31'), request.prompt)
    tools_metadata = request.tools
    if not request.tools:
        if request.user:  # get_access_user(request)
            tools_metadata = await RegistryMetadata.get_metadata(session, request.user)
        else:
            tools_metadata = AI_Tools + await app.state.func_manager.tools_metadata(
                model_name=request.model_metadata, registered=True)
            # print(tools_metadata)

    if not request.model_name:
        return JSONResponse(tools_metadata)

    if not tools_metadata:
        tools_metadata = await app.state.func_manager.get_tools_metadata(func_list=[], redis=app.state.redis)

    completion, _ = await ai_client_completions(messages=request.messages, client=None, model=request.model_name,
                                                get_content=False, tools=tools_metadata or AI_Tools,
                                                top_p=request.top_p, temperature=request.temperature,
                                                dbpool=app.state.collector)

    tool_messages = completion.choices[0].message
    if not tool_messages:
        raise HTTPException(status_code=500, detail="No response from AI model.")

    if request.tools:  # 自定义tools直接返回
        return JSONResponse(tool_messages)

    # 解析响应并调用工具
    return JSONResponse(await ai_tools_results(tool_messages, app.state.func_manager))


@app.get("/tools/mcp")
async def get_mcp_tool():
    await app.state.func_manager.register_mcp_tools()
    mcp_tools = await app.state.func_manager.get_mcp_tools()
    return JSONResponse(content={k: v.to_mcp_tool().model_dump() for k, v in mcp_tools.items()})


@app.get('/tools/classes/{class_name}')
async def get_class_info(class_name: str):
    """获取特定类的详细信息"""
    try:
        return app.state.func_manager.class_method.get_class_info(class_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post('/tools/call/<class_name>')
async def call_registered_method(class_name: str, request: FunctionCallItem):
    """调用注册的类方法"""
    try:
        method_name = request.function
        result = app.state.func_manager.class_method.call_method(class_name, method_name, *request.args,
                                                                 **request.kwargs)
        return JSONResponse({"result": dataclass2dict(result), "class": class_name, "method": method_name})
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except TypeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/metadata/")
async def generate_metadata_from_code(req: MetadataRequest, session: AsyncSession = Depends(get_db_session)):
    cache_key = generate_hash_key(req.metadata, req.function_code, req.description)
    cached_metadata = await RegistryMetadata.get(session, cache_key)
    if cached_metadata:
        metadata = cached_metadata.get("metadata")
        func_name = cached_metadata.get("name")  # metadata.get('function', {}).get('name')
        # print(f"Metadata already cached for function: {func_name}")
        return {"metadata": metadata, "name": func_name, "id": cache_key}

    metadata = await ai_generate_metadata(
        req.function_code, req.metadata, req.model,
        description=req.description, code_type=req.code_type or "Python",
        dbpool=app.state.collector)

    registry = metadata.get("function", {}).copy()
    func_name = registry.get("name")
    if not func_name:
        raise HTTPException(status_code=400, detail="函数名未能从模型生成结果中提取")
    await RegistryMetadata.add(session, req, metadata, cache_key, func_name)
    if req.callback:
        registry['callback'] = req.callback.model_dump()
    registry["x-user"] = req.user
    await FunctionManager.set_metadata_to_cache(registry, cache_key, func_name, redis=app.state.redis, ex=req.cache_sec)
    return {
        "metadata": metadata,
        "registry": registry,
        "name": func_name,
        "id": cache_key
    }


@app.post("/agent/")
async def agent_run(request: AgentRequest):
    redis = app.state.redis or get_redis()
    metadata = await app.state.func_manager.get_tools_metadata(redis=redis)
    description = request.messages[-1].content
    messages = [msg.dict() for msg in request.messages]

    task_id, smart_task = await TaskManager.add(redis=redis, action='ai_client_completions',
                                                function=ai_client_completions,
                                                description=description,
                                                params={'messages': messages, 'model': request.model})

    task_id, extract_task = await TaskManager.add(redis=redis, action='extract_json_struct',
                                                  function=extract_json_struct,
                                                  description='extract_json')

    graph_node = await app.state.ai_scheduler.set_node(smart_task)
    # await app.state.ai_scheduler.build_subgraph(nodes, edges)

    try:
        # scheduler.update_task_status(task_id, TaskStatus.COMPLETED)  # source status:edge["condition"]
        # 触发依赖任务
        app.state.ai_scheduler.check_trigger_tasks()
        return {"message": f"Task {task_id} executed successfully.{app.state.ai_scheduler.export_nodes()}"}
    except Exception as e:
        return {"error": str(e)}


# @app.api_route("/dask-dashboard/{full_path:path}", methods=["GET", "POST"], operation_id="proxy_dask_dashboard")
# async def proxy_dask_dashboard(full_path: str, request: Request):
#     return await proxy_http_html(Config.DASK_DASHBOARD_HOST, full_path, request)
#     return RedirectResponse(url=f'{Config.DASK_DASHBOARD_HOST}/status')  # edirect


@app.post("/assistant/")
async def assistant_run(request: AssistantRequest):
    # Call the ai_assistant_run function with user input
    result = await ai_assistant_run(
        user_request=request.question,
        instructions=request.prompt,
        user_name=request.user_name,
        tools_type=request.tools_type,
        model_id=request.model_id,
        max_retries=20,
        interval=5
    )

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return result


@app.post("/callback")
async def handle_callback(request: Request):
    raw_body = await request.body()
    try:
        data = await request.json()
    except Exception as e:
        print(f"[JSON 解析失败]: {e}")
        return JSONResponse(status_code=400, content={"error": f"Invalid JSON:{raw_body.decode()}"})

    if isinstance(data, dict):
        result = data.get('result') or data.get('transform') or data
        if not result and data.get('answer'):
            print(f"Answer: {data.get('answer')}")
    elif isinstance(data, list):
        result = data
    else:
        result = {}

    print('callback:', result)

    return {"status": "success", "result": result}


@app.post("/llm")
async def generate_batch_text(request: CompletionParams,
                              use_task: bool = Query(False, description="非流式是否启用任务异步处理模式")):
    prompt = request.prompt or System_content.get(request.agent, 'You are a helpful assistant.')
    questions = request.question if isinstance(request.question, list) else [request.question]

    async def process_one(item):
        sub_q, idx = item
        model_info, payload, refer = await get_generate_payload(
            prompt=prompt,  # 优先使用
            user_request=sub_q,  # 如果空则是 prompt
            suffix=request.suffix, stream=request.stream,
            **request.asdict(
                include={'temperature', 'top_p', 'max_tokens', "model_name", "model_id", "agent", "keywords"})
        )

        # 如果要流式，用异步生成器形式返回,非流式就直接拿完整字符串
        if request.stream:
            payload["stream"] = True
            return await ai_generate(model_info, payload, get_content=True)  # async generator

        @async_error_logger(max_retries=Config.MAX_RETRY_COUNT, delay=1)
        async def try_ai_generate():
            content = await ai_generate(model_info, payload, get_content=True)
            if 'Error code: 429' in content or 'error' in content.lower():
                raise RuntimeError(f"AI生成失败： {content.strip()}")

            return {'question': sub_q, 'answer': content, 'reference': refer,
                    'transform': extract_string(content, request.extract), 'id': idx}

        return await try_ai_generate()

    if request.stream:
        async def batch_stream_response() -> AsyncGenerator[str, None]:
            num = len(questions)
            for idx, sub_q in enumerate(questions):
                if num > 1:
                    yield f"\n({idx}) {sub_q}\n"  # 子请求之间加一个换行分隔
                stream_fn = await process_one((sub_q, idx))
                assert hasattr(stream_fn,
                               "__aiter__"), f"process_one did not return async generator, got {type(stream_fn)}"
                async for chunk in stream_fn:
                    yield chunk
                    await asyncio.sleep(0.01)  # Token 逐步返回的延迟
                await asyncio.sleep(0.03)

                # media_type="text/plain",纯文本数据

        return StreamingResponse(batch_stream_response(), media_type="text/plain")

    process_func = run_togather(max_concurrent=Config.REDIS_MAX_CONCURRENT)(process_one)

    if use_task:
        redis = app.state.redis or get_redis()

        @TaskManager.task_node(action="llm", redis=redis, description=prompt, params=request.model_dump(),
                               data={"questions": questions})
        async def process_task(sub_q_list: list):
            results = await process_func(inputs=sub_q_list)  # 这会得到 List[dict]
            for i, r in enumerate(results):
                if isinstance(r, Exception):
                    app.state.logger.error(f"[任务 {i} 异常] {r}")

            return [r for r in results if isinstance(r, dict)]

        task_id = await process_task(sub_q_list=[(sub_q, idx) for idx, sub_q in enumerate(questions)])
        return JSONResponse(content={'task_id': task_id, 'url': f'{Config.WEBUI_URL}/task/{task_id}',
                                     'result': f'{Config.WEBUI_URL}/get/{TaskManager.key_prefix}:{task_id}',
                                     'html': f'{Config.WEBUI_URL}/task/{task_id}?platform=html',
                                     'file': f'{Config.WEBUI_URL}/task/{task_id}?platform=file'})

    results = await process_func(inputs=[(sub_q, idx) for idx, sub_q in enumerate(questions)])
    return JSONResponse(results[0] if len(results) == 1 else results)


@app.post("/message/")
async def generate_message(request: ChatCompletionRequest,
                           session: AsyncSession = Depends(get_db_session)) -> StreamingResponse or JSONResponse:
    # data = await request.json()
    # print(request.__dict__)
    if not request.messages and not request.question:
        return JSONResponse(status_code=400,
                            content={'answer': 'error',
                                     'error': 'Please provide messages or a question to process.'})

    agent = request.agent
    extract = request.extract
    chat_history = ChatHistory(request.user, request.name, request.robot_id, agent, request.model_name,
                               timestamp=time.time(), session_uid=request.request_id)

    if not extract:
        agent_format = {
            '3': 'code.python',
            '2': 'json',
            '4': 'json',
            '5': 'code.sql',
            '6': 'header',
        }
        extract = agent_format.get(agent, request.extract)

    history = await chat_history.build(request.question, request.messages or [], request.use_hist,
                                       request.filter_limit, request.filter_time, session=session)

    system_prompt = request.prompt or System_content.get(agent, System_content['0'])  # system_instruction

    model_info, payload, refer = await get_chat_payload(
        messages=history, user_request=chat_history.user_request, system=system_prompt, **request.payload())

    if request.stream:
        async def generate_stream() -> AsyncGenerator[str, None]:
            if refer:
                first_data = json.dumps({'role': 'reference', 'content': refer}, ensure_ascii=False)
                yield f'data: {first_data}\n\n'

            async for content, data in ai_chat_stream(model_info, payload):
                # yield f'data: {data}\n\n'
                if content and content.strip():
                    yield f'data: {content}\n\n'
                await asyncio.sleep(0.01)

            completion = json.loads(data)
            assistant_content = completion.get('choices', [{}])[0].get('message', {}).get('content')
            transform = extract_string(assistant_content, extract)
            last_data = json.dumps({'role': 'assistant', 'content': assistant_content, 'transform': transform},
                                   ensure_ascii=False)  # 转换字节流数据
            yield f'data: {last_data}\n\n'
            yield 'data: [DONE]\n\n'

            await chat_history.save(assistant_content, refer, transform, model=payload['model'],
                                    usage=completion.get('usage', {}), session=session)
            if request.callback and request.callback.url:
                result = {'answer': assistant_content, 'reference': refer, 'transform': transform}
                await send_callback(request.callback.model_dump(), transform if isinstance(transform, dict) else result)

        return StreamingResponse(generate_stream(), media_type="text/event-stream")
    else:
        assistant_content, completion = await ai_chat(model_info, payload)
        transform = extract_string(assistant_content, extract)
        await chat_history.save(assistant_content, refer, transform, model=payload['model'],
                                usage=completion.get('usage', {}), session=session)

        result = {'answer': assistant_content, 'reference': refer, 'transform': transform}
        if request.callback and request.callback.url:
            await send_callback(request.callback.model_dump(), transform if isinstance(transform, dict) else result)

        return JSONResponse(result)


@app.post("/v1/embeddings")
async def get_embeddings(request: OpenAIEmbeddingRequest):
    kwargs = request.model_dump(exclude_unset=True, exclude={'input', 'model'})
    response = await ai_embeddings(request.input, model_name=request.model, model_id=0, get_embedding=False, **kwargs)
    return JSONResponse(content=response)


# /v1/files
# /v1/audio/transcriptions
@app.post("/v1/completions", response_model=OpenAIResponse)
async def completions(request: OpenAIRequest, session: AsyncSession = Depends(get_db_session)) -> Union[
    OpenAIResponse, StreamingResponse]:
    kwargs = request.model_dump(exclude_unset=True, exclude={'prompt', 'model', 'stream', 'user'})
    if request.stream:
        async def stream_response() -> AsyncGenerator[dict, None]:
            stream_fn = await ai_generate(model_info=None, prompt=request.prompt, user_request='',
                                          model_name=request.model, stream=True, get_content=False, **kwargs)
            async for chunk in stream_fn:
                yield f"data: {chunk}\n\n"
                await asyncio.sleep(0.01)

            yield 'data: [DONE]\n\n'

        return StreamingResponse(stream_response(), media_type="text/event-stream")  # 纯文本数据
    else:
        response = await ai_generate(model_info=None, prompt=request.prompt, user_request='',
                                     model_name=request.model, stream=False, get_content=False, **kwargs)

        instance = BaseReBot(user_content=request.prompt, model=request.model, agent='generate')
        if request.suffix:
            instance.user_content += request.suffix
        await instance.save(request.user, instance=instance, model_response=response, session=session)

        return JSONResponse(content=response)


# @require_api_key
@app.post("/v1/chat/completions", response_model=OpenAIResponse)
async def chat_completions(request: OpenAIRequestMessage):
    """
    兼容 OpenAI API 的 /v1/chat/completions 路径，返回类似 OpenAI API 的格式
    """
    # print(request.dict())
    kwargs = request.payload()
    messages = [msg.dict() for msg in request.messages]
    instance = BaseReBot(model=request.model, user=request.user, agent='chat')
    if request.stream:
        async def fake_stream_response():
            async for content, data in ai_chat_stream(model_info=None, messages=messages, user_request=None,
                                                      system=None, model_name=request.model, **kwargs):
                yield f"data: {data}\n\n"
                await asyncio.sleep(0.01)

            yield 'data: [DONE]\n\n'
            await instance.async_save(user=request.user, messages=messages, model_response=data,
                                      dbpool=app.state.collector, instance=instance)

        return StreamingResponse(fake_stream_response(), media_type="text/event-stream")

    content, completion = await ai_chat(model_info=None, messages=messages, user_request=None, system=None,
                                        model_name=request.model, model_id=0, **kwargs)

    await instance.async_save(user=request.user, messages=messages, model_response=completion,
                              dbpool=app.state.collector, instance=instance)

    return JSONResponse(content=completion)  # Response(content=json.dumps(data), media_type="application/json")


@app.get("/v1/models")
async def get_models(model: Optional[str] = Query(None,
                                                  description=f"Retrieves a model instance, providing basic information about the model such as the owner and permissioning. e.g., {','.join(model_api_keys().keys())} or custom models.")):
    if model:
        response_data = ModelList.get_model_data(model)
    else:
        response_data = ModelList.get_models(AI_Models)
    return JSONResponse(content=response_data)


@app.get("/models/")
async def get_models_list(model_type: Literal["model", "embedding", "reranker"] = Query("model",
                                                                                        description=f"custom models/embedding/reranker.")):
    if model_type == "model":
        return JSONResponse(content=ModelList.models)
    return JSONResponse(content=ModelList.extract_models(model_type, AI_Models))


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    db = next(get_db())  # async for db in get_db():
    await websocket.accept()
    try:
        while True:
            request = await websocket.receive_json()  # await websocket.receive_text()
            if not request.get('messages') and not request.get('question'):
                await websocket.send_text(
                    json.dumps({'error': 'Please provide messages or a question to process.'}))
                continue

            model_name = request.get('model_name', "moonshot")
            agent = request.get('agent', '0')
            extract = request.get('extract')
            name = request.get('username')
            user = request.get('user')
            robot_id = request.get('robot_id')
            current_timestamp = time.time()

            # 构建聊天历史记录
            history, user_request, hist_size = build_chat_history(
                request.get('question'), user, name, robot_id, db=db,
                user_history=request.get('messages', []), use_hist=request.get('use_hist', True),
                filter_limit=request.get('filter_limit', -500), filter_time=request.get('filter_time', 0.0),
                agent=agent, session_uid=request.get('session_uid')
            )

            # 生成系统提示和模型请求
            system_prompt = request.get('prompt') or System_content.get(agent, '')

            model_info, payload, refer = await get_chat_payload(
                messages=history, user_request=user_request,
                system=system_prompt, **request.payload()
            )
            if request.get('stream', True):
                async def generate_stream() -> AsyncGenerator[str, None]:
                    if refer:
                        first_data = json.dumps({'role': 'reference', 'content': refer}, ensure_ascii=False)
                        yield f'data: {first_data}\n\n'

                    async for content, data in ai_chat_stream(model_info, payload):
                        if content and content.strip():
                            yield f'data: {content}\n\n'
                        await asyncio.sleep(0.01)

                    completion = json.loads(data)
                    assistant_content = completion.get('choices', [{}])[0].get('message', {}).get('content')
                    transform = extract_string(assistant_content, extract)
                    last_data = json.dumps({'role': 'assistant', 'content': assistant_content, 'transform': transform},
                                           ensure_ascii=False)
                    yield f'data: {last_data}\n\n'
                    yield 'data: [DONE]\n\n'

                    # 保存聊天记录
                    save_chat_history(
                        user_request, assistant_content, user, name, robot_id, agent,
                        hist_size, model_name, current_timestamp, db=db,
                        refer=refer, transform=transform, session_uid=request.get('session_uid'))

                # 流式传输消息到 WebSocket
                async for stream_chunk in generate_stream():
                    await websocket.send_text(stream_chunk)

            else:  # 非流式响应处理
                assistant_content, completion = await ai_chat(model_info, payload)
                transform = extract_string(assistant_content, extract)

                # 保存聊天记录
                save_chat_history(
                    user_request, assistant_content, user, name, robot_id, agent,
                    hist_size, model_name, current_timestamp, db=db,
                    refer=refer, transform=transform, session_uid=request.get('session_uid')
                )

                await websocket.send_text(
                    json.dumps({'answer': assistant_content, 'reference': refer, 'transform': transform}))
                # await asyncio.sleep(0.1)

            if system_prompt.lower() == "bye":
                await websocket.send_text("Closing connection")
                break

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Connection error: {e}")
    finally:
        await websocket.close()
        db.close()


@app.post("/batch/submit")
async def create_batch(request: OpenAIRequestBatch):
    """
    上传消息列表并启动批处理任务
    """
    messages_list = [[msg.model_dump() for msg in m] for m in request.messages_list]
    kwargs = request.payload()
    total = len(messages_list)
    data = {"messages": messages_list, "size": total}
    params = {"model": request.model, **kwargs}

    redis = app.state.redis or get_redis()
    task_id, task = await TaskManager.add(redis=redis,
                                          description='批处理任务',
                                          action='batch',
                                          params=params,
                                          data=data,
                                          # status=TaskStatus.PENDING,
                                          start_time=time.time(),
                                          ex=86400 * 2)
    if request.completion_window == '24h':
        result = await ai_batch(inputs=messages_list, task_id=task_id, model=request.model, search_field="model",
                                **kwargs)
        if isinstance(result, dict) and result.get("batch_id"):
            params.update(result)
            await TaskManager.put_task_result(task, result=result, total=total, status=TaskStatus.READY, params=params,
                                              redis_client=redis)
        else:
            task.status = TaskStatus.FAILED
        return {
            "task_id": task_id, "status": task.status.value,
            "url": f'{Config.WEBUI_URL}/batch/{task_id}?interval=-1',
            "file": f'{Config.WEBUI_URL}/files/{result.get("input_file")}',
            **result
        }

    async def _process(task_id):
        results = await ai_batch_run(variable_name='messages', variable_values=messages_list, model=request.model,
                                     dbpool=app.state.collector, **kwargs)
        status = TaskStatus.FAILED if all(r.get("error") for r in results) else TaskStatus.COMPLETED
        return await TaskManager.update_task_result(task_id, result=results, status=status, redis_client=redis)

    asyncio.create_task(_process(task_id))
    return JSONResponse(content={'task_id': task_id, 'url': f'{Config.WEBUI_URL}/task/{task_id}',
                                 'result': f'{Config.WEBUI_URL}/get/{TaskManager.key_prefix}:{task_id}',
                                 'html': f'{Config.WEBUI_URL}/task/{task_id}?platform=html',
                                 'file': f'{Config.WEBUI_URL}/task/{task_id}?platform=file'})


@app.get("/batch/{task_id}")
async def get_batch_result(task_id: str, interval: int = 10, timeout: int = 300, oss_expires: int = 0):
    redis = app.state.redis or get_redis()
    task = await TaskManager.get_task(task_id, redis)
    waiting_status = (TaskStatus.PENDING, TaskStatus.READY, TaskStatus.IN_PROGRESS)
    if not task:
        error_data = {"error": "Invalid task ID", 'messages': task_id, "status": "not_found"}
        return JSONResponse(content=error_data, status_code=404)
    if task.action != 'batch':
        error_data = {"error": "Invalid task action", 'messages': task.action, "status": task.status.value}
        return JSONResponse(content=error_data, status_code=400)
    if task.count <= 0:
        error_data = {"error": "Invalid task count", 'messages': task.count, "status": task.status.value}
        return JSONResponse(content=error_data, status_code=400)

    if task.status not in waiting_status:
        result = task.result[-1]  # 如果任务状态不是等待状态，直接返回最后的结果
    else:
        params = task.params or task.result[0]
        batch_id = params.get("batch_id")
        client_name = params.get("client_name")

        result = await ai_batch_result(
            batch_id=batch_id,
            task_id=task_id,
            client_name=client_name,
            interval=interval,
            timeout=timeout,
            oss_expires=oss_expires
        )
        status_map = {"completed": TaskStatus.COMPLETED, "failed": TaskStatus.FAILED, "expired": TaskStatus.CANCELLED,
                      "cancelled": TaskStatus.CANCELLED, "timeout": TaskStatus.READY,
                      "in_progress": TaskStatus.IN_PROGRESS}
        mapped_status = status_map.get(result["status"], TaskStatus.IN_PROGRESS)
        total = -1 if mapped_status in waiting_status else 1
        await TaskManager.put_task_result(task, result=result.copy(), total=total, status=mapped_status,
                                          redis_client=redis)

        result['batch_id'] = batch_id

    result['url'] = f'{Config.WEBUI_URL}/task/{task_id}'
    if "result_file" in result:
        result['result'] = f'{Config.WEBUI_URL}/files/{result.get("result_file")}'
    if "error_file" in result:
        result['error'] = f'{Config.WEBUI_URL}/files/{result.get("error_file")}'
    return result


@app.get("/message/get/")
async def get_messages(request: Request, user: str = Query(""), robot_id: Optional[str] = None,
                       agent: Optional[str] = None, filter_time: float = Query(0.0),
                       session: AsyncSession = Depends(get_db_session)):
    """
    获取用户消息历史：
    1. 优先从 JWT token（Authorization / cookie / query）中识别真实用户；
    2. 否则使用 session 内的 user_id 作为匿名访客；
    """
    access_user = get_access_user(request)  # 优先：从 token 认证用户
    session_uid = request.session.get('user_id')
    uid = access_user or session_uid
    if not (user or uid):
        return JSONResponse(status_code=400, content={"error": "No user id found in session"})

    if filter_time > 1e12:  # 很可能是毫秒
        filter_time = filter_time / 1000.0
    # filter_history = get_user_history(user, access_user, robot_id, filter_time, db, agent=agent,session_uid=session_uid)
    filter_history = await BaseChatHistory.async_user_history(session, user, uid, robot_id, agent=agent,
                                                              filter_time=filter_time)
    for item in filter_history:
        if isinstance(item.get('created_at'), datetime):
            item['created_at'] = item['created_at'].isoformat()
    return JSONResponse(content=sorted(filter_history, key=lambda x: x['timestamp']))


# @app.route('/chat_ui')
# def chat_ui():
#     return redirect("http://host.docker.internal:8080")

@app.post("/message/submit")
async def submit_messages(request: Request, body: SubmitMessagesRequest,
                          session: AsyncSession = Depends(get_db_session)):
    if TaskManager.size() > Config.MAX_TASKS:
        return JSONResponse(status_code=400, content={'task_id': '', "error": "任务队列已满"})
    if not body.messages and not body.params:
        return JSONResponse(status_code=400,
                            content={'task_id': '', 'error': 'Please provide messages or a question to process.'})

    name = body.name or get_access_user(request)
    current_timestamp = time.time()
    chat_history = ChatHistory(body.user, name, body.robot_id, agent=None, model=None, timestamp=current_timestamp,
                               session_uid=body.request_id)

    history: List[dict] = await chat_history.build('', body.messages or [], body.use_hist,
                                                   body.filter_limit, body.filter_time, session=session)
    data = {"messages": history, "size": chat_history.index, "user": body.user, "name": name,
            "robot_id": body.robot_id, 'session_uid': body.request_id}
    # generate_hash_key(data,datetime.datetime.now().date().isoformat())
    redis = app.state.redis or get_redis()
    task_id, task = await TaskManager.add(redis=redis,
                                          description=chat_history.user_request,
                                          action='message',
                                          params=body.params,  # List[CompletionParams]
                                          data=data,
                                          # status=TaskStatus.PENDING,
                                          start_time=current_timestamp)

    if not body.params:  # task.params
        return JSONResponse(content={'task_id': task_id, 'url': f'{Config.WEBUI_URL}/message/{task_id}'})
    if not task:
        return JSONResponse(status_code=400,
                            content={'task_id': task_id, 'error': '[process_task_ai] Task not found in Task_map.'})

    async def process_task_ai(task: TaskNode):
        await TaskManager.set_task_status(task, TaskStatus.IN_PROGRESS, 0, redis)
        task_id = task.name
        params: List[CompletionParams] = task.params
        user_request = chat_history.user_request or task.description

        # run_generator
        @run_togather(max_concurrent=Config.MAX_CONCURRENT, batch_size=-1)
        async def single_param(data: tuple[int, CompletionParams]):
            i, param = data
            if param.stream:
                param.stream = False
            if not param.question:
                param.question = user_request
            # else:
            #     chat_history.user_request = param.question
            user_history = chat_history.rebuild(param.question, history) if history else []
            system_prompt = param.prompt or System_content.get(param.agent, '')
            model_info, payload, refer = await get_chat_payload(messages=user_history, user_request=param.question,
                                                                system=system_prompt, **param.payload())
            # **param.asdict(),payload=param.payload()
            assistant_content, completion = await ai_chat(model_info, payload)
            transform = extract_string(assistant_content, param.extract)

            chat_history.agent = param.agent
            await chat_history.save(assistant_content, refer, transform, param.question,
                                    model=payload.get('model', param.model_name), usage=completion.get('usage', {}))

            if param.extract == 'wechat':
                if chat_history.name and chat_history.robot_id:
                    await send_to_wechat(chat_history.name, transform or assistant_content)
                    # send_wechat(transform or assistant_content, chat_history.name)
            result = {'answer': assistant_content, 'reference': refer, 'transform': transform,
                      'question': param.question, 'id': i}

            if param.callback and param.callback.url:
                await send_callback(param.callback.model_dump(),
                                    result=transform if isinstance(transform, (dict, list)) else result)
            if body.stream:
                await TaskManager.put_task_result(task, result, len(params), redis_client=redis)

            return result

        results = await single_param(inputs=[(i, p) for i, p in enumerate(params)])
        status = TaskStatus.COMPLETED if all(isinstance(r, dict) for r in results) else TaskStatus.FAILED
        await TaskManager.update_task_result(task_id, result=[r for r in results if isinstance(r, dict)],
                                             status=status, redis_client=redis)

        for i, r in enumerate(results):
            if isinstance(r, Exception):
                app.state.logger.error(f"[任务 {i} 异常] {r}")

        return results
        # executor.submit(ai_chat, *args) future.result()   await asyncio.wait_for(ai_chat(*args), timeout=10)

    asyncio.create_task(process_task_ai(task))  # background_tasks:background_tasks.add_task(process_task_ai, task_id)
    return JSONResponse(content={'task_id': task_id, 'url': f'{Config.WEBUI_URL}/task/{task_id}',
                                 'result': f'{Config.WEBUI_URL}/get/{TaskManager.key_prefix}:{task_id}',
                                 'html': f'{Config.WEBUI_URL}/task/{task_id}?platform=html',
                                 'file': f'{Config.WEBUI_URL}/task/{task_id}?platform=file'})


async def get_ai_param(
        stream: bool = Query(False, description="Enable streaming response if set to True."),
        temperature: float = Query(0.8,
                                   description="Controls randomness. Lower values make output more focused and deterministic, while higher values increase creativity."),
        top_p: float = Query(0.8,
                             description="Nucleus sampling parameter. Limits the response to the smallest set of tokens whose cumulative probability exceeds this value."),
        max_tokens: Optional[int] = Query(1024,
                                          description="The maximum number of tokens the model can generate in the response."),
        thinking: Optional[int] = Query(0, ge=0,
                                        description="Number of thinking tokens. 0 disables thinking; >0 enables thinking and adds token budget."),
        prompt: Optional[str] = Query(default=None,
                                      description="The initial context or system message to guide the AI's behavior."),
        question: Optional[str] = Query(default=None,
                                        description="The main question or user prompt for the AI to respond to."),
        agent: Optional[str] = Query(default='0',
                                     description="Contextual identifier for different use cases, enabling selection of appropriate system behavior."),
        model_name: str = Query("moonshot",
                                description=f"Specify the model to use, e.g., {','.join(model_api_keys().keys())} or custom models."),
        model_id: int = Query(0,
                              description="An optional model ID for selecting different versions or configurations of a model."),
        extract: Optional[str] = Query(None,
                                       description="Specify the type of content to extract from the AI's response (e.g., key phrases, summaries)."),
        keywords: Optional[List[Any]] = Body(None,
                                             description="A list of keywords used to guide the retrieval of relevant information or sources based on search terms."),
        tools: Optional[List[Dict]] = Body(None, description="tools 参数"),
) -> CompletionParams:
    """
     Asynchronously retrieves the AI parameters based on user input and returns them as an CompletionParams object.
     把 query + body 合并到一个 CompletionParams
    """
    return CompletionParams(
        stream=stream,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        thinking=thinking,
        prompt=prompt,
        question=question,
        agent=agent,
        model_name=model_name,
        model_id=model_id,
        extract=extract,
        keywords=keywords,
        tools=tools
    )


@app.get("/message/{task_id}")
async def response_message(task_id: str, param: CompletionParams = Depends(get_ai_param),
                           session: AsyncSession = Depends(get_db_session)) -> StreamingResponse or JSONResponse:
    redis = app.state.redis or get_redis()
    task = await TaskManager.get_task(task_id, redis)

    if not task:
        error_data = {"error": "Invalid task ID", 'messages': task_id, "status": "not_found"}
        return JSONResponse(content=error_data, status_code=404)

    if task.action != 'message':
        error_data = {"error": "Invalid task action", 'messages': task.action, "status": task.status.value}
        return JSONResponse(content=error_data, status_code=400)

    if task.status == TaskStatus.IN_PROGRESS:
        # params: List[CompletionParams] = task.params
        error_data = {"error": "Task already in progress", 'messages': task.action, "status": task.status.value}
        return JSONResponse(content=error_data, status_code=202)

    if task.status in (TaskStatus.COMPLETED, TaskStatus.RECEIVED):
        await TaskManager.set_task_status(task, TaskStatus.RECEIVED, 100, redis)
        return JSONResponse(content=task.result)

    await TaskManager.set_task_status(task, TaskStatus.IN_PROGRESS, 10, redis)
    history: List[dict] = task.data.get('messages', [])
    chat_history = ChatHistory(task.data.get('user'), task.data.get('name'), task.data.get('robot_id'),
                               agent=param.agent, model=param.model_name,
                               timestamp=task.start_time, index=task.data.get("size"),
                               session_uid=task.data.get('session_uid'),
                               user_request=task.description, user_history=history)

    if not param.question:
        param.question = task.description

    system_prompt = param.prompt or System_content.get(param.agent, '')
    model_info, payload, refer = await get_chat_payload(messages=history, user_request=param.question,
                                                        system=system_prompt, **param.payload())

    if param.stream:
        async def generate():
            if refer:
                first_data = json.dumps({'role': 'reference', 'content': refer}, ensure_ascii=False)
                yield f'data: {first_data}\n\n'

            async for content, data in ai_chat_stream(model_info, payload):
                if content and content.strip():
                    yield f'data: {content}\n\n'
                await asyncio.sleep(0.01)

            completion = json.loads(data)
            assistant_content = completion.get('choices', [{}])[0].get('message', {}).get('content')
            transform = extract_string(assistant_content, param.extract)
            last_data = json.dumps({'role': 'assistant', 'content': assistant_content, 'transform': transform},
                                   ensure_ascii=False)
            yield f'data: {last_data}\n\n'
            yield 'data: [DONE]\n\n'

            await chat_history.save(assistant_content, refer, transform, user_request=param.question,
                                    model=payload['model'], usage=completion.get('usage', {}), session=session)

            result = {'answer': assistant_content, 'reference': refer, 'transform': transform, 'id': 0}
            await TaskManager.update_task_result(task_id, result=[result], status=TaskStatus.RECEIVED,
                                                 redis_client=redis)

        return StreamingResponse(generate(), media_type="text/event-stream")

    assistant_content, completion = await ai_chat(model_info, payload)
    transform = extract_string(assistant_content, param.extract)

    await chat_history.save(assistant_content, refer, transform, user_request=param.question,
                            model=payload['model'], usage=completion.get('usage', {}), session=session)

    result = {'answer': assistant_content, 'reference': refer, 'transform': transform, 'id': 0}
    await TaskManager.update_task_result(task_id, result=[result], status=TaskStatus.RECEIVED, redis_client=redis)
    return JSONResponse(content=result)


def cleanup_old_tempfiles(min_age=600, prefix="tmp_"):
    """删除超过 `min_age` 秒的临时文件（前缀匹配）"""
    temp_dir = tempfile.gettempdir()  # 获取系统临时目录
    # pattern = os.path.join(temp_dir, f"{prefix}*")  # 匹配前缀文件
    for filename in os.listdir(temp_dir):
        if not filename.startswith(prefix):
            continue
        file_path = os.path.join(temp_dir, filename)
        try:
            if not os.path.isfile(file_path):
                continue

            file_age = time.time() - os.path.getmtime(file_path)  # 文件最后修改时间
            if file_age > min_age:
                os.remove(file_path)
                print(f"已删除过期文件: {file_path}")
        except FileNotFoundError:
            pass  # 文件已被其他进程删除
        except PermissionError:
            print(f"权限不足，无法删除: {file_path}")
        except Exception as e:
            print(f"删除临时文件失败: {file_path} - {str(e)}")


@app.get("/task/{task_id}")
async def get_task_status(task_id: str, platform: Literal['json', 'file', 'html'] = 'json'):
    redis = app.state.redis or get_redis()
    task: TaskNode = await TaskManager.get_task(task_id, redis)
    if not task:
        return {'error': "Invalid task ID,Task not found", "status": "not_found"}

    status = task.status
    if status == TaskStatus.COMPLETED:
        await TaskManager.set_task_status(task, TaskStatus.RECEIVED, 100, redis)

    if task.result:
        if platform == 'file':
            tmp = tempfile.NamedTemporaryFile(mode="w+", delete=False, prefix="tmp_", suffix=".json", encoding="utf-8")
            json.dump(task.result, tmp, ensure_ascii=False, indent=2)
            tmp.flush()
            tmp.close()

            # file_path=tmp.name
            # 返回一个 FileResponse，客户端收到后就能下载 JSON 文件
            return FileResponse(tmp.name, media_type="application/json", filename=f"results_{task_id}.json")

        if platform == 'html':
            import constants
            text = format_summary_text(summary_data=task.result, title_map=constants.Title_Map)
            return HTMLResponse(content=format_for_html(text, True), status_code=200)

    return JSONResponse(
        content={"status": status.value, "action": task.action, "params": dataclass2dict(task.params),
                 "result": dataclass2dict(task.result), "count": task.count, "progress": task.progress})


# return [{"task_id": v["name"], "description": v["description"], "status": v["status"], 'action': v['action']} for v in Task_graph.vs]


# 执行任务
@app.post("/task/execute/")
def execute_task(task_id: str):
    scheduler = TaskGraphManager()
    try:
        scheduler.update_task_status(task_id, TaskStatus.COMPLETED)  # source status:edge["condition"]
        # 触发依赖任务
        scheduler.check_trigger_tasks()
        return {"message": f"Task {task_id} executed successfully."}
    except Exception as e:
        return {"error": str(e)}


@app.get("/location/")
async def search_location(query: str, region: str = Query('', description="Region or city to limit the search"),
                          platform: str = 'badiu'):
    if platform == 'badiu':
        result = await search_bmap_location(query, region) if region else get_bmap_location(query)
    else:
        result = await search_amap_location(query, region) if region else get_amap_location(query)
    return {"result": result}


@app.post("/translate")
async def translate_text(request: TranslateRequest):
    try:
        return await auto_translate(request.text, request.platform.lower(), request.source, request.target)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unsupported platform:{e}")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...), oss_expires: int = 86400):
    if not os.path.exists(Config.DATA_FOLDER):
        Path(Config.DATA_FOLDER).mkdir(parents=True, exist_ok=True)  # 确保目标文件夹存在

    raw_bytes = await file.read()
    if oss_expires != 0:
        object_name = f"upload/{file.filename}"
        file_obj = io.BytesIO(raw_bytes)
        url, _ = await asyncio.to_thread(upload_file_to_oss, AliyunBucket, file_obj, object_name, expires=oss_expires)
    else:
        file_path = Path(Config.DATA_FOLDER) / file.filename
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(raw_bytes)
        url = f"{Config.WEBUI_URL}/files/{file.filename}"

    return {"url": url}


@app.get("/files/{filename}")
async def file_handler(filename: str = None, url: str = None, stream: bool = False):
    data_folder = Path(Config.DATA_FOLDER)
    data_folder.mkdir(parents=True, exist_ok=True)  # 确保目标文件夹存在

    if url and is_url(url):
        file_path, file_name = await download_file(url, data_folder)
        if file_path:
            if stream:
                return StreamingResponse(file_path.open("rb"), media_type="application/octet-stream")
            return FileResponse(file_path)
    elif filename:
        file_path = data_folder / filename
        if file_path.exists():
            if stream:
                # media_type='application/octet-stream' 使用流式响应，二进制模式打开文件
                return StreamingResponse(file_path.open("rb"), media_type="application/octet-stream")
            return FileResponse(file_path)
    return {"error": "File not found"}


@app.post("/send_wechat_scheduler")
async def send_wechat_scheduler(request: Request, file: UploadFile = File(None)):
    form_data = await request.form()
    now = datetime.now()
    user_name = form_data.get('user_name')
    context = form_data.get('context')
    send_time = form_data.get('send_time')
    object_name = form_data.get('object_name', file.filename).strip()

    if user_name is None:
        raise HTTPException(status_code=500, detail="需要提供微信名称.")

    if send_time:
        try:
            send_time = datetime.fromisoformat(send_time)
        except ValueError:
            raise HTTPException(status_code=400,
                                detail="Invalid datetime format. Please use 'YYYY-MM-DDTHH:MM:SS' format.")
    else:
        send_time = now

    tigger_sec = (send_time - now).total_seconds()
    url = ''
    if file and file.filename:
        object_name = file.filename
        oss_expires = int(max(0, tigger_sec)) + 86400
        file_obj = io.BytesIO(await file.read())
        url, _ = upload_file_to_oss(AliyunBucket, file_obj, f"upload/{file.filename}", expires=oss_expires)

    if tigger_sec > 0:
        job_id = f"to_wechat_{generate_hash_key(user_name, context, url)}"
        if app.state.scheduler.get_job(job_id):
            print(f'job_existing and replaced: {job_id} | {form_data}')  # job.remove()
        else:
            print(f'job_created: {job_id} | scheduled at {send_time}')

        app.state.scheduler.add_job(send_to_wechat, 'date', id=job_id, run_date=send_time, misfire_grace_time=50,
                                    args=[user_name, context, url, object_name], replace_existing=True,
                                    jobstore='redis')
    else:
        await send_to_wechat(user_name, context, url, object_name)

    return {"name": user_name, "file": object_name, "url": url, "send_time": send_time, 'tigger_sec': tigger_sec}


# @app.post("/send_zero_mq")
# async def send_msg_zero_mq(message: str, topic: str = 'topic'):
#     await message_zero_mq.send_message(message, topic)
#     return {"status": "success", "message": f"Message '{message}' sent to ZeroMQ."}


@app.post("/conversations/extract")
async def extract_conversations(file: UploadFile = File(...), convo_type: Literal['gpt', 'deepseek'] = Form('gpt'),
                                stream: bool = Form(False)):
    from script.conversation import extract_conversations_records_gpt, extract_conversations_records_ds
    raw_bytes = await file.read()
    conversations_data = json.loads(raw_bytes)
    if convo_type == 'gpt':
        structured_data = await asyncio.to_thread(extract_conversations_records_gpt, conversations_data)
    else:
        structured_data = await asyncio.to_thread(extract_conversations_records_ds, conversations_data)

    if stream:
        json_bytes = json.dumps(structured_data, ensure_ascii=False).encode('utf-8')
        convo_io = io.BytesIO(json_bytes)
        convo_io.seek(0)
        return StreamingResponse(convo_io, media_type="application/json")

    tmp = tempfile.NamedTemporaryFile(mode="w+", delete=False, prefix="tmp_", suffix=".json", encoding="utf-8")
    json.dump(structured_data, tmp, ensure_ascii=False, indent=2)
    tmp.flush()
    tmp.close()
    return FileResponse(tmp.name, media_type="application/json", filename=f"conversations_structured.json")


@app.post("/conversations/filter")
async def filter_conversations(files: List[UploadFile] = File(...), after_date: str = Form(None),
                               stream: bool = Form(False), name: Optional[str] = Form(None)):
    from script.conversation import filter_messages_after, df_messages_sorted, messages_to_records_values

    structured_data = []
    for file in files:
        raw_bytes = await file.read()
        try:
            obj = json.loads(raw_bytes)
            if isinstance(obj, list):
                structured_data.extend(obj)
            else:
                structured_data.append(obj)
        except Exception:
            # 尝试纯文本
            try:
                structured_data.append(raw_bytes.decode('utf-8'))
            except Exception:
                raise HTTPException(status_code=400, detail=f"文件 {file.filename} 无法解析为 JSON/Excel/文本")

    if not structured_data:
        raise HTTPException(status_code=400, detail="没有可用的对话数据")

    after_ts = format_date_type(date=after_date).timestamp() if after_date else 0
    filtered_messages = await asyncio.to_thread(filter_messages_after, structured_data, after_ts)
    df_sorted = await asyncio.to_thread(df_messages_sorted, filtered_messages)
    if name:
        records = messages_to_records_values(df_sorted, user='conversations', name=name)
        asyncio.create_task(ChatHistory.update(records))
    if stream:
        json_bytes = json.dumps(df_sorted.to_dict(orient='records'), ensure_ascii=False).encode('utf-8')
        return StreamingResponse(io.BytesIO(json_bytes), media_type="application/json")

    tmp = tempfile.NamedTemporaryFile(delete=False, prefix="tmp_", suffix=".xlsx")
    tmp.close()

    df_sorted = df_sorted.astype(str).apply(lambda col: col.str.replace(r'[\x00-\x08\x0b-\x1f]', '', regex=True))
    df_sorted.to_excel(tmp.name, index=False)  # C:\Users\Admin\AppData\Local\Temp\tmp_a9c3axrq.xlsx
    return FileResponse(tmp.name, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        filename=f"conversations_filter_{after_date or 'all'}.xlsx")


@app.post("/summary")
async def summary_extract_text(file: Optional[UploadFile] = File(None), text: Optional[str] = Form(None),
                               extract_prompt: Optional[str] = Form(None), summary_prompt: Optional[str] = Form(None),
                               extract_model: str = Form('qwen:qwen-long'), summary_model: str = Form('qwen:qwen-plus'),
                               max_tokens: int = Form(4096), max_segment_length: int = Form(10000)):
    long_text: List = []

    # 处理 text
    if text:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                long_text.extend(parsed)
            else:
                long_text.append(parsed)
        except Exception:
            long_text.append(text)  # 当普通字符串

    # 处理文件
    if file:
        try:
            fname = (file.filename or "").lower()
            ctype = (file.content_type or "").lower()
            raw_bytes = await file.read()

            # Excel 文件
            if fname.endswith(('.xls', '.xlsx')) or 'sheet' in ctype or 'excel' in ctype:
                try:
                    import pandas as pd
                    df = pd.read_excel(io.BytesIO(raw_bytes))
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"无法解析 Excel 文件: {e}")
                long_text += df.to_dict(orient='records')
            else:
                # 尝试 JSON
                try:
                    structured_data = json.loads(raw_bytes)
                    if isinstance(structured_data, list):
                        long_text += structured_data
                    else:
                        long_text.append(structured_data)
                except Exception:
                    # 当作纯文本
                    try:
                        long_text.append(raw_bytes.decode('utf-8'))
                    except Exception:
                        raise HTTPException(status_code=400, detail="上传文件无法解析为 JSON/Excel/文本")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"处理文件失败: {e}")

    if not long_text:
        raise HTTPException(status_code=400, detail="No text or file provided")

    # 调用 AI 摘要
    async def event_generator():
        async for chunk in ai_summary(long_text, extract_prompt, summary_prompt, extract_model, summary_model,
                                      extract_max_tokens=max_tokens, summary_max_tokens=max_tokens,
                                      segment_max_length=max_segment_length, dbpool=app.state.collector):
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/ocr")
async def image_recognition(file: UploadFile = File(None), image_url: str = Form(None),
                            ocr_type: str = 'general'):
    try:
        image_data = await file.read() if file else None  # 从内存字节b"uploading.."
    except Exception as e:
        return {"error": f"Failed to process the uploaded file:{e}"}

    result = None
    if ocr_type in ['general', 'accurate', 'accurate_basic', 'general_basic', 'webimage', 'doc_analysis_office',
                    'table', 'numbers', 'qrcode', 'handwriting', 'idcard', 'account_opening']:
        result = await baidu_ocr_recognise(image_data, image_url, ocr_type=ocr_type)
    if ocr_type in ['GeneralBasicOCR', 'RecognizeTableDDSNOCR', 'GeneralAccurateOCR', 'VatInvoiceOCR',
                    'ImageEnhancement', 'QrcodeOCR', 'SmartStructuralOCRV2']:
        result = await tencent_ocr_recognise(image_data, image_url, ocr_type=ocr_type)
    if result is None:
        raise HTTPException(status_code=500, detail="OCR recognition failed.")

    return result


@app.post('/visual')
async def generate_image(file: UploadFile = File(None), image_urls: List[str] = Form(None), prompt: str = Form(None),
                         style_name: str = '角色特征保持', model_id: int = 0, return_url: bool = False):
    try:
        image_data = await file.read() if file else None  # 从内存字节b"uploading.."
    except Exception as e:
        return {"error": f"Failed to process the uploaded file:{e}"}

    image_urls = [url.strip() for url in image_urls if url.strip()] if image_urls else []

    if model_id == 0:
        image_decode, result = await ark_visual_picture(image_data, image_urls, prompt=prompt,
                                                        logo_info=None, style_name=style_name, return_url=return_url,
                                                        data_folder=None)
    elif model_id == 1:
        image_decode, result = await ark_drawing_picture(image_data, image_urls, whitening=1.0,
                                                         dermabrasion=1.2, logo_info=None, style_name=style_name,
                                                         return_url=return_url)
    else:
        if not image_urls and file:
            file_obj = io.BytesIO(image_data)
            object_name = f"webimg/{file.filename}"
            image_urls = [upload_file_to_oss(AliyunBucket, file_obj, object_name, expires=3600)[0]]

        if model_id == 2:
            image_decode, result = await wanx_image_generation(image_urls, style_name=style_name)
        elif model_id == 3:
            image_decode, result = await tencent_drawing_picture(image_data, image_urls[0] if image_urls else '',
                                                                 prompt=prompt, style_name=style_name,
                                                                 return_url=return_url)
            # image_decode = None
            # result = await ali_cartoon_picture(image_urls[0], style_name=style_name)
        else:
            image_decode, result = dashscope_image_call(prompt, image_url=image_urls[0] if image_urls else '',
                                                        style_name=style_name, model_name="wanx-v1", data_folder=None)

    if image_decode:
        file_basename = quote(os.path.splitext(file.filename)[0].strip()) if file else None
        content_disposition = f"attachment; filename*=UTF-8''{file_basename}.png" if file_basename else f"attachment; filename={result['id']}.png"
        # print(content_disposition)
        image_io = io.BytesIO(image_decode)
        image_io.seek(0)
        return StreamingResponse(image_io, media_type="image/png",  # data:image/jpeg;base64,xxxx
                                 headers={"Content-Disposition": content_disposition,
                                          "X-Request-ID": result['id'], "File-Url": result['urls'][0]})
    if 'image_urls' not in result:
        result['image_urls'] = image_urls
    return result


@app.post("/asr")
async def speech_to_text(file: UploadFile = File(None), file_urls: List[str] = Form(None),
                         platform: PlatformEnum = PlatformEnum.baidu, rate: int = 16000):  # File(...）
    # 获取音频数据
    if file:
        try:
            audio_data = io.BytesIO(await file.read())
            if file.content_type and file.content_type.startswith('audio/'):
                detected_format = file.content_type.split('/')[1]
            elif file.filename:
                detected_format = os.path.splitext(file.filename)[1].lower().lstrip('.')  # 'mpeg'
            else:
                detected_format = 'pcm'
            format_mapping = {
                'mpeg': 'mp3',
                'mp3': 'mp3',
                'x-mpeg': 'mp3',
                'mp4': 'm4a',
                'x-m4a': 'm4a',
                'aac': 'aac',
                'wav': 'wav',
                'x-wav': 'wav',
                'wave': 'wav',
                'flac': 'flac',
                'ogg': 'ogg',
                'oga': 'ogg'
            }  # wav、mp3、m4a、flv、mp4、wma、3gp、amr、aac、ogg-opus、flac
            format = format_mapping.get(detected_format, detected_format)
            print(format)
        except Exception as e:
            return {"error": f"Failed to process the uploaded file:{e}"}

        if platform == PlatformEnum.baidu:
            return await baidu_speech_to_text(audio_data, format, rate)
        elif platform == PlatformEnum.ali:
            return await ali_speech_to_text(audio_data, format, rate)
        elif platform == PlatformEnum.tencent:
            return await tencent_speech_to_text(audio_data)
        elif platform == PlatformEnum.dashscope:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{format}')
            try:
                temp_file.write(audio_data.getvalue())  # await file.read()
                temp_file.close()
                result = await dashscope_speech_to_text(temp_file.name, format=format, rate=rate)
            finally:
                os.remove(temp_file.name)
            return result

    elif file_urls:
        if isinstance(file_urls, list) and len(file_urls) == 1:
            file_urls = file_urls[0].split(',')
        elif isinstance(file_urls, str):
            file_urls = file_urls.split(',')

        file_urls = [url.strip(" '\"") for url in file_urls]
        if platform == PlatformEnum.tencent:
            return await tencent_speech_to_text(None, file_urls[0])
        result = await dashscope_speech_to_text_url(file_urls)
        return JSONResponse(content={"transcriptions": result})

    return HTTPException(status_code=400, detail="Unsupported platform.")

    # curl - X
    # POST
    # "http://127.0.0.1:8000/speech-to-text" - F
    # "file=@path_to_your_audio.pcm"


@app.post("/tts")
async def text_to_audio(sentences: str, platform: str = "cosyvoice-v1"):
    if platform in ["cosyvoice-v1"]:
        audio_data, request_id = await dashscope_text_to_speech(sentences, model=platform)
        audio_io = io.BytesIO(audio_data)
        audio_io.seek(0)
        return StreamingResponse(audio_io, media_type="audio/mpeg",
                                 headers={"Content-Disposition": f"attachment; filename={request_id}.mp3",
                                          "X-Request-ID": request_id})
    return HTTPException(status_code=400, detail="Unsupported platform.")


@app.post("/tti")
async def text_to_image(prompt: str, negative_prompt: str = '', style_name: str = '人像写真',
                        model_name: str = 'dashscope:wanx-v1'):
    owner = model_name
    if ':' in model_name:
        owner, model_name = model_name.split(':')
    if owner == 'dashscope':
        image_decode, result = dashscope_image_call(prompt, negative_prompt=negative_prompt, image_url='',
                                                    style_name=style_name, model_name=model_name, data_folder=None)

    elif owner == 'tencent':
        image_decode, result = await tencent_generate_image(prompt, negative_prompt, style_name, return_url=False)
    elif owner == 'siliconflow':
        image_decode, result = await siliconflow_generate_image(prompt, negative_prompt, model_name=model_name,
                                                                model_id=0)
    else:
        image_decode, result = await xunfei_picture(prompt, data_folder=None)

    if image_decode:
        image_io = io.BytesIO(image_decode)
        image_io.seek(0)
        return StreamingResponse(image_io, media_type="image/jpeg",
                                 headers={
                                     "Content-Disposition": f"attachment; filename={result.get('id', 'result')}.jpg"})
    return result


@app.post("/iu")
async def image_understanding(request: Optional[CompletionParams] = None, files: List[UploadFile] = File(...)):
    if request is None:
        example = CompletionParams.Config.json_schema_extra["examples"][1]
        request = CompletionParams(**example)

    urls = []
    for file in files:
        file_obj = io.BytesIO(await file.read())
        object_name = f"webimg/{file.filename}"
        url, _ = upload_file_to_oss(AliyunBucket, file_obj, object_name, expires=86400)
        urls.append(url)

    system_prompt = request.prompt or System_content.get(request.agent, '')
    model_info, payload, refer = await get_chat_payload(
        messages=[], user_request=request.question, system=system_prompt, images=urls,
        **{k: v for k, v in request.payload().items() if k not in {"images", "tools"}})

    # print(payload)
    assistant_content, completion = await ai_chat(model_info, payload)
    transform = extract_string(assistant_content, request.extract)
    return JSONResponse({'answer': assistant_content, 'reference': refer, 'transform': transform, "urls": urls})


@app.post("/fp")
async def files_process(files: List[UploadFile], question: str = None, model_name: str = 'qwen-long',
                        max_tokens: int = 4096):
    """
       接收文件并调用 AI 模型处理,基于文件内容生成消息。

       :param files: 上传的文件列表
       :param model_name: 模型名称
       :return: AI 处理结果
    """
    saved_file_paths = []
    for file in files:
        file_path = Path(Config.DATA_FOLDER) / file.filename
        # file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("wb") as f:
            f.write(await file.read())
        saved_file_paths.append(str(file_path))

    return JSONResponse(await ai_files_messages(saved_file_paths, question, model_name, max_tokens=max_tokens,
                                                dbpool=app.state.collector))


@app.get("/ppt")
async def ppt_create(text: str = Query(..., description="用于生成PPT的文本内容"), templateid: str = "20240718489569D"):
    url = await  xunfei_ppt_create(text, templateid)

    file_data, file_name = await download_file(url, data_folder=None)
    if file_data:
        file_io = io.BytesIO(file_data)
        file_io.seek(0)
        return StreamingResponse(file_io,
                                 media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                                 headers={"Content-Disposition": f"attachment; filename={file_name}.pptx",
                                          "File-Url": url})
    return {'url': url}


# class WebSocketCallback(ResultCallback):
#
#     _player = None
#     _stream = None
#
#     def __init__(self, websocket: WebSocket):
#         self.websocket = websocket
#
#     def on_open(self):
#         import pyaudio
#         print("websocket is open.")
#         self._player = pyaudio.PyAudio()
#         self._stream = self._player.open(
#             format=pyaudio.paInt16, channels=1, rate=22050, output=True
#         )
#
#     def on_complete(self):
#         print("speech synthesis task complete successfully.")
#
#     def on_error(self, message: str):
#         print(f"speech synthesis task failed, {message}")
#
#     def on_close(self):
#         print("websocket is closed.")
#         # stop player
#         self._stream.stop_stream()
#         self._stream.close()
#         self._player.terminate()
#
#     async def on_event(self, message):
#         print(f"recv speech synthesis message {message}")
#         await self.websocket.send_text(f"Synthesis Message: {message}")
#
#     def on_data(self, data: bytes) -> None:
#         print("audio result length:", len(data))
#         self._stream.write(data)
#         # 通过 websocket 发送音频数据
#         self.websocket.send_bytes(data)

# @app.websocket("/ws/speech_synthesis")
# async def websocket_speech_synthesis(websocket: WebSocket):
#     await websocket.accept()
#     try:
#         # 持续接收客户端文本消息
#         while True:
#             data = await websocket.receive_text()
#             print(f"Received text: {data}")
#
#             # 实例化回调并启动合成器
#             callback = WebSocketCallback(websocket)
#             synthesizer = SpeechSynthesizer(
#                 model=model,
#                 voice=voice,
#                 format=AudioFormat.PCM_22050HZ_MONO_16BIT,
#                 callback=callback,
#                 url=url,
#             )
#
#             # 使用文本生成模型
#             messages = [{"role": "user", "content": data}]
#             responses = Generation.call(
#                 model="qwen-turbo",
#                 messages=messages,
#                 result_format="message",  # 返回消息格式
#                 stream=True,  # 开启流式输出
#                 incremental_output=True,  # 增量输出
#             )
#
#             # 逐步将生成的内容进行语音合成
#             for response in responses:
#                 if response.status_code == 200:
#                     generated_text = response.output.choices[0]["message"]["content"]
#                     print(generated_text, end="")
#                     synthesizer.streaming_call(generated_text)  # 发送合成请求
#                 else:
#                     await websocket.send_text(f"Error: {response.message}")
#
#             synthesizer.streaming_complete()  # 完成合成
#             await websocket.send_text('Synthesis complete')
#     except WebSocketDisconnect:
#         print("Client disconnected")

# let socket = new WebSocket("ws://localhost:8000/ws/speech_synthesis");

if __name__ == "__main__":
    import uvicorn
    from uvicorn.config import LOGGING_CONFIG

    LOGGING_CONFIG["formatters"]["default"]["fmt"] = "%(asctime)s-%(levelprefix)s %(message)s"  # uvicorn / FastAPI

    # Config.debug()  # 测试环境配置,生产环境注释
    os.environ['UVICORN_WORKERS'] = '1'  # Config.update(WORKERS=1)
    try:
        uvicorn.run(app, host="0.0.0.0", port=7000)  # , log_level="warning"
    except KeyboardInterrupt:
        logging.warning("Server stopped by user（Ctrl+C）")
        close_dask_client()
    # pip install -r requirements.txt
    # uvicorn main:app --host 0.0.0.0 --port 7000 --workers 4
