# -*- coding: utf-8 -*-
import datetime
import difflib, copy
import tempfile
import logging
from logging.handlers import RotatingFileHandler

from typing import AsyncGenerator, Generator
from fastapi import FastAPI, Request, Header, Depends, Query, Body, File, UploadFile, BackgroundTasks, Form, \
    WebSocket, WebSocketDisconnect, HTTPException, status
from fastapi.responses import Response, StreamingResponse, JSONResponse, FileResponse, HTMLResponse
# from starlette.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer

# from fastmcp import FastMCP
from starlette.middleware.sessions import SessionMiddleware

# from apscheduler.schedulers.background import BackgroundScheduler
# from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.executors.asyncio import AsyncIOExecutor
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.jobstores.redis import RedisJobStore
# from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore

from contextlib import asynccontextmanager
from passlib.context import CryptContext
from config import Config

Config.load()
# Config.debug()  # 测试环境配置,生产环境注释

from structs import *
from database import *
from generates import *
from utils import configure_event_loop

configure_event_loop()


#  定义一个上下文管理器,初始化任务（如初始化数据库、调度器等） @app.lifespa
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        print("Starting up.")
        Base.metadata.create_all(bind=engine)
        # if not w3.is_connected():
        #     print("Failed to connect to Ethereum node")

        if not scheduler.get_job("tick_job"):
            scheduler.add_job(tick, 'interval', id="tick_job", seconds=60, misfire_grace_time=60,
                              jobstore='memory', executor='default')  # , max_instances=3
        if not scheduler.get_job("metadata_job"):
            scheduler.add_job(do_job_by_lock, 'cron', id="metadata_job", hour=5, minute=20,
                              misfire_grace_time=300,
                              args=[generate_tools_metadata, "lock:generate_tools_metadata", 600],
                              kwargs={"model_name": Config.DEFAULT_MODEL_METADATA}, jobstore='memory',  # 'redis',
                              max_instances=1, replace_existing=True)

        if not scheduler.running:
            scheduler.start()

        redis = await get_redis_connection()
        await init_ai_clients(AI_Models, get_data=True, redis=redis)

        # print(json.dumps(AI_Models, indent=4))
        # task1 = asyncio.create_task(message_zero_mq.start())
        # global_function_registry()

        yield


    except asyncio.CancelledError:
        logging.warning("Lifespan 被取消（应用关闭中）")
    finally:
        print("Shutting down.")
        scheduler.shutdown()
        engine.dispose()
        # MysqlData().close()

        # await async_engine.dispose()
        await shutdown_httpx()
        await shutdown_redis()

    # task1.cancel()
    # await asyncio.gather(task1, return_exceptions=True)


# executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
# scheduler = BackgroundScheduler(jobstores={'default': SQLAlchemyJobStore(engine=engine), 'memory': MemoryJobStore()},
#                                 executors={'default': ThreadPoolExecutor(4)}, timezone='Asia/Shanghai')  # 设置线程池大小
scheduler = AsyncIOScheduler(executors={'default': AsyncIOExecutor()},
                             jobstores={'memory': MemoryJobStore(),
                                        'redis': RedisJobStore(jobs_key='apscheduler.jobs',
                                                               run_times_key='apscheduler.run_times',
                                                               host=Config.REDIS_HOST, port=Config.REDIS_PORT, db=0)},
                             timezone='Asia/Shanghai')  # 异步调度器
# message_zero_mq = MessageZeroMQ()
app = FastAPI(lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key=Config.SECRET_KEY)
# mcp = FastMCP("aigc", app=app)  # lifespan=
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.WARNING,
                    handlers=[
                        logging.StreamHandler(),  # 输出到终端,控制台输出
                        RotatingFileHandler("app.log", maxBytes=1_000_000, backupCount=3)  # 文件日志logging.FileHandler
                    ])
logger = logging.getLogger(__name__)
dashscope.api_key = Config.DashScope_Service_Key
AliyunBucket = oss2.Bucket(oss2.Auth(Config.ALIYUN_oss_AK_ID, Config.ALIYUN_oss_Secret_Key), Config.ALIYUN_oss_endpoint,
                           Config.ALIYUN_Bucket_Name)
# 加密配置,密码哈希上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# OAuth2 密码令牌,设置 tokenUrl
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
os.makedirs(Config.DATA_FOLDER, exist_ok=True)
# 将文件目录映射为静态文件路径
# directory=os.path.abspath('.') + "/static")
app.mount("/static", StaticFiles(directory=Config.DATA_FOLDER), name="static")

# model_config['protected_namespaces'] = ()
try:
    from web3 import Web3

    w3 = Web3(
        Web3.HTTPProvider(f'https://mainnet.infura.io/v3/{Config.INFURA_PROJECT_ID}'))  # ("http://127.0.0.1:8545")
except:
    w3 = None

MTick_Time: datetime = None


async def tick():
    global MTick_Time
    tick_now = datetime.now()
    if not MTick_Time or MTick_Time.minute != tick_now.minute:
        MTick_Time = tick_now.replace(second=0, microsecond=0)
        # print(f"Tick new bar! The time is: {MTick_Time.strftime('%Y-%m-%d %H:%M:%S')}")

        if len(TaskManager.Task_queue):
            # print(len(TaskManager.Task_queue), 'Tick! The time is: %s' % tick_now)
            await TaskManager.clean_old_tasks()

        if len(BaseChatHistory.Chat_History_Cache):
            with SessionLocal() as session:
                ChatHistory.sequential_insert(session)


async def async_cron():
    while True:
        print('Async Tick! The time is: %s' % time.time())
        await asyncio.sleep(60)


# @app.on_event("startup")
# async def startup_event():
#     print("Starting up...")
#     Base.metadata.create_all(bind=engine)
#     asyncio.create_task(async_cron())
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


# 用户验证后的数据
fake_users_db = {
    "user1": {
        "username": "user1",
        "full_name": "John Doe",
        "hashed_password": "$2b$12$KixF3G.UZ.JYF4Jlk/EoWeVvMb2RJsCJdPtFPBW4wZBQGI3B1ysGm",  # "password"
        "disabled": False,
        "public_key": "user1_public_key"
    }
}


# @app.middleware("http")
# async def verify_api_key(request: Request, call_next):
#     api_key = request.headers.get("Authorization")
#     if api_key:
#         api_key = api_key.replace("Bearer ", "")  # 如果用的是 Bearer Token
#     if api_key not in Config.VALID_API_KEYS:
#         raise HTTPException(status_code=401, detail="Invalid API key")
#     response = await call_next(request)
#     return response
# 用于依赖注入
# async def verify_api_key(authorization: str = Header(None)):
#     if not authorization:
#         raise HTTPException(status_code=401, detail="Missing API key")
#     api_key = authorization.replace("Bearer ", "")  # 如果用的是 Bearer Token 格式
#     if api_key not in Config.VALID_API_KEYS:
#         raise HTTPException(status_code=401, detail="Invalid API key")

# @app.on_event("startup")
# async def startup():
#     sse_transport = SseServerTransport(app)
#     mcp.mount_transport(sse_transport)

def require_api_key(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        request: Request = kwargs.get("request")
        authorization = request.headers.get("Authorization")
        if not authorization:
            raise HTTPException(status_code=401, detail="Missing API key")
        api_key = authorization.replace("Bearer ", "")  # 如果用的是 Bearer Token
        if api_key not in Config.VALID_API_KEYS:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return await func(*args, **kwargs)

    return wrapper


# 如果提供了 eth_address 或 public_key，则不强制提供密码。
# 如果提供了 username 或 uuid，并且没有提供 eth_address 或 public_key，则需要提供密码进行注册。
@app.post("/register")
async def register_user(request: Registration, db: Session = Depends(get_db)):
    username = request.username
    public_key = request.public_key
    eth_address = request.eth_address
    signed_message = request.signed_message
    original_message = request.original_message

    if not (username or eth_address or public_key):
        raise HTTPException(status_code=400,
                            detail="At least one of username, eth_address, or public_key must be provided")

    db_user = User.get_user(db, username, public_key, eth_address)
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

    # 注册新用户
    db_user = User.create_user(db=db, username=username, password=request.password, role=request.role,
                               group=request.group, eth_address=eth_address, public_key=public_key)

    if not db_user:
        raise HTTPException(status_code=400,
                            detail="Password is required when neither eth_address nor public_key is provided")

    return {"message": "User registered successfully"}
    # User.update_user(user_id=db_user.id, eth_address=eth_address)


@app.post("/authenticate", response_model=Token)
async def authenticate_user(request: AuthRequest, db: Session = Depends(get_db)):
    '''
    登录路由，颁发访问令牌和刷新令牌,令牌生成 login_for_access_token,
    如果 eth_address 或 public_key 认证成功，通过公钥验证签名则不需要密码。
    使用 username 或 uuid 和密码登录。
    '''
    eth_address = request.eth_address
    public_key = request.public_key  # 用户的公钥
    signed_message = request.signed_message  # 签名的消息，Base64 编码格式（确保已正确编码）
    original_message = request.original_message  # 要验证的原始消息内容
    username = request.username
    is_verified = 0
    db_user = None
    # 验证签名
    if signed_message and original_message:
        if eth_address and w3:
            recovered_address = w3.eth.account.recover_message(text=original_message, signature=signed_message)
            if recovered_address.lower() != eth_address.lower():
                raise HTTPException(status_code=400, detail="Authentication failed")

            db_user = User.get_user(db=db, eth_address=eth_address)
            is_verified = 1 << 0

        if public_key:
            if not verify_ecdsa_signature(public_key, original_message, signed_message):
                raise HTTPException(status_code=400, detail="Public key authentication failed")

            db_user = User.get_user(db=db, public_key=public_key)
            is_verified |= 1 << 1

        if is_verified and db_user:
            access_token = create_access_token(
                data={"sub": db_user.username or db_user.user_id, 'user_id': db_user.user_id},
                expires_minutes=Config.ACCESS_TOKEN_EXPIRE_MINUTES)
            return {"access_token": access_token, "token_type": "bearer"}

    if request.password and username:
        db_user = User.get_user(db=db, username=username)  # user_id=request.uuid ,User.validate_credentials(
        if not db_user:
            raise HTTPException(status_code=400, detail="User not found")

        if not User.verify_password(request.password, db_user.password):
            raise HTTPException(status_code=400, detail="Invalid credentials")

        is_verified |= 1 << 2
        access_token = create_access_token(data={"sub": username, 'user_id': db_user.user_id},
                                           expires_minutes=Config.ACCESS_TOKEN_EXPIRE_MINUTES)
        return {"access_token": access_token, "token_type": "bearer"}

    raise HTTPException(status_code=400, detail="Invalid authentication request")


async def verify_request_signature(request: Request, api_secret_keys):
    # 请求签名验证的函数，主要用于确保请求的来源可信，防止请求在传输过程中被篡改
    api_key = request.headers.get("X-API-KEY")
    signature = request.headers.get("X-SIGNATURE")
    timestamp = request.headers.get("X-TIMESTAMP")

    if not all([api_key, signature, timestamp]):
        raise HTTPException(status_code=400, detail="Missing authentication headers")

    # 检查时间戳是否超时
    current_time = int(time.time())
    request_time = int(timestamp)
    if abs(current_time - request_time) > 300:  # 5分钟的时间窗口
        raise HTTPException(status_code=403, detail="Request timestamp expired")

    # 检查API Key是否合法
    secret = api_secret_keys.get(api_key)
    if not secret:
        raise HTTPException(status_code=403, detail="Invalid API Key")

    # 从请求中构造签名字符串
    method = request.method
    url = str(request.url)
    body = await request.body() if request.method in ["POST", "PUT"] else b""

    # 拼接签名字符串
    message = f"{method}{url}{body.decode()}{timestamp}"

    # 使用 HMAC-SHA256 生成服务器端的签名
    server_signature = hmac.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()

    if not hmac.compare_digest(server_signature, signature):
        raise HTTPException(status_code=403, detail="Invalid signature")

    return True


@app.post("/protected")
async def protected(request: Request, db: Session = Depends(get_db)):
    # api_key, secret_key = User.create_api_key(1, db)
    # User.update_user(1, db,public_key='83e2c687a44f839f6b3414d63e1a54ad32d8dbe4706cdd58dc6bd4233a592f78367ee1bff0e081ba678a5dfdf068d4b4c72f03085aa6ba5f0678e157fc15d305')
    api_keys = User.get_api_keys(db)
    await verify_request_signature(request, api_keys)
    return {"message": "Request authenticated successfully"}


# 检查该用户是否在数据库中有效或是否具备某些标志位
async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    username = verify_access_token(token)
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials,Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = User.get_user(db, username=username, user_id=username)
    if not user:
        raise HTTPException(status_code=404, detail="Access forbidden,User not found")
    if user.disabled:  # or user.expires_at <= time.time()
        raise HTTPException(status_code=400, detail="Access forbidden,Inactive user")
    return user


@app.post("/secure")
async def secure_route(user: User = Depends(get_current_user)):
    return {"message": "Access granted", "user": user.username}


@app.post("/refresh_token", response_model=Token)  # dict
async def refresh_access_token(username: str = Depends(verify_access_token)):
    if username is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")
    new_access_token = create_access_token(data={"sub": username},
                                           expires_minutes=Config.ACCESS_TOKEN_EXPIRE_MINUTES)
    return {"access_token": new_access_token, "token_type": "bearer"}


@app.get("/status/")
async def system_status():
    thread_count = threading.active_count()

    task_count = len(asyncio.all_tasks())  # asyncio 任务数
    job_count = len(scheduler.get_jobs())

    redis_info = await get_redis().info()
    pool = engine.pool

    return JSONResponse(content={
        "pid": os.getpid(),
        "threads": thread_count,
        "asyncio_task_count": task_count,
        "job_count": job_count,
        "memory_mb": round(get_memory_info(), 2),
        "cpu_time_sec": round(get_cpu_time(), 2),
        "open_fd_count": get_open_fds_count(),  # socket连接数 (打开的文件描述符数量)
        "http_connection_count": count_http_connections(7000),
        "redis_connected_clients": redis_info.get("connected_clients"),
        "db_connections_total": pool.size(),
        "db_connections_in_use": pool.checkedout(),
        "task_queue_count": len(TaskManager.Task_queue),
        'history_cache_count': len(BaseChatHistory.Chat_History_Cache),
        "event_loop_type": str(type(asyncio.get_event_loop())),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    })


@app.get("/logs")
async def get_logs(lines: int = 100):
    log_file = Path("app.log")
    if not log_file.exists():
        return {"error": "日志文件不存在"}

    try:
        lines_list = deque(maxlen=lines)
        async with aiofiles.open(log_file, "r") as f:
            # f.readlines(),await f.read()
            async for line in f:
                lines_list.append(line.strip())

        return {"logs": list(lines_list)}

    except Exception as e:
        return {"error": f"读取日志失败: {str(e)}"}


@app.get("/admin/")
async def admin(username: str = Depends(verify_access_token)):
    if username == 'admin':
        job_list = [
            {'id': job.id,
             'name': job.name,
             'next_run_time': job.next_run_time.isoformat() if job.next_run_time else None,
             'trigger': str(job.trigger)
             } for job in scheduler.get_jobs()
        ]
        return {'history': BaseChatHistory.Chat_History_Cache, 'task': TaskManager.Task_queue, 'job': job_list}
    return {"message": "Access denied: Admin privileges required"}


@app.get('/user/')
async def user(request: Request, token: str = None, db: Session = Depends(get_db)):
    if token:
        username = verify_access_token(token)
        if username is None:
            return {"error": "Invalid credentials"}
        user = User.get_user(db, username=username, user_id=username)
        if user and not user.disabled:
            return {"user": username}
        return {"error": "Access forbidden for virtual users"}

    user_id = request.session.get('user_id', '')
    if not user_id:
        user_id = str(uuid.uuid1())
        request.session['user_id'] = user_id  # 伪用户信息用于 session,临时用户标识

    return {"user": user_id}


@app.post("/data")
async def push_redis_data(payload: dict):
    redis = get_redis()
    try:
        await redis.lpush("task_queue", json.dumps(payload))
        return {"status": "queued"}
    except (ConnectionError, TimeoutError) as e:
        return JSONResponse(status_code=503, content={"error": "Redis 连接失败", "detail": str(e)})
    except Exception as e:
        return {"error": str(e)}


@app.get("/get/{key}")
async def read_redis_value(key: str = 'funcmeta:*'):
    redis = get_redis()
    try:
        if "*" in key or "?" in key or "[" in key:
            keys = await redis.keys(key)
            if not keys:
                return {"error": f"No keys match pattern '{key}'"}
            values = await redis.mget(*keys)
            result = {
                k.decode("utf-8") if isinstance(k, bytes) else k:
                    v.decode("utf-8") if isinstance(v, bytes) else v
                for k, v in zip(keys, values)
            }
            return {"result": result}
        else:
            value = await redis.get(key)
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            return {"result": value}

    except (ConnectionError, TimeoutError) as e:
        return JSONResponse(status_code=503, content={"error": "Redis 连接失败", "detail": str(e)})
    except Exception as e:  # Connect call failed,redis.exceptions.ConnectionError
        return {"error": str(e)}


@app.delete("/delete/{key}")
async def delete_redis_key(key: str):
    redis = get_redis()
    try:
        if "*" in key or "?" in key or "[" in key:
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
        results = patent_search(text, limit=3)
    elif platform == 'invest':
        results = company_search(text, search_type='invest', limit=10)
    else:
        results = await retrieved_reference(text, keywords=[text])

    return JSONResponse(results)


@app.get("/extract/")
async def extract(text: str = Query(...), extract: str = Query(default='all')):
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
    tokens = [x.replace("\n", " ").strip() for x in request.terms]
    results = []

    if request.method == 'levenshtein':  # char_unigr,计算两个词组之间的最小编辑次数,from nltk.metrics.distance import edit_distance
        for token in querys:
            matches = difflib.get_close_matches(token, tokens, n=request.top_n, cutoff=request.cutoff)
            matches = [(match, round(difflib.SequenceMatcher(None, token, match).ratio(), 3), tokens.index(match))
                       for match in matches]
            results.append({'query': token, 'matches': matches})
        # results = [{'token': token, 'matches': rapidfuzz.process.extract(token, tokens, limit=top_n, score_cutoff=cutoff)}
        #            for token in querys]
        # rapidfuzz.process.extractOne(token,choices)
    elif request.method == 'bm25':  # 初步检索,将词组转化为向量化表示（TF-IDF）,概率检索模型,适合在短词组或句子相似性上做简单匹配
        from script.bm25 import BM25
        bm25 = BM25(tokens)  # corpus,全库太大（如数亿条），先用关键词快速 filter 再做 embedding,用户输入极短关键词，embedding 向量不稳定
        for token in querys:
            scores = bm25.rank_documents(token)
            matches = [(tokens[match[0]], round(match[1], 3), match[0]) for match in scores[:request.top_n] if
                       match[1] >= request.cutoff]
            results.append({'query': token, 'matches': matches})
    elif request.method == 'reranker':  # 精细排序,BERT Attention,Cross-encoder / Bi-encoder,通过 Transformer 编码，进行对比分析,可以利用上下文信息
        async def process_token(token):
            scores = await ai_reranker(token, documents=tokens, top_n=request.top_n,
                                       model_name="BAAI/bge-reranker-v2-m3", model_id=0)
            matches = [(match[0], round(match[1], 3), match[2]) for match in scores if
                       match[1] >= request.cutoff]
            return {'query': token, 'matches': matches}

        results = await asyncio.gather(*(process_token(token) for token in querys))  # [(match,score,index)]
    elif request.method == 'embeddings':  # SBERT,MiniLM,将词组或句子嵌入为高维向量，通过余弦相似度衡量相似性
        similars = await get_similar_embeddings(querys, tokens, ai_embeddings, topn=request.top_n,
                                                cutoff=request.cutoff,
                                                model_name='qwen', model_id=0)
        results = [{'query': token, 'matches':
            [(match[0], round(match[1], 3), tokens.index(match[0])) for match in matches if match[1] >= request.cutoff]}
                   for token, matches in similars]
    elif request.method == 'wordnet':  # 词典或同义词库，将词组扩展为同义词词组列表进行匹配,PageRank基于链接或交互关系构建的图中节点间的“传递性”来计算排名
        pass

    # tokens = list({match for token in querys for match in difflib.get_close_matches(token, tokens)})
    # match, score = process.extractOne(token, tokens)
    # results.append({'token': token, 'match': match, 'score': score})
    return JSONResponse(content=results, media_type="application/json; charset=utf-8")
    # Response(content=list_to_xml('results', results), media_type='application/xml; charset=utf-8')


@app.post("/classify")
async def classify_text(request: ClassifyRequest):
    intents = []
    query = request.query.strip()
    intent_tokens = [(it, x.replace("\n", " ").strip()) for it, keywords in request.class_terms.items() for x in
                     keywords]
    tokens = [token for _, token in intent_tokens]
    last_c = request.class_default

    # 遍历 class_terms 字典，检查文本是否匹配任意关键词
    if lang_token_size(query, tokenizer=None) < 32:
        for intent, keywords in request.class_terms.items():
            if any(re.search(r'\b' + re.escape(keyword) + r'\b', query) for keyword in keywords):
                intents.append({"class": intent, 'score': 0, 'type': 'search'})

    matches = difflib.get_close_matches(query, tokens, n=10, cutoff=0.8)
    edit_scores = [(tokens.index(match), difflib.SequenceMatcher(None, query, match).ratio())
                   for match in matches]

    if edit_scores:
        intent = intent_tokens[edit_scores[0][0]][0]
        intents.append({"class": intent, 'score': edit_scores[0][1], 'type': 'edit'})
        if (all(intent_tokens[i[0]][0] == intent for i in edit_scores[:3]) and intent == last_c) or \
                (max(i[1] for i in edit_scores) > request.cutoff and all(i['class'] == last_c for i in intents)):
            return {"class": intent, 'match': intents}

    # 如果当前意图得分大于0.85并且与历史意图相同，更新历史记录并返回
    if any(i['score'] >= request.cutoff for i in intents if i['score']):
        if last_c and all(i['class'] == last_c for i in intents):
            return {"class": last_c, 'match': intents}

    # bm25_scores = BM25(tokens).rank_documents(query, sort=True, normalize=False)  # [(idx,max_score)]
    # if bm25_scores:
    #     best_match = max(bm25_scores, key=lambda x: x[1])
    #     intent = intent_tokens[best_match[0]][0]  # bm25_scores[0]
    #     if all(intent_tokens[i[0]][0] == intent for i in bm25_scores[:3]) and all(
    #             intent_tokens[i[0]][0] == intent for i in edit_scores[:3]):
    #         intents.append({"class": intent, 'score': None, 'type': 'bm25'})

    similar_scores = await get_similar_embeddings([query], tokens, embeddings_calls=ai_embeddings, topn=10,
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

    reranker_scores = await ai_reranker(query, documents=tokens, top_n=10, model_name=request.rerank_model,
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


@app.post("/knowledge/")
async def knowledge(text: str, rerank_model: Optional[str] = "BAAI/bge-reranker-v2-m3",
                    file: UploadFile = File(None), version: int = 0):
    '''search_knowledge_base'''
    from knowledge import ideatech_knowledge
    result = await ideatech_knowledge(text.strip(), rerank_model=rerank_model, file=file, version=version)
    return {'similar': result}


async def run_structure_sample(task: TaskNode, redis=None, limit: int = 100, yd_type='开户',
                               model='qwen:qwen3-32b'):
    task_id = task.name
    try:
        from script.insight_reject import structure_sample_applynote_api
        # 更新状态
        await TaskManager.set_task_status(task, TaskStatus.IN_PROGRESS, 10, redis)
        with engine.connect() as conn:
            df_template = pd.read_sql(
                f"SELECT * FROM semantic_template_library  WHERE 来源 = {yd_type} and 状态 = '启用'", conn)
            # 读取 multi_cleaned 数据，排除已抽样
            multi_cleaned = pd.read_sql(
                f"SELECT * FROM classify_sentences_clustered  WHERE 来源 = {yd_type} and 标注状态 = '未标注' and model_response is null",
                conn)  # AND (action IS NULL OR action IN ('update', 'insert')

        mask = (multi_cleaned['原始句子'].str.len() >= 10)
        df_sample = multi_cleaned[mask].groupby('cluster_id', group_keys=False).apply(
            lambda x: x.sample(n=min(2, len(x)), random_state=42)).sample(n=limit, random_state=42).reset_index()

        # 执行结构分类（已有的结构函数）
        merged_df, result_df = await structure_sample_applynote_api(df_sample, df_template, host='127.0.0.1',
                                                                    model=model)

        # 回写 Redis
        result = result_df[['id', 'action', 'path.一级类', 'path.二级类', 'path.三级类', 'template']].rename(
            columns={'path.一级类': '一级类', 'path.二级类': '二级类', 'path.三级类': '三级类'}).to_dict(
            orient='records')
        await TaskManager.update_task_result(task_id, result=result,
                                             status=TaskStatus.COMPLETED if all(isinstance(r, dict) for r in results)
                                             else TaskStatus.FAILED, redis_client=redis)
        update_sql = """
            UPDATE classify_sentences_clustered
            SET model_response=%s, model_response3=%s, action=%s, 一级类=%s, 二级类=%s, 三级类=%s, template=%s
            WHERE id=%s
        """

        params_list = [
            (
                json.dumps(row.get('model_response')),
                json.dumps(row.get('model_response3')),
                row.get('action'),
                row.get('path.一级类') or row.get('一级类'),
                row.get('path.二级类') or row.get('二级类'),
                row.get('path.三级类') or row.get('三级类'),
                row.get('template') or row.get('matched_template'),
                row.get('id') or row.get('index'),
            )
            for _, row in merged_df.iterrows()
        ]

        with MysqlData() as session:
            session.execute(update_sql, params_list)

    except Exception as e:
        print(e)
        await TaskManager.set_task_status(task, TaskStatus.FAILED, 100, redis)


class StructureSampleRequest(BaseModel):
    yd_type: str = Field(..., description="业务类型，例如 '开户'、'销户' 等")
    model: Optional[str] = Field("qwen:qwen3-32b", description="使用的模型名称")
    limit: Optional[int] = Field(100, description="返回的样本数量上限")


@app.get("/ideatech/sample", response_model=List[dict])
async def get_sample_batch(request: StructureSampleRequest):
    """
       返回 insert / update 类型的模型判定样本，用于人工标注处理。
       自动排除已标注或已入库模板的样本。
    """
    task_id = str(uuid.uuid4())
    redis = get_redis()
    task = TaskNode(
        name=task_id,
        description=request.yd_type,
        action='classify',
        data={"params": request.model_dump()},
        status=TaskStatus.PENDING,
        priority=10
    )
    await TaskManager.add_task(task_id, task, redis_client=redis, ex=3600)

    asyncio.create_task(
        run_structure_sample(task, redis, limit=request.limit, yd_type=request.yd_type, model=request.model))

    return {"task_id": task_id, 'url': f'{Config.WEBUI_URL}/task/{task_id}',
            'result': f'{Config.WEBUI_URL}/get/{TaskManager.key_prefix}:{task_id}'}


@app.get("/ideatech/templates/list", response_model=List[dict])
async def list_templates(only_active: bool = True, yd_type='开户'):
    ''' 查询模板，支持筛选路径、来源、状态。'''
    with MysqlData() as session:
        sql = f"SELECT * FROM semantic_template_library  WHERE 来源 = {yd_type}"
        if only_active:
            sql += " and 状态 = '启用'"
        return session.search(sql)


class SampleResult(BaseModel):
    id: int  # sample_id
    action: str
    一级类: str
    二级类: str
    三级类: str
    template: str
    status: bool


@app.post("/ideatech/confirm")
async def confirm_action(sample: SampleResult, yd_type='开户'):
    ''' 确认分类并更新模板库,人工确认 insert/update 的操作, status True 表示同意，False 表示拒绝'''
    with MysqlData() as session:
        # 更新 classify_sentences_clustered
        update_sql = """
            UPDATE classify_sentences_clustered
            SET action=%s, 一级类=%s, 二级类=%s, 三级类=%s, template=%s, 标注状态=%s
            WHERE id=%s
        """
        session.execute(update_sql, (
            sample.action, sample.一级类, sample.二级类,
            sample.三级类, sample.template, '已确认' if sample.status else '已弃用', sample.id
        ))

        # 若为 update/insert 则写入 semantic_template_library
        if sample.status and sample.action in ['update', 'insert']:
            insert_sql = """
                INSERT INTO semantic_template_library
                (来源, 一级类, 二级类, 三级类, template, sentences_sample_id)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            session.execute(insert_sql, (
                yd_type, sample.一级类, sample.二级类,
                sample.三级类, sample.template, sample.id
            ))


@app.get("/create_embeddings_collection/")
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


@app.post("/upsert_embeddings_points")
async def upsert_embeddings_points(payloads: List[Dict], inputs: Optional[List[str]], collection_name: str,
                                   text_field: str = None):
    collection_aliases = await QD_Client.get_collection_aliases(collection_name)
    model_name = collection_aliases.aliases[0].alias_name.split("_")[-1] if len(
        collection_aliases.aliases) else Config.DEFAULT_MODEL_EMBEDDING
    if text_field and not inputs:
        inputs = [p.get(text_field) for p in payloads]
    embeddings = await ai_embeddings(inputs=inputs, model_name=model_name, model_id=0, get_embedding=True)
    if embeddings:
        return {'operation_id': await upsert_points(payloads, vectors=embeddings,
                                                    collection_name=collection_name, client=QD_Client),
                'embeddings_size': len(embeddings),
                'embeddings_model': model_name}
    return {}


@app.post('/search_embeddings_points')
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


@app.post('/recommend_points')
async def recommend_points(ids: Union[List[int], Tuple[int]], collection_name: str, topn: int = 10,
                           score_threshold: float = 0.0, payload_key: str = None, field_key: str = None,
                           match_values=None):
    match = field_match(field_key, match_values) if field_key else []
    return await recommend_by_ids(ids, collection_name, client=QD_Client, payload_key=payload_key,
                                  match=match, not_match=[], topn=topn, score_threshold=score_threshold)


@app.get("/nlp/")
async def nlp(text: str, nlp_type: str = 'ecnet'):
    return await baidu_nlp(nlp_type=nlp_type, text=text)


@app.get("/markdown/", response_class=HTMLResponse)
async def get_markdown(text: str):
    html_content = format_for_html(text)
    return f"<html><body>{html_content}</body></html>"


@app.post("/prompts")
async def get_prompts(request: PromptRequest):
    result = {}
    if request.query:
        system_prompt = System_content.get(request.query, None)
        if system_prompt:
            result = {'prompt': system_prompt}
        elif request.model:
            depth_iter = iter(request.depth)
            to_do = next(depth_iter, None)
            if not to_do:
                return {"error": "depth 不能为空"}
            user_request = '构造或改进系统提示词：' + request.query
            bot_response = await ai_chat(user_request=user_request, system=System_content.get(to_do),
                                         model_name=request.model, temperature=0.3, max_tokens=4096, model_info={})
            reason, prompt = extract_tagged_split(bot_response, tag="reasoning")
            result = [{'reason': reason, 'prompt': prompt, 'depth': 0}]
            to_do = next(depth_iter, None)
            if to_do:
                messages = [
                    {"role": "system", "content": System_content.get(to_do)},
                    {"role": "user", "content": user_request},
                    {"role": "assistant", "content": prompt},
                    {"role": "user", "content": "再帮我调整优化下系统提示词:" + prompt}
                ]
                for i, to_do in enumerate(depth_iter, start=1):
                    try:
                        bot_response = await ai_chat(messages=messages, model_name=request.model, temperature=0.3,
                                                     max_tokens=4096, model_info={})

                        reason, prompt = extract_tagged_split(bot_response, tag="reasoning")
                        result.append({'reason': reason, 'prompt': prompt, 'depth': i})

                        messages[0] = {"role": "system", "content": System_content.get(to_do)}
                        messages.extend([{"role": "assistant", "content": bot_response},
                                         {"role": "user",
                                          "content": "根据上下文，再帮我调整优化下系统提示词:" + prompt}])
                    except Exception as e:
                        print(f"Error at depth {i}: {e}.{messages}")
                        break
            print(result)

    return JSONResponse(result if result else System_content)


@app.post("/tools")
async def get_tools(request: ToolRequest):
    """
    返回 OpenAI 兼容的 tools 定义，并调用模型接口后解析执行结果。
    如果 request.tools 为空且 request.user 存在，则从 Redis 获取用户缓存的 tools。否则调用 AI 自动生成 tool 元数据。
    最终根据 messages 与 tools 调用大模型，解析 tool_call 执行结果并返回。
    """
    if not request.messages:
        request.messages = [{"role": "system", "content": System_content.get('31')},
                            {"role": "user", "content": request.prompt}]

    tools_metadata = request.tools
    if not request.tools:
        if request.user:
            redis = get_redis()
            tools_metadata = await scan_from_redis(redis, "registry_meta", user=request.user)
        else:
            tools_metadata = await generate_tools_metadata(model_name=request.model_metadata)
            # print(tools_metadata)

    if not request.model_name:
        return JSONResponse(tools_metadata)

    tool_messages = await ai_tools_response(messages=request.messages, tools=tools_metadata or AI_Tools,
                                            model_name=request.model_name, model_id=request.model_id,
                                            top_p=request.top_p, temperature=request.temperature)

    if not tool_messages:
        raise HTTPException(status_code=500, detail="No response from AI model.")

    if request.tools:  # 自定义tools直接返回
        return JSONResponse(tool_messages)

    # 解析响应并调用工具
    return JSONResponse(await ai_tools_results(tool_messages))


@app.post("/metadata")
async def generate_metadata_from_code(req: MetadataRequest):
    cache_key = generate_hash_key(req.metadata, req.function_code, req.description)
    redis = get_redis()
    key_meta = f"registry_meta:{req.user}:{cache_key}"
    cached_metadata = await redis.get(key_meta)
    if cached_metadata:
        metadata = json.loads(cached_metadata)
        func_name = metadata.get('function', {}).get('name')
        print(f"Metadata already cached for function: {func_name}")
        key = f"registry:{func_name}"
        return {"metadata": metadata, 'key_meta': key_meta, "key": key}

    metadata = await ai_generate_metadata(
        req.function_code,
        req.metadata,
        req.model_name,
        description=req.description,
        code_type=req.code_type or "Python"
    )

    registry = metadata.get("function", {}).copy()
    func_name = registry.get("name")
    if not func_name:
        raise HTTPException(status_code=400, detail="函数名未能从模型生成结果中提取")

    if req.callback:  # 如果不是本地函数，生成远程调用 URL
        registry["x-url"] = req.callback.model_dump()
        # {
        #     "url": 'http://remote-system.com/api/function/{func_name}',
        #     "method": "POST",
        #     "headers": {
        #         "Authorization": "Bearer xxx"
        #     }
        # }
    registry["x-type"] = req.code_type.lower()
    registry["x-code"] = req.function_code
    registry["x-hash"] = cache_key
    registry["x-user"] = req.user
    key = f"registry:{func_name}"
    await redis.setex(key, req.cache_sec or Config.REDIS_CACHE_SEC, json.dumps(registry, ensure_ascii=False))
    await redis.setex(key_meta, req.cache_sec or Config.REDIS_CACHE_SEC, json.dumps(metadata, ensure_ascii=False))

    return {
        "metadata": metadata,
        "registry": registry,
        "key": key,
        'key_meta': key_meta,
    }


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

    async def process_one_limited(sub_q: str, idx: int, semaphore=None):
        model_info, payload, refer = await get_generate_payload(
            prompt=prompt,  # 优先使用
            user_request=sub_q,  # 如果空则是 prompt
            suffix=request.suffix,
            stream=request.stream, temperature=request.temperature, top_p=request.top_p, max_tokens=request.max_tokens,
            model_name=request.model_name, model_id=request.model_id,
            agent=request.agent, keywords=request.keywords
        )

        # 如果要流式，用异步生成器形式返回,非流式就直接拿完整字符串
        if request.stream:
            return await ai_generate(model_info, payload, get_content=True)  # async generator

        async with semaphore:
            for attempt in range(1, Config.MAX_RETRY_COUNT + 1):
                content = await ai_generate(model_info, payload, get_content=True)
                if 'Error code: 429' in content or 'error' in content.lower():
                    print(f"[重试] 第 {attempt} 次失败，内容: {content}")
                    await asyncio.sleep(attempt)
                    continue

                return {'question': sub_q, 'answer': content, 'reference': refer,
                        'transform': extract_string(content, request.extract), 'id': idx}

            # 最后失败也返回内容
            return {'question': sub_q, 'answer': content, 'reference': refer,
                    'transform': extract_string(content, request.extract), 'id': idx}

    if request.stream:
        async def batch_stream_response() -> AsyncGenerator[str, None]:
            num = len(questions)
            for idx, sub_q in enumerate(questions):
                if num > 1:
                    yield f"\n({idx}) {sub_q}\n"  # 子请求之间加一个换行分隔
                stream_fn = await process_one_limited(sub_q, idx)
                async for chunk in stream_fn:
                    yield chunk
                    await asyncio.sleep(0.01)  # Token 逐步返回的延迟

        # media_type="text/plain",纯文本数据
        return StreamingResponse(batch_stream_response(), media_type="text/plain")
    else:
        semaphore = asyncio.Semaphore(Config.MAX_CONCURRENT)
        tasks = [process_one_limited(sub_q, idx, semaphore) for idx, sub_q in enumerate(questions)]

        if use_task:
            task_id = str(uuid.uuid4())
            redis = get_redis()
            task = TaskNode(
                name=task_id,
                description=prompt,
                action='llm',
                data={
                    "params": request.model_dump(),  # CompletionParams
                    "questions": questions,  # list[str]
                },
                status=TaskStatus.PENDING,
                priority=10
            )
            await TaskManager.add_task(task_id, task, redis_client=redis, ex=3600)

            async def process_task():
                await TaskManager.set_task_status(task, TaskStatus.IN_PROGRESS, 10, redis)
                results = await asyncio.gather(*tasks, return_exceptions=True)  # 这会得到 List[dict]
                status = TaskStatus.COMPLETED if all(isinstance(r, dict) for r in results) else TaskStatus.FAILED
                await TaskManager.update_task_result(task_id, result=[r for r in results if isinstance(r, dict)],
                                                     status=status, redis_client=redis)

                for i, r in enumerate(results):
                    if isinstance(r, Exception):
                        print(f"[任务 {i} 异常] {r}")

                return results

            asyncio.create_task(process_task())

            return JSONResponse(content={'task_id': task_id, 'url': f'{Config.WEBUI_URL}/task/{task_id}',
                                         'result': f'{Config.WEBUI_URL}/get/{TaskManager.key_prefix}:{task_id}'})

        results = await asyncio.gather(*tasks)
        return JSONResponse(results[0] if len(results) == 1 else results)


# ,current_user: User = Depends(get_current_user)
@app.post("/message/")
async def generate_message(request: ChatCompletionRequest,
                           db: Session = Depends(get_db)) -> StreamingResponse or JSONResponse:
    # data = await request.json()
    # print(request.__dict__)
    if not request.messages and not request.question:
        return JSONResponse(status_code=400,
                            content={'answer': 'error',
                                     'error': 'Please provide messages or a question to process.'})

    model_name = request.model_name
    agent = request.agent
    extract = request.extract
    chat_history = ChatHistory(request.user, request.name, request.robot_id, agent, model_name,
                               timestamp=time.time(), request_uid=request.request_id)

    if not extract:
        agent_format = {
            '3': 'code.python',
            '2': 'json',
            '4': 'json',
            '5': 'code.sql',
            '6': 'header',
        }
        extract = agent_format.get(agent, request.extract)

    history = chat_history.build(request.question, [msg.dict() for msg in request.messages], request.use_hist,
                                 request.filter_limit, request.filter_time, db=db)

    system_prompt = request.prompt or System_content.get(agent, System_content['0'])  # system_instruction

    model_info, payload, refer = await get_chat_payload(
        messages=history, user_request=chat_history.user_request, system=system_prompt,
        temperature=request.temperature, top_p=request.top_p, max_tokens=request.max_tokens,
        model_name=model_name, model_id=request.model_id, agent=request.agent,
        tools=request.tools, keywords=request.keywords, images=request.images)

    if request.stream:
        async def generate_stream() -> AsyncGenerator[str, None]:
            if refer:
                first_data = json.dumps({'role': 'reference', 'content': refer},
                                        ensure_ascii=False)  # '\n'.join(refer)
                yield f'data: {first_data}\n\n'

            assistant_response = []
            async for content in ai_chat_stream(model_info, payload):
                if content:
                    if content.strip():
                        yield f'data: {content}\n\n'
                    assistant_response.append(content)
                    await asyncio.sleep(0.01)

            bot_response = ''.join(assistant_response)
            transform = extract_string(bot_response, extract)
            last_data = json.dumps({'role': 'assistant', 'content': bot_response, 'transform': transform},
                                   ensure_ascii=False)  # 转换字节流数据
            yield f'data: {last_data}\n\n'
            yield 'data: [DONE]\n\n'

            chat_history.save(bot_response, refer, transform, model_name=payload['model'], db=db)
            if request.callback and request.callback.url:
                result = {'answer': bot_response, 'reference': refer, 'transform': transform}
                await send_callback(request.callback.model_dump(), transform if isinstance(transform, dict) else result)

        return StreamingResponse(generate_stream(), media_type="text/event-stream")
    else:
        bot_response = await ai_chat(model_info, payload)
        # print(bot_response)
        transform = extract_string(bot_response, extract)
        chat_history.save(bot_response, refer, transform, model_name=payload['model'], db=db)

        result = {'answer': bot_response, 'reference': refer, 'transform': transform}
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
async def completions(request: OpenAIRequest, background_tasks: BackgroundTasks) -> Union[
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

        instance = BaseReBot(user_content=request.prompt, user=request.user, agent='generate')
        if request.suffix:
            instance.user_content += request.suffix
        if request.user and ':' in request.user:
            instance.user, instance.robot_id = request.user.split(':')
        background_tasks.add_task(instance.save, instance.user, instance.robot_id, request.model,
                                  instance=instance, model_response=response)  # 在请求结束后执行，并自动回收资源

        return JSONResponse(content=response)


# @require_api_key
@app.post("/v1/chat/completions", response_model=OpenAIResponse)
async def chat_completions(request: OpenAIRequestMessage, background_tasks: BackgroundTasks):
    """
    兼容 OpenAI API 的 /v1/chat/completions 路径，返回类似 OpenAI API 的格式
    """
    # print(request.dict())
    kwargs = request.model_dump(exclude_unset=True, exclude={'messages', 'model', 'stream', 'user'})
    messages = [msg.dict() for msg in request.messages]
    if request.stream:
        async def fake_stream_response():
            async for chunk in ai_chat_stream(model_info=None, messages=messages, user_request=None, system=None,
                                              model_name=request.model, model_id=0, get_content=False, **kwargs):
                # print(chunk.encode("utf-8"))
                yield f"data: {chunk}\n\n"

                await asyncio.sleep(0.01)

            yield 'data: [DONE]\n\n'

        return StreamingResponse(fake_stream_response(), media_type="text/event-stream")

    response = await ai_chat(model_info=None, messages=messages, user_request=None, system=None,
                             # temperature=request.temperature, max_tokens=request.max_tokens,top_p=request.top_p,
                             model_name=request.model, model_id=0, get_content=False, **kwargs)

    instance = BaseReBot(user=request.user, agent='chat')
    if request.user and ':' in request.user:
        instance.user, instance.robot_id = request.user.split(':')
    background_tasks.add_task(instance.save, instance.user, instance.robot_id, request.model,
                              instance=instance, messages=messages, model_response=response)
    return JSONResponse(content=response)  # Response(content=json.dumps(data), media_type="application/json")


@app.get("/v1/models")
async def get_models(model: Optional[str] = Query(None,
                                                  description=f"Retrieves a model instance, providing basic information about the model such as the owner and permissioning. e.g., {','.join(model_api_keys().keys())} or custom models.")):
    if model:
        try:
            model_info, model_id = find_ai_model(model, 0, 'model')
            model_data = next((item for item in model_info.get('data', []) if item['id'] == model_id), {})
            response_data = {
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": model_info['name']
            }
            for k, v in model_data.items():
                if k not in {"id", "owned_by"}:
                    response_data[k] = v
        except ValueError as e:
            response_data = {'error': str(e)}
    else:
        extracted_data = extract_ai_model("model")
        response_data = {
            "object": "list",
            "data": [
                {
                    "id": f'{owner}:{model_id}',  # 用于指定模型进行请求 fine-tuned-model
                    "object": "model",
                    "created": 1740386673,
                    "owned_by": owner,  # 拥有该模型的组织
                    "root": model_id,  # 根版本，与 ID 相同
                    "parent": None,  # 如果没有父模型，则为 None
                    # "max_model_len": 4096,#GPU内存限制而需要调整模型的最大序列长度
                    "permission": [
                        {
                            "id": f"modelperm-{owner}:{model_id}",
                            "object": "model_permission",
                            "created": 1740386673,
                            # "allow_create_engine": False,
                            # "allow_sampling": True,
                            # "allow_logprobs": True,
                            # "allow_search_indices": False,
                            # "allow_view": True,
                            # "allow_fine_tuning": False,
                            # "organization": "*",
                            # "group": None,
                            # "is_blocking": False
                        }
                    ],
                } for i, (owner, models) in enumerate(extracted_data)
                for j, model_id in enumerate(models)]
        }
        # print(len(ModelListExtract.models))
    return JSONResponse(content=response_data)


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
                agent=agent, request_uid=request.get('request_id')
            )

            # 生成系统提示和模型请求
            system_prompt = request.get('prompt') or System_content.get(agent, '')

            model_info, payload, refer = await get_chat_payload(
                messages=history, user_request=user_request,
                system=system_prompt, temperature=request.get('temperature', 0.4),
                top_p=request.get('top_p', 0.8), max_tokens=request.get('max_tokens', 1024),
                model_name=model_name, model_id=request.get('model_id', 0), agent=agent,
                keywords=request.get('keywords', []), images=request.get('images', [])
            )
            if request.get('stream', True):
                async def generate_stream() -> AsyncGenerator[str, None]:
                    if refer:
                        first_data = json.dumps({'role': 'reference', 'content': refer}, ensure_ascii=False)
                        yield f'data: {first_data}\n\n'

                    assistant_response = []
                    async for content in ai_chat_stream(model_info, payload):
                        if content and content.strip():
                            yield f'data: {content}\n\n'
                        assistant_response.append(content)
                        await asyncio.sleep(0.01)

                    bot_response = ''.join(assistant_response)
                    transform = extract_string(bot_response, extract)
                    last_data = json.dumps({'role': 'assistant', 'content': bot_response, 'transform': transform},
                                           ensure_ascii=False)
                    yield f'data: {last_data}\n\n'
                    yield 'data: [DONE]\n\n'

                    # 保存聊天记录
                    save_chat_history(
                        user_request, bot_response, user, name, robot_id, agent,
                        hist_size, model_name, current_timestamp, db=db,
                        refer=refer, transform=transform, request_uid=request.get('request_id'))

                # 流式传输消息到 WebSocket
                async for stream_chunk in generate_stream():
                    await websocket.send_text(stream_chunk)

            else:  # 非流式响应处理
                bot_response = await ai_chat(model_info, payload)
                transform = extract_string(bot_response, extract)

                # 保存聊天记录
                save_chat_history(
                    user_request, bot_response, user, name, robot_id, agent,
                    hist_size, model_name, current_timestamp, db=db,
                    refer=refer, transform=transform, request_uid=request.get('request_id')
                )

                await websocket.send_text(
                    json.dumps({'answer': bot_response, 'reference': refer, 'transform': transform}))
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


@app.get("/get_messages/")
async def get_messages(request: Request, user: str = Query(""), name: str = None,
                       robot_id: str = None, filter_time: float = Query(0.0),
                       agent: str = None, db: Session = Depends(get_db)):
    request_id = request.session.get('user', '')
    if not user and not request_id:
        return JSONResponse(status_code=400, content={"error": "No user id found in session"})
    # filter_time = filter_time / 1000.0
    filter_history = get_user_history(user, name, robot_id, filter_time, db, agent=agent,
                                      request_uid=request_id)
    for item in filter_history:
        if isinstance(item.get('created_at'), datetime):
            item['created_at'] = item['created_at'].isoformat()
    return JSONResponse(content=sorted(filter_history, key=lambda x: x['timestamp']))


# @app.route('/chat_ui')
# def chat_ui():
#     return redirect("http://host.docker.internal:8080")

@app.post("/submit_messages")
async def submit_messages(request: SubmitMessagesRequest,
                          db: Session = Depends(get_db)):  # background_tasks: BackgroundTasks,
    if len(TaskManager.Task_queue) > Config.MAX_TASKS:
        return JSONResponse(status_code=400, content={'task_id': '', "error": "任务队列已满"})
    if not request.messages and not request.params:
        return JSONResponse(status_code=400,
                            content={'task_id': '', 'error': 'Please provide messages or a question to process.'})

    current_timestamp = time.time()
    chat_history = ChatHistory(request.user, request.name, request.robot_id, agent=None, model_name=None,
                               timestamp=current_timestamp, request_uid=request.request_id)

    history: List[dict] = chat_history.build('', [msg.dict() for msg in request.messages], request.use_hist,
                                             request.filter_limit, request.filter_time, db=db)
    # generate_hash_key(request.user, request.name,request.robot_id, request.model_id,
    #                   datetime.datetime.now().date().isoformat())
    task_id = str(uuid.uuid4())
    redis = get_redis()
    task = TaskNode(
        name=task_id,
        description=chat_history.user_request,
        action='message',
        data={
            "params": request.params,  # List[CompletionParams]
            "messages": history,  # list[dict]
        },
        status=TaskStatus.PENDING,
        start_time=current_timestamp,
        priority=10
    )
    await TaskManager.add_task(task_id, task, redis_client=redis, ex=3600)
    if request.params:
        asyncio.create_task(process_task_ai(task, chat_history, redis))
        # asyncio.create_task(asyncio.to_thread(
        # background_tasks.add_task(process_task_ai, task_id)

    return JSONResponse(content={'task_id': task_id, 'url': f'{Config.WEBUI_URL}/task/{task_id}',
                                 'result': f'{Config.WEBUI_URL}/get/{TaskManager.key_prefix}:{task_id}'})


async def process_task_ai(task: TaskNode, chat_history: ChatHistory, redis=None):
    if not task:
        print(f"[process_task_ai] Task not found in Task_queue.")
        return

    await TaskManager.set_task_status(task, TaskStatus.IN_PROGRESS, 10, redis)
    task_id = task.name
    params: List[CompletionParams] = task.data.get('params')
    history = task.data.get('messages')
    user_request = chat_history.user_request or task.description

    with SessionLocal() as session:
        async def single_param(i: int, param: CompletionParams, semaphore):
            if not param.question:
                param.question = user_request
            # else:
            #     chat_history.user_request = param.question
            chat_history.model = param.model_name
            chat_history.agent = param.agent
            user_history = chat_history.rebuild(param.question, history) if history else []
            system_prompt = param.prompt or System_content.get(param.agent, '')
            model_info, payload, refer = await get_chat_payload(messages=user_history, user_request=param.question,
                                                                system=system_prompt, temperature=param.temperature,
                                                                top_p=param.top_p, max_tokens=param.max_tokens,
                                                                model_name=param.model_name, model_id=param.model_id,
                                                                agent=param.agent, keywords=param.keywords,
                                                                images=param.images)
            # **param.asdict(),payload=param.payload()
            async with semaphore:
                bot_response = await ai_chat(model_info, payload)
                transform = extract_string(bot_response, param.extract)

                chat_history.save(bot_response, refer, transform, param.question, model_name=payload['model'],
                                  db=session)

                if param.extract == 'wechat':
                    if chat_history.name and chat_history.robot_id:
                        await send_to_wechat(chat_history.name, transform or bot_response)
                        # send_wechat(transform or bot_response, chat_history.name)
                result = {'question': param.question, 'answer': bot_response, 'reference': refer,
                          'transform': transform, 'id': i, 'task_id': task_id}
                if param.callback and param.callback.url:
                    await send_callback(param.callback.model_dump(),
                                        result=transform if isinstance(transform, (dict, list)) else result)

                return result

    semaphore = asyncio.Semaphore(Config.MAX_CONCURRENT)
    tasks = [single_param(i, p, semaphore) for i, p in enumerate(params) if not p.stream]
    results = await asyncio.gather(*tasks, return_exceptions=True)  # 确保任务取消时不会引发异常
    status = TaskStatus.COMPLETED if all(isinstance(r, dict) for r in results) else TaskStatus.FAILED
    await TaskManager.update_task_result(task_id, result=[r for r in results if isinstance(r, dict)],
                                         status=status, redis_client=redis)

    for i, r in enumerate(results):
        if isinstance(r, Exception):
            print(f"[任务 {i} 异常] {r}")

    return results

    # async def process_multiple_tasks(task_ids: list, messages: list):
    #     # 使用 asyncio.gather 并发执行多个异步任务
    #     tasks = [process_message_in_threadpool(task_id, message) for task_id, message in zip(task_ids, messages)]
    #     await asyncio.gather(*tasks)

    # await asyncio.get_event_loop().run_in_executor(None, ai_chat,loop = asyncio.get_running_loop() await loop.run_in_executor(executor, ai_chat,
    # executor.submit(ai_chat, *args) future.result()   await asyncio.wait_for(ai_chat(*args), timeout=10)


async def get_ai_param(
        stream: bool = Query(False, description="Enable streaming response if set to True."),
        temperature: float = Query(0.8,
                                   description="Controls randomness. Lower values make output more focused and deterministic, while higher values increase creativity."),
        top_p: float = Query(0.8,
                             description="Nucleus sampling parameter. Limits the response to the smallest set of tokens whose cumulative probability exceeds this value."),
        max_tokens: Optional[int] = Query(1024,
                                          description="The maximum number of tokens the model can generate in the response."),
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
        keywords: Optional[List[str]] = Query(None,
                                              description="A list of keywords used to guide the retrieval of relevant information or sources based on search terms."),
) -> CompletionParams:
    """
     Asynchronously retrieves the AI parameters based on user input and returns them as an CompletionParams object.
     """
    return CompletionParams(
        stream=stream,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        prompt=prompt,
        question=question,
        agent=agent,
        model_name=model_name,
        model_id=model_id,
        extract=extract,
        keywords=keywords,
    )


@app.get("/message/{task_id}")
async def response_message(task_id: str, param: CompletionParams = Depends(get_ai_param),
                           db: Session = Depends(get_db)) -> StreamingResponse or JSONResponse:
    redis = get_redis()
    task = await TaskManager.get_task(task_id, redis)

    if not task:
        error_data = {"error": "Invalid task ID", 'messages': task_id, "status": "not_found"}
        return JSONResponse(content=error_data, status_code=404)

    if task.action != 'message':
        error_data = {"error": "Invalid task action", 'messages': task.action, "status": task.status.value}
        return JSONResponse(content=error_data, status_code=400)

    if task.status == TaskStatus.IN_PROGRESS:
        error_data = {"error": "Task already in progress", 'messages': task.action, "status": task.status.value}
        return JSONResponse(content=error_data, status_code=202)

    if task.status in (TaskStatus.COMPLETED, TaskStatus.RECEIVED):
        await TaskManager.set_task_status(task, TaskStatus.RECEIVED, 100, redis)
        return JSONResponse(content=task.result)

    await TaskManager.set_task_status(task, TaskStatus.IN_PROGRESS, 10, redis)

    chat_history: ChatHistory = task.data.get('obj')
    chat_history.model = param.model_name
    chat_history.agent = param.agent
    history: List[dict] = task.data.get('messages')

    if not param.question:
        param.question = chat_history.user_request or task.description

    system_prompt = param.prompt or System_content.get(param.agent, '')
    model_info, payload, refer = await get_chat_payload(messages=history, user_request=param.question,
                                                        system=system_prompt, temperature=param.temperature,
                                                        top_p=param.top_p, max_tokens=param.max_tokens,
                                                        model_name=param.model_name, model_id=param.model_id,
                                                        agent=param.agent, keywords=param.keywords, images=param.images)

    if param.stream:
        async def generate():
            if refer:
                first_data = json.dumps({'role': 'reference', 'content': refer}, ensure_ascii=False)
                yield f'data: {first_data}\n\n'

            assistant_response = []
            async for content in ai_chat_stream(model_info, payload):
                if content:
                    if content.strip():
                        yield f'data: {content}\n\n'
                    assistant_response.append(content)
                    await asyncio.sleep(0.01)

            bot_response = ''.join(assistant_response)
            transform = extract_string(bot_response, param.extract)
            last_data = json.dumps({'role': 'assistant', 'content': bot_response, 'transform': transform},
                                   ensure_ascii=False)
            yield f'data: {last_data}\n\n'
            yield 'data: [DONE]\n\n'

            chat_history.save(bot_response, refer, transform, user_request=param.question, model_name=payload['model'],
                              db=db)

            result = [{'answer': bot_response, 'reference': refer, 'transform': transform, 'id': 0, 'task_id': task_id}]
            await TaskManager.update_task_result(task_id, result=result, status=TaskStatus.RECEIVED, redis_client=redis)

        return StreamingResponse(generate(), media_type="text/event-stream")

    bot_response = await ai_chat(model_info, payload)
    transform = extract_string(bot_response, param.extract)

    chat_history.save(bot_response, refer, transform, user_request=param.question, model_name=payload['model'], db=db)

    result = [{'answer': bot_response, 'reference': refer, 'transform': transform, 'id': 0, 'task_id': task_id}]
    await TaskManager.update_task_result(task_id, result=result, status=TaskStatus.RECEIVED, redis_client=redis)
    # del Task_queue[task_id]
    return JSONResponse(content=result)


@app.get("/task/{task_id}")
async def get_task_status(task_id: str, as_file: bool = Query(False)):
    redis = get_redis()
    task = await TaskManager.get_task(task_id, redis)
    if not task:
        return {'error': "Invalid task ID,Task not found", "status": "not_found"}

    status = task.status
    if status == TaskStatus.COMPLETED:
        await TaskManager.set_task_status(task, TaskStatus.RECEIVED, 100, redis)

    if as_file:
        tmp = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json", encoding="utf-8")
        json.dump(task.result, tmp, ensure_ascii=False, indent=2)
        tmp.flush()
        tmp.close()

        async def delayed_remove(file_path, delay=300):
            await asyncio.sleep(delay)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"删除临时文件失败: {e}")

        asyncio.create_task(delayed_remove(tmp.name, 300))
        # 返回一个 FileResponse，客户端收到后就能下载 JSON 文件
        return FileResponse(tmp.name, media_type="application/json", filename=f"results_{task_id}.json")

    return JSONResponse(
        content={"status": status.value, "action": task.action, "result": task.result, "count": task.count})


# return [{"task_id": v["name"], "description": v["description"], "status": v["status"], 'action': v['action']} for v in Task_graph.vs]


# 执行任务
@app.post("/task/execute/")
def execute_task(task_id: str):
    scheduler = TaskScheduler()
    try:
        scheduler.update_task_status(task_id, TaskStatus.COMPLETED)  # source status:edge["condition"]
        # 触发依赖任务
        scheduler.check_and_trigger_tasks()
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

    file_path = Path(Config.DATA_FOLDER) / file.filename
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(await file.read())
        url = f"{Config.WEBUI_URL}/files/{file.filename}"

    if oss_expires != 0:
        object_name = f"upload/{file.filename}"
        with open(file_path, 'rb') as file_obj:
            url = upload_file_to_oss(AliyunBucket, file_obj, object_name, expires=oss_expires)
        os.remove(file_path)

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
        url = upload_file_to_oss(AliyunBucket, file_obj, f"upload/{file.filename}", expires=oss_expires)

    if tigger_sec > 0:
        job_id = f"to_wechat_{generate_hash_key(user_name, context, url)}"
        if scheduler.get_job(job_id):
            print(f'job_existing and replaced: {job_id} | {form_data}')  # job.remove()
        else:
            print(f'job_created: {job_id} | scheduled at {send_time}')

        scheduler.add_job(send_to_wechat, 'date', id=job_id, run_date=send_time, misfire_grace_time=50,
                          args=[user_name, context, url, object_name], replace_existing=True, jobstore='redis')
    else:
        await send_to_wechat(user_name, context, url, object_name)

    return {"name": user_name, "file": object_name, "url": url, "send_time": send_time, 'tigger_sec': tigger_sec}


@app.get("/", response_class=HTMLResponse)
async def send_page():
    return """
        <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>微信发送</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        margin: 20px;
                    }
                    h2 {
                        color: #4CAF50;
                    }
                    form {
                        max-width: 500px;
                        margin: 0 auto;
                    }
                    label {
                        display: block;
                        margin-bottom: 8px;
                        font-weight: bold;
                    }
                    input[type="file"], input[type="text"], input[type="datetime-local"], textarea {
                        margin-bottom: 10px;
                        width: 100%;
                        box-sizing: border-box;
                    }
                    textarea {
                        resize: vertical;
                        min-height: 100px;
                    }
                    input[type="submit"] {
                        background-color: #4CAF50;
                        color: white;
                        padding: 10px 20px;
                        border: none;
                        cursor: pointer;
                    }
                    input[type="submit"]:hover {
                        background-color: #45a049;
                    }
                </style>
            </head>
            <body>
                <h2>微信发送</h2>
                <form action="/send_wechat_scheduler" method="post" enctype="multipart/form-data">
                    <label for="user_name">User Name:</label>
                    <input type="text" name="user_name" required><br><br>

                    <label for="context">Context:</label>
                    <textarea name="context" rows="5" placeholder="Enter message"></textarea><br><br>

                    <label for="file">File:</label>
                    <input type="file" name="file"><br><br>
                
                    <label for="object_name">Object Name:</label>
                    <input type="text" name="object_name"><br><br>
                    
                    <label for="send_time">Send Time:</label>
                    <input type="datetime-local" name="send_time"><br><br>

                    <input type="submit" value="Upload">
                </form>

                 <script>
                    // 获取当前时间并格式化为 YYYY-MM-DDTHH:MM
                    let now = new Date();
                    let year = now.getFullYear();
                    let month = (now.getMonth() + 1).toString().padStart(2, '0');
                    let day = now.getDate().toString().padStart(2, '0');
                    let hours = now.getHours().toString().padStart(2, '0');
                    let minutes = now.getMinutes().toString().padStart(2, '0');
                    let formattedTime = `${year}-${month}-${day}T${hours}:${minutes}`;

                    // 设置到 input 的默认值
                    document.getElementById('send_time').value = formattedTime;
                </script>
            </body>
        </html>
        """


# @app.post("/send_zero_mq")
# async def send_msg_zero_mq(message: str, topic: str = 'topic'):
#     await message_zero_mq.send_message(message, topic)
#     return {"status": "success", "message": f"Message '{message}' sent to ZeroMQ."}


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
            image_urls = [upload_file_to_oss(AliyunBucket, file_obj, object_name, expires=3600)]

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
                         platform: PlatformEnum = PlatformEnum.baidu):  # File(...）
    # 获取音频数据
    if file:
        try:
            audio_data = io.BytesIO(await file.read())
            format = file.content_type.split('/')[1] if file.content_type.startswith('audio/') else 'pcm'
        except Exception as e:
            return {"error": f"Failed to process the uploaded file:{e}"}

        if platform == PlatformEnum.baidu:
            return await baidu_speech_to_text(audio_data, format)
        elif platform == PlatformEnum.ali:
            return await ali_speech_to_text(audio_data, format)
        elif platform == PlatformEnum.dashscope:
            if file:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{format}')
                try:
                    temp_file.write(audio_data.getvalue())  # await file.read()
                    temp_file.close()
                    result = await dashscope_speech_to_text(temp_file.name, format=format)
                finally:
                    os.remove(temp_file.name)
                return result

    elif file_urls:
        if isinstance(file_urls, list) and len(file_urls) == 1:
            file_urls = file_urls[0].split(',')
        elif isinstance(file_urls, str):
            file_urls = file_urls.split(',')

        file_urls = [url.strip(" '\"") for url in file_urls]

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
        url = upload_file_to_oss(AliyunBucket, file_obj, object_name, expires=86400)
        urls.append(url)

    system_prompt = request.prompt or System_content.get(request.agent, '')
    model_info, payload, refer = await get_chat_payload(
        messages=[], user_request=request.question, system=system_prompt,
        temperature=request.temperature, top_p=request.top_p, max_tokens=request.max_tokens,
        model_name=request.model_name, model_id=request.model_id, agent=request.agent,
        keywords=request.keywords, images=urls)

    # print(payload)
    bot_response = await ai_chat(model_info, payload)
    transform = extract_string(bot_response, request.extract)
    return {'answer': bot_response, 'reference': refer, 'transform': transform, "urls": urls}


@app.post("/fp")
async def files_process(files: List[UploadFile], question: str = None, model_name: str = 'qwen-long',
                        model_id: int = -1):
    """
       接收文件并调用 AI 模型处理,基于文件内容生成消息。

       :param files: 上传的文件列表
       :param model_name: 模型名称
       :param model_id: 模型 ID
       :return: AI 处理结果
    """
    saved_file_paths = []
    for file in files:
        file_path = Path(Config.DATA_FOLDER) / file.filename
        # file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("wb") as f:
            f.write(await file.read())
        saved_file_paths.append(str(file_path))

    return JSONResponse(await ai_files_messages(saved_file_paths, question, model_name, model_id, max_tokens=4096))


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
    uvicorn.run(app, host="0.0.0.0", port=7000)

    # pip install -r requirements.txt
    # uvicorn main:app --host 0.0.0.0 --port 7000 --workers 4
