# -*- coding: utf-8 -*-
import datetime
import difflib, copy
import tempfile
import logging
from functools import wraps

from typing import AsyncGenerator, Generator
from fastapi import FastAPI, Request, Header, Depends, Query, File, UploadFile, BackgroundTasks, Form, \
    WebSocket, WebSocketDisconnect, HTTPException, status
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse, HTMLResponse
from fastapi.security import OAuth2PasswordBearer
from starlette.middleware.sessions import SessionMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
# from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
# from apscheduler.jobstores.redis import RedisJobStore
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.executors.pool import ThreadPoolExecutor
from contextlib import asynccontextmanager
from passlib.context import CryptContext
from qdrant_client import AsyncQdrantClient

from structs import *
from generates import *
from config import *
from database import *
from agents.ai_tasks import *


#  定义一个上下文管理器,初始化任务（如初始化数据库、调度器等） @app.lifespa
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up.")
    Base.metadata.create_all(bind=engine)
    # if not w3.is_connected():
    #     print("Failed to connect to Ethereum node")

    if not scheduler.get_job("tick_job"):
        scheduler.add_job(tick, 'interval', id="tick_job", seconds=60, misfire_grace_time=60,
                          jobstore='memory')  # , max_instances=3
    if not scheduler.running:
        scheduler.start()

    await init_ai_clients(AI_Models, API_KEYS,get_data=True)

    # print(json.dumps(AI_Models, indent=4))
    # task1 = asyncio.create_task(message_zero_mq.start())

    yield

    print("Shutting down.")
    scheduler.shutdown()
    engine.dispose()

    # task1.cancel()
    # await asyncio.gather(task1, return_exceptions=True)  # 确保任务取消时不会引发异常


# executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
engine = create_engine(Config.SQLALCHEMY_DATABASE_URI, pool_recycle=28800, pool_size=8, max_overflow=20)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)  # isolation_level='SERIALIZABLE'
scheduler = BackgroundScheduler(jobstores={'default': SQLAlchemyJobStore(engine=engine), 'memory': MemoryJobStore()},
                                executors={'default': ThreadPoolExecutor(4)})  # 设置线程池大小
# scheduler = AsyncIOScheduler(executors={'default': ThreadPoolExecutor(4)})
# message_zero_mq = MessageZeroMQ()
app = FastAPI(lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key=Config.SECRET_KEY)
# 将文件目录映射为静态文件路径
# app.mount("/static", StaticFiles(directory=os.path.abspath('.') + "/static"), name="static")
qd_client = AsyncQdrantClient(host=Config.QDRANT_HOST, grpc_port=6334, prefer_grpc=True)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.WARNING)

dashscope.api_key = Config.DashScope_Service_Key
AliyunBucket = oss2.Bucket(oss2.Auth(Config.ALIYUN_oss_AK_ID, Config.ALIYUN_oss_Secret_Key), Config.ALIYUN_oss_endpoint,
                           Config.ALIYUN_Bucket_Name)
# 加密配置,密码哈希上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# OAuth2 密码令牌,设置 tokenUrl
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
os.makedirs(Config.DATA_FOLDER, exist_ok=True)

# model_config['protected_namespaces'] = ()
try:
    from web3 import Web3

    w3 = Web3(
        Web3.HTTPProvider(f'https://mainnet.infura.io/v3/{Config.INFURA_PROJECT_ID}'))  # ("http://127.0.0.1:8545")
except:
    w3 = None


def tick():
    db = next(get_db())
    if len(Task_queue):
        print(len(Task_queue), 'Tick! The time is: %s' % datetime.now())
        cleanup_tasks()

    if len(Chat_history):
        print(len(Chat_history), 'Tick! The time is: %s' % datetime.now())
        ChatHistory.sequential_insert(db, Chat_history)
    # db.close()


async def async_cron():
    while True:
        print('Async Tick! The time is: %s' % time.time())
        await asyncio.sleep(60)


def get_db():
    db = SessionLocal()
    try:
        yield db
    # except Exception:
    #     db.rollback()
    #     raise
    finally:
        db.close()


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
            access_token = create_access_token(data={"sub": db_user.username or db_user.user_id},
                                               expires_minutes=Config.ACCESS_TOKEN_EXPIRE_MINUTES)
            return {"access_token": access_token, "token_type": "bearer"}

    if request.password and username:
        db_user = User.get_user(db=db, username=username)  # user_id=request.uuid ,User.validate_credentials(
        if not db_user:
            raise HTTPException(status_code=400, detail="User not found")

        if not User.verify_password(request.password, db_user.password):
            raise HTTPException(status_code=400, detail="Invalid credentials")

        is_verified |= 1 << 2
        access_token = create_access_token(data={"sub": username},
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
        return {'history': Chat_history, 'task': Task_queue, 'job': job_list}
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


@app.get('/web_search/{text}')
async def web_search(text: str, platform: str = 'default'):
    if platform == 'wiki':
        return wikipedia_search(text)
    return JSONResponse(await web_search_async(text))


@app.get('/retrieval/{text}')
async def retrieval(text: str):
    return JSONResponse(await retrieved_reference(text, keywords=[text]))


@app.post("/embeddings")
async def embeddings(request: EmbeddingRequest):
    inputs = [x.replace("\n", " ") for x in request.texts]
    if request.model_name == 'baidu_bge':
        access_token = get_baidu_access_token()
        embedding = await get_baidu_embeddings(inputs, access_token=access_token)
        return {"embedding": embedding}
    if request.model_name == 'word_vec':
        pass

    embedding = await ai_embeddings(inputs, model_name=request.model_name, model_id=request.model_id)
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
        bm25 = BM25(tokens)  # corpus
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
                   for
                   token, matches in similars]
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
    last_intent = request.class_default

    # 遍历 class_terms 字典，检查文本是否匹配任意关键词
    if len(query) < 32:
        for intent, keywords in request.class_terms.items():
            if any(re.search(r'\b' + re.escape(keyword) + r'\b', query) for keyword in keywords):
                intents.append({"class": intent, 'score': None, 'type': 'search'})

    matches = difflib.get_close_matches(query, tokens, n=10, cutoff=0.8)
    edit_scores = [(tokens.index(match), difflib.SequenceMatcher(None, query, match).ratio())
                   for match in matches]

    if edit_scores:
        intent = intent_tokens[edit_scores[0][0]][0]
        intents.append({"class": intent, 'score': edit_scores[0][1], 'type': 'edit'})
        if (all(intent_tokens[i[0]][0] == intent for i in edit_scores[:3]) and intent == last_intent) or \
                (max(i[1] for i in edit_scores) > 0.85 and all(i['class'] == last_intent for i in intents)):
            return {"intent": intent, 'match': intents}

    # 如果当前意图得分大于0.85并且与历史意图相同，更新历史记录并返回
    if any(i['score'] >= 0.85 for i in intents if i['score']):
        if last_intent and all(i['class'] == last_intent for i in intents):
            return {"intent": last_intent, 'match': intents}

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

    if any(i['score'] >= 0.85 for i in intents if i['score']):
        if all(i['class'] == intents[0]['class'] for i in intents):
            intent = intents[0]['class']
            return {"intent": intent, 'match': intents}

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
            return {"intent": intent, 'match': intents}

    system_prompt = request.prompt + f'\n{request.class_terms}'
    model_info, payload, refer = await get_chat_payload(messages=None, user_request=query, system=system_prompt,
                                                        temperature=0.3, top_p=0.8, max_tokens=256,
                                                        model_name=request.llm_model, model_id=1)
    bot_response = await ai_chat(model_info, payload)
    result = extract_json_from_string(bot_response)
    intent = result.get("intent") if result else None
    if intent:
        intents.append({"class": intent, 'type': request.llm_model})
        result['match'] = intents
        return result

    print(bot_response, intents, payload)
    return {"intent": last_intent, 'match': intents}


@app.post("/knowledge/")
async def knowledge(text: str, rerank_model: Optional[str] = "BAAI/bge-reranker-v2-m3",
                    file: UploadFile = File(None), version: int = 0):
    '''search_knowledge_base'''
    from knowledge import ideatech_knowledge
    result = await ideatech_knowledge(text.strip(), rerank_model=rerank_model, file=file, version=version)
    return {'similar': result}


@app.get("/nlp/")
async def nlp(text: str, nlp_type: str = 'ecnet'):
    return await baidu_nlp(nlp_type=nlp_type, text=text)


@app.post("/tools")
async def text_tools(request: ToolRequest):
    # 调用模型接口
    if not request.messages:
        request.messages = [{"role": "system", "content": System_content.get('31')},
                            {"role": "user", "content": request.prompt}]

    response = await ai_tool_response(messages=request.messages, tools=request.tools or AI_Tools,
                                      model_name=request.model_name, model_id=request.model_id,
                                      top_p=request.top_p, temperature=request.temperature)

    if not response:
        raise HTTPException(status_code=500, detail="No response from AI model.")

    if request.tools:  # 自定义tools直接返回
        return JSONResponse(response)

    # 解析响应并调用工具
    return JSONResponse(await ai_tools_messages(response))


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


@app.post("/llm")  # response_model=OpenAIResponse
async def generate_text(request: CompletionParams):
    # f"You asked: {query}\nHere are some search results:\n{search_summary}\nBased on these results, here's some information:\n"
    # prompt = "以下是最近的对话内容，请生成一个摘要：\n\n"  # 请根据对话内容将会议的讨论内容整理成纪要,从中提炼出关键信息,将会议内容按照主题或讨论点分组,列出决定事项和待办事项。
    # prompt += "\n".join(str(msg) for msg in conversation),"\n".join(conversation_history[-10:])
    # prompt += "\n\n摘要：",Summary
    system_prompt = request.prompt or System_content.get(request.agent, '')
    user_request = request.question
    refer = await retrieved_reference(request.question, request.keywords, tool_calls=None)
    if refer:
        formatted_refer = '\n'.join(map(str, refer))
        user_request = f'参考材料:\n{formatted_refer}\n 材料仅供参考,请回答下面的问题:{request.question}'

    if request.stream:
        async def stream_response() -> AsyncGenerator[str, None]:
            async for chunk in await ai_generate(
                    prompt=system_prompt,
                    user_request=user_request,
                    suffix=request.suffix,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    model_name=request.model_name,
                    model_id=request.model_id,
                    stream=True, get_content=True):
                yield chunk
                await asyncio.sleep(0.01)  # 模拟 Token 逐步返回的延迟

        return StreamingResponse(stream_response(), media_type="text/plain")
    else:
        bot_response = await ai_generate(
            prompt=system_prompt,
            user_request=user_request,
            suffix=request.suffix,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            model_name=request.model_name,
            model_id=request.model_id,
            stream=False, get_content=True)
        return {"completion": bot_response, 'reference': refer,
                'transform': extract_string(bot_response, request.extract)}  # OpenAIResponse(response=generated_text)


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
    chat_history = ChatHistory(request.username, request.robot_id, request.user_id, agent, model_name,
                               timestamp=time.time(), db=db, request_uuid=request.uuid)

    if not extract:
        agent_format = {
            '3': 'code.python',
            '2': 'json',
            '4': 'json',
            '5': 'code.sql',
            '6': 'header',
        }
        extract = agent_format.get(agent, request.extract)

    history, user_request = chat_history.build(request.question, request.messages, request.use_hist,
                                               request.filter_limit, request.filter_time)

    system_prompt = request.prompt or System_content.get(agent, '')
    agent_funcalls = [agent_func_calls(agent)]
    model_info, payload, refer = await get_chat_payload(
        messages=history, user_request=user_request, system=system_prompt,
        temperature=request.temperature, top_p=request.top_p, max_tokens=request.max_tokens,
        model_name=model_name, model_id=request.model_id,
        tool_calls=agent_funcalls, keywords=request.keywords)

    if request.stream:
        async def generate_stream() -> AsyncGenerator[str, None]:
            if refer:
                first_data = json.dumps({'role': 'reference', 'content': refer},
                                        ensure_ascii=False)  # '\n'.join(refer)
                yield f'data: {first_data}\n\n'

            assistant_response = []
            async for content in ai_chat_async(model_info, payload):
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

            chat_history.save(user_request, bot_response, refer, transform, payload['model'])

        # generate() , media_type="text/plain"
        return StreamingResponse(generate_stream(), media_type="text/event-stream")
    else:
        bot_response = await ai_chat(model_info, payload)
        # print(bot_response)
        transform = extract_string(bot_response, extract)
        chat_history.save(user_request, bot_response, refer, transform, payload['model'])

        return JSONResponse({'answer': bot_response, 'reference': refer, 'transform': transform})


# /v1/completions
# /v1/embeddings
# /v1/audio/transcriptions
# @require_api_key
@app.post("/v1/chat/completions", response_model=OpenAIResponse)
async def chat_completions(request: OpenAIRequest):
    """
    兼容 OpenAI API 的 /v1/chat/completions 路径，返回类似 OpenAI API 的格式
    """
    # print(request.dict())
    if request.stream:
        async def fake_stream_response():
            async for chunk in ai_chat_async(model_info=None, messages=[msg.dict() for msg in request.messages],
                                             user_request=None, system=None,
                                             temperature=request.temperature, max_tokens=request.max_tokens,
                                             top_p=request.top_p,
                                             model_name=request.model, model_id=0, get_content=False):
                # print(chunk.encode("utf-8"))
                yield f"data: {chunk}\n\n"

                await asyncio.sleep(0.01)

            yield 'data: [DONE]\n\n'

        return StreamingResponse(fake_stream_response(), media_type="text/event-stream")
    else:
        response = await ai_chat(model_info=None, messages=[msg.dict() for msg in request.messages],
                                 user_request=None, system=None,
                                 temperature=request.temperature, max_tokens=request.max_tokens,
                                 top_p=request.top_p,
                                 model_name=request.model, model_id=0, get_content=False)
        # print(response)
        return JSONResponse(content=response)


@app.get("/v1/models")
async def get_models(model: Optional[str] = Query(None,
                                                  description=f"Retrieves a model instance, providing basic information about the model such as the owner and permissioning. e.g., 'moonshot', 'glm', 'qwen', 'ernie', 'hunyuan', 'doubao','spark','baichuan','deepseek' or custom models.")):
    if model:
        model_info, model_id = find_ai_model(model, 0, 'model')
        response_data = {
            "id": model_id,
            "object": "model",
            "created": 1686935002 if model_info['supported_openai'] else 0,
            "owned_by": model_info['name']
        }
    else:
        extracted_data = extract_ai_model("model")
        response_data = {
            "object": "list",
            "data": [
                {
                    "id": model_id,  # 用于指定模型进行请求 fine-tuned-model
                    "object": "model",
                    "created": 1677645600,
                    "owned_by": owner,
                    "permission": [
                        {
                            "id": f"modelperm-{owner}:{model_id}",
                            # "object": "model_permission",
                            # "created": 1677645600,
                            # "allow_create_engine": True,
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
                    "root": model_id,  # 根版本，与 ID 相同
                    "parent": None  # 如果没有父模型，则为 None
                } for i, (owner, models) in enumerate(extracted_data)
                for j, model_id in enumerate(models)]
        }
        # print(len(MODEL_LIST.models))
        # response_data['data'].append({"id": 'gpt-4o-mini',
        #                               "object": "model", "created": 0, "owned_by": 'openai'})
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
            user_name = request.get('username')
            robot_id = request.get('robot_id')
            user_id = request.get('user_id')
            current_timestamp = time.time()

            # 构建聊天历史记录
            history, user_request, hist_size = build_chat_history(
                user_name, request.get('question'), robot_id, user_id, db=db,
                user_history=request.get('messages', []), use_hist=request.get('use_hist', True),
                filter_limit=request.get('filter_limit', -500), filter_time=request.get('filter_time', 0.0),
                agent=agent, request_uuid=request.get('uuid')
            )

            # 生成系统提示和模型请求
            system_prompt = request.get('prompt') or System_content.get(agent, '')
            agent_funcalls = [agent_func_calls(agent)]
            model_info, payload, refer = await get_chat_payload(
                messages=history, user_request=user_request,
                system=system_prompt, temperature=request.get('temperature', 0.4),
                top_p=request.get('top_p', 0.8), max_tokens=request.get('max_tokens', 1024),
                model_name=model_name, model_id=request.get('model_id', 0),
                tool_calls=agent_funcalls, keywords=request.get('keywords', [])
            )
            if request.get('stream', True):
                async def generate_stream() -> AsyncGenerator[str, None]:
                    if refer:
                        first_data = json.dumps({'role': 'reference', 'content': refer}, ensure_ascii=False)
                        yield f'data: {first_data}\n\n'

                    assistant_response = []
                    async for content in ai_chat_async(model_info, payload):
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
                        user_name, user_request, bot_response, robot_id, user_id, agent,
                        hist_size, model_name, current_timestamp, db=db,
                        refer=refer, transform=transform, request_uuid=request.get('uuid'))

                # 流式传输消息到 WebSocket
                async for stream_chunk in generate_stream():
                    await websocket.send_text(stream_chunk)

            else:  # 非流式响应处理
                bot_response = await ai_chat(model_info, payload)
                transform = extract_string(bot_response, extract)

                # 保存聊天记录
                save_chat_history(
                    user_name, user_request, bot_response, robot_id, user_id, agent,
                    hist_size, model_name, current_timestamp, db=db,
                    refer=refer, transform=transform, request_uuid=request.get('uuid')
                )

                await websocket.send_text(
                    json.dumps({'answer': bot_response, 'reference': refer, 'transform': transform}))
                # await asyncio.sleep(0.1)

            if system_prompt.lower() == "bye":
                await websocket.send_text("Closing connection")
                await websocket.close()
                break

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Connection error: {e}")
        await websocket.close()


@app.get("/get_messages/")
async def get_messages(request: Request, user_name: str = Query(""), robot_id: str = Query(""),
                       user_id: str = Query(""), filter_time: float = Query(0.0),
                       agent: Optional[str] = Query(None), db: Session = Depends(get_db)):
    request_uuid = request.session.get('user_id', '')
    if not user_name and not request_uuid:
        return JSONResponse(status_code=400, content={"error": "No username found in session"})
    # filter_time = filter_time / 1000.0
    user_history = get_user_history(user_name, robot_id, user_id, filter_time, db, agent=agent,
                                    request_uuid=request_uuid)
    return JSONResponse(content=sorted(user_history, key=lambda x: x['timestamp']))


@app.post("/submit_messages")
async def submit_messages(request: SubmitMessagesRequest, background_tasks: BackgroundTasks,
                          db: Session = Depends(get_db)):
    if len(Task_queue) > Config.MAX_TASKS:
        return JSONResponse(status_code=400, content={'task_id': '', "error": "任务队列已满"})
    if not request.messages and not request.params:
        return JSONResponse(status_code=400,
                            content={'task_id': '', 'error': 'Please provide messages or a question to process.'})

    user_name = request.username
    current_timestamp = time.time()
    chat_history = ChatHistory(user_name, request.robot_id, request.user_id, agent=None, model_name=None,
                               timestamp=current_timestamp, db=db, request_uuid=request.uuid)

    history, user_request = chat_history.build('', request.messages, request.use_hist,
                                               request.filter_limit, request.filter_time)

    # history, user_request, hist_size = build_chat_history(
    #     user_name, "", request.robot_id, request.user_id, db, user_history=request.messages,
    #     use_hist=request.use_hist, filter_limit=request.filter_limit, filter_time=request.filter_time,
    #     agent=None,request_uuid=request.uuid)

    task_id = str(uuid.uuid4())
    Task_queue[task_id] = {
        "status": TaskStatus.PENDING,
        "action": 'message',
        "description": user_request,

        "username": user_name,
        "messages": history,
        "chat_history": chat_history,

        "response": None,
        "start_time": current_timestamp,
        "priority": 10,  # 优先级分数score
    }

    if request.params:
        background_tasks.add_task(process_task_ai, task_id, request.params)
        # asyncio.create_task(process_task_ai(task_id, request.params))

    return JSONResponse(content={'task_id': task_id})


# async def process_multiple_tasks(task_ids: list, messages: list):
#     # 使用 asyncio.gather 并发执行多个异步任务
#     tasks = [process_message_in_threadpool(task_id, message) for task_id, message in zip(task_ids, messages)]
#     await asyncio.gather(*tasks)

async def process_task_ai(task_id: str, params: List[CompletionParams]):
    task = Task_queue.get(task_id)
    if not task:
        return
    task['status'] = TaskStatus.IN_PROGRESS
    history: List[dict] = task.get('messages', [])
    user_request = task['description']

    async def single_param(i: int, param: CompletionParams):
        if param.stream:
            pass
        if not param.question:
            param.question = user_request

        local_history = copy.deepcopy(history)  # history.copy()
        if local_history[-1]["role"] == 'user':
            local_history[-1]['content'] = param.question

        agent_funcalls = [agent_func_calls(param.agent)]
        system_prompt = param.prompt or System_content.get(param.agent, '')
        model_info, payload, refer = await get_chat_payload(messages=local_history, user_request=param.question,
                                                            system=system_prompt, temperature=param.temperature,
                                                            top_p=param.top_p, max_tokens=param.max_tokens,
                                                            model_name=param.model_name, model_id=param.model_id,
                                                            tool_calls=agent_funcalls, keywords=param.keywords)
        # **param.asdict(),payload=param.payload()
        bot_response = await ai_chat(model_info, payload)
        transform = extract_string(bot_response, param.extract)
        return {'answer': bot_response, 'reference': refer, 'transform': transform, 'id': i}

    tasks = [single_param(i, p) for i, p in enumerate(params) if not p.stream]
    results = await asyncio.gather(*tasks)
    task["response"] = [r for r in results if r]
    task['status'] = TaskStatus.COMPLETED
    # await asyncio.get_event_loop().run_in_executor(None, ai_chat,loop = asyncio.get_running_loop() await loop.run_in_executor(executor, ai_chat,
    # executor.submit(ai_chat, *args) future.result()   await asyncio.wait_for(ai_chat(*args), timeout=10)
    # print(task)


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
                                description="Specify the model to use, e.g., 'moonshot', 'glm', 'qwen', 'ernie', 'hunyuan', 'doubao','spark','baichuan' or custom models."),
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
async def response_message(task_id: str,
                           param: CompletionParams = Depends(get_ai_param)) -> StreamingResponse or JSONResponse:
    task = Task_queue.get(task_id)
    if not task:
        error_data = {"error": "Invalid task ID", 'content': task_id}
        return JSONResponse(content=error_data, status_code=404)

    if task['action'] != 'message':
        error_data = {"error": "Invalid task action", 'content': task['action']}
        return JSONResponse(content=error_data, status_code=400)

    if task['status'] == TaskStatus.COMPLETED or task['status'] == TaskStatus.RECEIVED:
        task['status'] = TaskStatus.RECEIVED
        return JSONResponse(content=task["response"])

    if task['status'] == TaskStatus.IN_PROGRESS:
        error_data = {"error": "Task already in progress", 'content': task_id}
        return JSONResponse(content=error_data, status_code=409)

    task['status'] = TaskStatus.IN_PROGRESS
    history: List[dict] = task.get('messages', [])
    chat_history: ChatHistory = task['chat_history']

    if not param.question:
        param.question = task['description']

    agent_funcalls = [agent_func_calls(param.agent)]
    system_prompt = param.prompt or System_content.get(param.agent, '')
    model_info, payload, refer = await get_chat_payload(messages=history, user_request=param.question,
                                                        system=system_prompt, temperature=param.temperature,
                                                        top_p=param.top_p, max_tokens=param.max_tokens,
                                                        model_name=param.model_name, model_id=param.model_id,
                                                        tool_calls=agent_funcalls, keywords=param.keywords)

    if param.stream:
        async def generate():
            if refer:
                first_data = json.dumps({'role': 'reference', 'content': refer}, ensure_ascii=False)
                yield f'data: {first_data}\n\n'

            assistant_response = []
            async for content in ai_chat_async(model_info, payload):
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

            chat_history.save(param.question, bot_response, refer, transform, payload['model'])
            task['status'] = TaskStatus.RECEIVED

        return StreamingResponse(generate(), media_type="text/event-stream")

    bot_response = await ai_chat(model_info, payload)
    transform = extract_string(bot_response, param.extract)

    chat_history.save(param.question, bot_response, refer, transform, payload['model'])
    # del Task_queue[task_id]
    task["response"] = [{'answer': bot_response, 'reference': refer, 'transform': transform, 'id': 0}]
    task['status'] = TaskStatus.RECEIVED
    return JSONResponse(content=task["response"])


@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    task = Task_queue.get(task_id)
    if not task:
        return {"task_id": task_id, 'error': "Invalid task ID,Task not found"}
    return {"task_id": task_id, "status": task['status'], "action": task['action']}
    # return [{"task_id": v["name"], "description": v["description"], "status": v["status"], 'action': v['action']} for v in Task_graph.vs]


# 执行任务
@app.post("/execute_task/")
def execute_task(task_id: str):
    try:
        update_task_status(task_id, TaskStatus.COMPLETED)  # source status:edge["condition"]
        # 触发依赖任务
        check_and_trigger_tasks(Task_graph)
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
    file_path = Path(Config.DATA_FOLDER) / file.filename
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(await file.read())
        url = f"{Config.LOCAL_URL}/files/{file.filename}"

    if oss_expires != 0:
        object_name = f"upload/{file.filename}"
        with open(file_path, 'rb') as file_obj:
            url = upload_file_to_oss(AliyunBucket, file_obj, object_name, expires=oss_expires)
        os.remove(file_path)

    return {"url": url}


@app.get("/files/{filename}")
async def file_handler(filename: str = None, url: str = None):
    data_folder = Path(Config.DATA_FOLDER)
    data_folder.mkdir(parents=True, exist_ok=True)  # 确保目标文件夹存在

    if url and is_url(url):
        file_path, file_name = await download_file(url, data_folder)
        if file_path:
            return FileResponse(file_path)
    elif filename:
        file_path = data_folder / filename
        if file_path.exists():
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
        is_existing = False
        job_id = f"to_wechat_{generate_hash_key(user_name, context, url)}"
        if scheduler.get_job(job_id):
            print('job_existing:', job_id, form_data)  # job.remove()
            is_existing = True
        scheduler.add_job(send_to_wechat, 'date', id=job_id, run_date=send_time, misfire_grace_time=50,
                          args=[user_name, context, url, object_name], replace_existing=is_existing)
    else:
        send_to_wechat(user_name, context, url, object_name)

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
async def text_to_image(prompt: str, negative_prompt: str = '', style_name: str = '人像写真', model_id: int = 0):
    if model_id == 0:
        image_decode, result = dashscope_image_call(prompt, negative_prompt=negative_prompt, image_url='',
                                                    style_name=style_name, model_name="wanx-v1", data_folder=None)

    elif model_id == 1:
        image_decode, result = await tencent_generate_image(prompt, negative_prompt, style_name, return_url=False)
    elif model_id == 2:
        image_decode, result = await siliconflow_generate_image(prompt, negative_prompt, model_name=style_name,
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
    agent_funcalls = [agent_func_calls(request.agent)]
    model_info, payload, refer = await get_chat_payload(
        messages=[], user_request=request.question, system=system_prompt,
        temperature=request.temperature, top_p=request.top_p, max_tokens=request.max_tokens,
        model_name=request.model_name, model_id=request.model_id,
        tool_calls=agent_funcalls, keywords=request.keywords, images=urls)

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

    return JSONResponse(ai_files_messages(saved_file_paths, question, model_name, model_id, max_tokens=4096))


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

    uvicorn.run(app, host="0.0.0.0", port=7000)
