# -*- coding: utf-8 -*-
import string, difflib, re, time, copy, os, io, sys, uuid
import tempfile
import logging
import concurrent.futures
# import spacy
# import queue
# import numpy as np
from typing import AsyncGenerator
from web3 import Web3
from fastapi import FastAPI, Request, Response, Depends, Query, File, BackgroundTasks, UploadFile, Form, \
    Body, WebSocket, WebSocketDisconnect, HTTPException, status
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import OAuth2PasswordBearer
from starlette.middleware.sessions import SessionMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
# from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.executors.pool import ThreadPoolExecutor
from contextlib import asynccontextmanager
from passlib.context import CryptContext
from qdrant_client import AsyncQdrantClient

from structs import *
from generates import *
from config import *
from database import *


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up.")
    Base.metadata.create_all(bind=engine)
    init_ai_clients()
    # if not w3.is_connected():
    #     print("Failed to connect to Ethereum node")

    scheduler.add_job(tick, 'interval', seconds=60, misfire_grace_time=60)  # , max_instances=3
    scheduler.start()
    yield
    print("Shutting down.")
    scheduler.shutdown()


scheduler = BackgroundScheduler(executors={'default': ThreadPoolExecutor(2)})  # 设置线程池大小, AsyncIOScheduler()
# executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
engine = create_engine(Config.SQLALCHEMY_DATABASE_URI)  # echo=True
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
app = FastAPI(lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key=Config.SECRET_KEY)
# logging.basicConfig(level=logging.INFO)
qd_client = AsyncQdrantClient(host=Config.QDRANT_HOST, grpc_port=6334, prefer_grpc=True)
w3 = Web3(Web3.HTTPProvider(f'https://mainnet.infura.io/v3/{Config.INFURA_PROJECT_ID}'))  # ("http://127.0.0.1:8545")
Task_queue = {}  # queue.Queue(maxsize=Config.MAX_TASKS)

dashscope.api_key = Config.DashScope_Service_Key
# 加密配置,密码哈希上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# OAuth2 密码令牌,设置 tokenUrl
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# model_config['protected_namespaces'] = ()


def tick():
    db = next(get_db())
    if len(Task_queue):
        print(len(Task_queue), 'Tick! The time is: %s' % datetime.now())
        cleanup_tasks()

    if len(Chat_history):
        print(len(Chat_history), 'Tick! The time is: %s' % datetime.now())
        ChatHistory.sequential_insert(db, Chat_history)
    # db.close()


def cleanup_tasks(timeout_received=600, timeout=86400):
    current_time = time.time()
    task_ids_to_delete = []
    for task_id, task in Task_queue.items():
        t_sec = current_time - task['timestamp']
        if t_sec > timeout_received:
            if task['status'] == TaskStatus.RECEIVED:
                task_ids_to_delete.append(task_id)
                print(f"Task {task_id} has been marked for cleanup. Status: RECEIVED")
        elif t_sec > timeout:
            task_ids_to_delete.append(task_id)
            print(f"Task {task_id} has been marked for cleanup. Timeout exceeded")

    for task_id in task_ids_to_delete:
        del Task_queue[task_id]


# async def process_multiple_tasks(task_ids: list, messages: list):
#     # 使用 asyncio.gather 并发执行多个异步任务
#     tasks = [process_message_in_threadpool(task_id, message) for task_id, message in zip(task_ids, messages)]
#     await asyncio.gather(*tasks)
async def async_cron():
    while True:
        print('Async Tick! The time is: %s' % time.time())
        await asyncio.sleep(60)


def get_db():
    db = SessionLocal()
    try:
        yield db
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
        if eth_address:
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
        if eth_address:
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
        return {'history': Chat_history, 'task': Task_queue}
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


@app.get("/")
async def index():
    return {"message": "Hello World"}


@app.get('/web_search/{text}')
async def web_search(text: str, platform: str = 'default'):
    if platform == 'wikipedia':
        return wikipedia_search(text)
    return JSONResponse(await web_search_async(text))


@app.get('/retrieval/{text}')
async def retrieval(text: str, platform: str = 'default'):
    return {'knowledge', 'retrieval pass'}


@app.get("/embeddings/")
async def embeddings(texts: List[str] = Query(...), model_name: str = 'qwen', model_id: int = 0):
    inputs = [text.replace("\n", " ") for text in texts]
    if model_name == 'baidu':
        access_token = get_baidu_access_token()
        embedding = await get_baidu_embeddings(inputs, access_token=access_token)
        return {"embedding": embedding}
    return {"embedding": await ai_embeddings(inputs, model_name=model_name, model_id=model_id)}


@app.get("/fuzzy/")
async def fuzzy_matches(texts: List[str] = Query(...), terms: List[str] = Query(...),
                        top_n: int = 3, cutoff: float = 0.6, method: str = 'levenshtein'):
    querys = [text.replace("\n", " ").strip() for text in texts]
    tokens = [text.replace("\n", " ").strip() for text in terms]
    results = []
    if method == 'levenshtein':
        for token in querys:
            matches = difflib.get_close_matches(token, tokens, n=top_n, cutoff=cutoff)
            matches = [(match, round(difflib.SequenceMatcher(None, token, match).ratio(), 3), tokens.index(match))
                       for match in matches]
            results.append({'query': token, 'matches': matches})
        # results = [{'token': token, 'matches': rapidfuzz.process.extract(token, tokens, limit=top_n, score_cutoff=cutoff)}
        #            for token in querys]
        # fuzzywuzzy.process.extractOne(token,choices,scorer=fuzzywuzzy.fuzz.token_sort_ratio)
    elif method == 'bm25':
        bm25 = BM25(tokens)  # corpus
        for token in querys:
            scores = bm25.rank_documents(token)
            matches = [(tokens[match[0]], round(match[1], 3), match[0]) for match in scores[:top_n] if
                       match[1] >= cutoff]
            results.append({'query': token, 'matches': matches})
    elif method == 'reranker':
        async def process_token(token):
            scores = await ai_reranker(token, documents=tokens, top_n=top_n,
                                       model_name="BAAI/bge-reranker-v2-m3", model_id=0)
            matches = [(match[0], round(match[1], 3), match[2]) for match in scores if
                       match[1] >= cutoff]
            return {'query': token, 'matches': matches}

        results = await asyncio.gather(*(process_token(token) for token in querys))  # [(match,score,index)]
    elif method == 'embeddings':
        similars = await get_similar_embeddings(querys, tokens, ai_embeddings, topn=top_n, cutoff=cutoff,
                                                model_name='qwen', model_id=0)
        results = [{'query': token, 'matches':
            [(match[0], round(match[1], 3), tokens.index(match[0])) for match in matches if match[1] >= cutoff]} for
                   token, matches in similars]

    # tokens = list({match for token in querys for match in difflib.get_close_matches(token, tokens)})
    # match, score = process.extractOne(token, tokens)
    # results.append({'token': token, 'match': match, 'score': score})
    return JSONResponse(content=results, media_type="application/json; charset=utf-8")
    # Response(content=list_to_xml('results', results), media_type='application/xml; charset=utf-8')


@app.get("/nlp/")
async def nlp():
    return {"message": "ok"}


@app.post("/llm")  # response_model=OpenAIResponse
async def generate_text(request: CompletionParams):
    # f"You asked: {query}\nHere are some search results:\n{search_summary}\nBased on these results, here's some information:\n"
    # prompt = "以下是最近的对话内容，请生成一个摘要：\n\n"  # 请根据对话内容将会议的讨论内容整理成纪要,从中提炼出关键信息,将会议内容按照主题或讨论点分组,列出决定事项和待办事项。
    # prompt += "\n".join(str(msg) for msg in conversation),"\n".join(conversation_history[-10:])
    # prompt += "\n\n摘要：",Summary

    if request.stream:
        async def stream_response():
            async for chunk in await ai_generate(
                    prompt=request.prompt,
                    user_request=request.question,
                    suffix=request.suffix,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    model_name=request.model_name,
                    model_id=request.model_id,
                    stream=True,
            ):
                yield chunk

        return StreamingResponse(stream_response(), media_type="text/plain")
    else:
        user_request = request.question
        refer = await retrieved_reference(request.question, request.keywords, tool_calls=None)
        if refer:
            formatted_refer = '\n'.join(map(str, refer))
            user_request = f'参考材料:\n{formatted_refer}\n 材料仅供参考,请回答下面的问题:{request.question}'

        bot_response = await ai_generate(
            prompt=request.prompt,
            user_request=user_request,
            suffix=request.suffix,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            model_name=request.model_name,
            model_id=request.model_id,
            stream=False,
        )
        return {"completion": bot_response, 'reference': refer,
                'transform': extract_string(bot_response, request.extract)}  # OpenAIResponse(response=generated_text)


# ,current_user: User = Depends(get_current_user)
@app.post("/message/")
async def generate_message(request: OpenAIRequest,
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
        extract = agent_format.get(agent, extract)

    history, user_request = chat_history.build(request.question, request.messages, request.use_hist,
                                               request.filter_limit, request.filter_time)

    system_prompt = request.prompt or System_content.get(agent, '')
    agent_funcalls = [Agent_Functions.get(agent, lambda *args, **kwargs: [])]
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
                                   ensure_ascii=False)
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
                request_uuid=request.get('uuid')
            )

            # 生成系统提示和模型请求
            system_prompt = request.get('prompt') or System_content.get(agent, '')
            agent_funcalls = [Agent_Functions.get(agent, lambda *args, **kwargs: [])]
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
    #     request_uuid=request.uuid)

    task_id = str(uuid.uuid4())
    Task_queue[task_id] = {
        "status": TaskStatus.PENDING,
        "action": 'message',
        'username': user_name,
        "messages": history,
        "chat_history": chat_history,
        "user_request": user_request,

        "timestamp": current_timestamp,
        "response": None,
    }
    if request.params:
        background_tasks.add_task(process_task_ai, task_id, request.params)
        # asyncio.create_task(process_task_ai(task_id, request.params))

    return JSONResponse(content={'task_id': task_id})


async def process_task_ai(task_id: str, params: List[CompletionParams]):
    task = Task_queue.get(task_id)
    if not task:
        return
    task['status'] = TaskStatus.IN_PROGRESS
    history: List[dict] = task.get('messages', [])
    user_request = task['user_request']

    async def single_param(i: int, param: CompletionParams):
        if param.stream:
            pass
        if not param.question:
            param.question = user_request

        local_history = copy.deepcopy(history)  # history.copy()
        if local_history[-1]["role"] == 'user':
            local_history[-1]['content'] = param.question

        agent_funcalls = [Agent_Functions.get(param.agent, lambda *args, **kwargs: [])]
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
                                description="Specify the model to use, e.g., 'moonshot', 'glm', 'qwen', 'ernie', 'hunyuan', 'doubao','speark','baichuan' or custom models."),
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
        param.question = task['user_request']

    agent_funcalls = [Agent_Functions.get(param.agent, lambda *args, **kwargs: [])]
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
    return {"task_id": task_id, "status": task['status'], 'action': task['action']}


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
    platform = request.platform.lower()
    if platform == "baidu":
        translated_text = await baidu_translate(request.text, request.source, request.target)
        return {"platform": "Baidu", "translated": translated_text}
    elif platform == "tencent":
        translated_text = await tencent_translate(request.text, request.source, request.target)
        return {"platform": "Tencent", "translated": translated_text}
    elif platform == "xunfei":
        translated_text = await xunfei_translate(request.text, request.source, request.target)
        return {"platform": "XunFei", "translated": translated_text}
    else:
        system_prompt = System_content.get('9')
        translated_text = await ai_generate(
            prompt=system_prompt,
            user_request=request.text,
            model_name=platform,
            model_id=0,
            stream=False,
        )
        if translated_text:
            return {"platform": platform, "translated": translated_text}

    raise HTTPException(status_code=400, detail="Unsupported platform.")


@app.post("/ocr")
async def image_to_text(file: UploadFile = File(...), image_url: str = Form(None),
                        ocr_sign: str = 'accurate_basic'):
    try:
        image_data = await file.read()
    except Exception as e:
        return {"error": f"Failed to process the uploaded file:{e}"}

    access_token = get_baidu_access_token(Config.BAIDU_ocr_API_Key, Config.BAIDU_ocr_Secret_Key)
    result = baidu_ocr_recognise(image_data, image_url, access_token, ocr_sign=ocr_sign)
    if result is None:
        raise HTTPException(status_code=500, detail="OCR recognition failed.")

    return {"result": result}


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
    elif file_urls:
        if isinstance(file_urls, list) and len(file_urls) == 1:
            file_urls = file_urls[0].split(',')
        elif isinstance(file_urls, str):
            file_urls = file_urls.split(',')
        file_urls = [url.strip(" '\"") for url in file_urls]

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
        else:
            result = await dashscope_speech_to_text_url(file_urls)
            return JSONResponse(content={"transcriptions": result})

    return HTTPException(status_code=400, detail="Unsupported platform.")

    # curl - X
    # POST
    # "http://127.0.0.1:8000/speech-to-text" - F
    # "file=@path_to_your_audio.pcm"


@app.post("/tts")
async def text_to_audio(sentences: str, platform: str = "baidu"):
    audio_io, request_id = await dashscope_text_to_speech(sentences)
    return StreamingResponse(audio_io, media_type="audio/mpeg",
                             headers={"Content-Disposition": "attachment; filename=output.mp3",
                                      "X-Request-ID": request_id})


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
