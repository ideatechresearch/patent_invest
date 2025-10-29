from router.base import *

chat_router = APIRouter()


@chat_router.get("/", response_class=HTMLResponse)
async def chat_page(request: Request):
    # 渲染 templates/chat.html
    return templates.TemplateResponse("chat.html", {"request": request})


@chat_router.get('/prompt', response_class=HTMLResponse)
async def send_prompt(request: Request):
    return templates.TemplateResponse("prompt.html", {"request": request})


@chat_router.get("/debate", response_class=HTMLResponse)
async def send_debate(request: Request):
    return templates.TemplateResponse("debate.html", {"request": request})


@chat_router.get("/consensus", response_class=HTMLResponse)
async def send_consensus(request: Request):
    return templates.TemplateResponse("consensus.html", {"request": request})


@chat_router.get("/conversations", response_class=HTMLResponse)
async def send_conversations(request: Request):
    return templates.TemplateResponse("conversations.html", {"request": request})


AGENTS_NAME = [
    {"value": "0", "name": "问题助手"},
    {"value": "1", "name": "领域专家"},
    {"value": "2", "name": "技术专家"},
    {"value": "4", "name": "信息提取"},
    {"value": "5", "name": "SQL转换"},
    {"value": "6", "name": "著作助手"},
    {"value": "8", "name": "文本润色"},
    {"value": "11", "name": "任务规划"},
    {"value": "31", "name": "工具处理"},
    {"value": "32", "name": "工具助手"},
    {"value": "37", "name": "网络助手"},
    {"value": "43", "name": "书籍透视"},
    {"value": "72", "name": "变更描述"},
    {"value": "109", "name": "信息切片"},
    {"value": "121", "name": "文本禅师"},
    {"value": "131", "name": "周报撰写"},
    {"value": "134", "name": "英译中"},
]


@chat_router.get('/message', response_class=HTMLResponse)
async def message_page(request: Request):
    session_uid = request.session.get('user_id','')
    context = {
        "request": request,
        "user_id": session_uid,  # 后端至少提供一个匿名ID
        "agents": AGENTS_NAME,
    }
    return templates.TemplateResponse("message.html", context)


@chat_router.get('/ichat', response_class=HTMLResponse)
async def send_chat(request: Request):
    # username: str = Depends(verify_access_token)
    # username = get_access_user(request)

    session_user = request.session.get('user_id', '')
    # user_name = username or session_user
    # user = User.get_user(db, username=username, user_id=username)
    context = {
        "request": request,
        # "username": user_name,
        "uid": session_user,
        "agents": AGENTS_NAME,
    }
    # if user and not user.disabled:
    #     context["username"] = user_name
    # else:
    #     context["uid"] = session_user

    return templates.TemplateResponse("message_bak.html", context)
