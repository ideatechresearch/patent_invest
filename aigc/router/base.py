from fastapi.templating import Jinja2Templates
from fastapi import APIRouter, Request, Query, Body
from fastapi.responses import Response, HTMLResponse, JSONResponse
from urllib.parse import quote
from typing import Any
import uuid, json
from utils import format_for_html, clean_escaped_string, format_summary_text

templates = Jinja2Templates(directory="templates")

index_router = APIRouter()


@index_router.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    return templates.TemplateResponse("navigation.html", {"request": request})


@index_router.get("/login")
async def login_page(request: Request):
    """
    登录页路由：若无 session 则分配唯一 user_id。
    """
    session_uid = request.session.get('user_id')
    if not session_uid:
        session_uid = str(uuid.uuid4())
        request.session['user_id'] = session_uid
    return templates.TemplateResponse("login.html", {"request": request})


@index_router.get("/register")
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})


@index_router.get("/api")
async def api_user_page(request: Request):
    return templates.TemplateResponse("api_user.html", {"request": request})


@index_router.get("/markdown/", response_class=HTMLResponse)
async def get_markdown(text: str = Query(default="hello,word!")):
    text = clean_escaped_string(text)
    return HTMLResponse(format_for_html(text, True), status_code=200)


def make_md_link(text: str, max_len: int = 50):
    """
    浏览器和服务器对 URL 长度有上限，一般在 2000–8000 字符之间，过长可能会被截断。
    """
    # for row in data:
    #     for k, v in row.items():
    #         if "content" in k.lower():
    #             row[k] = make_md_link(v)

    if not isinstance(text, str):
        return ""
    preview = text[:max_len].replace("\n", " ") + ("..." if len(text) > max_len else "")
    url = f"/markdown/?text={quote(text)}"
    return f'<a href="{url}" target="_blank">{preview}</a>'


templates.env.filters["make_md_link"] = make_md_link


@index_router.post("/markdown/", response_class=HTMLResponse)
async def post_markdown(data: Any = Body(...)):
    """
     统一 Markdown 渲染接口：
     - 如果 data 是 str 或 {"text": str} → 渲染普通 Markdown
     - 如果 data 是 dict 或 list[dict] → 渲染结构化 summary Markdown
     """
    for_html = True
    if isinstance(data, str):  # 处理纯文本 string body
        text = clean_escaped_string(data)
    elif isinstance(data, dict):
        for_html = bool(data.pop("html", True))
        if isinstance(data.get("text", None), str):
            text = clean_escaped_string(data["text"])
        else:
            text = format_summary_text(data)
    elif isinstance(data, list):
        text = format_summary_text(data)
    else:
        return HTMLResponse("<em>无法解析的输入</em>", status_code=400)

    if isinstance(text, (dict, list)):
        print('object markdown:', text)
        text = format_summary_text(text)

    return HTMLResponse(format_for_html(text, for_html), status_code=200)


def make_md_data_link(data: Any, max_len: int = 75, is_json: bool = True):
    """
    将文本或 Python 对象生成 HTML 点击链接，点击后通过 POST 发送到 /markdown/。
    - data 可以是 str, dict, list
    - max_len 控制 preview 长度
    """
    import html
    if data is None:
        return ""

    # 预览文本
    preview_text = json.dumps(data, ensure_ascii=False) if isinstance(data, (dict, list)) else str(data)
    preview = preview_text[:max_len].replace("\n", " ") + ("..." if len(preview_text) > max_len else "")
    preview = html.escape(preview)  # 转义引号
    uid = "md_" + uuid.uuid4().hex
    if is_json and isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as e:
            print(f'[make_md_data_link]:{e}\n{data}')
    json_data = json.dumps(data, ensure_ascii=False)
    # 返回点击可触发 JS fetch 的链接
    html_link = f'''
    <a href="#" class="md-link" data-json-id="{uid}">{preview}</a>
    <script type="application/json" id="{uid}">{json_data}</script>
    '''
    return html_link.strip()


def md_data_script():
    return '''
    <script>
    document.addEventListener('DOMContentLoaded', () => {
        document.querySelectorAll('.md-link').forEach(el => {
            el.addEventListener('click', function(e) {
                e.preventDefault();
                const jsonId = el.dataset.jsonId;
                const scriptTag = document.getElementById(jsonId);
                if (!scriptTag) {
                    console.error("No script tag found for", jsonId);
                    return;
                }
                try {
                    const payload = JSON.parse(scriptTag.textContent.trim());
                    const w = window.open();
                    fetch("/markdown/", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(payload)
                    })
                    .then(resp => resp.text())
                    .then(html => {
                        w.document.write(html);
                        w.document.close();
                    })
                    .catch(err => {
                        w.close();
                        console.error(err);
                    });
                } catch(err) {
                    console.error("Invalid JSON in script tag:", err);
                }
            });
        });
    });
    </script>
    '''


@index_router.post("/markdown/check")
async def post_markdown_check(data: Any = Body(...)):
    from script.md_checker import run_all_checks
    if isinstance(data, str):  # 处理纯文本 string body
        text = clean_escaped_string(data)
    elif isinstance(data, dict) and isinstance(data.get("text", None), str):
        text = data["text"]
    elif isinstance(data, (dict, list)):
        text = format_summary_text(data)
    else:
        return JSONResponse("无法解析的输入")

    issues = run_all_checks(text)
    return JSONResponse(content=issues)
