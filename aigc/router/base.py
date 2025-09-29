from fastapi.templating import Jinja2Templates
from fastapi import APIRouter, Request, Query, Body
from fastapi.responses import HTMLResponse, JSONResponse
from urllib.parse import quote
from typing import Any
from utils import format_for_html, clean_escaped_string, format_summary_text

templates = Jinja2Templates(directory="templates")

index_router = APIRouter()


@index_router.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    return templates.TemplateResponse("navigation.html", {"request": request})


@index_router.get("/markdown/", response_class=HTMLResponse)
async def get_markdown(text: str = Query(default="hello,word!")):
    text = clean_escaped_string(text)
    return HTMLResponse(format_for_html(text, True), status_code=200)


def make_md_link(text: str, max_len: int = 50):
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
    if isinstance(data, str):  # 处理纯文本 string body
        text = clean_escaped_string(data)
    elif isinstance(data, dict) and isinstance(data.get("text", None), str):
        text = clean_escaped_string(data["text"])
    elif isinstance(data, (dict, list)):
        text = format_summary_text(data)
    else:
        return HTMLResponse("<em>无法解析的输入</em>", status_code=400)

    if isinstance(text, (dict, list)):
        print('object markdown:', text)
        text = format_summary_text(text)

    return HTMLResponse(format_for_html(text, bool(data.get("html", True))), status_code=200)


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
