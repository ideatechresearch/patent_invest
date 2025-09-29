from router.base import *
from fastapi import Request, Depends, Query, Form
from fastapi.responses import RedirectResponse
from typing import Optional, Dict
from service import DB_Client

table_router = APIRouter()
table_router_prefix = "/table"
REGISTERED_TABLES = ['agent_history', 'agent_robot', 'chat_history']  # 注册的表名列表


@table_router.get("/list", response_class=HTMLResponse)
async def list_table(request: Request):
    return templates.TemplateResponse("table_index.html",
                                      {"request": request, "tables": REGISTERED_TABLES, "prefix": table_router_prefix})


@table_router.get("/{table_name}", response_class=HTMLResponse)
async def show_table_form(request: Request, table_name: str, edit_id: Optional[str] = Query(None)):
    if table_name not in REGISTERED_TABLES:
        return RedirectResponse(f"{table_router_prefix}/list")

    data = None
    async with DB_Client.get_cursor() as cur:
        columns = await DB_Client.get_table_columns(table_name, cur)
        if edit_id:
            primary_key = await DB_Client.get_primary_key(table_name, cur)
            if primary_key:
                data = await DB_Client.query_one(f"SELECT * FROM {table_name} WHERE {primary_key} = %s", (edit_id,),
                                                 cursor=cur)
    return templates.TemplateResponse("table_form.html", {
        "request": request,
        "table_name": table_name,
        "columns": columns,
        "data": data,
        "is_edit": edit_id is not None,
        "prefix": table_router_prefix,
    })


@table_router.post("/submit/{table_name}", response_class=HTMLResponse)
async def submit_data(request: Request, table_name: str,
                      edit_id: Optional[str] = Form(None), form_data: Dict[str, str] = Depends(lambda x: dict(x))):
    if table_name not in REGISTERED_TABLES:
        return RedirectResponse(f"{table_router_prefix}/list")
    # 过滤掉csrf_token等非数据库字段
    form_data = {k: v for k, v in form_data.items() if not k.startswith("_")}
    conn = await DB_Client.get_conn()
    try:
        async with DB_Client.get_cursor(conn) as cur:
            # 获取表结构以验证数据
            columns = await DB_Client.get_table_columns(table_name, cur)
            column_names = [col["name"] for col in columns]
            valid_data = {k: v for k, v in form_data.items() if k in column_names}  # 过滤掉不存在的列

            primary_key = await DB_Client.get_primary_key(table_name, cur)
            if edit_id and primary_key:
                # 更新操作
                await DB_Client.async_update(table_name, valid_data, edit_id, primary_key, conn)
                message = "数据更新成功"
            else:
                # 插入操作
                await DB_Client.async_insert(table_name, valid_data, conn)
                message = "数据添加成功"

            # 获取操作后的数据（便于回显）
            order_col = primary_key or column_names[0]
            await cur.execute(
                f"SELECT * FROM {table_name} ORDER BY `{order_col}` DESC LIMIT 10")
            recent_data = await cur.fetchall()

        return templates.TemplateResponse("submit_success.html", {
            "request": request,
            "table_name": table_name,
            "data": valid_data,
            "recent_data": recent_data,
            "message": message,
            "primary_key": primary_key,
            "prefix": table_router_prefix
        })
    except Exception as e:
        return {"request": request, "error": str(e)}

    finally:
        DB_Client.release(conn)


@table_router.get("/view/{table_name}", response_class=HTMLResponse)
async def view_table_data(request: Request, table_name: str,
                          page: int = Query(None, ge=1), per_page: int = Query(10, ge=1, le=100)):
    if table_name not in REGISTERED_TABLES:
        return RedirectResponse(f"{table_router_prefix}/list")

    async with DB_Client.get_cursor() as cur:
        offset, page, total_pages, total = await DB_Client.get_offset(table_name, page=page, per_page=per_page,
                                                                      cursor=cur)
        columns = await DB_Client.get_table_columns(table_name, cur)
        primary_key = await DB_Client.get_primary_key(table_name, cur)

        order_clause = f"ORDER BY {primary_key} ASC" if primary_key else ""
        data_query = f"SELECT * FROM {table_name} {order_clause} LIMIT %s OFFSET %s"
        results = await DB_Client.async_query([(data_query, (per_page, offset))], cursor=cur)
        data = results[0] if results else None

    return templates.TemplateResponse("table_view.html", {
        "request": request,
        "table_name": table_name,
        "columns": columns,
        "data": data,
        "primary_key": primary_key,
        "page": page,
        "per_page": per_page,
        "total": total,
        "total_pages": total_pages,
        "prefix": table_router_prefix
    })


@table_router.get("/delete/{table_name}/{row_id}", response_class=HTMLResponse)
async def delete_data(
        request: Request,
        table_name: str,
        row_id: str
):
    if table_name not in REGISTERED_TABLES:
        return RedirectResponse(f"{table_router_prefix}/list")

    primary_key = await DB_Client.get_primary_key(table_name)
    if not primary_key:
        return {"request": request, "error": "找不到主键，无法删除"}
    res = await DB_Client.async_run(f"DELETE FROM {table_name} WHERE {primary_key} = %s", (row_id,))
    return RedirectResponse(f"{table_router_prefix}/view/{table_name}?message=删除成功")
