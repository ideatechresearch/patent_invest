from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from co_utils import *
from co_due_diligence import questions_router
from co_analysis_search import search_router

app = FastAPI(lifespan=lifespan)

app.include_router(search_router, prefix="/ext1")
app.include_router(questions_router, prefix="/ext2")


@app.get("/start_task/")
async def start_task(company_name: str):
    batch_no, _ = await run_enterprise_analysis_task_background(company_name, app.state.httpx_client)
    return {"batch_no": batch_no, 'url': f'http://127.0.0.1:8000/get_summary_answer/{batch_no}',
            'analysis_url': f'http://127.0.0.1:8000/get_analysis_answer/{batch_no}'}


@app.get("/task_status/{batch_no}")
async def get_task_status(batch_no: str):
    sql_list = [
        ("SELECT * FROM task_batch WHERE batch_no=%s", (batch_no,)),
        ("SELECT * FROM task_question WHERE batch_no=%s", (batch_no,)),
        ("SELECT * FROM task_question_content WHERE batch_no=%s", (batch_no,)),
        ("SELECT * FROM task_summary_question WHERE batch_no=%s", (batch_no,)),
        ("SELECT * FROM due_diligence_questions WHERE batch_no=%s", (batch_no,))
    ]
    batch_info, questions, question_content, summary, diligence_questions = await dbop.query(sql_list, fetch_all=True)
    return {"batch": batch_info, "questions": questions, "question_content": question_content, "summary": summary,
            "diligence_questions": diligence_questions}


@app.get("/get_summary_answer/{batch_no}", response_class=HTMLResponse)
async def get_summary_answer(batch_no: str, model: Optional[str] = 'deepseek-reasoner'):
    if model:
        row = await dbop.query_one(
            "SELECT summary_answer FROM task_summary_question WHERE batch_no=%s AND model=%s ORDER BY updated_at DESC LIMIT 1",
            (batch_no, model))
    else:
        row = await dbop.query_one(
            "SELECT summary_answer FROM task_summary_question WHERE batch_no=%s ORDER BY updated_at DESC LIMIT 1",
            (batch_no,))

    if not row or not row.get("summary_answer"):
        return HTMLResponse(f"<h3>暂无总结报告内容，请稍后重试或检查任务状态。</h3>", status_code=200)

    '''
    ### 整体风险与运营状况总结
    **报告主体：{}

    --

    '''
    field_map = {
        "core_strengths": "核心优势与稳定性",
        "major_risks": "重大风险与问题",
        "risk_table": "风险关联性分析",
        "conclusion_advice": "结论与建议",
        "score_table": "维度评分与总分",
        "account_limit_suggestion": "推荐非柜面开户额度区间"
    }
    style = """
    <style>
        table {
            border-collapse: collapse;
            width: 100%;
        }
        th, td {
            border: 1px solid #999;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
    </style>
    """
    summary_answer = row['summary_answer']
    try:
        summary_data = json.loads(summary_answer)
        summary_text = render_summary_text(field_map, summary_data)
    except json.JSONDecodeError:
        try:
            summary_data = json.loads(extract_json_str(summary_answer))
            summary_text = render_summary_text(field_map, summary_data)
        except:
            summary_text = summary_answer

    import markdown2
    html_content = markdown2.markdown(summary_text, extras=["tables", "fenced-code-blocks", "break-on-newline"])
    full_html = f"<html><head>{style}</head><body>{html_content}</body></html>"
    return HTMLResponse(content=full_html, status_code=200)


@app.get("/get_analysis_answer/{batch_no}", response_class=HTMLResponse)
async def get_analysis_answer(batch_no: str):
    rows = await dbop.run("SELECT origin_question,result FROM task_question WHERE batch_no=%s",
                          (batch_no,))
    if not rows:
        return HTMLResponse(f"<h3>暂无分析报告内容，请稍后重试或检查任务状态。</h3>", status_code=200)

    analysis_data = {row["origin_question"]: row["result"] for row in rows}
    analysis_text = render_summary_text({}, analysis_data)
    import markdown2
    html_content = markdown2.markdown(analysis_text, extras=["tables", "fenced-code-blocks", "break-on-newline"])
    return HTMLResponse(f"<html><body>{html_content}</body></html>", status_code=200)


@app.get("/get_question_content/{batch_no}", response_class=HTMLResponse)
async def get_question_content(batch_no: str, get_content: bool = True):
    rows = await dbop.run("SELECT origin_question,interface_data,content FROM task_question_content WHERE batch_no=%s",
                          (batch_no,))
    if not rows:
        return HTMLResponse(f"<h3>暂无分析报告内容，请稍后重试或检查任务状态。</h3>", status_code=200)

    def load_data(row):
        data = row['interface_data']
        try:
            data = json.loads(row['interface_data'])
        except json.JSONDecodeError:
            pass
        return data

    analysis_data = {row["origin_question"]: row['content'] if get_content else load_data(row) for row in rows}
    analysis_text = render_summary_text({}, analysis_data)
    import markdown2
    html_content = markdown2.markdown(analysis_text, extras=["tables", "fenced-code-blocks", "break-on-newline"])
    return HTMLResponse(f"<html><body>{html_content}</body></html>", status_code=200)


@app.get("/rebuild_summary/{batch_no}")
async def rebuild_summary_answer(batch_no: str,
                                 attention_origins: list[str] = Query(default=['严重违法信息', '失信信息', '被执行信息',
                                                                               '空壳企业识别', '经营异常名录',
                                                                               '简易注销公告']),
                                 model: str = 'deepseek-reasoner'):
    task_batch, summary_row = await dbop.query([
        ("SELECT title FROM task_batch WHERE batch_no=%s",
         (batch_no,)),
        ("SELECT summary_answer,status FROM task_summary_question WHERE batch_no=%s AND model=%s ",
         (batch_no, model))
    ])

    if not task_batch:
        return {'error': "暂无该 batch_no，请稍后重试或检查任务状态。", 'status': 'not_created'}

    status = 'created'
    last_data = None
    if summary_row:
        status = summary_row[0]["status"]
        last_data = summary_row[0]["summary_answer"]
        if status in ("completed", "failed"):
            if status == "failed":  # 删除现有记录，重新触发,完成的新建一个，不同模型比较
                await dbop.run("DELETE FROM task_summary_question WHERE batch_no=%s and status=failed", (batch_no,))

    full_title = task_batch[0]['title']  # 'title': f"企业分析任务 - {company_name}"
    company_name = full_title.split(' - ')[1].strip() if ' - ' in full_title else full_title
    asyncio.create_task(run_summary_result(batch_no, company_name, attention_origins, model=model))
    return {'url': f'http://127.0.0.1:8000/get_summary_answer/{batch_no}?model={model}',
            'last_data': last_data, 'status': status}


if __name__ == "__main__":
    import uvicorn

    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except KeyboardInterrupt:
        print("Server stopped by user")
