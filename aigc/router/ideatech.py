from fastapi import File, UploadFile
from router.base import *
from pydantic import BaseModel, Field
from typing import Optional, List

from database import engine
from service import OperationMysql, DB_Client, get_redis
from utils import *
from config import Config

from agents.ai_tasks import TaskManager, TaskStatus, TaskNode

ideatech_router = APIRouter()


@ideatech_router.get("/", response_class=HTMLResponse)
async def send_page(request: Request):
    return templates.TemplateResponse("send_wechat.html", {"request": request})


@ideatech_router.post("/knowledge/")
async def knowledge(text: str, rerank_model: Optional[str] = "BAAI/bge-reranker-v2-m3",
                    file: UploadFile = File(None), version: int = 0):
    '''search_knowledge_base'''
    from script.knowledge import ideatech_knowledge
    result = await ideatech_knowledge(text.strip(), rerank_model=rerank_model, file=file, version=version)
    return {'similar': result}


async def run_structure_sample(task: TaskNode, redis=None, limit: int = 100, yd_type='开户',
                               model='qwen:qwen3-32b'):
    task_id = task.name
    try:
        import pandas as pd
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
                                             status=TaskStatus.COMPLETED if all(isinstance(r, dict) for r in result)
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

        # with OperationMysql() as session:
        #     session.execute(update_sql, params_list)

        await DB_Client.async_run(update_sql, params_list)

    except Exception as e:
        print(e)
        await TaskManager.set_task_status(task, TaskStatus.FAILED, 100, redis)


class StructureSampleRequest(BaseModel):
    yd_type: str = Field('开户', description="业务类型，例如 '开户'、'销户' 等")
    model: Optional[str] = Field("qwen:qwen3-32b", description="使用的模型名称")
    limit: Optional[int] = Field(100, description="返回的样本数量上限")


@ideatech_router.post("/sample")
async def get_sample_batch(request: StructureSampleRequest):
    """
       返回 insert / update 类型的模型判定样本，用于人工标注处理。
       自动排除已标注或已入库模板的样本。
    """
    redis = get_redis()
    task_id, task = await TaskManager.add(redis=redis,
                                          description=request.yd_type,
                                          action='classify',
                                          params=request.model_dump(),
                                          data={}
                                          )

    asyncio.create_task(
        run_structure_sample(task, redis, limit=request.limit, yd_type=request.yd_type, model=request.model))

    return {"task_id": task_id, 'url': f'{Config.WEBUI_URL}/task/{task_id}',
            'result': f'{Config.WEBUI_URL}/get/{TaskManager.key_prefix}:{task_id}'}


@ideatech_router.get("/templates/list", response_model=List[dict])
async def list_templates(only_active: bool = True, yd_type='开户'):
    ''' 查询模板，支持筛选路径、来源、状态。'''
    sql = f"SELECT * FROM semantic_template_library  WHERE 来源 = {yd_type}"
    if only_active:
        sql += " and 状态 = '启用'"
    return DB_Client.search(sql)


class SampleResult(BaseModel):
    id: int  # sample_id
    action: str
    一级类: str
    二级类: str
    三级类: str
    template: str
    status: bool


@ideatech_router.post("/confirm")
async def confirm_action(sample: SampleResult, yd_type='开户'):
    ''' 确认分类并更新模板库,人工确认 insert/update 的操作, status True 表示同意，False 表示拒绝'''
    # 更新 classify_sentences_clustered
    update_sql = """
        UPDATE classify_sentences_clustered
        SET action=%s, 一级类=%s, 二级类=%s, 三级类=%s, template=%s, 标注状态=%s
        WHERE id=%s
    """
    await DB_Client.async_run(update_sql, (
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
        await DB_Client.async_run(insert_sql, (
            yd_type, sample.一级类, sample.二级类,
            sample.三级类, sample.template, sample.id
        ))


async def check_risk_task_question(date: str = None, is_test: bool = False):
    from script.md_checker import run_all_checks
    table_name = 'llm_infr_test.task_question' if is_test else 'llm_infr.task_question'
    sql = f'''
    SELECT id, batch_no, created_at, updated_at, origin_question, question_no, result
    FROM {table_name}  WHERE created_at >= %s
    AND has_valid_data=1 AND `status`='completed'
    '''
    date_str = date or datetime.now().strftime("%Y-%m-%d 00:00:00")
    df = await asyncio.to_thread(OperationMysql.get_dataframe, Config.Risk_DB_CONFIG, sql, date_str)
    if df.empty:
        return df
    df['check'] = df.result.apply(run_all_checks)
    df['check'] = df['check'].apply(lambda res: res if res and len(res) > 0 else None)
    print(f"[check_risk_task_question] 总行数={len(df)}, 有效检查结果={df['check'].notna().sum()}")
    # icheck = df['check'].dropna().explode()
    # icheck_df = pd.json_normalize(icheck)
    # icheck_df.index = icheck.index
    df.dropna(subset=['check'], inplace=True)
    return df


@ideatech_router.get("/task_question/check")
async def check_task_question(date: str = None, ret_json: bool = False, is_test: bool = False):
    df = await check_risk_task_question(date, is_test)
    if ret_json:
        return df.to_json(orient='records', force_ascii=False)

    df['result'] = df['result'].apply(make_md_link)
    html = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>Task Question Check</title>
        <style>
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; }}
            th {{ background-color: #f4f4f4; }}
        </style>
    </head>
    <body>
        <h2>Task Question 检查结果</h2>
        {df.to_html(escape=False, index=False)}
    </body>
    </html>
    """
    return HTMLResponse(content=html, status_code=200)
