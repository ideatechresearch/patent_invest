from fastapi import File, UploadFile
from router.base import *
from pydantic import BaseModel, Field
from typing import Optional, List
import pandas as pd

from database import engine, async_engine
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
    df = await asyncio.to_thread(OperationMysql.read, Config.Risk_DB_CONFIG, sql, (date_str,))
    if df.empty:
        return df
    df['check'] = df.result.map(run_all_checks)  # with ProcessPoolExecutor() as executor:
    df['check'] = df['check'].apply(lambda res: res if res and len(res) > 0 else None)
    print(f"[check_risk_task_question] 总行数={len(df)}, 有效检查结果={df['check'].notna().sum()}")
    # icheck = df['check'].dropna().explode()
    # icheck_df = pd.json_normalize(icheck)
    # icheck_df.index = icheck.index
    df.dropna(subset=['check'], inplace=True)
    return df


@ideatech_router.get("/task_question/check")
async def check_task_question(date: str = None, ret_object: bool = False, is_test: bool = False):
    """/ideatech/task_question/check?date=2025-10-01"""
    df = await check_risk_task_question(date, is_test)
    if ret_object:
        return df.to_json(orient='records', force_ascii=False)

    df['result'] = df['result'].apply(lambda x: make_md_data_link(x, is_json=False))  # apply(make_md_link)
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
        {md_data_script()}
    </body>
    </html>
    """
    return HTMLResponse(content=html, status_code=200)


async def filter_risk_task_summary_question(date: str = None, is_test: bool = False, final_score: float = 0,
                                            last_id: int = 0):
    table_task = 'llm_infr_test.task_batch' if is_test else 'llm_infr.task_batch'
    table_name = 'llm_infr_test.task_summary_question' if is_test else 'llm_infr.task_summary_question'
    sql = f"""
    SELECT 
        t1.id,
        t1.batch_no,
        t1.created_at,
        t1.updated_at,
        t2.completed_at,
        t2.company_name,
        t1.summary_answer
    FROM {table_name} t1
    JOIN {table_task} t2
        ON t1.batch_no = t2.batch_no
    WHERE ( 
        t1.created_at > %s
        OR (t1.created_at >= %s AND t1.id > %s)
        )
        AND t1.status = 'completed';
    """
    date_str = date or datetime.now().strftime("%Y-%m-%d 00:00:00")
    df = await asyncio.to_thread(OperationMysql.read, Config.Risk_DB_CONFIG, sql, (date_str, date_str, last_id))
    if df.empty:
        return df, date_str
    # last_id = df['id'].max()
    last_at = df['created_at'].max().strftime("%Y-%m-%d %H:%M:%S")

    def get_final_score(summary_answer):
        try:
            data = json.loads(summary_answer)
            res = data.get('final_score', None)
            if res is None:
                from co_analyis.parse_score import parse_score_table_elements
                total_score, adjustments, risk_value = parse_score_table_elements(data.get('score_table', ''))
                if total_score:
                    res = {"final_score": total_score, "adjustments_score": risk_value, "adjustments": adjustments}
            elif not isinstance(res, dict):
                res = {k: v for k, v in data.items() if k in ("final_score", "adjustments_score", "adjustments")}
            return res
        except json.decoder.JSONDecodeError:
            return None

    df['final_score_data'] = df['summary_answer'].apply(get_final_score)
    if final_score:
        mask = df['final_score_data'].apply(
            lambda d: isinstance(d, dict) and (d.get("final_score", 0) - abs(final_score)) * final_score >= 0)
    else:
        mask = df['final_score_data'].apply(lambda d: isinstance(d, dict) and d.get("adjustments_score", 0) < 0)
    print(f"[filter_risk_task_summary_question] 总行数={len(df)}, 筛选结果={mask.sum()}")
    df = df[mask]
    expanded = pd.json_normalize(df['final_score_data'])
    expanded.index = df.index
    return pd.concat([df, expanded], axis=1), last_at


@ideatech_router.get("/task_summary_question/filter")
async def get_task_summary_question(date: str = None, ret_object: bool = False, is_test: bool = False,
                                    final_score: float = 0):
    """
    /ideatech/task_summary_question/filter?final_score=-60
    final_score<0:score<=final_score
    final_score>0:score>=final_score
    final_score=0:adjustments_score<0
    """
    df, _ = await filter_risk_task_summary_question(date, is_test, final_score)
    if ret_object:
        show = ['batch_no', 'company_name', "final_score", "adjustments_score", "adjustments"]
        return df[show].to_dict(orient="list")

    df['summary_answer'] = df['summary_answer'].map(make_md_data_link)
    html = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>Task Summary Question Filter</title>
        <style>
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; }}
            th {{ background-color: #f4f4f4; }}
        </style>
    </head>
    <body>
        <h2>Task Summary Question 结果</h2>
        {df.to_html(escape=False, index=False)}
        {md_data_script()}
    </body>
    </html>
    """
    return HTMLResponse(content=html, status_code=200)


async def filter_risk_task_summary_question_adjustment(date: str = None, is_test: bool = False, final_score: float = 0,
                                                       last_id: int = 0):
    df, last_at = await filter_risk_task_summary_question(date, is_test, final_score, last_id)
    if df.empty:
        return df, last_at
    if "adjustments" not in df.columns:
        df['adjustments'] = None
    df_exp = df[df["adjustments"].notna() & (df["adjustments"].str.strip() != "")].copy()
    df_exp["adj_item"] = df_exp["adjustments"]
    df_exp = df_exp.explode("adj_item", ignore_index=True)
    df_exp["adjustment"] = df_exp["adj_item"].str.replace(r"\([+-]?\d+(\.\d+)?\)$", "", regex=True).str.strip()
    df_exp["adjustment_score"] = df_exp["adj_item"].str.extract(r"\(([+-]?\d+(\.\d+)?)\)$")[0].astype(float)
    df_exp.drop(columns=["adj_item"], inplace=True)
    df_exp.sort_values(by=["adjustment", "adjustment_score", "adjustments_score", "final_score", "id"], ascending=True,
                       inplace=True)
    return df_exp, last_at  # dropna(subset='adjustment')


async def sync_risk_task_summary_question_adjustment(is_test: bool = False, final_score: float = 0) -> dict:
    '''
        CREATE TABLE IF NOT EXISTS task_summary_co_sample (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        batch_no VARCHAR(64) NOT NULL,
        company_name VARCHAR(255),
        adjustment VARCHAR(255),
        adjustment_score FLOAT,
        adjustments TEXT,
        adjustments_score FLOAT,
        final_score FLOAT,
        created_at DATETIME NOT NULL,
        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        completed_at DATETIME,
        source_id BIGINT,
        is_test TINYINT,
        score_type VARCHAR(64),
        UNIQUE KEY uq_batch_adj_type (batch_no, adjustment, score_type)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    '''
    score_type = 'high_score' if final_score > 0 else 'low_score' if final_score < 0 else 'adjustment'
    sql = "SELECT MAX(created_at) as last_at, MAX(source_id) as last_id FROM task_summary_co_sample WHERE is_test = %s AND score_type = %s;"
    df_0 = await asyncio.to_thread(OperationMysql.read, Config.Risk_DB_CONFIG, sql, (int(is_test), score_type))
    if df_0.empty:
        error = "[sync] No adjustment table records found."
        print(error)
        return {'size': 0, 'type': score_type, 'error': error}

    last_id = df_0['last_id'].iloc[0] or 0
    last_at = df_0['last_at'].iloc[0] or "2025-01-01 00:00:00"
    # dbop = OperationMysql(async_mode=True, db_config=Config.Risk_DB_CONFIG)
    df, new_last_at = await filter_risk_task_summary_question_adjustment(last_at, is_test, final_score, last_id)
    if df.empty:
        error = "[sync] No new adjustment records found."
        print(error)
        return {"from": last_at, "to": last_at, 'size': 0, 'type': score_type, 'error': error}
    size = len(df)
    df['is_test'] = int(is_test)
    df['score_type'] = score_type
    df['adjustments'] = df['adjustments'].apply(lambda x: ';'.join(x) if isinstance(x, list) else x)
    df['updated_at'] = datetime.now()

    df.rename(columns={"id": "source_id"}, inplace=True)
    # 插入字段映射
    insert_cols = [
        "source_id", "batch_no", "company_name", "adjustment", "adjustment_score",
        "adjustments", "adjustments_score", "final_score",
        "created_at", "completed_at", "updated_at", 'is_test', 'score_type'
    ]
    update_fields = ['adjustment_score', 'adjustments', 'adjustments_score', 'final_score']  # , 'updated_at'
    # 构建参数
    sql, values = OperationMysql.build_insert_dataframe('task_summary_co_sample', df=df[insert_cols],
                                                        update_fields=update_fields, with_new=False)
    if not values:
        error = "[sync] ⚠️ No valid values to insert."
        print(error)
        return {"from": last_at, "to": last_at, 'size': size, 'type': score_type, 'error': error}
    # async_merge
    await asyncio.to_thread(OperationMysql.write, Config.Risk_DB_CONFIG, sql, values)
    print(f"[sync] ✅ {size} records synced ({score_type} up to {new_last_at}).")
    return {"from": last_at, "to": new_last_at, 'size': size, 'type': score_type}


async def run_adjustment_sync_tasks(concurrent: bool = False) -> list[dict]:
    tasks = [
        dict(is_test=False, final_score=0),
        dict(is_test=True, final_score=0),
        dict(is_test=False, final_score=-60),
        dict(is_test=True, final_score=-60),
    ]
    if concurrent:
        return await asyncio.gather(*[sync_risk_task_summary_question_adjustment(**params) for params in tasks])
    results = []
    for params in tasks:  # 顺序执行
        results.append(await sync_risk_task_summary_question_adjustment(**params))
    return results


@ideatech_router.post("/task_summary_question/filter_adjustment/sync")
async def run_sync_summary_question_adjustment():
    return await run_adjustment_sync_tasks()


@ideatech_router.get("/task_summary_question/filter_adjustment")
async def get_task_summary_question_adjustment(date: str = None, ret_object: bool = False, group: bool = True,
                                               is_test: bool = False, final_score: float = 0):
    """/ideatech/task_summary_question/filter_adjustment?ret_object=false&group=true"""
    date_str = date or get_month_range(shift=0, count=1)[0]
    df, last_at = await filter_risk_task_summary_question_adjustment(date_str, is_test, final_score)
    grouped = None
    if group:
        grouped = df.groupby("adjustment").agg({
            "batch_no": list,
            "company_name": list,  # ";".join,
            "final_score": list,
            "adjustment_score": list,
            "adjustments_score": list,
        }).reset_index()

    if ret_object:
        if grouped is not None:
            return grouped.to_dict(orient="records")
        show = ["adjustment", "adjustment_score", 'batch_no', 'company_name', "final_score", "adjustments_score",
                "adjustments"]
        return df[show].to_dict(orient="records")

    if grouped is not None:
        html_df = grouped.to_html(escape=False, index=False)
    else:
        df['summary_answer'] = df['summary_answer'].map(make_md_data_link)
        html_df = df.to_html(escape=False, index=False)

    html = f"""
      <html>
      <head>
          <meta charset="utf-8">
          <title>Task Summary Question Filter</title>
          <style>
              table {{ border-collapse: collapse; width: 100%; }}
              th, td {{ border: 1px solid #ddd; padding: 8px; }}
              th {{ background-color: #f4f4f4; }}
          </style>
      </head>
      <body>
          <h2>Task Summary Question,范围:[{date_str},{last_at}]</h2>
          {html_df}
          {md_data_script()}
      </body>
      </html>
      """
    return HTMLResponse(content=html, status_code=200)
