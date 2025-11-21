from fastapi import File, UploadFile
from router.base import *
from pydantic import BaseModel, Field
from typing import Optional, List
import pandas as pd

from database import get_database_manager
from utils import *
from config import Config
from service.task_ops import TaskManager
from service import BaseMysql, OperationMysql, DB_Client, get_redis

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


async def run_structure_sample(limit: int = 100, yd_type='开户', model='qwen:qwen3-32b'):
    from script.insight_reject import structure_sample_applynote_api
    # 更新状态
    db_manager = get_database_manager()
    with db_manager.get_engine().connect() as conn:
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

    if not all(isinstance(r, dict) for r in result):
        raise RuntimeError('structure_sample result error:{}'.format(result))

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
    return result


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
    process_task = TaskManager.task_node(action="classify", redis=redis, description=request.yd_type,
                                         params=request.model_dump(), data={})(run_structure_sample)

    task_id = await process_task(limit=request.limit, yd_type=request.yd_type, model=request.model)
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
    html = build_table_html(df.to_html(escape=False, index=False), 'Question Check')
    return HTMLResponse(content=html, status_code=200)


async def filter_risk_task_summary_question(date: str = None, is_test: bool = False, final_score: float = 0,
                                            last_id: int = 0):
    table_task = 'task_batch'
    table_name = 'task_summary_question'
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
    db_config = Config.Risk_DB_CONFIG.copy()
    db_config['database'] = 'llm_infr_test' if is_test else 'llm_infr'
    df = await asyncio.to_thread(OperationMysql.read, db_config, sql, (date_str, date_str, last_id))
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
async def get_task_summary_question(date: str = None, duplicate: bool = False, ret_object: bool = False,
                                    is_test: bool = False, final_score: float = 0):
    """
    /ideatech/task_summary_question/filter?final_score=-60
    pd.read_json('http://127.0.0.1:7000/ideatech/task_summary_question/filter?final_score=-60&ret_object=true')
    pd.read_json('http://127.0.0.1:7000/ideatech/task_summary_question/filter?final_score=0&ret_object=true&duplicate=true&date=2025-07-01')
    final_score<0:score<=final_score
    final_score>0:score>=final_score
    final_score=0:adjustments_score<0
    """
    df, _ = await filter_risk_task_summary_question(date, is_test, final_score)
    if duplicate:
        df = df.sort_values('created_at').drop_duplicates('company_name', keep='last').reset_index(drop=True)
    if ret_object:
        return df.to_dict(orient="list")

    df['summary_answer'] = df['summary_answer'].map(make_md_data_link)
    html = build_table_html(df.to_html(escape=False, index=False), 'Summary Question Filter')
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
                                                        update_fields=update_fields, with_new=False, dumps=False)
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
        dict(is_test=False, final_score=-41),  # 中高风险
        dict(is_test=True, final_score=-41),
        dict(is_test=False, final_score=58),  # 中低风险
        dict(is_test=True, final_score=58),
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
    """
    /ideatech/task_summary_question/filter_adjustment?ret_object=false&group=true
    /ideatech/task_summary_question/filter_adjustment?final_score=-57&group=false
    pd.read_json('http://127.0.0.1:7000/ideatech/task_summary_question/filter_adjustment?ret_object=true&group=false')
    """
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
        return df.to_dict(orient="records")

    if grouped is not None:
        html_df = grouped.to_html(escape=False, index=False)
    else:
        df['summary_answer'] = df['summary_answer'].map(make_md_data_link)
        html_df = df.to_html(escape=False, index=False)
    html = build_table_html(html_df, 'Summary Question Filter Adjustment',
                            additional=f',范围:[{date_str},{last_at}]')
    return HTMLResponse(content=html, status_code=200)


async def filter_risk_task_summary_question_co(date: str = None, is_test: bool = False, final_score: float = 0,
                                               last_id: int = 0):
    df, _ = await filter_risk_task_summary_question(date, is_test, final_score, last_id)
    # .groupby('company_name').tail(1)
    last_batch = df.sort_values('created_at').drop_duplicates('company_name', keep='last').reset_index(drop=True)
    if last_batch.empty:
        return last_batch
    if "adjustments" in last_batch.columns:
        # last_batch["adj_item"] = last_batch["adjustments"].str.split(';')
        last_batch["adj_item"] = last_batch["adjustments"].dropna().apply(
            lambda lst: [re.sub(r"\([+-]?\d+(?:\.\d+)?\)$", "", s).strip() for s in lst]).apply(
            lambda lst: [x for x in lst if isinstance(x, str) and x.strip()])
        remove = ['经营异常信息', '经营异常名录', '经营异常隐患', '被执行信息', '失信信息', '注销/吊销',
                  '严重违法信息']
        mask = last_batch["adj_item"].apply(lambda lst: any(r in item for item in lst for r in remove))
        last_batch = last_batch[~mask].reset_index(drop=True)
    else:
        last_batch['adjustments'] = None
        last_batch["adj_item"] = None

    batch = tuple(set(last_batch.batch_no.dropna()))
    if not batch:
        last_batch["question_result"] = None
        return last_batch

    conditions = "has_valid_data=1 AND `status`='completed' AND question_no in ('Q020','Q021','Q024','Q025')"
    db_config = Config.Risk_DB_CONFIG.copy()
    db_config['database'] = 'llm_infr_test' if is_test else 'llm_infr'
    with BaseMysql.get_cursor(db_config) as cur:
        res, cols = BaseMysql.query_batches(cur, batch, index_key='batch_no', table_name='task_question',
                                            fields=['batch_no', 'origin_question', 'result'],
                                            conditions=conditions)
        df_question = pd.DataFrame(res, columns=cols)
    grouped = df_question.groupby('batch_no', group_keys=False).apply(
        lambda x: dict(zip(x['origin_question'], x['result']))).reset_index(name='question_result')
    last_batch = last_batch.merge(grouped, on='batch_no', how='left')
    return last_batch


@ideatech_router.get("/task_summary_question/filter_co")
async def get_task_summary_question(date: str = None, ret_object: bool = False,
                                    is_test: bool = False, final_score: float = 0):
    """
    /ideatech/task_summary_question/filter_co?final_score=-60
    final_score<0:score<=final_score
    final_score>0:score>=final_score
    final_score=0:adjustments_score<0
    """
    df = await filter_risk_task_summary_question_co(date, is_test, final_score)
    if ret_object:
        return df.to_dict(orient="list")

    df['summary_answer'] = df['summary_answer'].map(make_md_data_link)
    df['question_result'] = df['question_result'].map(make_md_data_link)
    html = build_table_html(df.to_html(escape=False, index=False), 'Summary Question Filter Co')
    return HTMLResponse(content=html, status_code=200)


@ideatech_router.get("/task_summary_question/risk_case/{company_names}")
async def get_task_summary_question_result(company_names: str, duplicate: bool = False, ret_object: bool = False):
    '''
    http://127.0.0.1:7000/ideatech/task_summary_question/risk_case/云南新创智联供应链有限公司,云南创投易泰网络科技有限公司经开分公司
    '''
    co_names = company_names.replace("，", ",").split(',')
    with BaseMysql.get_cursor(Config.Risk_DB_CONFIG) as cur:
        res, cols = BaseMysql.query_batches(cur, co_names, index_key='company_name', table_name='risk_case_result')
    df = pd.DataFrame(res, columns=cols)
    if duplicate:
        df = df.sort_values('created_at').drop_duplicates(subset=['company_name', 'batch_no'], keep='last').reset_index(
            drop=True)
    if ret_object:
        return df.to_dict(orient="list")

    for col in ['summary_answer', 'question_result']:
        if col in df.columns:
            df[col] = df[col].map(make_md_data_link)
    for col in ['model_response', 'model_response2', 'case']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: make_md_data_link(x, is_json=False))

    html = build_table_html(df.to_html(escape=False, index=False), 'Summary Question Risk Case')
    return HTMLResponse(content=html, status_code=200)


async def get_risk_task_summary_question(batch_no: str, is_test: bool = False, question_no: list = None,
                                         duplicate: bool = True):
    sql = """
        SELECT 
            t1.id,
            t1.batch_no,
            t1.created_at,
            t1.updated_at,
            t2.completed_at,
            t2.company_name,
            t1.summary_answer,
            t3.origin_question,
            t3.result
        FROM task_summary_question t1
        JOIN task_batch t2
            ON t1.batch_no = t2.batch_no
        LEFT JOIN task_question t3
            ON t1.batch_no = t3.batch_no
            AND t3.status = 'completed'
            AND t3.has_valid_data = 1
        WHERE 
            t1.batch_no = %s
            AND t1.status = 'completed'
        """
    db_config = Config.Risk_DB_CONFIG.copy()
    db_config['database'] = 'llm_infr_test' if is_test else 'llm_infr'
    params = [batch_no]
    if question_no:  # ['Q020','Q021','Q024','Q025']
        placeholders = ', '.join(['%s'] * len(question_no))
        sql += f" AND t3.question_no in ({placeholders})"
        params.extend(question_no)

    params = tuple(params)
    df = await asyncio.to_thread(OperationMysql.read, db_config, sql, params)
    if df.empty or not duplicate:
        return df
    grouped = df.groupby('batch_no', group_keys=False).apply(
        lambda x: dict(zip(x['origin_question'], x['result']))).reset_index(name='question_result')
    last_row = df.drop(columns=['origin_question', 'result'], errors='ignore').drop_duplicates(
        subset=['id']).iloc[-1].to_frame().T
    return last_row.merge(grouped, on='batch_no', how='left')


@ideatech_router.get("/task_summary_question/batch/{batch_no}")
async def get_task_batch_summary_question_result(batch_no: str, is_test: bool = False, ret_object: bool = False,
                                                 duplicate: bool = False, question_no: str = None):
    '''
     http://127.0.0.1:7000/ideatech/task_summary_question/batch/TASK_1762845816635_afa5e0db?duplicate=true
    '''
    question_no = question_no.replace("，", ",").split(',') if question_no else None
    df = await get_risk_task_summary_question(batch_no, is_test, question_no, duplicate)
    if ret_object:
        return df.to_dict(orient="list" if duplicate else "records")

    for col in ['summary_answer', 'question_result']:
        if col in df.columns:
            df[col] = df[col].map(make_md_data_link)

    if 'result' in df.columns:
        df['result'] = df['result'].apply(lambda x: make_md_data_link(x, is_json=False))

    html = build_table_html(df.to_html(escape=False, index=False), 'Summary Question Batch')
    return HTMLResponse(content=html, status_code=200)


@ideatech_router.post("/task_summary_question/speech_analyze/v1/{batch_no}")
async def summary_speech_analyze_v1(batch_no: str, file: UploadFile = File(None), audio_url: Optional[str] = None,
                                    is_test: bool = False, question_no: str = None,
                                    model: str = 'deepseek-reasoner', max_tokens: int = 8192,
                                    oss_expires: int = 86400, interval: int = 3, timeout: int = 300):
    # TASK_1761719089866_69dbddae,云南创投易泰网络科技有限公司,11389271598
    question_no = question_no.replace("，", ",").split(',') if question_no else None
    df = await get_risk_task_summary_question(batch_no, is_test, question_no, True)
    if df.empty:
        return {"error": "no summary data found for this batch"}

    from agents.ai_generates import ai_analyze
    from agents.ai_multi import tencent_speech_to_text
    from agents.ai_prompt import System_content
    from utils import extract_transcription_segments
    from service import upload_file_to_oss, AliyunBucket

    if audio_url:  # 如果提供了 audio_url，优先使用
        result = await tencent_speech_to_text(None, audio_url, interval, timeout)
    elif file:
        raw_bytes = await file.read()
        file_obj = io.BytesIO(raw_bytes)
        file_obj.name = file.filename
        file_obj.seek(0, os.SEEK_END)
        total_size = file_obj.tell()  # os.path.getsize(file_path)
        file_obj.seek(0)

        if total_size > 1024 * 1024 * 5:  # 大文件 → OSS URL 模式
            object_name = f"upload/{file.filename}"
            audio_url, _ = await asyncio.to_thread(upload_file_to_oss, AliyunBucket, file_obj, object_name,
                                                   expires=oss_expires, total_size=total_size)
            result = await tencent_speech_to_text(None, audio_url, interval, timeout)
        else:
            result = await tencent_speech_to_text(file_obj, None, interval, timeout)
    else:
        data = {"error": "Either 'audio_url' or 'file' must be provided."}
        data.update(df.where(pd.notnull(df), None).to_dict(orient="list"))
        return data

    df["audio_url"] = audio_url
    if "error" in result:
        data = {"error": result["error"] or "ASR service returned unexpected result"}
        data.update(df.where(pd.notnull(df), None).to_dict(orient="list"))
        return data

    transcription_cleand = extract_transcription_segments(result['text'])
    df["transcription"] = transcription_cleand
    system_p = System_content.get('138')

    def build_prompt_template(company_name, summary_answer, transcription_text) -> str:
        prompt = f"""当前企业名称：{company_name}

    【任务要求】
    做一次完整的开户风险综合分析，不复述内容、不灌水、不搞成形式主义，重点：
    - 核实经营真实性
    - 识别可疑点（尤其是法人、业务、场地、供应链三个核心）
    - 给开户可行性结论
    - 如果可开户，明确要补的材料

    【A. 企业分析总结】：{summary_answer}

    【B. 沟通记录】:{transcription_text}
    """
        return prompt.strip()

    summary_answer = df["summary_answer"].iloc[0]
    company_name = df["company_name"].iloc[0]
    user_p = build_prompt_template(company_name, summary_answer, transcription_cleand)
    analyze = await ai_analyze(results=user_p, system_prompt=system_p, model=model, max_tokens=max_tokens)
    df["model_completion"] = analyze
    return df.to_dict(orient="list")


@ideatech_router.post("/task_summary_question/speech_analyze/{batch_no}")
async def summary_speech_analyze(batch_no: str, file: UploadFile = File(None), audio_url: Optional[str] = None,
                                 is_test: bool = False, question_no: str = None,
                                 model: str = 'deepseek-reasoner', max_tokens: int = 8192,
                                 oss_expires: int = 86400):
    """
    接收语音文件并调用 AI 模型处理,基于文件内容与文本内容生成消息。
    """
    # TASK_1761719089866_69dbddae,云南创投易泰网络科技有限公司,11389271598
    question_no = question_no.replace("，", ",").split(',') if question_no else None
    df = await get_risk_task_summary_question(batch_no, is_test, question_no, True)
    if df.empty:
        return {"error": "no summary data found for this batch"}
    row = df.iloc[0]
    full_res = row.to_dict()
    results = {
        "A. 企业分析总结": row["summary_answer"],
        "企业名称": row["company_name"],
        "C. 单项分析": row["question_result"],
    }

    file_obj = None
    if file:
        raw_bytes = await file.read()
        file_obj = io.BytesIO(raw_bytes)
        file_obj.name = file.filename

    redis = get_redis()
    from agents.ai_generates import ai_speech_analyze
    from agents.ai_prompt import System_content
    system_prompt = System_content.get('139')

    @TaskManager.task_node(action="speech_analyze", redis=redis, description="开户风险综合分析",
                           params={"batch_no": batch_no, "audio_url": audio_url, "question_no": question_no},
                           data=results)
    async def _run(_context: dict = None):
        analyze: dict = await ai_speech_analyze(file_obj, audio_url, results, system_prompt,
                                                model, max_tokens, oss_expires, interval=3, timeout=600,
                                                dbpool=DB_Client)
        full_res.update(analyze)
        keep = ['batch_no', 'company_name', 'summary_answer', 'question_result', 'audio_url', 'transcription',
                'model_completion', 'created_at']
        data = {k: v for k, v in full_res.items() if k in keep}
        data['summary_answer'] = BaseMysql.format_value(data.get('summary_answer'))
        data['question_result'] = BaseMysql.format_value(data.get('question_result'))
        data['status'] = 'completed'
        sql, values = BaseMysql.build_insert('task_asr_llm', params_data=data,
                                             explode=False, with_new=False)
        BaseMysql.write(Config.Risk_DB_CONFIG, sql, values)
        return full_res

    task_id = await _run()
    return {"task_id": task_id, 'url': f'{Config.WEBUI_URL}/task/{task_id}',
            'result': f'{Config.WEBUI_URL}/get/{TaskManager.key_prefix}:{task_id}', 'data': full_res}
