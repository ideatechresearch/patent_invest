from fastapi import APIRouter
from fastapi import Query, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
from co_utils import *

questions_router = APIRouter()


@questions_router.get("/generate_questions/{batch_no}")
async def trigger_summary_questions(batch_no: str):
    # 校验批次是否存在
    task_batch = await dbop.query_one("SELECT title FROM task_batch WHERE batch_no=%s", (batch_no,))
    if not task_batch:
        raise HTTPException(status_code=404, detail=f"batch_no '{batch_no}' 不存在")

    existing = await dbop.query_one(
        "SELECT status FROM due_diligence_questions WHERE batch_no=%s ORDER BY updated_at DESC LIMIT 1",
        (batch_no,)
    )

    if existing:
        status = existing["status"]
        if status in ("running", "created"):
            return {"status": status, "message": "已存在，当前状态无需重新触发。"}
        elif status in ("completed", "failed"):
            # 删除现有记录，重新触发
            await dbop.run("DELETE FROM due_diligence_questions WHERE batch_no=%s", (batch_no,))
            print(f'重新生成{batch_no}')

    full_title = task_batch['title']  # 'title': f"企业分析任务 - {company_name}"
    company_name = full_title.split(' - ')[1].strip() if ' - ' in full_title else full_title
    # 否则调用生成任务，首次触发任务；或者 failed 重试,有数据会等待任务执行完成
    asyncio.create_task(run_question_summary(batch_no, company_name))

    return {"status": "created", "message": "任务已触发，将在数据准备后自动运行"}


@questions_router.get("/get_generate_questions/{batch_no}")
async def query_summary_questions(batch_no: str):
    existing = await dbop.query_one(
        "SELECT question_text,question_id,status FROM due_diligence_questions WHERE batch_no=%s ORDER BY updated_at DESC LIMIT 1",
        (batch_no,)
    )

    if not existing:
        return {}  # { "status": "not_found","message": "尚未触发尽调提问任务，请先调用 /generate_questions/{batch_no} 接口启动问题生成流程。"}

    status = existing["status"]

    def parse_questions(raw: str):
        try:
            parsed = json.loads(raw)
            return [d if isinstance(d, dict) else {"question": str(d)} for d in parsed]
        except json.JSONDecodeError:
            return [{'question': q.strip()} for q in raw.split("\n\n") if q.strip()]

    if status == "completed":
        question_list = parse_questions(existing["question_text"])
        return {
            "status": status,
            "questions": [{'seq': i, **d} for i, d in enumerate(question_list)],
            "question_id": existing['question_id']
        }

    if status == "running":
        return {
            "status": status,
            "message": "任务执行中，将自动开始生成问题，请稍后刷新。"
        }
    # waiting for ready then running->completed.
    if status == "created":
        return {
            "status": status,
            "message": "任务已创建，正在等待所需数据，稍后将自动运行。"
        }

    return {
        "status": status,
        "message": "任务状态异常，请检查日志或重新触发。"
    }


class QuestionAnswerRequest(BaseModel):
    seq_list: Optional[List[int]] = Field(default_factory=list)
    role: Literal['positive', 'negative'] = 'negative'
    reference: List[str] = ['工商全维度信息', '股权结构', '被执行信息', "最终受益人识别", "立案信息"]
    judge: bool = False


@questions_router.post("/generate_questions_answer/{question_id}")
async def generate_questions_answer(question_id: str, request: QuestionAnswerRequest):
    existing = await dbop.query_one(
        "SELECT question_text,batch_no,question_id,question_type,status FROM due_diligence_questions WHERE question_id=%s",
        (question_id,)
    )
    if not existing:
        return {"status": "not_found",
                "message": "尚未找到对应尽调问题记录，请确认 question_id 是否正确，或先触发问题生成任务。"}

    def parse_questions(raw: str):
        try:
            parsed = json.loads(raw)
            return [d if isinstance(d, dict) else {"question": str(d)} for d in parsed]
        except json.JSONDecodeError:
            return [{'question': q.strip()} for q in raw.split("\n\n") if q.strip()]

    question_list = parse_questions(existing["question_text"])
    question_result = await due_diligence_questions_answer(question_list, request.seq_list, existing, request.reference,
                                                           role=request.role, judge=request.judge)
    return JSONResponse(content=question_result)


@questions_router.get("/get_summary_evaluate/{question_id}")
async def generate_summary_evaluate(question_id: str):
    existing = await dbop.query_one(
        "SELECT batch_no, company_name FROM due_diligence_answer WHERE question_id=%s",
        (question_id,)
    )
    if not existing:
        return {"status": "not_found",
                "message": "尚未找到对应尽调问题记录，请确认 question_id 是否正确，或先触发问题生成任务。"}

    batch_no, company_name = existing['batch_no'], existing['company_name']
    return await run_summary_evaluate(batch_no, company_name)


async def wait_then_retry_summary(batch_no, company_name,
                                  reference: tuple[str] = ('工商全维度信息', '股权结构', '被执行信息', "最终受益人识别",
                                                           "立案信息"),
                                  max_wait_secs=300, check_interval=10):
    waited = 0
    ready = False
    interface_keys = tuple(reference)
    placeholders = ', '.join(['%s'] * len(interface_keys))
    while waited < max_wait_secs:
        # 检查依赖数据是否就绪
        sql_list = [
            ("SELECT 1 FROM task_summary_question WHERE batch_no=%s AND status='completed'", (batch_no,)),
            (f"SELECT COUNT(*) AS cnt FROM task_question WHERE batch_no=%s AND origin_question IN ({placeholders}) AND status='completed'",
             (batch_no, *interface_keys)),
        ]
        summary_ready, task_ready = await dbop.query(sql_list, fetch_all=True)

        if summary_ready and task_ready and task_ready[0]['cnt'] >= 2:
            ready = True
        await asyncio.sleep(check_interval)
        waited += check_interval

    if ready:
        await run_question_summary(batch_no, company_name, reference)


async def run_question_summary(batch_no, company_name,
                               reference: tuple[str] = ('工商全维度信息', '股权结构', '被执行信息', "最终受益人识别",
                                                        "立案信息")):
    question_type = f"尽调提问 - {company_name}"
    await dbop.run("""
        INSERT INTO due_diligence_questions (batch_no, status, question_type)
        VALUES (%s, %s, %s) AS new
        ON DUPLICATE KEY UPDATE 
            question_type = new.question_type,
            updated_at = NOW()
    """, (batch_no, "created", question_type))

    sql_list = [
        ("SELECT summary_answer FROM task_summary_question WHERE batch_no=%s AND status= %s", (batch_no, 'completed')),
        ("SELECT origin_question,result,status,interface_result FROM task_question WHERE batch_no=%s AND status=%s",
         (batch_no, 'completed')),
        ("SELECT status, retry_count FROM due_diligence_questions WHERE batch_no=%s", (batch_no,)),
    ]
    summary_row, question_rows, existing = await dbop.query(sql_list, fetch_all=True)
    summary_answer = summary_row[0]['summary_answer'] if summary_row else None
    interface_filter = {row["origin_question"]: row["result"] or row["interface_result"] for row in question_rows if
                        row["origin_question"] in reference}

    # ready = await wait_for_dependencies(batch_no)
    if len(interface_filter) < len(reference) or not summary_answer:
        retry_count = existing[0].get("retry_count", 0) if existing else 0
        # 启动后台等待任务，waiting,若状态为 created/failed 且已重试过一次，则不再执行等待任务
        if existing and existing[0]["status"] in ("created", "failed") and retry_count >= 2:
            return None

        await dbop.run("""
             UPDATE due_diligence_questions SET retry_count=%s, updated_at=NOW() WHERE batch_no=%s
         """, (retry_count + 1, batch_no,))
        asyncio.create_task(wait_then_retry_summary(batch_no, company_name, reference))
        return None

    field_map = {
        "core_strengths": "核心优势与稳定性",
        "major_risks": "重大风险与问题",
        "risk_table": "风险关联性分析",
        "conclusion_advice": "结论与建议",
    }
    try:
        summary_data = json.loads(summary_answer)
        summary_text = render_summary_text(field_map, summary_data)
    except json.JSONDecodeError:
        try:
            summary_data = json.loads(extract_json_str(summary_answer))
            summary_text = render_summary_text(field_map, summary_data)
        except:
            summary_text = summary_answer

    question_prompt = SYS_PROMPT.get('question_prompt_0')

    try:
        await dbop.run("UPDATE due_diligence_questions SET status='running', updated_at=NOW() WHERE batch_no=%s",
                       (batch_no,))

        desc = f'企业【{company_name}】的《整体风险与运营状况总结》报告，报告内容涵盖了股东背景、治理结构、财务情况、法律合规、经营状态等方面的信息'
        question_result = await ai_analyze(question_prompt, {"总结报告文章内容": summary_text, **interface_filter},
                                           client=ai_client, desc=desc,
                                           model='deepseek-chat', max_tokens=8192, top_p=0.85)
        parsed = extract_json_array(question_result)
        question_list = [d for d in parsed if isinstance(d, (str, dict)) and str(d).strip()] if isinstance(parsed,
                                                                                                           list) else []

        combined_text = json.dumps(question_list,
                                   ensure_ascii=False) if question_list else question_result  # "\n\n".join(chunk_lists)

        question_id = uuid.uuid4().hex[:16]
        await dbop.run(
            "UPDATE due_diligence_questions SET question_text=%s,question_id=%s,status='completed',model=%s WHERE batch_no=%s",
            (combined_text, question_id, 'deepseek-chat', batch_no))

        return combined_text

    except Exception as e:
        print(f"[Question Error] => {str(e)}")
        await dbop.run("UPDATE due_diligence_questions SET status='failed' WHERE batch_no=%s",
                       (batch_no,))
    return None


async def due_diligence_questions_answer(question_list: list[dict], seq_list: list[int], existing: dict,
                                         reference: list[str], role='negative', judge=False):
    # '''   你的问题将用于 A/B 角色模拟问答：
    #
    #     A 模型代表真实陈述，B 模型尝试闪避或模糊表达；
    #     设计问题时，请注重逻辑清晰与信息可验证性。
    # '''
    batch_no = existing['batch_no']
    question_id = existing['question_id']
    question_type = existing['question_type']  # f"尽调提问 - {company_name}"
    company_name = question_type.split(' - ')[1].strip() if ' - ' in question_type else question_type
    interface_keys = tuple(reference)
    placeholders = ', '.join(['%s'] * len(interface_keys))
    task_question_rows = await dbop.run(
        f"SELECT origin_question,result,status,interface_result FROM task_question WHERE batch_no=%s AND origin_question IN ({placeholders})",
        (batch_no, *interface_keys))
    interface_result = {row["origin_question"]: row["interface_result"] or row['result'] for row in task_question_rows}
    prompt_template = (SYS_PROMPT.get('prompt_a') if role == 'positive' else SYS_PROMPT.get('prompt_b'))
    prompt_evaluate = SYS_PROMPT.get('evaluate')
    # {
    #     "question": "我们注意到公司法定代表人在一年内变更了两次，是否涉及管理层的重大调整？",
    #     "person": "法人代表",
    #     "action": "核实工商变更记录及董事会决议",
    #     "topic": "治理结构"
    # }
    tasks = []
    for i, q in enumerate(question_list):
        if seq_list and i not in seq_list:
            continue

        async def process_question(i=i, q=q):
            question_text = q.get("question", "").strip()
            action = q.get("action", "联系相关人员核实")
            topic = q.get("topic", "其他")
            params_data = {'batch_no': batch_no, 'question_id': question_id, 'seq': i, 'company_name': company_name,
                           'question': question_text, 'action': action, 'topic': topic, 'role': role,
                           'status': "running"}
            insert_id = await dbop.insert('due_diligence_answer', params_data)

            prompt_i = prompt_template.format(company_name=company_name, person=q.get("person", "相关负责人"))
            answer = await ai_analyze(
                system_prompt=prompt_i,
                results=interface_result,
                client=ai_client,
                desc=f'💬 问题：{question_text},对方可能需要：{action},企业基本信息',
                model='deepseek-chat',
                max_tokens=2048,
                top_p=0.7
            )
            await dbop.run("""UPDATE due_diligence_answer SET status='ready', answer=%s
                                WHERE id=%s
                            """, (answer, insert_id))
            evaluate_data = None
            if judge:
                prompt_j = prompt_evaluate.format(question=question_text, answer=answer)
                evaluate_result = await ai_analyze(
                    system_prompt=prompt_j,
                    results=interface_result,
                    client=ai_client,
                    desc=f'公司信息等背景材料（可参考但不强制逐条引用）',
                    model='deepseek-chat',
                    max_tokens=8192,
                    top_p=0.85
                )
                parsed = extract_json_struct(evaluate_result)
                evaluate_data = {k: d for k, d in parsed.items() if
                                 isinstance(d, (str, dict, list)) and str(d).strip()} if isinstance(parsed,
                                                                                                    dict) else {}
                evaluate = json.dumps(evaluate_data, ensure_ascii=False, indent=2) if evaluate_data else (
                        extract_json_str(evaluate_result) or evaluate_result)
                await dbop.run("""UPDATE due_diligence_answer SET status='completed', evaluate=%s
                                  WHERE id=%s
                               """, (evaluate, insert_id))

            return {
                'seq': i,
                'question': question_text,
                'answer': answer,
                'topic': topic,
                'evaluate': evaluate_data
            }

        tasks.append(process_question())

    return await asyncio.gather(*tasks)


async def run_summary_evaluate(batch_no, company_name):
    sql_list = [
        ("SELECT question, answer, action, topic FROM due_diligence_answer WHERE batch_no = %s AND answer IS NOT NULL",
         (batch_no,)),
        ("SELECT origin_question,result FROM task_question WHERE batch_no=%s", (batch_no,))]
    question_answer_list, task_analysis = await dbop.query(sql_list)

    all_results = [row["result"] for row in task_analysis]
    joined_title = "、".join(row["origin_question"] for row in task_analysis)
    data = {'question_answer': question_answer_list, 'analysis': all_results}
    evaluate_prompt = SYS_PROMPT.get('evaluate_prompt').format(joined_title=joined_title, company_name=company_name)
    try:
        # await dbop.run("""UPDATE due_diligence_answer SET status='completed', evaluate=%s
        #                     WHERE id=%s
        #                 """, (evaluate, insert_id))
        evaluate_result = await ai_analyze(
            system_prompt=evaluate_prompt,
            results=data,
            client=ai_client,
            desc=f'尽调问答与下公司信息等背景材料（可参考但不强制逐条引用）企业各项分析',
            model='deepseek-reasoner',
            max_tokens=8192,
            top_p=0.85
        )
        parsed = extract_json_struct(evaluate_result)
        evaluate_data = {k: d for k, d in parsed.items() if
                         isinstance(d, (str, dict, list)) and str(d).strip()} if isinstance(
            parsed, dict) else {}

        evaluate = json.dumps(evaluate_data, ensure_ascii=False, indent=2) if evaluate_data else (
                extract_json_str(evaluate_result) or evaluate_result)

        return evaluate_data

        # await dbop.execute([
        #     ("""
        #            UPDATE task_summary_question
        #            SET summary_answer=%s, model=%s, status='completed'
        #            WHERE batch_no=%s
        #            """,
        #      (combined_text, 'deepseek-reasoner', batch_no)
        #      ),
        #     ("""
        #            UPDATE task_batch SET status='completed', completed_at=NOW()
        #            WHERE batch_no=%s
        #            """,
        #      (batch_no,)
        #      )
        # ])
        # print(render_summary_text(summary_data) if summary_data else combined_text)
    except:
        pass


if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI

    app = FastAPI(lifespan=lifespan)
    try:
        uvicorn.run(app, host="0.0.0.0", port=8001)
    except KeyboardInterrupt:
        print("Server stopped by user")
