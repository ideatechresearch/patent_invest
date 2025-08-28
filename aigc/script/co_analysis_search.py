import time

from fastapi import APIRouter
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from co_utils import *

search_router = APIRouter()


@search_router.get('/search_points')
async def search_embeddings_points(batch_no: str,
                                   querys: Union[str, List[str]] = Query(None, description="查询语句，一个或多个"),
                                   reranker: Optional[Literal['KeyWord', 'BM25', 'Hybrid', 'Hybrid_LLM', 'LLM']] = None,
                                   topn: int = 30, score_threshold: float = 0.0, model: str = 'qwen-plus'):
    # {'text': chunk, 'chunk_id': i, 'batch_no': batch_no, 'desc': origin_question}

    if not querys:
        summary_chunks_keywords = await dbop.query_one(
            "SELECT summary_chunk,operation_id FROM task_summary_chunks WHERE batch_no=%s AND status=%s",
            (batch_no, 'completed'))
        if summary_chunks_keywords:
            querys = summary_chunks_keywords['summary_chunk'].split('\n\n')

    if not querys:
        raise HTTPException(status_code=400, detail="查询语句不能为空")

    tokens = [querys.strip()] if isinstance(querys, str) else [x.replace("\n", " ").strip() for x in querys]

    match = field_match(field_key='batch_no', match_values=batch_no)
    not_match = field_match(field_key='desc', match_values='整体风险与运营状况总结')
    search_hit = await search_by_embeddings(tokens, Collection_Name, client=qd_client, emb_client=emb_client,
                                            model=MODEL_EMBEDDING, payload_key=['text', 'desc'],
                                            match=match, not_match=not_match,
                                            topn=topn, score_threshold=score_threshold, exact=False, get_dict=True)
    if not reranker:
        return {'search_hit': search_hit}

    question_content = await dbop.run(
        "SELECT origin_question,content FROM task_question_content WHERE batch_no=%s", (batch_no,))

    if reranker == 'KeyWord':
        segments = [p.get("content", '') for p in question_content]
        corpus = [para for doc in segments for para in doc.split('\n\n---\n\n') if para.strip()]
        results = []
        bm25 = BM25(corpus)
        for token in tokens:
            scores = bm25.rank_documents(token, topn)
            matches = [(corpus[match[0]], round(match[1], 3)) for match in scores]
            results.append({'query': token, 'matches': matches})

        return {'keywords': results, 'search_hit': search_hit}

    if reranker == 'BM25':
        results = []
        corpus = [p.get("payload", {}).get('text', '') for p in search_hit]
        bm25 = BM25(corpus)
        # print(bm25.doc_count, bm25.corpus)
        for token in tokens:
            scores = bm25.rank_documents(token, topn, normalize=False)
            matches = [(search_hit[match[0]], round(match[1], 3)) for match in scores]
            results.append({'query': token, 'matches': matches})

        return {'keywords': results, 'search_hit': search_hit}

    if reranker == 'Hybrid':
        def hybrid_scores(query, alpha=0.6):
            dense_scores = [item.get('score', 0.0) for item in search_hit]  # dense embedding 相似度
            sparse_scores = bm25.get_scores(query, normalize=True)  # BM25 分数
            hybrid_scores = []
            for idx, item in enumerate(search_hit):
                hybrid_score = alpha * dense_scores[idx] + (1 - alpha) * sparse_scores[idx][1]
                hybrid_scores.append((item, hybrid_score))

            hybrid_scores.sort(key=lambda x: x[1], reverse=True)
            return hybrid_scores

        corpus = [p.get("payload", {}).get('text', '') for p in search_hit]
        bm25 = BM25(corpus)
        results = [{'query': token, 'matches': hybrid_scores(token)} for token in tokens]
        return {'keywords': results, 'search_hit': search_hit}

    '''
    1.用 dense embedding 做初步向量召回（返回 top-N）
    2. 对 top-N 结果再用 BM25 或关键词打分进行 rerank 或 filter
    final_score = α * dense_similarity + (1 - α) * sparse_score
    '''

    #  3. 如果某些文本明显无关、重复度过高，或者仅是模板性描述，可直接排除。
    if reranker == 'Hybrid_LLM':
        system_prompt = SYS_PROMPT.get('search_prompt')
        corpus = search_hit
    else:
        system_prompt = SYS_PROMPT.get('search_prompt2')
        corpus = [(p.get('origin_question'), p.get("content", '').split('\n\n---\n\n')) for p in question_content]
        print(corpus)

        # corpus = corpus * 5
        # qwen-plus:18862(14390+2476),75448(56188+1758),150896(111896+2078)
        # qwen-long:113172(84042+4155),188620(139733+1610),943100(696813+1731)
        # deepseek-chat:75448(47700+2906) 158,94310
        print(sum([len(y) for x in corpus for y in x[1]]), sum(len(p.get("content", '')) for p in question_content))

    desc = (f'请根据以上规则处理，并返回一个 JSON 数组（保留{topn}个相关结果并重排）。数据如下：'
            f'\nquery: {tokens}\nsearch_hit')
    x = time.time()
    result = await ai_analyze(system_prompt, corpus, client=emb_client, desc=desc,
                              model=model, max_tokens=8192, temperature=0.2, top_p=0.85)
    print(time.time() - x)
    parsed = extract_json_array(result)
    reranker_list = [d.strip() if isinstance(d, str) else d for d in parsed] if isinstance(parsed, list) else []

    return JSONResponse(content={'similar_list': reranker_list, 'search_hit': search_hit})


@search_router.get('/get_origin')
async def get_origin_points(batch_no: str):
    task_batch, summary_row, existing = await dbop.query([
        ("SELECT title FROM task_batch WHERE batch_no=%s",
         (batch_no,)),
        ("SELECT summary_answer FROM task_summary_question WHERE batch_no=%s AND status=%s",
         (batch_no, 'completed')),
        ("SELECT status FROM task_summary_chunks WHERE batch_no=%s ORDER BY updated_at DESC LIMIT 1",
         (batch_no,))
    ])
    if existing:
        status = existing[0]["status"]
        if status in ("completed", "failed"):
            # 删除现有记录，重新触发
            await dbop.run("DELETE FROM task_summary_chunks WHERE batch_no=%s", (batch_no,))
            print(f'重新生成:{batch_no}')

    summary_answer = summary_row[0]['summary_answer'] if summary_row else None
    full_title = task_batch[0]['title']  # 'title': f"企业分析任务 - {company_name}"
    company_name = full_title.split(' - ')[1].strip() if ' - ' in full_title else full_title
    try:
        summary_data = json.loads(summary_answer)
    except json.JSONDecodeError:
        try:
            summary_data = json.loads(extract_json_str(summary_answer))
        except:
            summary_data = None

    return JSONResponse(
        content=await run_summary_embedding(batch_no, summary_data, extract_json_str(summary_answer), company_name))


@search_router.get('/get_origin_result')
async def get_origin_res(batch_no: str):
    rows = await dbop.run(
        "SELECT * FROM keyword_summary_origin WHERE batch_no=%s AND status=%s",
        (batch_no, 'completed'))

    fields = ['analyze_matches', 'origin_matches', 'dense_matches', 'reranker_matches', 'keywords']
    data = []
    for row in rows:
        result = {}
        for k, v in row.items():
            if 'matches' in k and v:
                try:
                    v = json.loads(v)
                except json.JSONDecodeError:
                    pass
            if k in fields:
                result[k] = v
        data.append(result)

    return JSONResponse(content=data)
