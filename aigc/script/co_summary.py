from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from co_utils import *

app = FastAPI(lifespan=lifespan)


@app.get("/start_task/")
async def start_task(company_name: str):
    batch_no, _ = await run_enterprise_analysis_task_background(company_name, app.state.httpx_client)
    return {"batch_no": batch_no}


@app.get("/task_status/{batch_no}")
async def get_task_status(batch_no: str):
    sql_list = [
        ("SELECT * FROM task_batch WHERE batch_no=%s", (batch_no,)),
        ("SELECT * FROM task_question WHERE batch_no=%s", (batch_no,)),
        ("SELECT * FROM task_question_content WHERE batch_no=%s", (batch_no,)),
        ("SELECT * FROM task_summary_question WHERE batch_no=%s", (batch_no,))
    ]
    batch_info, questions, question_content, summary = await dbop.query(sql_list, fetch_all=True)
    return {"batch": batch_info, "questions": questions, "question_content": question_content, "summary": summary}


@app.get('/search_points')
async def search_embeddings_points(batch_no: str,
                                   querys: Union[str, List[str]] = Query(..., description="查询语句，一个或多个"),
                                   reranker: Optional[Literal['KeyWord', 'BM25', 'Hybrid', 'LLM']] = None,
                                   topn: int = 30, score_threshold: float = 0.0,
                                   payload_key: Optional[str] = None):
    # {'text': chunk, 'chunk_id': i, 'batch_no': batch_no, 'desc': origin_question}
    match = field_match(field_key='batch_no', match_values=batch_no)
    tokens = [querys.strip()] if isinstance(querys, str) else [x.replace("\n", " ").strip() for x in querys]

    search_hit = await search_by_embeddings(tokens, Collection_Name, client=qd_client, emb_client=emb_client,
                                            model=MODEL_EMBEDDING, payload_key=payload_key, match=match, not_match=[],
                                            topn=topn, score_threshold=score_threshold, exact=False, get_dict=True)
    if not reranker:
        return {'search_hit': search_hit}

    if reranker == 'KeyWord':
        question_content = await dbop.run(
            "SELECT id,origin_question,content FROM task_question_content WHERE batch_no=%s", (batch_no,))
        corpus = [p.get("content", '') for p in question_content]
        results = []
        bm25 = BM25(corpus)
        for token in tokens:
            scores = bm25.rank_documents(token, topn)
            matches = [(question_content[match[0]], round(match[1], 3)) for match in scores]
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

    system_prompt = f'''你是一个信息重排助手，任务是根据用户的查询意图（querys）对搜索结果（search_hit）进行语义相关性重排，并返回多个最相关的文本内容。

    ## 输入说明
    你将收到以下两个部分：
    - querys：一个查询词或问题的列表，表示用户希望了解的核心信息。
    - search_hit：一个由多条文本组成的列表，每条文本来自原始数据源，表示候选答案。
    
    ## 处理要求
    1. **请逐条判断每个文本与 querys 的语义相关性，而不仅仅依赖关键词重合。**
    2. **请根据相关性从高到低进行排序，返回前若干条最相关的结果。**
    3. **请保留多个结果（而不是只返回一个），保持每条为独立的完整文本。**
    4. 输出一个 JSON 数组，每个元素是一个字符串，对应表示 text 字段原文内容。
    
    ## 输出格式
    请严格按如下格式输出：
    ```json
    [
      "2024年11月19日，陕西省高级人民法院已就***申请陕西鱼化置业有限公司预重整一案作出裁定……",
      "2022年5月8日，陕西省西安市中级人民法院受理……",
      ...
    ]
    '''
    #  3. 如果某些文本明显无关、重复度过高，或者仅是模板性描述，可直接排除。

    desc = (f'请根据以上规则处理，并返回一个 JSON 数组（保留{topn}个相关结果并重排）。数据如下：'
            f'\nquerys: {tokens}\nsearch_hit')
    result = await ai_analyze(system_prompt, search_hit, client=ai_client, desc=desc,
                              model='deepseek-chat', max_tokens=8192, temperature=0.2, top_p=0.85)

    parsed = extract_json_array(result)
    similar_list = [s.strip() for s in parsed if isinstance(s, str) and s.strip()] if isinstance(parsed, list) else []
    return JSONResponse(content={'similar_list': similar_list, 'search_hit': search_hit})


if __name__ == "__main__":
    import uvicorn

    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except KeyboardInterrupt:
        print("Server stopped by user")
