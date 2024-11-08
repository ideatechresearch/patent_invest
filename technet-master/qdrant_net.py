from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, IsEmptyCondition, HasIdCondition, MatchValue, \
    PayloadField, SearchRequest, SearchParams, RecommendRequest, QuantizationSearchParams, VectorParams, \
    CollectionStatus, PointStruct, OrderBy, Batch, Distance, Range
from py2neo import Graph, Node, Relationship, Subgraph
import requests, json, httpx
import numpy as np
from enum import Enum as PyEnum


# from neo4j import Graph,GraphDatabase driver = GraphDatabase.driver(uri, auth=(username, password))
# import asyncio


def cosine_sim(A, B):
    dot_product = np.dot(A, B)
    similarity = dot_product / (np.linalg.norm(A) * np.linalg.norm(B))
    return similarity


# 计算次数：13
def calc_layers_total(layers=[3, 3, 3]):
    n = 0
    product = 1
    for layer in layers:
        n += product
        product *= layer
    return n
    # import math
    # for i in range(len(layers)):
    #     n += math.prod(layers[:i], start=1)


# 最大节点数：40
def calc_nodes_total(layers=[3, 3, 3]):
    n = 1
    product = 1
    for layer in layers:
        product *= layer
        n += product
    return n


def scale_to_range(numbers, new_min, new_max):
    min_value = min(numbers)
    max_value = max(numbers)

    # 避免除以零的情况
    if min_value == max_value:
        return [new_min + (new_max - new_min) / 2] * len(numbers)

    scaled = (new_max - new_min) / (max_value - min_value)
    scaled_numbers = [
        new_min + (num - min_value) * scaled for num in numbers
    ]

    return scaled_numbers


def download_file(url, local_filename):
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(local_filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
    return local_filename


def recover_snapshots_upload(host, collection_name, file_path):
    url = f"http://{host}:6333/collections/{collection_name}/snapshots/upload"
    with open(file_path, 'rb') as f:
        files = {'snapshot': f}
        # headers={"Content-Type": "multipart/form-data"}
        response = requests.post(url, files=files)

    return response.json()


def qdrant_livez(host, https=False, api_key=''):
    url = f"https://{host}:6333/livez" if https else f"http://{host}:6333/livez"
    headers = {"api-key": api_key} if api_key else {}
    try:
        response = requests.get(url, headers=headers, timeout=3)
        return response.text, response.status_code
    except requests.exceptions.RequestException as e:
        return e, 400
    except Exception as e:
        return e, 500


def del_collections_snapshots(client):
    for collection in client.get_collections().collections:
        snapshots_names = client.list_snapshots(collection.name)
        print(collection.name, snapshots_names)
        if snapshots_names:
            snapshots_name = snapshots_names[0].name
            client.delete_snapshot(collection_name=collection.name, snapshot_name=snapshots_name)


def create_vdb_collection(collection_name, size, new=False):
    if new and client.collection_exists(collection_name=collection_name):
        client.delete_collection(collection_name=collection_name)

    if not client.collection_exists(collection_name=collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=size, distance=Distance.COSINE),  # { "size": 128, "distance": "Cosine" }
            # optimizer_config=OptimizersConfigDiff(indexing_threshold=0),#上传期间禁用索引,避免不必要的向量索引，这些向量将被下一批覆盖
        )
    return client.count(collection_name=collection_name, exact=True)  # get_collection(collection_name)


def get_vdb_point(host, collection_name, ids):
    response = requests.get(f'http://{host}:6333/collections/{collection_name}/points/{ids}')  # 从单个点检索所有详细信息
    return response.json()


def get_bge_embeddings(texts=[], access_token=''):
    # global baidu_access_token
    # if not access_token:
    #     access_token = get_baidu_access_token()
    if not isinstance(texts, list):
        texts = [texts]

    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/embeddings/bge_large_zh?access_token=" + access_token
    headers = {
        'Content-Type': 'application/json'
    }
    batch_size = 16
    embeddings = []
    if len(texts) < batch_size:
        payload = json.dumps({
            "input": texts
        })
        response = requests.request("POST", url, headers=headers, data=payload)
        data = response.json().get('data')
        if len(texts) == 1:
            return data[0].get('embedding')

        embeddings = [d.get('embedding') for d in data]
    else:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            payload = json.dumps({
                "input": batch
            })
            # print(f"批次 {i // batch_size + 1}")
            response = requests.request("POST", url, headers=headers, data=payload)
            data = response.json().get('data')
            if data and len(data) == len(batch):
                embeddings += [d.get('embedding') for d in data]

    return embeddings


def most_similar_embeddings(query, collection_name, client, topn=10, score_threshold=0.0,
                            match=[], not_match=[], access_token=''):
    try:
        query_vector = get_bge_embeddings(query, access_token)
        if not query_vector:
            return []

        query_filter = Filter(must=match, must_not=not_match)
        search_hit = client.search(collection_name=collection_name,
                                   query_vector=query_vector,  # tolist()
                                   query_filter=query_filter,
                                   limit=topn,
                                   score_threshold=score_threshold,
                                   )
        return [(p.payload, p.score) for p in search_hit]
    except Exception as e:
        print('Error:', e)
        return []


def web_search(text: str, api_key: str) -> list:
    msg = [{"role": "user", "content": text}]
    url = "https://open.bigmodel.cn/api/paas/v4/tools"
    data = {
        "request_id": str(np.random.randint(1, 1e9)),
        "tool": "web-search-pro",
        "stream": False,
        "messages": msg
    }

    headers = {'Authorization': api_key}
    try:
        response = requests.post(url, json=data, headers=headers, timeout=60)
        response.raise_for_status()

        data = response.json()
        search_result = data.get('choices', [{}])[0].get('message', {}).get('tool_calls', [{}])[1].get('search_result')
        if search_result:
            return [{'title': result.get('title'),
                     'content': result.get('content'),
                     'link': result.get('link'),
                     'media': result.get('media')
                     } for result in search_result]
        return [{'content': response.content.decode()}]
    except (requests.exceptions.RequestException, KeyError, IndexError) as e:
        return [{'error': str(e)}]


# 模型编码:0默认，1小，-1最大
AI_Models = [
    # https://platform.moonshot.cn/console/api-keys
    {'name': 'moonshot', 'type': 'default', 'api_key': '',
     "model": ["moonshot-v1-32k", "moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"],
     'url': "https://api.moonshot.cn/v1/chat/completions", 'base_url': "https://api.moonshot.cn/v1"},

    # https://open.bigmodel.cn/console/overview
    {'name': 'glm', 'type': 'default', 'api_key': '',
     "model": ["glm-4-air", "glm-4-flash", "glm-4-air", "glm-4", "glm-4v", "glm-4-0520"],
     'url': 'https://open.bigmodel.cn/api/paas/v4/chat/completions',
     'base_url': "https://open.bigmodel.cn/api/paas/v4/"},

    # https://dashscope.console.aliyun.com/overview
    {'name': 'qwen', 'type': 'default', 'api_key': '',
     "model": ["qwen-turbo", "qwen1.5-7b-chat", "qwen1.5-32b-chat", "qwen2-7b-instruct", "qwen2.5-32b-instruct",
               'qwen-long', "qwen-turbo", "qwen-plus", "qwen-max"],  # "qwen-vl-plus"
     'embedding': ["text-embedding-v2", "text-embedding-v1", "text-embedding-v2", "text-embedding-v3"],
     'speech': ['paraformer-v1', 'paraformer-8k-v1', 'paraformer-mtl-v1'],
     'url': 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
     'base_url': "https://dashscope.aliyuncs.com/compatible-mode/v1"},

    # https://cloud.siliconflow.cn/playground/chat
    {'name': 'silicon', 'type': 'default', 'api_key': '',
     'model': ["Qwen/Qwen2-7B-Instruct", "Qwen/Qwen1.5-7B-Chat", "Qwen/Qwen1.5-32B-Chat", "01-ai/Yi-1.5-9B-Chat-16K",
               "THUDM/chatglm3-6b", "THUDM/glm-4-9b-chat", "Pro/THUDM/glm-4-9b-chat",
               "deepseek-ai/DeepSeek-V2-Chat", "deepseek-ai/DeepSeek-V2.5", "deepseek-ai/deepseek-llm-67b-chat",
               "internlm/internlm2_5-7b-chat", "Pro/internlm/internlm2_5-7b-chat", "Pro/OpenGVLab/InternVL2-8B",
               "google/gemma-2-9b-it", "meta-llama/Meta-Llama-3-8B-Instruct"],
     'embedding': ['BAAI/bge-large-zh-v1.5', 'BAAI/bge-m3', 'netease-youdao/bce-embedding-base_v1'],
     'completion': ['Qwen/Qwen2.5-Coder-7B-Instruct', "deepseek-ai/DeepSeek-V2.5",
                    'deepseek-ai/DeepSeek-Coder-V2-Instruct'],
     'reranker': ['BAAI/bge-reranker-v2-m3', 'netease-youdao/bce-reranker-base_v1'],
     'url': 'https://api.siliconflow.cn/v1/chat/completions',
     'base_url': 'https://api.siliconflow.cn/v1',
     'embeddings_url': "https://api.siliconflow.cn/v1/embeddings",
     'rerank_url': "https://api.siliconflow.cn/v1/rerank"},

    # https://platform.baichuan-ai.com/docs/api
    {'name': 'baichuan', 'type': 'default', 'api_key': '',
     "model": ['Baichuan3-Turbo', "Baichuan2-Turbo", 'Baichuan3-Turbo', 'Baichuan3-Turbo-128k', "Baichuan4",
               "Baichuan-NPC-Turbo"],
     "embedding": ["Baichuan-Text-Embedding"],
     'url': 'https://api.baichuan-ai.com/v1/chat/completions',
     'base_url': "https://api.baichuan-ai.com/v1/",  # assistants,files,threads
     'embeddings_url': 'https://api.baichuan-ai.com/v1/embeddings'},
]


class ModelNameEnum(str, PyEnum):
    moonshot = "moonshot"
    glm = "glm"
    qwen = "qwen"
    ernie = "ernie"
    hunyuan = "hunyuan"
    doubao = "doubao"
    silicon = "silicon"


AI_Client = {}


def find_ai_model(name: str, model_i: int = 0):
    model = next((model for model in AI_Models if model['name'] == name), None)
    if model:
        model_i = model_i if abs(model_i) < len(model['model']) else 0  # 默认选择第一个模型
        return model, model['model'][model_i]
    return None, None


def ai_chat(messages, model_id, temperature=0.4, top_p=0.8, max_tokens=1024, payload=None,
            client=None, model_info={}):
    if not payload:
        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }

    if client:
        completion = client.chat.completions.create(**payload)
        return completion.choices[0].message.content

    # 通过 requests 库直接发起 HTTP POST 请求
    url = model_info['url']
    api_key = model_info['api_key']
    headers = {
        'Content-Type': 'application/json',
        "Authorization": f'Bearer {api_key}'
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # 如果请求失败，则抛出异常
        data = response.json().get('choices')
        return data[0].get('message').get('content')
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None


def ai_chat_async(messages, model_id, temperature=0.4, top_p=0.8, max_tokens=1024,
                  payload=None, client=None, model_info={}):
    if not payload:
        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": True,
        }

    if client:
        stream = client.chat.completions.create(**payload)
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:  # 以两个换行符 \n\n 结束当前传输的数据块
                yield delta.content  # completion.append(delta.content)
            # if chunk.choices[0].finish_reason == 'stop':
            #     break
        # yield '[DONE]'
        return

    api_key = model_info['api_key']
    url = model_info['url']
    headers = {
        'Content-Type': 'text/event-stream',
        "Authorization": f'Bearer {api_key}'
    }

    try:
        # async with httpx.AsyncClient() as cx:
        #     async with cx.stream("POST", url, headers=headers, json=payload) as response:
        with httpx.Client() as cx:  # sse HTTP 响应 data=json.dumps(payload)
            response = cx.post(url, headers=headers, json=payload)
            response.raise_for_status()
            yield from process_line_stream(response)
    except httpx.RequestError as e:
        yield str(e)

    # yield "[DONE]"


def moonshot_chat(messages, temperature=0.4, top_p=0.8, payload=None, client=None, api_key=''):
    if not payload:
        payload = {
            "model": "moonshot-v1-32k",  # moonshot-v1-8k,32moonshot-v1-128k
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
        }

    if client:
        completion = client.chat.completions.create(**payload)
        return completion.choices[0].message.content

    url = "https://api.moonshot.cn/v1/chat/completions"
    headers = {
        'Content-Type': 'application/json',
        "Authorization": f'Bearer {api_key}'
    }

    try:
        response = requests.post(url, headers=headers, json=payload)  # data=json.dumps(payload)
        response.raise_for_status()  # 如果请求失败，则抛出异常
        data = response.json().get('choices')
        return data[0].get('message').get('content')
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None


def moonshot_chat_async(messages, temperature=0.4, payload=None, client=None, api_key=''):
    if not payload:
        payload = {
            "model": "moonshot-v1-32k",  # moonshot-v1-8k,moonshot-v1-128k
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }

    if client:
        stream = client.chat.completions.create(**payload)
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:  # 以两个换行符 \n\n 结束当前传输的数据块
                yield delta.content  # completion.append(delta.content)
            # if chunk.choices[0].finish_reason == 'stop':
            #     break
        # yield '[DONE]'
        return

    url = "https://api.moonshot.cn/v1/chat/completions"
    headers = {
        'Content-Type': 'text/event-stream',
        "Authorization": f'Bearer {api_key}'
    }

    try:
        response = httpx.post(url, headers=headers, json=payload)  # sse HTTP 响应 data=json.dumps(payload)
        response.raise_for_status()
        yield from process_line_stream(response)
    except httpx.RequestError as e:
        yield str(e)

    # yield "[DONE]"


def process_data_chunk(data):
    try:
        chunk = json.loads(data)
        delta = chunk.get('choices')[0]["delta"]
        content = delta.get("content")
        if content:
            return content
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        return str(e)


def process_line_stream(response):
    data = ""
    for line in response.iter_lines():
        line = line.strip()
        if len(line) == 0:  # 开头的行 not line
            if data:
                yield process_data_chunk(data)  # 一个数据块的结束 yield from + '\n'
                data = ""
            continue
        if line.startswith("data: "):
            line_data = line.lstrip("data: ")
            if line_data == "[DONE]":
                # if data:  # 在结束时处理可能的残留数据
                #     yield process_data_chunk(data)
                # print(data)
                break
            content = process_data_chunk(line_data)
            if content:
                yield content
        else:
            data += "\n" + line

    if data:
        yield process_data_chunk(data)


def most_similar_by_name(name, collection_name, client, match=[], exclude=[], topn=10, score_threshold=0.5):
    match_name = FieldCondition(key='word', match=MatchValue(value=name, ), )
    not_match_name = [FieldCondition(key='word', match=MatchValue(value=w), ) for w in exclude]
    scroll_filter = Filter(must=match + [match_name])
    scroll_result = client.scroll(collection_name=collection_name,
                                  scroll_filter=scroll_filter,
                                  with_vectors=True,
                                  with_payload=True,
                                  limit=1,  # 只获取一个匹配的结果
                                  # order_by=OrderBy(key='df',direction='desc')
                                  )

    query_vector = scroll_result[0][0].vector

    query_filter = Filter(must=match, must_not=[match_name] + not_match_name)  # 缩小查询范围
    search_hit = client.search(collection_name=collection_name,
                               query_vector=query_vector,  # tolist()
                               query_filter=query_filter,
                               limit=topn,
                               score_threshold=score_threshold,
                               search_params=SearchParams(exact=True,  # Turns on the exact search mode
                                                          quantization=QuantizationSearchParams(rescore=True),
                                                          ),  # 使用原始向量对 top-k 结果重新评分
                               # offset=1
                               )

    return [(p.payload['word'], p.score) for p in search_hit]


# SimilarByIds
def most_similar_by_ids(ids, collection_name, client, key_name='word', match=[], exclude=[], topn=10,
                        score_threshold=0.0, exact=True):
    id_record = client.retrieve(collection_name=collection_name, ids=ids, with_vectors=True)

    not_match_name = [FieldCondition(key=key_name, match=MatchValue(value=w), ) for w in exclude]
    not_match_ids = [HasIdCondition(has_id=ids)]
    query_filter = Filter(must=match, must_not=not_match_ids + not_match_name)  # 缩小查询范围

    search_queries = [
        SearchRequest(vector=p.vector, filter=query_filter, limit=topn, score_threshold=score_threshold,
                      with_payload=[key_name], params=SearchParams(exact=exact), )
        for p in id_record]

    search_hit = client.search_batch(collection_name=collection_name, requests=search_queries)  # ScoredPoint

    return [(id, [(p.payload[key_name], p.score) for p in hit]) for id, hit in
            zip(ids, search_hit)]  # [:topn]


# 选择合适的索引类型和参数,性能优先: HNSW 索引,资源优先:IVF_FLAT
# 流式插入和批量导入
# 谨慎使用标量过滤，删除特性
# k8s 部署
def names_to_ids(names, collection_name, client, match=[], key_name='word'):
    shoulds = [FieldCondition(key=key_name, match=MatchValue(value=w)) for w in names]
    scroll_filter = Filter(must=match, should=shoulds)
    scroll_result = client.scroll(collection_name=collection_name,
                                  scroll_filter=scroll_filter,
                                  with_payload=True,
                                  # with_vectors=True,
                                  limit=len(names))

    return {i.payload[key_name]: i.id for i in scroll_result[0]}


def ids_to_names(ids, collection_name, client, key_name='word'):
    id_record = client.retrieve(collection_name=collection_name, ids=ids, with_payload=[key_name])
    return {p.id: p.payload[key_name] for p in id_record}


def ids_match(ids):
    return [HasIdCondition(has_id=ids)]


def empty_match(field_key):
    if not field_key:
        return []
    return [IsEmptyCondition(is_empty=PayloadField(key=field_key), )]


def field_match(field_key, match_values):
    if not field_key or match_values == 'all':
        return []
    if isinstance(match_values, (str, int, float)):
        return [FieldCondition(key=field_key, match=MatchValue(value=match_values, ), )]
    return [FieldCondition(key=field_key, match=MatchValue(value=v), ) for v in match_values]


def rerank_similar_by_search(names, search_hit, topn=10, duplicate=0, key_name='word'):
    similar_next = {}  # 数据格式:{co:[(next,val)*topn]}
    if duplicate == 0:
        similar_next = {name: [(p.payload[key_name], p.score) for p in hit[:topn]] for name, hit in
                        zip(names, search_hit)}
        # return [(name,[(p.payload['word'],p.score) for p in hit[:topn]]) for name,hit in zip(names,search_hit)]

    if duplicate == 1:
        for name, hit in zip(names, search_hit):
            y = [(p.payload[key_name], p.score) for p in hit]
            y2 = [_[0] for x in similar_next.values() for _ in x]
            similar_next[name] = [y3 for y3 in y if y3[0] not in y2][:topn]

    if duplicate == 2:  # 各自排序后,由下一节点各自选最近的上节点,最后由上节点各自匹配重复关系
        similar_with = {}
        for name, hit in zip(names, search_hit):
            y = [(p.payload[key_name], p.score) for p in hit]
            for z in y:
                if z[0] not in similar_with or z[1] > similar_with[z[0]][1]:  # 对关系去重选最大的,反向选择
                    similar_with[z[0]] = (name, z[1])
        for y, x in similar_with.items():
            similar_next[x[0]] = sorted(similar_next.get(x[0], []) + [(y, x[1])],
                                        key=lambda z: z[1], reverse=True)[:topn]

    if duplicate == 3:
        similar_sort = sorted(
            [(name, p.payload[key_name], p.score) for name, hit in zip(names, search_hit) for p in hit],
            key=lambda x: x[2], reverse=True)
        pos = []
        for ix, iy, v in similar_sort:
            y = [p[1] for p in pos]
            x = [p[0] for p in pos if p[0] == ix]
            if len(x) < topn and iy not in y:
                pos.append((ix, iy))
                similar_next[ix] = similar_next.get(ix, []) + [(iy, v)]

    # print(similar_next)
    return [(w, similar_next[w]) for w in names if w in similar_next]


def rerank_similar_by_recommend(ids, recommend_hit, topn=10, duplicate=0):
    similar_next = {}  # 数据格式:{_id:[(next_id,score)*topn]}
    if duplicate == 0:
        similar_next = {i: [(p.id, p.score) for p in hit[:topn]] for i, hit in zip(ids, recommend_hit)}
        # return [(_id, [(p.id, p.score) for p in hit]) for _id, hit in zip(ids, search_hit)]

    if duplicate == 1:
        for i, hit in zip(ids, recommend_hit):
            y = [(p.id, p.score) for p in hit]
            y2 = [_[0] for x in similar_next.values() for _ in x]
            similar_next[i] = [y3 for y3 in y if y3[0] not in y2][:topn]

    if duplicate == 2:  # 各自排序后,由下一节点各自选最近的上节点,最后由上节点各自匹配重复关系
        similar_with = {}
        for i, hit in zip(ids, recommend_hit):
            y = [(p.id, p.score) for p in hit]
            for z in y:
                if z[0] not in similar_with or z[1] > similar_with[z[0]][1]:  # 对关系去重选最大的,反向选择
                    similar_with[z[0]] = (i, z[1])
        for y, x in similar_with.items():
            similar_next[x[0]] = sorted(similar_next.get(x[0], []) + [(y, x[1])],
                                        key=lambda z: z[1], reverse=True)[:topn]

    if duplicate == 3:
        similar_sort = sorted(
            [(i, p.id, p.score) for i, hit in zip(ids, recommend_hit) for p in hit],
            key=lambda x: x[2], reverse=True)
        pos = []
        for ix, iy, v in similar_sort:
            y = [p[1] for p in pos]
            x = [p[0] for p in pos if p[0] == ix]
            if len(x) < topn and iy not in y:
                pos.append((ix, iy))
                similar_next[ix] = similar_next.get(ix, []) + [(iy, v)]

    # print(similar_next)
    return [(i, similar_next[i]) for i in ids if i in similar_next]


def SearchByIds(ids, collection_name, client, key_name='word', match=[], exclude=[], topn=10, duplicate=0,
                score_threshold=0, exact=True):
    id_record = client.retrieve(collection_name=collection_name, ids=ids, with_vectors=True)

    names = [p.payload[key_name] for p in id_record]
    not_match_name = [FieldCondition(key=key_name, match=MatchValue(value=w), ) for w in names + exclude]
    query_filter = Filter(must=match, must_not=not_match_name)  # 缩小查询范围

    search_queries = [
        SearchRequest(vector=p.vector, filter=query_filter, limit=topn * len(names), score_threshold=score_threshold,
                      with_payload=[key_name], params=SearchParams(exact=exact))
        for p in id_record]

    search_hit = client.search_batch(collection_name=collection_name, requests=search_queries)  # ScoredPoint

    return rerank_similar_by_search(names, search_hit, topn=topn, duplicate=duplicate, key_name=key_name)


def recommend_by_ids(ids, collection_name, client, key_name='word', match=[], exclude=[], topn=10, score_threshold=0):
    not_match_name = [FieldCondition(key=key_name, match=MatchValue(value=w), ) for w in exclude]
    query_filter = Filter(must=match, must_not=not_match_name)  # 缩小查询范围

    recommend_queries = [
        RecommendRequest(positive=[_id], filter=query_filter, limit=topn, score_threshold=score_threshold,
                         with_payload=[key_name]) for _id in ids]

    search_hit = client.recommend_batch(collection_name=collection_name, requests=recommend_queries)  # ScoredPoint

    return [(_id, [(p.payload[key_name], p.score) for p in hit]) for _id, hit in zip(ids, search_hit)]


def RecommendByIds(ids, collection_name, client, match=[], not_ids=[], topn=10, duplicate=0, score_threshold=0):
    not_match_ids = [HasIdCondition(has_id=not_ids)]
    query_filter = Filter(must=match, must_not=not_match_ids)  # 缩小查询范围

    recommend_queries = [
        RecommendRequest(positive=[_id], filter=query_filter, limit=topn * len(ids), score_threshold=score_threshold,
                         with_payload=False) for _id in ids]

    search_hit = client.recommend_batch(collection_name=collection_name, requests=recommend_queries)  # ScoredPoint

    return rerank_similar_by_recommend(ids, search_hit, topn=topn, duplicate=duplicate)


class VDBSimilar:
    client = None
    collection_name = ''
    key_name = 'word'
    match_first = []

    def __init__(self, client, collection_name, key_name='word', match_first=[]):
        self.client = client
        self.collection_name = collection_name
        self.key_name = key_name  # .lower()
        self.match_first = match_first

        self.name_ids = {}  # unique,映射记忆

    def get_ids(self, names):
        finds = set(names) - self.name_ids.keys()
        if len(finds):  # 更新未记录id
            result = names_to_ids(finds, self.collection_name, self.client, self.match_first, self.key_name)
            if result:
                self.name_ids.update(result)

        return [self.name_ids[n] for n in names if n in self.name_ids]

    def get_id(self, name):
        if name not in self.name_ids:
            result = names_to_ids([name], self.collection_name, self.client, self.match_first, self.key_name)
            if result:
                self.name_ids.update(result)

        return self.name_ids.get(name, -1)

    def get_names(self, ids):
        finds = set(ids) - set(self.name_ids.values())
        if finds:
            result = ids_to_names(list(finds), self.collection_name, self.client, self.key_name)
            if result:
                self.name_ids.update({n: i for i, n in result.items()})
        id_to_name = {i: n for n, i in self.name_ids.items()}
        return [id_to_name[i] for i in ids if i in id_to_name]

    def get_name(self, _id):
        if _id not in self.name_ids.values():
            result = ids_to_names([_id], self.collection_name, self.client, self.key_name)
            if result:
                self.name_ids.update({n: i for i, n in result.items()})

        for n, i in self.name_ids.items():
            if i == _id:
                return n
        return None

    def get_menory(self):
        return self.name_ids

    def get_keyname(self):
        return self.key_name

    def get_vecs(self, names, ids=[]):
        if not ids:
            ids = self.get_ids(names)
        id_record = self.client.retrieve(collection_name=self.collection_name, ids=ids, with_vectors=True)
        return {p.payload[self.key_name]: p.vector for p in id_record}

    def get_vec(self, name, _id=-1):  # id or name
        get_id = _id if _id > 0 else self.name_ids.get(name, -1)
        if get_id > 0:
            point_result = self.client.retrieve(collection_name=self.collection_name, ids=[get_id], with_vectors=True)
        else:
            match_name = FieldCondition(key=self.key_name, match=MatchValue(value=name, ), )
            scroll_filter = Filter(must=self.match_first + [match_name])
            scroll_result = self.client.scroll(collection_name=self.collection_name,
                                               scroll_filter=scroll_filter,
                                               with_vectors=True,
                                               with_payload=[self.key_name],
                                               limit=1,  # 只获取一个匹配的结果
                                               # order_by=OrderBy(key='df',direction='desc')
                                               )

            point_result = scroll_result[0]  # (points,next_page_offset)
            if not point_result:
                return []

            self.name_ids[point_result[0].payload[self.key_name]] = point_result[0].id

        return point_result[0].vector

    def get_payload(self, names, ids=[], fields=[]):
        if not ids:
            ids = self.get_ids(names)
        id_record = self.client.retrieve(collection_name=self.collection_name, ids=ids,
                                         with_payload=fields if len(fields) else True)
        return [(p.id, p.payload) for p in id_record]  # [(id,{payload}),]

    def RecommendBatch(self, ids, exclude=[], not_ids=[], topn=10, duplicate=0, score_threshold=0.0):
        not_match_name = [FieldCondition(key=self.key_name, match=MatchValue(value=w), ) for w in exclude]
        not_match_ids = [HasIdCondition(has_id=not_ids)]
        query_filter = Filter(must=self.match_first, must_not=not_match_ids + not_match_name)  # 缩小查询范围

        recommend_queries = [
            RecommendRequest(positive=[i], filter=query_filter, limit=topn * len(ids), with_payload=[self.key_name],
                             score_threshold=score_threshold)
            for i in ids]

        recommend_hit = self.client.recommend_batch(collection_name=self.collection_name,
                                                    requests=recommend_queries)  # ScoredPoint

        self.name_ids.update({p.payload[self.key_name]: p.id for hit in recommend_hit for p in hit})

        return rerank_similar_by_recommend(ids, recommend_hit, topn=topn, duplicate=duplicate)

    def SearchBatch(self, names, ids=[], exclude=[], not_ids=[], topn=10, duplicate=0, score_threshold=0.0):
        if not ids:
            ids = self.get_ids(names)
        id_record = self.client.retrieve(collection_name=self.collection_name, ids=ids, with_vectors=True)

        names = [p.payload[self.key_name] for p in id_record]
        not_match_name = [FieldCondition(key=self.key_name, match=MatchValue(value=w), ) for w in names + exclude]
        not_match_ids = [HasIdCondition(has_id=ids + not_ids)]
        query_filter = Filter(must=self.match_first, must_not=not_match_ids + not_match_name)  # 缩小查询范围

        search_queries = [
            SearchRequest(vector=p.vector, filter=query_filter, limit=topn * len(names), with_payload=[self.key_name],
                          score_threshold=score_threshold)
            for p in id_record]

        search_hit = self.client.search_batch(collection_name=self.collection_name,
                                              requests=search_queries)  # ScoredPoint

        self.name_ids.update({p.payload[self.key_name]: p.id for hit in search_hit for p in hit})

        return rerank_similar_by_search(names, search_hit, topn=topn, duplicate=duplicate, key_name=self.key_name)

    def Recommend(self, ids, exclude=[], not_ids=[], topn=10, score_threshold=0.0, use_name=False):
        not_match_name = [FieldCondition(key=self.key_name, match=MatchValue(value=w), ) for w in exclude]
        not_match_ids = [HasIdCondition(has_id=not_ids)]
        query_filter = Filter(must=self.match_first, must_not=not_match_ids + not_match_name)
        hit = self.client.recommend(collection_name=self.collection_name,
                                    positive=ids,  # [ID]
                                    query_filter=query_filter,
                                    limit=topn,
                                    score_threshold=score_threshold,
                                    with_payload=[self.key_name],
                                    )  # ScoredPoint

        self.name_ids.update({p.payload[self.key_name]: p.id for p in hit})

        return [(p.payload[self.key_name] if use_name else p.id, p.score) for p in hit]

    def Search(self, name, _id=-1, exclude=[], not_ids=[], topn=10, score_threshold=0.0):
        query_vector = self.get_vec(name, _id)
        if not query_vector:
            return []

        not_match_name = [FieldCondition(key=self.key_name, match=MatchValue(value=w), ) for w in exclude]
        not_match_ids = [HasIdCondition(has_id=not_ids)]
        query_filter = Filter(must=self.match_first, must_not=not_match_ids + not_match_name)  # [match_name] + 缩小查询范围
        hit = self.client.search(collection_name=self.collection_name,
                                 query_vector=query_vector,  # neighbor_vector then result_vector tolist()
                                 query_filter=query_filter,
                                 with_payload=True,
                                 limit=topn,  # max_neighbors
                                 score_threshold=score_threshold,
                                 offset=(0 if name in exclude else 1),
                                 # search_params=SearchParams(exact=True,  # Turns on the exact search mode
                                 #                            # quantization=QuantizationSearchParams(rescore=True),
                                 #                            ),
                                 )  # ScoredPoint

        self.name_ids.update({p.payload[self.key_name]: p.id for p in hit})

        return [(p.payload[self.key_name], p.score) for p in hit]


def CreateNode(word, depth, graph, node_name='Word', **kwargs):
    nodes = graph.nodes.match(node_name)
    node = nodes.where(name=word).first()  # Node("Word", name=word,id=mid)
    if not node:
        node = Node(node_name, name=word, depth=depth, nid=len(nodes), **kwargs)
        graph.create(node)
    return node


# 查询图数据的函数
# {"query": "MATCH (n)-[r]->(m) RETURN n, r, m"}
# fetch('/query', {
#     method: 'POST',
#     headers: {
#         'Content-Type': 'application/json'
#     },
#     body: JSON.stringify({query: 'MATCH (n)-[r]->(m) RETURN n, r, m'})
# })
def query_graph(tx, query):
    result = tx.run(query)
    nodes = []
    relationships = []

    for record in result:
        for key, value in record.items():
            if isinstance(value, dict) and 'labels' in value:  # 检查是否为节点
                nodes.append({
                    "id": value.id,
                    "labels": value.labels,
                    "properties": dict(value.items())
                })
            elif isinstance(value, dict) and 'type' in value:  # 检查是否为关系
                relationships.append({
                    "id": value.id,
                    "type": value.type,
                    "start": value.start,
                    "end": value.end,
                    "properties": dict(value.items())
                })

    return {"nodes": nodes, "relationships": relationships}


class VDBRelationships:
    graph = None
    vdb = {}  # functions.get(key)(,)

    def load(self, graph, client, collection_name='专利_w2v', prefix='Word', field_key='行业', match_values=['all']):
        self.graph = graph
        self.append(client, collection_name=collection_name, prefix=prefix, field_key=field_key,
                    match_values=match_values)

    def append(self, client, collection_name='专利_w2v', prefix='Word', field_key='行业', match_values=['all']):
        if not field_key:
            match_values = ['all']
        for k in match_values:
            match = [] if k == 'all' else [FieldCondition(key=field_key, match=MatchValue(value=k, ))]
            vdb_key = f"{prefix}_{k}"
            self.vdb[vdb_key] = VDBSimilar(client, collection_name=collection_name, key_name=prefix.lower(),
                                           match_first=match)
            # count = client.count(collection_name=collection_name,
            #                      count_filter=Filter(must=match), exact=True)
            # print(vdb_key, ':', count)

    def switch_clients(self, client):
        for v in self.vdb.values():
            if isinstance(v, VDBSimilar):
                v.client = client  # 切换备用线路

    def get_payload(self, vdb_key, names=[], ids=[], fields=[]):
        if vdb_key not in self.vdb:
            return []
        return self.vdb[vdb_key].get_payload(names, ids, fields)  # [(id,{payload}),]

    def get_names(self, vdb_key, ids):
        if vdb_key not in self.vdb:
            return None
        _instance = self.vdb[vdb_key]
        id_to_name = {i: n for n, i in _instance.name_ids.items()}
        result = {}
        for i in ids:
            if i in id_to_name:
                result[i] = id_to_name[i]
            else:
                name = _instance.get_name(i)
                if name:
                    result[i] = name

        return result

    def get_ids(self, vdb_key, names):
        if vdb_key not in self.vdb:
            return None
        name_ids = self.vdb[vdb_key].get_menory()
        return {n: name_ids.get(n, self.vdb[vdb_key].get_id(n)) for n in names}

    def get_id(self, vdb_key, name):
        if vdb_key not in self.vdb:  # self.vdb.get(vdb_key, None)
            return None
        return self.vdb[vdb_key].get_id(name)

    def similar(self, _id, name='', not_ids=[], vdb_key='Word_all', topn=10, exclude=[], score_threshold=0.0):
        if vdb_key not in self.vdb:  # _id > 0,name
            return []
        try:
            if _id > 0:
                return self.vdb[vdb_key].Recommend([_id], exclude=exclude, not_ids=not_ids, topn=topn,
                                                   score_threshold=score_threshold, use_name=True)

            return self.vdb[vdb_key].Search(name=name, _id=_id, exclude=exclude, not_ids=not_ids, topn=topn,
                                            score_threshold=score_threshold)

        except Exception as e:
            print(f"Error occurred in similar function: {e}")
            return []

    def similar_by_names(self, tokens, vdb_key='Word_all', topn=10, duplicate=0, exclude=[], score_threshold=0.0):
        tokens = [token.strip().lower() for token in tokens]
        if vdb_key not in self.vdb:
            return tokens, []
        # tokens = [token for token in tokens if token in self.data.index]  # data.index.isin(tokens)
        if len(tokens) == 0:
            return [], []
        return tokens, self.vdb[vdb_key].SearchBatch(tokens, ids=[], exclude=exclude, not_ids=[],
                                                     topn=topn, duplicate=duplicate, score_threshold=score_threshold)

    def match_relationships(self, name, prefix='Word'):
        if self.graph:
            node_name = f"{prefix}_{name}"
            return self.graph.run(
                "MATCH (source:" + node_name + " {name:\"" + name + "\"})-[relation:SIMILAR_TO]->(target) RETURN source,relation,target").data()
        return []

    def create_relationships(self, width=3, depth=0, similar_names=[], names_depth={}, relationships_edges=[],
                             vdb_key='Word_all', duplicate=3, create=0, score_threshold=0.0, exclude=[]):

        name_first = [k for k, v in names_depth.items() if v == 0]  # list(words_depth)[0]
        try:
            not_ids = self.vdb[vdb_key].get_ids(list(names_depth))
            similar_next = self.vdb[vdb_key].SearchBatch(similar_names,
                                                         ids=[],
                                                         exclude=exclude,  # +list(names_depth)
                                                         not_ids=not_ids,
                                                         topn=width,
                                                         duplicate=duplicate,
                                                         score_threshold=score_threshold)  # 分支会有合并,下层关系去重

            similar_names = list(set(y[0] for x in similar_next for y in x[1]))  # 迭代词组
            print("Depth:", depth, "Similar:", [x[0] for x in similar_next], "->", similar_next)
            names_depth.update({w: depth + 1 for w in similar_names if w not in names_depth})

            if self.graph and create:
                prefix = vdb_key.split('_')[0]
                node_name = f"{prefix}_{name_first}"

                for x in similar_next:
                    word_node = CreateNode(x[0], names_depth[x[0]], self.graph, node_name)  # w=x[0]
                    relationships = [Relationship(word_node, "SIMILAR_TO",
                                                  CreateNode(y[0], names_depth[y[0]], self.graph, node_name),
                                                  rank=i, similarity=float(y[1]))
                                     for i, y in enumerate(x[1])]

                    self.graph.create(Subgraph(relationships=relationships))  # 创建关系
                    relationships_edges += relationships
            else:
                relationships = [
                    {'source': x[0], 'target': y[0], 'relation': str(round(y[1], 5)), 'value': 1.0 - y[1], 'rank': i}
                    for x in similar_next for i, y in enumerate(x[1])]
                relationships_edges += relationships

        except KeyError:
            print(f"Data '{name_first, similar_names}' not in vocabulary.")
            similar_names = []

        return similar_names

    def Relations(self, width=3, depth=0, ids=[], ids_depth={}, relationships_edges=[],
                  vdb_key='Word_all', duplicate=3, score_threshold=0.0, exclude=[]):
        try:
            similar_next = self.vdb[vdb_key].RecommendBatch(ids=ids,
                                                            exclude=exclude,  # +list(names_depth)
                                                            not_ids=list(ids_depth),
                                                            topn=width,
                                                            duplicate=duplicate,
                                                            score_threshold=score_threshold)  # 分支会有合并,下层关系去重

            ids = list(set(y[0] for x in similar_next for y in x[1]))
            # print("Depth:", depth, "Similar:", [x[0] for x in similar_next], "->", similar_next)
            ids_depth.update({i: depth + 1 for i in ids if i not in ids_depth})
            relationships = [
                {'source': x[0], 'target': y[0], 'relation': str(round(y[1], 5)), 'value': 1.0 - y[1], 'rank': i}
                for x in similar_next for i, y in enumerate(x[1])]
            relationships_edges += relationships

        except KeyError:
            id_first = [k for k, v in ids_depth.items() if v == 0]  # list(ids_depth)[0]
            print(f"Data '{id_first, ids}' not in vocabulary.")
            ids = []

        return ids

    def create_relationship(self, width, depth, name, names_depth={}, vdb_key='Word_all', exclude_all=True, create=0,
                            score_threshold=0.0, exclude=[]):
        name_first = [k for k, v in names_depth.items() if v == 0]
        relationships = []
        try:
            exclude_names = list(names_depth) if exclude_all else [k for k in names_depth if
                                                                   names_depth[k] <= depth]  # 深度遍历时,排除上几层出现的
            not_ids = self.vdb[vdb_key].get_ids(exclude_names)
            similar_next = self.vdb[vdb_key].Search(name, _id=-1, exclude=exclude,
                                                    not_ids=not_ids,
                                                    topn=width,
                                                    score_threshold=score_threshold)
            similar_names = [y[0] for y in similar_next]
            if len(similar_names) == 0:
                return names_depth, relationships  # 返回递归

            print("Depth:", depth, "Similar:", name, "->", similar_next)
            new_depth = {w: depth + 1 for w in similar_names if w not in names_depth}
            names_depth.update(new_depth)  # 当前层去重

            if self.graph and create:
                prefix = vdb_key.split('_')[0]
                node_name = f"{prefix}_{name_first}"
                word_node = CreateNode(name, names_depth[name], self.graph, node_name)  # 创建节点
                relationships = [Relationship(word_node, "SIMILAR_TO",
                                              CreateNode(y[0], names_depth[y[0]], self.graph, word_node),
                                              similarity=y[1], rank=i)
                                 for i, y in enumerate(similar_next)]
                self.graph.create(Subgraph(relationships=relationships))  # 创建关系
            else:
                relationships = [
                    {'source': name, 'target': y[0], 'relation': str(round(y[1], 5)), 'value': 1.0 - y[1], 'rank': i}
                    for i, y in enumerate(similar_next)]

        except KeyError:
            print(f"Data '{name_first, name}' not in vocabulary.")
            # new_depth = {}

        return new_depth, relationships

    def Relation(self, width, depth, _id, ids_depth={}, vdb_key='Word_all', exclude_all=True, score_threshold=0.0,
                 exclude=[]):
        relationships = []
        try:
            not_ids = list(ids_depth) if exclude_all else [k for k in ids_depth if ids_depth[k] <= depth]
            similar_next = self.vdb[vdb_key].Recommend(ids=[_id], exclude=exclude, not_ids=not_ids,
                                                       topn=width, score_threshold=score_threshold, use_name=False)
            ids = [y[0] for y in similar_next]  # p.payload[self.key_name] if use_name else p.id
            if len(ids) == 0:
                return ids_depth, relationships  # 返回递归

            # print("Depth:", depth, "Similar:", _id, "->", similar_next)
            new_depth = {i: depth + 1 for i in ids if i not in ids_depth}
            ids_depth.update(new_depth)  # 当前层去重

            relationships = [
                {'source': _id, 'target': y[0], 'relation': str(round(y[1], 5)), 'value': 1.0 - y[1], 'rank': i}
                for i, y in enumerate(similar_next)]

        except KeyError:
            id_first = [k for k, v in ids_depth.items() if v == 0]
            print(f"Data '{id_first, _id}' not in vocabulary.")
            # new_depth = {}

        return new_depth, relationships

    def create_relationships_breadth(self, name, vdb_key='Word_all', layers=[3, 3, 3], duplicate=3,
                                     max_calc=30, max_node=0, create=0, score_threshold=0.0, exclude=[]):
        if vdb_key not in self.vdb:
            return {}, []

        names_depth = {name: 0}
        relationships_edges = []

        similar_names = [name]
        depth = 0
        count = 0

        for width in layers:  # while depth <= max_depth and len(names_depth) <= max_node
            if max_node > 0 and len(names_depth) + len(similar_names) * width > max_node:
                cut_len = (max_node - len(names_depth)) // width
                if cut_len <= 0:
                    break
                similar_names = similar_names[:cut_len]

            if max_calc > 0 and count + len(similar_names) > max_calc:  # calc_layers_total
                cut_len = (max_calc - count) // width
                if cut_len <= 0:
                    break
                similar_names = similar_names[:cut_len]

            count += len(similar_names)

            similar_names = self.create_relationships(width=width, depth=depth,
                                                      similar_names=similar_names,
                                                      names_depth=names_depth,
                                                      relationships_edges=relationships_edges,
                                                      vdb_key=vdb_key, duplicate=duplicate, create=create,
                                                      score_threshold=score_threshold, exclude=exclude)

            depth += 1

            if len(similar_names) == 0:
                break

        return names_depth, relationships_edges

    def RelationsBreadth(self, _id, vdb_key='Word_all', layers=[3, 3, 3], duplicate=3, max_calc=30, max_node=0,
                         score_threshold=0.0, exclude=[]):
        ids_depth = {_id: 0}
        relationships_edges = []

        ids = [_id]
        depth = 0
        count = 0

        for width in layers:
            if max_node > 0 and len(ids_depth) + len(ids) * width > max_node:
                cut_len = (max_node - len(ids_depth)) // width
                if cut_len <= 0:
                    break
                ids = ids[:cut_len]

            if max_calc > 0 and count + len(ids) > max_calc:  # calc_layers_total
                cut_len = (max_calc - count) // width
                if cut_len <= 0:
                    break
                ids = ids[:cut_len]

            count += len(ids)

            ids = self.Relations(width=width, depth=depth, ids=ids,
                                 ids_depth=ids_depth, relationships_edges=relationships_edges, vdb_key=vdb_key,
                                 duplicate=duplicate, score_threshold=score_threshold, exclude=exclude)

            depth += 1

            if len(ids) == 0:
                break

        return ids_depth, relationships_edges

    def create_relationships_depth(self, name, vdb_key='Word_all', layers=[3, 3, 3], max_calc=30, max_node=0,
                                   create=0, score_threshold=0.0, exclude_all=True, exclude=[],
                                   depth=0, names_depth={}, relationships_edges=[]):
        if vdb_key not in self.vdb:
            return {}, []

        if names_depth is None:
            names_depth = {}
        if relationships_edges is None:
            relationships_edges = []
        if depth == 0:
            relationships_edges.clear()
            names_depth.clear()
            names_depth[name] = 0

        if depth >= len(layers):
            return names_depth, relationships_edges

        if max_node > 0 and len(names_depth) + layers[depth] > max_node:
            return names_depth, relationships_edges

        if max_calc > 0 and len(names_depth) > 1:
            count = 1  # len(relationships_edges)
            for d, w in enumerate(layers):
                ndepth = [dn for dn in names_depth.values() if dn == d + 1]
                count += len(ndepth) // w
            if count >= max_calc:
                return names_depth, relationships_edges

        new_depth, relationships = self.create_relationship(width=layers[depth], depth=depth,
                                                            name=name, names_depth=names_depth,
                                                            vdb_key=vdb_key, exclude_all=exclude_all,
                                                            create=create, score_threshold=score_threshold,
                                                            exclude=exclude)

        relationships_edges += relationships

        # 递归创建更深层次的关系,只找新的
        for w in new_depth:
            self.create_relationships_depth(w, vdb_key, layers, max_calc, max_node,
                                            create, score_threshold, exclude_all, exclude,
                                            depth + 1, names_depth, relationships_edges)

        # return [name] + [
        #     self.create_relationships_depth(w, vdb_key, names_depth, relationships_edges,
        #                                     width, depth + 1, max_depth, max_node,
        #                                     create, score_threshold, exclude_all)
        #     if w in new_depth else [w] for w in similar_words]

        return names_depth, relationships_edges

    def RelationsDepth(self, _id, vdb_key='Word_all', layers=[3, 3, 3], max_calc=30, max_node=0,
                       score_threshold=0.0, exclude_all=True, exclude=[],
                       depth=0, ids_depth={}, relationships_edges=[]):
        if ids_depth is None:
            ids_depth = {}
        if relationships_edges is None:
            relationships_edges = []
        if depth == 0:
            relationships_edges.clear()
            ids_depth.clear()
            ids_depth[_id] = 0

        if depth >= len(layers):
            return ids_depth, relationships_edges

        if max_node > 0 and len(ids_depth) + layers[depth] > max_node:
            return ids_depth, relationships_edges

        if max_calc > 0 and len(ids_depth) > 1:
            count = 1  # len(relationships_edges)
            for d, w in enumerate(layers):
                ndepth = [dn for dn in ids_depth.values() if dn == d + 1]
                count += len(ndepth) // w
            if count >= max_calc:
                return ids_depth, relationships_edges

        new_depth, relationships = self.Relation(width=layers[depth], depth=depth, _id=_id, ids_depth=ids_depth,
                                                 vdb_key=vdb_key, exclude_all=exclude_all,
                                                 score_threshold=score_threshold, exclude=exclude)

        relationships_edges += relationships

        for i in new_depth:
            self.RelationsDepth(i, vdb_key, layers, max_calc, max_node, score_threshold, exclude_all, exclude,
                                depth + 1, ids_depth, relationships_edges)

        return ids_depth, relationships_edges

    # functions
    def SimilarRelationships(self, name, vdb_key='Word_all', width=3, max_depth=3, layers=[], batch=True,
                             duplicate=3, create=0, max_calc=30, max_node=0, score_threshold=0.0, exclude=[]):

        if not layers:
            layers = [width] * max_depth

        if batch:
            return self.create_relationships_breadth(name, vdb_key, layers, duplicate=duplicate,
                                                     max_calc=max_calc, max_node=max_node, create=create,
                                                     score_threshold=score_threshold, exclude=exclude)
        else:
            return self.create_relationships_depth(name, vdb_key, layers,
                                                   max_calc=max_calc, max_node=max_node, create=create,
                                                   score_threshold=score_threshold, exclude=exclude,
                                                   exclude_all=(duplicate > 0))

        # 老方法:self.SimulationNodes(names_depth, relationships_edges, params['vdb_key'], params.get('key_radius', ''))

    # 多层次关系探索:max_neighbors max_depth search_tree append(children)
    def SimilarRelations(self, name, vdb_key='Word_all', key_radius='', width=3, max_depth=3, layers=[], batch=True,
                         duplicate=3, draw=0, max_calc=30, max_node=0, score_threshold=0.0, exclude=[]):

        if vdb_key not in self.vdb:
            return {}, []

        _id = self.vdb[vdb_key].get_id(name)
        if _id is None:
            return {}, []

        if not layers:
            layers = [width] * max_depth

        if batch:
            ids_depth, relationships_edges = self.RelationsBreadth(_id, vdb_key, layers, duplicate=duplicate,
                                                                   max_calc=max_calc, max_node=max_node,
                                                                   score_threshold=score_threshold, exclude=exclude)
        else:
            ids_depth, relationships_edges = self.RelationsDepth(_id, vdb_key, layers,
                                                                 max_calc=max_calc, max_node=max_node,
                                                                 score_threshold=score_threshold,
                                                                 exclude_all=(duplicate > 0), exclude=exclude)

        if draw == 0:  # Simulation
            if key_radius:
                key_name = self.vdb[vdb_key].get_keyname()  # vdb_key.split('_')[0].lower() 'word' get_payload
                payloads = self.get_payload(vdb_key, names=[], ids=list(ids_depth), fields=[key_name, key_radius])
                values = [p.get(key_radius, 1) for i, p in payloads]  # [(id, {payload}), ]
                radius = scale_to_range(np.log(values), 10, 30)

                nodes = [{"id": i, "name": p[key_name], "depth": ids_depth[i], "radius": radius[idx]}
                         for idx, (i, p) in enumerate(payloads)]  # zip(*payloads)
            else:
                ids_name = self.get_names(vdb_key, list(ids_depth))
                nodes = [{"id": i, "name": n, "depth": ids_depth[i], "radius": 20} for i, n in ids_name.items()]

            key_nodes = {n['id']: n for n in nodes}
            edges = relationships_edges.copy()
            for rel in edges:
                rel['source'] = key_nodes.get(rel['source'], rel['source'])
                rel['target'] = key_nodes.get(rel['target'], rel['target'])

            return {"nodes": nodes, "edges": edges}

        if draw == 1:  # Cytoscape
            key_name = self.vdb[vdb_key].get_keyname()
            payloads = self.get_payload(vdb_key, names=[], ids=list(ids_depth))
            nodes = [{"id": i, 'label': p[key_name], 'properties': {**p, 'depth': ids_depth[i]}}
                     for i, p in payloads]

            edges = relationships_edges.copy()
            for i, rel in enumerate(edges):
                rel['id'] = str(i)
                rel['label'] = rel.pop('relation', 'none')
                rel['rank'] = rel.get('rank', 0)
                # edges[i] = {"data": rel}

            return {"nodes": nodes, "edges": edges}

        if draw == 2:
            # NetworkxPlot(names_depth, relationships_edges)
            pass

        return ids_depth, relationships_edges

    def SimilarRelation(self, node_id, node_name, existing_nodes, vdb_key='Word_all', width=3, duplicate=3,
                        score_threshold=0.0, exclude=[]):
        if vdb_key not in self.vdb:
            return {}

        if node_id > 0:
            ids_depth = {n['id']: n['depth'] for n in existing_nodes}
            new_depth, new_relationships = self.Relation(width=width, depth=ids_depth.get(node_id, 1),
                                                         _id=node_id, ids_depth=ids_depth,
                                                         vdb_key=vdb_key, exclude_all=(duplicate > 0),
                                                         score_threshold=score_threshold, exclude=exclude)

            ids_name = self.get_names(vdb_key, list({**ids_depth, **new_depth}))
            nodes = [{"id": i, "name": ids_name.get(i, 'None'), "depth": d, "radius": 20} for i, d in new_depth.items()]
        else:  # 老方法
            # self.vdb[vdb_key].get_id(node_name)
            names_depth = {n['name']: n['depth'] for n in existing_nodes}
            new_depth, new_relationships = self.create_relationship(width=width, depth=names_depth.get(node_name, 1),
                                                                    name=node_name, names_depth=names_depth,
                                                                    vdb_key=vdb_key, exclude_all=(duplicate > 0),
                                                                    create=0,
                                                                    score_threshold=score_threshold, exclude=exclude)

            nodes = [{"id": vdr.get_id(vdb_key, n), "name": n, 'depth': d, "radius": 20} for n, d in new_depth.items()]
            for r in new_relationships:
                r['source'] = self.get_id(vdb_key, r['source'])  # get_ids
                r['target'] = self.get_id(vdb_key, r['target'])

        return {"nodes": nodes, "edges": new_relationships}

    def SimulationNodes(self, names_depth, relationships_edges, vdb_key='Word_all', key_radius=''):
        if vdb_key not in self.vdb:
            nodes = [{"name": w, "id": i + 1, 'depth': names_depth[w]} for i, w in enumerate(names_depth)]
        else:
            if key_radius:
                key_name = self.vdb[vdb_key].get_keyname()  # vdb_key.split('_')[0].lower() 'word'
                payloads = self.get_payload(vdb_key, list(names_depth), ids=[], fields=[key_name, key_radius])
                values = [p.get(key_radius, 1) for i, p in payloads]
                radius = scale_to_range(np.log(values), 10, 30)
                nodes = [{"id": i, "name": p[key_name], "depth": names_depth[p[key_name]], "radius": radius[idx]}
                         for idx, (i, p) in enumerate(payloads)]  # zip(*payloads)
            else:
                name_ids = self.get_ids(vdb_key, list(names_depth))
                nodes = [{"id": i, "name": n, "depth": names_depth[n], "radius": 20} for n, i in name_ids.items()]

        key_nodes = {n['name']: n for n in nodes}  # {p['word']: i for i, p in payloads}
        edges = relationships_edges.copy()
        for r in edges:
            r['source'] = key_nodes.get(r['source'])
            r['target'] = key_nodes.get(r['target'])

        return {"nodes": nodes, "edges": edges}


# def NetworkxPlot(names_depth, relationships_edges):
#     import networkx as nx
#     import plotly.graph_objects as go
#     G = nx.Graph()
#     for rel in relationships_edges:
#         G.add_edge(rel['source'], rel['target'], label=rel['relation'], value=rel['value'], rank=rel['rank'])
#     pos = nx.spring_layout(G)
#     edge_trace = go.Scatter(
#         x=[],
#         y=[],
#         line=dict(width=0.5, color='#888'),
#         hoverinfo='none',
#         mode='lines'
#     )
#     for rel in G.edges(data=True):
#         x0, y0 = pos[rel[0]]
#         x1, y1 = pos[rel[1]]
#         edge_trace['x'] += [x0, x1, None]
#         edge_trace['y'] += [y0, y1, None]
#
#     node_trace = go.Scatter(
#         x=[],
#         y=[],
#         text=[],
#         mode='markers+text',
#         hoverinfo='text',
#         marker=dict(
#             showscale=True,
#             colorscale='YlGnBu',
#             size=10,
#             color=[],
#             colorbar=dict(
#                 thickness=15,
#                 title='Node Connections',
#                 xanchor='left',
#                 titleside='right'
#             ),
#             line_width=2
#         )
#     )
#
#     for node in G.nodes():
#         x, y = pos[node]
#         node_trace['x'] += [x]
#         node_trace['y'] += [y]
#
#     node_adjacencies = []
#     node_text = []
#     for node, adjacencies in enumerate(G.adjacency()):
#         node_adjacencies.append(len(adjacencies[1]))
#         node_text.append(f'# of connections: {len(adjacencies[1])}')
#
#     node_trace.marker.color = node_adjacencies
#     node_trace.text = node_text
#
#     # 创建图形对象
#     fig = go.Figure(data=[edge_trace, node_trace],
#                     layout=go.Layout(
#                         title='Network Graph',
#                         titlefont_size=16,
#                         showlegend=False,
#                         hovermode='closest',
#                         margin=dict(b=20, l=5, r=5, t=40),
#                         annotations=[dict(
#                             text="Python code by <a href='https://plotly.com/'>Plotly</a>",
#                             showarrow=False,
#                             xref="paper", yref="paper",
#                             x=0.005, y=-0.002
#                         )],
#                         xaxis=dict(showgrid=False, zeroline=False),
#                         yaxis=dict(showgrid=False, zeroline=False))
#                     )
#
#     graph_div = fig.to_html(full_html=False)


if __name__ == '__main__':
    client = QdrantClient(host="10.10.10.5", grpc_port=6334, prefer_grpc=True)
    match_first = field_match("行业", '金融')
    vdbs = VDBSimilar(client, collection_name='专利_w2v', key_name='word', match_first=match_first)
    res = vdbs.SearchBatch(['计算机', '训练方法', '电子设备'], ids=[], exclude=[], topn=3, duplicate=3,
                           score_threshold=0.3)
    print(res)
    res = vdbs.Search('领域', _id=-1, exclude=[], topn=10, score_threshold=0.5)
    print(res)

    vdr = VDBRelationships()
    vdr.load(None, client, collection_name='专利_w2v', prefix='Word',
             field_key='行业', match_values=['金融', '传统制造'])
    words_depth, relationships_edges = vdr.create_relationships_breadth('计算机', 'Word_金融', [2, 3, 4],
                                                                        duplicate=3,
                                                                        create=0, max_node=400,
                                                                        score_threshold=0.0)
    print(words_depth, '\n', relationships_edges)

    words_depth, relationships_edges = vdr.create_relationships_depth('计算机', 'Word_金融',
                                                                      [2, 3, 3],
                                                                      max_node=500, create=0, score_threshold=0.0,
                                                                      exclude_all=False)

    print(words_depth, '\n', relationships_edges, '\n')

    words_depth, relationships_edges = vdr.SimilarRelationships('计算机', 'Word_金融', width=3, max_depth=3, layers=[],
                                                                batch=False)
    print(words_depth, '\n', relationships_edges)

    name = '电子装置'
    next = vdbs.Search(name, _id=-1, exclude=words_depth.keys(), not_ids=[], topn=10, score_threshold=0.0)
    print(next)

    new_depth, relationships = vdr.create_relationship(1, words_depth.get(name, 1), name, names_depth=words_depth,
                                                       vdb_key='Word_金融', exclude_all=True, create=0,
                                                       score_threshold=0.0, exclude=[])
    print(new_depth, relationships)

    # collection_name='专利_w2v_188_37'
    # client.create_snapshot(collection_name=collection_name)
    # snapshots_name=client.list_snapshots(collection_name)[0].name
    # url=f'http://10.10.10.5:6333/collections/{collection_name}/snapshots/{snapshots_name}'
    # local_filename=f'E:\Downloads\{snapshots_name}'
    # download_file(url, local_filename)
    # recover_snapshots_upload('47.110.156.41', collection_name, local_filename)
    # client.delete_snapshot(
    # collection_name=collection_name, snapshot_name=snapshots_name
    # )
    # client.recover_snapshot
    client = QdrantClient(host="47.110.156.41", grpc_port=6334, prefer_grpc=True)
    from lda_topics import LdaTopics

    lt = LdaTopics(suffix='xjzz', len_below=2, top_n_topics=4, minimum_probability=0.03, weight_threshold_topics=0.03)
    vec = lt.topic_vec('一种电风扇摇头装置，其特征在于由两个同轴安 装的上传动盘和下传动盘组成，上传动盘的下表面设')

    SP = client.search(collection_name='专利_先进制造_w2v_lda_120', query_vector=vec, limit=5)
    print(SP)
    api_key = "sk-7"
    messages = [{"role": "user", "content": '你好我的朋友'}, ]
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url="https://api.moonshot.cn/v1")
    except Exception as e:
        print(f"An error occurred while initializing the OpenAI client: {e}")
        client = None

    for content in moonshot_chat_async(messages, temperature=0.4, payload=None, client=None, api_key=api_key):
        print(content)
