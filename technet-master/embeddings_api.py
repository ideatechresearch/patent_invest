import requests
import json


def get_baidu_access_token(API_Key='rlSIpsk5IChIGLLRRlCXxqCN', Secret_Key='3pqla0QLkuCLZ82opbGZ0uG49GbpSfOl'):
    """
    使用 API Key，Secret Key 获取access_token，替换下列示例中的应用API Key、应用Secret Key
    """
    url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={API_Key}&client_secret={Secret_Key}"

    payload = json.dumps("")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json().get("access_token")


def get_bge_embeddings(texts=[], access_token=''):
    # global baidu_access_token
    if not access_token:
        access_token = get_baidu_access_token()
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


def get_embeddings_2(texts=[], access_token='Bearer f62e903205bb597c98a853e80f4ff66f.obvD49XnyTdC1HIN'):
    if not isinstance(texts, list):
        texts = [texts]

    url = 'https://open.bigmodel.cn/api/paas/v4/embeddings'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': access_token
    }
    payload = {"model": "embedding-2",
               "input": texts
               }

    if isinstance(texts, str):
        response = requests.post(url, headers=headers, json=payload)  # data=json.dumps(
        return response.json().get('data')[0].get('embedding')

    # if isinstance(texts, list)
    batch_size = 16
    embeddings = []
    if len(texts) < batch_size:
        response = requests.post(url, headers=headers, json=payload)
        data = response.json().get('data')
        embeddings = [d.get('embedding') for d in data]
    else:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            payload["input"] = batch
            # print(f"批次 {i // batch_size + 1}")
            response = requests.post(url, headers=headers, json=payload)
            data = response.json().get('data')
            if data and len(data) == len(batch):
                embeddings += [d.get('embedding') for d in data]

    return embeddings


# "https://api-inference.huggingface.co/models/bert-base-uncased"
# "https://api-inference.huggingface.co/models/bert-base-chinese"
# "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
# "https://api-inference.huggingface.co/models/BAAI/bge-reranker-large"
def query(inputs, url, access_token="Bearer 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"):
    headers = {"Authorization": access_token}  # f"Bearer {API_TOKEN}"
    payload = {"inputs": inputs}
    response = requests.post(url, headers=headers, json=payload)
    return response.json()
    #'error': 'Model BAAI/bge-large-zh-v1.5 is currently loading','estimated_time': 52.08882141113281}


def get_embeddings_1_5(text, access_token="Bearer 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"):
    url = "https://api-inference.huggingface.co/models/BAAI/bge-large-zh-v1.5"
    headers = {"Authorization": access_token}
    payload = {"inputs": text}
    response = requests.post(url, headers=headers, json=payload)
    return response.json()

