import base64
import hmac, ecdsa, hashlib
from jose import JWTError, jwt
import requests, json
from urllib.parse import quote_plus, urlencode, urlparse, quote, unquote
from datetime import datetime, timedelta
from wsgiref.handlers import format_date_time
import time
import uuid


class Config(object):
    SQLALCHEMY_DATABASE_URI = f'mysql+pymysql://technet:{quote_plus("**")}@****.aliyuncs.com:3306/technet?charset=utf8'
    # SQLALCHEMY_DATABASE_URI = (f'mysql+pymysql://***?charset=utf8')
    SQLALCHEMY_COMMIT_ON_TEARDOWN = False
    SQLALCHEMY_TACK_MODIFICATIONS = True
    SQLALCHEMY_ECHO = True
    SECRET_KEY = '***'
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 60
    HTTP_TIMEOUT_SEC = 60
    MAX_TASKS = 1024
    MAX_CACHE = 1024
    DEVICE_ID = '***'
    INFURA_PROJECT_ID = ''
    DATA_FOLDER = 'data'
    QDRANT_HOST = 'qdrant'
    LOCAL_URL = 'http://47.***:7000'
    WECHAT_URL = 'http://*robot:*'
    QDRANT_URL = "http://47.***:6333"

    BAIDU_API_Key = '***'
    BAIDU_Secret_Key = '***'
    # https://console.bce.baidu.com/qianfan/ais/console/applicationConsole/application/v1
    BAIDU_qianfan_API_Key = '***'  # 45844683
    BAIDU_qianfan_Secret_Key = '***'
    # https://console.bce.baidu.com/ai/#/ai/nlp/overview/index
    BAIDU_nlp_API_Key = '***'  # 11266517
    BAIDU_nlp_Secret_Key = '***'
    # https://console.bce.baidu.com/ai/#/ai/unit/overview/index,https://aip.baidubce.com/rpc/2.0/unit/v3/
    BAIDU_unit_API_Key = '***'  # 115610933,
    BAIDU_unit_Secret_Key = '***'
    # https://console.bce.baidu.com/ai/#/ai/imagesearch/overview/index
    BAIDU_image_API_Key = '***'
    BAIDU_image_Secret_Key = '***'
    # https://console.bce.baidu.com/ai/#/ai/ocr/app/list
    BAIDU_ocr_API_Key = '***'  # 115708755
    BAIDU_ocr_Secret_Key = '***'
    # https://console.bce.baidu.com/ai/#/ai/speech/overview/index
    BAIDU_speech_API_Key = '***'  # '115520761'
    BAIDU_speech_Secret_Key = '***'
    # https://fanyi-api.baidu.com/api/trans/product/desktop
    BAIDU_trans_AppId = '***'
    BAIDU_trans_Secret_Key = '***'
    # https://lbsyun.baidu.com/apiconsole/center
    BMAP_API_Key = '***'
    # https://console.amap.com/dev/key/app
    AMAP_API_Key = '***'

    DashScope_Service_Key = '***' 
    DashVectore_Service_Key = '***'
    ALIYUN_AK_ID = '***'
    ALIYUN_Secret_Key = '***'
    ALIYUN_nls_AppId = '***'

    ALIYUN_oss_AK_ID = '***'
    ALIYUN_oss_Secret_Key = '***'
    ALIYUN_Bucket_Name = '***'  # 存储桶名称*
    ALIYUN_Bucket_Domain = "***"  # 加速域名**
    # https://console.cloud.tencent.com/hunyuan/api-key
    TENCENT_SecretId = '***'
    TENCENT_Secret_Key = '***'
    TENCENT_Service_Key = '***'

    # https://console.xfyun.cn/services
    XF_AppID = '***'
    XF_API_Key = '***'
    XF_Secret_Key = '***'  # XF_API_Key:XF_Secret_Key
    XF_API_Password = ['**', '', '']

    Silicon_Service_Key = '***'
    Moonshot_Service_Key = "***" 

    # https://open.bigmodel.cn/console/overview
    GLM_Service_Key = "***"
    # https://platform.baichuan-ai.com/console/apikey
    Baichuan_Service_Key = '***'
    HF_Service_Key = '***'
    # https://ai.youdao.com/console/#/
    YOUDAO_AppID = '***'
    YOUDAO_Service_Key = '***'
    CaiYun_Token = "***"
    HF_Service_Key = '***'

    VOLCE_AK_ID = '***' 
    VOLCE_Secret_Key = '***' 
    ARK_Service_Key = '***'

    Weather_Service_Key = '***'

    SOCKS_Proxies = {
        'http': 'socks5://your_socks_proxy_address:port',
        'https': 'socks5://u:p@proxy_address:port', }


# 模型编码:0默认，1小，-1最大
AI_Models = [
    # https://platform.moonshot.cn/console/api-keys
    {'name': 'moonshot', 'type': 'default', 'api_key': '',
     "model": ["moonshot-v1-32k", "moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"],
     'url': "https://api.moonshot.cn/v1/chat/completions", 'base_url': "https://api.moonshot.cn/v1",
     'file_url': "https://api.moonshot.cn/v1/files"},
    # https://open.bigmodel.cn/console/overview
    {'name': 'glm', 'type': 'default', 'api_key': '',
     "model": ["glm-4-air", "glm-4-flash", "glm-4-air", "glm-4", 'glm-4-plus', "glm-4v", "glm-4-0520"],
     "embedding": ["embedding-2", "embedding-3"],
     'url': 'https://open.bigmodel.cn/api/paas/v4/chat/completions',
     'base_url': "https://open.bigmodel.cn/api/paas/v4/",
     'embedding_url': 'https://open.bigmodel.cn/api/paas/v4/embeddings',
     'tool_url': "https://open.bigmodel.cn/api/paas/v4/tools"},
    # https://platform.baichuan-ai.com/docs/api
    {'name': 'baichuan', 'type': 'default', 'api_key': '',
     "model": ['Baichuan3-Turbo', "Baichuan2-Turbo", 'Baichuan3-Turbo', 'Baichuan3-Turbo-128k', "Baichuan4",
               "Baichuan-NPC-Turbo"],
     "embedding": ["Baichuan-Text-Embedding"],
     'url': 'https://api.baichuan-ai.com/v1/chat/completions',
     'base_url': "https://api.baichuan-ai.com/v1/",  # assistants,files,threads
     'embedding_url': 'https://api.baichuan-ai.com/v1/embeddings'},
    # https://dashscope.console.aliyun.com/overview
    # https://bailian.console.aliyun.com/#/home
    # https://pai.console.aliyun.com/?regionId=cn-shanghai&spm=5176.pai-console-inland.console-base_product-drawer-right.dlearn.337e642duQEFXN&workspaceId=567545#/quick-start/models
    {'name': 'qwen', 'type': 'default', 'api_key': '',
     "model": ["qwen-turbo", "qwen1.5-7b-chat", "qwen1.5-32b-chat", "qwen2-7b-instruct", "qwen2.5-32b-instruct",
               'qwen-long', "qwen-turbo", "qwen-plus", "qwen-max",
               'baichuan2-7b-chat-v1', 'baichuan2-turbo', 'abab6.5s-chat'],  # "qwen-vl-plus"
     'generation': ['dolly-12b-v2', 'baichuan2-7b-chat-v1', 'belle-llama-13b-2m-v1', 'billa-7b-sft-v1'],
     'embedding': ["text-embedding-v2", "text-embedding-v1", "text-embedding-v2", "text-embedding-v3"],
     'speech': ['paraformer-v1', 'paraformer-8k-v1', 'paraformer-mtl-v1'],
     'url': 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
     'base_url': "https://dashscope.aliyuncs.com/compatible-mode/v1",
     'generation_url': 'https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation',
     'embedding_url': 'https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding'},
    # https://cloud.baidu.com/doc/WENXINWORKSHOP/s/clntwmv7t#%E8%AF%B7%E6%B1%82%E8%AF%B4%E6%98%8E
    # https://cloud.baidu.com/doc/WENXINWORKSHOP/s/mlm0nonsv#%E5%AF%BC%E5%85%A5hf%E7%B3%BB%E5%88%97%E6%A8%A1%E5%9E%8B
    {'name': 'ernie', 'type': 'baidu', 'api_key': '',
     "model": ["ERNIE-4.0-8K", "ERNIE-3.5-8K", "ERNIE-4.0-8K", "ERNIE-4.0-8K-Preview", "ERNIE-4.0-8K-Latest",
               "ERNIE-3.5-128K"],
     'url': "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro",
     'base_url': "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/",
     },
    # https://console.bce.baidu.com/qianfan/ais/console/onlineService
    # https://console.bce.baidu.com/ai/#/ai/nlp/overview/index
    {'name': 'baidu', 'type': 'baidu', 'api_key': '',
     'model': ['llama_3_8b', 'Qianfan-Chinese-Llama-2-7B', 'Qianfan-Chinese-Llama-2-7B-32K',
               'Llama-2-7B-Chat', 'llama_3_8b', 'Llama-2-13B-Chat', 'Llama-2-70B-Chat',
               'ChatGLM3-6B', 'ChatGLM2-6B-32K', 'ChatGLM3-6B-32K',
               'Baichuan2-7B-Chat', 'Baichuan2-13B-Chat', 'Fuyu-8B', 'Yi-34B-Chat',
               'BLOOMZ-7B', 'Qianfan-BLOOMZ-7B-compressed'],
     "generation": ['sqlcoder_7b', 'CodeLlama-7b-Instruct', 'Yi-34B'],
     'embedding': ['bge_large_zh'],
     "nlp": ["txt_mone", "address", "simnet", "word_emb_sim", "ecnet", "text_correction", "keyword", "topic"],
     'url': "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/",
     'generation_url': "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/completions",
     'embedding_url': 'https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/embeddings',
     'nlp_url': "https://aip.baidubce.com/rpc/2.0/nlp/v1/",
     'base_url': ''},
    # https://console.volcengine.com/ark/region:ark+cn-beijing/endpoint?config=%7B%7D
    # https://console.volcengine.com/ark/region:ark+cn-beijing/model?vendor=Bytedance&view=LIST_VIEW
    {'name': 'doubao', 'type': 'default', 'api_key': '',
     "model": ['ep-20240919160005-chzhb', 'ep-20240919160119-7rbsn', 'ep-20240919160005-chzhb',
               'ep-20240919161410-7k5d8', 'ep-20241017105930-drfm8', 'ep-20241017110248-fr7z6'],
     # ["doubao-pro-32k", "doubao-lite-4k", "doubao-lite-32k", "doubao-pro-4k", "doubao-pro-32k", "doubao-pro-128k","GLM3-130B",chatglm3-130-fin],
     'url': 'https://ark.cn-beijing.volces.com/api/v3/chat/completions',
     'base_url': "https://ark.cn-beijing.volces.com/api/v3"},  # open.volcengineapi.com
    # https://cloud.tencent.com/document/product/1729
    {'name': 'hunyuan', 'type': 'tencent', 'api_key': '',
     'model': ["hunyuan-pro", "hunyuan-lite", "hunyuan-turbo", "hunyuan-code", 'hunyuan-functioncall'],
     'embedding': ['hunyuan-embedding'],
     'url': 'https://hunyuan.tencentcloudapi.com',  # 'hunyuan.ap-shanghai.tencentcloudapi.com'
     'base_url': "https://api.hunyuan.cloud.tencent.com/v1",
     'embedding_url': "https://api.hunyuan.cloud.tencent.com/v1/embeddings",
     'nlp_url': "nlp.tencentcloudapi.com", 'ocr_url': 'ocr.tencentcloudapi.com'},
    # https://cloud.siliconflow.cn/playground/chat
    {'name': 'silicon', 'type': 'default', 'api_key': '',
     'model': ["Qwen/Qwen2-7B-Instruct", "Qwen/Qwen1.5-7B-Chat", "Qwen/Qwen1.5-32B-Chat",
               "Qwen/Qwen2.5-Coder-7B-Instruct",
               "THUDM/chatglm3-6b", "THUDM/glm-4-9b-chat", "Pro/THUDM/glm-4-9b-chat",
               "deepseek-ai/DeepSeek-V2-Chat", "deepseek-ai/DeepSeek-V2.5", "deepseek-ai/DeepSeek-Coder-V2-Instruct",
               "internlm/internlm2_5-7b-chat", "Pro/internlm/internlm2_5-7b-chat", "Pro/OpenGVLab/InternVL2-8B",
               "01-ai/Yi-1.5-9B-Chat-16K", 'TeleAI/TeleChat2',
               "google/gemma-2-9b-it", "meta-llama/Meta-Llama-3-8B-Instruct"],
     'embedding': ['BAAI/bge-large-zh-v1.5', 'BAAI/bge-m3', 'netease-youdao/bce-embedding-base_v1'],
     'generation': ['Qwen/Qwen2.5-Coder-7B-Instruct', "deepseek-ai/DeepSeek-V2.5",
                    'deepseek-ai/DeepSeek-Coder-V2-Instruct'],
     'reranker': ['BAAI/bge-reranker-v2-m3', 'netease-youdao/bce-reranker-base_v1'],
     'url': 'https://api.siliconflow.cn/v1/chat/completions',
     'base_url': 'https://api.siliconflow.cn/v1',
     'embedding_url': "https://api.siliconflow.cn/v1/embeddings",
     'reranker_url': "https://api.siliconflow.cn/v1/rerank"},
    # https://console.xfyun.cn/services/sparkapiCenter
    {'name': 'spark', 'type': 'default', 'api_key': [],
     'model': ['pro', 'lite', 'max-32k', 'pro', 'pro-128k', '4.0Ultra', 'generalv3', 'generalv3.5'],
     'url': 'https://spark-api-open.xf-yun.com/v1/chat/completions',
     'base_url': 'https://spark-api-open.xf-yun.com/v1',
     'file_url': 'https://spark-api-open.xf-yun.com/v1/files',
     'embedding_url': 'https://emb-cn-huabei-1.xf-yun.com/',
     'translation_url': 'https://itrans.xf-yun.com/v1/its',
     'ws_url': 'wss://spark-api.xf-yun.com/v3.5/chat'
     },
    {'name': 'gpt', 'type': 'default', 'api_key': '', 'model': ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
     'generation': ["text-davinci-003", "text-davinci-002", "text-davinci-003", "text-davinci-004"],
     'embedding': ["text-embedding-ada-002", "text-search-ada-doc-001", "text-similarity-babbage-001",
                   "code-search-ada-code-001", "search-babbage-text-001"],
     'url': 'https://api.openai.com/v1/chat/completions',
     'embedding_url': 'https://api.openai.com/v1/embeddings',
     'base_url': "https://api.openai.com/v1",
     },
]
# moonshot,glm,qwen,ernie,hunyuan,doubao,silicon,spark,baichuan
API_KEYS = {
    'moonshot': Config.Moonshot_Service_Key,
    'glm': Config.GLM_Service_Key,
    'qwen': Config.DashScope_Service_Key,
    'doubao': Config.ARK_Service_Key,
    'silicon': Config.Silicon_Service_Key,
    'spark': Config.XF_API_Password,
    'baichuan': Config.Baichuan_Service_Key,
    'hunyuan': Config.TENCENT_Service_Key
}


# Api_Tokens = [
#     {"type": 'baidu', "func": get_baidu_access_token, "access_token": None, "expires_at": None, "expires_delta": 1440}]


# for tokens in Api_Tokens:
def scheduled_token_refresh(token_info):
    if token_info["expires_at"] is None or datetime.utcnow() > token_info["expires_at"] - timedelta(minutes=5):
        try:
            token_info["access_token"] = token_info["func"]()
            token_info["expires_at"] = datetime.utcnow() + timedelta(minutes=token_info["expires_delta"])
            # response = requests.post(f"{BASE_URL}/refresh", json={"refresh_token": tokens["refresh_token"]})
        except Exception as e:
            print(f"Error refreshing token for {token_info['type']}: {e}")


def md5_sign(content: str) -> str:
    return hashlib.md5(content.encode('utf-8')).hexdigest()


def hmac_sha256(key: bytes, content: str):
    """生成 HMAC-SHA256 签名"""
    return hmac.new(key, content.encode("utf-8"), hashlib.sha256).digest()  # hexdigest()


# sha256 hash
def hash_sha256(content: str):
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def generate_hash_key(*args, **kwargs):
    """
    根据任意输入参数生成唯一的缓存键。
    :param args: 任意位置参数（如模型名称、模型 ID 等）
    :param kwargs: 任意关键字参数（如其他描述性信息）
    :return: 哈希键
    """
    # 将位置参数和关键字参数统一拼接成一个字符串
    inputs = []
    for arg in args:
        if isinstance(arg, list):
            inputs.extend(map(str, arg))  # 如果是列表，逐个转换为字符串
        else:
            inputs.append(str(arg))

    for key, value in kwargs.items():
        inputs.append(f"{key}:{value}")  # 格式化关键字参数为 key:value

    joined_key = "|".join(inputs)  # [:1000]
    # 返回 MD5 哈希
    return hashlib.md5(joined_key.encode()).hexdigest()


# 获取百度的访问令牌
def get_baidu_access_token(secret_id=Config.BAIDU_qianfan_API_Key, secret_key=Config.BAIDU_qianfan_Secret_Key):
    # payload = json.dumps("")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    params = {
        "grant_type": "client_credentials",
        "client_id": secret_id,
        "client_secret": secret_key
    }
    url = "https://aip.baidubce.com/oauth/2.0/token"
    # url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={secret_id}&client_secret={secret_key}"
    response = requests.request("POST", url, params=params, headers=headers)  # data=payload
    response.raise_for_status()
    return response.json().get("access_token")


# 使用 HMAC 进行数据签名 https://ram.console.aliyun.com/users
# 阿里云服务交互时的身份验证: Base64( HMAC-SHA1(stringToSign, accessKeySecret + "&") );
def get_aliyun_access_token(service="nls-meta", region: str = "cn-shanghai",
                            access_key_id=Config.ALIYUN_AK_ID, access_key_secret=Config.ALIYUN_Secret_Key):
    parameters = {
        'AccessKeyId': access_key_id,
        'Action': 'CreateToken',
        'Format': 'JSON',
        'RegionId': region,
        'SignatureMethod': 'HMAC-SHA1',
        'SignatureNonce': str(uuid.uuid1()),
        'SignatureVersion': '1.0',
        'Timestamp': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        'Version': '2019-02-28'  # "2020-06-29"
    }

    def encode_text(text):
        return quote_plus(text).replace('+', '%20').replace('*', '%2A').replace('%7E', '~')

    def encode_dict(dic):
        dic_sorted = sorted(dic.items())
        return urlencode(dic_sorted).replace('+', '%20').replace('*', '%2A').replace('%7E', '~')

    query_string = encode_dict(parameters)
    string_to_sign = f"GET&{encode_text('/')}&{encode_text(query_string)}"

    secreted_string = hmac.new(bytes(f"{access_key_secret}&", 'utf-8'),
                               bytes(string_to_sign, 'utf-8'),
                               hashlib.sha1).digest()
    signature = base64.b64encode(secreted_string).decode()
    signature = encode_text(signature)

    full_url = f"http://{service}.{region}.aliyuncs.com/?Signature={signature}&{query_string}"
    response = requests.get(full_url)
    response.raise_for_status()

    if response.ok:
        token_info = response.json().get('Token', {})
        return token_info.get('Id'), token_info.get('ExpireTime')

    print(response.text)
    return None, None  # token, expire_time


def get_xfyun_authorization(api_key=Config.XF_API_Key, api_secret=Config.XF_Secret_Key,
                            host="spark-api.xf-yun.com", path="/v3.5/chat", method='GET'):
    # "itrans.xf-yun.com",/v1/its
    # Step 1: 生成当前日期
    cur_time = datetime.now()
    date = format_date_time(time.mktime(cur_time.timetuple()))  # RFC1123格式
    # datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%a, %d %b %Y %H:%M:%S GMT")
    # Step 2: 拼接鉴权字符串tmp
    signature_origin = f"host: {host}\ndate: {date}\n{method} {path} HTTP/1.1"

    # Step 3: 生成签名
    signature_sha = hmac.new(api_secret.encode('utf-8'), signature_origin.encode('utf-8'),
                             digestmod=hashlib.sha256).digest()
    signature = base64.b64encode(signature_sha).decode('utf-8')

    # Step 5: 生成 authorization_origin
    authorization_origin = (
        f"api_key=\"{api_key}\", algorithm=\"hmac-sha256\", "
        f"headers=\"host date request-line\", signature=\"{signature}\""
    )

    # Step 6: 对 authorization_origin 进行base64编码
    authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode('utf-8')

    # Step 7: 生成最终URL
    headers = {
        "authorization": authorization,  # 鉴权生成的authorization
        "date": date,  # 生成的date
        "host": host  # 请求的主机名
    }
    return headers
    # return f"https://{host}{path}?" + urlencode(headers)  #  https:// .wss:// requset_url + "?" +


def get_xfyun_signature(appid, api_secret, timestamp):
    # timestamp = int(time.time())
    try:
        # 对app_id和时间戳进行MD5加密
        auth = md5_sign(appid + str(timestamp))
        # 使用HMAC-SHA1算法对加密后的字符串进行加密 encrypt_key,encrypt_text
        return base64.b64encode(
            hmac.new(api_secret.encode('utf-8'), auth.encode('utf-8'), hashlib.sha1).digest()).decode('utf-8')
    except Exception as e:
        print(e)
        return None


# 火山引擎生成签名
def get_ark_signature(action: str, service: str, host: str, region: str = "cn-north-1", version: str = "2018-01-01",
                      access_key_id: str = Config.VOLCE_AK_ID, secret_access_key: str = Config.VOLCE_Secret_Key,
                      timenow=None):
    if not host:
        host = f"{service}.volcengineapi.com"  # 'open.volcengineapi.com'
    if not timenow:
        timenow = datetime.datetime.utcnow()
    date = timenow.strftime('%Y%m%dT%H%M%SZ')
    date_short = date[:8]

    # 构建Canonical Request
    http_method = "GET"
    canonical_uri = "/"
    canonical_querystring = f"Action={action}&Version={version}"
    canonical_headers = f"host:{host}\nx-date:{date}\n"
    signed_headers = "host;x-date"
    payload_hash = hashlib.sha256("".encode('utf-8')).hexdigest()  # 空请求体的哈希

    canonical_request = f"{http_method}\n{canonical_uri}\n{canonical_querystring}\n{canonical_headers}\n{signed_headers}\n{payload_hash}"

    # 构建String to Sign
    algorithm = "HMAC-SHA256"
    credential_scope = f"{date_short}/{region}/{service}/request"
    canonical_request_hash = hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()
    string_to_sign = f"{algorithm}\n{date}\n{credential_scope}\n{canonical_request_hash}"

    # 计算签名
    def get_signing_key(secret_key, date, region, service):
        k_date = hmac_sha256(f"VOLC{secret_key}".encode('utf-8'), date)
        k_region = hmac_sha256(k_date, region)
        k_service = hmac_sha256(k_region, service)
        k_signing = hmac_sha256(k_service, "request")
        return k_signing

    signing_key = get_signing_key(secret_access_key, date_short, region, service)
    signature = hmac.new(signing_key, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()

    #  构建Authorization头
    authorization_header = f"{algorithm} Credential={access_key_id}/{credential_scope}, SignedHeaders={signed_headers}, Signature={signature}"

    headers = {
        "Authorization": authorization_header,
        "Content-Type": "application/json",
        "Host": host,
        "X-Date": date
    }
    url = f"https://{host}/?{canonical_querystring}"

    return headers, url


def get_tencent_signature(service, host=None, params=None, action='ChatCompletions',
                          secret_id: str = Config.TENCENT_SecretId,
                          secret_key: str = Config.TENCENT_Secret_Key, timestamp: int = None, version='2023-09-01'):
    if not host:
        host = f"{service}.tencentcloudapi.com"
    if not timestamp:
        timestamp = int(time.time())
        # 支持 POST 和 GET 方式
    if not params:
        http_request_method = "GET"  # GET 请求签名
        params = {
            'Action': action,  # 'DescribeInstances'
            'InstanceIds.0': 'ins-09dx96dg',
            'Limit': 20,
            'Nonce': str(uuid.uuid1().int >> 64),  # 随机数,确保唯一性
            'Offset': 0,
            'Region': 'ap-shanghai',
            'SecretId': secret_id,
            'Timestamp': timestamp,
            'Version': version  # '2017-03-12'
        }
        # f"{k}={quote(str(v), safe='')}"
        query_string = '&'.join("%s=%s" % (k, str(v)) for k, v in sorted(params.items()))
        string_to_sign = f"{http_request_method}{host}/?{query_string}"
        signature = hmac.new(secret_key.encode("utf8"), string_to_sign.encode("utf8"), hashlib.sha1).digest()
        params["Signature"] = quote_plus(signature)  # 进行 URL 编码
        # quote_plus(signature.decode('utf8')) if isinstance(signature, bytes)  base64.b64encode(signature)
        return params

    algorithm = "TC3-HMAC-SHA256"
    date = datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d")

    # ************* 步骤 1：拼接规范请求串 *************
    http_request_method = "POST"
    canonical_uri = "/"
    canonical_querystring = ""
    ct = "application/json; charset=utf-8"  # 使用签名方法 v3（TC3-HMAC-SHA256）
    payload = json.dumps(params)
    canonical_headers = f"content-type:{ct}\nhost:{host}\nx-tc-action:{action.lower()}\n"
    signed_headers = "content-type;host;x-tc-action"
    hashed_request_payload = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    canonical_request = (http_request_method + "\n" +
                         canonical_uri + "\n" +
                         canonical_querystring + "\n" +
                         canonical_headers + "\n" +
                         signed_headers + "\n" +
                         hashed_request_payload)

    # ************* 步骤 2：拼接待签名字符串 *************
    credential_scope = f"{date}/{service}/tc3_request"
    hashed_canonical_request = hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()
    string_to_sign = (algorithm + "\n" +
                      str(timestamp) + "\n" +
                      credential_scope + "\n" +
                      hashed_canonical_request)

    # ************* 步骤 3：计算签名 *************
    secret_date = hmac_sha256(("TC3" + secret_key).encode("utf-8"), date)
    secret_service = hmac_sha256(secret_date, service)
    secret_signing = hmac_sha256(secret_service, "tc3_request")
    signature = hmac.new(secret_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

    # ************* 步骤 4：拼接 Authorization *************
    authorization = (algorithm + " " +
                     "Credential=" + secret_id + "/" + credential_scope + ", " +
                     "SignedHeaders=" + signed_headers + ", " +
                     "Signature=" + signature)

    # return authorization
    # 公共参数需要统一放到 HTTP Header 请求头部
    headers = {
        "Authorization": authorization,  # "<认证信息>"
        "Content-Type": ct,  # "application/json"
        "Host": host,  # "hunyuan.tencentcloudapi.com","tmt.tencentcloudapi.com"
        "X-TC-Action": action,  # "ChatCompletions","TextTranslate"
        # 这里还需要添加一些认证相关的Header
        "X-TC-Timestamp": str(timestamp),
        "X-TC-Version": version,  # "<API版本号>"
        "X-TC-Region": 'ap-shanghai'  # region,"<区域>",
    }
    return headers


def build_url(url: str, access_token: str = get_baidu_access_token(), **kwargs) -> str:
    url = url.strip().strip('"')
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url

    params = {"access_token": access_token}
    params.update(kwargs)
    query_string = urlencode(params)
    return f"{url}?{query_string}"


def is_url(url: str) -> bool:
    """更准确地判断是否为URL"""
    # url.startswith("http://") or url.startswith("https://")
    parsed = urlparse(url)
    # return all([parsed.scheme, parsed.netloc])
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


# 生成API请求签名
def generate_hmac_signature(secret_key: str, method: str, params: dict):
    """
     生成 HMAC 签名

     参数：
     - secret_key: 用于生成签名的共享密钥
     - http_method: HTTP 请求方法（如 GET、POST）
     - params: 请求参数的字典
     """
    # 对参数进行排序并构造签名字符串
    # string_to_sign = method.upper() + "&" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    # hashed = hmac.new(secret_key.encode('utf-8'), string_to_sign.encode('utf-8'), hashlib.sha1)  # hashlib.sha256
    # signature = base64.b64encode(hashed.digest()).decode()

    sorted_params = sorted(params.items())  # , key=lambda x: x[0]
    canonicalized_query_string = '&'.join(f'{quote_plus(k)}={quote_plus(str(v))}' for k, v in sorted_params)
    string_to_sign = f'{method}&%2F&{quote_plus(canonicalized_query_string)}'

    secreted_string = hmac.new(bytes(f'{secret_key}&', 'utf-8'), bytes(string_to_sign, 'utf-8'), hashlib.sha1).digest()
    signature = base64.b64encode(secreted_string).decode('utf-8')
    return signature


# 生成 JWT 令牌,带有效期的 Access Token
def create_access_token(data: dict, expires_minutes: int = None):
    to_encode = data.copy()
    expires_delta = timedelta(minutes=15 if expires_minutes else Config.ACCESS_TOKEN_EXPIRE_MINUTES)
    expire = datetime.utcnow() + expires_delta  # datetime.utcnow() + timedelta(days=7)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, Config.SECRET_KEY, algorithm=Config.ALGORITHM)
    return encoded_jwt


# 验证和解码 Token,Access Token 有效性，并返回 username
def verify_access_token(token: str) -> str:
    try:
        # rsa.verify(original_message, signed_message, public_key)
        payload = jwt.decode(token, Config.SECRET_KEY, algorithms=[Config.ALGORITHM])
        username: str = payload.get("sub")
        if not username:
            return None
    except JWTError:
        return None
    return username


# 通过公钥验证签名，使用公钥 public_key 非对称密钥,验证与私钥签名的消息 message 是否被篡改
def verify_ecdsa_signature(public_key: str, message: str, signature: str):
    try:
        signature_bytes = base64.b64decode(signature)  # 从 base64 解码签名
        vk = ecdsa.VerifyingKey.from_string(bytes.fromhex(public_key), curve=ecdsa.SECP256k1)  # 生成验证公钥对象
        vk.verify(signature_bytes, message.encode('utf-8'), hashfunc=hashlib.sha256)  # 使用相同的 hash 函数验证
        return True
    except ecdsa.BadSignatureError:
        return False


# 验证基于共享密钥的 HMAC 签名,共享密钥生成签名，对称密钥,比较生成的签名与提供的签名是否匹配。
def verify_hmac_signature(shared_secret: str, data: str, signature: str):
    """
     使用 HMAC-SHA256 验证签名是否有效。
     参数：
     - shared_secret: 用于生成签名的共享密钥（对称密钥）
     - data: 需要验证的消息数据
     - signature: 需要验证的签名
     """
    hmac_signature = hmac.new(shared_secret.encode(), data.encode(), hashlib.sha256).digest()
    expected_signature = base64.urlsafe_b64encode(hmac_signature).decode()
    return hmac.compare_digest(signature, expected_signature)


# def encode_id(raw_id):
#     return base64.urlsafe_b64encode(raw_id.encode()).decode().rstrip('=')
#
# def decode_id(encoded_id):
#     padded_encoded_id = encoded_id + '=' * (-len(encoded_id) % 4)
#     return base64.urlsafe_b64decode(padded_encoded_id.encode()).decode()

if __name__ == "__main__":
    key = 'e439bcf5e2b3f706b05a4962aaeac0cc'
    secret = 'MDRjODdhNDcwYjQyNGIyMjQzOGQ2ZDE4'
    api_key = f"{key}:{secret}"
    api_key_base64 = base64.b64encode(api_key.encode('utf-8')).decode('utf-8')
    print(api_key_base64)