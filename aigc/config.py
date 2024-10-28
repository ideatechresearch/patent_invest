import base64
import hmac, ecdsa, hashlib
from fastapi import HTTPException
from jose import JWTError, jwt
import requests, json
from urllib.parse import quote_plus, urlencode, quote
from datetime import datetime, timedelta
import time
import uuid


class Config(object):
    SQLALCHEMY_DATABASE_URI = f'mysql+pymysql://technet:{quote_plus("***")}@***.mysql.rds.aliyuncs.com:3306/technet?charset=utf8'
    SQLALCHEMY_COMMIT_ON_TEARDOWN = False
    SQLALCHEMY_TACK_MODIFICATIONS = True
    SQLALCHEMY_ECHO = True
    SECRET_KEY = '***'
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 60
    HTTP_TIMEOUT_SEC = 60
    MAX_TASKS = 1024
    DEVICE_ID = '***'
    INFURA_PROJECT_ID = ''
    DATA_FOLDER = 'data'
    QDRANT_HOST = 'qdrant'  # '47.***'#
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

    DashScope_Service_Key = '***' 
    ALIYUN_AK_ID = '***'
    ALIYUN_Secret_Key = '***'
    ALIYUN_nls_AppId = '***'
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
    Baichuan_Service_Key = '***'
    HF_Service_Key = '***'

    VOLCE_AK_ID = '***' 
    VOLCE_Secret_Key = '***' 
    ARK_Service_Key = '***'

    SOCKS_Proxies = {
        'http': 'socks5://your_socks_proxy_address:port',
        'https': 'socks5://u:p@proxy_address:port', }



def md5_sign(q: str, salt: str, appid: str, secret_key: str) -> str:
    sign_str = appid + q + salt + secret_key
    return hashlib.md5(sign_str.encode('utf-8')).hexdigest()


# sha256 HMAC 签名
def hmac_sha256(key: bytes, content: str):
    return hmac.new(key, content.encode("utf-8"), hashlib.sha256).digest()


# sha256 hash
def hash_sha256(content: str):
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


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
def get_aliyun_access_token(access_key_id=Config.ALIYUN_AK_ID, access_key_secret=Config.ALIYUN_Secret_Key):
    parameters = {
        'AccessKeyId': access_key_id,
        'Action': 'CreateToken',
        'Format': 'JSON',
        'RegionId': 'cn-shanghai',
        'SignatureMethod': 'HMAC-SHA1',
        'SignatureNonce': str(uuid.uuid1()),
        'SignatureVersion': '1.0',
        'Timestamp': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        'Version': '2019-02-28'
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

    full_url = f"http://nls-meta.cn-shanghai.aliyuncs.com/?Signature={signature}&{query_string}"
    response = requests.get(full_url)
    response.raise_for_status()

    if response.ok:
        token_info = response.json().get('Token', {})
        return token_info.get('Id'), token_info.get('ExpireTime')

    print(response.text)
    return None, None  # token, expire_time


# 火山引擎生成签名
def get_ark_signature(service: str, host: str, action: str, region: str = "cn-north-1", version: str = "2018-01-01",
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
                          secret_key: str = Config.TENCENT_Secret_Key, timestamp: int = None):
    if not host:
        host = f"{service}.tencentcloudapi.com"
    if not timestamp:
        timestamp = int(time.time())
    if not params:
        http_request_method = "GET"  # GET 请求签名
        headers = {
            'Action': action,  # 'DescribeInstances'
            'InstanceIds.0': 'ins-09dx96dg',
            'Limit': 20,
            'Nonce': 11886,  # 随机数,确保唯一性
            'Offset': 0,
            'Region': 'ap-shanghai',
            'SecretId': secret_id,
            'Timestamp': timestamp,
            'Version': '2017-03-12'
        }
        query_string = '&'.join(f"{k}={quote(str(v), safe='')}" for k, v in sorted(headers.items()))
        string_to_sign = f"{http_request_method}{host}/?{query_string}"
        signature = hmac.new(secret_key.encode("utf8"), string_to_sign.encode("utf8"), hashlib.sha1).digest()
        headers["Signature"] = quote_plus(signature)  # base64.b64encode(signature)
        return headers

    algorithm = "TC3-HMAC-SHA256"
    date = datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d")

    # ************* 步骤 1：拼接规范请求串 *************
    http_request_method = "POST"
    canonical_uri = "/"
    canonical_querystring = ""
    ct = "application/json; charset=utf-8"
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

    headers = {
        "Authorization": authorization,  # "<认证信息>"
        "Content-Type": ct,  # "application/json"
        "Host": host,  # "hunyuan.tencentcloudapi.com"
        "X-TC-Action": action,  # "ChatCompletions"
        # 这里还需要添加一些认证相关的Header
        "X-TC-Timestamp": str(timestamp),
        'X-TC-Version': '2023-09-01'  # version,"<API版本号>",'2017-03-12'
        # "X-TC-Region": 'ap-shanghai 'region,"<区域>",
    }
    return headers


def build_url(url: str, access_token: str = get_baidu_access_token()) -> str:
    url = url.strip().strip('"')
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url

    params = {"access_token": access_token}
    query_string = urlencode(params)
    return f"{url}?{query_string}"


#  API 签名
def generate_tencent_signature(secret_key: str, method: str, params: dict):
    """
    生成腾讯云 API 请求签名

    参数：
    - secret_key: 用于生成签名的腾讯云 API 密钥
    - http_method: HTTP 请求方法（如 GET、POST）
    - params: 请求参数的字典
    sign_str = f"{TRANSLATE_KEY}{timestamp}{nonce}
    """
    # string_to_sign =method+f"{service}.tencentcloudapi.com" + "/?" + "&".join("%s=%s" % (k, params[k]) for k in sorted(params))
    string_to_sign = method + "&" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    hashed = hmac.new(secret_key.encode('utf-8'), string_to_sign.encode('utf-8'), hashlib.sha1)  # hashlib.sha256
    signature = base64.b64encode(hashed.digest()).decode()
    return signature


# 生成请求签名
def generate_hmac_signature(secret_key: str, method: str, params: dict):
    """
     生成 HMAC 签名

     参数：
     - secret_key: 用于生成签名的共享密钥
     - http_method: HTTP 请求方法（如 GET、POST）
     - params: 请求参数的字典
     """
    # 对参数进行排序并构造签名字符串
    sorted_params = sorted(params.items())
    canonicalized_query_string = '&'.join(f'{quote_plus(k)}={quote_plus(str(v))}' for k, v in sorted_params)
    string_to_sign = f'{method}&%2F&{quote_plus(canonicalized_query_string)}'

    secreted_string = hmac.new(bytes(f'{secret_key}&', 'utf-8'), bytes(string_to_sign, 'utf-8'), hashlib.sha1).digest()
    signature = base64.b64encode(secreted_string).decode('utf-8')
    return signature


# 带有效期的 Access Token
def create_access_token(data: dict, expires_minutes: None):
    to_encode = data.copy()
    expires_delta = timedelta(minutes=15 if expires_minutes else Config.ACCESS_TOKEN_EXPIRE_MINUTES)
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, Config.SECRET_KEY, algorithm=Config.ALGORITHM)
    return encoded_jwt


# 验证和解码 Token,Access Token 有效性，并返回 username
def verify_access_token(token: str):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Invalid token,Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # rsa.verify(original_message, signed_message, public_key)
        payload = jwt.decode(token, Config.SECRET_KEY, algorithms=[Config.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return username


# 通过公钥验证签名，使用公钥 public_key 非对称密钥,验证与私钥签名的消息 message 是否被篡改
def verify_ecdsa_signature(public_key: str, message: str, signature: str):
    try:
        # 从 base64 解码签名
        signature_bytes = base64.b64decode(signature)
        vk = ecdsa.VerifyingKey.from_string(bytes.fromhex(public_key), curve=ecdsa.SECP256k1)
        vk.verify(signature_bytes, message.encode('utf-8'))

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
