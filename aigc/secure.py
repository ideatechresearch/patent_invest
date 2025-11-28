from jose import JWTError, jwt
import hmac, ecdsa, hashlib
import uuid
import base64
from urllib.parse import quote_plus, urlencode, urlparse, quote, unquote
from datetime import datetime, timedelta, timezone
import time
import requests, json
from functools import wraps

# from starlette import status
from starlette.requests import Request
from starlette.exceptions import HTTPException

from config import Config


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

# 生成 JWT 令牌,带有效期的 Access Token
def create_access_token(data: dict, expires_minutes: int = None) -> str:
    now = datetime.now(timezone.utc)
    expires_delta = timedelta(minutes=expires_minutes or Config.ACCESS_TOKEN_EXPIRE_MINUTES)
    expire = now + expires_delta  # datetime.utcnow() + timedelta(days=7)
    to_encode = {**data, "exp": expire, "iat": now}  # 可以携带更丰富的 payload:{"sub": username, 'user_id': user_id}
    return jwt.encode(to_encode, Config.SECRET_KEY, algorithm=Config.ALGORITHM)  # encoded_jwt


# 验证和解码 Token,Access Token 有效性，并返回 payload
def decode_token(token: str) -> dict | None:
    try:
        # rsa.verify(original_message, signed_message, public_key)
        return jwt.decode(token, Config.SECRET_KEY, algorithms=[Config.ALGORITHM])
    except JWTError:
        return None


# 验证和解码 Token,Access Token 有效性
def verify_access_token(token: str) -> str:
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Could not validate credentials. Invalid or expired token.",
                            headers={"WWW-Authenticate": "Bearer"}, )
    username = payload.get("sub") or payload.get("user_id")
    if not username:
        raise HTTPException(status_code=401, detail="Could not validate credentials. Missing subject.",
                            headers={"WWW-Authenticate": "Bearer"}, )
    return username


# 使用 Refresh Token 刷新新的 Access Token
def refresh_access_token(refresh_token: str, expires_minutes: int = None) -> str:
    payload = decode_token(refresh_token)
    if payload is None or payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token", )
    username = payload.get("sub") or payload.get("user_id")
    if not username:
        raise HTTPException(status_code=401, detail="Invalid refresh token payload", )
    # 生成新的 access token
    new_access_token = create_access_token({"sub": username, "type": "access"}, expires_minutes)
    return new_access_token


# for tokens in Api_Tokens:
def scheduled_token_refresh(token_info: dict, func):
    now = datetime.now(timezone.utc)
    if token_info["expires_at"] is None or now > token_info["expires_at"] - timedelta(minutes=5):
        try:
            token_info["access_token"] = func()
            token_info["expires_at"] = now + timedelta(minutes=token_info["expires_delta"])
            # response = requests.post(f"{BASE_URL}/refresh", json={"refresh_token": tokens["refresh_token"]})
        except Exception as e:
            print(f"Error refreshing token for {token_info['type']}: {e}")


# 用于依赖注入
async def verify_api_key(authorization: str = None):  # Header(None)
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing API key")
    api_key = authorization.replace("Bearer ", "").strip()  # 如果用的是 Bearer Token 格式
    # active_keys = await User.get_active_api_keys(redis=redis)
    if api_key not in Config.VALID_API_KEYS:  # 其他redis...
        raise HTTPException(status_code=401, detail="Invalid API key")


def require_api_key(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        request: Request = kwargs.get("request")
        authorization = request.headers.get("Authorization")
        await verify_api_key(authorization)
        return await func(*args, **kwargs)

    return wrapper


def extract_token(request: Request) -> str | None:
    # Token 提取工具（优先级: Authorization header -> cookie)
    # 1) Authorization header
    auth = request.headers.get("Authorization")
    if auth and auth.startswith("Bearer "):
        return auth.split(" ", 1)[1].strip()

    # 2) HttpOnly cookie (选择cookies方式存 token)
    cookie_token = request.cookies.get("access_token")
    if cookie_token:
        return cookie_token

    return None


def get_access_user(request: Request) -> str | None:
    token_from_request = extract_token(request)
    if token_from_request:
        try:
            username = verify_access_token(token_from_request)
            return username
        except HTTPException as e:
            # print(f"Error getting access token from request,{e.detail}")
            pass

    return None


def split_tokens(auth: str):
    """分割token

    Args:
        auth: Authorization头部值
    Returns:
        List[str]: token列表
    """
    if not auth:
        return []
    auth = auth.replace('Bearer', '').strip()
    return [t.strip() for t in auth.split(',') if t.strip()]


async def verify_request_signature(request: Request, api_secret_keys: dict, time_out: int = None):
    # 请求签名验证的函数，主要用于确保请求的来源可信，防止请求在传输过程中被篡改（防伪造、防篡改、防重放）
    api_key = request.headers.get("X-API-KEY")
    signature = request.headers.get("X-SIGNATURE")
    timestamp = request.headers.get("X-TIMESTAMP")

    if not all([api_key, signature, timestamp]):
        raise HTTPException(status_code=400, detail="Missing authentication headers")

    # 检查时间戳是否超时
    current_time = int(time.time())
    request_time = int(timestamp)
    timeout = time_out or Config.VERIFY_TIMEOUT_SEC
    if abs(current_time - request_time) > timeout:  # 5分钟的时间窗口
        raise HTTPException(status_code=403, detail="Request timestamp expired")

    # 检查API Key是否合法
    secret = api_secret_keys.get(api_key)
    if not secret:
        raise HTTPException(status_code=403, detail="Invalid API Key")

    # 从请求中构造签名字符串
    method = request.method.upper()
    url = str(request.url)  # 避免包含 query 参数顺序问题
    body = await request.body() if request.method in ["POST", "PUT", "PATCH"] else b""

    # 拼接签名字符串 build_signature
    message = f"{method}|{url}|{body.decode()}|{timestamp}"

    # 使用 HMAC-SHA256 生成服务器端的签名
    server_signature = hmac.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()

    if not hmac.compare_digest(server_signature, signature):
        raise HTTPException(status_code=403, detail="Invalid signature")

    return True


def md5_sign(content: str) -> str:
    return hashlib.md5(content.encode('utf-8')).hexdigest()


# sha256 非对称加密
def hmac_sha256(key: bytes, content: str) -> bytes:
    """生成 HMAC-SHA256 签名"""
    return hmac.new(key, content.encode("utf-8"), digestmod=hashlib.sha256).digest()


# sha256 hash
def hash_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# calculate sha256 and encode to base64
def sha256base64(data):
    sha256 = hashlib.sha256()
    sha256.update(data)
    digest = base64.b64encode(sha256.digest()).decode(encoding='utf-8')
    return digest


def construct_sorted_query(params: dict):
    sort_query_string = []
    for key in sorted(params.keys()):
        if isinstance(params[key], list):
            for k in params[key]:
                sort_query_string.append(quote(key, safe="-_.~") + "=" + quote(k, safe="-_.~"))
        else:
            sort_query_string.append(quote(key, safe="-_.~") + "=" + quote(params[key], safe="-_.~"))

    query = "&".join(sort_query_string)  # query[:-1]
    return query.replace("+", "%20").replace('*', '%2A').replace('%7E', '~')  # .encode("utf-8")


def build_tts_stream_headers(api_key) -> dict:
    headers = {
        'accept': 'application/json, text/plain, */*',
        'content-type': 'application/json',
        'authorization': "Bearer " + api_key,
    }
    return headers


def generate_uuid(with_hyphen: bool = True) -> str:
    """生成UUID

    Args:
        with_hyphen: 是否包含连字符

    Returns:
        str: UUID字符串
    """
    _uuid = str(uuid.uuid4())
    return _uuid if with_hyphen else _uuid.replace('-', '')


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
# 阿里云服务交互时的身份验证
# https://help.aliyun.com/zh/ocr/developer-reference/signature-method?spm=a2c4g.11186623.help-menu-252763.d_3_2_3.42df53e75y9ZST
# https://help.aliyun.com/zh/viapi/developer-reference/request-a-signature?spm=a2c4g.11186623.help-menu-142958.d_4_4.525e16d1nt551a
def get_aliyun_access_token(service: str = "nls-meta", region: str = "cn-shanghai",
                            action: str = 'CreateToken', http_method="GET", body=None, version: str = '2019-02-28',
                            access_key_id=Config.ALIYUN_AK_ID, access_key_secret=Config.ALIYUN_Secret_Key):
    # 公共请求参数
    parameters = {
        'AccessKeyId': access_key_id,
        'Action': action,
        'Format': 'JSON',  # 返回消息的格式
        'RegionId': region,
        'SignatureMethod': 'HMAC-SHA1',
        'SignatureNonce': str(uuid.uuid1()),
        'SignatureVersion': '1.0',
        'Timestamp': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        'Version': version  # '2019-02-28',"2020-06-29"
    }
    if body:
        parameters.update(body)

    def encode_text(text):
        # 特殊URL编码,加号+替换为%20,星号*替换为%2A,波浪号~替换为%7E
        return quote_plus(text).replace('+', '%20').replace('*', '%2A').replace('%7E', '~')

    def encode_dict(dic):
        # urlencode 会将字典转为查询字符串
        dic_sorted = sorted(dic.items())
        return urlencode(dic_sorted).replace('+', '%20').replace('*', '%2A').replace('%7E', '~')

    # "https://%s/?%s" % (endpoint, '&'.join('%s=%s' % (k, v) for k, v in parameters.items()))
    query_string = encode_dict(parameters)  # construct_sorted_query
    # HTTPMethod + “&” + UrlEncode(“/”) + ”&” + UrlEncode(sortedQueryString)
    string_to_sign = f"{http_method.upper()}&{encode_text('/')}&{encode_text(query_string)}"
    # 签名采用HmacSHA1算法+Base64
    secreted_string = hmac.new(bytes(f"{access_key_secret}&", 'utf-8'),
                               bytes(string_to_sign, 'utf-8'),
                               hashlib.sha1).digest()

    signature = base64.b64encode(secreted_string).decode()
    signature = encode_text(signature)  # Base64( HMAC-SHA1(stringToSign, accessKeySecret + "&"));

    full_url = f"http://{service}.{region}.aliyuncs.com/?Signature={signature}&{query_string}"
    if action == 'CreateToken':
        response = requests.get(full_url)
        response.raise_for_status()

        if response.ok:
            token_info = response.json().get('Token', {})
            return token_info.get('Id'), token_info.get('ExpireTime')  # token, expire_time
        print(response.text)

    return parameters, full_url


def get_xfyun_authorization(api_key=Config.XF_API_Key, api_secret=Config.XF_Secret_Key,
                            host="spark-api.xf-yun.com", path="/v3.5/chat", method='GET'):
    # "itrans.xf-yun.com",/v1/its
    # Step 1: 生成当前日期
    cur_time = datetime.now()
    from wsgiref.handlers import format_date_time
    date = format_date_time(time.mktime(cur_time.timetuple()))  # RFC1123格式
    # datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%a, %d %b %Y %H:%M:%S GMT")
    # Step 2: 拼接鉴权字符串tmp
    signature_origin = f"host: {host}\ndate: {date}\n{method} {path} HTTP/1.1"

    # Step 3: 生成签名
    signature_sha = hmac_sha256(api_secret.encode('utf-8'), signature_origin)

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
    url = f"https://{host}{path}?" + urlencode(headers)  # https:// .wss:// requset_url + "?" +
    return headers, url


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
# https://www.volcengine.com/docs/6793/127781
# https://www.volcengine.com/docs/6369/67269
# https://www.volcengine.com/docs/6369/67270
# https://github.com/volcengine/volc-openapi-demos/blob/main/signature/python/sign.py
def get_ark_signature(action: str, service: str, host: str = None, region: str = "cn-north-1",
                      version: str = "2018-01-01", http_method="GET", body=None,
                      access_key_id: str = Config.VOLC_AK_ID, secret_access_key: str = Config.VOLC_Secret_Key,
                      timenow=None):
    if not host:
        host = f"{service}.volcengineapi.com"  # 'open.volcengineapi.com'
    if not timenow:
        timenow = datetime.utcnow()
    date = timenow.strftime('%Y%m%dT%H%M%SZ')  # YYYYMMDD'T'HHMMSS'Z'
    date_short = date[:8]  # Date 精确到日, YYYYMMDD

    # 构建Canonical Request
    canonical_uri = "/"  # 如果 URI 为空，那么使用"/"作为绝对路径
    canonical_querystring = f"Action={action}&Version={version}"  # construct_sorted_query
    # "X-Expires"
    canonical_headers = f"host:{host}\nx-date:{date}\n"  # 将需要参与签名的header的key全部转成小写， 然后以ASCII排序后以key-value的方式组合后换行构建,注意需要在结尾增加一个回车换行\n。
    signed_headers = "host;x-date"  # host、x-date如果存在header中则必选参与 content-type;host;x-content-sha256;x-date
    # HexEncode(Hash(RequestPayload))
    payload_hash = hash_sha256("" if body is None else json.dumps(body))  # GET空请求体的哈希
    canonical_request = "\n".join([http_method.upper(), canonical_uri, canonical_querystring,
                                   canonical_headers, signed_headers, payload_hash])
    # print(canonical_request)
    # 构建String to Sign
    algorithm = "HMAC-SHA256"
    credential_scope = "/".join([date_short, region, service, 'request'])  # ${YYYYMMDD}/${region}/${service}/request
    canonical_request_hash = hash_sha256(canonical_request)
    string_to_sign = "\n".join([algorithm, date, credential_scope, canonical_request_hash])

    # print(string_to_sign)

    # 计算签名
    def get_signing_key(secret_key, date_short, region, service):
        k_date = hmac_sha256(secret_key.encode('utf-8'), date_short)  # VOLC
        k_region = hmac_sha256(k_date, region)
        k_service = hmac_sha256(k_region, service)
        k_signing = hmac_sha256(k_service, "request")
        return k_signing

    signing_key = get_signing_key(secret_access_key, date_short, region, service)
    # HexEncode(HMAC(Signingkey, StringToSign)) hmac.new(signing_key, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()
    signature = hmac_sha256(signing_key, string_to_sign).hex()

    # 构建Authorization头: HMAC-SHA256 Credential={AccessKeyId}/{CredentialScope}, SignedHeaders={SignedHeaders}, Signature={Signature}
    authorization_header = f"{algorithm} Credential={access_key_id}/{credential_scope}, SignedHeaders={signed_headers}, Signature={signature}"
    # 签名参数可以在query中，也可以在header中
    headers = {
        "Authorization": authorization_header,
        "Content-Type": "application/json",  # application/x-www-form-urlencoded
        "Host": host,
        "X-Date": date
        # 'X-Security-Token'
    }
    # Action和Version必须放在query当中
    url = f"https://{host}/?{canonical_querystring}"

    return headers, url


def get_tencent_signature(service, host=None, body=None, action='ChatCompletions',
                          secret_id: str = Config.TENCENT_SecretId, secret_key: str = Config.TENCENT_Secret_Key,
                          timestamp: int = None, region: str = "ap-shanghai", version='2023-09-01'):
    # https://cloud.tencent.com/document/api/1093/35641
    if not host:
        host = f"{service}.tencentcloudapi.com"  # url.split("//")[-1]
    if not timestamp:
        timestamp = int(time.time())
        # 支持 POST 和 GET 方式
    if not body:
        http_request_method = "GET"  # GET 请求签名
        params = {
            'Action': action,  # 'DescribeInstances'
            'InstanceIds.0': 'ins-09dx96dg',
            'Limit': 20,
            'Nonce': str(uuid.uuid1().int >> 64),  # 随机数,确保唯一性
            'Offset': 0,
            'Region': region,
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

    algorithm = "TC3-HMAC-SHA256"  # 使用签名方法 v3
    date = datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m-%d")  # UTC+0

    # ************* 步骤 1：拼接规范请求串 *************
    http_request_method = "POST"
    canonical_uri = "/"
    canonical_querystring = ""
    ct = "application/json; charset=utf-8"
    canonical_headers = (
        f"content-type:{ct}\n"
        f"host:{host}\n"
        f"x-tc-action:{action.lower()}\n"
    )
    signed_headers = "content-type;host;x-tc-action"  # content-type 和 host 为必选头部,以分号（;）分隔
    hashed_request_payload = hash_sha256(
        "" if body is None else json.dumps(body, separators=(",", ":"), ensure_ascii=False))
    # Lowercase(HexEncode(Hash.SHA256(RequestPayload)))
    canonical_request = "\n".join([http_request_method, canonical_uri, canonical_querystring,
                                   canonical_headers, signed_headers, hashed_request_payload])

    # ************* 步骤 2：拼接待签名字符串 *************
    credential_scope = f"{date}/{service}/tc3_request"  # 待签名字符串
    hashed_canonical_request = hash_sha256(canonical_request)  # Lowercase(HexEncode(Hash.SHA256(CanonicalRequest)))
    string_to_sign = "\n".join([algorithm, str(timestamp), credential_scope, hashed_canonical_request])

    # ************* 步骤 3：计算签名 *************
    secret_date = hmac_sha256(("TC3" + secret_key).encode("utf-8"), date)
    secret_service = hmac_sha256(secret_date, service)
    secret_signing = hmac_sha256(secret_service, "tc3_request")
    signature = hmac_sha256(secret_signing, string_to_sign).hex()
    # Signature = HexEncode(HMAC_SHA256(SecretSigning, StringToSign))
    # signature = hmac.new(secret_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
    # ************* 步骤 4：拼接 Authorization *************
    authorization = (algorithm + " " +
                     "Credential=" + secret_id + "/" + credential_scope + ", " +
                     "SignedHeaders=" + signed_headers + ", " +
                     "Signature=" + signature)

    # return authorization
    # 公共参数需要统一放到 HTTP Header 请求头部
    headers = {
        "Authorization": authorization,  # "<认证信息>"
        "Content-Type": ct,  # Content-Type "application/json"
        "Host": host,  # 主机名 "hunyuan.tencentcloudapi.com","tmt.tencentcloudapi.com"
        "X-TC-Action": action,  # 请求接口名 "ChatCompletions","TextTranslate"
        # 这里还需要添加一些认证相关的Header
        "X-TC-Timestamp": str(timestamp),
        "X-TC-Version": version,  # "<API版本号>"
        "X-TC-Region": region  # region,"<区域>",
    }
    return headers


def build_url(url: str, access_token: str = None, **kwargs) -> str:
    url = url.strip().strip('"')
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url
    if not access_token:
        access_token = get_baidu_access_token()

    params = {"access_token": access_token}
    params.update(kwargs)
    query_string = urlencode(params)
    return f"{url}?{query_string}"
