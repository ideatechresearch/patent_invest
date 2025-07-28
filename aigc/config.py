import base64
import hmac, ecdsa, hashlib
from jose import JWTError, jwt
import requests, json
from urllib.parse import quote_plus, urlencode, urlparse, quote, unquote, parse_qs, unquote_plus
from datetime import datetime, timedelta, timezone
import time
import uuid
import os
from dataclasses import dataclass, asdict


@dataclass(kw_only=True)  # , frozen=True
class Config(object):
    """
       全局参数配置（示例结构），涉及配置中的敏感信息，真实环境用从 YAML文件加载配置覆盖，以下为占位内容，仅供结构参考，非真实配置
    """
    # {{!IGNORE_START!}} (请忽略以下内容)
    DATABASE_PWD = "***"
    SQLALCHEMY_DATABASE_URI = f'mysql+pymysql://technet:{quote_plus(DATABASE_PWD)}@***.mysql.rds.aliyuncs.com:3306/technet?charset=utf8mb4'
    ASYNC_SQLALCHEMY_DATABASE_URI = f'mysql+aiomysql://ideatech:{quote_plus(DATABASE_PWD)}@***.mysql.rds.aliyuncs.com:3306/h3yun?charset=utf8'
    SQLALCHEMY_COMMIT_ON_TEARDOWN = False
    SQLALCHEMY_TACK_MODIFICATIONS = True
    SQLALCHEMY_ECHO = True
    SECRET_KEY = '***'
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 60
    VERIFY_TIMEOUT_SEC = 300
    HTTP_TIMEOUT_SEC = 100.0
    MAX_TASKS = 1024
    MAX_CACHE = 1024
    MAX_CONCURRENT = 100
    MAX_RETRY_COUNT = 3
    RETRY_DELAY = 5
    DEVICE_ID = '***'
    INFURA_PROJECT_ID = ''
    DATA_FOLDER = 'data'
    Version = 'v1.2.5'
    _config_path = 'config.yaml'
    __config_data = {}  # 动态加载的数据，用于还原
    __config_dynamic = {}  # 其他默认配置项，用于运行时
    WEBUI_NAME = 'aigc'
    WEBUI_URL = 'http://***:7000'
    # https://api.qdrant.tech/api-reference
    OLLAMA_HOST = 'localhost'
    NEO_URI = "bolt://neo4j:7687"
    NEO_Username = "neo4j"
    NEO_Password = '***'
    DASK_Cluster = 'tcp://10.10.10.20:8786'
    QDRANT_HOST = 'qdrant'  # "47.110.156.41"
    QDRANT_GRPC_PORT = 6334
    QDRANT_URL = "http://***:6333"  # ":memory:"
    WECHAT_URL = 'http://idea_ai_robot:28089'
    REDIS_HOST = "redis_aigc"
    REDIS_PORT = 6379  # 7007
    REDIS_CACHE_SEC = 99999  # 86400
    REDIS_MAX_CONCURRENT = 50

    VALID_API_KEYS = {"token-abc123", "token-def456"}
    DEFAULT_LANGUAGE = 'Chinese'
    DEFAULT_MODEL = 'moonshot'  # 'qwen'
    DEFAULT_MODEL_ENCODING = "gpt-3.5-turbo"
    DEFAULT_MODEL_EMBEDDING = 'BAAI/bge-large-zh-v1.5'
    DEFAULT_MODEL_METADATA = 'qwen:qwen-coder-plus'  # qwen-coder-turbo
    DEFAULT_MODEL_FUNCTION = 'qwen:qwen-max'
    DEFAULT_MAX_TOKENS = 4000

    BAIDU_API_Key = '***'
    BAIDU_Secret_Key = '***'
    # https://console.bce.baidu.com/qianfan/ais/console/applicationConsole/application/v1
    # https://console.bce.baidu.com/iam/?_=1739264215214#/iam/apikey/list
    BAIDU_qianfan_API_Key = '***'  # 45844683
    BAIDU_qianfan_Secret_Key = '***' 
    QIANFAN_Service_Key = '***'
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
    # https://console.bce.baidu.com/ai/#/ai/machinetranslation/app/list
    BAIDU_translate_API_Key = '***'  # '116451173'
    BAIDU_translate_Secret_Key = '***'
    # https://fanyi-api.baidu.com/api/trans/product/desktop
    BAIDU_trans_AppId = '20240525002061478'
    BAIDU_trans_Secret_Key = '***'
    # https://lbsyun.baidu.com/apiconsole/center
    BMAP_API_Key = '***'
    # https://console.amap.com/dev/key/app
    AMAP_API_Key = '***'

    DashScope_Service_Key = '***'  # Bailian,DashScope
    Bailian_Service_Key = '***'
    DashVectore_Service_Key = '***'
    ALIYUN_AK_ID = '***'
    ALIYUN_Secret_Key = '***'
    ALIYUN_nls_AppId = 'CjIryfx0wqdvCHZk'

    ALIYUN_oss_AK_ID = '***'
    ALIYUN_oss_Secret_Key = '***'
    ALIYUN_oss_region = "oss-cn-hangzhou"
    ALIYUN_oss_internal = False  # 是否使用内网地址
    ALIYUN_oss_endpoint = f'https://{ALIYUN_oss_region}.aliyuncs.com'  # 'https://oss-cn-shanghai.aliyuncs.com'
    ALIYUN_Bucket_Name = 'idea-aigc-images'  # 存储桶名称 'rime'
    ALIYUN_Bucket_Domain = f"https://{ALIYUN_Bucket_Name}.{ALIYUN_oss_region}{'-internal' if ALIYUN_oss_internal else ''}.aliyuncs.com"  # 加速域名"https://rime.oss-accelerate.aliyuncs.com"
    # https://console.cloud.tencent.com/hunyuan/api-key
    TENCENT_SecretId = '***'
    TENCENT_Secret_Key = '***'
    TENCENT_Service_Key = '***'

    # https://console.xfyun.cn/services
    XF_AppID = 'e823df98'
    XF_API_Key = '***'
    XF_Secret_Key = '***'
    Spark_Service_Key = '***'  # AK:SK,{XF_API_Key}:{XF_Secret_Key} ws协议的apikey和apisecret 按照ak:sk格式拼接
    XF_API_Password = ['***','***']  # http协议的APIpassword
    # https://aistudio.google.com/apikey
    GEMINI_Service_Key = '***'
    # https://console.anthropic.com/settings/keys
    # https://www.cursor.com/cn/settings
    Anthropic_Service_Key = '***'

    # https://cnb.cool/profile/token
    CNB_Server_Token = "***"

    # https://cloud.siliconflow.cn/models
    Silicon_Service_Key = '***'
    Moonshot_Service_Key = "***"
    # https://modelscope.cn/my/myaccesstoken
    ModelScope_Service_Key = "***"
    # https://aihubmix.com/token
    AiHubMix_Service_Key = "***"
    # https://tokenflux.ai/dashboard/api-keys
    TokenFlux_Service_Key = "***"
    # https://console.x.ai/team/457298eb-40a8-4308-907d-0a36ec706042/api-keys/create
    Grok_Service_Key = '***'
    # https://build.nvidia.com/explore/discover
    NVIDIA_Service_Key = '***'  
    # https://platform.openai.com/settings/organization/admin-keys
    OPENAI_Admin_Service_Key = '***'
    OPENAI_Service_Key = '***'
    # https://open.bigmodel.cn/console/overview
    GLM_Service_Key = "***"  # f'Bearer {}'
    # https://platform.baichuan-ai.com/console/apikey
    Baichuan_Service_Key = '***'
    # https://platform.deepseek.com/usage
    DeepSeek_Service_Key = '***'  
    # https://platform.minimaxi.com/user-center/basic-information
    MINIMAX_Group_ID = '1887488082316366022'
    MINIMAX_Service_Key = '***'
    # https://platform.lingyiwanwu.com/apikeys
    Lingyiwanwu_Service_Key = '***'
    # https://ai.youdao.com/console/#/
    YOUDAO_AppID = '775f6562be8c52e4'
    YOUDAO_Service_Key = '***'
    CaiYun_Token = "***"
    HF_Service_Key = '***'
    # https://console.mistral.ai/api-keys/
    MISTRAL_API_KEY = '***'
    Codestral_API_KEY = '***'
    # https://jina.ai/api-dashboard/key-manager
    JINA_Service_Key = '***'
    # https://www.firecrawl.dev/app
    # https://docs.firecrawl.dev/introduction
    Firecrawl_Service_Key = '***' 
    # https://console.volcengine.com/iam/keymanage
    VOLC_AK_ID = '***'
    VOLC_Secret_Key = '***'
    VOLC_AK_ID_admin = '***'  
    VOLC_Secret_Key_admin = '***'
    # https://console.volcengine.com/ark/region:ark+cn-beijing/apiKey?apikey=%7B%7D
    ARK_Service_Key = '***'

    # https://vanna.ai/account/models 基于大模型和RAG的Text2SQL
    # https://vanna.ai/docs/app/
    VANNA_Api_Key = '***'
    # https://platform.stability.ai/account/keys
    Stability_Api_Key = '***'
    # https://www.comet.com/opik/dooven-prime/get-started
    # https://www.comet.com/opik/dooven-prime/home
    OPIK_Api_Key = '***'
    OPIK_Workspace = 'dooven-prime'
    # https://www.searchapi.io/
    SearchApi_Key = '***'  # TLvdjLW2QAgdotQyYMpcXeqx
    # https://api-dashboard.search.brave.com/app/keys
    # https://api-dashboard.search.brave.com/app/dashboard
    Brave_Api_Key = "***"
    # https://serper.dev/api-key
    SERPER_Api_Key = '***'
    # https://console.deepgram.com/project/
    Deepgram_Api_Key = '***'
    # https://fish.audio/zh-CN/discovery/
    FishAudio_Api_Key = '***'
    # https://serpapi.com/dashboard
    SERPAPI_Api_Key = '***'
    # https://www.alphavantage.co/support/#api-key
    AlphaVantage_Api_Key = '***'
    # https://app.tavily.com/home
    TAVILY_Api_Key = '***'  # 88YHAKX5YS9EMHGW6EGZN871
    # https://dashboard.exa.ai/api-keys
    Exa_Api_Key = '***'
    # https://github.com/settings/personal-access-tokens
    GITHUB_Access_Tokens = '***'
    # https://studio.d-id.com/
    DID_Service_Key = '***'
    # https://www.weatherapi.com/my/
    Weather_Service_Key = '***'
    # https://openweathermap.org/api
    WeatherMap_APPID = '***'
    # https://vectorizer.ai/account#Account-apiKey0
    Vectorizer_Api_Key = '***'
    Vectorizer_Secret_Key = '***'
    # https://console.deepgram.com/project/fb261c77-9eb1-4c7a-bf47-1eef8297bafc/keys
    Deepgram_Service_Key = '***'

    # https://app.roboflow.com/main-36irj/settings/api
    Roboflow_Api_Key = '***'

    # https://gitee.com/personal_access_tokens
    GITEE_Access_Tokens = '***'
    # 'https://matrix.ideatech.info'
    Ideatech_API_Key = '***'
    Ideatech_Host = '***.info'

    HTTP_Proxy = 'http://***@***:7930'
    HTTP_Proxies = {
        'http': HTTP_Proxy,
        'https': HTTP_Proxy,
    }

    # {{!IGNORE_END!}}

    @classmethod
    def debug(cls):
        if cls.__config_dynamic.get('IS_DEBUG'):
            return
        cls.SQLALCHEMY_DATABASE_URI = f'mysql+pymysql://***:***@10.10.10.5:3306/kettle?charset=utf8mb4'
        cls.ASYNC_SQLALCHEMY_DATABASE_URI = f'mysql+aiomysql://***:***@10.10.10.5:3306/kettle?charset=utf8'
        cls.WEBUI_URL = 'http://127.0.0.1:7000'
        # https://api.qdrant.tech/api-reference
        cls.QDRANT_GRPC_PORT = None
        cls.WECHAT_URL = 'http://***:28089'
        cls.REDIS_HOST = '10.10.10.5'  # "localhost"
        cls.NEO_URI = "bolt://localhost:7687"
        cls.NEO_Password = '***'

        cls.HTTP_Proxy = 'http://***:***@10.10.10.3:7890'
        # cls.HTTP_Proxies = {'http': cls.HTTP_Proxy,'https': cls.HTTP_Proxy}
        os.environ['HTTP_PROXY'] = cls.HTTP_Proxies['http']
        os.environ['HTTPS_PROXY'] = cls.HTTP_Proxies['https']

        cls.Moonshot_Service_Key = "sk-***"
        cls.DeepSeek_Service_Key = 'sk-***'
        cls.DashScope_Service_Key = 'sk-***'
        cls.__config_dynamic['IS_DEBUG'] = True
        print(cls.get_config_data())

    @classmethod
    def save(cls, filepath=None):
        """将配置项保存到YAML文件"""
        import joblib, yaml
        config_data = {key: getattr(cls, key) for key in dir(cls)
                       if not key.startswith('__')
                       and not key.startswith('_')
                       and not callable(getattr(cls, key))
                       and not isinstance(getattr(cls, key), (classmethod, staticmethod))}

        if not cls.__config_data and any(isinstance(value, str) and set(value) == {"*"}
                                         for value in config_data.values()):
            cls.load(filepath)

        config_data.update(cls.__config_data)
        for key, value in config_data.items():
            if isinstance(value, str) and set(value) == {"*"}:
                print(f"配置文件保存失败，包含占位符 '***','{key}': 当前值 = {value}")
                return config_data

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f'data/config_{timestamp}_{cls.Version}.pkl'
        joblib.dump(config_data, backup_path)
        path = filepath or getattr(cls, '_config_path', 'data/.config.yaml')
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)

        print(f"已保存配置文件 {path},备份目录 {backup_path}")
        return config_data

    @classmethod
    def load(cls, filepath=None):
        """从YAML文件加载配置项"""
        if cls.__config_data:  # is_loaded
            return cls.__config_dynamic

        import yaml
        path = filepath or getattr(cls, '_config_path', 'data/.config.yaml')
        if not os.path.exists(path):
            print(f"配置文件 {path} 不存在，使用默认配置")
            return {}

        with open(path, "r", encoding='utf-8') as f:
            cls.__config_data = yaml.safe_load(f)

        # print('load yaml:', config_data)
        # 将文件中的配置覆盖类中的默认值,cls优先,无值则导入
        for key, value in cls.__config_data.items():
            if hasattr(cls, key):
                current_value = getattr(cls, key)
                if current_value is None or (isinstance(current_value, str) and all(c == '*' for c in current_value)):
                    setattr(cls, key, value)
                elif key.endswith('DATABASE_URI'):
                    setattr(cls, key, value)
                elif current_value != value:
                    print(f"配置项 '{key}' 不一致: 当前值 = {current_value}, 加载值 = {value}")
            else:
                # 添加到动态配置
                cls.__config_dynamic[key] = value

        cls.__config_dynamic['IS_LOADED'] = True
        print(f"配置已加载并更新,文件: {path}")
        return cls.__config_dynamic

    @classmethod
    def get(cls, key, default=None):
        """
        获取配置值，优先级顺序：
        1. 类属性（静态字段）
        2. __config_dynamic 动态字段
        若字段值为 ***遮蔽，尝试从 __config_data 中还原真实值,访问不存在的属性时被调用,第一次访问属性时触发
        """
        if hasattr(cls, key):
            value = getattr(cls, key, default)
        else:
            value = cls.__config_dynamic.get(key, default)

        if isinstance(value, str) and all(c == '*' for c in value):
            if not getattr(cls, "__config_data", None):
                cls.load()
            return cls.__config_data.get(key, default)
        return value

    @classmethod
    def update(cls, **kwargs):
        cls.__config_data.update(**kwargs)

    @classmethod
    def get_config_data(cls):
        """打印当前的配置项"""
        config_data = {f"{key}": f"{value}" for key, value in cls.__dict__.items()
                       if not key.startswith('__') and not callable(value)
                       and not isinstance(value, (classmethod, staticmethod))}
        #   print(f"{key}: {value}")
        return config_data

    @staticmethod
    def mask_sensitive(filepath="config.py"):
        import re
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()

        # 匹配 class Config: 块类体（包括空行/注释）
        config_pattern = re.compile(r'^class\s+Config\s*(\(.*?\))?\s*:\s*$', re.MULTILINE)
        class_match = config_pattern.search(code)
        if not class_match:
            print("❌ 未找到 Config 类。")
            return

        class_start = class_match.end()
        lines = code[class_start:].splitlines(keepends=True)

        # 提取 Config 类体（缩进的部分）
        class_body_lines = []
        for line in lines:
            stripped = line.strip()
            if re.match(r'^(@|def |class )', stripped):
                break  # 跳过方法定义、装饰器、嵌套类等
            if re.match(r'^(async\s+)?def\b', stripped):
                break  # async def 支持
            if not line.strip() == "" and not re.match(r'^[ \t]+', line):
                break  # 到达类体外部，结束
            # 保留赋值语句、空行、注释
            class_body_lines.append(line)

        class_header = class_match.group(0)
        class_body = "".join(class_body_lines)
        print(class_header, class_body)

        # 匹配每一行的赋值语句
        line_pattern = re.compile(
            r'^([ \t]+)([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([\'"])(.*?)(\3)',
            re.MULTILINE
        )
        # 在 class Config 体内部，查找敏感字段赋值
        sensitive_keywords = {'password', 'pwd', 'token', 'tokens', 'secret', 'apikey', 'key', 'access', 'auth',
                              'ak_id'}

        def is_sensitive(key):
            key_lower = key.lower()
            return any(k in key_lower for k in sensitive_keywords)

        def replacer(m):
            indent, key, quote, value, _ = m.groups()
            if is_sensitive(key):
                print(f"替换字段：{key} = {quote}{value}{quote}")
                return f"{indent}{key} = {quote}***{quote}"
            return m.group(0)

        new_body = line_pattern.sub(replacer, class_body)

        # 替换整体代码中的 config 类体
        modified_code = code.replace(class_body, new_body)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"data/{filepath}.{timestamp}.bak"

        with open(backup_path, "w", encoding="utf-8") as f:  # 备份
            f.write(code)

        with open(filepath, "w", encoding="utf-8") as f:  # 写回
            f.write(modified_code)

        print(f"✅ 已更新 Config 类中的敏感字段，并备份到 {backup_path}")

        return class_body, new_body


# 模型编码:0默认，1小，-1最大
AI_Models = [
    # https://dashscope.console.aliyun.com/overview
    # https://bailian.console.aliyun.com/#/home
    # https://pai.console.aliyun.com/?regionId=cn-shanghai&spm=5176.pai-console-inland.console-base_product-drawer-right.dlearn.337e642duQEFXN&workspaceId=567545#/quick-start/models
    {'name': 'qwen', 'type': 'default', 'api_key': '',
     "model": ["qwen-turbo", "qwen1.5-7b-chat", "qwen1.5-32b-chat", "qwen2-7b-instruct",
               "qwen2.5-32b-instruct", "qwen2.5-72b-instruct",
               "qwen3-14b", "qwen3-32b", "qwq-32b", "qwq-32b-preview", "qwen-72b-chat",
               "qwen-omni-turbo", "qwen-vl-plus", "qwen-coder-plus", "qwen-math-plus",
               'qwen-long', "qwen-turbo", "qwen-turbo-latest", "qwen-plus", "qwen-plus-latest",
               "qwen-max", "qwen-max-latest", 'qwq-plus', "qwq-plus-latest", "qwen-plus-2025-04-28",
               "tongyi-intent-detect-v3", "abab6.5t-chat", 'abab6.5s-chat',
               "deepseek-r1-distill-qwen-1.5b", "deepseek-r1-distill-qwen-7b",
               "deepseek-r1-distill-qwen-14b", "deepseek-r1-distill-qwen-32b",
               "deepseek-r1-distill-llama-8b", "deepseek-r1-distill-llama-70b",
               "deepseek-v3", "deepseek-r1", ],
     # "qwen-math-plus",'baichuan2-7b-chat-v1','baichuan2-turbo'
     "generation": ["qwen-coder-turbo", "qwen2.5-coder-7b-instruct", "qwen2.5-coder-14b-instruct",
                    "qwen2.5-coder-32b-instruct", "qwen-coder-turbo-latest"],
     'embedding': ["text-embedding-v2", "multimodal-embedding-v1", "text-embedding-v1", "text-embedding-v2",
                   "text-embedding-v3", "text-embedding-v4"],
     'speech': ['paraformer-v1', 'paraformer-8k-v1', 'paraformer-mtl-v1'],
     'url': 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
     'base_url': "https://dashscope.aliyuncs.com/compatible-mode/v1",
     'supported_openai': True, 'supported_list': True, "timeout": 100,
     'generation_url': 'https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation',
     'embedding_url': 'https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding',
     },
    # https://platform.moonshot.cn/console/api-keys
    {'name': 'moonshot', 'type': 'default', 'api_key': '',
     "model": ["moonshot-v1-auto", "moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k", "kimi-latest"],
     'url': "https://api.moonshot.cn/v1/chat/completions", 'base_url': "https://api.moonshot.cn/v1",
     'file_url': "https://api.moonshot.cn/v1/files",
     'supported_openai': True, 'supported_list': True, "timeout": 100},
    # https://open.bigmodel.cn/console/overview
    {'name': 'glm', 'type': 'default', 'api_key': '',
     "model": ["glm-4-air", "glm-4-flash", "glm-4-flashx", "glm-4v-flash", "glm-4-air", "glm-4-airx", 'glm-4-plus',
               "glm-4-long", "glm-z1-air", "glm-z1-airx", "glm-z1-flash", "glm-4-air-250414",
               "codegeex-4", "glm-4", "glm-4v", "glm-4-9b", "glm-3-turbo"],  # "glm-4-assistant"
     #  "glm-4-0520"
     "embedding": ["embedding-2", "embedding-3"],
     "reranker": ["rerank"],
     'url': 'https://open.bigmodel.cn/api/paas/v4/chat/completions',
     'base_url': "https://open.bigmodel.cn/api/paas/v4/",
     'embedding_url': 'https://open.bigmodel.cn/api/paas/v4/embeddings',
     'reranker_url': "https://open.bigmodel.cn/api/paas/v4/rerank",
     'tool_url': "https://open.bigmodel.cn/api/paas/v4/tools",
     'supported_openai': True, 'supported_list': False, "timeout": 100},
    # https://platform.baichuan-ai.com/docs/api
    {'name': 'baichuan', 'type': 'default', 'api_key': '',
     "model": ['Baichuan3-Turbo', "Baichuan2-Turbo", 'Baichuan3-Turbo', 'Baichuan3-Turbo-128k', 'Baichuan3-Turbo-128k',
               "Baichuan4", 'Baichuan4-Turbo', 'Baichuan4-Air',
               "Baichuan-NPC-Turbo"],
     "embedding": ["Baichuan-Text-Embedding"],
     'url': 'https://api.baichuan-ai.com/v1/chat/completions',
     'base_url': "https://api.baichuan-ai.com/v1/",  # assistants,files,threads
     'assistants_url': 'https://api.baichuan-ai.com/v1/assistants',
     'embedding_url': 'https://api.baichuan-ai.com/v1/embeddings',
     'supported_openai': True, 'supported_list': True, "timeout": 100},
    # https://platform.lingyiwanwu.com/docs/api-reference#api-keys
    {'name': 'lingyiwanwu', 'type': 'default', 'api_key': '',
     "model": ['yi-lightning', 'yi-medium', 'yi-large', 'yi-large-fc', 'yi-spark'],
     "embedding": ["Baichuan-Text-Embedding"],
     'url': 'https://api.lingyiwanwu.com/v1/chat/completions',
     'base_url': "https://api.lingyiwanwu.com/v1",
     'supported_openai': True, 'supported_list': True, "timeout": 100},
    # https://api-docs.deepseek.com/zh-cn/
    {'name': 'deepseek', 'type': 'default', 'api_key': '',
     'model': ["deepseek-chat", "deepseek-reasoner"],  # DeepSeek-V3,DeepSeek-R1
     'url': 'https://api.deepseek.com/chat/completions',
     'generation_url': "https://api.deepseek.com/beta",  # https://api.deepseek.com/beta/completions
     'base_url': 'https://api.deepseek.com',
     'supported_openai': True, 'supported_list': True, "timeout": 300},  # /v1
    # https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Zm2ycv77m
    # https://cloud.baidu.com/doc/WENXINWORKSHOP/s/mlm0nonsv
    {'name': 'ernie', 'type': 'default', 'api_key': '',
     # "ERNIE-4.0-8K", "ERNIE-3.5-8K", "ERNIE-4.0-8K","ERNIE-4.0-8K-Preview", "ERNIE-4.0-8K-Latest","ERNIE-3.5-128K"
     "model": ["ernie-4.0-8k-latest", 'ernie-speed-8k', "ernie-4.0-8k-preview", "ernie-4.0-turbo-8k",
               'ernie-4.0-turbo-8k', 'ernie-4.0-turbo-128k', 'ernie-speed-pro-128k',
               'deepseek-v3', 'deepseek-r1',
               'deepseek-r1-distill-qwen-14b', 'deepseek-r1-distill-qwen-32b',
               ],
     'embedding': ['tao-8k', 'embedding-v1', 'bge-large-zh', 'bge-large-en'],
     'reranker': ['bce-reranker-base'],
     'url': 'https://qianfan.baidubce.com/v2/chat/completions',
     'embedding_url': 'https://qianfan.baidubce.com/v2/embeddings',
     'reranker_url': "https://qianfan.baidubce.com/v2/rerankers",
     'base_url': "https://qianfan.baidubce.com/v2",
     'supported_openai': True, 'supported_list': False, "timeout": 100
     },
    # https://cloud.baidu.com/doc/WENXINWORKSHOP/s/clntwmv7t
    # https://console.bce.baidu.com/qianfan/ais/console/onlineService
    # https://console.bce.baidu.com/ai/#/ai/nlp/overview/index
    {'name': 'baidu', 'type': 'baidu', 'api_key': '',
     'model': ['completions_pro', 'qianfan-agent-lite-8k', 'qianfan-agent-speed-8k', 'qianfan-agent-speed-32k',
               'qianfan-dynamic-8k',
               'ernie-lite-8k', 'ernie_speed', 'ernie-novel-8k', 'ernie-speed-128k',
               'qianfan_chinese_llama_2_7b', 'qianfan_chinese_llama_2_13b', 'qianfan_chinese_llama_2_70b',
               'llama_2_7b', 'llama_2_13b', 'llama_2_70b', 'llama_3_8b',
               'gemma_7b_it', 'mixtral_8x7b_instruct', 'aquilachat_7b',
               'bloomz_7b1', 'qianfan_bloomz_7b_compressed',
               'chatglm2_6b_32k', 'baichuan2-7b-chat', 'baichuan2-13b-chat',
               'Fuyu-8B', 'yi_7b_chat', 'yi_34b_chat',
               ],  # 'chatglm3-6b', 'chatglm3-6b-32k',
     "generation": ['sqlcoder_7b', 'CodeLlama-7b-Instruct', 'Yi-34B'],
     'embedding': ['bge_large_zh'],
     "nlp": ["txt_mone", "address", "simnet", "word_emb_sim", "ecnet", "text_correction", "keyword", "topic"],
     'url': "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/",
     # 'https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/{model}'.format(model)
     'generation_url': "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/completions",
     'embedding_url': 'https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/embeddings',
     'nlp_url': "https://aip.baidubce.com/rpc/2.0/nlp/v1/",
     'base_url': '', 'supported_openai': False, 'supported_list': False},
    # https://jina.ai/reader/
    # https://jina.ai/api-dashboard/embedding
    {'name': 'jina', 'type': 'default', 'api_key': '',
     "model": ["jina-deepsearch-v1"],
     "embedding": ["jina-clip-v2", "jina-embeddings-v3", "jina-embeddings-v2-base-code", "jina-colbert-v2"],
     "reranker": ["jina-reranker-v2-base-multilingual", "jina-reranker-m0", 'jina-colbert-v2', ],
     'url': 'https://deepsearch.jina.ai/v1/chat/completions',
     'base_url': "https://r.jina.ai",
     'embedding_url': 'https://api.jina.ai/v1/embeddings',
     'reranker_url': "https://api.jina.ai/v1/rerank",
     'classify_url': "https://api.jina.ai/v1/classify",
     'segment_url': 'https://api.jina.ai/v1/segment',
     'supported_openai': False, 'supported_list': False, 'proxy': True, "timeout": 200},
    # https://console.volcengine.com/ark/region:ark+cn-beijing/endpoint?config=%7B%7D
    # https://console.volcengine.com/ark/region:ark+cn-beijing/model?vendor=Bytedance&view=LIST_VIEW
    {'name': 'doubao', 'type': 'default', 'api_key': '',
     "model": {"doubao-lite-32k": 'ep-20241206154509-gwsp9',
               "doubao-1.5-lite-32k-250115": "doubao-1-5-lite-32k-250115",
               "doubao-1.5-thinking-pro-250415": "doubao-1-5-thinking-pro-250415",
               "doubao-1.5-thinking-pro-m-250428": "doubao-1-5-thinking-pro-m-250428",
               "doubao-1.5-pro-32k-250115": "doubao-1-5-pro-32k-250115",
               "doubao-1.5-pro-256k-250115": "doubao-1-5-pro-256k-250115",
               'doubao-1-5-pro-32k-250115': 'ep-20240919160119-7rbsn',
               "doubao-pro-32k": 'ep-20241018103141-7hqr7',
               "doubao-pro-32k-browsing-241115": "doubao-pro-32k-browsing-241115",
               "doubao-pro-32k-character-241215": "doubao-pro-32k-character-241215",
               "doubao-pro-32k-functioncall-241028": "doubao-pro-32k-functioncall-241028",
               "doubao-pro-32k-functioncall-preview": 'ep-20241018103141-fwpjd',
               # The model or endpoint doubao-pro-32k does not exist or you do not have access to it
               "doubao-pro-256k-241115": "ep-m-20250416135143-wb2hz",
               'doubao-pro-128k': 'ep-20240919161410-7k5d8',
               "doubao-character-pro-32k": 'ep-20241206120328-msvt7',
               "chatglm3-130-fin": 'ep-20241017110248-fr7z6',
               "chatglm3-130b-fc": 'ep-20241017105930-drfm8',
               "doubao-vision-lite-32k": "ep-20241219174540-rdlfj", "doubao-vision-pro-32k": "ep-20241217182411-kdg49"},
     # [ "GLM3-130B",chatglm3-130-fin,functioncall-preview],
     'embedding': {'Doubao-embedding': 'ep-20241219165520-lpqrl', 'Doubao-embedding-large': 'ep-20241219165636-kttk2',
                   "doubao-embedding-vision-250328": "doubao-embedding-vision-250328"},
     'url': 'https://ark.cn-beijing.volces.com/api/v3/chat/completions',
     'tokenization_url': 'https://ark.cn-beijing.volces.com/api/v3/tokenization',
     'base_url': "https://ark.cn-beijing.volces.com/api/v3",
     'supported_openai': True, 'supported_list': False, "timeout": 100},
    # open.volcengineapi.com
    # https://cloud.tencent.com/document/product/1729
    # https://cloud.tencent.com/document/product/1729/104753
    {'name': 'hunyuan', 'type': 'tencent', 'api_key': '',
     'model': ['hunyuan-standard', "hunyuan-lite", "hunyuan-pro", "hunyuan-turbo", "hunyuan-large",
               "hunyuan-large-longcontext", "hunyuan-code", 'hunyuan-role', 'hunyuan-functioncall', 'hunyuan-vision',
               "hunyuan-turbo-latest", 'hunyuan-turbos-latest'],
     'embedding': ['hunyuan-embedding'],
     'url': 'https://hunyuan.tencentcloudapi.com',  # 'hunyuan.ap-shanghai.tencentcloudapi.com'
     'base_url': "https://api.hunyuan.cloud.tencent.com/v1",
     'embedding_url': "https://api.hunyuan.cloud.tencent.com/v1/embeddings",
     'nlp_url': "nlp.tencentcloudapi.com", 'ocr_url': 'ocr.tencentcloudapi.com',
     'supported_openai': True, 'supported_list': True, "timeout": 100},
    # https://cloud.siliconflow.cn/playground/chat
    {'name': 'silicon', 'type': 'default', 'api_key': '',
     'model': ["Qwen/Qwen2-7B-Instruct", "Qwen/Qwen2-1.5B-Instruct",
               "Qwen/Qwen2.5-Coder-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct", 'Qwen/Qwen2.5-Coder-32B-Instruct',
               "Qwen/Qwen2.5-72B-Instruct",
               'Qwen/QwQ-32B-Preview', 'Qwen/QwQ-32B',
               "Qwen/Qwen3-8B", "Qwen/Qwen3-14B", "Qwen/Qwen3-32B", "Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-235B-A22B",
               "THUDM/chatglm3-6b", "THUDM/glm-4-9b-chat", "Pro/THUDM/glm-4-9b-chat",
               "THUDM/GLM-Z1-9B-0414", "THUDM/GLM-4-9B-0414", "THUDM/GLM-Z1-32B-0414", "THUDM/GLM-4-32B-0414",
               "THUDM/GLM-Z1-Rumination-32B-0414",
               "deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-R1",
               "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
               "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
               "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
               "Pro/deepseek-ai/DeepSeek-V3",
               "internlm/internlm2_5-7b-chat", "Pro/internlm/internlm2_5-7b-chat", "internlm/internlm2_5-20b-chat",
               "Pro/OpenGVLab/InternVL2-8B",
               ],
     # "deepseek-ai/DeepSeek-V2.5", "deepseek-ai/deepseek-vl2","deepseek-ai/DeepSeek-V2-Chat","Qwen/Qwen1.5-7B-Chat", "Qwen/Qwen1.5-32B-Chat"
     # "01-ai/Yi-1.5-9B-Chat-16K",  'TeleAI/TeleChat2',"google/gemma-2-9b-it", "meta-llama/Meta-Llama-3-8B-Instruct"
     # https://huggingface.co/BAAI
     'embedding': ['BAAI/bge-large-zh-v1.5', "BAAI/bge-large-en-v1.5", 'BAAI/bge-m3',
                   'netease-youdao/bce-embedding-base_v1', 'Pro/BAAI/bge-m3'],
     'generation': ['Qwen/Qwen2.5-Coder-7B-Instruct', "deepseek-ai/DeepSeek-V2.5",
                    'deepseek-ai/DeepSeek-Coder-V2-Instruct'],
     'reranker': ['BAAI/bge-reranker-v2-m3', 'netease-youdao/bce-reranker-base_v1', 'Pro/BAAI/bge-reranker-v2-m3'],
     # "BAAI/bge-reranker-large","BAAI/bge-reranker-base"
     'speech': ['FunAudioLLM/SenseVoiceSmall', ],
     'image': ['stabilityai/stable-diffusion-3-medium', 'stabilityai/stable-diffusion-3-5-large',
               'black-forest-labs/FLUX.1-schnell', 'black-forest-labs/FLUX.1-dev',
               'deepseek-ai/Janus-Pro-7B',
               "Wan-AI/Wan2.1-T2V-14B", "Wan-AI/Wan2.1-T2V-14B-Turbo", "Wan-AI/Wan2.1-I2V-14B-720P",
               "Wan-AI/Wan2.1-I2V-14B-720P-Turbo", ],
     'url': 'https://api.siliconflow.cn/v1/chat/completions',
     'base_url': 'https://api.siliconflow.cn/v1',
     'embedding_url': "https://api.siliconflow.cn/v1/embeddings",
     'reranker_url': "https://api.siliconflow.cn/v1/rerank",
     'image_url': 'https://api.siliconflow.cn/v1/images/generations',
     'speech_url': "https://api.siliconflow.cn/v1/uploads/audio/voice",
     'supported_openai': True, 'supported_list': True, "timeout": 100},

    # https://console.xfyun.cn/services/sparkapiCenter
    # https://www.xfyun.cn/doc/spark/HTTP%E8%B0%83%E7%94%A8%E6%96%87%E6%A1%A3.html#_3-%E8%AF%B7%E6%B1%82%E8%AF%B4%E6%98%8E
    {'name': 'spark', 'type': 'default', 'api_key': [],
     'model': ['lite', 'max-32k', 'pro-128k', '4.0Ultra', 'generalv3', 'generalv3.5', "x1"],
     'url': 'https://spark-api-open.xf-yun.com/v1/chat/completions',
     'base_url': 'https://spark-api-open.xf-yun.com/v2/',
     'file_url': 'https://spark-api-open.xf-yun.com/v1/files',
     'embedding_url': 'https://emb-cn-huabei-1.xf-yun.com/',
     'translation_url': 'https://itrans.xf-yun.com/v1/its',
     'ws_url': 'wss://spark-api.xf-yun.com/v3.5/chat',
     'supported_openai': False, 'supported_list': False, "timeout": 100
     },
    # https://platform.minimaxi.com/document/ChatCompletion%20v2?key=66701d281d57f38758d581d0
    {'name': 'minimax', 'type': 'default', 'api_key': '',
     'model': ["MiniMax-Text-01", 'abab6.5s-chat', 'DeepSeek-R1'],
     'speech': ['speech-02-hd', 'speech-01-hd', 'speech-02-turbo', 'speech-01-turbo'],
     'image': ['image-01'],
     'url': 'https://api.minimax.chat/v1/text/chatcompletion_v2',
     'speech_url': 'https://api.minimax.chat/v1/t2a_v2',  # t2a_async_v2
     'base_url': 'https://api.minimax.chat/v1',
     'supported_openai': True, 'supported_list': False},
    # https://docs.mistral.ai/api/#tag/chat/operation/chat_completion_v1_chat_completions_post
    {'name': 'mistral', 'type': 'default', 'api_key': '',
     'model': ["mistral-small-latest", 'open-mistral-nemo', 'open-codestral-mamba', 'codestral-latest',
               'pixtral-12b-2409'],
     'embedding': ["mistral-embed"],
     'url': 'https://api.mistral.ai/v1/chat/completions',  # https://codestral.mistral.ai/v1/chat/completions
     'agent_url': 'https://api.mistral.ai/v1/agents/completions',
     'embedding_url': 'https://api.mistral.ai/v1/embeddings',
     'base_url': 'https://api.mistral.ai/v1',
     'supported_openai': True, 'supported_list': True, 'proxy': True, "timeout": 200},
    # https://ai.google.dev/gemini-api/docs/openai?hl=zh-cn#python
    {'name': 'gemini', 'type': 'default', 'api_key': '',
     'model': ["gemini-1.5-flash", 'gemini-1.5-flash-8b', 'gemini-1.5-pro',
               'gemini-2.0-flash-lite', 'gemini-2.0-flash'],  # 'aqa'
     'embedding': ['text-embedding-004', 'gemini-embedding-exp'],
     'url': 'https://generativelanguage.googleapis.com/v1beta/models/',
     'base_url': "https://generativelanguage.googleapis.com/v1beta/openai/",
     'supported_openai': True, 'supported_list': True, 'proxy': True},
    # https://docs.x.ai/docs/overview
    {'name': 'grok', 'type': 'default', 'api_key': '',
     'model': ["grok-2-latest", "grok-2-vision-1212", "grok-3-mini", "grok-3-mini-fast", "grok-3", "grok-3-fast",
               "grok-4-0709"],
     'url': 'https://api.x.ai/v1/chat/completions',
     'base_url': "https://api.x.ai/v1",
     'supported_openai': True, 'supported_list': True, 'proxy': True, "timeout": 200},
    # https://modelscope.cn/my/modelService/deploy?page=1&type=free
    {'name': 'modelscope', 'type': 'default', 'api_key': '',
     'model': ["LLM-Research/c4ai-command-r-plus-08-2024",
               "mistralai/Mistral-Small-Instruct-2409", "mistralai/Ministral-8B-Instruct-2410",
               "mistralai/Mistral-Large-Instruct-2407",
               "Qwen/Qwen2.5-Coder-32B-Instruct", "Qwen/Qwen2.5-Coder-14B-Instruct", "Qwen/Qwen2.5-Coder-7B-Instruct",
               "Qwen/Qwen2.5-72B-Instruct", "Qwen/Qwen2.5-32B-Instruct", "Qwen/Qwen2.5-14B-Instruct",
               "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2-VL-7B-Instruct", "Qwen/Qwen2.5-VL-32B-Instruct",
               "Qwen/QwQ-32B-Preview", "Qwen/QwQ-32B", "Qwen/QVQ-72B-Preview",
               "opencompass/CompassJudger-1-32B-Instruct",
               "LLM-Research/Meta-Llama-3.1-405B-Instruct", "LLM-Research/Meta-Llama-3.1-8B-Instruct",
               "LLM-Research/Meta-Llama-3.1-70B-Instruct", "LLM-Research/Llama-3.3-70B-Instruct",
               "Qwen/Qwen2.5-14B-Instruct-1M", "Qwen/Qwen2.5-7B-Instruct-1M", "Qwen/Qwen2.5-VL-3B-Instruct",
               "Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2.5-VL-72B-Instruct",
               "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
               "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
               "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
               "deepseek-ai/DeepSeek-R1", "deepseek-ai/DeepSeek-V3",
               "XGenerationLab/XiYanSQL-QwenCoder-32B-2412", "XGenerationLab/XiYanSQL-QwenCoder-32B-2504",
               "deepseek-ai/DeepSeek-V3-0324",
               "Wan-AI/Wan2.1-T2V-1.3B",
               "LLM-Research/Llama-4-Scout-17B-16E-Instruct", "LLM-Research/Llama-4-Maverick-17B-128E-Instruct",
               "Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B", "Qwen/Qwen3-14B",
               "Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-32B", "Qwen/Qwen3-235B-A22B"
               ],
     # 'unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF'
     'url': 'https://ms-fc-283a6661-de97.api-inference.modelscope.cn/v1',
     'base_url': "https://api-inference.modelscope.cn/v1",
     'supported_openai': True, 'supported_list': True, 'proxy': False, "timeout": 100},
    # https://docs.aihubmix.com/cn/index
    {'name': 'aihubmix', 'type': 'default', 'api_key': '',
     'model': ["gpt-4o-mini", "gpt-4.1-mini", "gpt-3.5-turbo", 'gpt-4.1-nano', 'o3-mini', 'o4-mini',
               "gemini-2.0-flash-exp", "gemini-2.5-flash-preview-05-20", 'gemini-2.0-flash',
               "jina-deepsearch-v1", "aihubmix-command-r-08-2024", "aihubmix-Cohere-command-r",
               "Qwen/Qwen3-30B-A3B", 'Qwen/Qwen3-8B', "Qwen/Qwen3-32B", "qwen2.5-vl-72b-instruct",
               'qwen2.5-math-72b-instruct', "Qwen/QwQ-32B", "qwen-long",
               "grok-3-mini", "grok-3-mini-fast-beta", "grok-3",
               "kimi-latest", "yi-large-turbo", "yi-34b-chat-200k",
               "THUDM/GLM-4-32B-0414", "THUDM/GLM-4-32B-0414", "THUDM/GLM-Z1-32B-0414",
               "Doubao-1.5-pro-32k", "Doubao-pro-128k", "nvidia/llama-3.1-nemotron-70b-instruct",
               "claude-3-5-sonnet-20241022", "claude-3-7-sonnet-20250219", "claude-sonnet-4-20250514"],
     'embedding': ["jina-clip-v2", "jina-colbert-v2", 'jina-embeddings-v3', "jina-embeddings-v2-base-code",
                   "gemini-embedding-exp-03-07", "text-embedding-3-large"],
     'url': 'https://aihubmix.com/v1/chat/completions',
     'embedding_url': 'https://aihubmix.com/v1/embeddings',
     'reranker_url': 'https://aihubmix.com/v1/rerank',
     'base_url': "https://aihubmix.com/v1",
     'supported_openai': True, 'supported_list': True, 'proxy': False, "timeout": 200},
    # https://docs.tokenflux.ai/quickstart
    # https://tokenflux.ai/models
    {'name': 'tokenflux', 'type': 'default', 'api_key': '',
     'model': ['gemini-pro', 'gemini-2.5-pro', 'gemma-3-12b', 'gemma-3-27b',
               'claude-3.5-haiku', 'claude-3-7-sonnet', 'claude-sonnet-4', 'grok-3-mini-beta',
               'deepseek-r1', 'deepseek-v3', 'doubao-1.5-pro-32k', 'glm-4-9b-chat',
               'gpt-4o-mini', 'gpt-4.1-mini', 'gpt-4.1-nano', 'gpt-4.1'],
     'embedding': ["BAAI bge-m3", 'text-embedding-3-large'],
     'url': 'https://aihubmix.com/v1/chat/completions',
     'embedding_url': 'https://aihubmix.com/v1/embeddings',
     'reranker_url': 'https://aihubmix.com/v1/rerank',
     'mcp_url': 'https:/tokenflux.ai/v1/mcps',
     'base_url': "https://tokenflux.ai/v1",
     'supported_openai': True, 'supported_list': True, 'proxy': False, "timeout": 200},
    # https://docs.anthropic.com/zh-CN/api/overview#python
    # https://docs.anthropic.com/zh-CN/docs/about-claude/models/overview
    {'name': 'claude', 'type': 'anthropic', 'api_key': '',
     'model': ["claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022", "claude-3-7-sonnet-20250219",
               "claude-sonnet-4-20250514", "claude-opus-4-20250514",
               "claude-3-5-sonnet-latest", "claude-3-7-sonnet-latest"],
     'embedding': ["BAAI bge-m3", 'text-embedding-3-large'],
     'generation': ['claude-2'],
     'url': 'https://api.anthropic.com/v1/messages',
     'generation_url': 'https://api.anthropic.com/v1/complete',
     'base_url': 'https://api.anthropic.com/v1',
     'supported_openai': False, 'supported_list': False, 'proxy': True, "timeout": 200},
    # https://platform.openai.com/docs/overview
    {'name': 'gpt', 'type': 'default', 'api_key': '',
     'model': ["o3-mini", "o3-mini-2025-01-31", "o1", "o1-2024-12-17", "o1-preview", "o1-preview-2024-09-12", "o1-mini",
               "o1-mini-2024-09-12", "gpt-4o", "gpt-4o-2024-11-20", "gpt-4o-2024-08-06", "gpt-4o-2024-05-13",
               "gpt-4o-audio-preview", "gpt-4o-audio-preview-2024-10-01", "gpt-4o-audio-preview-2024-12-17",
               "gpt-4o-mini-audio-preview", "gpt-4o-mini-audio-preview-2024-12-17", "chatgpt-4o-latest", "gpt-4o-mini",
               "gpt-4o-mini-2024-07-18", "gpt-4-turbo", "gpt-4-turbo-2024-04-09", "gpt-4-0125-preview",
               "gpt-4-turbo-preview", "gpt-4-1106-preview", "gpt-4-vision-preview", "gpt-4", "gpt-4-0314", "gpt-4-0613",
               "gpt-4-32k", "gpt-4-32k-0314", "gpt-4-32k-0613", "gpt-3.5-turbo", "gpt-3.5-turbo-16k",
               "gpt-3.5-turbo-0301", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-1106", "gpt-3.5-turbo-0125",
               "gpt-3.5-turbo-16k-0613"],
     'generation': ["text-davinci-003", "text-davinci-002", "text-davinci-003", "text-davinci-004"],
     'embedding': ["text-embedding-ada-002", "text-search-ada-doc-001", "text-similarity-babbage-001",
                   "code-search-ada-code-001", "search-babbage-text-001"],
     'url': 'https://api.openai.com/v1/chat/completions',
     'embedding_url': 'https://api.openai.com/v1/embeddings',
     'base_url': "https://api.openai.com/v1",
     'ws_url': 'wss://api.openai.com/v1/realtime',
     'supported_openai': True, 'supported_list': True, 'proxy': True, "timeout": 200,
     },

]


# moonshot,glm,qwen,ernie,hunyuan,doubao,silicon,spark,baichuan,deepseek
def model_api_keys(name: str = None):
    api_keys = {
        'moonshot': Config.Moonshot_Service_Key,
        'glm': Config.GLM_Service_Key,
        'qwen': Config.DashScope_Service_Key,
        'doubao': Config.ARK_Service_Key,
        'spark': Config.XF_API_Password,
        'ernie': Config.QIANFAN_Service_Key,
        'baichuan': Config.Baichuan_Service_Key,
        'lingyiwanwu': Config.Lingyiwanwu_Service_Key,
        'hunyuan': Config.TENCENT_Service_Key,
        'deepseek': Config.DeepSeek_Service_Key,
        'minimax': Config.MINIMAX_Service_Key,
        'mistral': Config.MISTRAL_API_KEY,
        'jina': Config.JINA_Service_Key,
        'gemini': Config.GEMINI_Service_Key,
        'grok': Config.Grok_Service_Key,
        'claude': Config.Anthropic_Service_Key,
        # 'gpt': Config.OPENAI_Service_Key
        'silicon': Config.Silicon_Service_Key,
        'modelscope': Config.ModelScope_Service_Key,
        'aihubmix': Config.AiHubMix_Service_Key,
        'tokenflux': Config.TokenFlux_Service_Key,
    }
    if not name:
        return api_keys
    api_key = api_keys.get(name, None)
    if isinstance(api_key, str) and set(api_key) == {"*"}:
        return None
    return api_key


# SUPPORTED_OPENAI_MODELS = {'moonshot', 'glm', 'qwen', 'hunyuan', 'silicon', 'doubao', 'baichuan', 'deepseek', 'minimax',
#                            'mistral', 'gemini'}


def extract_ai_model(search_field: str = "model"):
    """
    提取 AI_Models 中的 name 以及 search_field 中的所有值（列表或字典 key）。

    参数：
    - search_field: 需要提取的字段名称，默认为 'model'

    返回：
    - List[Tuple[str, List[str]]]: 每个模型的名称及其对应的模型列表
    """
    extracted_data = []

    for model in AI_Models:
        name = model["name"]
        field_value = model.get(search_field, [])
        if model.get('supported_openai', True) and not model.get('api_key'):
            continue

        if isinstance(field_value, list):
            extracted_data.append((name, list(dict.fromkeys(field_value))))
        elif isinstance(field_value, dict):
            extracted_data.append((name, list(field_value.keys())))
        else:
            extracted_data.append((name, [field_value]))

    return extracted_data


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


# sha256 非对称加密
def hmac_sha256(key: bytes, content: str):
    """生成 HMAC-SHA256 签名"""
    return hmac.new(key, content.encode("utf-8"), digestmod=hashlib.sha256).digest()


# sha256 hash
def hash_sha256(content: str):
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


def token_split(auth: str):
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
    date = datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d")

    # ************* 步骤 1：拼接规范请求串 *************
    http_request_method = "POST"
    canonical_uri = "/"
    canonical_querystring = ""
    ct = "application/json; charset=utf-8"
    canonical_headers = f"content-type:{ct}\nhost:{host}\nx-tc-action:{action.lower()}\n"
    signed_headers = "content-type;host;x-tc-action"

    payload = json.dumps(body)
    hashed_request_payload = hash_sha256(payload)
    canonical_request = "\n".join([http_request_method, canonical_uri, canonical_querystring,
                                   canonical_headers, signed_headers, hashed_request_payload])

    # ************* 步骤 2：拼接待签名字符串 *************
    credential_scope = f"{date}/{service}/tc3_request"
    hashed_canonical_request = hash_sha256(canonical_request)
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
        "X-TC-Region": region  # region,"<区域>",
    }
    return headers


def parse_database_uri(uri):
    parsed = urlparse(uri)
    query = parse_qs(parsed.query)
    # parsed.scheme  # e.g., mysql+aiomysql
    return {
        "host": parsed.hostname or 'localhost',
        "port": parsed.port or 3306,
        "user": unquote_plus(parsed.username),
        "password": unquote_plus(parsed.password),
        "db_name": parsed.path.lstrip('/'),  # 去掉前面的 /,parsed.path[1:]
        "charset": query.get("charset", ["utf8mb4"])[0]
    }


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
    to_encode = data.copy()  # 可以携带更丰富的 payload
    expires_delta = timedelta(minutes=expires_minutes or Config.ACCESS_TOKEN_EXPIRE_MINUTES)
    expire = datetime.utcnow() + expires_delta  # datetime.utcnow() + timedelta(days=7)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, Config.SECRET_KEY, algorithm=Config.ALGORITHM)  # encoded_jwt


# 验证和解码 Token,Access Token 有效性，并返回 payload
def verify_access_token(token: str) -> str | None:
    try:
        # rsa.verify(original_message, signed_message, public_key)
        payload = jwt.decode(token, Config.SECRET_KEY, algorithms=[Config.ALGORITHM])
        username: str = payload.get("sub") or payload.get('user_id')
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
    from openai import OpenAI

    # for backward compatibility, you can still use `https://api.deepseek.com/v1` as `base_url`.
    # cx = OpenAI(api_key=Config.DeepSeek_Service_Key, base_url="https://api.deepseek.com")
    # print(cx.models.list().model_dump_json())

    print(Config.save())

    # from utils import backup_to_webdev
    # import asyncio, nest_asyncio
    #
    # nest_asyncio.apply()
    # backup_id = asyncio.run(
    #     backup_to_webdev(Config._config_path, api_url='http://10.10.10.3:8090', username='dooven',
    #                      password='***'))
    # print(backup_id)

    Config.mask_sensitive()

    Config.load()
    print(Config.get_config_data())
