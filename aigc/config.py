from urllib.parse import quote_plus
from datetime import datetime
import os
from dataclasses import dataclass, asdict


@dataclass(kw_only=True)  # , frozen=True
class Config(object):
    """
       全局参数配置（示例结构），涉及配置中的敏感信息，真实环境用从 YAML文件加载配置覆盖，以下为占位内容，仅供结构参考，非真实配置
    """
    # {{!IGNORE_START!}} (请忽略以下内容)
    DATABASE_PWD = "***"
    SQLALCHEMY_DATABASE_URI = f'mysql+pymysql://xxx:{quote_plus(DATABASE_PWD)}@rm-xxx.mysql.rds.aliyuncs.com:3306/technet?charset=utf8mb4'
    ASYNC_SQLALCHEMY_DATABASE_URI = f'mysql+aiomysql://xxx:{quote_plus(DATABASE_PWD)}@rm-xxx.mysql.rds.aliyuncs.com:3306/technet?charset=utf8mb4'
    IDEATECH_SQLALCHEMY_DATABASE_URI = f'mysql+aiomysql://xxx:{quote_plus(DATABASE_PWD)}@rm-xxx.mysql.rds.aliyuncs.com:3306/h3yun?charset=utf8'

    SQLALCHEMY_COMMIT_ON_TEARDOWN = False
    SQLALCHEMY_TACK_MODIFICATIONS = True
    SQLALCHEMY_ECHO = True
    SECRET_KEY = '***'
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 60
    VERIFY_TIMEOUT_SEC = 300
    HTTP_TIMEOUT_SEC = 100.0
    LLM_TIMEOUT_SEC = 300.0
    MAX_TASKS = 1024
    MAX_CACHE = 1024
    MAX_CONCURRENT = 100
    MAX_HTTP_CONNECTIONS = 300
    MAX_KEEPALIVE_CONNECTIONS = 100
    MAX_RETRY_COUNT = 3
    RETRY_DELAY = 5
    DEVICE_ID = 'IusetTHaVvDmji5odDRZBQIhcTvcGWs6'
    INFURA_PROJECT_ID = ''
    DATA_FOLDER = 'data'
    Version = 'v1.3.0'
    _config_path = 'config.yaml'
    __config_data = {}  # 动态加载的数据，用于还原
    __config_dynamic = {}  # 其他默认配置项，用于运行时
    WEBUI_NAME = 'aigc'
    WEBUI_URL = 'http://xxx:7000'
    # https://api.qdrant.tech/api-reference
    OLLAMA_HOST = 'localhost'
    NEO_URI = "bolt://neo4j:7687"
    NEO_Username = "neo4j"
    NEO_Password = '***'
    DASK_Cluster = 'tcp://127.0.0.1:8786'
    DASK_DASHBOARD_HOST = "http://127.0.0.1:8787"
    QDRANT_HOST = 'qdrant' 
    QDRANT_GRPC_PORT = 6334
    QDRANT_URL = "http://xxx:6333"  # ":memory:"
    WECHAT_URL = 'http://idea_ai_robot:28089'
    REDIS_HOST = "redis_aigc"
    REDIS_PORT = 6379  # 7007
    REDIS_CACHE_SEC = 99999  # 86400
    REDIS_MAX_CONCURRENT = 300
    DB_MAX_SIZE = 20

    VALID_API_KEYS = {"token-abc123", "token-def456"}
    DEFAULT_LANGUAGE = 'Chinese'
    DEFAULT_MODEL = 'moonshot'  # 'qwen'
    DEFAULT_MODEL_ENCODING = "gpt-3.5-turbo"
    DEFAULT_MODEL_EMBEDDING = 'BAAI/bge-large-zh-v1.5'
    DEFAULT_MODEL_METADATA = 'qwen:qwen3-coder-plus'  # qwen-coder-turbo
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
    XF_API_Password = ['***']  # http协议的APIpassword
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
    DeepSeek_Service_Keys = ['***']
    # https://platform.minimaxi.com/user-center/basic-information
    MINIMAX_Group_ID = '1887488082316366022'
    MINIMAX_Service_Key = '***'
    # https://platform.lingyiwanwu.com/apikeys
    Lingyiwanwu_Service_Key = '***'
    # https://o3.fan/token
    OThree_Service_Key = "***"
    ZZZ_Service_Key = "***"
    AI147_Service_Key = '***'
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
    SearchApi_Key = '***' 
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
    TAVILY_Api_Key = '***' 
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

    Ideatech_API_Key = '***'
    Ideatech_Host = '***.info' 

    Risk_BASE_URL = "https://risk-qa.ezhanghu.cn"
    Risk_API_KEY = "***"
    Risk_DB_CONFIG = {
        "host": "rm-***.mysql.rds.aliyuncs.com",  # 数据库地址
        "port": 3306,  # 端口
        "user": "***",  # 用户名
        "password": "***",  # 密码
        "database": "***",  # 库名
        "charset": "utf8mb4"
    }

    HTTP_Proxy = 'http://***:***@www.***.***:7930'
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
        cls.ASYNC_SQLALCHEMY_DATABASE_URI = f'mysql+aiomysql://***:***@10.10.10.5:3306/kettle?charset=utf8mb4'
        cls.WEBUI_URL = 'http://127.0.0.1:7000'
        # https://api.qdrant.tech/api-reference
        cls.QDRANT_GRPC_PORT = None
        cls.WECHAT_URL = 'http://***:28089'
        cls.REDIS_HOST = '10.10.10.5'  # "localhost"
        cls.NEO_URI = "bolt://localhost:7687"
        cls.NEO_Password = '***'
        cls.DASK_Cluster = 'tcp://10.10.10.20:8786'
        cls.DASK_DASHBOARD_HOST = "http://host.docker.internal:8787"

        cls.HTTP_Proxy = 'http://***:***@10.10.10.3:7890'
        # cls.HTTP_Proxies = {'http': cls.HTTP_Proxy,'https': cls.HTTP_Proxy}
        os.environ['HTTP_PROXY'] = cls.HTTP_Proxies['http']
        os.environ['HTTPS_PROXY'] = cls.HTTP_Proxies['https']

        cls.Moonshot_Service_Key = "sk-***" 
        cls.DeepSeek_Service_Key = 'sk-***'
        cls.DashScope_Service_Key = 'sk-***'
        cls.ARK_Service_Key = '***'
        cls.__config_dynamic['IS_DEBUG'] = True

    @staticmethod
    def is_invalid(value):
        return isinstance(value, str) and set(value) == {"*"}  # all(s == '*' for s in value)

    @staticmethod
    def norm_version(s):
        return s.lstrip('vV') if s else '0.0.0'

    @classmethod
    def save(cls, filepath=None):
        """将配置项保存到YAML文件"""
        import joblib, yaml
        config_data = {key: getattr(cls, key) for key in dir(cls)
                       if not key.startswith('__')
                       and not key.startswith('_')
                       and not callable(getattr(cls, key))
                       and not isinstance(getattr(cls, key), (classmethod, staticmethod))}

        if not cls.__config_data and any(cls.is_invalid(value) for value in config_data.values()):
            cls.load(filepath)

        config_data.update(cls.__config_data)
        for key, value in config_data.items():
            if cls.is_invalid(value):
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
        load_version = cls.norm_version(cls.__config_data.get('Version'))
        cur_version = cls.norm_version(getattr(cls, 'Version'))
        for key, value in cls.__config_data.items():
            if hasattr(cls, key):
                current_value = getattr(cls, key)
                if current_value is None or cls.is_invalid(current_value):
                    setattr(cls, key, value)
                elif key.endswith('DATABASE_URI'):
                    setattr(cls, key, value)
                elif current_value != value:
                    print(f"配置项 '{key}' 不一致: 当前值 = {current_value}, 加载值 = {value}")
                    if load_version > cur_version:
                        print(f"由于加载版本较新 ({load_version} > {cur_version})，覆盖配置项 '{key}'")
                        setattr(cls, key, value)
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

        if cls.is_invalid(value):
            if not getattr(cls, "__config_data", None):
                cls.load()
            return cls.__config_data.get(key, default)
        return value

    @classmethod
    def update(cls, **kwargs):
        if all(key in cls.__config_data for key in kwargs):
            cls.__config_data.update(**kwargs)
        else:
            for key, value in kwargs.items():
                if hasattr(cls, key):
                    cls.__config_data[key] = value
                else:
                    cls.__config_dynamic[key] = value

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
            return None, None

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
    # https://help.aliyun.com/zh/model-studio/use-qwen-by-calling-api?spm=5176.28197632.console-base_help.dexternal.58192e36ZiOCjR#dc02de8e5earc
    # https://pai.console.aliyun.com/?regionId=cn-shanghai&spm=5176.pai-console-inland.console-base_product-drawer-right.dlearn.337e642duQEFXN&workspaceId=567545#/quick-start/models
    {'name': 'qwen', 'type': 'default', 'api_key': '',
     "model": ["qwen-turbo", "qwen1.5-7b-chat", "qwen1.5-32b-chat", "qwen2-7b-instruct",
               "qwen2.5-32b-instruct", "qwen2.5-72b-instruct",
               "qwen3-8b", "qwen3-14b", "qwen3-32b", "qwen3-coder", "qwen3-coder-plus",
               "qwq-32b", "qwq-32b-preview", "qwen-72b-chat",  # 128K
               "qwen-omni-turbo", "qwen-vl-plus",
               "qwen-coder-plus", "qwen-math-plus",  # 128K,4K
               'qwen-long', 'qwen-long-latest',  # 10M
               "qwen-turbo", "qwen-turbo-latest",  # 1M
               "qwen-plus", "qwen-plus-latest",  # 128K
               "qwen-max", "qwen-max-latest",  # 128K
               "qwen-mt-plus", "qwen-mt-turbo",  # 4K
               'qwq-plus', "qwq-plus-latest", "qwen-plus-2025-04-28",
               "qwen3-235b-a22b", "qwen3-235b-a22b-thinking-2507", "qwen3-235b-a22b-instruct-2507",  # 128K
               "tongyi-intent-detect-v3", "abab6.5t-chat", 'abab6.5s-chat',
               "deepseek-r1-distill-qwen-1.5b", "deepseek-r1-distill-qwen-7b",
               "deepseek-r1-distill-qwen-14b", "deepseek-r1-distill-qwen-32b",  # 32k
               "deepseek-r1-distill-llama-8b", "deepseek-r1-distill-llama-70b",
               "deepseek-v3", "deepseek-r1", "deepseek-v3.1"],
     # 'baichuan2-7b-chat-v1','baichuan2-turbo'
     "generation": ["qwen-coder-turbo", "qwen2.5-coder-7b-instruct", "qwen2.5-coder-14b-instruct",
                    "qwen2.5-coder-32b-instruct", "qwen-coder-turbo-latest"],
     'embedding': ["text-embedding-v2", "multimodal-embedding-v1", "text-embedding-v1", "text-embedding-v2",
                   "text-embedding-v3", "text-embedding-v4"],
     'speech': ['paraformer-v1', 'paraformer-8k-v1', 'paraformer-mtl-v1'],
     'url': 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
     'base_url': "https://dashscope.aliyuncs.com/compatible-mode/v1",
     'supported_openai': True, 'supported_list': True, "timeout": 300,
     'generation_url': 'https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation',
     'embedding_url': 'https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding',
     },
    # https://api-docs.deepseek.com/zh-cn/
    {'name': 'deepseek', 'type': 'default', 'api_key': '',
     'model': ["deepseek-chat", "deepseek-reasoner"],  # DeepSeek-V3 64K,DeepSeek-R1 128K
     'url': 'https://api.deepseek.com/chat/completions',
     'generation_url': "https://api.deepseek.com/beta",  # https://api.deepseek.com/beta/completions
     'base_url': 'https://api.deepseek.com',
     'extra_url': ["https://api.deepseek.com/v3.1_terminus_expires_on_20251015"],
     'supported_openai': True, 'supported_list': True, "timeout": 300},  # /v1
    # https://platform.moonshot.cn/console/api-keys
    {'name': 'moonshot', 'type': 'default', 'api_key': '',
     "model": ["moonshot-v1-auto", "moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k", "kimi-latest"],
     'url': "https://api.moonshot.cn/v1/chat/completions", 'base_url': "https://api.moonshot.cn/v1",
     'file_url': "https://api.moonshot.cn/v1/files",
     'supported_openai': True, 'supported_list': True, "timeout": 300},
    # https://open.bigmodel.cn/console/overview
    # https://docs.bigmodel.cn/cn/guide/start/introduction
    {'name': 'glm', 'type': 'default', 'api_key': '',
     "model": ["glm-4-air", "glm-4-flash", "glm-4-flashx", "glm-4v-flash", "glm-4-air", "glm-4-airx", 'glm-4-plus',
               "glm-4-long",
               "glm-4.5-flash", "glm-4.5-air", "glm-4.5-airx", "glm-4.5", "glm-4.5-x",
               "glm-z1-flash", "glm-z1-air", "glm-z1-airx", "glm-4-air-250414",
               "codegeex-4", "glm-4", "glm-4v", "glm-4-9b", "glm-3-turbo"],  # "glm-4-assistant"
     #  "glm-4-0520"
     "embedding": ["embedding-2", "embedding-3"],
     "reranker": ["rerank"],
     'url': 'https://open.bigmodel.cn/api/paas/v4/chat/completions',
     'base_url': "https://open.bigmodel.cn/api/paas/v4/",
     'embedding_url': 'https://open.bigmodel.cn/api/paas/v4/embeddings',
     'reranker_url': "https://open.bigmodel.cn/api/paas/v4/rerank",
     'realtime_url': "wss://open.bigmodel.cn/api/paas/v4/realtime",
     'speech_url': "https://open.bigmodel.cn/api/paas/v4/audio/speech",
     'image_url': 'https://open.bigmodel.cn/api/paas/v4/images/generations',
     'video_url': 'https://open.bigmodel.cn/api/paas/v4/videos/generations',
     'file_url': "https://open.bigmodel.cn/api/paas/v4/files",
     'tool_url': "https://open.bigmodel.cn/api/paas/v4/tools",
     'agent_url': "https://open.bigmodel.cn/api/v1/agents",
     'supported_openai': True, 'supported_list': False, "timeout": 300},
    # https://console.volcengine.com/ark/region:ark+cn-beijing/endpoint?config=%7B%7D
    # https://console.volcengine.com/ark/region:ark+cn-beijing/model?vendor=Bytedance&view=LIST_VIEW
    # https://www.volcengine.com/docs/82379/1494384
    {'name': 'doubao', 'type': 'default', 'api_key': '',
     "model": ["doubao-1-5-lite-32k-250115", "doubao-1-5-pro-32k-250115", "doubao-1-5-pro-256k-250115",
               "doubao-pro-32k-browsing-241115", "doubao-pro-32k-character-241215",
               "doubao-pro-32k-functioncall-241028", 'doubao-pro-32k-functioncall-preview',
               "doubao-pro-256k-241115",
               "doubao-seed-1-6-flash-250715", "doubao-seed-1-6-thinking-250715", "doubao-seed-1-6-250615",
               "kimi-k2-250905", "deepseek-v3-1-250821", "deepseek-v3-1-terminus",
               ],
     "model_map": {
         # 自定义接入点调用 The model or endpoint doubao-pro-32k does not exist or you do not have access to it
         "doubao-lite-32k": 'ep-20241206154509-gwsp9',
         "doubao-pro-32k": 'ep-20241018103141-7hqr7',
         "doubao-character-pro-32k": 'ep-20241206120328-msvt7',
         "chatglm3-130-fin": 'ep-20241017110248-fr7z6',
         "chatglm3-130b-fc": 'ep-20241017105930-drfm8',
         "doubao-vision-lite-32k": "ep-20241219174540-rdlfj",
         "doubao-vision-pro-32k": "ep-20241217182411-kdg49",
     },
     # [ "GLM3-130B",chatglm3-130-fin,functioncall-preview],
     'embedding': {'Doubao-embedding': 'doubao-embedding-text-240715',  # 'ep-20241219165520-lpqrl'
                   'Doubao-embedding-large': 'doubao-embedding-large-text-250515',  # 'ep-20241219165636-kttk2',
                   "Doubao-embedding-vision": "doubao-embedding-vision-250615"},
     'url': 'https://ark.cn-beijing.volces.com/api/v3/chat/completions',
     'tokenization_url': 'https://ark.cn-beijing.volces.com/api/v3/tokenization',
     'base_url': "https://ark.cn-beijing.volces.com/api/v3",
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
               "THUDM/GLM-Z1-Rumination-32B-0414", "THUDM/GLM-4.1V-9B-Thinking",
               "deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-R1",
               "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
               "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
               "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
               "Pro/deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-V3.1", "deepseek-ai/DeepSeek-V3.1-Terminus",
               "moonshotai/Kimi-K2-Instruct", "moonshotai/Kimi-Dev-72B", "moonshotai/Kimi-K2-Instruct-0905",
               "zai-org/GLM-4.5V", "zai-org/GLM-4.5", "zai-org/GLM-4.5",
               "inclusionAI/Ring-flash-2.0", "ByteDance-Seed/Seed-OSS-36B-Instruct",
               "internlm/internlm2_5-7b-chat", "Pro/internlm/internlm2_5-7b-chat", "internlm/internlm2_5-20b-chat",
               "baidu/ERNIE-4.5-300B-A47B", "tencent/Hunyuan-A13B-Instruct", "tencent/Hunyuan-MT-7B",
               "MiniMaxAI/MiniMax-M1-80k", "stepfun-ai/step3", "ascend-tribe/pangu-pro-moe",
               ],
     # "deepseek-ai/DeepSeek-V2.5", "deepseek-ai/deepseek-vl2","deepseek-ai/DeepSeek-V2-Chat","Qwen/Qwen1.5-7B-Chat", "Qwen/Qwen1.5-32B-Chat"
     # "01-ai/Yi-1.5-9B-Chat-16K",  'TeleAI/TeleChat2',"google/gemma-2-9b-it", "meta-llama/Meta-Llama-3-8B-Instruct"
     # https://huggingface.co/BAAI
     'embedding': ['BAAI/bge-large-zh-v1.5', "BAAI/bge-large-en-v1.5", 'BAAI/bge-m3',
                   'netease-youdao/bce-embedding-base_v1', 'Pro/BAAI/bge-m3',
                   "Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-4B", "Qwen/Qwen3-Embedding-8B"],
     'generation': ['Qwen/Qwen2.5-Coder-7B-Instruct', "deepseek-ai/DeepSeek-V2.5",
                    'deepseek-ai/DeepSeek-Coder-V2-Instruct'],
     'reranker': ['BAAI/bge-reranker-v2-m3', 'netease-youdao/bce-reranker-base_v1', 'Pro/BAAI/bge-reranker-v2-m3',
                  "Qwen/Qwen3-Reranker-0.6B", "Qwen/Qwen3-Reranker-4B", "Qwen/Qwen3-Reranker-8B"],
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
     'supported_openai': True, 'supported_list': True, 'proxy': True, "timeout": 300},
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
     'model': ["gpt-4o-mini", "gpt-3.5-turbo", 'o3-mini', 'o4-mini', 'gpt-4o',
               'gpt-4-32k', 'gpt-4', 'gpt-4-turbo', "gpt-4.1-mini", 'gpt-4.1-nano',
               'gpt-5', 'gpt-5-mini',
               "grok-3-mini", "grok-3-mini-fast-beta", "grok-3",
               "gemini-2.0-flash-exp", "gemini-2.5-flash-preview-05-20", 'gemini-2.0-flash',
               'glm-4.5-airx',
               "jina-deepsearch-v1", "aihubmix-command-r-08-2024", "aihubmix-Cohere-command-r",
               "Qwen/Qwen3-30B-A3B", 'Qwen/Qwen3-8B', "Qwen/Qwen3-32B", "qwen2.5-vl-72b-instruct",
               'qwen2.5-math-72b-instruct', "Qwen/QwQ-32B", "Qwen/QVQ-72B-Preview", "qwen-long",
               "kimi-latest", 'moonshotai/Kimi-Dev-72B',
               "yi-large-turbo", "yi-34b-chat-200k",
               "THUDM/GLM-4-32B-0414", "THUDM/GLM-4-32B-0414", "THUDM/GLM-Z1-32B-0414",
               "Doubao-1.5-pro-32k", "Doubao-pro-128k", "nvidia/llama-3.1-nemotron-70b-instruct",
               "claude-3-5-sonnet-20241022", 'claude-3-5-sonnet-latest',
               "claude-3-7-sonnet-20250219", "claude-sonnet-4-20250514"],
     'embedding': ["jina-clip-v2", "jina-colbert-v2", 'jina-embeddings-v3', "jina-embeddings-v2-base-code",
                   "gemini-embedding-exp-03-07", 'text-embedding-3-small', "text-embedding-3-large",
                   'text-embedding-004'],
     'url': 'https://aihubmix.com/v1/chat/completions',
     'embedding_url': 'https://aihubmix.com/v1/embeddings',
     'reranker_url': 'https://aihubmix.com/v1/rerank',
     'base_url': "https://aihubmix.com/v1",
     'supported_openai': True, 'supported_list': True, 'proxy': False, "timeout": 300},
    # https://docs.tokenflux.ai/quickstart
    # https://tokenflux.ai/models
    {'name': 'tokenflux', 'type': 'default', 'api_key': '',
     'model': ['google/gemini-pro', 'google/gemini-2.5-pro', 'google/gemini-2.5-flash',
               'google/gemma-3-12b-it', 'google/gemma-3-27b-it',
               'anthropic/claude-3-sonnet', 'anthropic/claude-3.5-sonnet', 'anthropic/claude-3.5-haiku',
               'anthropic/claude-3-7-sonnet', 'anthropic/claude-3.7-sonnet:thinking', 'anthropic/claude-sonnet-4',
               'microsoft/phi-4',
               'mistralai/mistral-large', 'mistralai/mistral-large-2407', 'mistralai/mistral-medium-3',
               'mistralai/mistral-nemo',
               'bytedance/doubao-1.5-pro-32k', 'bytedance/doubao-seed-1.6-thinking', 'baidu/ernie-4.5-300b-a47b',
               'moonshotai/kimi-k2', 'moonshotai/kimi-vl-a3b-thinking',
               'deepseek/deepseek-r1', 'deepseek/deepseek-v3', 'deepseek/deepseek-r1-0528',
               'meta-llama/llama-3.1-405b', 'meta-llama/llama-3.1-405b-instruct',
               'meta-llama/llama-3.1-70b-instruct', 'meta-llama/llama-3-70b-instruct',
               'z-ai/glm-4-32b', 'z-ai/glm-4.5', 'z-ai/glm-4.5-air',
               'thudm/glm-4-32b', 'thudm/glm-4.1v-9b-thinking', 'thudm/glm-z1-32b',
               'x-ai/grok-3', 'x-ai/grok-3-beta', 'x-ai/grok-3-mini', 'x-ai/grok-3-mini-beta', 'x-ai/grok-4',
               'tencent/hunyuan-a13b-instruct', 'qwen/qwen3-235b-a22b', 'qwen/qwen3-coder',
               'openai/o1', 'openai/o1-mini', 'openai/o1-pro', 'openai/o3', 'openai/o3-mini', 'openai/o3-pro',
               'openai/o4-mini', 'openai/gpt-4o-mini', 'openai/gpt-4o', 'openai/chatgpt-4o-latest',
               'openai/gpt-3.5-turbo', 'openai/gpt-3.5-turbo-instruct',
               'openai/gpt-4', 'openai/gpt-4-turbo',
               'openai/gpt-4.1-mini', 'openai/gpt-4.1-nano', 'openai/gpt-4.1',
               'openai/gpt-5', 'openai/gpt-5-chat', 'openai/gpt-5-mini', 'openai/gpt-5-nano'],

     'embedding': ["BAAI bge-m3", 'text-embedding-3-small', 'text-embedding-3-large', 'qwen/text-embedding-v3',
                   'qwen/text-embedding-v4'],
     'url': 'https://aihubmix.com/v1/chat/completions',
     'embedding_url': 'https://aihubmix.com/v1/embeddings',
     'reranker_url': 'https://aihubmix.com/v1/rerank',
     'mcp_url': 'https:/tokenflux.ai/v1/mcps',
     'base_url': "https://tokenflux.ai/v1",
     'supported_openai': True, 'supported_list': True, 'proxy': False, "timeout": 300},
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
     'supported_openai': False, 'supported_list': True, 'proxy': True, "timeout": 300},
    # https://o3.fan/info/models/
    # https://o3.fan/account/profile
    # https://o3.fan/account/pricing
    {'name': 'othree', 'type': 'default', 'api_key': '',
     'model': ['codegeex-4', "glm-3-turbo", "kimi-k2-0711-preview", "moonshot-v1-8k", "abab6.5s-chat",
               "Doubao-1.5-pro-256k", "SparkDesk-Lite", "SparkDesk-Pro",
               "gpt-5-mini", "gpt-5-nano", "gpt-4o",
               "gemini-1.5-flash-latest", "gemini-2.0-flash", "gemini-2.5-flash"],
     'embedding': ["embedding-2", 'embedding-3', 'Doubao-embedding'],
     'reranker': ['bge-reranker-v2-m3', 'bge-large-zh', "bce-reranker-base_v1"],
     'url': 'https://api.o3.fan/v1/chat/completions',
     'embedding_url': 'https://api.o3.fan/v1/embeddings',
     'reranker_url': 'https://api.o3.fan/v1/rerank',
     'model_url': 'https://api.o3.fan/v1/models',
     'base_url': "https://api.o3.fan/v1",
     'supported_openai': True, 'supported_list': True, 'proxy': False, "timeout": 300},
    # https://doc.zhizengzeng.com/doc-3979947
    {'name': 'zzz', 'type': 'default', 'api_key': '',
     'model': ['gpt-5', 'gpt-5-2025-08-07', 'gpt-5-mini', 'gpt-5-mini-2025-08-07', 'gpt-5-nano',
               'gpt-5-nano-2025-08-07', 'gpt-5-chat-latest', 'gpt-oss-120b', 'gpt-oss-20b', 'o4-mini',
               'o4-mini-2025-04-16', 'o3-pro', 'o3-pro-2025-06-10', 'o3', 'o3-2025-04-16', 'gpt-4.1',
               'gpt-4.1-2025-04-14', 'gpt-4.1-mini', 'gpt-4.1-mini-2025-04-14', 'gpt-4.1-nano',
               'gpt-4.1-nano-2025-04-14', 'gpt-4.5-preview', 'gpt-4.5-preview-2025-02-27', 'o3-mini',
               'o3-mini-2025-01-31', 'o1-pro', 'o1-pro-2025-03-19', 'o1', 'o1-2024-12-17', 'o1-preview',
               'o1-preview-2024-09-12', 'o1-mini', 'o1-mini-2024-09-12', 'gpt-3.5-turbo', 'gpt-3.5-turbo-0301',
               'gpt-3.5-turbo-0613', 'gpt-3.5-turbo-1106', 'gpt-3.5-turbo-0125', 'gpt-3.5-turbo-16k',
               'gpt-3.5-turbo-16k-0613', 'gpt-3.5-turbo-instruct', 'gpt-3.5-turbo-instruct-0914', 'gpt-4', 'gpt-4-0314',
               'gpt-4-0613', 'gpt-4-1106-preview', 'gpt-4-vision-preview', 'gpt-4-1106-vision-preview',
               'gpt-4-0125-preview', 'gpt-4-turbo', 'gpt-4-turbo-preview', 'gpt-4-turbo-2024-04-09', 'gpt-4o',
               'gpt-4o-2024-11-20', 'gpt-4o-2024-08-06', 'gpt-4o-2024-05-13', 'gpt-4o-audio-preview',
               'gpt-4o-audio-preview-2024-10-01', 'gpt-4o-mini', 'gpt-4o-mini-2024-07-18', 'gpt-4o-mini-audio-preview',
               'gpt-4o-mini-audio-preview-2024-12-17', 'chatgpt-4o-latest', 'gpt-4-32k', 'gpt-4-32k-0314',
               'gpt-4-32k-0613', 'gpt-4o-transcribe',
               'gpt-4o-mini-transcribe', 'dall-e-2', 'dall-e-3', 'claude-opus-4-1-20250805',
               'claude-opus-4-1-20250805-thinking', 'claude-opus-4-20250514', 'claude-sonnet-4-20250514',
               'claude-opus-4-20250514-thinking', 'claude-sonnet-4-20250514-thinking',
               'claude-3-7-sonnet-20250219-thinking', 'claude-3-7-sonnet-thinking', 'claude-3-7-sonnet-20250219',
               'claude-3-7-sonnet-latest', 'claude-3-opus-20240229', 'claude-3-5-sonnet-20241022',
               'claude-3-5-sonnet-20240620', 'claude-3-sonnet-20240229', 'claude-3-5-haiku-20241022',
               'claude-3-haiku-20240307', 'claude-2.1', 'claude-2.0', 'gemini-2.5-pro', 'gemini-2.5-pro-thinking',
               'gemini-2.5-flash', 'gemini-2.5-flash-thinking', 'gemini-2.5-flash-lite',
               'gemini-2.5-flash-lite-thinking', 'gemini-2.5-flash-preview-05-20',
               'gemini-2.5-flash-preview-05-20-thinking', 'gemini-2.5-flash-preview-04-17',
               'gemini-2.5-flash-preview-04-17-thinking', 'gemini-2.5-pro-preview-06-05',
               'gemini-2.5-pro-preview-06-05-thinking', 'gemini-2.5-pro-preview-05-06',
               'gemini-2.5-pro-preview-05-06-thinking', 'gemini-2.0-flash', 'gemini-2.0-flash-lite',
               'gemini-2.0-flash-lite-preview-02-05', 'gemini-1.5-flash', 'gemini-1.5-pro',
               'gemini-2.0-flash-thinking-exp', 'gemini-2.0-flash-thinking-exp-1219', 'gemini-2.0-flash-exp',
               'gemini-2.0-pro-exp-02-05', 'gemini-exp-1206', 'gemini-exp-1121', 'gemini-exp-1114', 'grok-4',
               'grok-4-0709', 'grok-4-latest', 'grok-3', 'grok-3-latest', 'grok-3-beta', 'grok-3-fast',
               'grok-3-fast-latest', 'grok-3-fast-beta', 'grok-3-mini', 'grok-3-mini-beta', 'grok-3-mini-latest',
               'grok-3-mini-fast', 'grok-3-mini-fast-latest', 'grok-3-mini-fast-beta', 'grok-2-vision-1212',
               'grok-2-vision', 'grok-2-vision-latest', 'grok-2-1212', 'grok-2', 'grok-2-latest', 'grok-beta',
               'grok-vision-beta', 'ernie-x1-turbo-32k', 'ernie-x1-32k-preview', 'ernie-x1-32k',
               'ernie-4.5-turbo-vl-32k', 'ernie-4.5-turbo-vl-32k-preview', 'ernie-4.5-8k-preview',
               'ernie-4.5-turbo-32k', 'ernie-4.5-turbo-128k', 'ERNIE-Bot', 'ERNIE-Bot-4', 'ERNIE-4.0-8K',
               'ernie-4.0-8k', 'ernie-4.0-8k-0613', 'ernie-4.0-8k-latest', 'ernie-4.0-8k-preview',
               'ernie-4.0-turbo-128k', 'ernie-4.0-turbo-8k', 'ernie-4.0-turbo-8k-0628', 'ernie-4.0-turbo-8k-0927',
               'ernie-4.0-turbo-8k-latest', 'ernie-4.0-turbo-8k-preview', 'ERNIE-Bot-turbo', 'ERNIE-3.5-8K',
               'ERNIE-Speed-128K', 'ERNIE-Speed-8K', 'ERNIE-Lite-8K', 'ERNIE-Tiny-8K', 'ernie-3.5-128k',
               'ernie-3.5-128k-preview', 'ernie-3.5-8k', 'ernie-3.5-8k-0613', 'ernie-3.5-8k-0701',
               'ernie-3.5-8k-preview', 'glm-4.5', 'glm-4.5v', 'glm-4.5-x', 'glm-4.5-air', 'glm-4.5-airx',
               'glm-4.5-flash', 'glm-4-plus', 'glm-4-air', 'glm-4-air-250414', 'glm-4-airx', 'glm-4-flashx',
               'glm-4-flashx-250414', 'glm-4-long', 'glm-z1-air', 'glm-z1-airx', 'glm-z1-flash', 'glm-4v-plus',
               'glm-4v-plus-0111', 'glm-4v-flash', 'glm-4-0520', 'glm-4-flash', 'glm-4', 'glm-4v', 'glm-3-turbo',
               'chatglm-6b-v2', 'chatglm3-6b', 'qwen3-235b-a22b', 'qwen3-32b', 'qwen3-30b-a3b', 'qwen3-14b', 'qwen3-8b',
               'qwen3-4b', 'qwen3-1.7b', 'qwen3-0.6b', 'qwen3-coder-plus', 'qwen3-coder-plus-2025-07-22',
               'qwen-coder-plus', 'qwen-coder-plus-latest', 'qwen-coder-plus-2024-11-06', 'qwen-coder-turbo',
               'qwen-coder-turbo-latest', 'qwen-coder-turbo-2024-09-19', 'qwen-max', 'qwen-max-latest',
               'qwen-max-2025-01-25', 'qwen-max-1201', 'qwen-max-longcontext', 'qwen-plus', 'qwen-turbo', 'qwen-vl-v1',
               'qwen-vl-chat-v1', 'qwen2.5-vl-72b-instruct', 'Qwen2.5-VL-72B-Instruct', 'qwen2.5-vl-32b-instruct',
               'qwen2.5-vl-7b-instruct', 'qwen2.5-vl-3b-instruct', 'qwen2.5-3b-instruct', 'qwen2.5-72b-instruct',
               'qwen2.5-32b-instruct', 'qwen2.5-14b-instruct', 'qwen2.5-7b-instruct', 'qwen2.5-1.5b-instruct',
               'qwen2.5-0.5b-instruct', 'qwen2-72b-instruct', 'qwen2-57b-instruct', 'qwen2-7b-instruct',
               'qwen2-1.5b-instruct', 'qwen2-0.5b-instruct', 'qwen1.5-110b-chat', 'qwen1.5-72b-chat', 'qwen-72b-chat',
               'x1', '4.0Ultra', 'generalv3.5', 'max-32k', 'generalv3', 'pro-128k', 'lite', 'hunyuan-t1-latest',
               'hunyuan-t1-20250403', 'hunyuan-t1-20250321', 'hunyuan-turbos-latest', 'hunyuan-turbos-20250515',
               'hunyuan-turbos-20250416', 'hunyuan-turbos-20250313', 'hunyuan-turbos-longtext-128k-20250325',
               'hunyuan-large', 'hunyuan-turbo', 'hunyuan-standard', 'hunyuan-standard-256k', 'hunyuan-translation',
               'hunyuan-translation-lite', 'hunyuan-role', 'hunyuan-functioncall', 'hunyuan-code', 'hunyuan-t1-vision',
               'hunyuan-turbos-vision', 'hunyuan-vision', 'hunyuan-pro', 'hunyuan-lite', 'Baichuan4-Turbo',
               'Baichuan4-Air', 'Baichuan4', 'Baichuan3-Turbo', 'Baichuan3-Turbo-128k', 'Baichuan2-Turbo',
               'Baichuan2-Turbo-192k', 'Baichuan2-53B', 'baichuan2-7b-chat-v1', 'baichuan2-13b-chat-v1',
               'baichuan-7b-v1', '360zhinao2-o1', '360gpt2-o1', '360gpt-turbo', '360gpt-pro', '360gpt2-pro',
               '360gpt-pro-trans', 'moonshot-v1-8k', 'moonshot-v1-32k', 'moonshot-v1-128k',
               'moonshot-v1-8k-vision-preview', 'moonshot-v1-32k-vision-preview', 'moonshot-v1-128k-vision-preview',
               'kimi-latest', 'kimi-k2', 'kimi-k2-0711-preview', 'kimi-k2-turbo-preview', 'kimi-thinking-preview',
               'deepseek-chat', 'deepseek-coder', 'deepseek-v3', 'DeepSeek-V3', 'deepseek-reasoner', 'deepseek-r1',
               'deepseek-r1-250528', 'yi-lightning', 'yi-large', 'yi-large-turbo', 'yi-large-rag', 'yi-medium',
               'yi-medium-200k', 'yi-spark', 'yi-vision', 'yi-6b-chat', 'yi-34b-chat', 'MiniMax-M1', 'MiniMax-Text-01',
               'abab6.5-chat', 'abab6.5s-chat', 'abab6-chat', 'abab5.5-chat', 'abab5.5s-chat', 'doubao-seed-1.6',
               'doubao-seed-1.6-250615', 'doubao-seed-1.6-thinking', 'doubao-seed-1.6-thinking-250615',
               'doubao-seed-1.6-flash', 'doubao-seed-1.6-flash-250615', 'doubao-1.5-thinking-vision-pro',
               'doubao-1.5-thinking-vision-pro-250428', 'doubao-1.5-thinking-pro', 'doubao-1.5-thinking-pro-250415',
               'doubao-1.5-thinking-pro-m-250428', 'doubao-1.5-thinking-pro-m-250415', 'doubao-1.5-vision-pro',
               'doubao-1.5-vision-pro-250328', 'doubao-1.5-vision-pro-32k', 'doubao-1.5-vision-pro-32k-250115',
               'doubao-1.5-vision-lite', 'doubao-1.5-vision-lite-250315', 'doubao-1.5-pro-32k',
               'doubao-1.5-pro-32k-250115', 'doubao-1.5-pro-256k', 'doubao-1.5-pro-256k-250115', 'Doubao-lite-4k',
               'Doubao-lite-32k', 'Doubao-1.5-lite-32k', 'Doubao-lite-128k', 'Doubao-pro-4k', 'Doubao-pro-32k',
               'bot-Doubao-pro-32k-browsing-240828', 'bot-Doubao-pro-32k-functioncall-241028', 'Doubao-pro-128k',
               'llama-4-maverick-17b-128e-instruct', 'llama-4-scout-17b-16e-instruct', 'llama3-8b-instruct',
               'llama3-70b-instruct'],
     'embedding': ['text-embedding-ada-002', 'text-embedding-3-large', 'text-embedding-3-small'],
     'reranker': ['bge-reranker-v2-m3', 'bge-large-zh', "bce-reranker-base_v1"],
     'speech': ['tts-1', 'tts-1-1106', 'tts-1-hd', 'tts-1-hd-1106', 'gpt-4o-mini-tts', 'whisper-1'],
     'url': 'https://api.zhizengzeng.com/v1/chat/completions',
     'embedding_url': 'https://api.zhizengzeng.com/v1/embeddings',
     'base_url': "https://api.zhizengzeng.com/v1",
     'supported_openai': True, 'supported_list': True, 'proxy': False, "timeout": 300},
    {'name': '147', 'type': 'default', 'api_key': '', 'model': ['deepseek-v3-2-exp'],
     'url': 'https://147ai.com/v1/chat/completions',
     'base_url': "https://147ai.com/v1",
     'supported_openai': True, 'supported_list': True, 'proxy': False, "timeout": 600},
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
        'othree': Config.OThree_Service_Key,
        'zzz': Config.ZZZ_Service_Key,
        '147': Config.AI147_Service_Key
    }
    if not name:
        return api_keys
    api_key = api_keys.get(name, None)
    if Config.is_invalid(api_key):
        return None
    return api_key


# SUPPORTED_OPENAI_MODELS = {'moonshot', 'glm', 'qwen', 'hunyuan', 'silicon', 'doubao', 'baichuan', 'deepseek', 'minimax',
#                            'mistral', 'gemini'}

# Api_Tokens = [
#     {"type": 'baidu', "func": get_baidu_access_token, "access_token": None, "expires_at": None, "expires_delta": 1440}]


if __name__ == "__main__":
    from openai import OpenAI

    # for backward compatibility, you can still use `https://api.deepseek.com/v1` as `base_url`.
    # cx = OpenAI(api_key=Config.DeepSeek_Service_Key, base_url="https://api.deepseek.com")
    # print(cx.models.list().model_dump_json())

    print(Config.save())

    Config.mask_sensitive()

    Config.load()
    print(Config.get_config_data())
