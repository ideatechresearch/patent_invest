from pydantic import BaseModel, Field, field_validator, model_validator, condecimal, conint
from typing import Optional, Literal, Annotated, Generator, Dict, List, Tuple, Union, Any
from abc import ABC, abstractmethod
from collections.abc import Callable as IsCallable
from enum import IntEnum, Enum
from dataclasses import asdict, dataclass, is_dataclass, field
from datetime import datetime
import random, time, asyncio, json

from utils import pickle_serialize
from service import ModelListExtract

MODEL_LIST = ModelListExtract()

_KB = 1 << 10
_MB = 1 << 20
_GB = 1 << 30
_T = 1e12
SESSION_ID_MAX = 1 << 31 - 1  # 2147483647,2 ** 31


class BaseTool(ABC, BaseModel):
    name: str
    description: str
    parameters: Optional[Union[Dict, str]] = None

    class Config:
        arbitrary_types_allowed = True

    async def __call__(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""
        return await self.execute(**kwargs)

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""

    def to_param(self) -> Dict:
        """Convert tool to function call format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def dataclass2dict(data):
    """
    通用的递归转换函数，支持 dataclass、BaseModel、Enum、datetime、列表、字典等嵌套结构。serialize
    """

    def convert(obj):
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, BaseModel):
            return obj.model_dump()
        elif is_dataclass(obj):
            return {k: convert(v) for k, v in asdict(obj).items()}
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple, set, frozenset)):
            return [convert(v) for v in obj]  # list(obj)
        elif isinstance(obj, IsCallable):  # 用于运行时判断
            return pickle_serialize(obj)
        elif hasattr(obj, "model_dump") and callable(obj.model_dump):  # 兼容 Pydantic v2 的 BaseMode
            return obj.model_dump()
        elif hasattr(obj, "dict") and callable(obj.dict):  # 兼容 Pydantic v1 或其他自定义对象实现的 .dict() 方法
            return obj.dict()  # convert(obj.dict())
        elif hasattr(obj, "__dict__"):  # 通用对象（包含 __dict__）
            return {k: convert(v) for k, v in vars(obj).items() if not k.startswith("_")}
        else:
            return pickle_serialize(obj)

    return convert(data)


def variables2dict(variables: Optional[Union[Dict[str, str], BaseModel, Any]]) -> Dict[str, str]:
    """
    Convert variables to a dictionary.

    Args:
        variables (Optional[Union[Dict[str, str], BaseModel, Any]]):
            Variables to convert.

    Returns:
        Dict[str, str]: The converted dictionary.

    Raises:
        ValueError: If the variables type is unsupported.Dict[str, str]
    """
    if variables is None:
        return {}
    if isinstance(variables, BaseModel):
        return variables.dict()
    if is_dataclass(variables):
        return dataclass2dict(variables)
    if isinstance(variables, dict):
        return variables
    raise ValueError('Unsupported variables type.')


class GeneratorWithReturn:
    """Generator wrapper to capture the return value."""

    def __init__(self, generator: Generator):
        self.generator = generator
        self.ret = None

    def __iter__(self):
        self.ret = yield from self.generator
        return self.ret


class GenerationParams(BaseModel):
    inputs: Union[str, List[Dict]]
    session_id: int = Field(default_factory=lambda: random.randint(0, SESSION_ID_MAX))
    agent_cfg: Dict = Field(default_factory=dict)


class SearchQueryList(BaseModel):
    query: List[str] = Field(description="A list of search queries to be used for web research.")
    rationale: str = Field(description="A brief explanation of why these queries are relevant to the research topic.")


class Reflection(BaseModel):
    is_sufficient: bool = Field(
        description="Whether the provided summaries are sufficient to answer the user's question."
    )
    knowledge_gap: str = Field(description="A description of what information is missing or needs clarification.")
    follow_up_queries: List[str] = Field(description="A list of follow-up queries to address the knowledge gap.")


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "developer", "tool"]
    content: Union[str, List[Dict[str, Union[str, Dict[str, str]]]]]
    name: Optional[str] = None  # 可以包含 a-z、A-Z、0-9 和下划线，最大长度为 64 个字符

    @model_validator(mode="before")
    @classmethod
    def set_default_name(cls, values):
        role = values.get("role")
        content = values.get("content")
        if role != "user" and isinstance(content, list):
            raise ValueError(f"Role '{role}' must not have list content (only 'user' can have list).")

        name = values.get("name", "")
        if not name or not str(name).strip():
            values.pop("name", None)
        return values

    # @field_validator("name")
    # def validate_name(cls, value):
    #     re.fullmatch(r"^\w{1,64}$", value)
    #     return value or ''


class IMessage(ChatMessage):
    """Represents a chat message in the conversation"""
    tool_calls: Optional[List] = Field(default=None)
    tool_call_id: Optional[str] = Field(default=None)

    def __add__(self, other) -> List["IMessage"]:
        """支持 Message + list 或 Message + Message 的操作"""
        if isinstance(other, list):
            return [self] + other
        elif isinstance(other, IMessage):
            return [self, other]
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(self).__name__}' and '{type(other).__name__}'"
            )

    def __radd__(self, other) -> List["IMessage"]:
        """支持 list + Message 的操作"""
        if isinstance(other, list):
            return other + [self]
        raise TypeError(
            f"unsupported operand type(s) for +: '{type(other).__name__}' and '{type(self).__name__}'")

    def to_dict(self) -> dict:
        """Convert message to dictionary format"""
        message = {"role": self.role}
        if self.content is not None:
            message["content"] = self.content
        if self.tool_calls is not None:
            message["tool_calls"] = [tool_call.dict() for tool_call in self.tool_calls]
        if self.name is not None:
            message["name"] = self.name
        if self.tool_call_id is not None:
            message["tool_call_id"] = self.tool_call_id
        return message

    @classmethod
    def create_message(cls, role: Literal["system", "user", "assistant", "developer", "tool"] = "user",
                       content: Optional[Union[str, List[dict]]] = None, **kwargs) -> "IMessage":
        return cls(role=role, content=content, **kwargs)

    @classmethod
    def tool_message(cls, content: str, name: str, tool_call_id: str) -> "IMessage":
        """Create a tool message"""
        return cls(role="tool", content=content, name=name, tool_call_id=tool_call_id)

    @classmethod
    def from_tool_calls(cls, tool_calls: List[Any], content: Optional[Union[str, List[str]]] = "", **kwargs):
        """Create ToolCallsMessage from raw tool calls.

        Args:
            tool_calls: Raw tool calls from LLM
            content: Optional message content
        """
        formatted_calls = [
            {"id": call.id, "function": call.function.model_dump(), "type": "function"}
            for call in tool_calls
        ]
        return cls(role="assistant", content=content, tool_calls=formatted_calls, **kwargs)


class OpenAIRequest(BaseModel):
    # model: Literal[*tuple(MODEL_LIST.models)]
    model: Annotated[str, lambda v: MODEL_LIST.contains(v)]
    prompt: Optional[Union[str, List[str]]] = None
    suffix: Optional[str] = None
    temperature: Optional[float] = 1.0  # 介于 0 和 2 之间
    top_p: Optional[float] = 1.0
    max_tokens: Optional[conint(ge=1)] = 512
    stream: Optional[bool] = False
    store: Optional[bool] = False

    tools: Optional[List[dict]] = None  # 工具参数,在生成过程中调用外部工具
    stop: Optional[Union[str, List[str]]] = None  # 停止词，用于控制生成的停止条件 ["\n"],
    presence_penalty: Optional[float] = 0.0  # 避免生成重复内容 condecimal(ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = 0.0  # 避免过于频繁的内容 condecimal(ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[int, float]] = None  # 特定 token 的偏置
    logprobs: Optional[int] = None
    user: Optional[str] = None  # 用户标识符,最终用户的唯一标识符

    prediction: Optional[dict] = None  # 预测输出,减少模型响应的延迟
    stream_options: Optional[dict] = None  # 在流式输出的最后一行展示token使用信息
    extra_body: Optional[dict] = None
    extra_query: Optional[Dict[str, Any]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "model": "silicon:Qwen/Qwen2.5-Coder-7B-Instruct",
                "prompt": "User:你好,有问题要问你; Assistant:好的; User:请问1到100的和怎么计算?",
                "suffix": "Assistant:",
                "temperature": 1,
                "user": 'test:test',
                "top_p": 1,
                "max_tokens": 512,
                "stream": False,
                # "stop": '\n\n',
                # "extra_body": {"enable_thinking": False}
            }
        }

    @classmethod
    @field_validator("model")
    def validate_model(cls, value):
        if not MODEL_LIST.contains(value):
            raise ValueError(f"Model '{value}' is not in the supported models list {MODEL_LIST.models}")
        return value


class OpenAIRequestMessage(BaseModel):
    # model: Literal[*tuple(MODEL_LIST.models)]
    model: Annotated[str, lambda v: MODEL_LIST.contains(v)]
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0  # 介于 0 和 2 之间
    top_p: Optional[float] = 1.0
    max_tokens: Optional[conint(ge=1)] = 512
    stream: Optional[bool] = False
    store: Optional[bool] = False

    tools: Optional[List[dict]] = None  # 工具参数,在生成过程中调用外部工具
    stop: Optional[Union[str, List[str]]] = None  # 停止词，用于控制生成的停止条件 ["\n"],
    presence_penalty: Optional[float] = 0.0  # 避免生成重复内容 condecimal(ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = 0.0  # 避免过于频繁的内容 condecimal(ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[int, float]] = None  # 特定 token 的偏置
    user: Optional[str] = None  # 用户标识符,最终用户的唯一标识符

    reasoning_effort: Optional[str] = None  # 探索高级推理和问题解决模型 low,medium,high
    # max_completion_tokens
    response_format: Optional[dict] = None  # 结构化输出,响应符合 JSON 架构
    prediction: Optional[dict] = None  # 预测输出,减少模型响应的延迟
    stream_options: Optional[dict] = None  # 在流式输出的最后一行展示token使用信息
    extra_body: Optional[dict] = None
    extra_query: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, str]] = None  # 可选，附加的元数据

    # file =
    # modalities= ["text", "audio"]
    # audio={voice: "alloy", format: "wav"}
    # metadata: {
    #     role: "manager",
    #     department: "accounting",
    #     source: "homepage"
    # }

    class Config:
        json_schema_extra = {
            "example": {
                "model": "qwen-turbo",
                "messages": [
                    {
                        "role": "user",
                        "content": "你好,有问题要问你",
                        "name": 'test'
                    }
                ],
                "temperature": 1,
                "user": 'test:test',
                "top_p": 1,
                "max_tokens": 512,
                "stream": False,
                "extra_body": {"enable_thinking": False}
            }
        }

    @classmethod
    @field_validator("model")
    def validate_model(cls, value):
        if not MODEL_LIST.contains(value):
            raise ValueError(f"Model '{value}' is not in the supported models list {MODEL_LIST.models}")
        return value

    @classmethod
    @field_validator("messages")
    def check_messages(cls, values):
        if not values:
            raise ValueError('Messages are required')
        return values


class OpenAIResponse(BaseModel):
    id: str
    object: str
    created: int  # Unix 时间戳
    model: str
    choices: List[dict]
    usage: dict
    # completion_tokens,
    # prompt_tokens(prompt_cache_hit_tokens,prompt_cache_miss_tokens),
    # total_tokens（prompt + completion）
    # completion_tokens_details

    service_tier: Optional[str] = None  # 标识当前 API 请求或用户的服务层级
    system_fingerprint: Optional[str] = None  # 标识与请求关联的特定客户端、设备或会话,后端配置的指纹


class OpenAIEmbeddingRequest(BaseModel):
    model: str  # 模型名称
    input: Union[str, List[str]]  # 输入的文本数组
    encoding_format: str = "float"  # 默认设置为 "float"，表示返回浮点数嵌入
    dimensions: Optional[int] = None  # 指定向量维度（text-embedding-v3及 text-embedding-v4）

    user: Optional[str] = None  # 可选，标识用户

    @model_validator(mode="before")
    def set_default_value(cls, values):
        if isinstance(values.get("input"), str):
            values["input"] = [values["input"]]
        values["input"] = [x.replace("\n", " ") for x in values["input"]]
        user = values.get("user", "")
        if not user or not str(user).strip():
            values.pop("user", None)
        return values

    class Config:
        json_schema_extra = {
            "example": {
                "model": "qwen:text-embedding-v4",
                "input": [
                    "role", "user",
                    "content", "你好,有问题要问你", "第二条文本\n带换行"
                ],
                "encoding_format": "float",
                "dimensions": 1024
            }
        }


class AuthRequest(BaseModel):
    eth_address: Optional[str] = None
    signed_message: Optional[str] = None
    original_message: Optional[str] = None

    username: Optional[str] = None
    password: Optional[str] = None
    public_key: Optional[str] = None
    code: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "username": "test",
                "password": "123456",
                "public_key": "0x123456789ABCDEF",
                "eth_address": None,
                "signed_message": None,
                "original_message": None,
                "code": None,  # "NUM"
            }
        }


class Registration(AuthRequest):
    role: Optional[str] = 'user'
    group: Optional[str] = '0'

    class Config:
        json_schema_extra = {
            "example": {
                "username": "test",
                "password": "secure_password",
                "role": "user",
                "group": '0',
                "public_key": "0x123456789ABCDEF",
                "eth_address": "0x123456789ABCDEF",
                "signed_message": "signed_message_here",
                "original_message": "original_message_here",
                "code": '123456',  # "NUM"
            }
        }


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str = None


class IError(Exception):
    def __init__(self, message="Could not validate credentials"):
        self.message = message
        super().__init__(self.message)


class PromptRequest(BaseModel):
    query: Optional[str] = None
    model: Optional[str] = "deepseek:deepseek-reasoner"
    depth: List[str] = Field(default_factory=lambda: ["73", "71", "72", "74"])

    class Config:
        json_schema_extra = {
            "example": {
                "depth": ["73", "71", "72", "74"],
                "model": "deepseek:deepseek-reasoner"
            }
        }


class TranslateRequest(BaseModel):
    text: str = "你好我的朋友。"
    source: str = "auto"
    target: str = "auto"
    platform: str = "baidu"


class EmbeddingRequest(BaseModel):
    texts: List[str]
    model_name: str = 'qwen:text-embedding-v2'
    model_id: int = 0
    normalize: bool = False

    class Config:
        protected_namespaces = ()
        json_schema_extra = {
            "example": {
                "model_name": "qwen:text-embedding-v2",
                "texts": [
                    "role", "user",
                    "content", "你好,有问题要问你"
                ],
                "normalize": False
            }
        }


class FuzzyMatchRequest(BaseModel):
    texts: List[str]
    terms: List[str]
    top_n: int = 3
    cutoff: float = 0.6
    method: Literal['levenshtein', 'bm25', 'reranker', 'embeddings'] = 'levenshtein'


class PlatformEnum(str, Enum):
    baidu = "baidu"
    ali = "ali"
    dashscope = "dashscope"


class ToolRequest(BaseModel):
    messages: Optional[List[ChatMessage]] = Field(default_factory=list)
    tools: Optional[List[Any]] = Field(default_factory=list)
    user: Optional[str] = Field(default=None)
    prompt: Optional[str] = Field(default=None)
    model_name: Optional[str] = 'qwen:qwen-max'
    model_metadata: str = 'qwen:qwen-coder-plus'
    model_id: int = -1
    top_p: float = 0.95
    temperature: float = 0.01

    class Config:
        protected_namespaces = ()
        json_schema_extra = {
            "example": {
                "messages": [
                    {
                        "role": "system",
                        "content": "请根据用户的提问分析意图，请转换用户的问题，提取所需的关键参数，并自动选择最合适的工具进行处理。"
                    },
                    {
                        "role": "user",
                        "content": "请告诉我 2023-11-22 前面三周的日期范围。"
                    }
                ],
                "tools": [],
                "prompt": '',
                "user": '',
                "model_name": "qwen:qwen-max",
                "model_metadata": "qwen:qwen-coder-plus",
                "model_id": -1,
                "top_p": 0.95,
                "temperature": 0.01
            }
        }

        @classmethod
        @field_validator("tools")
        def clean_tools(cls, values):
            if not values:
                return []
            return [t for t in values if isinstance(t, dict) and t]


class AssistantToolsEnum(str, Enum):
    code = "code_interpreter"
    web = "web_search"
    function = "function_calling"


class AssistantRequest(BaseModel):
    question: str
    prompt: str = "You are a personal math tutor. Write and run code to answer math questions."
    user_name: str = 'test'
    tools_type: AssistantToolsEnum = AssistantToolsEnum.code
    model_id: int = 4

    class Config:
        protected_namespaces = ()


class KeywordItem(BaseModel):
    function: Optional[str] = None
    args: Optional[List[Any]] = None
    kwargs: Optional[Dict[str, Any]] = None
    env: Optional[Dict[str, Any]] = None


class CallbackUrl(BaseModel):
    format: Literal["query", "json", "form"] = "json"
    url: Optional[str] = None
    payload: Optional[Dict[str, Union[str, int]]] = None
    mapping: Optional[Dict[str, Union[str, int]]] = None
    params: Optional[Dict[str, str]] = None
    headers: Optional[Dict[str, str]] = None

    @model_validator(mode="before")
    @classmethod
    def clean_url(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        if "url" in data and isinstance(data["url"], str) and not data["url"].strip():
            data["url"] = None
        return data

    class Config:
        json_schema_extra = {"example": {'url': 'http://127.0.0.1:7000/callback', 'format': 'json'}}


class MetadataRequest(BaseModel):
    function_code: str = Field(..., description="函数源码")
    user: str = Field(default='test', description="用户ID")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="预设元数据结构")
    model_name: Optional[str] = Field(default='qwen:qwen-coder-plus', description="使用的大模型名称")
    description: Optional[str] = Field(default=None, description="函数功能描述")
    callback: Optional[CallbackUrl] = Field(default=None, description="外部函数调用信息")
    code_type: Optional[str] = Field(default="Python", description="代码类型")
    cache_sec: Optional[int] = Field(default=0, description="缓存秒数，0 表示使用默认")

    class Config:
        protected_namespaces = ()


class AgentRequest(BaseModel):
    messages: Optional[List[ChatMessage]] = Field(default_factory=list)
    user: Optional[str] = Field(default=None)
    model: Optional[str] = 'deepseek-chat'
    callback: Optional[CallbackUrl] = Field(default=None, description="外部函数调用信息")

    class Config:
        protected_namespaces = ()


class CompletionResponse(BaseModel):
    answer: Optional[str] = None
    transform: Optional[Any] = None
    reference: Optional[Any] = None
    id: Optional[int] = None


class CompletionParams(BaseModel):
    stream: bool = Field(False, description="Whether to stream the response")
    temperature: float = Field(0.8, description="Temperature for response generation")
    top_p: float = Field(0.8, description="The probability threshold setting for the model.")
    max_tokens: Optional[int] = Field(1024, description="Maximum number of tokens the model can generate.")

    prompt: Optional[str] = Field(default=None,
                                  description="The initial system content or prompt used to guide the AI's response.")
    question: Optional[Union[str, List[str]]] = Field(None,
                                                      description="The primary question or prompt for the AI to respond to. ")
    agent: Optional[str] = Field(default=None,
                                 description="System content identifier. This index represents different scenarios or contexts for AI responses, allowing the selection of different system content.")
    suffix: Optional[str] = Field(None, description="The suffix for the AI to respond to completion. ")
    extract: Optional[Union[str, List[str]]] = Field(None,
                                                     description="Response Format,The type of content to extract from response(e.g., code.python,code.bash,code.cpp,code.sql,json,header,links)")
    callback: Optional[CallbackUrl] = Field(default=None,
                                            description='Callback info: a URL string or a dict with url and optional payload,params,headers')
    model_name: str = Field("moonshot",
                            description=("Specify the name of the model to be used. It can be any available model, "
                                         "such as 'moonshot', 'glm', 'qwen', 'ernie', 'hunyuan', 'doubao','spark','baichuan','deepseek', or other models."))
    model_id: int = Field(0, description="Model ID to be used")
    keywords: Optional[List[Union[
        str,
        Tuple[str, Any],
        Tuple[str, List[Any]],
        Tuple[str, Dict[str, Any]],
        Tuple[str, List[Any], Dict[str, Any]]
    ]]] = Field(
        None, description=(
            "A list of keywords or tuples of (keyword, function, *args,*kwargs) used to search for relevant information across various sources, "
            "A list of tools represented as tuples, where each tuple consists of a callable and its corresponding arguments. "
            "such as online searches, database queries, or vector-based search systems. "
            "These keywords help guide the retrieval of data based on the specific terms provided."))
    # keywords: Optional[List[KeywordItem]] = None
    tools: Optional[List[Dict]] = Field(
        None, description=(
            "This allows the AI to call specific functions with the provided arguments to perform tasks such as data processing, "
            "API calls, or other utility operations. Each tool can be invoked to enhance the AI's capabilities and provide more "
            "dynamic responses based on the context."
        ))

    images: Optional[List[str]] = None

    # score_threshold: float = Field(default=0, ge=-1, le=1, description="The score threshold setting for the model.")
    # top_n: int = Field(10,description="The number of top results to retrieve during vector search. This determines how many of the highest-scoring items will be returned.")
    # keywords = [
    #     "simple_keyword",  #  纯字符串（符合 `str`）
    #     ("func_name", "arg1"),  #  (函数名, 单个参数)
    #     ("func_name", [[1, 2, 3]]),  #  (函数名, 单个参数,列表)
    #     ("another_func", [1, 2, 3]),  # (函数名, 多个参数列表)
    #     ("some_tool", {"key": "value"}),  #  (函数名, 关键字参数字典)
    #     ("complex_tool", [1, 2], {"key": "v"})  #  (函数名, 位置参数列表, 关键字参数)
    # ]

    class Config:
        protected_namespaces = ()
        json_schema_extra = {
            "examples": [
                {
                    'prompt': '你是一个知识广博且乐于助人的助手，擅长分析和解决各种问题。请根据我提供的信息进行帮助。',
                    "question": ["请解释人工智能的原理。", "AI是什么啊,可以描述一下吗?"],
                    "agent": "0",
                    "stream": False,
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "model_name": "silicon",
                    "model_id": 0,
                    "extract": "code.python",
                    "max_tokens": 4000,
                    "keywords": ["AI智能"],
                    "tools": [],
                    "callback": None
                },
                {
                    "stream": False,
                    "extract": "wechat",
                    "callback": None,
                    "model_name": "doubao",
                    "model_id": -1,
                    "prompt": "这是什么啊,可以描述一下吗?。",
                    "agent": "42",
                    "top_p": 0.8,
                    "question": "",
                    "keywords": [('web_search', "大象")],  # [("tool_name", {"key": "value"})]
                    "suffix": "这是",
                    "temperature": 0.7,
                    "max_tokens": 4096,
                    "tools": []
                }
            ]
        }

    @classmethod
    @field_validator("model_name")
    def validate_model(cls, value):
        if not MODEL_LIST.contains(value):
            raise ValueError(f"Model '{value}' is not in the supported models list {MODEL_LIST.models}")
        return value

    @classmethod
    @field_validator("tools")
    def clean_tools(cls, values):
        if not values:
            return []
        return [t for t in values if isinstance(t, dict) and t]

    def asdict(self):
        return self.model_dump()  # .dict()

    def payload(self):
        return self.model_dump(include={'temperature', 'top_p', 'max_tokens', 'stream'})
        # {k: v for k, v in self.dict().items() if k in ['temperature', 'top_p', 'max_tokens', 'stream']}


class SubmitMessagesRequest(BaseModel):
    request_id: Optional[str] = None
    name: Optional[str] = None
    user: Optional[str] = None
    robot_id: Optional[str] = None
    use_hist: bool = Field(default=False, description="Use historical messages.")
    filter_limit: Optional[int] = Field(-500,
                                        description="The limit count(<0) or max len(>0) to filter historical messages.")
    filter_time: Optional[int] = Field(0, description="The timestamp to filter historical messages.")

    messages: Optional[List[ChatMessage]] = Field(default_factory=list,
                                                  description="A list of message objects representing the current conversation. "
                                                              "If no messages are provided and `use_hist` is set to `True`, "
                                                              "the system will filter existing chat history using the fields "
                                                              "`name`, `user`, and `filter_time`. "
                                                              "If `messages` are provided, the last user message will be used as the question.")

    params: Optional[List[CompletionParams]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "request_id": None,
                "name": None,
                "user": "aigc_test",
                "robot_id": None,
                "use_hist": False,
                "filter_limit": -500,
                "filter_time": 0.0,
                "messages": [
                    {
                        "role": "user",
                        "content": "你好，我的朋友",
                        "name": 'test',
                    }
                ],
                "params": [
                    {
                        "name": None,
                        "stream": False,
                        "temperature": 0.8,
                        "top_p": 0.8,
                        "max_tokens": 1024,
                        "prompt": "",
                        "question": "中国今年多少新生儿",
                        "keywords": ["中国2024新生儿人口"],
                        "tools": [],
                        'images': [],
                        "agent": "0",
                        "extract": "json",
                        "callback": {'url': 'http://127.0.0.1:7000/callback'},
                        "model_name": "moonshot",
                        "model_id": 0,
                    },
                    {
                        "agent": "0",
                        "callback": {"url": "http://127.0.0.1:7000/callback"},
                        "extract": "wechat",
                        "images": [],
                        "keywords": [('web_search', "感冒灵")],
                        "max_tokens": 1024,
                        "model_id": 0,
                        "model_name": "qwen",
                        "prompt": "",
                        "question": "感冒灵颗粒好用吗",
                        "stream": False,
                        "temperature": 0.8,
                        "tools": [],
                        "top_p": 0.8
                    }
                ]
            }
        }


class ChatCompletionRequest(CompletionParams):
    request_id: Optional[str] = None
    name: Optional[str] = None
    user: Optional[str] = None
    robot_id: Optional[str] = None
    use_hist: bool = Field(default=False, description="use historical messages.")
    filter_limit: Optional[int] = Field(-500,
                                        description="The limit count(<0) or max len(>0) to filter historical messages.")
    filter_time: float = Field(default=0, description="The timestamp to filter historical messages.")

    messages: Optional[List[ChatMessage]] = Field(default_factory=list,
                                                  description="A list of message objects representing the current conversation. "
                                                              "If no messages are provided and `use_hist` is set to `True`, "
                                                              "the system will filter existing chat history using the fields "
                                                              "`name`, `user`, and `filter_time`. "
                                                              "If `messages` are provided, the last user message will be used as the question.")
    question: Optional[str] = Field(None,
                                    description="The primary question or prompt for the AI to respond to. "
                                                "If `messages` are provided, this field will be automatically overridden by "
                                                "the last user message content in `messages`. Otherwise, this `question` field "
                                                "will be used directly as the prompt for the AI.")

    class Config:
        json_schema_extra = {
            "example": {
                "request_id": None,
                "name": None,
                "user": "test",
                "robot_id": None,
                "agent": "0",

                "use_hist": False,
                "filter_limit": -500,
                "filter_time": 0.0,
                "model_name": "moonshot",
                "model_id": 0,
                "prompt": '',
                "question": "什么是区块链金融?",
                "messages": [],
                "keywords": [],
                "tools": [],
                'images': [],
                "extract": 'json',
                "callback": None,
                "stream": False,
                "temperature": 0.4,
                "top_p": 0.8,
                "max_tokens": 1024,
                # "score_threshold": 0.0,
                # "top_n": 10,
            }
        }


class SummaryRequest(BaseModel):
    text: Union[str, List[str]]
    extract_prompt: Optional[str] = None
    summary_prompt: Optional[str] = None
    model: str = 'qwen:qwen-max'
    max_tokens: int = 4096
    max_segment_length: int = 1024


class ClassifyRequest(BaseModel):
    query: str
    class_terms: Dict[str, List[str]]
    class_default: Optional[str] = Field(None, description="default or last history to fallback.")

    # robot_id: str = None
    # user: str = None
    emb_model: Optional[str] = "text-embedding-v2"
    rerank_model: Optional[str] = "BAAI/bge-reranker-v2-m3"
    cutoff: float = 0.85

    class Config:
        json_schema_extra = {
            "example": {
                'query': '今天几号？',
                "class_terms": {
                    "经营数据查询": ["经营数据查询", "经营分析", "公司经营状况", "经营报告"],
                    "财务报销": ["财务报销", "报销流程", "报销单", "财务审批", "报销申请"],
                    "跟进记录录入": ["跟进记录", "跟进情况", "客户跟进", "记录跟进", "跟进内容"],
                    "商机录入": ["商机录入", "商机信息", "录入商机", "商机跟进", "商机记录"],
                    "新增客户信息": ["新增客户", "添加客户", "客户信息", "客户录入", "客户添加"],
                    "查询销售额": ["销售额查询", "查询销售额", "销售收入", "销售总额", "销售情况"],
                    "查询回款额": ["回款额查询", "查询回款额", "回款情况", "回款金额", "回款记录"],
                    "研发详情": ["研发详细情况", "研发的单据情况", "研发明细", "研发单据", "研发进展"],
                    "研发进度": ["产研进度", "工作完成进度", "单据完成进度", "研发进展", "工作进度"],
                    "研发质量": ["产品缺陷", "产品质量", "质量问题", "产品问题", "质量报告"],
                    "项目情况": ["项目情况", "项目进度", "项目跟进", "项目状态", "项目详情"],
                    "未验收的项目数": [
                        "待验收的项目数", "尚未验收的项目数", "未完成验收的项目数", "未交付的项目数",
                        "验收未完成的项目数"],
                    "本季度计划验收的项目数": [
                        "本季度预定验收的项目数量", "本季度预计完成验收的项目数", "本季度计划验收的项目数",
                        "本季度安排验收的项目数", "本季度计划交付的项目数量"]
                },
                "class_default": '聊天',
                'emb_model': 'text-embedding-v2',
                "rerank_model": "BAAI/bge-reranker-v2-m3",
                "cutoff": 0.85
            }
        }
