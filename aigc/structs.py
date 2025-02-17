from pydantic import BaseModel, Field, field_validator, condecimal, conint
from typing import Optional, Literal, Annotated, Generator, Dict, List, Tuple, Union, Any
from enum import Enum as PyEnum
from enum import IntEnum
from dataclasses import asdict, dataclass, is_dataclass
import random

SESSION_ID_MAX = 2 ** 31 - 1  # 2147483647


@dataclass
class FunctionCall:
    name: str
    parameters: Union[Dict, str]


def dataclass2dict(data):
    def enum_dict_factory(inputs):
        return {key: (value.value if isinstance(value, IntEnum) else value)
                for key, value in inputs}

    return asdict(data, dict_factory=enum_dict_factory)


def variables2dict(variables: Optional[Union[Dict[str, str], BaseModel, Any]]) -> Dict[str, str]:
    """
    Convert variables to a dictionary.

    Args:
        variables (Optional[Union[Dict[str, str], BaseModel, Any]]):
            Variables to convert.

    Returns:
        Dict[str, str]: The converted dictionary.

    Raises:
        ValueError: If the variables type is unsupported.
    """
    if variables is None:
        return {}
    if isinstance(variables, BaseModel):
        return variables.dict()
    if is_dataclass(variables):
        return asdict(variables)
    if isinstance(variables, dict):
        return variables
    raise ValueError(
        'Unsupported variables type. Must be a dict, BaseModel, or '
        'dataclass.')


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


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "developer"]
    content: Union[str, List[dict]]  # str
    name: Optional[str] = None  # 可以包含 a-z、A-Z、0-9 和下划线，最大长度为 64 个字符


from config import ModelListExtract
MODEL_LIST = ModelListExtract()

class OpenAIRequest(BaseModel):
    # model: Literal[*tuple(MODEL_LIST.models)]
    model: Annotated[str, lambda v: MODEL_LIST.contains(v)]
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0  # 介于 0 和 2 之间
    top_p: Optional[float] = 1.0
    max_tokens: Optional[conint(ge=1)] = 512
    stream: Optional[bool] = True
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
                        "content": "你好,有问题要问你"
                    }
                ],
                "temperature": 1,
                "top_p": 1,
                "max_tokens": 512,
                "stream": True
            }
        }

    @field_validator("model")
    @classmethod
    def validate_model(cls, value):
        if value == 'gpt-4o-mini':
            return 'qwen-turbo'
        if not MODEL_LIST.contains(value):
            raise ValueError(f"Model '{value}' is not in the supported models list {MODEL_LIST.models}")
        return value

    @field_validator("messages")
    @classmethod
    def check_messages(cls, values):
        if not values:
            raise ValueError('Messages are required')
        return values


class OpenAIResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[dict]
    usage: dict

    service_tier: Optional[str] = None  # 标识当前 API 请求或用户的服务层级
    system_fingerprint: Optional[str] = None  # 标识与请求关联的特定客户端、设备或会话


class AuthRequest(BaseModel):
    eth_address: Optional[str] = None
    signed_message: Optional[str] = None
    original_message: Optional[str] = None

    username: Optional[str] = None
    password: Optional[str] = None
    public_key: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "username": "test",
                "password": "123456",
                "public_key": "0x123456789ABCDEF",
                "eth_address": "0x123456789ABCDEF",
                "signed_message": None,
                "original_message": None
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
                "original_message": "original_message_here"
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


class TranslateRequest(BaseModel):
    text: str = "你好我的朋友。"
    source: str = "auto"
    target: str = "auto"
    platform: str = "baidu"


class EmbeddingRequest(BaseModel):
    texts: List[str]
    model_name: str = 'qwen'
    model_id: int = 0

    class Config:
        protected_namespaces = ()


class FuzzyMatchRequest(BaseModel):
    texts: List[str]
    terms: List[str]
    top_n: int = 3
    cutoff: float = 0.6
    method: str = 'levenshtein'


class PlatformEnum(str, PyEnum):
    baidu = "baidu"
    ali = "ali"
    dashscope = "dashscope"


class ToolRequest(BaseModel):
    messages: Optional[List[ChatMessage]] = Field(None)
    tools: Optional[List[Any]] = Field(None)
    prompt: Optional[str] = Field(default=None)
    model_name: str = 'moonshot'
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
                "model_name": "moonshot",
                "model_id": -1,
                "top_p": 0.95,
                "temperature": 0.01
            }
        }


class AssistantToolsEnum(str, PyEnum):
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


class CompletionParams(BaseModel):
    stream: bool = Field(False, description="Whether to stream the response")
    temperature: float = Field(0.8, description="Temperature for response generation")
    top_p: float = Field(0.8, description="The probability threshold setting for the model.")
    max_tokens: Optional[int] = Field(1024, description="Maximum number of tokens the model can generate.")

    prompt: Optional[str] = Field(default=None,
                                  description="The initial system content or prompt used to guide the AI's response.")
    question: Optional[str] = Field(None, description="The primary question or prompt for the AI to respond to. ")
    agent: Optional[str] = Field(default=None,
                                 description="System content identifier. This index represents different scenarios or contexts for AI responses, allowing the selection of different system content.")
    suffix: Optional[str] = Field(None, description="The suffix for the AI to respond to completion. ")
    extract: Optional[str] = Field(None,
                                   description="Response Format,The type of content to extract from response(e.g., code.python,code.bash,code.cpp,code.sql,json,header,links)")

    model_name: str = Field("moonshot",
                            description=("Specify the name of the model to be used. It can be any available model, "
                                         "such as 'moonshot', 'glm', 'qwen', 'ernie', 'hunyuan', 'doubao','spark','baichuan','deepseek', or other models."))
    model_id: int = Field(0, description="Model ID to be used")

    keywords: Optional[List[Union[str, Tuple[str, Any]]]] = Field(
        None, description=(
            "A list of keywords or tuples of (keyword, function, *args) used to search for relevant information across various sources, "
            "such as online searches, database queries, or vector-based search systems. "
            "These keywords help guide the retrieval of data based on the specific terms provided."))
    tools: Optional[List[Tuple[str, Any]]] = Field(
        None, description=(
            "A list of tools represented as tuples, where each tuple consists of a callable and its corresponding arguments. "
            "This allows the AI to call specific functions with the provided arguments to perform tasks such as data processing, "
            "API calls, or other utility operations. Each tool can be invoked to enhance the AI's capabilities and provide more "
            "dynamic responses based on the context."
        ))

    # score_threshold: float = Field(default=0, ge=-1, le=1, description="The score threshold setting for the model.")
    # top_n: int = Field(10,description="The number of top results to retrieve during vector search. This determines how many of the highest-scoring items will be returned.")

    class Config:
        protected_namespaces = ()
        json_schema_extra = {
            "examples": [
                {
                    'prompt': '请解释人工智能的原理。',
                    "question": "",
                    "agent": "0",
                    "suffix": "",
                    "stream": False,
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "model_name": "silicon",
                    "model_id": 0,
                    "extract": "code.python",
                    "max_tokens": 4096,
                    "keywords": ["AI智能"],
                    "tools": [],  # [("tool_name", {"key": "value"})]
                },
                {
                    "stream": False,
                    "extract": "wechat",
                    "model_name": "doubao",
                    "model_id": -1,
                    "prompt": "",
                    "agent": "42",
                    "top_p": 0.8,
                    "question": "这是什么啊,可以描述一下吗?",
                    "keywords": [],
                    "suffix": "",
                    "temperature": 0.7,
                    "max_tokens": 4096,
                    "tools": []
                }
            ]
        }

    @field_validator("model_name")
    @classmethod
    def validate_model(cls, value):
        if not MODEL_LIST.contains(value):
            raise ValueError(f"Model '{value}' is not in the supported models list {MODEL_LIST.models}")
        return value

    def asdict(self):
        return self.model_dump()  # .dict()

    def payload(self):
        return self.model_dump(include={'temperature', 'top_p', 'max_tokens', 'stream'})
        # {k: v for k, v in self.dict().items() if k in ['temperature', 'top_p', 'max_tokens', 'stream']}


class SubmitMessagesRequest(BaseModel):
    uuid: Optional[str] = None
    username: Optional[str] = None
    robot_id: Optional[str] = None
    user_id: Optional[str] = None
    use_hist: bool = Field(default=False, description="Use historical messages.")
    filter_limit: Optional[int] = Field(-500,
                                        description="The limit count(<0) or max len(>0) to filter historical messages.")
    filter_time: Optional[int] = Field(0, description="The timestamp to filter historical messages.")

    messages: Optional[List[ChatMessage]] = Field(None,
                                                  description="A list of message objects representing the current conversation. "
                                                              "If no messages are provided and `use_hist` is set to `True`, "
                                                              "the system will filter existing chat history using the fields "
                                                              "`username`, `user_id`, and `filter_time`. "
                                                              "If `messages` are provided, the last user message will be used as the question.")

    params: Optional[List[CompletionParams]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "uuid": None,
                "username": "test",
                "robot_id": None,
                "user_id": None,
                "use_hist": False,
                "filter_limit": -500,
                "filter_time": 0.0,
                "messages": [
                    {
                        "role": "user",
                        "content": "你好，我的朋友"
                    }
                ],
                "params": [{
                    "stream": False,
                    "temperature": 0.8,
                    "top_p": 0.8,
                    "max_tokens": 1024,
                    "prompt": "",
                    "question": "",
                    "keywords": [],
                    "tools": [],
                    "agent": "0",
                    "extract": "json",
                    "model_name": "moonshot",
                    "model_id": 0,
                    # "score_threshold": 0.0,
                    # "top_n": 10,
                }]
            }
        }


class ChatCompletionRequest(CompletionParams):
    uuid: Optional[str] = None
    username: Optional[str] = None
    robot_id: Optional[str] = None
    user_id: Optional[str] = None
    use_hist: bool = Field(default=False, description="use historical messages.")
    filter_limit: Optional[int] = Field(-500,
                                        description="The limit count(<0) or max len(>0) to filter historical messages.")
    filter_time: float = Field(default=0, description="The timestamp to filter historical messages.")

    messages: Optional[List[ChatMessage]] = Field(None,
                                                  description="A list of message objects representing the current conversation. "
                                                              "If no messages are provided and `use_hist` is set to `True`, "
                                                              "the system will filter existing chat history using the fields "
                                                              "`username`, `user_id`, and `filter_time`. "
                                                              "If `messages` are provided, the last user message will be used as the question.")
    question: Optional[str] = Field(None,
                                    description="The primary question or prompt for the AI to respond to. "
                                                "If `messages` are provided, this field will be automatically overridden by "
                                                "the last user message content in `messages`. Otherwise, this `question` field "
                                                "will be used directly as the prompt for the AI.")

    class Config:
        json_schema_extra = {
            "example": {
                "uuid": None,
                "username": "test",
                "robot_id": None,
                "user_id": None,

                "use_hist": False,
                "filter_limit": -500,
                "filter_time": 0.0,
                "agent": "0",
                "model_name": "moonshot",
                "model_id": 0,
                "prompt": '',
                "question": "什么是区块链金融?",
                "messages": [],
                "keywords": [],
                "tools": [],
                "extract": 'json',
                "stream": False,
                "temperature": 0.4,
                "top_p": 0.8,
                "max_tokens": 1024,
                # "score_threshold": 0.0,
                # "top_n": 10,
            }
        }


class ClassifyRequest(BaseModel):
    query: str
    class_terms: Dict[str, List[str]]
    class_default: Optional[str] = Field(None, description="default or last history to fallback.")

    # robot_id: str = None
    # user_id: str = None
    emb_model: Optional[str] = "text-embedding-v2"
    rerank_model: Optional[str] = "BAAI/bge-reranker-v2-m3"
    llm_model: Optional[str] = 'moonshot'
    prompt: Optional[str] = None

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
                "llm_model": 'moonshot',
                'prompt': ('你是群聊中的智能助手。任务是根据给定内容，识别并分类用户的意图，并返回相应的 JSON 格式，例如：{"intent":"xx"}'
                           '对于意图分类之外的任何内容，请归类为 "聊天",如果用户输入的内容不属于意图类别，直接返回 `{"intent": "聊天"}`，即表示这条内容不涉及明确的工作任务或查询。'
                           '以下是常见的意图类别与对应可能的关键词或者类似的意思，请帮我判断用户意图:')
            }
        }
