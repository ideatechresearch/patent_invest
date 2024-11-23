from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Tuple, Union, Any
from enum import Enum as PyEnum


class TaskStatus(PyEnum):
    PENDING = "pending"
    IN_PROGRESS = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RECEIVED = "received"


class OpenAIResponse(BaseModel):
    response: str


class Message(BaseModel):
    role: str
    content: str
    # name: str = None


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


class ToolRequest(BaseModel):
    messages: Optional[List[Message]] = Field(None)
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


class PlatformEnum(str, PyEnum):
    baidu = "baidu"
    ali = "ali"
    dashscope = "dashscope"


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
                                   description="The type of content to extract from response(e.g., code.python,code.bash,code.cpp,code.sql,json,header,links)")

    model_name: str = Field("moonshot",
                            description=("Specify the name of the model to be used. It can be any available model, "
                                         "such as 'moonshot', 'glm', 'qwen', 'ernie', 'hunyuan', 'doubao','speark','baichuan', or other models."))
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
            "example": {
                'prompt': '请解释区块链的原理。',
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
                "keywords": ["区块链"],
                "tools": [],
            }
        }

    def asdict(self):
        return self.dict()

    def payload(self):
        return {k: v for k, v in self.dict().items() if
                k in ['temperature', 'top_p', 'max_tokens', 'stream']}


class SubmitMessagesRequest(BaseModel):
    uuid: Optional[str] = None
    username: Optional[str] = None
    robot_id: Optional[str] = None
    user_id: Optional[str] = None
    use_hist: bool = Field(default=False, description="Use historical messages.")
    filter_limit: Optional[int] = Field(-500,
                                        description="The limit count(<0) or max len(>0) to filter historical messages.")
    filter_time: Optional[int] = Field(0, description="The timestamp to filter historical messages.")

    messages: Optional[List[Message]] = Field(None,
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


class OpenAIRequest(CompletionParams):
    uuid: Optional[str] = None
    username: Optional[str] = None
    robot_id: Optional[str] = None
    user_id: Optional[str] = None
    use_hist: bool = Field(default=False, description="use historical messages.")
    filter_limit: Optional[int] = Field(-500,
                                        description="The limit count(<0) or max len(>0) to filter historical messages.")
    filter_time: float = Field(default=0, description="The timestamp to filter historical messages.")

    messages: Optional[List[Message]] = Field(None,
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
