from pydantic import BaseModel, Field
from typing import Optional, List
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
    eth_address: str
    signed_message: str
    original_message: str


class Registration(AuthRequest):
    username: str
    password: str
    role: Optional[str] = 'user'
    group: Optional[str] = '0'

    public_key: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "username": "test_user",
                "password": "secure_password",
                "role": "user",
                "public_key": "your_public_key_here",
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


class TranslateRequest(BaseModel):
    text: str = "你好我的朋友。"
    source: str = "auto"
    target: str = "en"
    platform: str = "baidu"


class PlatformEnum(str, PyEnum):
    baidu = "baidu"
    ali = "ali"
    dashscope = "dashscope"


class AIParams(BaseModel):
    stream: bool = Field(False, description="Whether to stream the response")
    temperature: float = Field(0.8, description="Temperature for response generation")
    top_p: float = Field(0.8, description="The probability threshold setting for the model.")
    max_tokens: Optional[int] = Field(1024, description="Maximum number of tokens the model can generate.")

    prompt: Optional[str] = Field(default=None,
                                  description="The initial system content or prompt used to guide the AI's response.")
    question: Optional[str] = Field(None, description="The primary question or prompt for the AI to respond to. ")
    agent: Optional[str] = Field(default='0',
                                 description="System content identifier. This index represents different scenarios or contexts for AI responses, allowing the selection of different system content.")
    extract: Optional[str] = Field(None,
                                   description="The type of content to extract from response(e.g., code.python,code.bash,code.cpp,code.sql,json,header,links)")

    model_name: str = Field("moonshot",
                            description="Specify the name of the model to be used. It can be any available model, such as 'moonshot', 'glm', 'qwen', 'ernie', 'hunyuan', 'doubao', or other models.")
    model_id: int = Field(0, description="Model ID to be used")

    keywords: Optional[List[str]] = Field(None,
                                          description="A list of keywords used to search for relevant information across various sources, such as online searches, database queries, or vector-based search systems. These keywords help guide the retrieval of data based on the specific terms provided.")
    score_threshold: float = Field(default=0, ge=-1, le=1, description="The score threshold setting for the model.")
    top_n: int = Field(10,
                       description="The number of top results to retrieve during vector search. This determines how many of the highest-scoring items will be returned.")

    class Config:
        protected_namespaces = ()

    def asdict(self):
        return self.dict()

    def payload(self):
        return {k: v for k, v in self.dict().items() if
                k in ['temperature', 'top_p', 'max_tokens', 'stream']}


class CompletionRequest(AIParams):
    prompt: str

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "请解释区块链的原理。",
                "stream": 0,
                "temperature": 0.7,
                "top_p": 0.8,
                "model_name": "gpt",
                "model_id": 1,
                "extract": "code.python",
                "max_tokens": 1024
            }
        }


class SubmitMessagesRequest(BaseModel):
    uuid: Optional[str] = None
    username: Optional[str] = None
    user_id: Optional[str] = None
    use_hist: bool = Field(default=False, description="use historical messages.")
    filter_time: Optional[int] = Field(0, description="The timestamp to filter historical messages.")

    messages: Optional[List[Message]] = Field(None,
                                              description="A list of message objects representing the current conversation. "
                                                          "If no messages are provided and `use_hist` is set to `True`, "
                                                          "the system will filter existing chat history using the fields "
                                                          "`username`, `user_id`, and `filter_time`. "
                                                          "If `messages` are provided, the last user message will be used as the question.")

    params: Optional[List[AIParams]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "uuid": "",
                "username": "test",
                'user_id': "test_2",
                "use_hist": False,
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
                    "agent": "0",
                    "extract": "json",
                    "model_name": "moonshot",
                    "model_id": 0,
                    "keywords": [],
                    "score_threshold": 0.0,
                    "top_n": 10,
                }]
            }
        }


class OpenAIRequest(AIParams):
    # max_tokens: int = 300
    uuid: Optional[str] = None
    username: Optional[str] = None
    user_id: Optional[str] = None
    use_hist: bool = Field(default=False, description="use historical messages.")
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
                "uuid": "",
                "username": "test",
                'user_id': "test_1",
                "use_hist": False,
                "agent": "0",
                "model_name": "moonshot",
                "model_id": 0,
                "prompt": '',
                "messages": [],
                "question": "什么是区块链金融?",
                "extract": 'json',
                "stream": False,
                "filter_time": 0.0,
                "temperature": 0.4,
                "top_p": 0.8,
                "keywords": [],
                "score_threshold": 0.0,
                "top_n": 10,
                "max_tokens": 1024
            }
        }
