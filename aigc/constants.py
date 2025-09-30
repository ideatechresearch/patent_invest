from typing import Final

MODEL: Final[str] = "model"
EMBEDDING: Final[str] = "embedding"
RERANKER: Final[str] = "reranker"

VIEW: Final[str] = "view"
KEYWORDS: Final[str] = "keywords"
PARAMS: Final[str] = "params"
EXAMPLES: Final[str] = "examples"

NAME: Final[str] = "name"
USER: Final[str] = "user"
USER_ID: Final[str] = "user_id"
USERNAME: Final[str] = "username"
TIMESTAMP: Final[str] = "timestamp"
CONTEXT: Final[str] = "context"


class Constants:
    MODEL = "model"
    VIEW = "view"

    # 防止修改常量值
    def __setattr__(self, name, value):
        raise AttributeError("常量不可修改")


Title_Map: Final[dict] = {'question': '问题', 'reference': '参考', 'refer': '参考', 'answer': '回答', 'content': '内容',
                          'transform': '转换', 'reason': '原因', 'suggest': '建议', 'title': '标题',
                          "result": "结果","error": "错误"}
