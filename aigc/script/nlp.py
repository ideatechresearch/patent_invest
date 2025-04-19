from generates import *
import difflib

class Classifier:
    def __init__(self, class_terms: Dict[str, List[str]], intent_history_manager, bm25_model, llm_client):
        """
        :param class_terms: 关键词定义，每个意图类对应一个关键词列表。
        :param intent_history_manager: 管理意图历史记录的对象。
        :param bm25_model: 用于 BM25 排序的对象。
        :param llm_client: 用于调用大语言模型的客户端。
        """
        self.class_terms = class_terms
        self.intent_history = intent_history_manager
        self.bm25 = bm25_model
        self.llm_client = llm_client

    def match_by_keywords(self, query: str) -> Optional[Dict[str, Any]]:
        """
        使用关键词直接匹配意图。
        :param query: 用户输入的查询字符串。
        :return: 匹配的意图和相关信息，或 None 如果没有匹配到。
        """
        for intent, keywords in self.class_terms.items():
            for keyword in keywords:
                if keyword in query:
                    return {"intent": intent, "score": None,"type": "keyword_match"}
        return None

    def match_by_regex(self, query: str) -> Optional[Dict[str, Any]]:
        """通过关键词匹配意图"""
        for intent, keywords in self.class_terms.items():
            if any(re.search(r'\b' + re.escape(keyword) + r'\b', query) for keyword in keywords):
                return {"class": intent, "score": None, "type": "regex_match"}
        return None

    def match_by_similarity(self, query: str) -> List[Dict[str, Any]]:
        """通过相似度匹配意图"""
        intent_tokens = [(intent, x.strip()) for intent, keywords in self.class_terms.items() for x in keywords]
        tokens = [token for _, token in intent_tokens]
        matches = difflib.get_close_matches(query, tokens, n=10, cutoff=0.8)
        return [
            {"class": intent_tokens[tokens.index(match)][0], "score": difflib.SequenceMatcher(None, query, match).ratio(), "type": "similarity"}
            for match in matches
        ]

    def match_by_bm25(self, query: str) -> Optional[Dict[str, Any]]:
        """通过 BM25 匹配意图"""
        intent_tokens = [(intent, x.strip()) for intent, keywords in self.class_terms.items() for x in keywords]
        tokens = [token for _, token in intent_tokens]
        scores = self.bm25.rank_documents(query, sort=True)
        if scores:
            best_match = max(scores, key=lambda x: x[1])
            intent = intent_tokens[best_match[0]][0]
            return {"class": intent, "score": best_match[1], "type": "bm25"}
        return None

    async def match_by_llm(self, query: str, prompt: str, llm_model: str) -> Optional[Dict[str, Any]]:
        """通过大模型匹配意图"""
        response = await self.llm_client(query=query, prompt=prompt, model_name=llm_model)
        if response:
            intent = response.get("intent")
            if intent:
                return {"class": intent, "score": None, "type": "llm"}
        return None

    def handle_short_query(self, query: str) -> Optional[Dict[str, Any]]:
        """处理短文本为闲聊意图"""
        chat_keywords = ["你好", "您好", "问候", "闲聊", "随便聊聊", "帮忙", "请求帮助", "请教", "咨询"]
        if len(query) < 8 and any(re.search(r'\b' + re.escape(keyword) + r'\b', query) for keyword in chat_keywords):
            return {"class": "聊天", "type": "short_query"}
        return None