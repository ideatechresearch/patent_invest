from utils import load_dictjson, save_dictjson, extract_json_struct, ts2datetime, str_to_ts
from datetime import datetime
import time
import pandas as pd


def extract_conversations_records_gpt(conversations: list):
    structured = []

    for convo in conversations:
        mapping = convo.get("mapping", {})
        start_nodes = []

        # 找出 root 节点的 children，跳过 message=None 的 root
        roots = [node_id for node_id, node in mapping.items() if node.get("parent") is None]
        for root_id in roots:
            root = mapping[root_id]
            children = root.get("children", [])
            start_nodes.extend(children)

        stack = list(start_nodes)
        messages = []
        index = 0
        while stack:
            node_id = stack.pop(0)
            node = mapping.get(node_id)
            if not node:
                continue
            msg = node.get("message")
            if not msg:
                continue

            _id = msg.get("id")
            role = msg["author"]["role"]
            create_time = msg.get("create_time")
            model_slug = msg.get("metadata", {}).get("model_slug")  # 有的没有
            content_block = msg.get("content", {})
            content_type = content_block.get("content_type")

            if content_type == "text" and "parts" in content_block:
                content = "\n\n---\n\n".join(content_block["parts"])
            else:
                content = f"[非文本内容: {content_type}]"

            messages.append({
                "role": role,
                "content": content,
                "create_time": create_time,
                "model": model_slug,
                "id": _id,
                "search_results": None,
                "index": index
            })
            index += 1

            stack.extend(node.get("children", []))

        # 保留至少一个消息的会话
        if messages:
            structured.append({
                "title": convo.get("title") or convo.get("id"),
                "create_time": convo.get("create_time"),
                "update_time": convo.get("update_time"),
                "messages": messages
            })

    return structured


def extract_conversations_records_ds(conversations: list):
    structured = []
    for convo in conversations:
        mapping = convo.get("mapping", {})
        start_nodes = []

        # 找出 root 节点的 children，跳过 message=None 的 root
        roots = [node_id for node_id, node in mapping.items() if node.get("parent") is None]
        for root_id in roots:
            root = mapping.get(root_id, {})
            children = root.get("children", [])
            start_nodes.extend(children)

        stack = list(start_nodes)
        messages = []
        index = 0
        while stack:
            node_id = stack.pop(0)
            node = mapping.get(node_id)
            if not node:
                continue
            msg = node.get("message")
            if not msg:
                # stack.extend(node.get("children", []) or [])
                continue

            create_time = datetime.fromisoformat(msg.get("inserted_at")).timestamp()
            model = msg.get("model")  # 有的没有

            fragments = msg.get("fragments") or []
            frag_types = [(frag.get("type") or "").upper() for frag in fragments]
            role = "user" if all(ft == "REQUEST" for ft in frag_types) else "assistant"
            if role == "user":
                content = "\n\n".join([frag.get("content") for frag in fragments if frag.get("content")])
                search_results = None
            else:
                # fragments：逐条展开
                parts = []
                search_results = []
                for frag in fragments:
                    ftype = (frag.get("type") or "").upper()
                    text = frag.get("content")
                    if not text and ftype != "SEARCH":
                        # 非 SEARCH 且无文本，则跳过
                        continue
                    if ftype == "THINK":
                        parts.append(f"<think>{text}</think>")
                    elif ftype == "RESPONSE":
                        parts.append(f"{text}")
                    elif ftype == "SEARCH":
                        if frag.get("results") is not None:
                            search_results += frag.get("results", [])
                    else:
                        # 其它类型，直接追加文本
                        parts.append(text)

                content = "\n\n---\n\n".join([p for p in parts if p])
                if not search_results:
                    search_results = None

            messages.append({
                "role": role,
                "content": content,
                "create_time": create_time,
                "model": model,
                "search_results": search_results,
                "index": index
            })
            index += 1
            stack.extend(node.get("children", []))

        # 保留至少一个消息的会话
        if messages:
            structured.append({
                "title": convo.get("title") or convo.get("id"),
                "create_time": datetime.fromisoformat(convo.get("inserted_at")).timestamp(),
                "update_time": datetime.fromisoformat(convo.get("updated_at")).timestamp(),
                "messages": messages
            })

    return structured


def filter_messages_after(structured_data: list, after_timestamp: int | float = 0):
    result = []

    for convo in structured_data:
        title = convo.get("title")
        create_time = convo.get("create_time")
        update_time = convo.get("update_time")

        for i, msg in enumerate(convo["messages"]):
            created_timestamp = msg.get("create_time")
            if created_timestamp and created_timestamp > after_timestamp:
                result.append({
                    "role": msg["role"],
                    "content": msg["content"],
                    "search_results": msg.get("search_results", None),
                    "created_timestamp": created_timestamp,
                    "model": msg.get("model"),

                    "title": title,
                    "create_time": create_time,
                    "update_time": update_time,
                    "index": msg.get("index", i),
                })

    return result


def df_messages_sorted(messages: list[dict]):
    df = pd.DataFrame(messages)

    df["create_time_dt"] = df["create_time"].apply(ts2datetime)
    df["update_time_dt"] = df["update_time"].apply(ts2datetime)
    df_sorted = df.sort_values(by=["create_time_dt", "title", "update_time_dt", "index", "created_timestamp"])

    df_sorted["create_time"] = df_sorted["create_time_dt"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df_sorted["update_time"] = df_sorted["update_time_dt"].dt.strftime("%Y-%m-%d %H:%M:%S")

    print('messages len sum:', df_sorted["content"].str.len().sum())
    return df_sorted.drop(columns=["create_time_dt", "update_time_dt"])


def messages_to_records_values(df_sorted, user: str, name: str) -> list[dict]:
    """
    将外部的消息格式转换为与表列名一致的 dict 列表。
    假设表字段包含: user, agent, role, timestamp, model, content, search_results, reference
    create_time/create_at -> timestamp (unix seconds)
    """
    if df_sorted is None or df_sorted.empty:
        return []

    df_filtered = df_sorted.loc[(df_sorted['role'] != 'system') & (df_sorted['content'].str.len() > 0)].copy()
    if df_filtered.empty:
        return []

    now_ts = int(time.time())
    df_filtered['user'] = user
    df_filtered['name'] = name
    df_filtered["timestamp"] = df_filtered["created_timestamp"].apply(lambda v: str_to_ts(v, now_ts))
    df_filtered["created_at"] = df_filtered["timestamp"].apply(ts2datetime)
    df_filtered.rename(columns={"title": "agent", "search_results": "reference"}, inplace=True)
    cols = [c for c in
            ("user", "name", "agent", "role", "timestamp", "model", "content", "reference", "created_at", "index") if
            c in df_filtered.columns]

    return df_filtered[cols].where(pd.notna(df_filtered[cols]), None).to_dict(orient="records")


if __name__ == "__main__":
    def main():
        conversations = load_dictjson("data/conversations.json")
        structured_data = extract_conversations_records_gpt(conversations)
        save_dictjson(structured_data, 'data/conversations_structured.json')

        print(len(structured_data))
        after_ts = datetime(2025, 8, 1).timestamp()
        filtered_messages = filter_messages_after(structured_data, after_ts)
        df_sorted = df_messages_sorted(filtered_messages)
        df_sorted.to_excel('data/extract_conversations_2508.xlsx')
