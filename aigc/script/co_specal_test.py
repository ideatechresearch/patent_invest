import requests
import threading
import json, re, time
from queue import Queue
from utils import extract_json_struct
import pymysql
import pandas as pd

url_batch = "http://localhost:8080/api/start_task_batch"
co_name = [
    # "云南维普建设项目管理有限公司勐海分公司",
    # "勐腊万泽农业科技发展有限公司",
    # "昌宁宇航建筑装饰有限公司",
    # "景东德宇林农产品开发有限公司",
    # "云南济华建筑装饰工程有限公司",
    # "云南大众安保服务集团有限公司",
    # "普洱市基础设施投资开发管理有限公司",
    # "石林国有资本投资有限公司",
    # "西双版纳蓝帆诺城建设工程有限公司",
    # "玉溪泰航建设工程有限公司",
    # "云南柏源建筑工程有限公司"

    # "昌宁宇航建筑装饰有限公司"
    # "云南恒梓石化有限公司", "江苏旺德建筑工程有限公司"
    "昌宁宇航建筑装饰有限公司"
]
url = "http://localhost:8080/test/special-conclusion/execute"
# batchNos = [
#     {"batch_no": "TASK_1755244011158_25d6192c", "channel_id": None, "record_id": None},
#     {"batch_no": "TASK_1755495454019_f2f30569", "channel_id": None, "record_id": None},
#     # "TASK_1755244011388_61270f38",
#     {"batch_no": "TASK_1755495575215_a7611e0e", "channel_id": None, "record_id": None},
#     # "TASK_1755244011757_8a8a3ace",
#     {"batch_no": "TASK_1755244012011_ff4862ed", "channel_id": None, "record_id": None},
#     {"batch_no": "TASK_1755244012104_b371c5e8", "channel_id": None, "record_id": None},
#     {"batch_no": "TASK_1755244012224_2d7a735f", "channel_id": None, "record_id": None},
#     {"batch_no": "TASK_1755244012332_c5988795", "channel_id": None, "record_id": None},
#     {"batch_no": "TASK_1755244012439_54da3444", "channel_id": None, "record_id": None},
#     {"batch_no": "TASK_1755244012605_33d2ec58", "channel_id": None, "record_id": None},
#     {"batch_no": "TASK_1755244012725_efcdc5c4", "channel_id": None, "record_id": None},
#     {"batch_no": "TASK_1755244012841_deb760c0", "channel_id": None, "record_id": None}
# ]
# batchNos_INPUT = [
#     {
#         "batch_no": "TASK_1755504009191_50c795e3",
#
#     },
#     {
#         "batch_no": "TASK_1755504009291_b3ac34a8",
#
#     },
#     {
#         "batch_no": "TASK_1755504009346_c7159dfb",
#
#     },
#     {
#         "batch_no": "TASK_1755504009398_8b40eb9a",
#
#     },
#     {
#         "batch_no": "TASK_1755504009454_cc25934e",
#
#     },
#     {
#         "batch_no": "TASK_1755504009504_05181463",
#
#     },
#     {
#         "batch_no": "TASK_1755504009562_4eda8cea",
#
#     },
#     {
#         "batch_no": "TASK_1755504009611_dfceafc3",
#
#     },
#     {
#         "batch_no": "TASK_1755504009664_66ed5a1f",
#
#     },
#     {
#         "batch_no": "TASK_1755504009717_66c93696",
#
#     },
#     {
#         "batch_no": "TASK_1755504009765_2e24dc7f",
#
#     }
# ]

# [{"batch_no":"TASK_1755507698210_e53f0fb6","channel_id":null,"record_id":null},{"batch_no":"TASK_1755507698261_6b70d8c6","channel_id":null,"record_id":null},{"batch_no":"TASK_1755507698313_abc77983","channel_id":null,"record_id":null},{"batch_no":"TASK_1755507698367_0e29d3cc","channel_id":null,"record_id":null},{"batch_no":"TASK_1755507698420_cc3f05bf","channel_id":null,"record_id":null},{"batch_no":"TASK_1755507698473_2806fc01","channel_id":null,"record_id":null},{"batch_no":"TASK_1755507698523_1ea62bc2","channel_id":null,"record_id":null},{"batch_no":"TASK_1755507698572_668c1158","channel_id":null,"record_id":null},{"batch_no":"TASK_1755507698622_12c123d3","channel_id":null,"record_id":null},{"batch_no":"TASK_1755507698670_e0a2ebf9","channel_id":null,"record_id":null},{"batch_no":"TASK_1755507698714_9d4c96f6","channel_id":null,"record_id":null}]
# batchNos_INPUT = []  # [{"batch_no": "TASK_1755660570955_b75f3d8a"}, {"batch_no": "TASK_1755660571067_27450a39"}],[{"batch_no":"TASK_1755748760481_892948e3","channel_id":null,"record_id":null},{"batch_no":"TASK_1755748760605_055d8ea8","channel_id":null,"record_id":null}]
# [{"batch_no":"TASK_1755850350934_e2e3ca2d","channel_id":null,"record_id":null},{"batch_no":"TASK_1755850351031_8af42087","channel_id":null,"record_id":null},{"batch_no":"TASK_1755850351090_08a33ab0","channel_id":null,"record_id":null},{"batch_no":"TASK_1755850351143_3191f1f7","channel_id":null,"record_id":null},{"batch_no":"TASK_1755850351190_057c820b","channel_id":null,"record_id":null},{"batch_no":"TASK_1755850351238_ed82721b","channel_id":null,"record_id":null},{"batch_no":"TASK_1755850351293_97b742a0","channel_id":null,"record_id":null},{"batch_no":"TASK_1755850351377_81100c7e","channel_id":null,"record_id":null},{"batch_no":"TASK_1755850351429_5e349073","channel_id":null,"record_id":null},{"batch_no":"TASK_1755850351477_9e6b7448","channel_id":null,"record_id":null},{"batch_no":"TASK_1755850351527_e0c712f4","channel_id":null,"record_id":null}]
#[['昌宁宇航建筑装饰有限公司']] -> 200 [{"batch_no":"TASK_1755852231761_9ac492b1","channel_id":null,"record_id":null}]
# batchNos_INPUT =[{"batch_no":"TASK_1755850350934_e2e3ca2d",},{"batch_no":"TASK_1755850351031_8af42087",},{"batch_no":"TASK_1755852231761_9ac492b1",},{"batch_no":"TASK_1755850351143_3191f1f7",},{"batch_no":"TASK_1755850351190_057c820b",},
#                  {"batch_no":"TASK_1755850351238_ed82721b",},{"batch_no":"TASK_1755850351293_97b742a0",},{"batch_no":"TASK_1755850351377_81100c7e",},{"batch_no":"TASK_1755850351429_5e349073",},{"batch_no":"TASK_1755850351477_9e6b7448",},
#                  {"batch_no":"TASK_1755850351527_e0c712f4",}]
batchNos_INPUT =[{"batch_no": "TASK_1755944881567_85c838fa"}]
DB_CONFIG = {
    "host": "rm-bp17785oqv814s5cjeo.mysql.rds.aliyuncs.com",  # 数据库地址
    "port": 3306,  # 端口
    "user": "llm_infr",  # 用户名
    "password": "llm_infr123",  # 密码
    "database": "llm_infr_test",  # 库名
    "charset": "utf8mb4"
}
questionNos = ["SC001", "SC002", "SC003", "SC004", "SC005", "SC006"]
headers = {
    'Authorization': 'Bearer eyJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjozMzI4MDg0MTQ0ODIxMjA3MDQsInVzZXJfbmFtZSI6ImFkbWluIiwiaWF0IjoxNzU1ODQ3MzUxLCJzdWIiOiJhZG1pbiIsImV4cCI6MTc1NTkzMzc1MX0.tpSSCsJ-TuldVEc8aeSP6YpGxQNsYjYWn-vyVGhJjmQ'}

max_concurrency = 6
file_n = '7'
results = []
lock = threading.Lock()


def submit_batch(co_name):
    """提交批量任务，返回 batchNos"""
    try:
        r = requests.post(url_batch, json=co_name, headers=headers, timeout=100)
        print(f"[{co_name}] -> {r.status_code} {r.text}")
        return r.json()
    except Exception as e:
        print(f"[{co_name}] 请求失败: {e}")
        return []


if not batchNos_INPUT:
    batchNos = submit_batch(co_name)
else:
    batchNos = batchNos_INPUT

if not batchNos:
    exit(0)

if not batchNos_INPUT:
    time.sleep(len(co_name) * 90)  # 15 * 60

batch_nos = [b["batch_no"] for b in batchNos]


def call_api(batch_no, question_no):
    data = {
        "batchNo": batch_no,
        "questionNo": question_no
    }
    try:
        r = requests.post(url, json=data, headers=headers, timeout=300)
        print(f"[{batch_no} | {question_no}] -> {r.status_code} {r.text}")
        with lock:
            results.append({
                "batchNo": batch_no,
                "questionNo": question_no,
                "status_code": r.status_code,
                "response": r.json() if r.headers.get("Content-Type", "").startswith("application/json") else r.text
            })
    except Exception as e:
        print(f"[{batch_no} | {question_no}] 请求失败: {e}")
        with lock:
            results.append({
                "batchNo": batch_no,
                "questionNo": question_no,
                "error": str(e)
            })


def worker(q):
    while True:
        item = q.get()
        if item is None:
            break
        batch_no, qn = item
        call_api(batch_no, qn)
        q.task_done()


q = Queue()

threads = []
for i in range(max_concurrency):
    t = threading.Thread(target=worker, args=(q,))
    t.start()
    threads.append(t)

for no in batch_nos:
    for qn in questionNos:
        q.put((no, qn))

q.join()

for _ in range(max_concurrency):
    q.put(None)
for t in threads:
    t.join()

print("全部请求完成 ✅")

with open(f"data/co_results{file_n}.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

map_id = {no: c for no, c in zip(batch_nos, co_name)}

print(sum([x['response']['success'] for x in results]))

extracted = [
    {
        "batchNo": r["batchNo"],
        "questionNo": r["questionNo"],
        "originQuestion": r["response"]["data"]["originQuestion"],
        "result": r["response"]["data"]["result"],
        "companyName": map_id.get(r["batchNo"], 'UNKNOWN'),
        **extract_json_struct(r["response"]["data"]["result"])
    }
    for r in results
    if r["response"].get("success")
]

sorted_extracted = sorted(extracted, key=lambda x: (x["batchNo"], x["questionNo"]))

with open(f"data/co_results_data{file_n}.json", "w", encoding="utf-8") as f:
    json.dump(sorted_extracted, f, ensure_ascii=False, indent=2)

print(len(sorted_extracted))

print("=== API调用结果 ===")
for r in results:
    print(r)


def clean_markdown_text(md_text):
    # 删除孤立的 YAML 区块起始符号（如 "---"）
    md_text = re.sub(r'(?m)^---\s*$', '', md_text)
    # 将 /n/n/ 或 /n/n 替换成真正的换行
    md_text = md_text.replace('/n/n/', '\n\n').replace('/n/n', '\n\n')
    # 替换掉可能影响 Pandoc 的控制符
    md_text = md_text.replace('\r', '')
    return md_text.encode('utf-8', errors='ignore').decode('utf-8')


placeholders = ",".join(["%s"] * len(batch_nos))

SQL = f"""
SELECT batch_no,summary_answer
FROM task_summary_question
WHERE batch_no IN ({placeholders})
ORDER BY batch_no, id DESC;
"""
conn = pymysql.connect(**DB_CONFIG)

try:
    # df_summary_answer = pd.read_sql(SQL, conn, params=batch_nos)
    with conn.cursor() as cursor:
        cursor.execute(SQL, batch_nos)
        rows = cursor.fetchall()
        cols = [desc[0] for desc in cursor.description]
        df_summary_answer = pd.DataFrame(rows, columns=cols)
finally:
    conn.close()

df_struct = pd.concat([df_summary_answer,
                       pd.json_normalize(df_summary_answer['summary_answer'].apply(extract_json_struct))], axis=1)

df_result = pd.merge(pd.DataFrame(sorted_extracted), df_struct, left_on='batchNo', right_on='batch_no', how='left')


def parse_total_from_score_table(score_table):
    """
    从 score_table 的 markdown 文本中解析出总分（浮点数）。
    支持多种写法，例如：
      | **总分**     | **49.5**      |
    或行中出现 "总分" 或 "总分：" 后跟数字的场景。
    返回 float 或 None。
    """
    if not isinstance(score_table, str):
        return None
    # 1) 尝试匹配 **总分** | **49.5**
    m = re.search(r"\*\*\s*总分\s*\*\*\s*\|\s*\*\*\s*([+-]?\d+(?:\.\d+)?)\s*\*\*", score_table)
    if m:
        try:
            return float(m.group(1))
        except:
            pass
    # 2) 匹配 "总分" 后面的数字（更宽松）
    m = re.search(r"总分[^\d\-+]*([+-]?\d+(?:\.\d+)?)", score_table)
    if m:
        try:
            return float(m.group(1))
        except:
            pass
    # 3) 匹配行以 | **总分** | **49.5** | 之类
    m = re.search(r"\|\s*\*\*\s*总分\s*\*\*\s*\|\s*\*\*?\s*([+-]?\d+(?:\.\d+)?)\s*\*?\s*\|", score_table)
    if m:
        try:
            return float(m.group(1))
        except:
            pass
    return None


df_result['main_score'] = df_result['score_table'].apply(parse_total_from_score_table)
print(df_result['main_score'].unique())

df_result.to_excel(f'data/风险排查{file_n}.xlsx', index=False)
print(df_result.shape)
