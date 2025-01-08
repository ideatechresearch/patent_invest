from generates import *
from igraph import Graph
import PyPDF2


def parse_toc(pdf_path):
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        toc_page = reader.pages[0].extract_text()  # 假设第一页为目录
    toc = []
    # 正则解析目录
    toc_pattern = re.compile(r'(\d+(\.\d+)*).*?(\d+)$')
    for line in toc_page.splitlines():
        match = toc_pattern.match(line.strip())
        if match:
            title_number = match.group(1)
            title = line.strip().split(title_number)[-1].strip().rsplit(' ', 1)[0]
            # 清理标题中的无效字符，例如多余的点
            title = re.sub(r'[\s\.]+$', '', title)
            page = int(match.group(3))
            level = title_number.count('.')
            toc.append({'title_number': title_number, 'title': title, 'page': page, 'level': level})
    return toc


def extract_content(pdf_path, toc):
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        content = {}
        for idx, item in enumerate(toc):
            start_page = item['page'] - 1
            end_page = toc[idx + 1]['page'] - 1 if idx + 1 < len(toc) else len(reader.pages)
            # 按页提取内容
            text = ""
            for page_num in range(start_page, end_page):
                text += reader.pages[page_num].extract_text()
            content[item['title_number']] = {
                "title": item['title'],
                "text": text.strip(),
                "level": item['level'],
                "parent": toc[idx - 1]['title_number'] if item['level'] > 1 else None
            }
    return content


def refine_content_with_titles(content):
    for key, value in content.items():
        text = value['text']
        for sub_key, sub_value in content.items():
            # 如果子条目的 title 在当前条目的 text 中，且 text 尚未被更新
            if sub_value['title'] in text and content[sub_key]['text'] != sub_value['text']:
                content[sub_key]['text'] = sub_value['text']
    return content


def get_full_parent_path(content, current_key):
    path = []
    while current_key:
        # 获取当前节点
        node = content.get(current_key)
        if node:
            path.append(node['title'])  # current_key
            current_key = node['parent']
        else:
            break
    return " > ".join(reversed(path))  # 返回拼接的父级路径，倒序拼接


def clean_text(raw_text):
    """
    清理原始文本，去掉多余的不可见字符、异常数字等。
    """
    # 替换不可见字符（如 \xa0）为普通空格
    cleaned_text = raw_text.replace('\xa0', ' ').strip()

    # 去掉多余空格和连续空格
    cleaned_text = re.sub(r'[^\S\r\n]+', ' ', cleaned_text)  # r'\s+'

    # 将形如 "1.\n2." 的情况保留，避免误合并多个序号
    cleaned_text = re.sub(r'(\d+)\n\.\n(\d+)', r'\1.\2', cleaned_text)

    # 去掉16位数字
    # cleaned_text = re.sub(r'\d{16}\b', '', cleaned_text)
    cleaned_text = re.sub(r'\d{16}[，。、\s]*', '', cleaned_text)

    return cleaned_text


def split_paragraphs(text: str, max_length=512,
                     pattern=r'(?=[。！？])|(?=\b[一二三四五六七八九十]+\、)|(?=\b[（(][一二三四五六七八九十]+[）)])|(?=\b\d+\、)|(?=\r\n)'):
    '''
    :param text: 输入的文本
    :param max_length: 每个段落的最大长度
    :param pattern: 用于段落分割的正则表达式（根据标点或换行符）
    :return: 处理后的段落列表
    '''
    # merged_pattern = r'\b[一二三四五六七八九十]+\、|\b[（(][一二三四五六七八九十]+[）)]|\b\d+\、|^\d+\.'

    if not pattern:
        pattern = r'(?=[。！？])'

    sentences = split_sentences(text, pattern)
    paragraphs = []
    current_paragraph = ""

    for sentence in sentences:
        # 匹配结构化序号的句子合并 re.match(merged_pattern, sentence):
        # 如果当前段落已经接近长度限制，保存段落
        if len(current_paragraph) + len(sentence) > max_length:
            # 检查并处理超长段落,是否超出长度限制
            if len(current_paragraph) > max_length:
                # 超过 max_length，优先寻找标点符号处或换行分割段落,避免了无意义的截断
                sub_paragraphs = re.split(pattern, current_paragraph)
                buffer = ""
                for sub in sub_paragraphs:
                    if len(buffer) + len(sub) > max_length:
                        paragraphs.append(buffer.strip())
                        buffer = sub
                    else:
                        buffer += sub
                if buffer:
                    current_paragraph = buffer.strip()
                else:
                    current_paragraph = ""  # 重置段落
            else:
                if current_paragraph:
                    paragraphs.append(current_paragraph.strip())
                current_paragraph = sentence  # 开始新的段落
        else:
            current_paragraph += sentence  # 未超出长度累积

    # 添加最后一段
    if current_paragraph:
        paragraphs.append(current_paragraph.strip())

    final_paragraphs = []
    # 对每个段落进行截断，确保每个段落不超过 max_length
    for paragraph in paragraphs:
        while len(paragraph) > max_length:
            final_paragraphs.append(paragraph[:max_length].strip())
            paragraph = paragraph[max_length:]  # 更新剩余部分
        if paragraph.strip():
            final_paragraphs.append(paragraph.strip())

    return final_paragraphs or paragraphs


def generate_embeddings(sentences: list):
    url = 'http://47.110.156.41:7000/embeddings/'
    payload = {'texts': sentences, 'model_name': 'qwen', 'model_id': 0}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()['embedding']

    return []


async def graph_icc_edge(pdf_path='data/ideatech-251124-1758-255.pdf'):
    toc = parse_toc(pdf_path)
    content = extract_content(pdf_path, toc)
    structured_content = refine_content_with_titles(content)
    for key in structured_content:
        structured_content[key]['full_path'] = get_full_parent_path(content, key)

    g = Graph(directed=True)
    # 添加节点和边
    for title_number, item in structured_content.items():
        tx = clean_text(item["text"])
        paragraphs = split_paragraphs(tx, max_length=1000,
                                      pattern=r'(?=[。！？])|(?=\b[一二三四五六七八九十]+\、)|(?=\b[（(][一二三四五六七八九十]+[）)])|(?=\b\d+\、)|(?=\r\n)')
        print(title_number, [len(x) for x in paragraphs], len(paragraphs), item["full_path"])
        # print(paragraphs)
        # paragraph_embeddings = generate_embeddings(paragraphs)
        # path_embedding = generate_embeddings(item['full_path'])

        paragraph_embeddings, path_embedding = await asyncio.gather(
            ai_embeddings(paragraphs, model_name='qwen', model_id=0),
            ai_embeddings(item['full_path'], model_name='qwen', model_id=0))

        if len(paragraphs) == 1 and len(tx) <= 8:  # "短段落"
            paragraph_embeddings = [[0.0] * 1536]

        combined_paragraph_embedding = np.mean(paragraph_embeddings, axis=0) if paragraph_embeddings else np.zeros(1536)
        # [generate_embedding(p) for p in paragraphs]

        # 添加节点
        g.add_vertex(
            name=title_number,  # 节点的唯一标识
            title=item["title"],  # 节点标题
            text=item["text"],  # 节点内容
            level=item["level"],  # 节点层级
            path=item['full_path'],
            paragraph=paragraphs,
            paragraph_embeddings=paragraph_embeddings,
            combined_paragraph_embedding=combined_paragraph_embedding.tolist(),  # 合并段落嵌入
            path_embedding=path_embedding[0]  # 路径嵌入
        )
        # 如果有父级，添加边
        if item["parent"]:
            g.add_edge(item["parent"], title_number)

    # with open(f"{Config.DATA_FOLDER}/ideatech_pdf_graph.pkl", "wb") as f:
    #     pickle.dump(g, f)

    async with aiofiles.open(f"{Config.DATA_FOLDER}/ideatech_pdf_graph.pkl", "wb") as f:
        await f.write(pickle.dumps(g))

    g.save(f"{Config.DATA_FOLDER}/ideatech_pdf_graph.graphml")
    return g


def find_similar_paragraphs(target_embedding, graph=None, node=None):
    """
    找到图或节点中与目标嵌入最相似的段落。

    :param target_embedding: 目标嵌入
    :param graph: 图对象（用于查询整个图）
    :param node: 节点对象（用于查询单个节点）
    :return: [(段落文本, 相似度), ...] 按相似度降序排列
    """
    paragraphs = []
    paragraph_embeddings = []

    if graph:
        # 查询整个图
        for v in graph.vs:
            if "paragraph" in v.attributes() and "paragraph_embeddings" in v.attributes():
                paragraphs.extend(v["paragraph"])
                paragraph_embeddings.extend(v["paragraph_embeddings"])
        # print(len(paragraphs), len(paragraph_embeddings))
    elif node:
        # 查询单个节点
        if "paragraph" in node.attributes() and "paragraph_embeddings" in node.attributes():
            paragraphs = node["paragraph"]
            paragraph_embeddings = node["paragraph_embeddings"]

    # 没有提供有效的 graph 或 node
    # 如果没有段落或嵌入，返回空结果
    if not paragraphs or not paragraph_embeddings:
        return []

    # 计算相似度
    similarities = cosine_similarity_np(target_embedding, np.array(paragraph_embeddings)).flatten()
    # similarities = cosine_similarity([target_embedding], paragraph_embeddings).flatten()
    # sim_matrix = np.array(query_vector) @ np.array(tokens_vector).T
    # 按相似度降序排序
    return sorted(zip(paragraphs, similarities), key=lambda x: x[1], reverse=True)


def query_similar_content(graph, target_embedding, top_n_nodes=3, top_n_paragraphs=3):
    """
    查询图中与目标嵌入最相似的节点和段落。
    :param graph: 图对象
    :param target_embedding: 查询嵌入
    :param top_n_nodes: 返回前 N 个最相似的节点
    :param top_n_paragraphs: 在每个节点中返回前 N 个最相似段落
    :return: {节点: [(段落, 相似度), ...]}
    """
    # 计算目标嵌入与所有节点的相似度

    node_embeddings = np.array(graph.vs["combined_paragraph_embedding"])
    similarities = cosine_similarity_np(target_embedding, node_embeddings).flatten()

    # 按相似度降序排序
    similar_nodes = sorted(zip(graph.vs, similarities), key=lambda x: x[1], reverse=True)[:top_n_nodes]
    # 在每个节点中查找最相似段落
    results = {}
    for node, node_similarity in similar_nodes:
        similar_paragraphs = find_similar_paragraphs(target_embedding, node=node)[:top_n_paragraphs]
        results[node["name"]] = {
            "path": node["path"],
            # "child": get_children(graph, node),
            "node_similarity": float(node_similarity),
            "similar_paragraphs": similar_paragraphs
        }

    return results


def get_children(g, node):
    return [g.vs[neighbor]["name"] for neighbor in g.neighbors(node, mode="OUT")]


def get_parent(g, node):
    neighbors = g.neighbors(node, mode="IN")
    return g.vs[neighbors[0]]["name"] if neighbors else None


IdeaTech_Graph = None


async def ideatech_knowledge(query, rerank_model="BAAI/bge-reranker-v2-m3", file=None, version=0):
    global IdeaTech_Graph
    if not IdeaTech_Graph:
        file_path = f"{Config.DATA_FOLDER}/ideatech_pdf_graph.pkl"
        # with open(file_path, "rb") as f:
        #     IdeaTech_Graph = pickle.load(f)
        async with aiofiles.open(file_path, "rb") as f:
            IdeaTech_Graph = pickle.loads(await f.read())
        print(f"Graph loaded.{IdeaTech_Graph.vs.attributes()}")

    if file and version == 3:
        _, file_extension = os.path.splitext(file.filename.lower())
        if file_extension == '.pdf':
            file_path = Path(Config.DATA_FOLDER) / file.filename
            async with aiofiles.open(file_path, "wb+") as f:
                await f.write(await file.read())
            # 重新加载图
            IdeaTech_Graph = await graph_icc_edge(pdf_path=file_path)
            print(f"load graph {file_path},nodes:{len(IdeaTech_Graph.vs)}")

    query_embedding = np.array(await ai_embeddings(query, model_name='qwen', model_id=0))
    if version:  # nodes and paragraphs
        similar_content = query_similar_content(IdeaTech_Graph, query_embedding, top_n_nodes=3, top_n_paragraphs=3)
        print(similar_content)
        paragraphs = [j[0] for i in similar_content.values() for j in i.get("similar_paragraphs")]
    else:
        similar_content = find_similar_paragraphs(query_embedding, graph=IdeaTech_Graph)[:9]
        paragraphs = [i[0] for i in similar_content]

    if rerank_model and paragraphs:
        similar_content = await ai_reranker(query, documents=paragraphs, top_n=4, model_name=rerank_model)

    return similar_content
