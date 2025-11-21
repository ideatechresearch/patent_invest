import numpy as np
import struct, re


def none_count(my_data: list):  # err：np.nansum([1,2,None])
    try:  # (data.count(np.nan))
        # filter(function, iterable)函数对满足条件的属性或值进行过滤，返回 True 的元素放到新列表中
        return [i is None for i in my_data].count(True)  # (np.sum([i is None for i in my_data]))
    except:
        # 它接收一个函数 f 和一个 list，并通过把函数 f 依次作用在 list 的每个元素上，得到一个新的 list 并返回
        return np.sum(list(map(lambda x: x is None, my_data)))  # list(map)被显示转化为列表


def value_marked(condition, values):
    """
    返回第一个满足条件的值，如果没有匹配返回 default
    condition: 可迭代的布尔序列
    values: 与 condition 对应的数据序列
    zip([iterable,arr])如果各个迭代器的元素个数不一致，则返回列表长度与最短的对象相同
    """
    if not condition.any():
        return 0
    for cond, val in zip(condition, values):
        if cond:
            return val  # np.take(close, np.where(dates == i))
    return 0


def value_when(condition, values, occurrence=0, default=np.nan):
    """
    VALUEWHEN(COND, DATA, occurrence)
    模拟 TA-Lib：每个位置返回最近 occurrence 次满足条件的值
    condition: 布尔数组
    values: 数据数组
    occurrence: 0 表示最近一次, 1 表示倒数第二次, ...
    """
    result = np.full_like(values, default, dtype=float)
    idx = np.where(condition)[0]

    for i in range(len(values)):
        matches = idx[idx <= i]
        if len(matches) > occurrence:
            result[i] = values[matches[-(occurrence + 1)]]
    return result


def is_upeak(data, index: int):
    vi = data[index]
    length = len(data)
    if length == 1:
        return True
    lmax = max(data[:index]) if index > 0 else float('-inf')
    rmax = max(data[index + 1:]) if index < length - 1 else float('-inf')
    return vi > lmax and vi > rmax


def is_finite(value) -> bool:
    """判断是否为有限数字"""
    try:
        float_val = float(value)
        return not (float_val == float('inf') or float_val == float('-inf') or float_val != float_val)
    except (TypeError, ValueError):
        return False


def float16_to_bin(num):
    # 将float16数打包为2字节16位，使用struct.pack 处理二进制数据的模块
    packed_num = struct.pack('e', num)  # e 半精度浮点数（float16,16-bit) b'\x00<'
    # 解包打包后的字节以获取整数表示
    int_value = struct.unpack('H', packed_num)[0]
    # 将整数表示转换为二进制
    binary_representation = bin(int_value)[2:].zfill(16)
    return binary_representation


def levenshtein_distance_np(s: str, t: str) -> int:
    """
    基于 NumPy 的 Levenshtein 编辑距离计算
    """
    if len(s) == 0:
        return len(t)
    if len(t) == 0:
        return len(s)

    rows: int = len(s) + 1
    cols: int = len(t) + 1

    # 初始化矩阵
    dist_matrix = np.zeros((rows, cols), dtype=int)
    dist_matrix[:, 0] = np.arange(rows)
    dist_matrix[0, :] = np.arange(cols)

    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            dist_matrix[i, j] = min(
                dist_matrix[i - 1, j] + 1,  # 删除
                dist_matrix[i, j - 1] + 1,  # 插入
                dist_matrix[i - 1, j - 1] + cost  # 替换
            )

    return dist_matrix[-1, -1]


def cosine_sim(vecs1, vecs2):
    # 两个 单个向量（1D 数组）之间的余弦相似度
    dot_product = np.dot(vecs1, vecs2)
    similarity = dot_product / (np.linalg.norm(vecs1) * np.linalg.norm(vecs2))
    return similarity


def cluster_similarity_mean(df, text_embeddings):
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_means = np.full(len(df), np.nan)  # [np.nan] * len(df)

    for cluster_id, group in df.groupby('cluster'):
        if cluster_id == -1:
            # 忽略 HDBSCAN 中的 noise 点
            continue
        indices = group.index.tolist()
        if len(indices) <= 1:
            # similarity_means[indices[0]] = np.nan
            continue

        embs = text_embeddings[indices]
        sim_matrix = cosine_similarity(embs)

        for i, idx in enumerate(indices):
            sim_row = np.delete(sim_matrix[i], i)  # 删除对角线
            similarity_means[idx] = sim_row.mean()

    return similarity_means


def pairwise_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray):
    """
    计算 emb1[i] 和 emb2[i] 的余弦相似度[cosine_sim(a, b) for a, b in zip(emb1, emb2)]
    :param emb1: shape = (N, D)
    :param emb2: shape = (N, D)
    :return: shape = (N,) 的相似度数组
    """
    emb1 = np.asarray(emb1)
    emb2 = np.asarray(emb2)
    dot_product = np.sum(emb1 * emb2, axis=1)
    norm1 = np.linalg.norm(emb1, axis=1)
    norm2 = np.linalg.norm(emb2, axis=1)
    similarity = dot_product / (norm1 * norm2 + 1e-8)
    return similarity


def fast_dot_np(vecs1, vecs2):
    # 用 NumPy 批量计算点积,形状相同的 2D 数组逐行点积,矩阵逐元素相乘后按行求和
    return np.einsum('ij,ij->i', vecs1, vecs2)  # np.sum(A * B, axis=1)


# from sklearn.preprocessing import normalize
def normalize_np(vecs) -> list[float]:
    # 手动归一化
    # norms = np.sqrt(np.einsum('ij,ij->i', vecs, vecs)) #模长,L2 范数 ||ndarr1|| for each row
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def normalize_embeddings(vectors: list[list[float]], to_list=False):
    normalized = [vec / np.linalg.norm(vec) if np.linalg.norm(vec) != 0 else vec for vec in np.array(vectors)]
    return [vec.tolist() if isinstance(vec, np.ndarray) else vec for vec in normalized] if to_list else normalized


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


# from sklearn.metrics.pairwise import cosine_similarity
def cosine_similarity_np(ndarr1, ndarr2):
    denominator = np.outer(np.linalg.norm(ndarr1, axis=1), np.linalg.norm(ndarr2, axis=1))
    dot_product = np.dot(ndarr1, ndarr2.T)  # np.einsum('ik,jk->ij', ndarr1, ndarr2)
    with np.errstate(divide='ignore', invalid='ignore'):
        similarity = np.where(denominator != 0, dot_product / denominator, 0)
    return similarity


# from scipy.special import softmax
def softmax_np(x):
    if x.ndim == 1:
        e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return e_x / e_x.sum()
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # 对每行减去最大值
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def create_shared_array(shape, dtype='int32'):
    """创建跨进程 Worker 共享的数组"""
    from multiprocessing.shared_memory import SharedMemory
    shm = SharedMemory(create=True, size=np.prod(shape) * np.dtype(dtype).itemsize)
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    # mmap_array = np.memmap(file_path='shared_array.dat', dtype=np.float32, mode='w+', shape=shape) create
    return arr, shm.name


def attach_shared_array(name, shape, dtype='int32'):
    """附加到现有共享数组,子进程读取"""
    from multiprocessing.shared_memory import SharedMemory
    shm = SharedMemory(name=name)
    # np.memmap(file_path='shared_array.dat', dtype=np.float32, mode='r+', shape=shape) 'readwrite' access
    return np.ndarray(shape, dtype=dtype, buffer=shm.buf)


def generate_loss_mask(input_ids, bos_id, eos_id, max_length):
    loss_mask = [0] * len(input_ids)
    i = 0
    while i < len(input_ids):
        if input_ids[i:i + len(bos_id)] == bos_id:
            start = i + len(bos_id)
            end = start
            while end < len(input_ids):
                if input_ids[end:end + len(eos_id)] == eos_id:
                    break
                end += 1
            for j in range(start + 1, min(end + len(eos_id) + 1, max_length)):
                loss_mask[j] = 1
            i = end + len(eos_id) if end < len(input_ids) else len(input_ids)
        else:
            i += 1
    return loss_mask


def get_similar_nodes(embeddings, base_nodes, top_k=3):
    """
    计算 base_nodes（初步召回的记录）与所有记录之间的余弦相似度，找到最相似的 top_k 记录
    :param embeddings: 所有节点的嵌入矩阵 (Tensor)
    :param base_nodes: 需要查询相似记录的节点索引列表
    :param top_k: 每个节点要找的相似记录数
    :return: 召回的相似记录索引列表
    """
    # 提取 base_nodes 对应的向量
    base_embeddings = embeddings[base_nodes]
    # all_embeddings = embeddings.cpu().detach().numpy()

    # 计算余弦相似度 (sklearn)
    similarity_matrix = cosine_similarity_np(base_embeddings, embeddings)
    # 对每个 base_node 取最相似的 top_k 记录（排除自身）
    similar_nodes = set()
    for i, node in enumerate(base_nodes):
        sorted_indices = np.argsort(-similarity_matrix[i])  # 获取该记录的相似度排序,降序排序
        for idx in sorted_indices:
            if idx != node:  # 排除自身
                similar_nodes.add(idx)
            if len(similar_nodes) >= top_k:
                break

    return list(similar_nodes)


def math_solver(input_str=None, math_expr=None, operation="evaluate", values=None, symbol="x", limits=None):
    """
    自动数学求解 + 数值代入计算

    Params:
    - input_str: 原始自然语言描述，尝试从自然语言中粗略提取表达式，可以不填
    - math_expr: 数学表达式（如 "x^2 + 3x + 1"）
    - operation: 操作类型（"evaluate" 为数值计算，还有 diff，integrate，factorial，sum..）
    - values: dict，变量数值，如 {"x": 2}
    - symbol: 默认变量符号 "x"
    - limits: 对于求和，可传如 ("i", 1, 10),其他情况可以不填

    Returns:
    - str：结果数值
    """
    expr_text = math_expr or input_str
    if math_expr is None and input_str:
        # 尝试从自然语言中粗略提取表达式
        patterns = {
            r'加|plus|add|sum': '+',
            r'减|minus': '-',
            r'乘|times|multiplied by|dot': '*',
            r'除以|除|divided by|divide': '/',
            r'平方|square': '**2',
            r'\^|次方|power': '**',
        }
        for pattern, repl in patterns.items():
            expr_text = re.sub(pattern, repl, expr_text)
    if not expr_text:
        return "请提供数学表达式或输入描述"

    try:
        import sympy
        expr = sympy.sympify(expr_text.replace("^", "**"))
        sym = sympy.symbols(symbol)
        if operation == "evaluate":
            if values:
                subs_dict = {sympy.symbols(k): v for k, v in values.items()}
                return float(expr.evalf(subs=subs_dict))
            else:
                return float(expr.evalf())
        elif operation == "diff":  # 对表达式关于某个变量求导
            return str(sympy.diff(expr, sym))
        elif operation == "integrate":  # 求不定积分
            return str(sympy.integrate(expr, sym))
        elif operation == "factorial":  # 整数的阶乘
            return sympy.factorial(int(expr))
        elif operation == "sum":
            if limits and len(limits) == 3:
                var = sympy.symbols(limits[0])
                return sympy.summation(expr, (var, limits[1], limits[2]))
            else:
                return "请提供 limits 参数，如 ('i', 1, 10)"
        else:
            return f"未知操作类型：{operation}"

    except Exception as e:
        return f"解析失败：{e}"

def fibonacci_generator(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b