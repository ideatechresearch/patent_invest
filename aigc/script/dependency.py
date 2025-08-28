import os
import ast
import networkx as nx


def build_module_dependency_graph(base_path):
    """
    生成Python文件依赖关系图
    :param base_path: 要分析的文件夹路径（或单个py文件）
    :return: networkx.DiGraph 对象
    """
    graph = nx.DiGraph()
    file_map = {}  # 模块名 -> 文件路径

    # 递归扫描所有 Python 文件
    py_files = []
    if os.path.isfile(base_path) and base_path.endswith(".py"):
        py_files = [os.path.abspath(base_path)]
    else:
        for root, _, files in os.walk(base_path):
            for f in files:
                if f.endswith(".py"):
                    py_files.append(os.path.abspath(os.path.join(root, f)))

    # 建立模块名映射（utils/file_ops.py -> utils.file_ops）
    for f in py_files:
        rel_path = os.path.relpath(f, base_path)
        module_name = rel_path[:-3].replace(os.sep, ".")
        file_map[module_name] = f

    # 解析每个文件的 import 依赖
    for module_name, file_path in file_map.items():
        with open(file_path, "r", encoding="utf-8") as fp:
            try:
                tree = ast.parse(fp.read(), filename=file_path)
            except SyntaxError as e:
                print(f"⚠️ 语法错误: {file_path} - {e}")
                continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    dep = alias.name
                    if dep in file_map:  # 只记录项目内依赖
                        graph.add_edge(module_name, dep)

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    dep = node.module
                    if dep in file_map:
                        graph.add_edge(module_name, dep)

    return graph


def build_internal_dependency_graph(py_file):
    """
    构建单个 Python 文件内函数之间的依赖图
    :param py_file: Python 文件路径
    :return: networkx.DiGraph
    """
    with open(py_file, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source)
    graph = nx.DiGraph()

    # 收集所有函数定义
    func_defs = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_defs[node.name] = node

    # 分析每个函数内部调用了哪些其他函数
    for func_name, func_node in func_defs.items():
        for child in ast.walk(func_node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    called_func = child.func.id
                    if called_func in func_defs:  # 只记录文件内的调用
                        graph.add_edge(func_name, called_func)

                elif isinstance(child.func, ast.Attribute):
                    # 处理 self.xxx() 这种方法调用
                    if isinstance(child.func.value, ast.Name) and child.func.value.id == "self":
                        called_func = child.func.attr
                        if called_func in func_defs:
                            graph.add_edge(func_name, called_func)

    return graph


def plot_dependency_graph(graph, title="Dependency Graph"):
    """
    绘制依赖关系图
    """
    import matplotlib
    matplotlib.use("Agg")  # 强制使用无GUI后端
    import matplotlib.pyplot as plt
    # 找出所有循环依赖的节点
    cycles = list(nx.simple_cycles(graph))
    cycle_nodes = set()
    for cycle in cycles:
        cycle_nodes.update(cycle)

    node_colors = ["lightcoral" if node in cycle_nodes else "lightblue"
                   for node in graph.nodes()]

    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(graph, k=1.0, iterations=100)
    nx.draw_networkx_nodes(graph, pos, node_size=1500, node_color=node_colors)
    nx.draw_networkx_edges(graph, pos, arrowstyle="->", arrowsize=15)
    nx.draw_networkx_labels(graph, pos, font_size=10, font_weight="bold")

    plt.title(title, fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("data/dependency_graph.png", dpi=300)
    # plt.show()


if __name__ == "__main__":
    # base_path = "../utils/utils.py"
    # g = build_internal_dependency_graph(base_path)
    base_path = "../utils"
    g = build_module_dependency_graph(base_path)

    print("依赖关系边数:", g.number_of_edges())

    # 检查循环依赖
    cycles = list(nx.simple_cycles(g))
    if cycles:
        print("🔄 循环依赖检测到:")
        for c in cycles:
            print(" -> ".join(c))

    plot_dependency_graph(g)
