import igraph as ig
# import queue
import json, time
from enum import Enum as PyEnum

Task_queue = {}  # queue.Queue(maxsize=Config.MAX_TASKS)
Task_graph = ig.Graph(directed=True)  # 创建有向图


class TaskStatus(PyEnum):
    PENDING = "pending"  # 等待条件满足
    READY = "ready"  # 条件满足，可以执行
    IN_PROGRESS = "running"  # processing

    COMPLETED = "done"
    FAILED = "failed"
    RECEIVED = "received"


class TaskNode:
    name: str  # task_id
    description: str
    status: str = TaskStatus.PENDING  # 默认状态
    action = None  # 任务的执行逻辑（可调用对象函数、脚本或引用的操作类型)
    event = None  # 事件是标识符，用于任务之间的触发,指示触发的事件类型和附加数据
    priority: int = 10
    start_time: float = 0

    def __init__(self, name, description, action=None, event=None, status="PENDING", start_time=0, priority=10):
        self.name = name
        self.description = description
        self.action = action
        self.event = event
        self.priority = priority
        self.status = status
        self.start_time = start_time


class TaskEdge:
    source: str  # 依赖的起始任务
    target: str  # 被触发的任务
    condition: str  # 边上的触发条件（与源任务的状态相关),["done",{"deadline": time.time() + 60}]
    # type: str  # status_condition基于任务的状态触发,event_trigger,基于某个任务触发的事件来激活任务
    trigger_time: dict = None  # absolute,relative,[None, {"relative": 5}]
    trigger_event = None  # 任务触发事件,None无依赖
    rule = None  # 复杂条件,函数或复杂的逻辑判断

    def __init__(self, source, target, condition, trigger_time=None, trigger_event=None, rule=None):
        self.source = source
        self.target = target
        self.condition = condition
        self.trigger_time = trigger_time
        self.trigger_event = trigger_event
        self.rule = rule


# for task_id, attributes in Task_queue.items():
# 添加节点到图中
def set_task_node(task_id, attributes):
    nodes = Task_graph.vs["name"] if "name" in Task_graph.vs.attributes() else []
    if task_id in nodes:  # Task_graph
        # 更新已有节点属性
        # attributes= {**Task_graph.nodes[task_id], **Task_queue[task_id]}
        node_idx = Task_graph.vs.find(name=task_id).index
        Task_graph.vs[node_idx].update_attributes(attributes)
        # Task_graph.nodes[task_id].update(attributes)
    else:
        Task_graph.add_vertex(name=task_id, **attributes)
        # Task_graph.add_node(task_id, **Task_queue[task_id])自动处理不存在的节点


def update_task_status(task_id, new_status):
    task_index = Task_graph.vs.find(name=task_id).index
    Task_graph.vs[task_index]["status"] = new_status


def set_task_condition(edges, conditions=["done"]):
    for (source, target), condition in zip(edges, conditions):
        if source in Task_graph.vs["name"] and target in Task_graph.vs["name"]:
            Task_graph.add_edge(source, target, condition=condition)


# 检查依赖并触发任务
def check_and_trigger_tasks(graph):
    for edge in graph.es:
        source_task = graph.vs.find(name=edge.source)
        target_task = graph.vs.find(name=edge.target)

        if target_task["status"] == "pending":
            condition = edge["condition"]
            event_ready = 0
            # 状态条件
            if isinstance(condition, str) and source_task["status"] == condition:
                trigger_event = edge.get("trigger_event", None)
                trigger_time = edge.get("trigger_time", None)
                # 任务状态变化后触发事件驱动的任务边或者任务A完成时触发多个事件
                event_ready = 1 if not trigger_event else source_task.get("event") == trigger_event
                # 检查时间条件>
                if trigger_time:
                    if "absolute" in trigger_time:
                        event_ready = time.time() >= trigger_time["absolute"]
                    elif "relative" in trigger_time:
                        event_ready = time.time() >= source_task.get("start_time", 0) + trigger_time["relative"]
            # 时间条件<
            elif isinstance(condition, dict) and "deadline" in condition:
                if time.time() <= condition["deadline"]:
                    event_ready = 1 << 1
            # 自定义条件
            elif callable(condition) and condition():
                event_ready = 1 << 2

            if event_ready:
                target_task["status"] = "ready"
                print(f"Task {target_task['name']} is now ready.")


# 任务执行主循环
def task_execution_loop(graph):
    while True:
        # 查找状态为 ready 的任务并执行
        ready_tasks = [v for v in graph.vs if v["status"] == "ready"]
        if not ready_tasks:
            break  # 无可执行任务时退出
        for task in ready_tasks:
            print(f"Executing task {task['name']}...")
            time.sleep(1)  # 模拟执行时间
            task["status"] = "done"  # 'completed'
            print(f"Task {task['name']} is now done.")

        # 检查并触发新的任务
        check_and_trigger_tasks(graph)

        time.sleep(1)


def cleanup_tasks(timeout_received=600, timeout=86400):
    current_time = time.time()
    task_ids_to_delete = []
    for task_id, task in Task_queue.items():
        t_sec = current_time - task['timestamp']
        if t_sec > timeout_received:
            if task['status'] == TaskStatus.RECEIVED:
                task_ids_to_delete.append(task_id)
                print(f"Task {task_id} has been marked for cleanup. Status: RECEIVED")
        elif t_sec > timeout:
            task_ids_to_delete.append(task_id)
            print(f"Task {task_id} has been marked for cleanup. Timeout exceeded")

    for task_id in task_ids_to_delete:
        del Task_queue[task_id]


def export_to_json_from_graph(graph, filename):
    data = {"nodes": [{"id": v.index, **v.attributes()} for v in graph.vs],
            "edges": [{"source": e.source, "target": e.target, **e.attributes()} for e in graph.es], }
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

    # graph.write_pickle("graph.pkl")


def import_to_graph_from_json(filename):
    with open(filename, "r") as f:
        data = json.load(f)

    g = ig.Graph(directed=True)  # ig.Graph.TupleList(
    g.add_vertices(len(data["nodes"]))
    # g.vs["name"] = [node["id"] for node in data["nodes"]]
    # 为节点设置属性
    for idx, node in enumerate(data["nodes"]):
        g.vs[idx].update_attributes(node)  # node.items()
    g.add_edges([(edge["source"], edge["target"]) for edge in data["edges"]])  # [("task1", "task2")]

    # 为边设置属性
    for idx, edge in enumerate(data["edges"]):
        # 跳过 source 和 target 属性
        g.es[idx].update_attributes({k: v for k, v in edge.items() if k not in ["source", "target"]})
        # g.es[idx][key] = value

    # g.es["condition"] = ["done"]
    return g
