import igraph as ig
import json, time, random
from enum import Enum as PyEnum
from collections import defaultdict
import asyncio
import queue
from redis.asyncio import Redis
from typing import Dict, List
from threading import Thread
import uuid
import zmq, zmq.asyncio

REDIS_HOST = "localhost"
REDIS_PORT = 6379
QUEUE_NAME = "message_queue"

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


class WebSearchGraph:
    def __init__(self):

        # 初始化节点内容字典
        self.nodes: Dict[str, Dict[str, str]] = {}
        # 初始化邻接表
        self.adjacency_list: Dict[str, List[dict]] = defaultdict(list)
        self.task_queue = queue.Queue()
        self.n_active_tasks = 0

    async def add_root_node(self, node_content: str, node_name: str = 'root'):
        # 添加根节点
        self.nodes[node_name] = dict(content=node_content, type="root")
        # 在图中添加节点

        self.adjacency_list[node_name] = []
        return node_name

    async def add_node(self, node_name: str, node_content: str):
        # 添加子问题节点
        self.nodes[node_name] = dict(content=node_content, type="search")

        self.adjacency_list[node_name] = []

        # 处理父节点，查找相关的历史上下文
        parent_response = []
        for start_node, adj in self.adjacency_list.items():
            for neighbor in adj:
                if (node_name == neighbor["name"]  # 判断是否有连接,是否是当前节点的父节点，并且该父节点包含 response 信息
                        and start_node in self.nodes and "response" in self.nodes[start_node]):
                    parent_response.append(
                        dict(question=self.nodes[start_node]["content"], answer=self.nodes[start_node]["response"]))

        await self._async_node_stream(node_name, node_content)

        self.n_active_tasks += 1  # f"{node_name}-{node_content}"
        return self.n_active_tasks

    async def _async_node_stream(self, node_name: str, node_content: str, parent_response: List[dict]):
        """执行异步搜索"""
        cfg = {"search_config": "value"}  # 配置搜索
        session_id = random.randint(0, 999999)  # 会话ID
        agent = None

        try:
            # 模拟搜索过程
            searcher_message = "mock_search_message"  # 假设的搜索消息
            self.nodes[node_name]["response"] = searcher_message  # 更新节点响应
            self.nodes[node_name]["session_id"] = session_id
            self.task_queue.put((node_name, self.nodes[node_name]))  # 将结果放入队列
        except Exception as exc:
            self.task_queue.put((exc, None))

    async def add_response_node(self, node_name: str = 'response'):
        # 添加回复节点
        self.nodes[node_name] = dict(content='Search completed, thought response added.', type="response")
        # self.adjacency_list[node_name] = []
        self.task_queue.put((node_name, self.nodes[node_name], []))

    async def add_edge(self, start_node: str, end_node: str):
        self.adjacency_list[start_node].append(dict(id=str(uuid.uuid4()), name=end_node, state=2))

        self.task_queue.put((start_node, self.nodes[start_node], self.adjacency_list[start_node]))

    async def reset(self):
        # 重置图和节点
        self.nodes.clear()
        self.adjacency_list.clear()

    def node(self, node_name: str):
        # 获取节点信息
        if node_name in self.nodes:
            return self.nodes[node_name].copy()

        return None

    def graph(self):
        """根据节点信息和邻接表生成图
        nodes = {
            "root": {"content": "What is AI?", "type": "root"},
            "node1": {"content": "What is machine learning?", "type": "search"},
            "node2": {"content": "What is deep learning?", "type": "search"}
        }

        adjacency_list = {
            "root": [{"name": "node1"}, {"name": "node2"}],
            "node1": [{"name": "node2"}],
            "node2": []
        }

        Returns:
            ig.Graph: 返回生成的图对象
        """
        # 初始化图对象
        graph = ig.Graph(directed=True)  # 创建有向图

        # 添加节点到图中
        node_names = list(self.nodes.keys())  # 获取节点名称列表
        graph.add_vertices(node_names)  # 添加图的所有节点

        # 为每个节点添加属性
        for node_name, node_data in self.nodes.items():
            graph.vs[node_name]["content"] = node_data["content"]
            graph.vs[node_name]["type"] = node_data.get("type", "search")

        # 添加边到图中
        for start_node, neighbors in self.adjacency_list.items():
            for neighbor in neighbors:
                # 根据邻接表建立连接，添加边
                graph.add_edge(start_node, neighbor["name"])

        return graph


# message_queue = asyncio.Queue()
# 异步生产者
async def producer(messages):
    redis = Redis(host=REDIS_HOST, port=REDIS_PORT)
    try:
        for message in messages:
            await redis.rpush(QUEUE_NAME, message)  # 异步放入队列
            await redis.hset("task_status", message, "pending")
            # await message_queue.put(message)
            print(f"Produced: {message}")
            await asyncio.sleep(1)
    finally:
        await redis.close()


# 异步消费者
async def consumer():
    redis = Redis(host=REDIS_HOST, port=REDIS_PORT)
    while True:
        message = await redis.blpop(QUEUE_NAME, timeout=5)  # 异步阻塞消费
        if message:
            task = message[1].decode()
            print(f"Consumed: {task}")
            # message = await message_queue.get()
            # print(f"Consumed: {message}")
            # message_queue.task_done()  # 标记任务完成
            yield task

            await redis.hset("task_status", task, "completed")
        else:
            break
    await redis.close()


# async for task in consumer():
#     await asyncio.sleep(1)


class MessageZeroMQ:

    def __init__(self, pull_port="7556", push_port="7557", req_port="7555", process_callback=None):
        self.context = zmq.asyncio.Context(io_threads=2)  # zmq.Context()

        # 设置接收消息的 socket
        self.pull_socket = self.context.socket(zmq.PULL)
        self.pull_socket.bind(f"tcp://*:{pull_port}")  # 绑定接收端口

        # 设置发送消息的 socket
        self.push_socket = self.context.socket(zmq.PUSH)
        self.push_socket.connect(f"tcp://localhost:{push_port}")  # 连接到 Java 的接收端口

        # 设置 REQ socket 用于请求-响应模式
        self.req_socket = self.context.socket(zmq.REQ)  # zmq.DEALER
        if req_port:
            self.req_socket.connect(f"tcp://localhost:{req_port}")  # 连接到服务端

        self.process_callback = process_callback or self.default_process_message
        # self.push_socket.send_string('topic1 Hello, world!')

    def __del__(self):
        self.context.destroy(linger=0)

    @staticmethod
    def default_process_message(message):
        # 处理逻辑
        print(f"Processing message: {message}")
        # 对消息进行某些处理后需要将其转发回 Java
        return f"Processed: {message}"

    async def send_request(self, message: str = "Hello, server!", topic: str = 'Request'):
        """
        使用 REQ socket 主动发送消息并接收响应,主动请求-响应
        """
        await self.req_socket.send_string(f'{topic} {message}')
        print(f"Sent request: {message} under topic: {topic}")
        response = await self.req_socket.recv_string()
        print(f"Received response: {response}")
        return response

    async def call_service(self, data):
        """
        使用 REQ socket 发送 JSON 数据并接收 JSON 响应 zmq.Context()
        """
        self.req_socket.send_json(data)
        response  = self.req_socket.recv_json()
        # message = await self.req_socket.recv()
        # response = json.loads(message.decode())
        print(f"Received response: {response}")
        return response

    # 主动发送消息到 ZeroMQ
    async def send_message(self, message: str, topic: str = "Default"):
        await self.push_socket.send_string(f'{topic} {message}', flags=zmq.DONTWAIT, encoding='utf-8')
        print(f"Sent message: {message}")

    async def send_data(self, data=b'Hello in binary'):
        await self.push_socket.send(data, flags=zmq.DONTWAIT, copy=True, track=False)

    async def recv_messages(self):
        while True:
            # 将接收到的消息 yield 出去
            message = await self.pull_socket.recv_string()
            yield message

    async def stream_start(self):
        async for message in self.recv_messages():
            print(f"Received from ZeroMQ: {message}")
            processed_msg = self.process_callback(message)
            await self.send_message(processed_msg, topic="Processed")

    async def start(self):
        # 使用 asyncio 和 run_in_executor 进行阻塞操作
        # loop = asyncio.get_event_loop()

        while True:
            # 接收消息
            message = await self.pull_socket.recv_string()
            print(f"Received from ZeroMQ: {message}")

            # 处理消息
            processed_msg = self.process_callback(message)

            # 将处理后的消息发送
            await self.push_socket.send_string(processed_msg)
            print(f"Sent processed message back.")


import numpy as np


class ABO:
    def __init__(self):
        self._A = None
        self._B = None
        self._O = None

    def AO(self):
        pass

    def BO(self):
        pass

    def _AB(self):
        return self._A + self._B


class ABONode(ABO):

    def __init__(self, node_id: str, neighbors={}, d_v: int = 1024):
        super().__init__()
        """
        初始化节点对象
        :param node_id: 节点唯一标识
        :param d_v: 观点向量维度
        """
        self.node_id = node_id
        self._A = {}  # fact_history{timestamp: fact_vector}
        self._O = None
        self.neighbors = neighbors  # 记录与其他节点的关系
        self._B = np.zeros((len(neighbors), d_v))  # 节点对全局的观点矩阵,view_matrix

    def update_fact(self, timestamp: int, fact_vector: np.ndarray):
        """
        更新节点的事实向量
        :param timestamp: 时间戳
        :param fact_vector: 新的事实向量
        """
        self._A[timestamp] = fact_vector

    def update_neighbors(self, node_id: str, relation_data: np.ndarray):
        """
        更新节点与其他节点的关系数据
        :param node_id: 关系节点的 ID
        :param relation_data: 节点之间的关系数据（例如：观点向量、影响力等）
        """
        self.neighbors[node_id] = relation_data

    def update_view_matrix(self, alpha: float = 0.7):
        """
        根据邻居关系和节点的事实数据生成观点矩阵
        :param alpha: 权重因子，用于控制事实和关系的影响
        """
        for i, (neighbor_id, relation_data) in enumerate(self.neighbors.items()):
            fact_vector = self._A.get(1, None)  # 使用某个时间戳的事实向量
            self._B[i] = alpha * fact_vector + (1 - alpha) * relation_data


if __name__ == "__main__":
    import sys

    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    message_zero_mq = MessageZeroMQ()  # 创建消息处理器实例
    asyncio.run(message_zero_mq.stream_start())
    # asyncio.run(message_zero_mq.start())

    # 运行异步任务
    # async def main():
    #     await asyncio.gather(producer([f"Message {i}" for i in range(5)]), consumer())
    #     # redis = Redis(host="localhost", port=6379)
    #     # await redis.set("key", "value")
    #     # value = await redis.get("key")
    #     # print(value.decode())  # 输出: value
    #     # await redis.close()
    #
    #
    # asyncio.run(main())
