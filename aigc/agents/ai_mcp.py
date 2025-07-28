# from mcp.server.fastmcp import FastMCP
# from mcp.server.sse import SseServerTransport
from fastmcp import FastMCP, Context, Client as MCPClient

# from neo4j import GraphDatabase
# driver = GraphDatabase.driver(uri="bolt://localhost:7687", auth=('neo4j', '77774ge7'))
# with driver.session() as session:
#     session.run("CREATE (n:Person {name: 'Alice'})")
from elasticsearch import Elasticsearch
es = Elasticsearch("http://localhost:9200")


mcp = FastMCP("MCP Assistant Server")
config = {
    "mcpServers": {
        "weather": {"url": "https://weather-api.example.com/mcp"},
        "assistant": {"command": "python", "args": ["./assistant_server.py"]}
    }
}
@mcp.tool
def hello(name: str) -> str:
    return f"Hello, {name}!"

@mcp.tool
def multiply(a: float, b: float) -> float:
    """Multiplies two numbers."""
    return a * b

@mcp.prompt
def summarize_request(text: str) -> str:
    """Generate a prompt asking for a summary."""
    return f"Please summarize the following text:\n\n{text}"




if __name__ == "__main__":
    import asyncio
    import nest_asyncio

    nest_asyncio.apply()
    doc = {
        "title": "Elasticsearch 是一款强大的搜索引擎",
        "content": "适用于全文搜索、结构化查询和分析"
    }

    # es.index(index="my_index", document=doc)


    # mcp.run(transport="streamable-http", host="127.0.0.1", port=8000, path="/mcp")#"sse
    # async def main():
    #
    #     # asyncio.run(tencent_translate('tencent translate is ok', 'en', 'cn'))
    #     # from lagent import list_tools, get_tool
    #     #
    #     # list_tools()
    #     try:
    #         # from mcp.server.fastmcp import FastMCP
    #         #
    #         # mcp = FastMCP("aigc")
    #         # mcp.run(transport='stdio')
    #         # mcp.mount()  # 将MCP服务器挂载到FastAPI应用中
    #
    #         # from fastmcp import FastMCP
    #         #
    #         # mcp_server = FastMCP(host="0.0.0.0", port=8001)
    #         # register_handlers(mcp_server)
    #         # await mcp_server.start()
    #         #
    #         # def register_handlers(mcp_server):
    #         #     """注册本地所有非私有方法为 handler"""
    #         #     for method_name in dir(local_service):
    #         #         if not method_name.startswith('_'):
    #         #             method = getattr(local_service, method_name)
    #         #             if callable(method):
    #         #                 mcp_server.add_handler(
    #         #                     action=method_name,
    #         #                     handler=partial(_call_local_method, method)
    #         #                 )
    #
    #
    #         # message = {"action": action, "data": data}#action: str, data: dict
    #         message = {"action": "greet", "data": "Hello from TokenFlux from FastAPI"}
    #
    #         async def call_mcp_server(message: dict, host: str = "modelshape.server.ip", port: int = 8001):
    #             """作为客户端连接另一个 MCP 服务"""
    #             client = FastMCPClient(host=host, port=port)
    #             try:
    #                 await client.connect()
    #                 response = await client.send_and_receive(message)
    #                 return response
    #             except Exception as e:
    #                 return {"error": str(e)}
    #             finally:
    #                 await client.disconnect()
    #
    #         # {"role": "tool", "content": json.dumps(mcp_response), "tool_call_id": tool_call.id}
    #         remote_response = await call_mcp_server(message, host="modelshape.server.ip", port=8001)
    #
    #         client = FastMCPClient(host="other.server.ip", port=8001)
    #         await client.send_message(message)
    #         response = await client.receive_message()
    #         print("Received from remote MCP:", remote_response)

        #
        # except  Exception as e:
        #     print('err', str(e))
        # # finally:
        # #     await mcp_server.stop()

    async def main():
        # Connect via in-memory transport
        # async with Client(mcp) as client:
        #     pass

        # async with Client("utils.py") as client:
        #     tools = await client.list_tools()
        #     print(f"Available tools: {tools}")
        #     result = await client.call_tool("add", {"a": 5, "b": 3})
        #     print(f"Result: {result.text}")

        async with MCPClient(config) as client:
            # Access tools and resources with server prefixes
            forecast = await client.call_tool("weather_get_forecast", {"city": "London"})
            answer = await client.call_tool("assistant_answer_question", {"query": "What is MCP?"})

        # Connect via in-memory transport
        async with MCPClient(mcp) as client:
            pass
        # Connect via SSE
        async with MCPClient("http://localhost:8000/sse") as client:
            # ... use the client
            pass


    asyncio.run(main())


    # stdio：进程间通信，适用于命令行或脚本工具执行
    # SSE：基于http服务器实现事件推送，适合web应用、实时推送等场景
    # 使用标准输入输出传输来运行
    # if transport == "stdio":
    #     anyio.run(self.run_stdio_async)
    # else:  # 作为独立服务器传输事件
    #     anyio.run(self.run_sse_async)