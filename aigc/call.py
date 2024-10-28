import httpx
import asyncio
import requests
import websockets
import json


async def call_generate_stream_response(stream: bool = False):
    url = "http://localhost:7000/message/"
    request_data = {
        "uuid": "",
        "username": "",
        "agent": "1",
        "model_name": "moonshot",
        'user_id': 'test',
        "prompt": '',
        "question": "什么是区块链金融?",
        'stream': int(stream),
        "filter_time": 0,
        "temperature": 0.4,
        "topn": 10,
        "score_threshold": 0
    }

    try:
        async with httpx.AsyncClient(timeout=100.0) as client:
            if stream:
                async with client.stream("POST", url, json=request_data) as response:
                    print(f"Response status: {response.status_code}")
                    async for line in response.aiter_lines():
                        line = line.strip()
                        if line:
                            print(line)
                            if line == "data: [DONE]":
                                print("Stream finished.")
                                break
            else:
                response = await client.post(url, json=request_data)
                print(f"Response status: {response.status_code}")
                print(response.json())
    except httpx.RequestError as exc:
        print(f"An error occurred while requesting {exc.request.url!r}. Error details: {exc}")
    except httpx.HTTPStatusError as exc:
        print(f"Error response {exc.response.status_code} while requesting {exc.request.url!r}.")
    except Exception as exc:
        print(f"An unexpected error occurred: {exc}")



async def connect_chat():
    uri = "ws://127.0.0.1:7000/ws/chat"
    async with websockets.connect(uri) as websocket:
        while True:
            question = input("You: ")
            if question.lower() == 'exit':
                print("Exiting the chat.")
                break
            try:
                await websocket.send(json.dumps({"question": question, "username": "test"}))
                print("AI: ", end="")

                done = False
                while not done:
                    response = await websocket.recv()
                    for line in response.splitlines():  # 逐行读取流式响应
                        if not line:
                            continue
                        if line.startswith("data: "):
                            line_data = line.lstrip("data: ")

                            # 检查是否为结束标识 [DONE]
                            if line_data == "[DONE]":
                                print("\nEnd.")
                                done = True
                                break  # 收到结束标识后，终止解析
                            try:
                                # 尝试将数据解析为 JSON
                                parsed_content = json.loads(line_data)
                                if isinstance(parsed_content, dict):
                                    print(parsed_content.get("content", ""))
                                    # 假设服务器发送了结束标识，跳出循环继续下一轮对话
                                    # if parsed_content.get('role') == 'assistant':
                                    #     print("\nEnd of response.")
                                    #     break
                                else:
                                    print(f"Received non-dict content: {parsed_content}")

                            except json.JSONDecodeError:
                                # 如果 JSON 解析失败，返回原始文本数据
                                print(line_data, end="", flush=True)

            except websockets.ConnectionClosedOK:
                print("Connection closed normally.")
                return
            except websockets.ConnectionClosedError as e:
                print(f"Connection closed with error: {e}")
                return


def send_request(endpoint: str, data: dict):
    url = f"http://localhost:7000/{endpoint}"
    response = requests.post(url, json=data)
    return response.json()


def main():
    import streamlit as st
    st.title("Data Interaction with FastAPI")
    URL = "http://127.0.0.1:7000"

    text_input = st.text_input("Enter some text:")
    if st.button("Send Text"):
        data = {"text": text_input}
        result = send_request("text", data)
        st.write(result)

    image_url = st.text_input("Enter image URL:")
    if st.button("Send Image"):
        data = {"image_url": image_url}
        result = send_request("image", data)
        st.write(result)

    uploaded_file = st.file_uploader("Choose a file")
    if st.button("Send File"):
        if uploaded_file is not None:
            files = {"file": uploaded_file.getvalue()}
            response = requests.post(f"{URL}/upload", files=files)
            st.write(response.json())

    user_input = st.text_input("You:", key="user_input")
    if st.button("Send Message"):  # if prompt := st.chat_input("Shift + Enter 换行，Enter 发送"):
        if user_input:
            response = requests.post(f"{URL}/send_message", json={"question": user_input})
            st.session_state.messages = response.json()
            if 'rerun' not in st.session_state:
                st.session_state.rerun = True
            else:
                st.experimental_rerun()


# 如果你在异步环境中，例如 FastAPI 应用程序中
# 否则你可以直接运行
if __name__ == "__main__":
    # main()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(connect_chat())

    # asyncio.run(call_generate_stream_response(False))

    # url = "http://localhost:7000/message/"
    # request_data = {
    #     "uuid": "",
    #     "username": "test",
    #     "agent": "1",
    #     "model_name": "moonshot",
    #     "prompt": '',
    #     "question": "什么是区块链金融?",
    #     'stream': 0,
    #     "filter_time": 0,
    #     "temperature": 0.4,
    #     "topn": 10,
    #     "score_threshold": 0
    # }
    # response = requests.post(url, json=request_data)
    # print(response.text)
    # response_data = response.json()
    # print(response_data['answer'])

    # url = "http://localhost:7000/get_messages/"
    # request_data = {
    #     "uuid": "",
    #     "username": "test",
    #     "agent": "1",
    #     "filter_time": 0,
    # }
    # response = requests.get(url, json=request_data)
    # print(response.text)
    # response_data = response.json()
    # print(response_data)

    # 需要处理的批量数据
    batch_data = [
        {"data": "item1"},
        {"data": "item2"},
        {"data": "item3"},
        # 添加更多数据项
    ]

    # # 用于存储每个任务的 taskid 和数据
    # tasks = []
    #
    # # 上传每个数据并获取对应的 taskid
    # for idx, data_item in enumerate(batch_data):
    #     response = requests.post("https://example.com/upload", json=data_item)
    #     if response.status_code == 200:
    #         taskid = response.json().get("taskid")
    #         # 将taskid添加到批量数据中
    #         batch_data[idx]['taskid'] = taskid
    #         tasks.append({"taskid": taskid, "index": idx})
    #         print(f"Uploaded data: {data_item}, Task ID: {taskid}")
    #     else:
    #         print(f"Failed to upload data: {data_item}")
    #
    # # 循环检查每个任务的状态并获取结果
    # while tasks:
    #     for task in tasks[:]:  # 使用切片来避免在迭代时修改列表
    #         taskid = task['taskid']
    #         status_url = "https://example.com/status/{taskid}".format(taskid=taskid)
    #         result_url =  "https://example.com/result/{taskid}".format(taskid=taskid)
    #
    #         # 查询任务状态
    #         status_response = requests.get(status_url)
    #         if status_response.status_code == 200:
    #             status = status_response.json().get("status")
    #             print(f"Task {taskid} status: {status}")
    #
    #             if status == "completed":
    #                 # 获取结果
    #                 result_response = requests.get(result_url)
    #                 if result_response.status_code == 200:
    #                     result = result_response.json().get("result")
    #                     print(f"Result for task {taskid}: {result}")
    #
    #                     # 将结果保存回对应的批量数据项
    #                     batch_data[task['index']]['result'] = result
    #                 else:
    #                     print(f"Failed to get result for task {taskid}")
    #
    #                 # 移除已完成的任务
    #                 tasks.remove(task)
    #             elif status == "failed":
    #                 print(f"Task {taskid} failed.")
    #                 tasks.remove(task)
    #             else:
    #                 print(f"Task {taskid} is still processing.")
    #         else:
    #             print(f"Failed to check status for task {taskid}")
