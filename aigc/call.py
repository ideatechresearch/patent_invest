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


# def hmac_sha256(secret_key: str, message: str) -> str:
#     # HMAC-SHA256 生成签名
#     return hmac.new(secret_key.encode(), message.encode(), hashlib.sha256).hexdigest()

async def send_authenticated():
    api_key, secret_key = '6b01600d-5162-4709-997c-d6fc3274c3b0', b'8g4bri8oVRCHbvEyURk20LQX5P1gRAcnL5Phl3vJz3s'

    method = "POST"
    data = {"key": "value"}
    body = json.dumps(data)
    url = "http://127.0.0.1:7000/protected"
    timestamp = str(int(time.time()))
    message = f"{method}{url}{body}{timestamp}"
    signature = hmac_sha256(secret_key, message).hex()
    headers = {
        "X-API-KEY": api_key,
        "X-SIGNATURE": signature,
        "X-TIMESTAMP": timestamp,
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(verify=False) as client:
        response = await client.post(url, headers=headers, json=data)
        print(response.text)
        return response
    #
    # duration = 60
    # success_count = 0
    # failure_count = 0
    #
    # start_time = time.time()
    # end_time = start_time + duration
    #
    # async with httpx.AsyncClient(verify=False) as client:
    #     while time.time() < end_time:
    #         # 每次请求生成新的时间戳和签名
    #         timestamp = str(int(time.time()))
    #         message = f"{method}{url}{body}{timestamp}"
    #         signature = hmac.new(secret_key.encode(), message.encode(), hashlib.sha256).hexdigest()
    #
    #         headers = {
    #             "X-API-KEY": api_key,
    #             "X-SIGNATURE": signature,
    #             "X-TIMESTAMP": timestamp,
    #             "Content-Type": "application/json",
    #         }
    #
    #         try:
    #             response = await client.post(url, headers=headers, json=data)
    #             if response.status_code == 200:
    #                 success_count += 1
    #             else:
    #                 failure_count += 1
    #
    #         except Exception as e:
    #             failure_count += 1
    #             print(f"Request failed: {e}")
    #
    # print(response.text, success_count, failure_count)


async def authenticate_user():
    original_message = f"Authenticate request at {int(time.time())}"
    # 生成 ECDSA 私钥 (SECP256k1 曲线)
    # private_key = ecdsa.SigningKey.generate(curve=ecdsa.SECP256k1)
    # private_key_hex = private_key.to_string().hex()

    # public_key_hex='83e2c687a44f839f6b3414d63e1a54ad32d8dbe4706cdd58dc6bd4233a592f78367ee1bff0e081ba678a5dfdf068d4b4c72f03085aa6ba5f0678e157fc15d305'
    private_key_hex = "20f0"  # 请使用安全的方式存储和获取私钥
    private_key = ecdsa.SigningKey.from_string(bytes.fromhex(private_key_hex), curve=ecdsa.SECP256k1)
    signature = private_key.sign(original_message.encode('utf-8'), hashfunc=hashlib.sha256)
    # 将签名转换为Base64格式
    signed_message = base64.b64encode(signature).decode('utf-8')

    # 生成公钥并转为十六进制
    public_key_hex = private_key.get_verifying_key().to_string().hex()
    print(f"Private Key: {private_key_hex}")
    print(f"Public Key: {public_key_hex}")

    data = {
        "username": "admin",
        "password": "",
        # "eth_address": "",
        "public_key": public_key_hex,
        "signed_message": signed_message,
        "original_message": original_message,
    }

    url = "http://127.0.0.1:7000/authenticate"
    headers = {
        "Content-Type": "application/json",
    }
    token = None
    async with httpx.AsyncClient(verify=False) as client:
        response = await client.post(url, headers=headers, json=data)
        if response.status_code == 200:
            # 假设返回的响应数据包含 token
            token_data = response.json()
            token = token_data.get("access_token")
            print(response.text)
        else:
            print("Response:", response.text)

    if not token:
        return

    duration = 60
    success_count = 0
    failure_count = 0

    start_time = time.time()
    end_time = start_time + duration
    headers = {
        "Authorization": f"Bearer {token}"  # 设置Bearer Token
    }
    url = "http://127.0.0.1:7000/secure"
    async with httpx.AsyncClient(verify=False) as client:
        while time.time() < end_time:
            try:
                response = await client.post(url, headers=headers)
                if response.status_code == 200:
                    success_count += 1
                else:
                    failure_count += 1

            except Exception as e:
                failure_count += 1
                print(f"Request failed: {e}")

    print(response.text, success_count, failure_count)


# 如果你在异步环境中，例如 FastAPI 应用程序中
# 否则你可以直接运行
if __name__ == "__main__":
    # main()
    from config import *

    #asyncio.run(send_authenticated())
    asyncio.run(authenticate_user())
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(connect_chat())

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
