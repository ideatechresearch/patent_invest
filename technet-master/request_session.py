import requests
import logging
import json


class Res:
    def set_result(self, context, code):
        self.context = context
        self.res_code = code


class RunMethod:
    def __init__(self):
        self.session = requests.Session()  # 使用session保持连接
        self.result = Res()

    def _log_request(self, method, url, data, headers):
        logging.info(f'{method.upper()} method,request url={url}')
        logging.info(f'{method.upper()} method,request data={data}')
        if headers:
            logging.info(f'{method.upper()} method,request headers={headers}')

    def _log_response(self, method, res):
        logging.info(f'{method.upper()} method,response status={res.status_code}')
        logging.info(f'{method.upper()} method,response content={res.text}')

    def _make_request(self, method, url, data=None, headers=None, verify=False, **kwargs):
        self._log_request(method, url, data, headers)

        try:
            if method == 'post':
                res = self.session.post(url, data=data, headers=headers, verify=verify, **kwargs)
            elif method == 'get':
                res = self.session.get(url, params=data, headers=headers, verify=verify, **kwargs)
            elif method == 'put':
                res = self.session.put(url, data=data, headers=headers, verify=verify, **kwargs)
            else:
                raise ValueError(f"Unsupported method: {method}")

            self._log_response(method, res)
            return res
        except requests.RequestException as e:
            logging.error(f'{method.upper()}方法请求失败: {e}')
            raise

    def post_main(self, url, data, headers=None, verify=False, **kwargs):
        return self._make_request('post', url, data, headers, verify, **kwargs)

    def get_main(self, url, data=None, headers=None, verify=False, **kwargs):
        return self._make_request('get', url, data, headers, verify, **kwargs)

    def put_main(self, url, data, headers=None, verify=False, **kwargs):
        return self._make_request('put', url, data, headers, verify, **kwargs)


def request_aigc(messages, question, system, model_name, stream=False, host='aigc', **kwargs):
    headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
    url = f"http://{host}:7000/message"
    body = {
        "agent": '0',
        "extract": "json",
        "filter_time": 0,
        "max_tokens": 1024,
        "messages": messages,
        "model_id": 0,
        "keywords": [],
        "model_name": model_name,
        "prompt": system,
        "question": question,
        "score_threshold": 0,
        "stream": stream,
        "temperature": 0.4,
        "top_n": 10,
        "top_p": 0.8,
        "use_hist": False,
        "user_id": "",
        "username": "technet",
        "uuid": ""
    }

    body.update(kwargs)
    try:
        return RunMethod().post_main(url, json.dumps(body), headers=headers, timeout=(10, 60), stream=stream)
    except Exception as e:
        return str(e)

# async def request_aigc_async(messages, question, agent, model_name, system=None, stream=False, host='aigc', **kwargs):
#     url = f"http://{host}:7000/message"
#     headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
#
#     payload = {
#         "agent": agent,
#         "extract": "json",
#         "filter_time": 0,
#         "max_tokens": 1024,
#         "messages": messages,
#         "model_id": 0,
#         "keywords": [],
#         "model_name": model_name,
#         "prompt": "",
#         "question": question,
#         "score_threshold": 0,
#         "stream": stream,  # 是否使用流式返回
#         "temperature": 0.4,
#         "top_n": 10,
#         "top_p": 0.8,
#         "use_hist": False,
#         "user_id": "",
#         "username": "technet",
#         "uuid": ""
#     }
#
#     payload.update(kwargs)
#
#     # 如果有 system 信息，将其插入消息队列
#     if system:
#         payload["messages"].insert(0, {"role": "system", "content": system})
#
#     async with httpx.AsyncClient() as client:
#         # 如果开启了流式返回
#         if stream:
#             async with client.stream('POST', url, json=payload, headers=headers) as response:
#                 async for chunk in response.aiter_text(): #response.aiter_lines():
#                     yield chunk
#         else:
#             response = await client.post(url, json=payload, headers=headers)
#             return response.json()
