import requests, httpx
import logging
import json, time


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
        "messages": messages,
        "keywords": [],
        "tools": [],

        "model_id": 0,
        "model_name": model_name,
        "prompt": system,
        "question": question,
        "stream": stream,
        "max_tokens": 1024,
        "temperature": 0.4,
        "top_p": 0.8,

        "robot_id": "technet",
        "user_id": "",
        "username": "technet",
        "uuid": "",
        "filter_time": 0,
        "filter_limit": -500,
        "use_hist": False,
    }

    body.update(kwargs)
    try:
        return RunMethod().post_main(url, json.dumps(body), headers=headers, timeout=(10, 60), stream=stream)
    except Exception as e:
        return str(e)



async def request_aigc_async(messages, question, system,  model_name, assistant_response=None, host='aigc',
                             **kwargs):
    url = f"http://{host}:7000/message"
    headers = {'accept': 'application/json', 'Content-Type': 'application/json'}

    payload = {
        "agent": "0",
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
        "stream": True,
        "temperature": 0.4,
        "top_n": 10,
        "top_p": 0.8,
        "use_hist": False,
        "user_id": "",
        "username": "technet",
        "uuid": ""
    }

    payload.update(kwargs)

    async with httpx.AsyncClient() as client:
        async with client.stream('POST', url, json=payload, headers=headers) as response:
            async for chunk in response.iter_lines(decode_unicode=True):  # .aiter_text()
                if not chunk:
                    continue
                if chunk.startswith("data: "):
                    if assistant_response is None:
                        yield f"{chunk}\n\n"
                    else:
                        line_data = chunk.lstrip("data: ")
                        if line_data == "[DONE]":
                            break
                        try:
                            parsed_content = json.loads(line_data)
                            yield f'data: {json.dumps(parsed_content, ensure_ascii=False)}\n\n'
                            # if isinstance(parsed_content, dict) and parsed_content.get("content", ""):
                            #     if parsed_content.get('role') == 'assistant':
                            #         assistant_response = [parsed_content.get('content')]
                        except json.JSONDecodeError:
                            yield f'data: {line_data}\n\n'
                            assistant_response.append(line_data)

                time.sleep(0.01)

    # for content in forward_stream(
    #         RunMethod().post_main(url, json.dumps(payload), headers=headers, timeout=(10, 60), stream=True)):
    #     if 'text' in content:
    #         yield f'data: {content["text"]}\n\n'
    #         assistant_response.append(content["text"])
    #     elif 'json' in content:
    #         yield f'data: {json.dumps(content["json"], ensure_ascii=False)}'
    #         # if content["json"].get('content') and content["json"].get('role') == 'assistant':
    #         #     assistant_response = [content["json"].get('content')]
    #     time.sleep(0.01)


def forward_stream(response):
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("data: "):
            line_data = line.lstrip("data: ")
            if line_data == "[DONE]":
                break
            try:
                parsed_content = json.loads(line_data)
                yield {"json": parsed_content}
            except json.JSONDecodeError:
                yield {"text": line_data}

    # for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):  # 二进制数据
    #     for line in chunk.splitlines():
    #         if isinstance(line, bytes):
    #             line = line.decode('utf-8', errors='ignore')
    #         if line:
    #             yield line
