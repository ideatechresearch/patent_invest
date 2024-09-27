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
        logging.info(f'{method.upper()}方法的request url={url}')
        logging.info(f'{method.upper()}方法的request data={data}')
        if headers:
            logging.info(f'{method.upper()}方法的request headers={headers}')

    def _log_response(self, method, res):
        logging.info(f'{method.upper()}方法的response status={res.status_code}')
        logging.info(f'{method.upper()}方法的response content={res.text}')

    def _make_request(self, method, url, data=None, headers=None, verify=False):
        self._log_request(method, url, data, headers)

        try:
            if method == 'post':
                res = self.session.post(url, data=data, headers=headers, verify=verify)
            elif method == 'get':
                res = self.session.get(url, params=data, headers=headers, verify=verify)
            elif method == 'put':
                res = self.session.put(url, data=data, headers=headers, verify=verify)
            else:
                raise ValueError(f"Unsupported method: {method}")

            self._log_response(method, res)
            return res
        except requests.RequestException as e:
            logging.error(f'{method.upper()}方法请求失败: {e}')
            raise

    def post_main(self, url, data, headers=None, verify=False):
        return self._make_request('post', url, data, headers, verify)

    def get_main(self, url, data=None, headers=None, verify=False):
        return self._make_request('get', url, data, headers, verify)

    def put_main(self, url, data, headers=None, verify=False):
        return self._make_request('put', url, data, headers, verify)


def request_aigc(messages, question, agent, model_name, stream=False, host='aigc', **kwargs):
    headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
    url = f"http://{host}:7000/message"
    body = {
        "agent": agent,
        "extract": "json",
        "filter_time": 0,
        "max_tokens": 1024,
        "messages": messages,
        "model_id": 0,
        "keywords": [],
        "model_name": model_name,
        "prompt": "",
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
    if stream:
        return RunMethod().post_main(url, json.dumps(body), headers=headers)

    response = RunMethod().post_main(url, json.dumps(body), headers=headers)
    return json.loads(response.content)
