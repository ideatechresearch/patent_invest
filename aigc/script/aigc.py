import requests
import logging
import json
import httpx
import re, hashlib

try:
    import aiohttp

    USE_AIOHTTP = True
except ImportError:
    import httpx

    USE_AIOHTTP = False

AI_API_KEYS = {'moonshot': "",
               'aigc': "token-abc123"}
AIGC_HOST = 'aigc'


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
        try:
            data = json.dumps(json.loads(data), ensure_ascii=False)
        except:
            pass
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

    @classmethod
    def _request_helper(cls, url, body, headers, timeout, stream=False):
        try:
            return cls.post_main(url, json.dumps(body), headers=headers, timeout=timeout, stream=stream)
        except Exception as e:
            return str(e)

    def run_main(self, application, method, url, data=None, header=None):
        if application == "jira":
            application = "https://j.ideatech.info"
        if application == "confluence":
            application = "https://c.ideatech.info"

        # Log().info('请求方法的request data=' + str(data))
        try:
            if method == 'post':
                res = self.post_main(application + url, data, header)
                if res.status_code == 500:
                    context = ''
                else:
                    context = res.text
                res_code = res.status_code
                self.result.set_result(context, res_code)
                res.close()
                return self.result  # return res
            if method == "get":
                res = self.get_main(application + url, data, header)
                if res.status_code == 500:
                    context = ''
                else:
                    context = res.text
                res_code = res.status_code
                self.result.set_result(context, res_code)
                res.close()
                return self.result  # return res
            if method == "put":
                res = self.put_main(application + url, data, header)
                if res.status_code == 500:
                    context = ''
                else:
                    context = res.text
                res_code = res.status_code
                self.result.set_result(context, res_code)
                res.close()
                return self.result  # return res
        except requests.RequestException as e:
            logging.error("send_request_json_data_post请求出现异常:{0}".format(e))

    # AIGC_HOST= '47.110.156.41'


async def call_http_request(url, headers=None, time_out=100.0, **kwargs):
    async with httpx.AsyncClient() as cx:
        try:
            response = await cx.get(url, headers=headers, timeout=time_out, **kwargs)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(e)
            return None


class Local_Aigc:
    HOST = 'aigc'  # '47.110.156.41'
    AI_Models = [
        # https://platform.moonshot.cn/console/api-keys
        {'name': 'moonshot', 'type': 'default', 'api_key': '', 'data': None,
         "model": ["moonshot-v1-32k", "moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"],
         'url': "https://api.moonshot.cn/v1/chat/completions", 'base_url': "https://api.moonshot.cn/v1"},
        # {'name': 'aigc', 'type': 'default', 'api_key': '', 'data': None,
        #  "model": [],
        #  'url': f"http://{AIGC_HOST}:7000/v1/chat/completions", 'base_url': f"http://{AIGC_HOST}:7000/v1"}
    ]
    _client = None  # httpx.AsyncClient | aiohttp.ClientSession
    _is_aiohttp = False
    _is_sync = False
    _timeout = 100

    def __init__(self, host=AIGC_HOST, haio=USE_AIOHTTP, use_sync=False, time_out: int | float = 100.0):
        self.__class__.HOST = host
        self.__class__._is_aiohttp = haio
        self.__class__._is_sync = use_sync
        self.__class__._timeout = time_out

    async def __aenter__(self):
        await self.__class__.init_clients()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.__class__.close_http_client()

    @classmethod
    async def init_clients(cls):
        """获取每个模型的数据"""
        if cls._client is None:
            if cls._is_sync:
                cls._client = requests.Session()
            elif cls._is_aiohttp:
                connector = aiohttp.TCPConnector(limit=100, limit_per_host=20)
                timeout = aiohttp.ClientTimeout(total=cls._timeout, sock_read=cls._timeout,
                                                sock_connect=5.0)
                cls._client = aiohttp.ClientSession(connector=connector, timeout=timeout)
            else:
                limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
                timeout = httpx.Timeout(cls._timeout, read=cls._timeout, write=30.0, connect=5.0)
                cls._client = httpx.AsyncClient(limits=limits, timeout=timeout)

        if not any(model['name'] == 'aigc' for model in cls.AI_Models):
            cls.AI_Models.append(
                {'name': 'aigc', 'type': 'default', 'api_key': '', 'data': None, "model": [],
                 'url': f"http://{cls.HOST}:7000/v1/chat/completions",
                 'base_url': f"http://{cls.HOST}:7000/v1"
                 })

        for model in cls.AI_Models:
            model_name = model.get('name')
            api_key = AI_API_KEYS.get(model_name)
            if api_key:
                model['api_key'] = api_key

            # ai_client = OpenAI(api_key=model['api_key'] or 'empty',base_url=model['base_url'])
            url = model['base_url'] + '/models'
            try:
                models = await cls.get_resp(url)  # await call_http_request(url)
                # aigc_models = [m.model_dump() for m in await client.models.list().data]
                if models:
                    model['data'] = models['data']
                    model["model"] = [m['id'] for m in models['data']]

            except Exception as e:
                print(e)

    @classmethod
    async def close_http_client(cls):
        if cls._client:
            if cls._is_sync:
                cls._client.close()
            elif cls._is_aiohttp:
                if not cls._client.closed:
                    await cls._client.close()
            else:
                if not cls._client.is_closed:
                    await cls._client.aclose()
            cls._client = None

    @classmethod
    async def get_resp(cls, url, headers=None, **kwargs):
        if cls._is_sync:
            resp = cls._client.get(url, headers=headers, timeout=(cls._timeout, cls._timeout), **kwargs)  # params=data
            resp.raise_for_status()
            return resp.json()
        elif cls._is_aiohttp:
            async with cls._client.get(url, headers=headers or {}, **kwargs) as resp:
                resp.raise_for_status()
                return await resp.json()  # await resp.text()
        else:
            resp = await cls._client.get(url, headers=headers, **kwargs)
            resp.raise_for_status()  # 如果请求失败，则抛出异常
            return resp.json()

    @classmethod
    async def post_resp(cls, url, json=None, headers=None, **kwargs):
        if cls._is_sync:
            resp = cls._client.post(url, json=json, headers=headers, timeout=(cls._timeout, cls._timeout), **kwargs)
            resp.raise_for_status()
            return resp.json()
        elif cls._is_aiohttp:
            async with cls._client.post(url, json=json, headers=headers or {}, **kwargs) as resp:
                resp.raise_for_status()
                return await resp.json()
        else:
            resp = await cls._client.post(url, json=json, headers=headers, **kwargs)
            resp.raise_for_status()  # 如果请求失败，则抛出异常
            return resp.json()

    @classmethod
    def find_model(cls, name: str, model_id: int = 0):
        model = next((model for model in cls.AI_Models if model['name'] == name or name in model['model']), None)
        if model:
            model_items = model['model']
            if name in model_items:
                return model, name

            model_id %= len(model_items)
            return model, model_items[model_id]
        return None, None

    @classmethod
    async def get_chat_payload(cls, messages: list[dict] = None, user_request: str = '', system: str = '',
                               temperature: float = 0.4, top_p: float = 0.8, max_tokens: int = 1024,
                               model='moonshot', model_id=0, images: list = None):

        model_info, name = cls.find_model(model, model_id)

        if isinstance(messages, list) and messages:
            if system:
                if messages[0].get('role') != 'system':
                    messages.insert(0, {"role": "system", "content": system})
                # messages[-1]['content'] = messages[0]['content'] + '\n' + messages[-1]['content']

            if user_request:
                if messages[-1]["role"] != 'user':
                    messages.append({'role': 'user', 'content': user_request})
                else:
                    pass
                    # if messages[-1]["role"] == 'user':
                    #     messages[-1]['content'] = user_request
            else:
                if messages[-1]["role"] == 'user':
                    user_request = messages[-1]["content"]
        else:
            if messages is None:
                messages = []
            if system:  # system_message
                messages = [{"role": "system", "content": system}]
            messages.append({'role': 'user', 'content': user_request})

        if images:  # 图片内容理解,(str, list[dict[str, str | Any]]))
            messages[-1]['content'] = [{"type": "text", "text": user_request}]  # text-prompt 请详细描述一下这几张图片。这是哪里？
            messages[-1]['content'] += [{"type": "image_url", "image_url": {"url": image}} for image in images]

        # print(messages)
        payload = dict(
            model=name,  # 默认选择第一个模型
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            # top_k=50,
            max_tokens=max_tokens,
            stream=False,
            # extra_body = {"prefix": "```python\n", "suffix":"后缀内容"} 希望的前缀内容,基于用户提供的前缀信息来补全其余的内容
            # response_format={"type": "json_object"}
            # "tools":retrieval、web_search、function
        )
        # payload = {**payload, **kwargs}

        return model_info, payload

    @classmethod
    async def ai_chat(cls, model_info: dict = None, payload: dict = None, get_content: bool = True,
                      **kwargs) -> str | dict:
        """
        模拟发送请求给AI模型并接收响应。
        :param model_info: 模型信息（如名称、ID、配置等）
        :param payload: 请求的负载，通常是对话或文本输入
        :param get_content: 返回模型响应类型
        :param kwargs: 其他额外参数
        :return: 返回模型的响应
        """
        if not cls._client or not any(model['model'] for model in cls.AI_Models):
            await cls.init_clients()

        if not payload:
            model_info, payload = await cls.get_chat_payload(**kwargs)
        else:
            payload.update(kwargs)  # {**payload, **kwargs}

        if not model_info:
            return f"error occurred: no model,{payload}" if get_content else {"error": "no model"}

        # completion = await client.chat.completions.create(**payload)
        # return completion.choices[0].message.content if get_content else completion.model_dump()

        url = model_info['url'] if model_info.get('url') else model_info['base_url'] + '/chat/completions'
        api_key = model_info['api_key']
        headers = {'Content-Type': 'application/json', }
        # payload = payload.copy()
        if api_key:
            if isinstance(api_key, list):
                idx = model_info['model'].index(payload["model"])
                api_key = model_info['api_key'][idx]
            headers["Authorization"] = f'Bearer {api_key}'

        # print(headers, payload, url)

        try:
            data = await cls.post_resp(url, headers=headers, json=payload)

            if get_content:
                result = data.get('choices', [{}])[0].get('message', {}).get('content')
                if result:
                    return result
                # print(response.text)
            return data

        except Exception as e:
            print(f"HTTP error occurred: {e}", model_info, url)  # can't start new thread
            return f"HTTP error occurred: {e}" if get_content else {
                "error": str(e),
                "status": "failed"
            }


def request_aigc(history, question, system, model_name, agent='0', stream=False, host=AIGC_HOST, get_content=False,
                 time_out: int | float = 100, **kwargs) -> str | dict:
    headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
    url = f"http://{host}:7000/message/"
    # url = 'http://47.110.156.41:7000/message'  # 外网
    body = {
        "agent": agent,
        "extract": "json",
        "messages": history,
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

        "name": None,
        "uuid": None,
        "user_id": "idea_ai_robot",
        "robot_id": None,

        "filter_time": 0,
        "filter_limit": -500,
        "use_hist": False,
    }

    body.update(kwargs)
    try:
        response = requests.post(url, headers=headers, json=body, timeout=(10, time_out), stream=stream)
        data = response.json()

    except Exception as e:
        print(body)
        return f"HTTP error occurred: {e}" if get_content else {"error": str(e), "status": "failed"}

    if get_content:
        return data.get('transform') or data.get('answer') or data
    return data


async def request_aigc_async(history: list, question, system, model_name, agent='0', stream=False, host=AIGC_HOST,
                             get_content=False, time_out: int | float = 100, **kwargs) -> str | dict:
    headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
    url = f"http://{host}:7000/message/"
    # url = 'http://47.110.156.41:7000/message'  # 外网
    payload = {
        "agent": agent,
        "extract": "json",
        "messages": history,
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

        "name": None,
        "uuid": None,
        "user_id": "idea_ai_robot",
        "robot_id": None,

        "filter_time": 0,
        "filter_limit": -500,
        "use_hist": False,
    }

    payload.update(kwargs)  # 只更新 kwargs 里面的 key

    def extract(data):
        return data.get('transform') or data.get('answer') or data if get_content else data

    # print(headers, payload, url)
    limits = httpx.Limits(max_connections=100, max_keepalive_connections=50)
    timeout = httpx.Timeout(time_out, read=60.0, write=30.0, connect=5.0)
    try:
        async with httpx.AsyncClient(limits=limits, timeout=timeout) as cx:
            response = await cx.post(url, headers=headers, json=payload)
            response.raise_for_status()  # 如果请求失败，则抛出异常
            data = response.json()
            return extract(data)

    except (RuntimeError, httpx.HTTPStatusError, httpx.RequestError) as e:
        print(f"[httpx] 请求失败: {e}，尝试 fallback requests...")
        response = RunMethod().post_main(url, json.dumps(payload), headers=headers, timeout=(30, time_out),
                                         stream=stream)

        if response.status_code == 200:
            data = json.loads(response.content)  # response.content.decode('utf-8')
            return extract(data)
        else:
            raise RuntimeError(
                f"[requests fallback] 返回为空或状态码异常: {response.status_code}, 内容: {response.text}")

    except Exception as e:
        print(f"[requests] fallback error: {e}，尝试最终 fallback 函数...")
        # fallback to sync version:can't start new thread

    return request_aigc(history, question, system, model_name, agent, stream, host, get_content, time_out, **kwargs)


async def aigc_completion_wechat(question='', system='', name=None, robot_id='机器人小D',
                                 model_id='qwen', agent='0', filter_time=0, time_out=100, **kwargs):
    payload = {
        "history": [],
        "question": question,
        "system": system,
        "model_name": model_id,
        "agent": agent,
        "extract": "wechat",
        "stream": False,
        "temperature": 0.4,
        "max_tokens": 4096,
        "robot_id": robot_id,
        "name": name,
        "filter_time": filter_time,
        "filter_limit": -100,
        "use_hist": True,
    }

    kwargs.pop("get_content", None)
    payload.update(kwargs)

    return await request_aigc_async(**payload, get_content=True, time_out=time_out)


def aigc_message_callback(question, system, history: list[dict] = None, keywords: list = None,
                          callback_data: str | dict = None, model_id='moonshot', agent='0', host=AIGC_HOST,
                          time_out: int | float = 100, **kwargs) -> dict:
    headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
    url = f"http://{host}:7000/submit_messages"
    param = {
        "stream": False,
        "temperature": 0.6,
        "top_p": 0.7,
        "max_tokens": 4096,
        "prompt": system,
        "question": question,
        "keywords": keywords,
        "tools": [],
        'images': [],
        "callback": {},
        "agent": agent,
        "extract": "json",
        "model_name": model_id,
        "model_id": 0,
    }
    body = {
        "uuid": None,
        "name": None,
        "robot_id": None,
        "user_id": "idea_ai_robot",
        "use_hist": False,
        "filter_limit": -500,
        "filter_time": 0.0,
        "messages": history,  # not user_messages and use_hist get message_records from db, else use chat_history
        "params": [param]
    }
    if callback_data:
        if isinstance(callback_data, str):
            param["callback"]["url"] = callback_data
        elif isinstance(callback_data, dict):
            param["callback"] = callback_data

    for k, v in kwargs.items():
        if k in param:
            param[k] = v
        else:
            body[k] = v  # 非 params 里的放顶层

    body["params"][0] = param

    try:
        response = RunMethod().post_main(url, json.dumps(body), headers=headers, timeout=(30, time_out))
        data = json.loads(response.content)  # response.content.decode('utf-8')

    except Exception as e:
        print(body)
        raise RuntimeError(f"[requests] 返回为空或状态码异常: {e}")

    return data


def aigc_message_wechat(question, system, name=None, history: list[dict] = None, keywords: list = None,
                        model_id='moonshot', agent='0', host=AIGC_HOST, time_out: int | float = 100, **kwargs) -> dict:
    headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
    url = f"http://{host}:7000/submit_messages"

    body = {
        "uuid": None,
        "name": name,
        "robot_id": None,
        "user_id": "idea_ai_robot",
        "use_hist": True,
        "filter_limit": -500,
        "filter_time": 0.0,
        "messages": history,  # not user_messages and use_hist get message_records from db, else use chat_history
        "params": [{
            "stream": False,
            "temperature": 0.8,
            "top_p": 0.8,
            "max_tokens": 4096,
            "prompt": system,
            "question": question,
            "keywords": keywords,
            "tools": [],
            'images': [],
            "callback": {},
            "agent": agent,
            "extract": "wechat",
            "model_name": model_id,
            "model_id": 0,
            # "score_threshold": 0.0,
            # "top_n": 10,
        }]
    }
    for k, v in kwargs.items():
        if k in body["params"][0]:
            body["params"][0][k] = v
        else:
            body[k] = v  # 非 params 里的放顶层

    try:
        response = RunMethod().post_main(url, json.dumps(body), headers=headers, timeout=(30, time_out))
        data = json.loads(response.content)  # response.content.decode('utf-8')

    except Exception as e:
        print(body)
        raise RuntimeError(f"[requests] 返回为空或状态码异常: {e}")

    return data


def aigc_chat(user_request: str, system: str, model="moonshot:moonshot-v1-8k", get_content: bool = True,
              host=AIGC_HOST, time_out: int | float = 100, **kwargs):
    history = [
        {"role": "system", "content": system or "你是一个智能助手，帮助用户解决问题"},
        {"role": "user", "content": user_request}
    ]
    payload = dict(
        model=model,
        messages=history,
        temperature=0.3,
        top_p=0.8,
        # top_k=50,
        max_tokens=1024,
        stream=False,
    )
    payload.update(kwargs)

    headers = {'Content-Type': 'application/json', }
    url = f"http://{host}:7000/v1/chat/completions" 
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=(10, time_out))
        data = response.json()
        return data.get('choices', [{}])[0].get('message', {}).get('content') if get_content else data
    except Exception as e:
        print(payload)
        return f"HTTP error occurred: {e}" if get_content else {"error": str(e), "status": "failed"}


async def ai_chat_async(messages: list[dict] = None, user_request: str = '', system: str = '',
                        temperature: float = 0.4, top_p: float = 0.8, max_tokens: int = 1024, _client=None,
                        model='moonshot', host=AIGC_HOST, get_content: bool = True, api_key: str = None,
                        time_out: int | float = 100, **kwargs) -> str | dict:
    if not messages:
        messages = []
    if system and (not messages or messages[0].get("role") != "system"):
        messages.insert(0, {"role": "system", "content": system})
    if user_request and (not messages or messages[-1].get("role") != "user"):
        messages.append({"role": "user", "content": user_request})

    payload = dict(
        model=model,  # 默认选择第一个模型
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        # top_k=50,
        max_tokens=max_tokens,
        stream=False,
        # extra_body = {"prefix": "```python\n", "suffix":"后缀内容"} 希望的前缀内容,基于用户提供的前缀信息来补全其余的内容
        # response_format={"type": "json_object"}
        # "tools":retrieval、web_search、function
    )
    payload.update(kwargs)

    if _client:
        completion = _client.chat.completions.create(**payload)
        return completion.choices[0].message.content if get_content else completion.model_dump()

    url = f"http://{host}:7000/v1/chat/completions"
    headers = {'Content-Type': 'application/json', }
    if api_key:
        headers["Authorization"] = f'Bearer {api_key}'

    # print(headers, payload, url)
    limits = httpx.Limits(max_connections=100, max_keepalive_connections=50)
    timeout = httpx.Timeout(time_out, read=60.0, write=30.0, connect=5.0)
    try:
        async with httpx.AsyncClient(limits=limits, timeout=timeout) as cx:
            response = await cx.post(url, headers=headers, json=payload)
            response.raise_for_status()  # 如果请求失败，则抛出异常
            data = response.json()

            if get_content:
                result = data.get('choices', [{}])[0].get('message', {}).get('content')
                if result:
                    return result
                print(response.text)
            return data

    except (RuntimeError, httpx.HTTPStatusError, httpx.RequestError) as e:
        print(f"[httpx] 请求失败: {e}，尝试 fallback requests...")
        response = RunMethod().post_main(url, json.dumps(payload), headers=headers, timeout=(30, time_out))
        if response.status_code == 200:
            data = json.loads(response.content)  # response.content.decode('utf-8')
            return data.get('choices', [{}])[0].get('message', {}).get('content') if get_content else data
        else:
            raise RuntimeError(
                f"[requests fallback] 返回为空或状态码异常: {response.status_code}, 内容: {response.text}")

    except Exception as e:
        print(f"[requests] fallback error: {e}，尝试最终 fallback 函数...")

    return aigc_chat(user_request, system, model, get_content, host, time_out, **kwargs)


async def request_llm(question, system, keywords=[], model_name='qwen', agent='0', host=AIGC_HOST, **kwargs):
    headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
    url = f"http://{host}:7000/llm"
    body = {
        'prompt': system,
        "question": question,
        "agent": agent,
        "suffix": "",
        "stream": False,
        "temperature": 0.4,
        "top_p": 0.8,
        "model_name": model_name,
        "model_id": 0,
        "extract": "json",
        "max_tokens": 4096,
        "keywords": keywords,
        "tools": [],
    }

    body.update(kwargs)
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(url, json=body, headers=headers)
            response.raise_for_status()
            return response.json()
    except (RuntimeError, httpx.HTTPStatusError, httpx.RequestError) as e:
        response = RunMethod().post_main(url, json.dumps(body), headers=headers, timeout=(10, 60))
        return json.loads(response.content)
    except Exception as e:
        print(e)


async def request_classify(question, intent_keywords: dict[str, list] = None, last_intents=None,
                           host=AIGC_HOST, **kwargs):
    headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
    url = f"http://{host}:7000/classify"
    body = {
        'query': question,
        "class_terms": intent_keywords,
        "class_default": last_intents,
        'emb_model': 'text-embedding-v2',
        "rerank_model": "BAAI/bge-reranker-v2-m3",
        'cutoff': 0.85,
    }

    body.update(kwargs)
    timeout = httpx.Timeout(60, read=60.0, write=30.0, connect=5.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as cx:
            response = await cx.post(url, json=body, headers=headers)
            response.raise_for_status()
            return response.json()
    except (RuntimeError, httpx.HTTPStatusError, httpx.RequestError) as e:
        response = RunMethod().post_main(url, json.dumps(body), headers=headers, timeout=(10, 60))
        return json.loads(response.content)
    except Exception as e:
        print(e)


async def request_tools(question, tools=[], model_name='qwen-turbo', host=AIGC_HOST, **kwargs):
    headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
    url = f"http://{host}:7000/tools"
    body = {
        "messages": [
            # {
            #     "content": question,
            #     "role": "user"
            # }
        ],
        "tools": tools,
        "model_id": 0,
        "model_name": model_name,
        "prompt": question,
        "temperature": 0.01,
        "top_p": 0.95
    }

    body.update(kwargs)
    timeout = httpx.Timeout(60, read=60.0, write=30.0, connect=5.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=body, headers=headers)
            response.raise_for_status()
            return response.json()
    except (RuntimeError, httpx.HTTPStatusError, httpx.RequestError) as e:
        response = RunMethod().post_main(url, json.dumps(body), headers=headers, timeout=(10, 60))
        return json.loads(response.content)
    except Exception as e:
        print(e)


def generate_embeddings(sentences: list, model_name='qwen', host=AIGC_HOST):
    url = f'http://{host}:7000/embeddings/'
    payload = {'texts': sentences, 'model_name': model_name, 'model_id': 0}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()['embedding']

    return []


async def request_knowledge(question, model_name='BAAI/bge-reranker-v2-m3', version=0, host=AIGC_HOST, **kwargs):
    headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
    url = f"http://{host}:7000/knowledge/"
    body = {
        "text": question,
        "rerank_model": model_name,
        "version": version,
    }

    body.update(kwargs)
    timeout = httpx.Timeout(60, read=60.0, write=30.0, connect=5.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, params=body, headers=headers)
            response.raise_for_status()
            return response.json()
    except (RuntimeError, httpx.HTTPStatusError, httpx.RequestError) as e:
        response = RunMethod().get_main(url, json.dumps(body), headers=headers, timeout=(10, 60))
        return json.loads(response.content)
    except Exception as e:
        print(e)


def random_responses():
    import random
    responses = ['机器人暂时没有回复哦~',
                 "对不起，我不明白",
                 "我正在尝试",
                 "我不能这么做",
                 "请再重复一次",
                 "有什么可以帮助您的吗?",
                 "请问具体问题是什么?",
                 "你好，我现在正在线上"
                 # "当然可以",
                 # "我能做到",
                 ]
    weights = [0.25, 0.15, 0.1, 0.05, 0.07, 0.09, 0.14, 0.15]
    # 根据权重选择响应
    return random.choices(responses, weights=weights, k=1)[0]


def format_for_wechat(text):
    formatted_text = text
    formatted_text = re.sub(r'\*\*(.*?)\*\*', r'✦\1✦', formatted_text)  # **粗体** 转换为 ✦粗体✦样式
    formatted_text = re.sub(r'!!(.*?)!!', r'❗\1❗', formatted_text)  # !!高亮!! 转换为 ❗符号包围
    # formatted_text = re.sub(r'__(.*?)__', r'※\1※', formatted_text)  # __斜体__ 转换为星号包围的样式
    formatted_text = re.sub(r'~~(.*?)~~', r'_\1_', formatted_text)  # ~~下划线~~ 转换为下划线包围
    formatted_text = re.sub(r'\^\^(.*?)\^\^', r'||\1||', formatted_text)  # ^^重要^^ 转换为 ||重要|| 包围
    formatted_text = re.sub(r'######\s+(.*?)(\n|$)', r'[\1]\n', formatted_text)  # ###### 六级标题
    formatted_text = re.sub(r'#####\s+(.*?)(\n|$)', r'《\1》\n', formatted_text)  # ##### 五级标题
    formatted_text = re.sub(r'####\s+(.*?)(\n|$)', r'【\1】\n', formatted_text)  # #### 标题转换
    formatted_text = re.sub(r'###\s+(.*?)(\n|$)', r'— \1 —\n', formatted_text)  # ### 三级标题
    formatted_text = re.sub(r'##\s+(.*?)(\n|$)', r'—— \1 ——\n', formatted_text)  # ## 二级标题
    formatted_text = re.sub(r'#\s+(.*?)(\n|$)', r'※ \1 ※\n', formatted_text)  # # 一级标题
    # formatted_text = re.sub(r'```([^`]+)```',
    #                         lambda m: '\n'.join([f'｜ {line}' for line in m.group(1).splitlines()]) + '\n',
    #                         formatted_text)
    # formatted_text = re.sub(r'`([^`]+)`', r'「\1」', formatted_text)  # `代码` 转换为「代码」样式
    # formatted_text = re.sub(r'>\s?(.*)', r'💬 \1', formatted_text)  # > 引用文本，转换为聊天符号包围
    # formatted_text = re.sub(r'^\s*[-*+]\s+', '• ', formatted_text, flags=re.MULTILINE)  # 无序列表项
    # formatted_text = re.sub(r'^\s*\d+\.\s+',f"{next(ordinal_iter)} ", formatted_text, flags=re.MULTILINE)  # 有序列表项
    formatted_text = re.sub(r'\n{2,}', '\n\n', formatted_text)  # 转换换行以避免多余空行

    return formatted_text.strip()


def generate_hash_key(*args, **kwargs):
    """
    根据任意输入参数生成唯一的缓存键。
    :param args: 任意位置参数（如模型名称、模型 ID 等）
    :param kwargs: 任意关键字参数（如其他描述性信息）
    :return: 哈希键
    """
    # 将位置参数和关键字参数统一拼接成一个字符串
    inputs = []
    for arg in args:
        if isinstance(arg, list):
            inputs.extend(map(str, arg))  # 如果是列表，逐个转换为字符串
        else:
            inputs.append(str(arg))

    for key, value in kwargs.items():
        inputs.append(f"{key}:{value}")  # 格式化关键字参数为 key:value

    joined_key = "|".join(inputs)  # [:1000]
    # 返回 MD5 哈希
    return hashlib.md5(joined_key.encode()).hexdigest()


if __name__ == '__main__':
    import asyncio


    async def main():
        config = Local_Aigc(host="47.110.156.41", haio=True)
        # await Local_Aigc.init_clients()
        try:
            res = await Local_Aigc.ai_chat(model='moonshot:moonshot-v1-8k', user_request='你好')
            print(res)
            print(await Local_Aigc.ai_chat(model='qwen:qwq-plus', user_request='你好'))
            print(Local_Aigc.AI_Models)
            res = await ai_chat_async(model='moonshot:moonshot-v1-8k', user_request='你好')
            print(res)
        finally:
            await Local_Aigc.close_http_client()


    asyncio.run(main())
