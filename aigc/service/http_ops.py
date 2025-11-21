import requests, httpx, aiohttp, aiofiles
import json
from starlette.requests import Request
from starlette.responses import Response
# from starlette.routing import Route, Mount

from .base import *


async def aiohttp_request(method: str, url: str, session: aiohttp.ClientSession = None, json=None, data=None,
                          params=None, headers=None, timeout=30, **kwargs):
    headers = headers or {}

    # 拼接 GET query
    if method.upper() == "GET" and params:
        from urllib.parse import urlencode
        query_string = urlencode(params)
        url = f"{url}?{query_string}"

    async def fetch_url(session: aiohttp.ClientSession):
        async with session.request(method, url, json=json, data=data, headers=headers, timeout=timeout,
                                   **kwargs) as resp:
            try:
                resp.raise_for_status()
                return await resp.json()
            except aiohttp.ContentTypeError:
                return {"status": resp.status, "error": "Non-JSON response", "body": await resp.text()}

    try:
        if session:
            return await fetch_url(session)

        async with aiohttp.ClientSession(headers=headers, timeout=aiohttp.ClientTimeout(total=timeout)) as ts:
            return await fetch_url(ts)

    except aiohttp.ClientResponseError as e:
        logging.error(f"[HTTP Error] {method} {url} | Status: {e.status}, Message: {e.message}, Body: {json or data}")
        return {"status": e.status, "error": e.message, "url": url}
    except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as e:
        logging.error(f"[Connection Error] {method} {url} | Message: {e}")
        return {"status": 503, "error": f"Connection failed: {str(e)}", "url": url}
    except Exception as e:
        logging.error(f"[Unknown Error] {method} {url} | Message: {e}")
        return {"status": 500, "error": str(e), "url": url}


async def post_aiohttp_stream(url, payload: dict = None, time_out=60, headers: dict = None, **kwargs):
    timeout = aiohttp.ClientTimeout(total=time_out, sock_read=max(60, time_out), sock_connect=5.0)
    async with aiohttp.ClientSession(timeout=timeout, headers=headers or {}) as session:
        async with session.post(url, json=payload, **kwargs) as response:
            response.raise_for_status()
            buffer = bytearray()  # b""
            async for chunk in response.content.iter_any():  # .iter_chunked(1024)
                if not chunk:
                    continue
                buffer.extend(chunk.tobytes() if isinstance(chunk, memoryview) else chunk)
                while b"\n" in buffer:  # 处理缓冲区中的所有完整行
                    line, buffer = buffer.split(b"\n", 1)
                    if line.strip():
                        try:
                            yield json.loads(line.decode("utf-8").strip())
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            continue

            if buffer:
                tail_bytes = bytes(buffer).rstrip(b"\r\n")
                if tail_bytes and tail_bytes.strip():
                    try:
                        yield json.loads(tail_bytes.decode("utf-8").strip())  # if decoded_line.startswith(('{', '[')):
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        pass


async def post_httpx_sse(url, payload: dict = None, headers: dict = None, time_out=60,
                         client: httpx.AsyncClient = None, **kwargs):
    def parse_line(data):
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return None

    async def _run(cx: httpx.AsyncClient):
        try:
            async with cx.stream('POST', url, json=payload, headers=headers, **kwargs) as response:
                response.raise_for_status()
                buffer = ""
                async for line in response.aiter_lines():  # 使用 aiter_bytes() 处理原始字节流
                    line = line.strip()
                    if not line:  # 空行 -> 事件结束,开头的行 not line
                        if buffer:
                            if buffer == "[DONE]":
                                yield {"type": "done"}
                                return
                            parse = parse_line(buffer)  # 一个数据块的结束
                            if parse:
                                yield {"type": "data", "data": parse}
                            else:
                                yield {"type": "text", "data": buffer}
                            buffer = ""  # 重置清空
                        continue

                    if line.startswith("data: "):
                        content = line[6:]  # line.lstrip("data: ")
                        if content in ("[DONE]", '"[DONE]"', "DONE"):
                            yield {"type": "done"}
                            return
                        parse = parse_line(content)  # 单行 JSON
                        if parse:
                            yield {"type": "data", "data": parse}
                    else:  # 处理非 data: 行或 JSON 解析失败时
                        buffer += ("\n" + line) if buffer else line

                if buffer:  # 处理最后遗留的 buffer
                    parse = parse_line(buffer)
                    if parse:
                        yield {"type": "data", "data": parse}
                    else:
                        yield {"type": "text", "data": buffer}

        except Exception as e:
            # yield error event and return
            yield {"type": "error", "data": f"HTTP error: {e}"}

    if client:
        async for item in _run(client):
            yield item
        return

    timeout = httpx.Timeout(time_out, read=60.0, connect=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async for item in _run(client):
            yield item


async def call_http_request(url: str, headers=None, timeout=100.0, httpx_client: httpx.AsyncClient = None, **kwargs):
    """
    异步调用HTTP请求并返回JSON响应，如果响应不是JSON格式则返回文本内容
    国内，无代理，可能内容无解析
    :param url: 请求地址
    :param headers: 请求头
    :param timeout: 请求超时时间
    :param httpx_client: 外部传入的 AsyncClient 实例（可复用连接）
    :param kwargs: 其他传递给 httpx 的参数
    :return: dict 或 None
    """

    async def fetch_url(cx: httpx.AsyncClient):
        response = await cx.get(url, headers=headers, timeout=timeout, **kwargs)  # 请求级别的超时优先级更高
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type or not content_type:
            try:
                return response.json()
            except json.JSONDecodeError:
                pass

        return {'text': response.text}

    if httpx_client:
        return await fetch_url(httpx_client)

    async with httpx.AsyncClient() as cx:
        return await fetch_url(cx)


async def download_by_aiohttp(url: str, session: aiohttp.ClientSession, save_path, chunk_size=4096, in_decode=False):
    async with session.get(url) as response:
        # response = await asyncio.wait_for(session.get(url), timeout=timeout)
        if response.status == 200:
            if save_path:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                async with aiofiles.open(save_path, mode='wb') as f:
                    # response.content 异步迭代器（流式读取）,iter_chunked 非阻塞调用，逐块读取
                    async for chunk in response.content.iter_chunked(chunk_size):
                        if isinstance(chunk, (bytes, bytearray)):
                            await f.write(chunk)
                        elif isinstance(chunk, str):
                            await f.write(chunk.encode('utf-8'))  # 将字符串转为字节
                        else:
                            raise TypeError(
                                f"Unexpected chunk type: {type(chunk)}. Expected bytes or bytearray.")

                return save_path

            content = await response.read()  # 单次异步读取完整内容，小文件,await response.content.read(chunk_size)
            return content.decode('utf-8') if in_decode else content  # 将字节转为字符串,解码后的字符串 await response.text()

        print(f"Failed to download {url}: {response.status}")

    return None


async def download_by_httpx(url: str, client: httpx.AsyncClient, save_path, chunk_size=4096, in_decode=False):
    async with client.stream("GET", url) as response:  # 长时间的流式请求
        # response = await client.get(url, stream=True)
        response.raise_for_status()  # 如果响应不是2xx  response.status_code == 200:
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(save_path, mode="wb") as f:
                # response.aiter_bytes() 异步迭代器
                async for chunk in response.aiter_bytes(chunk_size=chunk_size):
                    await f.write(chunk)

            return save_path

        content = bytearray()  # 使用 `bytearray` 更高效地拼接二进制内容  b""  # raw bytes
        async for chunk in response.aiter_bytes(chunk_size=chunk_size):
            content.extend(chunk)
            # content += chunk
            # yield chunk

        return content.decode(response.encoding or 'utf-8') if in_decode else bytes(content)
        # return response.text if in_decode else response.content


async def upload_by_httpx(url: str, client: httpx.AsyncClient = None, files_path=('example.txt', b'Hello World')):
    '''
    with open("local.txt", "rb") as f:
        await upload_by_httpx("http://127.0.0.1:8000/upload", files_path=("local.txt", f))
    '''
    files = {'file': files_path}  # (filename, bytes/文件对象)

    if client:
        resp = await client.post(url, files=files)
    else:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, files=files)

    resp.raise_for_status()
    return resp.json()


def download_by_requests(url: str, save_path, chunk_size=4096, in_decode=False, timeout=30):
    """
    同步下载的流式方法
    如果目标是保存到文件，直接使用 content（无需解码）。（如图片、音频、视频、PDF 等）
    如果目标是处理和解析文本数据，且确定编码正确，使用 text。（如 HTML、JSON）
    """
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()  # 确保请求成功
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as f:
                # requests iter_content 同步使用,阻塞调用
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:  # <class 'bytes'>
                        f.write(chunk)

            return save_path

        return response.text if in_decode else response.content  # 直接获取文本,同步直接返回全部内容


def upload_by_requests(url: str, file_path, file_key='snapshot'):
    with open(file_path, "rb") as f:
        files = {file_key: f}
        response = requests.post(url, files=files)
    return response.json()


async def send_callback(callback_data: dict, result, **kwargs):
    url = callback_data.get("url")
    fallback_url = "http://127.0.0.1:7000/callback"
    if not url:
        print(f"[Missing callback URL]:{callback_data}")
        url = fallback_url

    payload = callback_data.get("payload", {})
    if not isinstance(payload, dict):
        payload = {}

    mapping = callback_data.get("mapping", {})

    def apply_mapping(data: dict):
        return {mapping.get(k, k): v for k, v in data.items()} if mapping else data

    def filter_payload(data: dict):
        return {mapping[k]: data[k] for k in mapping if k in data} if mapping else data

    if isinstance(result, dict):
        payload.update(result)
    elif isinstance(result, list) and len(result) == 1 and isinstance(result[0], dict):
        payload.update(result[0])
    else:
        payload = {**payload, "result": result}

    payload = apply_mapping(payload)
    headers = callback_data.get("headers") or {}
    params = callback_data.get("params")
    timeout = 30
    if params and isinstance(params, dict):
        timeout = params.pop("timeout", timeout)
        kwargs.update(params)

    res = None
    format_type = callback_data.get("format", 'json').lower()  # "query" or "json" form"

    async with aiohttp.ClientSession() as session:
        if format_type == "json":  # post_http_json
            res = await aiohttp_request("POST", url, session, json=payload, headers=headers, timeout=timeout, **kwargs)
        elif format_type == "query":  # get_http_query query 参数或表单参数
            query_payload = filter_payload(payload)
            res = await aiohttp_request("GET", url, session, params=query_payload, headers=headers, timeout=timeout)
        elif format_type == "form":  # post_http_form 支持 query 或 form
            headers["Content-Type"] = "application/x-www-form-urlencoded"
            res = await aiohttp_request("POST", url, session, data=payload, headers=headers, timeout=timeout)

        if res and res.get('error'):
            return await aiohttp_request("POST", fallback_url, session, json=payload, headers=headers, **kwargs)
        return res


class OperationHttp:
    def __init__(self, use_sync=False, time_out: int | float = 100.0, proxy: str = None):
        self.client = None  # httpx.AsyncClient | aiohttp.ClientSession | requests.Session
        self._is_httpx: bool = False
        self._is_sync: bool = use_sync
        self._timeout: int = time_out
        self._proxy: str = proxy  # 支持 None / "http://host:port" / "socks5://host:port"
        # self._mode: str = mode.lower()

        try:
            import httpx
            self._is_httpx = True
        except ImportError:
            try:
                import aiohttp
                self._is_httpx = False
            except ImportError:
                import requests
                self._is_sync = True

    async def __aenter__(self):
        self.init_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close_async()

    def __enter__(self):
        self.init_client()
        return self

    def __exit__(self, *args):
        self.close_sync()

    def init_client(self):
        if self.client is not None:
            return

        if self._is_sync:
            if self._is_httpx:
                transport = httpx.HTTPTransport(proxy=self._proxy or None)
                self.client = httpx.Client(timeout=self._timeout, transport=transport)  # 底层为 httpcore
            else:
                self.client = requests.Session()  # 基于 urllib3
                if self._proxy:
                    self.client.proxies.update({"http": self._proxy, "https": self._proxy})
        else:
            if self._is_httpx:
                transport = httpx.AsyncHTTPTransport(proxy=self._proxy or None)
                limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
                timeout = httpx.Timeout(self._timeout, read=self._timeout, write=30.0, connect=5.0)
                self.client = httpx.AsyncClient(limits=limits, timeout=timeout, transport=transport)
            else:
                connector = aiohttp.TCPConnector(limit=100, limit_per_host=20)
                timeout = aiohttp.ClientTimeout(total=self._timeout, sock_read=self._timeout, sock_connect=5.0)
                self.client = aiohttp.ClientSession(connector=connector, timeout=timeout)  # headers=headers or {}

            # self.semaphore = asyncio.Semaphore(30)

    async def close_client(self):
        if self._is_sync:
            self.close_sync()
        else:
            await self.close_async()

    def close_sync(self):
        if self.client:
            self.client.close()
            self.client = None

    async def close_async(self):
        if self.client:
            if self._is_httpx:
                if not self.client.is_closed:
                    await self.client.aclose()
            else:
                if not self.client.closed:
                    await self.client.close()
            self.client = None

    async def get(self, url, headers=None, **kwargs):
        timeout = kwargs.pop("timeout", self._timeout) or self._timeout
        if self._is_sync:
            if self._is_httpx:
                resp = self.client.get(url, headers=headers, timeout=timeout, **kwargs)  # params=data
            else:
                resp = self.client.get(url, headers=headers, timeout=(timeout, timeout), **kwargs)
            resp.raise_for_status()
            return resp.json()

        if self._is_httpx:
            resp = await self.client.get(url, headers=headers, timeout=timeout, **kwargs)
            resp.raise_for_status()  # 如果请求失败，则抛出异常
            return resp.json()

        async with self.client.get(url, headers=headers or {}, timeout=timeout, **kwargs) as resp:
            resp.raise_for_status()
            return await resp.json()  # await resp.text()

    async def post(self, url, json=None, headers=None, **kwargs):
        timeout = kwargs.pop("timeout", self._timeout) or self._timeout
        if self._is_sync:
            if self._is_httpx:
                resp = self.client.post(url, json=json, headers=headers, timeout=timeout, **kwargs)
            else:
                resp = self.client.post(url, json=json, headers=headers, timeout=(timeout, timeout), **kwargs)
            resp.raise_for_status()
            return resp.json()

        if self._is_httpx:
            resp = await self.client.post(url, json=json, headers=headers, timeout=timeout, **kwargs)
            resp.raise_for_status()  # 如果请求失败，则抛出异常
            return resp.json()

        async with self.client.post(url, json=json, headers=headers or {}, timeout=timeout, **kwargs) as resp:
            resp.raise_for_status()  # 抛出 4xx/5xx 错误
            return await resp.json()

    def fallback_post(self, url, json_payload, headers=None, stream=False):
        try:
            resp = requests.post(url, headers=headers, json=json_payload, timeout=(5, self._timeout), stream=stream,
                                 proxies={"http": self._proxy, "https": self._proxy} if self._proxy else None)
            if resp.status_code == 200:
                return resp.json()  # json.loads(resp.content)
            else:
                raise RuntimeError(f"[requests fallback] 返回异常: {resp.status_code}, 内容: {resp.text}")
        except Exception as e:
            print(f"[requests fallback] 请求失败: {e}")
            return None


@async_error_logger(max_retries=1, delay=3, exceptions=(Exception, httpx.HTTPError))
async def follow_http_html(url, time_out: float = 100.0, **kwargs):
    async with httpx.AsyncClient(timeout=time_out, follow_redirects=True) as cx:
        response = await cx.get(url, **kwargs)
        response.raise_for_status()
        return response.text


async def proxy_http_html(base_url: str, full_path: str, request: Request):
    url = f"{base_url}/{full_path}"
    async with httpx.AsyncClient(follow_redirects=True) as cx:
        method = request.method
        headers = dict(request.headers)
        body = await request.body()

        proxy_req = cx.build_request(method, url, headers=headers, content=body)
        resp = await cx.send(proxy_req)

        return Response(content=resp.content, status_code=resp.status_code, headers=dict(resp.headers))
