import asyncio
import httpx, requests, logging
import random, json, time
import uuid, base64
from typing import List, Dict, Literal

from utils import convert_to_pinyin

from secure import md5_sign, get_baidu_access_token, get_xfyun_authorization, get_tencent_signature
from service import get_httpx_client, async_error_logger
from config import Config


# if os.getenv('AIGC_DEBUG', '0').lower() in ('1', 'true', 'yes'):
# Config.load('../config.yaml')
# Config.debug()

async def web_search_async(text: str, api_key: str = Config.GLM_Service_Key, **kwargs) -> List[Dict]:
    """
    异步执行网页搜索，使用提供的文本,调用远程工具API，并返回搜索结果。
    :param text:要搜索的文本内容
    :param api_key:用于授权API请求的密钥,不需要提供
    :param kwargs:其他可选的关键字参数，将被合并到请求数据中
    :return:
    """

    msg = [{"role": "user", "content": text}]
    tool = "web-search-pro"
    url = "https://open.bigmodel.cn/api/paas/v4/tools"
    data = {
        "request_id": str(uuid.uuid4()),
        "tool": tool,
        "stream": False,
        "messages": msg
    }
    if kwargs:
        data.update(kwargs)

    headers = {'Authorization': api_key}
    cx = get_httpx_client(time_out=Config.HTTP_TIMEOUT_SEC)
    try:
        response = await cx.post(url, json=data, headers=headers)
        response.raise_for_status()

        data = response.json()
        results = data['choices'][0]['message']['tool_calls'][1]['search_result']
        return [{
            'title': result.get('title'),
            'content': result.get('content'),
            'link': result.get('link'),
            'media': result.get('media')
        } for result in results]

        # return resp.text  # resp.content.decode()

    except httpx.HTTPStatusError as exc:
        if exc.response is not None and exc.response.status_code == 429:
            print(f"请求过快，等待 5 秒后重试...")
        return [{'error': f"HTTP error: {exc.response.status_code} -{exc}"}]
    except Exception as exc:
        return [{'error': str(exc), 'text': text, 'data': data}]
        # https://portal.azure.com/#home


@async_error_logger(max_retries=1, delay=5, exceptions=(Exception, httpx.HTTPError, httpx.HTTPStatusError))
async def web_search_intent(text: str, engine: Literal[
    "search_std", "search_pro", "search-pro-sogou", "search-pro-quark"] = "search_std",
                            recency_filter: Literal['oneDay', 'oneWeek', 'oneMonth', 'oneYear', 'noLimit'] = 'noLimit',
                            api_key: str = Config.GLM_Service_Key, **kwargs) -> Dict:
    # https://docs.bigmodel.cn/api-reference/%E5%B7%A5%E5%85%B7-api/%E7%BD%91%E7%BB%9C%E6%90%9C%E7%B4%A2
    """
    该函数通过调用外部大模型服务（GLM 大模型 API），基于输入的搜索请求 `text` 自动解析用户意图，智能生成多个查询语句，并检索多引擎网页搜索结果。
    返回的内容包括：
    - **search_intent**：结构化的意图识别结果；
    - **search_result**：每个意图对应的多条网页摘要结果，具备时效性和多角度洞察。

    该接口适用于构建 **AI问答、智能搜索推荐、信息聚合分析** 等场景。

    :param text:要搜索的文本内容
    :param api_key:用于授权API请求的密钥,不需要提供
    :param kwargs:其他可选的关键字参数，将被合并到请求数据中
    :param engine:搜索引擎版本,多引擎协作显著降低空结果率，召回率和准确率大幅提升,搜狗：覆盖腾讯生态（新闻/企鹅号）和知乎内容，在百科、医疗等垂直领域权威性强,夸克：精准触达垂直内容
    :param recency_filter:搜索指定时间范围内的网页。
    :return:"search_result" "search_intent"
    """
    url = "https://open.bigmodel.cn/api/paas/v4/web_search"
    payload = {
        "search_query": text,
        "search_engine": engine,
        "search_intent": True,  # 执行搜索意图识别，有搜索意图后执行搜索
        "search_recency_filter": recency_filter,
        "count": 10
    }
    if kwargs:
        payload.update(kwargs)

    headers = {"Authorization": f'Bearer {api_key}', "Content-Type": "application/json"}
    cx = get_httpx_client(time_out=Config.HTTP_TIMEOUT_SEC)
    response = await cx.post(url, json=payload, headers=headers)
    response.raise_for_status()
    data = response.json()
    return {k: data.get(k) for k in ("search_intent", "search_result")}  # title,content,link,media,refer,publish_date


@async_error_logger(1)
async def tokenize_with_zhipu(content: str, model: str = "glm-4-plus", api_key: str = Config.GLM_Service_Key):
    url = "https://open.bigmodel.cn/api/paas/v4/tokenizer"
    headers = {
        "Authorization": api_key,
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}]
    }
    cx = get_httpx_client()
    response = await cx.post(url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    return data["usage"]  # ["prompt_tokens"]


@async_error_logger(1)
async def web_search_tavily(text: str, topic: Literal["general", "news"] = "general",
                            time_range: Literal['day', 'week', 'month', 'year', 'd', 'w', 'm', 'y'] = 'month',
                            search_depth: Literal["basic", "advanced"] = "basic", days: int = 7,
                            api_key: str = Config.TAVILY_Api_Key, **kwargs):
    """
    执行基于 Tavily API 的 Web 搜索，支持指定主题、时间范围、搜索深度和天数等参数。",
    :param text:搜索查询的文本内容。
    :param topic:搜索的主题，可以是 'general' 或 'news'
    :param time_range:搜索的时间范围，可以是 'day', 'week', 'month', 'year', 'd', 'w', 'm', 'y'。
    :param search_depth:搜索的深度，可以是 'basic' 或 'advanced'
    :param days:当主题为 'news' 时，指定搜索的天数。
    :param api_key:Tavily API 的访问密钥。不需要提供
    :param kwargs:其他可选参数，用于扩展搜索请求。以关键字参数的形式传递。
    :return:
    """
    # https://docs.tavily.com/api-reference/endpoint/search
    url = "https://api.tavily.com/search"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    payload = {
        "query": text,
        "search_depth": search_depth,
        "topic": topic,  # finance,news
        "time_range": time_range,
        "max_results": 5,  # 0 <= x <= 20
        "include_answer": True,  # basic,true,advanced,包括 LLM 生成的对所提供查询的答案。 或返回快速答案。 返回更详细的答案
        "include_raw_content": False,  # 包括每个搜索结果的已清理和解析的 HTML 内容。 或以 Markdown 格式返回搜索结果内容。 从结果中返回纯文本，这可能会增加延迟
        # "include_images": False,
        # "include_image_descriptions": False,
        # "include_domains": [],
        # "exclude_domains": [],
        # "country": None #提升来自特定国家/地区的搜索结果,仅当 topic 为 时可用
    }
    if topic == 'news':
        payload['days'] = days  # x >= 0,Available only if topic is .news
    if kwargs:
        payload.update(kwargs)

    cx = get_httpx_client(time_out=Config.HTTP_TIMEOUT_SEC, proxy=Config.HTTP_Proxy)
    # base_url="https://api.tavily.com",client.post("/search", content=json.dumps(data))
    response = await cx.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        # print(json.dumps(data, indent=4))  # 打印响应数据
        return data['results']  # "url,title,score,published_date,content
    else:
        logging.error(f"Error: {response.status_code}")
        return [{'error': response.text}]


@async_error_logger(1)
async def web_extract_tavily(urls: str | list[str], extract_depth: Literal["basic", "advanced"] = "basic",
                             api_key: str = Config.TAVILY_Api_Key):
    """
    提取提取给定 URL 的网页信息,从一个或多个指定的 URL 中提取网页内容,比如获取github内容，已初步解析
    Tavily是一家专注于AI搜索的公司，他们的搜索会为了LLM进行优化，以便于LLM进行数据检索。
    :param urls:url or list of urls,要提取信息的 URL 或者列表
    :param api_key:Tavily API 的访问密钥，不需要指定
    :return:"url","raw_content"
    """
    url = "https://api.tavily.com/extract"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    payload = {
        "urls": urls,
        "include_images": False,
        "extract_depth": extract_depth,  # basic, advanced
        "format": "markdown"  # markdown, text
    }

    cx = get_httpx_client(time_out=Config.HTTP_TIMEOUT_SEC, proxy=Config.HTTP_Proxy)
    response = await cx.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        # print(json.dumps(data, indent=4))  # 打印响应数据
        return data.get("failed_results") or data['results']
    else:
        logging.error(f"Error: {response.status_code},{response.text}")
        return [{'error': response.text}]


@async_error_logger(1)
async def web_search_jina(text: str, api_key: str = Config.JINA_Service_Key, **kwargs):
    """
    搜索网络并获取 SERP,搜索网络并将结果转换为大模型友好文本，返回 url,title,description,content
    """
    # https://docs.tavily.com/api-reference/endpoint/search
    url = "https://s.jina.ai/"
    headers = {
        "Accept": "application/json",
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
        "X-Engine": "direct"
        # "X-Respond-With": "no-content"
    }
    payload = {
        "q": text,
        "gl": "CN"
    }
    if kwargs:
        payload.update(kwargs)

    cx = get_httpx_client(time_out=Config.HTTP_TIMEOUT_SEC, proxy=Config.HTTP_Proxy)
    response = await cx.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data["data"]
    else:
        logging.error(f"Error: {response.status_code}")
        return [{'error': response.text}]


@async_error_logger(1)
async def web_extract_jina(url: str, api_key: str = Config.JINA_Service_Key):
    """
    读取 URL 并获取其内容，20-500 RPM，Reader API 可免费使用，并提供灵活的速率限制和定价。可以提取GitHub项目等页面的文本信息
    支持 PDF 读取。它兼容大多数 PDF，包括包含大量图片的 PDF,比如获取github内容，已部分解析并结构化
    返回"title","description","content","url"，"usage"
    """
    url = f"https://r.jina.ai/{url}"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
        "X-Return-Format": "markdown"

    }
    cx = get_httpx_client(time_out=Config.HTTP_TIMEOUT_SEC)
    response = await cx.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        # print(json.dumps(data, indent=4))  # 打印响应数据
        return data["data"]
    else:
        logging.error(f"Error: {response.status_code},{response.text}")
        return [{'error': response.text}]


@async_error_logger(1)
async def segment_with_jina(content: str, tokenizer: str = "o200k_base", api_key: str = Config.JINA_Service_Key):
    # 对长文本进行分词分句，20-200 RPM
    url = 'https://api.jina.ai/v1/segment'
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"

    }
    payload = {
        "content": content,
        "tokenizer": tokenizer,  # cl100k_base
        # "return_tokens": True,
        "return_chunks": True,
        "max_chunk_length": 1000
    }
    cx = get_httpx_client()
    response = await cx.post(url, headers=headers, json=payload)
    return response.json()  # "chunk_positions","chunks"


@async_error_logger(1)
async def search_by_api(query: str, location: str = None,
                        engine: Literal["google", "bing", "baidu", "naver", "yahoo", "youtube",
                        "google_videos", "google_news", "google_images", "amazon_search", "shein_search"] | None = 'google',
                        api_key=Config.SearchApi_Key):
    """
    通过指定的搜索引擎API进行搜索，并返回搜索结果。
    :param query:搜索查询字符串，必填项。
    :param location:搜索的位置信息，可选，默认为空。
    :param engine:使用的搜索引擎，可选值包括：'google', 'bing', 'baidu', 'naver', 'yahoo', 'youtube', 'google_videos', 'google_news', 'google_images', 'amazon_search', 'shein_search'，默认为'google'。
    :param api_key:API密钥，用于访问搜索API。默认从配置中读取
    :return:
    """
    # https://www.searchapi.io/docs/google
    if engine is None:
        if "视频" in query or "movie" in query:
            engine = "google_videos"
        elif "新闻" in query or "news" in query:
            engine = "google_news"
        elif "图片" in query or "image" in query:
            engine = "google_images"
        elif "购物" in query or "buy" in query:
            engine = "amazon_search"
        elif "shein" in query.lower():
            engine = "shein_search"
        elif location:
            if "中国" in location:
                engine = "baidu"
            elif "韩国" in location:
                engine = "naver"
            elif "日本" in location:
                engine = "yahoo"
            else:
                engine = "google"
        else:
            engine = "google"

    url = "https://www.searchapi.io/api/v1/search"
    params = {
        "engine": engine,
        "q": query,
        "api_key": api_key,
        # "google_domain": "google.com",
        # "hl": "en",
        # "gl": "us"
        # 'country_code'
        # 'language': 'zh-hans',
    }
    if location:
        params['location'] = location

    async with httpx.AsyncClient(timeout=Config.HTTP_TIMEOUT_SEC, proxy=Config.HTTP_Proxy, verify=False) as cx:
        response = await cx.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            return data.get("organic_results", data)  # "knowledge_graph,answer_box,top_searches,top_stories
        else:
            logging.error(f"Error: {response.status_code}")
            return [{'error': response.text}]


@async_error_logger(1)
async def brave_search(query: str, api_key=Config.Brave_Api_Key):
    """
    查询 Brave Search 并从 Web 取回搜索结果。以下部分介绍如何将请求（包括参数和标头）整理到 Brave Web Search API 并返回 JSON 响应。
    :param query:
    :param api_key:
    :return:
    """
    url = "https://api.search.brave.com/res/v1/web/search"  # "https://api.search.brave.com/res/v1/news/search"
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "x-subscription-token": api_key
    }
    params = {
        "q": query,
        "search_lang": "zh-hans",  # en
        "country": "CH",  # US
        "safesearch": "strict",  # Drops all adult content from search results.
        "count": "10",  # 20
        "summary": False  # enables summary key generation in web search results
    }
    cx = get_httpx_client(time_out=Config.HTTP_TIMEOUT_SEC, proxy=Config.HTTP_Proxy)
    response = await cx.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        # print(json.dumps(data, indent=4))  # 打印响应数据
        return data.get('web', {}).get('results', [])
    else:
        logging.error(f"Error: {response.status_code}")
        return [{'error': response.text}]


@async_error_logger(1)
async def exa_search(query: str, category: Literal[
    "research paper", "company", "news", "pdf", "github", "tweet", "personal site", "linkedin profile", "financial report"] = "research paper",
                     api_key=Config.Exa_Api_Key):
    """
    The search endpoint lets you intelligently search the web and extract contents from the results.
    By default, it automatically chooses between traditional keyword search and Exa’s embeddings-based model, to find the most relevant results for your query.
    """
    url = "https://api.exa.ai/search"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key
    }
    payload = {
        "query": query,
        "text": True,
        'category': category,
        "numResults": 10,
    }
    cx = get_httpx_client(time_out=Config.HTTP_TIMEOUT_SEC)
    response = await cx.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data.get('results', [])
    else:
        logging.error(f"Error: {response.status_code}")
        return [{'error': response.text}]


@async_error_logger(1)
async def web_extract_exa(urls: list[str], api_key: str = Config.Exa_Api_Key):
    """
    Get the full page contents, summaries, and metadata for a list of URLs.
    Returns instant results from our cache, with automatic live crawling as fallback for uncached pages.
    """
    url = "https://api.exa.ai/contents"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key
    }
    payload = {
        "urls": urls,
        "text": True
    }
    cx = get_httpx_client(time_out=Config.HTTP_TIMEOUT_SEC)
    response = await cx.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data.get('results', [])
    else:
        logging.error(f"Error: {response.status_code},{response.text}")
        return [{'error': response.text}]


@async_error_logger(1)
async def exa_retrieved(query: str, api_key=Config.Exa_Api_Key):
    """
    https://docs.exa.ai/reference/answer
    Get an LLM answer to a question informed by Exa search results. /answer performs an Exa search and uses an LLM to generate either:

    A direct answer for specific queries. (i.e. “What is the capital of France?” would return “Paris”)
    A detailed summary with citations for open-ended queries (i.e. “What is the state of ai in healthcare?” would return a summary with citations to relevant sources)
    :param query:
    :param api_key:
    :return:
    """
    url = "https://api.exa.ai/answer"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key
    }
    payload = {
        "query": query,
        "text": True,
        "model": 'exa'  # exa-pro
    }
    cx = get_httpx_client(time_out=Config.HTTP_TIMEOUT_SEC)
    response = await cx.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        return {"answer": data.get("answer"), "citations": data.get("citations", [])}
    else:
        logging.error(f"Error: {response.status_code},{response.text}")
        return {'error': response.text}


@async_error_logger(1)
async def firecrawl_search(query: str, api_key=Config.Firecrawl_Service_Key):
    """
    使用自然语言搜索已爬网数据。
    返回结构 {"title","description","url","markdown","metadata"}
    """
    url = 'https://api.firecrawl.dev/v1/search'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    payload = {
        "query": query,
        "limit": 5,  # Maximum number of results to return,1 <= x <= 100
        "lang": "zh",  # "en"
        "country": "ch",  # "us"
        "tbs": "",  # Time-based search parameter
        "ignoreInvalidURLs": False,
    }
    cx = get_httpx_client(time_out=Config.HTTP_TIMEOUT_SEC, proxy=Config.HTTP_Proxy)
    response = await cx.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        # print(json.dumps(data, indent=4))  # 打印响应数据
        return data.get('data', [])
    else:
        logging.error(f"Error: {response.status_code},{response.text}")
        return [{'error': response.text}]


@async_error_logger(1)
async def firecrawl_scrape(url, api_key=Config.Firecrawl_Service_Key):
    #   https://docs.firecrawl.dev/features/scrape
    """
    用于抓取单个 URL。返回带有 URL 内容的 markdown。
    返回结构化数据，比如{"description","title","hostname","favicon"},简要描述网站数据集的细节
    """
    api_url = 'https://api.firecrawl.dev/v1/scrape'  # /crawl
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    payload = {
        "url": url,
        "formats": ["markdown"],  # "json",'html',
        "onlyMainContent": True
    }
    cx = get_httpx_client(time_out=Config.HTTP_TIMEOUT_SEC, proxy=Config.HTTP_Proxy)
    response = await cx.post(api_url, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        # print(json.dumps(data, indent=4))  # 打印响应数据
        return data.get('data', {}).get("metadata")
    else:
        logging.error(f"Error: {response.status_code},{response.text}")
        return {'error': response.text}


@async_error_logger(1)
async def web_extract_firecrawl(prompt: str, urls: list[str], api_key=Config.Firecrawl_Service_Key):
    # https://docs.firecrawl.dev/features/extract
    """
    提取,使用 AI 从单个页面、多个页面或整个网站中提取结构化数据。
    可能没有返回内容或请求失败
    - 目标网页可能有反爬虫措施（如验证码、限制IP、动态加载等）
    - 请求头或参数设置不正确
    - 网页内容较复杂，需特殊解析
    /extract 端点简化了从任意数量的 URL 或整个域收集结构化数据的过程。提供 URL 列表，以及描述所需信息的提示或架构。Firecrawl 处理抓取、解析和整理大型或小型数据集的细节。
     :param prompt:描述所需信息的提示或架构
     :param urls:URL 列表
    """
    api_url = 'https://api.firecrawl.dev/v1/extract'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    payload = {
        "urls": urls,
        "prompt": prompt,
        "enableWebSearch": True,
        # "schema": {"type": "object", "properties": {}}
    }
    cx = get_httpx_client(time_out=Config.HTTP_TIMEOUT_SEC, proxy=Config.HTTP_Proxy)
    response = await cx.post(api_url, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data.get('data', {})
    else:
        logging.error(f"Error: {response.status_code},{response.text}")
        return {'error': response.text}


@async_error_logger(1)
async def serper_search(query, engine: Literal['search', 'news', 'scholar', 'patents'] | None = 'search', page=2,
                        api_key=Config.SERPER_Api_Key):
    if engine:
        url = f"https://google.serper.dev/{engine}"

        payload = {
            "q": query,
            "gl": "cn",
            "hl": "zh-cn",
            "page": page
        }
    else:
        url = f"https://www.serper.dev"
        payload = {'url': query}

    headers = {
        'X-API-KEY': api_key,
        'Content-Type': 'application/json'
    }
    cx = get_httpx_client(time_out=Config.HTTP_TIMEOUT_SEC, proxy=Config.HTTP_Proxy)
    response = await cx.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        try:
            data = response.json()
            return data.get("organic") or data.get(engine)
        except json.JSONDecodeError as e:
            return {'text': response.text}
    else:
        logging.error(f"Error: {response.status_code},{response.text}")
        return [{'error': response.text}]
    # response = requests.request("POST", url, headers=headers, data=json.dumps(payload))


@async_error_logger(1)
async def duckduckgo_search(query):
    url = "https://api.duckduckgo.com/"
    params = {
        'q': query,  # 搜索查询
        'format': 'json',  # 返回 JSON 格式
        'no_redirect': 1,  # 防止重定向
        'no_html': 1,  # 去除 HTML
        'skip_disambig': 1,  # 跳过歧义提示
    }
    cx = get_httpx_client(time_out=Config.HTTP_TIMEOUT_SEC, proxy=Config.HTTP_Proxy)
    response = await cx.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        logging.error(f"Error: {response.status_code}")
        return [{'error': response.text}]


def bing_search(query, bing_api_key):
    url = f"https://api.bing.microsoft.com/v7.0/search?q={query}"
    headers = {"Ocp-Apim-Subscription-Key": bing_api_key}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    search_results = response.json()

    results = []
    for result in search_results.get('webPages', {}).get('value', []):
        title = result.get('name')
        snippet = result.get('snippet')
        link = result.get('url')
        results.append((title, snippet, link))
    return results  # "\n".join([f"{i+1}. {title}: {snippet} ({link})" for i, (title, snippet, link) in enumerate(search_results[:5])])


# https://ziyuan.baidu.com/fastcrawl/index
def baidu_search(query, baidu_api_key, baidu_secret_key):
    access_token = get_baidu_access_token(baidu_api_key, baidu_secret_key)
    search_url = "https://aip.baidubce.com/rest/2.0/knowledge/v1/search"  # https://aip.baidubce.com/rpc/2.0/unit/service/v3/chat
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    search_params = {
        "access_token": access_token,
        "query": query,
        "scope": 1,  # 搜索范围
        "page_no": 1,
        "page_size": 5
    }
    search_response = requests.post(search_url, headers=headers, data=search_params)

    if search_response.status_code == 401:  # 如果token失效
        # global baidu_access_token
        # baidu_access_token = None
        search_params["access_token"] = get_baidu_access_token(baidu_api_key, baidu_secret_key)
        search_response = requests.post(search_url, headers=headers, data=search_params)

    search_response.raise_for_status()

    search_results = search_response.json()
    results = []
    for result in search_results.get('result', []):
        title = result.get('title')
        content = result.get('content')
        url = result.get('url')
        results.append((title, content, url))
    return results  # "\n".join([f"{i+1}. {title}: {content} ({url})" for i, (title, content, url) in enumerate(search_results[:5])])


async def wikipedia_search(query: str) -> dict:
    """
    异步搜索 Wikipedia 词条并返回结构化摘要信息

    参数:
        query: 查询词

    返回:
        包含 title, page_id, summary, url 的字典
    """
    base_url = "https://en.wikipedia.org/w/api.php"

    cx = get_httpx_client(time_out=Config.HTTP_TIMEOUT_SEC, proxy=Config.HTTP_Proxy)
    try:
        # ① 搜索页面
        search_resp = await cx.get(base_url, params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json"
        })
        search_resp.raise_for_status()

        search_results = search_resp.json().get('query', {}).get('search', [])
        if not search_results:
            return {"success": False, "msg": "No search results found."}

        top_result = search_results[0]
        page_id = top_result['pageid']
        title = top_result['title']

        # ② 获取摘要内容
        detail_resp = await cx.get(base_url, params={
            "action": "query",
            "prop": "extracts",
            "pageids": page_id,
            "format": "json",
            "explaintext": 1
        })
        detail_resp.raise_for_status()

        page_data = detail_resp.json()['query']['pages'][str(page_id)]
        summary = page_data.get('extract', 'No extract found.')

        return {
            "success": True,
            "title": title,
            "page_id": page_id,
            "summary": summary.strip(),
            "url": f"https://en.wikipedia.org/?curid={page_id}"
        }

    except httpx.RequestError as e:
        response = requests.get(f"{base_url}?action=query&list=search&srsearch={query}&format=json",
                                timeout=Config.HTTP_TIMEOUT_SEC, proxies=Config.HTTP_Proxies)
        search_results = response.json().get('query', {}).get('search', [])
        if search_results:
            page_id = search_results[0]['pageid']
            page_response = requests.get(f"{base_url}?action=query&prop=extracts&pageids={page_id}&format=json",
                                         timeout=Config.HTTP_TIMEOUT_SEC, proxies=Config.HTTP_Proxies)
            page_data = page_response.json()['query']['pages'][str(page_id)]
            return {"success": True, "summary": page_data.get('extract', 'No extract found.')}
        return {"success": False, "msg": f"HTTP请求失败: {e}"}
    except Exception as e:
        return {"success": False, "msg": f"解析失败: {e}"}


@async_error_logger(1, extra_msg='arXiv请求错误', exceptions=(httpx.HTTPError, Exception))
async def arxiv_search(query: str, arxiv_id: str = None, max_results: int = 10,
                       sort_by: str = 'submittedDate') -> list[dict]:
    """
    异步方式检索arXiv论文

    参数:
        query: 搜索查询字符串
        arxiv_id: 如果提供则按 ID 查询（如 1706.03762）
        max_results: 返回结果数量
        sort_by: 排序方式(submittedDate, lastUpdatedDate, relevance)

    返回:
        论文信息字典列表
    """
    base_url = "https://export.arxiv.org/api/query"
    headers = {
        'User-Agent': 'Mozilla/5.0'
    }
    if arxiv_id:
        query = f"id:{arxiv_id}"
    params = {
        'search_query': query,
        'start': 0,
        'max_results': max_results,
        'sortBy': sort_by,
        'sortOrder': 'descending'
    }

    def parse_arxiv_xml(xml_text):
        import xml.etree.ElementTree as ET
        ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
        root = ET.fromstring(xml_text)
        entries = []

        for entry in root.findall("atom:entry", ns):
            title = entry.findtext("atom:title", default="", namespaces=ns)
            summary = entry.findtext("atom:summary", default="", namespaces=ns)
            published = entry.findtext("atom:published", default="", namespaces=ns)
            updated = entry.findtext("atom:updated", default="", namespaces=ns)
            _id = entry.findtext("atom:id", default="", namespaces=ns).split("/abs/")[-1]

            authors = [author.findtext("atom:name", default="", namespaces=ns)
                       for author in entry.findall("atom:author", ns)]

            primary_cat_elem = entry.find("arxiv:primary_category", ns)
            primary_category = primary_cat_elem.get("term") if primary_cat_elem is not None else ""

            categories = [cat.get("term") for cat in entry.findall("atom:category", ns)]

            pdf_url = ""
            for link in entry.findall("atom:link", ns):
                if link.get("type") == "application/pdf":
                    pdf_url = link.get("href")
                    break

            entries.append({
                "title": title.strip(),
                "summary": summary.strip(),
                "authors": authors,
                "published": published,
                "updated": updated,
                "arxiv_id": _id,
                "primary_category": primary_category,
                "categories": categories,
                "pdf_url": pdf_url,
            })

        return entries

    cx = get_httpx_client(time_out=Config.HTTP_TIMEOUT_SEC, proxy=Config.HTTP_Proxy)
    # import feedparser  # 用于解析arXiv的Atom feed
    response = await cx.get(base_url, params=params, headers=headers, follow_redirects=True)
    response.raise_for_status()
    return parse_arxiv_xml(response.text)


def is_city(city, region='全国'):
    # https://restapi.amap.com/v3/geocode/geo?parameters
    response = requests.get(url="http://api.map.baidu.com/place/v2/suggestion",
                            params={'query': city, 'region': region,
                                    "output": "json", "ak": Config.BMAP_API_Key, })
    data = response.json()

    # 判断返回结果中是否有城市匹配
    for result in data.get('result', []):
        if result.get('city') == city:
            return True
    return False


def get_bmap_location(address, city=''):
    response = requests.get(url="https://api.map.baidu.com/geocoding/v3",
                            params={"address": address,
                                    "city": city,
                                    "output": "json",
                                    "ak": Config.BMAP_API_Key, })
    if response.status_code == 200:
        locat = response.json()['result']['location']
        return round(locat['lng'], 6), round(locat['lat'], 6)
    else:
        print(response.text)
    return None, None


# https://lbsyun.baidu.com/faq/api?title=webapi/place-suggestion-api
@async_error_logger(1)
async def search_bmap_location(query, region='', limit=True):
    url = "http://api.map.baidu.com/place/v2/suggestion"  # 100
    params = {
        "query": query,
        "region": region,
        "city_limit": 'true' if (region and limit) else 'false',
        "output": "json",
        "ak": Config.BMAP_API_Key,
    }
    cx = get_httpx_client(time_out=Config.HTTP_TIMEOUT_SEC)
    response = await cx.get(url, params=params)
    res = []
    if response.status_code == 200:
        js = response.json()
        for result in js.get('result', []):
            res.append({'lng_lat': (round(result['location']['lng'], 6), round(result['location']['lat'], 6)),
                        'name': result["name"], 'address': result['address']})
    else:
        logging.info(response.text)
    return res  # baidu_nlp(nlp_type='address', text=region+query+result["name"]+ result['address'])


def get_amap_location(address, city=''):
    response = requests.get(url="https://restapi.amap.com/v3/geocode/geo?parameters",
                            params={"address": address,
                                    "city": city,
                                    "output": "json",
                                    "key": Config.AMAP_API_Key, })

    if response.status_code == 200:
        js = response.json()
        if js['status'] == '1':
            s1, s2 = js['geocodes'][0]['location'].split(',')
            return float(s1), float(s2)  # js['geocodes'][0]['formatted_address']
    else:
        logging.info(response.text)

    return None, None


# https://lbs.amap.com/api/webservice/guide/api-advanced/search
@async_error_logger(1)
async def search_amap_location(query, region='', limit=True):
    url = "https://restapi.amap.com/v5/place/text?parameters"  # 100
    params = {
        "keywords": query,
        "region": region,
        "city_limit": 'true' if (region and limit) else 'false',
        "output": "json",
        "key": Config.AMAP_API_Key,
    }
    cx = get_httpx_client(time_out=Config.HTTP_TIMEOUT_SEC)
    response = await cx.get(url, params=params)
    res = []
    if response.status_code == 200:
        js = response.json()
        if js['status'] == '1' and int(js['count']) > 0:
            for result in js.get('pois', []):
                s1, s2 = result['location'].split(',')
                res.append({'lng_lat': (float(s1), float(s2)),
                            'name': result["name"], 'address': result['address']})
        else:
            logging.info(response.text)  # {"count":"0","infocode":"10000","pois":[],"status":"1","info":"OK"}
    return res


# https://www.weatherapi.com/api-explorer.aspx#forecast
def get_weather(city: str, days: int = 0, date: str = None):
    # 使用 WeatherAPI 的 API 来获取天气信息
    api_key = Config.Weather_Service_Key
    base_url = "http://api.weatherapi.com/v1/current.json"
    city = convert_to_pinyin(city)
    params = {
        'key': api_key,
        'q': city,
        # Pass US Zipcode, UK Postcode, Canada Postalcode, IP address, Latitude/Longitude (decimal degree) or city name.
        'aqi': 'no'  # Get air quality data 空气质量数据
    }
    # Number of days of weather forecast. Value ranges from 1 to 10
    if days > 0:
        params['days'] = days
        params['alerts'] = 'no'
    elif date:
        # Date on or after 1st Jan, 2010 in yyyy-MM-dd format
        # Date between 14 days and 300 days from today in the future in yyyy-MM-dd format
        params['dt'] = date

    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        weather = data['current']['condition']['text']
        temperature = data['current']['temp_c']
        return f"The weather in {city} is {weather} with a temperature of {temperature}°C."
    else:
        return f"Could not retrieve weather information for {city}."


# https://console.bce.baidu.com/ai/#/ai/machinetranslation/overview/index
@async_error_logger(1)
async def baidu_translate(text: str, from_lang: str = 'zh', to_lang: str = 'en', trans_type='texttrans'):
    """百度翻译 API"""
    salt = str(random.randint(32768, 65536))  # str(int(time.time() * 1000))
    sign_str = Config.BAIDU_trans_AppId + text + salt + Config.BAIDU_trans_Secret_Key
    sign = md5_sign(sign_str)  # 需要计算 sign = MD5(appid+q+salt+密钥)
    url = "https://fanyi-api.baidu.com/api/trans/vip/translate"
    lang_map = {'fa': 'fra', 'ja': 'jp', 'ar': 'ara', 'ko': 'kor', 'es': 'spa', 'zh-TW': 'cht', 'vi': 'vie'}

    if from_lang in lang_map.keys():
        from_lang = lang_map[from_lang]
    if to_lang in lang_map.keys():
        to_lang = lang_map[to_lang]

    if to_lang == 'auto':
        to_lang = 'zh'

    params = {
        "q": text,
        "from": from_lang,
        "to": to_lang,
        "appid": Config.BAIDU_trans_AppId,
        "salt": salt,
        "sign": sign
    }
    cx = get_httpx_client(time_out=Config.HTTP_TIMEOUT_SEC)
    response = await cx.get(url, params=params)

    data = response.json()
    if "trans_result" in data:
        return data["trans_result"][0]["dst"]

    logging.info(response.text)
    # texttrans-with-dict
    url = f"https://aip.baidubce.com/rpc/2.0/mt/{trans_type}/v1?access_token=" + get_baidu_access_token(
        Config.BAIDU_translate_API_Key, Config.BAIDU_translate_Secret_Key)
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    payload = json.dumps({
        "from": from_lang,
        "to": to_lang
    })

    response = await cx.post(url, headers=headers, json=payload)
    data = response.json()
    if "trans_result" in data:
        return data["trans_result"][0]["dst"]

    # print(response.text)
    return {'error': data.get('error_msg', 'Unknown error')}


# https://cloud.tencent.com/document/product/551/15619
async def tencent_translate(text: str, source: str, target: str):
    payload = {
        "SourceText": text,
        "Source": source,
        "Target": target,
        "ProjectId": 0
    }
    url = "https://tmt.tencentcloudapi.com"
    headers = get_tencent_signature(service="tmt", host="tmt.tencentcloudapi.com", body=payload,
                                    action='TextTranslate',
                                    secret_id=Config.TENCENT_SecretId, secret_key=Config.TENCENT_Secret_Key,
                                    timestamp=int(time.time()), version='2018-03-21')

    cx = get_httpx_client(time_out=Config.HTTP_TIMEOUT_SEC)
    response = await cx.post(url, headers=headers, json=payload)

    # 检查响应状态码和内容
    if response.status_code != 200:
        logging.error(f"Error: Received status code {response.status_code},Response content: {response.text}")
        return {'error': f'{response.status_code},Request failed'}

    try:
        data = response.json()
    except Exception as e:
        print(f"Failed to decode JSON: {e},Response text: {response.text}")
        return {'error': f"Failed to decode JSON: {e},Response text: {response.text}"}

    if "Response" in data and "TargetText" in data["Response"]:
        return data["Response"]["TargetText"]
    else:
        print(f"Unexpected response: {data}")
        return {'error': f"Tencent API Error: {data.get('Response', 'Unknown error')}"}


# https://www.xfyun.cn/doc/nlp/xftrans/API.html
async def xunfei_translate(text: str, source: str = 'en', target: str = 'cn'):
    # 将文本进行base64编码
    encoded_text = base64.b64encode(text.encode('utf-8')).decode('utf-8')

    # 构造请求数据
    request_data = {
        "header": {
            "app_id": Config.XF_AppID,  # 你在平台申请的appid
            "status": 3,
            # "res_id": "your_res_id"  # 可选：自定义术语资源id
        },
        "parameter": {
            "its": {
                "from": source,
                "to": target,
                "result": {}
            }
        },
        "payload": {
            "input_data": {
                "encoding": "utf8",
                "status": 3,
                "text": encoded_text
            }
        }
    }

    headers, url = get_xfyun_authorization(api_key=Config.XF_API_Key, api_secret=Config.XF_Secret_Key,
                                           host="itrans.xf-yun.com", path="/v1/its", method='POST')
    url = 'https://itrans.xf-yun.com/v1/its'

    # 异步发送请求
    cx = get_httpx_client(time_out=Config.HTTP_TIMEOUT_SEC)
    response = await cx.post(url, json=request_data, headers=headers)
    if response.status_code == 200:
        response_data = response.json()

        # 解码返回结果中的text字段
        if "payload" in response_data and "result" in response_data["payload"]:
            base64_text = response_data["payload"]["result"]["text"]
            decoded_result = base64.b64decode(base64_text).decode('utf-8')
            data = json.loads(decoded_result)
            if "trans_result" in data:
                return data["trans_result"]["dst"]
        else:
            return {"error": "Unexpected response format"}
    else:
        return {"error": f"HTTP Error: {response.status_code}"}


# https://docs.caiyunapp.com/lingocloud-api/
def caiyun_translate(source, direction="auto2zh"):
    url = "http://api.interpreter.caiyunai.com/v1/translator"

    # WARNING, this token is a test token for new developers,
    token = Config.CaiYun_Token

    payload = {
        "source": source,
        "trans_type": direction,
        "request_id": "demo",
        "detect": True,
    }

    headers = {
        "content-type": "application/json",
        "x-authorization": "token " + token,
    }

    response = requests.request("POST", url, data=json.dumps(payload), headers=headers)
    return json.loads(response.text)["target"]


__all__ = ['arxiv_search', 'web_search_intent', 'baidu_search', 'baidu_translate', 'bing_search', 'brave_search',
           'caiyun_translate', 'duckduckgo_search', 'exa_retrieved', 'exa_search', 'firecrawl_scrape',
           'firecrawl_search', 'get_amap_location', 'get_bmap_location', 'get_httpx_client', 'get_weather', 'is_city',
           'search_amap_location', 'search_bmap_location', 'search_by_api', 'segment_with_jina', 'serper_search',
           'tencent_translate', 'tokenize_with_zhipu', 'web_extract_exa', 'web_extract_firecrawl',
           'web_extract_jina', 'web_extract_tavily', 'web_search_async', 'web_search_jina', 'web_search_tavily',
           'wikipedia_search', 'xunfei_translate']

if __name__ == "__main__":
    from utils import get_module_functions

    funcs = get_module_functions('agents.ai_search')
    print([i[0] for i in funcs])


    async def main():
        # print(await web_search_tavily('季度业绩报告'))
        # r = await web_extract_tavily('https://en.wikipedia.org/wiki/Artificial_intelligence')
        # print(r)

        r = await  brave_search('季度业绩报告')  # r.keys(),
        print(r)
        result = await web_search_async('易得融信是什么公司')
        print(result)
        paper = await  arxiv_search(query='', arxiv_id="1706.03762")
        print(paper)
        # [{'title': 'Attention Is All You Need', 'summary': 'The dominant sequence transduction models are based on complex recurrent or\nconvolutional neural networks in an encoder-decoder configuration. The best\nperforming models also connect the encoder and decoder through an attention\nmechanism. We propose a new simple network architecture, the Transformer, based\nsolely on attention mechanisms, dispensing with recurrence and convolutions\nentirely. Experiments on two machine translation tasks show these models to be\nsuperior in quality while being more parallelizable and requiring significantly\nless time to train. Our model achieves 28.4 BLEU on the WMT 2014\nEnglish-to-German translation task, improving over the existing best results,\nincluding ensembles by over 2 BLEU. On the WMT 2014 English-to-French\ntranslation task, our model establishes a new single-model state-of-the-art\nBLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction\nof the training costs of the best models from the literature. We show that the\nTransformer generalizes well to other tasks by applying it successfully to\nEnglish constituency parsing both with large and limited training data.', 'authors': ['Ashish Vaswani', 'Noam Shazeer', 'Niki Parmar', 'Jakob Uszkoreit', 'Llion Jones', 'Aidan N. Gomez', 'Lukasz Kaiser', 'Illia Polosukhin'], 'published': '2017-06-12T17:57:34Z', 'updated': '2023-08-02T00:41:18Z', 'arxiv_id': '1706.03762v7', 'primary_category': 'cs.CL', 'categories': ['cs.CL', 'cs.LG'], 'pdf_url': 'http://arxiv.org/pdf/1706.03762v7'}]
        paper = await  wikipedia_search('道法自然')
        print(paper)

        se = await  segment_with_jina(
            '''{"status": "000000", "message": "查询成功", "data": {"name": "宁波尚缔电器有限公司", "historyName": "宁波久刈网络科技有限公司", "registNo": "330215000140739", "unityCreditCode": "91330201MA281N399X", "type": "有限责任公司(自然人投资或控股)", "legalPerson": "黄发虎", "registFund": "100万人民币", "openDate": "2016年03月21日", "startDate": "2016年03月21日", "endDate": "9999年12月31日", "registOrgan": "宁波市市场监督管理局高新技术产业开发区分局", "licenseDate": "2017年04月26日", "state": "吊销，未注销", "address": "宁波高新区创苑路750号001幢759室", "scope": "电器、日用百货、母婴用品的批发、零售及网上销售；网络技术、电子商务技术开发、技术服务；网络工程设计；网页设计；市场营销推广宣传；国内各类广告设计、制作、发布、代理；图文设计、制作；企业形象设计；摄影服务。（依法须经批准的项目，经相关部门批准后方可开展经营活动）", "revokeDate": "2022年04月29日", "isOnStock": null, "lastUpdateDate": "2024年06月28日", "priIndustry": "批发和零售业", "industryCategoryCode": "F", "subIndustry": "批发业", "industryLargeClassCode": "51", "middleCategory": "纺织、服装及家庭用品批发", "middleCategoryCode": "513", "smallCategory": "", "smallCategoryCode": null, "legalPersonSurname": "黄", "legalPersonName": "发虎", "registCapital": "1,000,000.000", "registCurrency": "CNY", "typeCode": "F", "country": "中国", "province": "浙江省", "city": "宁波市", "area": "市辖区", "partners": [{"name": "黄发虎", "type": "自然人股东", "identifyType": null, "identifyNo": null, "shouldType": "", "shouldCapi": "50.0", "shoudDate": "2016年03月21日", "realType": "", "realCapi": "0.0", "realDate": null}, {"name": "曹颖", "type": "自然人股东", "identifyType": null, "identifyNo": null, "shouldType": "", "shouldCapi": "30.0", "shoudDate": "2016年03月21日", "realType": "", "realCapi": "0.0", "realDate": null}, {"name": "王菊美", "type": "自然人股东", "identifyType": null, "identifyNo": null, "shouldType": "", "shouldCapi": "20.0", "shoudDate": "2016年03月21日", "realType": "", "realCapi": "0.0", "realDate": null}], "employees": [{"name": "黄发虎", "job": "执行董事兼总经理"}, {"name": "曹颖", "job": "监事"}], "branchs": [], "changes": [{"changesType": "行业代码变更", "changesBeforeContent": "6420:互联网信息服务", "changesAfterContent": "5137:家用电器批发", "changesDate": "2017年04月26日"}, {"changesType": "其他事项备案", "changesBeforeContent": "6420:互联网信息服务", "changesAfterContent": "5137:家用电器批发", "changesDate": "2017年04月26日"}, {"changesType": "经营范围变更（含业务范围变更）", "changesBeforeContent": "网络技术、电子商务技术开发、技术服务；网络工程设计；网页设计；市场营销推广宣传；国内各类广告设计、制作、发布、代理；图文设计、制作；企业形象设计；摄影服务；日用百货、母婴用品、电器网上销售及批发、零售。", "changesAfterContent": "电器、日用百货、母婴用品的批发、零售及网上销售；网络技术、电子商务技术开发、技术服务；网络工程设计；网页设计；市场营销推广宣传；国内各类广告设计、制作、发布、代理；图文设计、制作；企业形象设计；摄影服务。（依法须经批准的项目，经相关部门批准后方可开展经营活动）", "changesDate": "2017年04月26日"}, {"changesType": "名称变更（字号名称、集团名称等）", "changesBeforeContent": "宁波久刈网络科技有限公司", "changesAfterContent": "宁波尚缔电器有限公司", "changesDate": "2017年04月26日"}]}}''')
        print(se)

        se = await web_search_intent("宁波尚缔电器有限公司是什么公司")
        print(se)


    asyncio.run(main())
