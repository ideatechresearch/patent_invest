import asyncio
import httpx, aiohttp, requests
import random
from typing import List, Dict, Tuple, Any, Union, Iterator, Sequence, Literal
from utils import convert_to_pinyin

from config import *


# Config.load('../config.yaml')
# Config.debug()
# from selectolax.parser import HTMLParser

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

    async with httpx.AsyncClient(timeout=Config.HTTP_TIMEOUT_SEC) as cx:
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

    async with httpx.AsyncClient() as cx:
        response = await cx.post(url, headers=headers, json=payload)
        data = response.json()
        return data["usage"]  # ["prompt_tokens"]


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
        "include_answer": True,  # basic,true,advanced
        # "include_raw_content": False,
        # "include_images": False,
        # "include_image_descriptions": False,
        # "include_domains": [],
        # "exclude_domains": []
    }
    if topic == 'news':
        payload['days'] = days  # x >= 0,Available only if topic is .news
    if kwargs:
        payload.update(kwargs)

    async with httpx.AsyncClient(timeout=Config.HTTP_TIMEOUT_SEC, proxy=Config.HTTP_Proxy) as cx:
        # base_url="https://api.tavily.com",client.post("/search", content=json.dumps(data))
        response = await cx.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            data = response.json()
            # print(json.dumps(data, indent=4))  # 打印响应数据
            return data['results']  # "url,title,score,published_date,content
        else:
            print(f"Error: {response.status_code}")
            return [{'error': response.text}]


async def web_extract_tavily(urls: str | list[str], api_key: str = Config.TAVILY_Api_Key):
    """
    提取提取给定 URL 的网页信息,从一个或多个指定的 URL 中提取网页内容,比如获取github内容，已初步解析
    Tavily是一家专注于AI搜索的公司，他们的搜索会为了LLM进行优化，以便于LLM进行数据检索。
    :param urls:url or list of urls,要提取信息的 URL 或者列表
    :param api_key:Tavily API 的访问密钥，不需要指定
    :return:
    """
    url = "https://api.tavily.com/extract"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    payload = {
        "urls": urls,
        "include_images": False,
        "extract_depth": "basic"  # basic, advanced
    }
    async with httpx.AsyncClient(timeout=Config.HTTP_TIMEOUT_SEC, proxy=Config.HTTP_Proxy) as cx:
        response = await cx.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            data = response.json()
            # print(json.dumps(data, indent=4))  # 打印响应数据
            return data['results']
        else:
            print(f"Error: {response.status_code},{response.text}")
            return [{'error': response.text}]


async def search_by_api(query: str, location: str = None,
                        engine: Literal['google', 'bing', "baidu", 'naver', "yahoo", "youtube",
                        "google_videos", "google_news", "google_images", "amazon_search", "shein_search"] | None = 'google',
                        api_key=Config.SearchApi_Key):
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
            print(f"Error: {response.status_code}")
            return [{'error': response.text}]


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
    async with httpx.AsyncClient(timeout=Config.HTTP_TIMEOUT_SEC, proxy=Config.HTTP_Proxy) as cx:
        response = await cx.get(url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            # print(json.dumps(data, indent=4))  # 打印响应数据
            return data.get('web', {}).get('results', [])
        else:
            print(f"Error: {response.status_code}")
            return [{'error': response.text}]


async def firecrawl_search(query: str, api_key=Config.Firecrawl_Service_Key):
    """
    使用自然语言搜索已爬网数据。
    "title","description","url","markdown","metadata"
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
    async with httpx.AsyncClient(timeout=Config.HTTP_TIMEOUT_SEC, proxy=Config.HTTP_Proxy) as cx:
        response = await cx.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            data = response.json()
            # print(json.dumps(data, indent=4))  # 打印响应数据
            return data.get('data', [])
        else:
            print(f"Error: {response.status_code},{response.text}")
            return [{'error': response.text}]


async def firecrawl_scrape(url, api_key=Config.Firecrawl_Service_Key):
    """
    用于抓取单个 URL。返回带有 URL 内容的 markdown。
    https://docs.firecrawl.dev/features/scrape
    """
    api_url = 'https://api.firecrawl.dev/v1/scrape'  # /crawl
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    payload = {
        "url": url,
        "formats": ["markdown"],  # 'html',"json"
        "onlyMainContent": True
    }
    async with httpx.AsyncClient(timeout=Config.HTTP_TIMEOUT_SEC, proxy=Config.HTTP_Proxy) as cx:
        response = await cx.post(api_url, headers=headers, json=payload)
        if response.status_code == 200:
            data = response.json()
            # print(json.dumps(data, indent=4))  # 打印响应数据
            return data.get('data', {}).get("metadata")
        else:
            print(f"Error: {response.status_code},{response.text}")
            return {'error': response.text}


async def web_extract_firecrawl(prompt: str, urls: list[str], api_key=Config.Firecrawl_Service_Key):
    """
    提取,使用 AI 从单个页面、多个页面或整个网站中提取结构化数据。
    /extract 端点简化了从任意数量的 URL 或整个域收集结构化数据的过程。提供 URL 列表，可选使用通配符（例如 example.com/*）以及描述所需信息的提示或架构。Firecrawl 处理抓取、解析和整理大型或小型数据集的细节。
    https://docs.firecrawl.dev/features/extract
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
    async with httpx.AsyncClient(timeout=Config.HTTP_TIMEOUT_SEC, proxy=Config.HTTP_Proxy) as cx:
        response = await cx.post(api_url, headers=headers, json=payload)
        if response.status_code == 200:
            data = response.json()
            # print(json.dumps(data, indent=4))  # 打印响应数据
            return data.get('data', {})
        else:
            print(f"Error: {response.status_code},{response.text}")
            return {'error': response.text}


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
    async with httpx.AsyncClient(timeout=Config.HTTP_TIMEOUT_SEC, proxy=Config.HTTP_Proxy) as cx:
        response = await cx.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            try:
                data = response.json()
                # print(json.dumps(data, indent=4))  # 打印响应数据
                return data.get("organic") or data.get(engine)
            except json.JSONDecodeError as e:
                return {'text': response.text}
        else:
            print(f"Error: {response.status_code},{response.text}")
            return [{'error': response.text}]
    # response = requests.request("POST", url, headers=headers, data=json.dumps(payload))


async def duckduckgo_search(query):
    url = "https://api.duckduckgo.com/"
    params = {
        'q': query,  # 搜索查询
        'format': 'json',  # 返回 JSON 格式
        'no_redirect': 1,  # 防止重定向
        'no_html': 1,  # 去除 HTML
        'skip_disambig': 1,  # 跳过歧义提示
    }

    async with httpx.AsyncClient(timeout=Config.HTTP_TIMEOUT_SEC, proxy=Config.HTTP_Proxy) as cx:
        response = await cx.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            # print(json.dumps(data, indent=4))  # 打印响应数据
            return data
        else:
            print(f"Error: {response.status_code}")
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


def wikipedia_search(query):
    response = requests.get(f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&format=json",
                            timeout=Config.HTTP_TIMEOUT_SEC, proxies=Config.HTTP_Proxies)
    search_results = response.json().get('query', {}).get('search', [])
    if search_results:
        page_id = search_results[0]['pageid']
        page_response = requests.get(
            f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&pageids={page_id}&format=json",
            timeout=Config.HTTP_TIMEOUT_SEC, proxies=Config.HTTP_Proxies)
        page_data = page_response.json()['query']['pages'][str(page_id)]
        return page_data.get('extract', 'No extract found.')
    return "No information found."


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
async def search_bmap_location(query, region='', limit=True):
    url = "http://api.map.baidu.com/place/v2/suggestion"  # 100
    params = {
        "query": query,
        "region": region,
        "city_limit": 'true' if (region and limit) else 'false',
        "output": "json",
        "ak": Config.BMAP_API_Key,
    }

    async with httpx.AsyncClient(timeout=Config.HTTP_TIMEOUT_SEC) as client:
        response = await client.get(url, params=params)
        res = []
        if response.status_code == 200:
            js = response.json()
            for result in js.get('result', []):
                res.append({'lng_lat': (round(result['location']['lng'], 6), round(result['location']['lat'], 6)),
                            'name': result["name"], 'address': result['address']})
        else:
            print(response.text)
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
        print(response.text)

    return None, None


# https://lbs.amap.com/api/webservice/guide/api-advanced/search
async def search_amap_location(query, region='', limit=True):
    url = "https://restapi.amap.com/v5/place/text?parameters"  # 100
    params = {
        "keywords": query,
        "region": region,
        "city_limit": 'true' if (region and limit) else 'false',
        "output": "json",
        "key": Config.AMAP_API_Key,
    }

    async with httpx.AsyncClient(timeout=Config.HTTP_TIMEOUT_SEC) as client:
        response = await client.get(url, params=params)
        res = []
        if response.status_code == 200:
            js = response.json()
            if js['status'] == '1' and int(js['count']) > 0:
                for result in js.get('pois', []):
                    s1, s2 = result['location'].split(',')
                    res.append({'lng_lat': (float(s1), float(s2)),
                                'name': result["name"], 'address': result['address']})
            else:
                print(response.text)  # {"count":"0","infocode":"10000","pois":[],"status":"1","info":"OK"}
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

    async with httpx.AsyncClient(timeout=Config.HTTP_TIMEOUT_SEC) as cx:
        response = await cx.get(url, params=params)

    data = response.json()
    if "trans_result" in data:
        return data["trans_result"][0]["dst"]

    print(response.text)
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
    async with httpx.AsyncClient(timeout=Config.HTTP_TIMEOUT_SEC) as cx:
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

    async with httpx.AsyncClient(timeout=Config.HTTP_TIMEOUT_SEC) as client:
        response = await client.post(url, headers=headers, json=payload)

    # 检查响应状态码和内容
    if response.status_code != 200:
        print(f"Error: Received status code {response.status_code},Response content: {response.text}")
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
    async with httpx.AsyncClient(timeout=Config.HTTP_TIMEOUT_SEC) as client:
        response = await client.post(url, json=request_data, headers=headers)
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


if __name__ == "__main__":
    from utils import print_functions

    print_functions('agents.ai_search')


    async def main():
        # print(await web_search_tavily('季度业绩报告'))
        # r = await web_extract_tavily('https://en.wikipedia.org/wiki/Artificial_intelligence')
        # print(r)

        r = await  brave_search('季度业绩报告')  # r.keys(),
        print(r)


    asyncio.run(main())
