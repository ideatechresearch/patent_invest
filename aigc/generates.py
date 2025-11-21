from openai import AsyncOpenAI, BadRequestError

from utils import *
from service import *
from secure import *
from agents import *
from database import BaseReBot

Function_Registry_Global: Optional[FunctionManager] = None


def get_func_manager() -> FunctionManager:
    global Function_Registry_Global
    if Function_Registry_Global is None:
        Function_Registry_Global = FunctionManager(keywords_registry=default_keywords_registry(), copy=True,
                                                   mcp_server=mcp)
    if not Function_Registry_Global.keywords_registry:
        Function_Registry_Global.update_registry(default_keywords_registry())
    return Function_Registry_Global


def default_keywords_registry():
    # 初始化 keywords_registry 的工厂函数
    return {
        'no_func': lambda *args, **kwargs: f"[❌] Function not loaded",
        'map_search': search_amap_location,
        'web_search': web_search_async,
        'intent_search': web_search_intent,
        'tavily_search': web_search_tavily,
        'tavily_extract': web_extract_tavily,
        'jina_extract': web_extract_jina,
        'news_search': named_partial('news_search', web_search_tavily, topic='news', time_range='w'),
        'baidu_search': named_partial('baidu_search', search_by_api, engine='baidu'),
        'wiki_search': wikipedia_search,
        'arxiv_search': arxiv_search,
        'translate': auto_translate,
        "execute_code": execute_code_results,
        'auto_calls': ai_auto_calls,  # 名称不同，需指定调用，防止重复 ai_auto_calls
        'baidu_nlp': baidu_nlp,
        'ai_chat': ai_client_completions,

        'visitor_records': ideatech_visitor_records,
    }


def agent_func_calls(agent: str = 'default', messages: list = None,
                     tools: list[dict] = None, keywords: list[str | tuple[str, ...]] = None, prompt: str = None,
                     model: str = Config.DEFAULT_MODEL_FUNCTION, **kwargs) -> List[Callable[..., Any]]:
    from script.knowledge import ideatech_knowledge
    """
    根据 agent 类型和传入的 tools 配置，返回对应的函数调用列表
    :param agent: 代理编号
    :param messages: 历史对话消息
    :param tools: 工具配置列表，每个元素为 dict，包含 'type' 和对应参数
    :param keywords: 输入关键词列表，用于组合调用参数
    :param model: 模型信息，可用于部分工具函数
    :param prompt: 用户请求内容 prompt，用于某些 agent, 单独文本作为 fallback keyword
    :return: 函数列表，每个函数为 partial 结构，可直接调用
    """
    items_to_process = []
    if keywords:
        if isinstance(keywords, list):
            items_to_process = [item for item in keywords if isinstance(item, str)]
    else:
        if prompt:
            items_to_process = [prompt]  # ','.join(keywords)

    callable_map_agent = {
        'default': [lambda *args, **kwargs: [], ],  # 默认返回一个空函数列表
        '2': [named_partial('baidu_search', search_by_api, engine='baidu'), web_search_async],
        '9': [named_partial('auto_translate_baidu', auto_translate, model_name='baidu')],
        '29': [ideatech_knowledge],
        '31': [named_partial('auto_calls_any', ai_auto_calls, user_messages=messages,
                             system_prompt=System_content.get('31'), model_name=model, get_messages=False)],
        '32': [ai_auto_calls],
        '37': [web_search_async]
        # 扩展更多的 agent 类型，映射到多个函数
    }

    func_calls: list[Callable[..., Any]] = callable_map_agent.get(agent, [])
    # 如果有关键词且 func_calls 全部非空 lambda，添加一个默认函数（如意图识别）append auto func to run. ai_auto_calls
    if keywords and all(not is_empty_lambda(_func) for _func in func_calls):  # and _func()
        func_calls.append(web_search_intent)

    tool_calls: list[Callable[..., Any]] = []  # callable_list
    # 对每个关键词，绑定代理函数,多个 tool_calls to process items,kwargs func参数,agent控制
    for _func in filter(callable, func_calls):
        if is_empty_lambda(_func):
            continue
        tool_calls.extend([partial(_func, item, **kwargs) for item in items_to_process])
        # print(bound_func.func.__name__, bound_func.args, bound_func.keywords,bound_func.arguments)

    # 解析 tools 并绑定, 函数绑定特定的配置参数,无参数可调用函数
    if tools:
        # retrieval、web_search、function, https://docs.bigmodel.cn/cn/guide/tools/web-search
        registry = default_keywords_registry()  # tool_calls.extend(convert_to_callable_list())
        tools_params: list = get_tools_params(tools)
        for tool_name, params in tools_params:
            if tool_name not in registry:
                continue
            _func = registry[tool_name]
            if params and isinstance(params, dict):
                tool_calls.append(partial(_func, **params))  # 有参数的情况，直接 partial
            else:
                tool_calls.extend([partial(_func, item) for item in items_to_process])  # 无参数的情况，遍历每个 item

    return tool_calls


async def ideatech_visitor_records(prompt: str, customer_name: str | list | tuple, **kwargs):
    print(prompt, customer_name, kwargs)
    match = field_match('客户名称', customer_name)
    payload = dict(querys=[prompt], collection_name='拜访记录', client=QD_Client,
                   payload_key='record_text', match=match, not_match=[],
                   topn=20, score_threshold=0.5, exact=False,
                   embeddings_calls=ai_embeddings, model_name='BAAI/bge-large-zh-v1.5')
    payload.update(kwargs)
    return await search_by_embeddings(**payload)


async def ai_tools_results(tool_messages, func_manager: FunctionManager) -> list[dict]:
    """
    解析模型响应，动态调用工具并生成后续消息列表。

    :param tool_messages: 模型响应的消息对象
    :param func_manager:
    :return: 包含原始响应和工具调用结果的消息列表
    """

    messages = [tool_messages.to_dict()]  # ChatCompletionMessage,ChatMessage
    tool_calls = tool_messages.tool_calls
    if not tool_calls:
        results = execute_code_results(tool_messages.content)
        # print(results)
        for res in results:
            messages.append({
                'role': 'tool',
                'content': res['error'] if res['error'] else res['output'],
                'tool_call_id': 'output.getvalue',
                'name': 'exec'
            })
        return messages

    for tool in tool_calls:
        func_name = tool.function.name  # function_name
        if not func_name:
            print(f'Error in tool_message:{tool.to_dict()}')
            continue

        func_args = tool.function.arguments  # function_args = json.loads(tool_call.function.arguments)
        # func_reg = global_function_registry(func_name)  # 从注册表中获取函数 tools_map[function_name](**function_args)
        # print(func_args)
        func_out = await func_manager.run_function(func_name, func_args, tool.id)
        messages.append(func_out)

    return messages  # [*tool_mmessages,]


async def ai_auto_calls(question, user_messages: list = None, system_prompt: str = None,
                        model_name=Config.DEFAULT_MODEL_FUNCTION,
                        get_messages=False, **kwargs) -> list:
    """
    call_tool
    自动推理并调用工具，调用 AI 模型接口，使用提供的工具集和对话消息，返回模型的响应,返回调用结果或消息列表
    :param question: 从问题里面获取意图选择工具调用
    :param user_messages: 从对话上下文里面获取意图选择工具调用，两者选一项
    :param system_prompt:获取系统提示词
    :param model_name: 默认使用qwen-max
    :param get_messages: 是否组装成方法：结果列表
    :param kwargs:
    :return: 结果列表或者 模型messages回复+本地执行方法结果
    """
    # 构造对话上下文
    if not system_prompt:
        system_prompt = System_content.get('31')
    if user_messages and isinstance(user_messages, list):
        user_messages = user_messages[-10:].copy()
        if not any(msg.get('role') == 'system' for msg in user_messages):
            user_messages.insert(0, {"role": "system", "content": system_prompt})
    else:
        user_messages = create_analyze_messages(system_prompt, question)
    # 获取工具列表
    func_manager = get_func_manager()
    tools_metadata = AI_Tools + await func_manager.tools_metadata(registered=False)
    # 模型决定使用哪些工具
    # tools = [{"type": "web_search",
    #          "web_search": {
    #         "enable": True  # 启用网络搜索
    #         "search_query": "自定义搜索的关键词",
    #          "search_result": True,#默认为禁用,允许用户获取详细的网页搜索来源信息
    #     }}]
    completion, row_id = await ai_client_completions(user_messages, client=None, model=model_name, get_content=False,
                                                     tools=tools_metadata, top_p=0.95, temperature=0.01, **kwargs)
    tool_messages = completion.choices[0].message
    if not tool_messages:
        return []

    # 执行工具调用
    final_messages = await ai_tools_results(tool_messages, func_manager)  # 组合了模型回复后的结果
    tool_results = [{'function': msg['name'], 'result': msg['content']} for msg in final_messages if
                    msg['role'] == "tool"]

    await BaseReBot.async_save(
        data={"user_content": question, "reference": json.dumps(tool_results, ensure_ascii=False)},
        row_id=row_id, dbpool=DB_Client)
    if not get_messages:
        return [tuple(item.values()) for item in tool_results]  # [{func_name:func_result}

    result_messages = user_messages
    for msg in final_messages:
        if msg["role"] == "assistant" and "tool_calls" in msg:
            # 处理 tool_calls 为 JSON 格式
            msg["content"] = json.dumps(msg["tool_calls"], indent=2, ensure_ascii=False)
            result_messages.append(msg)
    # 构造消息列表，适合继续对话
    formatted_results = "\n".join(f"(function:{item['function']},result:{item['result']})" for item in tool_results)
    result_messages.append({
        "role": "system",  # "user"
        "content": f"question:{question},\nresults:\n{formatted_results}"
    })  # {"role": "system", "content": f"已调用函数 {func_name}，结果如下：{func_result}"}]
    return result_messages  # [-1]


@mcp.tool
async def generate_analyze(results: dict | list | str, ctx: MCPContext, system_prompt: str = None,
                           model: str = 'deepseek-reasoner', max_tokens: int = 4096, temperature: float = 0.2) -> str:
    """
    Perform intelligent analysis and summarization on provided results.

    Args:
        results: Input content (dict, list, or raw text)
        ctx: FastMCP context (for logging, sampling, etc.)
        system_prompt: Optional high-level system instruction
        model: Model name for reasoning
        max_tokens: Maximum output token length
        temperature: Sampling temperature (lower = more deterministic)
    Returns:
        str: The summarized or analyzed text output
    """
    # prompt = f"Please provide a concise summary of the following content:\n\n{results}"
    analyzed = await ai_analyze(results, system_prompt, desc='数据分析', model=model,
                                max_tokens=max_tokens, temperature=temperature, dbpool=DB_Client)

    await ctx.info(f"Analysis and Summary generation complete {analyzed}")
    response = await ctx.sample(analyzed, system_prompt=system_prompt, temperature=temperature, max_tokens=max_tokens)
    return response.text


async def ai_summary(long_text: str | list[str | dict],
                     extract_prompt: str = None, summary_prompt: str = None,
                     extract_model='qwen:qwen-plus', summary_model='qwen:qwen-plus',
                     encoding_model=Config.DEFAULT_MODEL_ENCODING,
                     extract_max_tokens=4096, summary_max_tokens=8182, segment_max_length=10000, dbpool=None):
    tokenizer = get_tokenizer(encoding_model)
    segments_chunk: list = []
    if isinstance(long_text, str):
        lang = lang_detect_to_trans(long_text)
        if lang == 'zh':
            # sentences = split_sentences_clean(long_text, h_symbols=True, h_tables=True)
            segments_chunk: list[str] = structure_aware_chunk(long_text, max_size=segment_max_length)
        else:
            tokens = tokenizer.encode(long_text)
            small_chunks, large_chunks = organize_segments(tokens,
                                                           small_chunk_size=max(175, int(segment_max_length * 0.34)),
                                                           large_chunk_size=segment_max_length,
                                                           overlap=max(20, int(segment_max_length * 0.04)))
            segments_chunk = [tokenizer.decode(chunk) for chunk in large_chunks if chunk]

    elif isinstance(long_text, list):
        if all(isinstance(x, str) for x in long_text):
            segments_chunk: list[list[str]] = organize_segments_chunk(long_text, chunk_size=7, overlap_size=2,
                                                                      max_length=segment_max_length,
                                                                      tokenizer=tokenizer)
        if all(isinstance(x, dict) for x in long_text):
            segments_chunk: list[list[dict]] = list(
                df2doc_split(long_text, max_tokens=segment_max_length, tokenizer=tokenizer))

    if not segments_chunk:
        raise ValueError("long_text 必须是 str 或 list[str|dict]")

    if not summary_prompt:
        summary_prompt = '你是总结助手，请根据以下多个信息片段进行归纳总结，提炼主旨、关键事件或共同规律。如片段内容存在重复或上下文重叠，可适当整合。'
    else:
        if not extract_prompt:
            extract_prompt = '你是信息摘要助手，请提炼以下关键信息:\n' + summary_prompt

    @run_generator(max_concurrent=Config.MAX_CONCURRENT, return_exceptions=True)
    async def extract_item(segment: list | str):
        if not segment:
            raise ValueError("bad input")
        content = format_content_str(segment, exclude_null=True)

        if lang_token_size(content, tokenizer=tokenizer) > extract_max_tokens:
            messages = [{"role": "system",
                         "content": extract_prompt or '你是信息摘要助手，请提炼以下文本中的关键信息、数据或事件。'},
                        {'role': 'user', 'content': content}]
            content, _ = await ai_client_completions(messages, client=None, model=extract_model, get_content=True,
                                                     max_tokens=extract_max_tokens, temperature=0.3, dbpool=dbpool)
        print(segment, '>', content)
        return content

    raw_results = []
    async for idx, extract in extract_item(inputs=segments_chunk):
        if isinstance(extract, Exception):
            yield {'seq': idx, 'content': None, 'type': 'extract', 'error': str(extract), 'item': segments_chunk[idx]}
            continue
        res = {'seq': idx, 'content': extract, 'type': 'extract'}
        yield res
        raw_results.append(res)
    results = sorted(raw_results, key=lambda x: x['seq'], reverse=True)
    summary = await ai_analyze(results, system_prompt=summary_prompt, desc='信息片段', model=summary_model,
                               max_tokens=summary_max_tokens, dbpool=dbpool)
    yield {'seq': -1, 'content': summary, 'type': 'summary'}  # 'extract': results,


# Callable[[参数类型], 返回类型]
async def get_chat_payload(messages: list[dict] = None, user_request: str = '', system: str = '',
                           temperature: float = 0.4, top_p: float = 0.8, max_tokens: int = 1024,
                           model_name=Config.DEFAULT_MODEL, model_id=0,
                           agent: str = None, tools: List[dict] = None,
                           keywords: List[Union[str, Tuple[str, ...]]] = None, images: List[str] = None,
                           thinking: int = 0, **kwargs):
    model_info, name = find_ai_model(model_name, model_id, 'model')
    model_type = model_info['type']
    if messages is None:
        messages = []

    if isinstance(messages, list) and messages:
        if model_type in ('baidu', 'tencent'):
            # system_message = next((msg for msg in messages if msg["role"] == "system"), None)
            if messages[0].get('role') == 'system':  # ['system,assistant,user,tool,function']
                system = messages[0].get('content')
                del messages[0]

            # the role of first message must be user
            if messages[0].get('role') != 'user':  # user（tool）
                messages.insert(0, {'role': 'user', 'content': user_request or '请问您有什么问题？'})

            # 确保 user 和 assistant 消息交替出现
            messages = alternate_chat_history(messages)

        for message in messages:
            if 'name' in message:
                if (model_info['name'] == 'mistral' or
                        not isinstance(message["name"], str) or
                        not message["name"].strip()):
                    del message['name']

        if model_type != 'baidu' and system:
            if not any(msg.get('role') == 'system' for msg in messages):
                messages.insert(0, {"role": "system", "content": system})
            # messages[0].get('role') != 'system':
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
        if model_type != 'baidu' and system:  # system_message
            messages = [{"role": "system", "content": system}]
        messages.append({'role': 'user', 'content': user_request})

    tool_callable: List[Callable[[...], Any]] = agent_func_calls(agent, messages, tools, keywords, user_request)

    func_manager = get_func_manager()
    refer = await func_manager.retrieved_reference(keywords, tool_callable)
    if refer:
        # """Answer the users question using only the provided information below:{docs}""".format(docs=formatted_refer)
        if model_type != 'baidu':
            formatted_refer = [{"type": "text", "text": str(text)} for text in refer]
            messages.append({"role": "system", "content": formatted_refer})
        else:
            formatted_refer = '\n'.join(map(str, refer))  # 百度模型对结构化 role 不敏感
            messages[-1]['content'] = (f'以下是相关检索结果或函数调用信息:\n{formatted_refer}\n'
                                       f'请结合以上参考内容或根据上下文，针对下面的问题进行解答：\n{user_request}')

    if images:  # 图片内容理解,(str, list[dict[str, str | Any]]))
        image_data = [{"type": "image_url", "image_url": {"url": image}} for image in images]
        if isinstance(messages[-1]['content'], list):  # text-prompt 请详细描述一下这几张图片。这是哪里？
            messages[-1]['content'].extend(image_data)
        else:
            messages[-1]['content'] = [{"type": "text", "text": user_request}] + image_data

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
    )
    payload.update(kwargs)

    if tools:
        filter_tools: list[dict] = deduplicate_functions_by_name(tools)
        if any(filter_tools):  # 一旦你传了 tools 字段，它 必须 至少包含一个合法的 tool
            payload['tools'] = filter_tools

    if model_type == 'baidu':
        payload['system'] = system

    if name in ('o3-mini', 'o4-mini', 'openai/o3', "openai/o1", 'openai/o1-mini', 'openai/o1-pro',
                'openai/o3-mini', 'openai/o3-pro', 'openai/o4-mini'):
        payload['max_completion_tokens'] = payload.pop('max_tokens', max_tokens)
        payload.pop('top_p', None)  # https://www.cnblogs.com/xiaoxi666/p/18827733
    if name in ('gpt-5', 'gpt-5-mini', 'gpt-5-nano',
                'openai/gpt-5', 'openai/gpt-5-chat', 'openai/gpt-5-mini', 'openai/gpt-5-nano'):
        for param in ['top_p', 'presence_penalty', 'frequency_penalty']:
            payload.pop(param, None)

    extra_body = payload.get("extra_body")
    if not isinstance(extra_body, dict):
        extra_body = {}
    enable_thinking = extra_body.get('enable_thinking', thinking > 0)
    thinking_budget = max(0, thinking) or extra_body.get('thinking_budget', 0)
    # https://help.aliyun.com/zh/model-studio/deep-thinking?spm=a2c4g.11186623.0.0.31a44823XJPxi9#e7c0002fe4meu
    if name in ('qwen3-32b', "qwen3-14b", "qwen3-8b", "qwen3-235b-a22b", "qwq-32b", "qwen-turbo", "qwen-plus",
                "qwen-plus-2025-04-28", "deepseek-v3.1",
                "Qwen/Qwen3-8B", "Qwen/Qwen3-14B", "Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-32B", "Qwen/Qwen3-235B-A22B",
                "tencent/Hunyuan-A13B-Instruct"
                "deepseek-ai/DeepSeek-V3.1", "deepseek-ai/DeepSeek-V3.1-Terminus"):
        # 开启深度思考,参数开启思考过程，该参数对 qwen3-30b-a3b-thinking-2507、qwen3-235b-a22b-thinking-2507、QwQ 模型无效
        extra_body["enable_thinking"] = enable_thinking
        if thinking_budget > 0:
            extra_body["thinking_budget"] = thinking_budget
    if name in ('deepseek-v3-1-terminus', "deepseek-v3-1-250821", "doubao-seed-1-6-250615",
                "doubao-seed-1-6-flash-250715"):
        # enable_thinking = extra_body.get('thinking', {}).get("type") == "enabled"
        if not "thinking" in extra_body:
            extra_body["thinking"] = {"type": "enabled" if enable_thinking else "disabled"}  # /"auto"
        if 'max_completion_tokens' in payload:
            payload.pop('max_tokens', None)  # 不可与 max_tokens 字段同时设置，会直接报错
        else:
            payload['max_completion_tokens'] = payload.pop('max_tokens', max_tokens) + thinking_budget  # 最大输出长度
    if name in ("deepseek-reason",):
        extra_body["thinking_budget"] = thinking_budget or 2048
    if name in ("qwen-mt-plus", "qwen-mt-turbo"):
        extra_body["translation_options"] = {"source_lang": "auto", "target_lang": "English"}

    if extra_body:
        if isinstance(payload.get("extra_body"), dict):
            payload["extra_body"].update(extra_body)
        else:
            payload["extra_body"] = extra_body

    # payload['response_format'] = {"type": "json_object"}

    # 1. 系统设定（system）
    # 2. 原始提问（user）
    # 3. 检索结果 or 函数调用信息（system / user）
    # 4. 最终 assistant 回复

    return model_info, payload, refer


def get_chat_payload_post(model_info: Dict[str, Any], payload: dict):
    # 通过 requests 库直接发起 HTTP POST 请求
    url = model_info['url'] if model_info.get('url') else model_info['base_url'] + '/chat/completions'
    api_key = model_info['api_key']
    headers = {'Content-Type': 'application/json', }
    payload = payload.copy()
    if api_key:
        if isinstance(api_key, list):
            idx = next((i for i, m in enumerate(model_info['model']) if m == payload["model"]), 0)
            # model_info['model'].index(payload["model"])
            if api_key[idx]:
                headers["Authorization"] = f'Bearer {api_key[idx]}'
        else:
            headers["Authorization"] = f'Bearer {api_key}'

    if model_info['type'] == 'baidu':  # 'ernie'
        url = build_url(f'{url}{payload["model"]}',
                        get_baidu_access_token(Config.BAIDU_qianfan_API_Key,
                                               Config.BAIDU_qianfan_Secret_Key))  # ?access_token=" + get_access_token()
        # print(url)
        # payload.pop('model', None)
        payload['max_output_tokens'] = payload.pop('max_tokens', 1024)
        # payload['enable_system_memory'] = False
        # payload["enable_citation"]= False
        # payload["disable_search"] = False
        # payload["user_id"]=
        # payload["user_ip"]=
        # payload['system'] = system
    if model_info['type'] == 'tencent':
        service = 'hunyuan'
        host = url.split("//")[-1]
        payload = convert_keys_to_pascal_case(payload)
        payload.pop('MaxTokens', None)
        headers = get_tencent_signature(service, host, payload, action='ChatCompletions',
                                        secret_id=Config.TENCENT_SecretId,
                                        secret_key=Config.TENCENT_Secret_Key)
        headers['X-TC-Version'] = '2023-09-01'
        # headers["X-TC-Region"] = 'ap-shanghai'

    if model_info['type'] == 'anthropic':
        headers["x-api-key"] = api_key
        headers.pop("Authorization", None)
        headers["anthropic-version"] = '2023-06-01'

    # if model_info['name'] == 'silicon':
    #     headers = {
    #         "accept": "application/json",
    #         "content-type": "application/json",
    #         "authorization": "Bearer sk-tokens"
    #     }
    return url, headers, payload


async def get_generate_payload(prompt: str, user_request: str = '', suffix: str = None, stream=False,
                               temperature: float = 0.7, top_p: float = 1, max_tokens: int = 4096,
                               model_name=Config.DEFAULT_MODEL, model_id=0, agent: str = None,
                               keywords: List[Union[str, Tuple[str, ...]]] = None,
                               **kwargs):
    chat = False
    try:
        model_info, name = find_ai_model(model_name, model_id, search_field="generation")
    except:
        model_info, name = find_ai_model(model_name, model_id)
        chat = True

    if chat or not name:
        model_info, payload, refer = await get_chat_payload(messages=None, user_request=user_request, system=prompt,
                                                            temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                                                            model_name=model_name, model_id=model_id,
                                                            agent=agent, tools=None, keywords=keywords, thinking=0,
                                                            **kwargs)

    else:
        tool_callable: List[Callable[[...], Any]] = agent_func_calls(agent, keywords=keywords, prompt=user_request)
        func_manager = get_func_manager()
        # 检索参考资料
        refer = await func_manager.retrieved_reference(keywords, tool_callable)
        # 如果检索到内容，就把它拼到 user_request 里
        if refer:
            formatted_refer = "\n".join(map(str, refer))
            user_request = f"参考材料:\n{formatted_refer}\n材料仅供参考,请回答下面的问题: {user_request}"

        if user_request:
            prompt += '\n\n' + user_request

        payload = dict(model=name,
                       prompt=prompt,
                       suffix=suffix,
                       max_tokens=max_tokens,
                       temperature=temperature,
                       top_p=top_p,
                       stream=stream,
                       stop=kwargs.get("stop", None),
                       )

    return model_info, payload, refer


# 生成:conversation or summary,Fill-In-the-Middle
async def ai_generate(model_info: Optional[Dict[str, Any]], payload: dict = None, get_content=True,
                      **kwargs):
    '''
    Completions足以解决几乎任何语言处理任务，包括内容生成、摘要、语义搜索、主题标记、情感分析等等。
    需要注意的一点限制是，对于大多数模型，单个API请求只能在提示和完成之间处理最多4096个标记。
    '''
    if not payload:
        model_info, payload, _ = await get_generate_payload(**kwargs)
    else:
        payload.update(kwargs)  # 修改更新 payload

    stream = payload.get("stream", False)
    if 'messages' in payload:
        if stream:
            async def stream_data():
                async for content, data in ai_chat_stream(model_info, payload):
                    yield content if get_content else data

            return stream_data()
        else:
            content, completion = await ai_chat(model_info, payload)
            return content if get_content else completion

    if model_info['name'] == 'qwen' and not stream:
        response = dashscope.Generation.call(model=payload['model'], prompt=payload['prompt'])
        return response.output.text if get_content else response.model_dump()

    client = AI_Client.get(model_info['name'], None)
    if not client:
        raise ValueError(f"Client for model {model_info['name']} not found")

    # <|fim_prefix|>{prefix_content}<|fim_suffix|>
    # <|fim_prefix|>{prefix_content}<|fim_suffix|>{suffix_content}<|fim_middle|>
    response = await client.completions.create(**payload)

    if stream:
        async def stream_data():
            async for chunk in response:
                yield chunk.choices[0].text if get_content else chunk.model_dump_json()

        return stream_data()  # async generator 对象

    return response.choices[0].text.strip() if get_content else response.model_dump()


async def ai_chat(model_info: Optional[Dict[str, Any]], payload: dict = None, **kwargs) -> Tuple[str, Dict]:
    """
    模拟发送请求给AI模型并接收响应。
    :param model_info: 模型信息（如名称、ID、配置等）
    :param payload: 请求的负载，通常是对话或文本输入
    :param kwargs: 其他额外参数
    :return: 返回模型的响应
    """
    if not payload:
        model_info, payload, _ = await get_chat_payload(**kwargs)
    elif not model_info:
        model_info, new_payload, _ = await get_chat_payload(**kwargs)
        payload = {**(payload or {}), **new_payload}  # 合并： {**payload, **kwargs}
    else:
        payload.update(kwargs)  # 修改更新 payload

    fake_response = {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": payload.get('model', 'unknown'),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": 'null',
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }
    client = AI_Client.get(model_info['name'], None)
    if client:
        try:
            # await asyncio.to_thread(client.chat.completions.create, **payload)
            # await asyncio.wait_for(client.chat.completions.create, **payload)
            completion = await client.chat.completions.create(**payload)
            if completion is None:
                raise ValueError("OpenAI API returned None instead of a valid response")
            if not hasattr(completion, "choices") or not completion.choices:
                raise ValueError(f"Unexpected API response, missing choices: {completion}")
            first_choice = completion.choices[0]
            if not hasattr(first_choice, "message") or not hasattr(first_choice.message, "content"):
                raise ValueError(f"Unexpected API response, missing message: {completion}")
            return first_choice.message.content, completion.model_dump()  # 自动序列化为 JSON
            # content = getattr(first_choice.message, "content", None)
            # json.loads(completion.model_dump_json())

        except BadRequestError as e:
            try:
                return await ai_try(client, payload, e)
            except:
                raise
        except Exception as e:
            logging.error(f"OpenAI error:{e}, {payload}")
            error_message = f"OpenAI error occurred: {e}"
            fake_response["choices"][0]["message"]["content"] = error_message
            return error_message, fake_response

    url, headers, payload = get_chat_payload_post(model_info, payload)
    # print(headers, payload)
    parse_rules = {
        'baidu': lambda d: d.get('result'),
        'tencent': lambda d: d.get('Response', {}).get('Choices', [{}])[0].get('Message', {}).get('Content'),
        # d.get('Choices')[0].get('Message').get('Content')
        'default': lambda d: d.get('choices', [{}])[0].get('message', {}).get('content')
    }
    cx = get_httpx_client(proxy=Config.HTTP_Proxy if model_info.get('proxy') else None)
    try:
        response = await cx.post(url, headers=headers, json=payload, timeout=Config.LLM_TIMEOUT_SEC)
        response.raise_for_status()  # 如果请求失败，则抛出异常
        data = response.json()
        result = parse_rules.get(model_info['type'], parse_rules['default'])(data)
        if not result:
            print(response.text)

        if model_info['type'] == 'tencent':
            return result, convert_keys_to_lower_case(data)

        if model_info['type'] == 'baidu':
            content = data.pop('result', data.get("error_msg", ""))
            data.update({
                "model": payload.get('model', 'unknown'),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content, },
                        "finish_reason": "stop",
                    }
                ],
            })
            return content, data

        return result, data

    except Exception as e:
        # print(response.text)
        error_message = f"HTTP error occurred: {e}"
        fake_response["choices"][0]["message"]["content"] = error_message
        return error_message, fake_response


async def ai_chat_stream(model_info: Optional[Dict[str, Any]], payload=None, **kwargs):
    if not payload:
        model_info, payload, _ = await get_chat_payload(**kwargs)
    else:
        # payload=copy(payload)
        payload.update(kwargs)

    client = AI_Client.get(model_info['name'], None)
    if client:
        async for item in stream_chat_completion(client, payload):
            yield item

        # with client.chat.completions.with_streaming_response.create(**payload) as response:
        #     print(response.headers.get("X-My-Header"))
        #     for line in response.iter_lines():
        #         yield line

        return  # 异步生成器的结束无需返回值

    fake_response = {
        "id": f"chatcmpl-{int(time.time())}", "object": "chat.completion.chunk",
        "created": int(time.time()), "model": payload.get("model", "unknown"),
        "choices": [{"index": 0, "delta": {"content": 'null'}, "finish_reason": "stop"}],
    }
    url, headers, payload = get_chat_payload_post(model_info, payload)
    payload["stream"] = True
    cx = get_httpx_client(proxy=Config.HTTP_Proxy if model_info.get('proxy') else None)
    model_type = model_info['type']
    async for item in post_httpx_sse(url, payload, headers, Config.LLM_TIMEOUT_SEC, cx):
        data_type = item.get("type")
        if data_type == 'done':
            break

        data = item.get("data", None)
        if not data:
            continue

        if data_type != "data":
            # text / error 直接返回原始字符串 yield
            content = f"OpenAI error occurred: {data}" if data_type == "error" else data
            fake_response["choices"][0]["delta"]["content"] = content
            yield data, json.dumps(fake_response)
            print(item)
            continue

        if model_type == 'baidu':
            if data.get('is_end') is True:
                break
            content = data.get("result") or data.get("error_msg")
            fake_response["choices"][0]["delta"]["content"] = content
            yield content, json.dumps(fake_response)
        elif model_type == 'tencent':
            reason = data.get('Choices', [{}])[0].get('FinishReason')
            if reason == "stop":
                # raise StopIteration(2)
                break
            data = convert_keys_to_lower_case(data)

        choices = data.get('choices', [])  # 通用逻辑：处理 choices -> delta -> content
        if choices:
            delta = choices[0].get('delta', {})
            yield delta.get("content", ""), json.dumps(data)
            # print(delta.get("content", ""), end="", flush=True)

    # yield "[DONE]"


async def forward_stream(response):
    async for line in response.iter_lines(decode_unicode=True):
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


async def ai_parse(model_info: Optional[Dict[str, Any]], format_model: Optional[Dict | BaseModel] = None, payload=None,
                   **kwargs):
    if not payload:
        model_info, payload, _ = await get_chat_payload(**kwargs)
    else:
        # payload=copy(payload)
        payload.update(kwargs)

    client = AI_Client.get(model_info['name'], None)
    messages = create_analyze_messages("你是一位数学辅导老师。", "使用中文解题: 8x + 9 = 32 and x + y = 1")
    completion = await client.beta.chat.completions.parse(
        model="doubao-seed-1-6-250615",  # 替换为您需要使用的模型
        messages=messages,
        response_format=format_model or {"type": "json_object"},  # BaseModel:json_schema/json_object
    )
    resp = completion.choices[0].message.parsed
    return resp.model_dump()  # resp.model_dump_json(indent=2)


Assistant_Cache = {}


# https://platform.baichuan-ai.com/docs/assistants
async def ai_assistant_create(instructions: str, user_name: str,
                              tools_type: Literal['web_search', 'code_interpreter', 'function'] = "code_interpreter",
                              model_id=4):
    # 如果您判断是一次连续的对话，则无需自己拼接上下文，只需将最新的 message 添加到对应的 thread id 即可得到包含了 thread id 历史上下文的回复，历史上下文超过我们会帮您自动截断。
    global Assistant_Cache
    cache_key = generate_hash_key(instructions, user_name, tools_type, model_id)
    if cache_key in Assistant_Cache:
        return Assistant_Cache[cache_key]

    model_info, model_name = find_ai_model('baichuan', model_id, 'model')
    assistants_url = model_info.get('assistants_url', f"{model_info['base_url']}assistants")
    threads_url = f"{model_info['base_url']}threads"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {model_info['api_key']}"
    }
    payload = {
        "instructions": instructions,
        "name": user_name,
        "tools": [{"type": tools_type}],  # web_search,code_interpreter,function
        "model": model_name
    }
    cx = get_httpx_client(time_out=Config.HTTP_TIMEOUT_SEC)
    try:
        assistant_response = await cx.post(assistants_url, headers=headers, json=payload)
        assistant_response.raise_for_status()
        threads_response = await cx.post(threads_url, headers=headers, json={})
        threads_response.raise_for_status()
        results = {'headers': headers,
                   'assistant_data': assistant_response.json(),
                   'threads_url': threads_url,
                   'threads_id': threads_response.json()['id']}
        Assistant_Cache[cache_key] = results  # assistant_response.json()["id"],assistant_id
        return results
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


async def ai_assistant_run(user_request: str, instructions: str, user_name: str, tools_type="code_interpreter",
                           model_id=4, max_retries=20, interval=5, backoff_factor=1.5):
    assistant = await ai_assistant_create(instructions, user_name, tools_type, model_id)
    if "error" in assistant:
        return assistant

    assistant_data = assistant['assistant_data']
    headers: dict = assistant.get('headers', {})
    messages_url = f'{assistant["threads_url"]}/{assistant["threads_id"]}/messages'
    threads_url = f'{assistant["threads_url"]}/{assistant["threads_id"]}/runs'

    cx = get_httpx_client(time_out=Config.HTTP_TIMEOUT_SEC)
    messages_response = await cx.post(messages_url, headers=headers, json={"role": "user", "content": user_request})
    messages_response.raise_for_status()
    run_response = await cx.post(threads_url, headers=headers, json={"assistant_id": assistant_data['id']})
    run_response.raise_for_status()
    run_id = run_response.json().get('id')
    run_url = f'{threads_url}/{run_id}'
    retries = 0
    current_interval = interval
    while retries < max_retries:
        try:
            run_status_response = await cx.get(run_url, headers=headers)  # 定期检索 run id 的运行状态
            # print(retries,run_status_response.status_code, run_status_response.headers, run_status_response.text)
            if run_status_response.status_code == 429:
                await asyncio.sleep(current_interval)
                retries += 1
                current_interval *= backoff_factor
                continue

            run_status_response.raise_for_status()
            status_data = run_status_response.json()

            if status_data.get('status') == 'completed':
                messages_status_response = await cx.get(messages_url, headers=headers)
                run_status_response.raise_for_status()
                status_data = messages_status_response.json()
                # print(status_data)
                return status_data.get('data')

            await asyncio.sleep(current_interval)
            retries += 1

        except httpx.HTTPStatusError as http_error:
            return {"error": f"HTTP error: {str(http_error)}"}  # status_data['error']

        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}

    return {"error": "Max retries reached, task not completed"}


# （1）文本数量不超过 16。 （2）每个文本长度不超过 512 个 token，超出自动截断，token 统计信息，token 数 = 汉字数+单词数*1.3 （仅为估算逻辑，以实际返回为准)。
async def get_baidu_embeddings(texts: List[str], access_token: str, model_name='bge_large_zh') -> List:
    if not isinstance(texts, list):
        texts = [texts]

    url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/embeddings/{model_name}?access_token={access_token}"
    headers = {
        'Content-Type': 'application/json',
        # "Authorization: Bearer $BAICHUAN_API_KEY"
    }
    batch_size = 16
    embeddings = []
    async with httpx.AsyncClient(timeout=Config.HTTP_TIMEOUT_SEC) as cx:
        if len(texts) < batch_size:
            payload = json.dumps({"input": texts})  # "model":
            response = await cx.post(url, headers=headers, data=payload)
            data = response.json().get('data')
            # if len(texts) == 1:
            #     return data[0].get('embedding') if data else None
            embeddings = [d.get('embedding') for d in data]
        else:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                payload = {"input": batch}
                try:
                    response = await cx.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json().get('data')
                    if data and len(data) == len(batch):
                        embeddings += [d.get('embedding') for d in data]
                except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                    print(exc)

    return embeddings


def get_hf_embeddings(texts, model_name='BAAI/bge-large-zh-v1.5', access_token=Config.HF_Service_Key):
    # "https://api-inference.huggingface.co/models/BAAI/bge-reranker-large"
    url = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": f'Bearer {access_token}'}
    payload = {"inputs": texts}
    response = requests.post(url, headers=headers, json=payload, proxies=Config.HTTP_Proxies,
                             timeout=Config.HTTP_TIMEOUT_SEC)
    data = response.json().get('data')
    return [emb.get('embedding') for emb in data]


# https://ai.youdao.com/
# https://hcfy.ai/docs/services/youdao-api
async def auto_translate(text: str, model_name='baidu', source: str = 'auto', target: str = 'auto'):
    """
       自动翻译函数，根据输入的文本和指定的翻译模型自动完成语言检测和翻译。
        功能描述:
    1. 自动检测源语言，如果 `source` 为 "auto"：
        - 使用 `detect` 方法检测语言。
        - 如果检测结果是中文，标准化为 "zh"。
        - 如果检测结果不明确，默认根据内容是否包含中文设定语言。
    2. 自动设定目标语言，如果 `target` 为 "auto"：
        - 如果源语言是中文，则目标语言为英文 ("en")。
        - 如果源语言是其他语言，则目标语言为中文 ("zh")。
    3. 根据指定的 `model_name` 调用对应的翻译模型处理文本。
        - 如果未找到对应模型，则调用 `ai_generate` 生成翻译。
    4. 返回翻译结果及相关信息。
    """
    if source == 'auto':
        source = lang_detect_to_trans(text)
    if target == 'auto':
        target = 'en' if source == 'zh' else 'zh'

    translate_map: dict = {
        "baidu": baidu_translate,
        "tencent": tencent_translate,
        "xunfei": xunfei_translate
    }
    error = ''
    handler = translate_map.get(model_name)
    if handler:
        translated_text = await handler(text, source, target)
        if isinstance(translated_text, str):
            return {"translated": translated_text, 'from': source, 'to': target, "model": model_name}

        error = translated_text.get('error')

    system_prompt = System_content.get('9').format(source_language=source, target_language=target)

    # model_map = {"baidu": 'ernie', "tencent": "hunyuan", "xunfei": 'spark'}
    # if model_name in model_map.keys():
    #     model_name = model_map[model_name]
    if not model_name:
        model_name = 'qwen'

    translated_text = await ai_generate(
        model_info=None,
        prompt=system_prompt,
        user_request=text,
        model_name=model_name,
        stream=False,
    )
    if translated_text:
        return {"translated": translated_text, 'from': source, 'to': target, "model": model_name}

    return {"translated": error, 'from': source, 'to': target, "model": model_name}


async def siliconflow_generate_image(prompt: str = '', negative_prompt: str = '', model_name='siliconflow', model_id=0):
    model_info, name = find_ai_model(model_name, model_id, 'image')

    url = model_info.get('image_url') or "https://api.siliconflow.cn/v1/images/generations"
    payload = {
        "model": name,
        "prompt": prompt,
        "seed": random.randint(1, 9999999998),  # Required seed range:0 < x < 9999999999
        'prompt_enhancement': True,
    }
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt

    headers = {
        "Authorization": f"Bearer {Config.Silicon_Service_Key}",
        "Content-Type": "application/json"
    }
    cx = get_httpx_client()
    response = await cx.post(url, headers=headers, json=payload, timeout=Config.LLM_TIMEOUT_SEC)
    response.raise_for_status()
    if response.status_code != 200:
        return None, {'error': f'{response.status_code},Request failed,Response content: {response.text}'}

    response_data = response.json()
    return None, {"urls": [i['url'] for i in response_data["images"]], 'id': response_data["seed"]}


@mcp.tool
async def generate_code_example(concept: str, ctx: MCPContext) -> str:
    """Generate a Python code example for a given concept."""
    response = await ctx.sample(
        messages=f"Write a simple Python code example demonstrating '{concept}'.",
        system_prompt="You are an expert Python programmer. Provide concise, working code examples without explanations.",
        temperature=0.7,
        max_tokens=300
    )

    code_example = response.text
    return f"```python\n{code_example}\n```"


@mcp.tool
async def technical_analysis(data: str, ctx: MCPContext) -> str:
    """Perform technical analysis with a reasoning-focused model."""
    response = await ctx.sample(
        messages=f"Analyze this technical data and provide insights: {data}",
        model_preferences=["claude-3-opus", "gpt-4"],  # Prefer reasoning models
        temperature=0.2,  # Low randomness for consistency
        max_tokens=800
    )

    return response.text


async def baidu_nlp(text: str | list[str] = None, nlp_type='ecnet', **kwargs):  # text': text
    '''
    nlp_type:
        address:地址识别,kw:text
        sentiment_classify:情感倾向分析
        emotion:对话情绪识别
        entity_analysis:实体分析,kw:text,mention
        simnet:短文本相似度,kw: text_1,text_2,
        ecnet,text_correction:文本纠错,搜索引擎、语音识别、内容审查等功能
        txt_keywords_extraction:关键词提取,kw:text[],num
        txt_monet:文本信息提取,content_list[{content,query_lis[{query}]}]
        sentiment_classify:
        depparser:依存句法分析,利用句子中词与词之间的依存关系来表示词语的句法结构信息
        lexer:词法分析,提供分词、词性标注、专名识别三大功能
        keyword:文章标签,kw:title,content
        topic:文章分类,kw:title,content
        news_summary:新闻摘要,title,content,max_summary_len
        titlepredictor,文章标题生成,kw:doc
    '''
    # https://aip.baidubce.com/rpc/2.0/ernievilg/v1/txt2imgv2
    url = f'https://aip.baidubce.com/rpc/2.0/nlp/v1/{nlp_type}'
    access_token = get_baidu_access_token(Config.BAIDU_nlp_API_Key, Config.BAIDU_nlp_Secret_Key)
    url = build_url(url, access_token, charset='UTF-8')  # f"{url}&access_token={access_token}"
    headers = {
        'Content-Type': 'application/json',  # 'application/x-www-form-urlencoded'
        # 'host': 'aip.baidubce.com',
        # 'authorization': 'bce-auth',
        # 'x-bce-date': '2015-03-24T13: 02:00Z',
        # 'x-bce-request-id':'',
    }
    body = {**kwargs}
    if text:
        body['text'] = text

    cx = get_httpx_client(time_out=Config.HTTP_TIMEOUT_SEC)
    response = await cx.post(url, headers=headers, json=body)
    response.raise_for_status()
    data = response.json()  # .get('items')
    # 遍历键列表，返回第一个找到的键对应的值
    for key in ['items', 'item', 'results', 'results_list', 'error_msg']:
        if key in data:
            if key == 'error_msg':
                print(response.text)
            return data[key]
    return data


async def ali_nlp(text):
    # https://help.aliyun.com/zh/sdk/product-overview/v3-request-structure-and-signature?spm=a2c4g.11186623.0.0.38ee5703p01VbZ#section-mqj-l8f-ak0
    url = 'alinlp.cn-hangzhou.aliyuncs.com'
    token, _ = get_aliyun_access_token(service="alinlp", region="cn-hangzhou", access_key_id=Config.ALIYUN_AK_ID,
                                       access_key_secret=Config.ALIYUN_Secret_Key)
    if not token:
        print("No permission!")

    headers = {
        "Authorization": f"Bearer {Config.ALIYUN_AK_ID}",
        'Content-Type': "application/x-www-form-urlencoded",
        "X-NLS-Token": token,

    }
    params = {
        "appkey": Config.ALIYUN_nls_AppId,
        'ServiceCode': 'alinlp',
        'Action': 'GetWeChGeneral',
        'Text': text,
        'TokenizerId': 'GENERAL_CHN',
    }


async def download_file(url: str, dest_folder: Path = None, chunk_size=4096,
                        in_decode=False, in_session=False, retries=3, delay=3, time_out=Config.HTTP_TIMEOUT_SEC):
    """
    下载URL中的文件到目标文件夹
    :param url: 下载链接
    :param dest_folder: 保存文件的路径
    :param chunk_size: 每次读取的字节大小（默认4096字节）
    :param in_decode: 是否解码为字符串
    :param in_session: 是否使用 session（长连接优化）
    :param retries: 下载失败后的重试次数
    :param delay: 重试之间的等待时间（秒）
    :param time_out: 超时时间（秒）
    """
    # filename = url.split("/")[-1]  # 提取文件名
    file_name = unquote(url.split("/")[-1].split("?")[0])
    save_path = None
    # print(file_name)
    if dest_folder:
        save_path = dest_folder / file_name  # 提取文件名

    attempt = 0
    while attempt < retries:
        try:
            if in_session:  # aiohttp长连接优化，适合发送多个请求或需要更好的连接复用,维护多个请求之间的连接
                timeout = aiohttp.ClientTimeout(total=time_out) if time_out > 0 else None
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    return await download_by_aiohttp(url, session, save_path, chunk_size, in_decode), file_name
            else:  # httpx少量请求场景,适合简单的、单个请求场景
                timeout = httpx.Timeout(time_out) if time_out > 0 else None
                async with httpx.AsyncClient(timeout=timeout) as client:
                    return await download_by_httpx(url, client, save_path, chunk_size, in_decode), file_name

        except (httpx.RequestError, aiohttp.ClientError, asyncio.TimeoutError, requests.Timeout) as e:
            print(f"Download attempt {attempt + 1} failed: {e}")
            attempt += 1
            await asyncio.sleep(delay)  # 等待重试
        except httpx.HTTPStatusError as exc:
            print(f"Failed to download {url},HTTP error occurred: {exc.response.status_code} - {exc.response.text}")
        except TypeError:
            return download_by_requests(url, save_path, chunk_size, in_decode, time_out), file_name
        except Exception as e:
            print(f"Error: {e}, downloading url: {url} file: {file_name}")
            break

    return None, None


if __name__ == "__main__":
    AccessToken = 'd04149e455d44ac09432f0f89c3e0a41'

    import nest_asyncio

    nest_asyncio.apply()


    async def test():
        result = await baidu_nlp(text="百度是一家人工只能公司", nlp_type="ecnet")
        print(result)

    # asyncio.run(test())
