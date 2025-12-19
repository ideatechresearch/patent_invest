from typing import List, Tuple, Dict, Union, Optional, Any, Callable
import asyncio, aiofiles, aiohttp, httpx
import json, requests, time, os
from pathlib import Path
from openai import AsyncOpenAI
import numpy as np

from agents.ai_prompt import System_content
from utils import build_prompt, create_analyze_messages, deduplicate_functions_by_name, run_togather, get_tokenizer, \
    lang_token_size, generate_hash_key, extract_json_struct, normalize_embeddings, cosine_similarity_np
from service import AI_Client, DB_Client, AliyunBucket, BaseMysql, AsyncAbortController, LRUCache, logger, get_redis, \
    get_httpx_client, post_aiohttp_stream, upload_file_to_oss, upload_file_to_oss_from_file, find_ai_model, \
    async_error_logger, async_polling_check
from database import BaseReBot
from config import Config


@async_error_logger(1)
async def ai_client_completions(messages: list[dict], client: Optional[AsyncOpenAI] = None,
                                model: str = 'deepseek-chat', get_content=True, tools: list[dict] = None,
                                max_tokens: int = 4096, top_p: float = 0.95, temperature: float = 0.1, dbpool=None,
                                **kwargs):
    """
    return: str or 模型响应的消息对象, lastrowid
    """
    chat = True
    if not client:
        try:
            model_info, name = find_ai_model(model, 0, search_field='model')
        except:
            chat = False
            model_info, name = find_ai_model(model, search_field="generation")

        client = AI_Client.get(model_info['name'], None)
        if not client:
            raise ValueError(f"Client for model {model_info['name']}:{model} not found")
        model = name

    reference = kwargs.pop('reference', None)
    payload = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stream=False,
        **kwargs
    )
    try:
        if chat:
            if tools:
                filter_tools: list[dict] = deduplicate_functions_by_name(tools)
                if any(filter_tools):  # 一旦你传了 tools 字段，它 必须 至少包含一个合法的 tool
                    payload['tools'] = filter_tools
                    # tool_choice="auto",

            completion = await client.chat.completions.create(**payload)
            if completion is None:
                raise ValueError("OpenAI API returned None instead of a valid response")
            if not completion.choices or not hasattr(completion.choices[0], "message"):
                raise ValueError("Unexpected API response.Incomplete,missing choices or message")
            row_id = await BaseReBot.async_save(model=model, messages=messages, model_response=completion.model_dump(),
                                                reference=reference, dbpool=dbpool or DB_Client,
                                                user='local', agent='chat')
            return (completion.choices[0].message.content if get_content else completion), row_id
        else:
            payload['prompt'] = build_prompt(messages, use_role=False)
            payload.pop('messages', None)
            completion = await client.completions.create(**payload)
            row_id = await BaseReBot.async_save(model=model, messages=messages, model_response=completion.model_dump(),
                                                reference=reference, dbpool=dbpool or DB_Client,
                                                user='local', agent='generate')
            return (completion.choices[0].text if get_content else completion), row_id
    except Exception as e:
        logger.error(f"OpenAI error occurred: {e}")
        raise


async def ai_generate_metadata(function_code: str, metadata: dict = None, model_name=Config.DEFAULT_MODEL_METADATA,
                               description: str = None, code_type: str = "Python", dbpool=None, **kwargs) -> dict:
    if not model_name:
        model_name = Config.DEFAULT_MODEL_METADATA

    prompt = System_content.get('84').format(code_type=code_type.lower(), function_code=function_code)
    lines = [f"帮我根据函数代码生成提取函数元数据（JSON格式）。"]
    if metadata:
        lines.append(f"当前已有初始元数据如下，请在此基础上补全或修正:\n{json.dumps(metadata, ensure_ascii=False)}")
    if description:
        lines.append(f"工具描述为:{description}，请结合此信息完善函数用途和说明")

    messages = create_analyze_messages(prompt, "\n".join(lines))
    content, row_id = await ai_client_completions(messages, client=None, model=model_name, get_content=True,
                                                  reference=function_code, max_tokens=1000, temperature=0.3,
                                                  dbpool=dbpool, **kwargs)
    if content:
        parsed = extract_json_struct(content)
        if parsed:
            metadata = parsed
            await BaseReBot.async_save(data={"transform": BaseMysql.format_value(parsed)},
                                       row_id=row_id, dbpool=dbpool or DB_Client)
        else:
            logger.warning(f'metadata extract failed:{content}')

    return metadata or {}


async def ai_batch_run(variable_name: str, variable_values: list, messages: list[dict] = None,
                       model='deepseek-reasoner', max_tokens=4096, temperature: float = 0.2, get_content: bool = True,
                       max_retries: int = 2, max_concurrent: int = Config.MAX_CONCURRENT, dbpool=None, **kwargs):
    '''
    多变量并发分析，对同一数据使用不同变量值进行多角度分析,返回分析结果列表
    顺序与variable_values对应,variable_name: 要变化的变量名(覆盖)
    '''
    if variable_name == 'model':
        client = None
    else:
        model_info, name = find_ai_model(model, 0, search_field='model')
        client = AI_Client.get(model_info['name'], None)

    payload = {"messages": messages, "model": model, "max_tokens": max_tokens, "temperature": temperature, **kwargs}

    @run_togather(max_concurrent=max_concurrent, batch_size=-1, return_exceptions=True)
    async def _run(item):
        params = {**payload, variable_name: item}
        msg = params.pop("messages", messages)
        response, row_id = await ai_client_completions(msg, client=client, get_content=get_content,
                                                       dbpool=dbpool, **params, max_retries=max_retries)
        result = {"content": response} if get_content else response.model_dump()
        return {**result, f"variable_{variable_name}": item}

    results = await _run(inputs=variable_values)
    return [{'id': i, 'error': str(r)} if isinstance(r, Exception) else {'id': i, **r}
            for i, r in enumerate(results)]


async def ai_analyze(results: dict | list | str, system_prompt: str = None, desc: str = None,
                     model: str = 'deepseek-reasoner', max_tokens: int = 4096, temperature: float = 0.2,
                     dbpool=None, **kwargs):
    reference = None if isinstance(results, str) else results
    user_request = results
    if desc:
        user_request = f"{desc}:\n{user_request}"

    messages = create_analyze_messages(system_prompt, user_request)
    content, _ = await ai_client_completions(messages, client=None, model=model, get_content=True, reference=reference,
                                             max_tokens=max_tokens, temperature=temperature, dbpool=dbpool, **kwargs)
    print(f'</{desc}>: {content}')
    return content.split("</think>")[-1]


async def ai_files_messages(files: List[str], question: str = None, messages: list = None,
                            model_name: str = 'qwen-long', max_tokens: int = 4096, dbpool=None, **kwargs):
    """
    处理文件并生成 AI 模型的对话结果。

    :param files: 文件路径列表
    :param question:问题提取
    :param messages
    :param model_name: 模型名称
    :param max_tokens
    :param dbpool
    :return: 模型生成的对话结果和文件对象列表
    """
    model_info, name = find_ai_model(model_name, -1)
    client = AI_Client.get(model_info['name'], None)
    messages = messages or []
    for file_path in files:
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():  # .is_file()
            continue

        if client:  # purpose=Literal["assistants", "batch", "fine-tune", "vision","batch_output"]
            file_object = await client.files.create(file=file_path_obj, purpose="file-extract")
            if model_info['name'] in ('qwen', 'openai'):  # fileid 引用协议
                messages.append({"role": "system", "content": f"fileid://{file_object.id}", })
                # client.files.list()
                # 文件信息 file_info = await client.files.retrieve(file_object.id)
            else:  # 直接读取内容注入到 prompt 中,内容大小有限制,通用性最强,Claude、Mistral、Gemini、moonshot
                file_content = await client.files.content(file_id=file_object.id)  # 文件内容
                messages.append({"role": "system", "content": file_content.text, })

            await asyncio.sleep(0.03)
        else:
            dashscope_file_upload(messages, file_path=str(file_path_obj))

    if question:
        messages.append({"role": "user", "content": question})
        content, _ = await ai_client_completions(messages, client, model=name, get_content=True, reference=files,
                                                 max_tokens=max_tokens, dbpool=dbpool, **kwargs)
        messages.append({"role": "assistant", "content": content})
        return messages

    return messages


def dashscope_file_upload(messages, file_path='.pdf', api_key=Config.DashScope_Service_Key):
    url = 'https://dashscope.aliyuncs.com/compatible-mode/v1/files'
    headers = {
        # 'Content-Type': 'application/json',
        "Authorization": f'Bearer {api_key}',
    }

    try:
        with open(file_path, 'rb') as f:
            files = {
                'file': f,
                'purpose': (None, 'file-extract'),
            }
            response = requests.post(url, headers=headers, files=files, timeout=Config.HTTP_TIMEOUT_SEC)
            response.raise_for_status()

            file_object = response.json()
            file_id = file_object.get('id', 'unknown_id')

            messages.append({"role": "system", "content": f"fileid://{file_id}"})
            return file_object, file_id

    except Exception as e:
        return {"error": str(e)}, None


async def ai_speech_analyze(file_obj, audio_url: str = None, results: dict = None, system_prompt: str = None,
                            model: str = 'deepseek-reasoner', max_tokens: int = 8192, oss_expires: int = 86400,
                            interval: int = 3, timeout: int = 300, dbpool=None, **kwargs) -> dict:
    from agents.ai_multi import tencent_speech_to_text
    from utils import extract_transcription_segments

    if audio_url:  # 如果提供了 audio_url，优先使用
        result = await tencent_speech_to_text(None, audio_url, interval, timeout)
    else:
        file_obj.seek(0, os.SEEK_END)
        total_size = file_obj.tell()  # os.path.getsize(file_path)
        file_obj.seek(0)

        if total_size > 1024 * 1024 * 5:  # 大文件 → OSS URL 模式
            audio_url, _ = await asyncio.to_thread(upload_file_to_oss, AliyunBucket, file_obj, object_name=None,
                                                   expires=oss_expires, total_size=total_size)
            result = await tencent_speech_to_text(None, audio_url, interval, timeout)
        else:
            result = await tencent_speech_to_text(file_obj, None, interval, timeout)

    data = {"audio_url": audio_url}
    if "error" in result:
        data["error"] = result["error"] or "ASR service returned unexpected result"
        return data

    data["asr_task_id"] = result.get("task_id")
    transcription_cleand = extract_transcription_segments(result.get('text'))
    data["transcription"] = transcription_cleand
    if isinstance(results, dict):
        results["沟通记录"] = transcription_cleand
        user_request = results
    else:
        user_request = f"沟通记录:\n{transcription_cleand}"
    messages = create_analyze_messages(system_prompt, user_request)
    content, _ = await ai_client_completions(messages, client=None, model=model, get_content=True, reference=data,
                                             max_tokens=max_tokens, dbpool=dbpool, **kwargs)
    data["model_completion"] = content
    return data


Active_Stream_Events = AsyncAbortController(redis_client=None)


async def stream_chat_completion(client: AsyncOpenAI, payload: dict, stream_id: str = None):
    payload["stream"] = True
    set_usage = payload.get("stream_options", {}).get("include_usage", False)
    if not set_usage:
        if not isinstance(payload.get("stream_options"), dict):
            payload["stream_options"] = {}
        payload["stream_options"]["include_usage"] = True  # 可选，配置以后会在流式输出的最后一行展示token使用信息

    # print(payload, client)
    try:
        abort_event = await Active_Stream_Events.set_abort_event(stream_id)
        stream = await client.chat.completions.create(**payload)
        if not stream:
            raise ValueError("OpenAI API returned an empty response")
        if not hasattr(stream, "__aiter__"):
            raise TypeError("OpenAI API returned a non-streaming response")

        reasoning_content = ''
        assistant_content = []  # answer_content
        completion_chunk = None
        async for chunk in stream:
            if not chunk:
                continue

            completion_chunk = chunk
            if abort_event and abort_event.is_set():  # 被 poll_abort 触发 or should_abort_async
                yield '[DONE]', chunk.model_dump_json()
                await stream.close()
                logger.warning(f"检测到中断信号，已停止流: {stream_id}")
                break  # return

            # 若 choices 为空但包含 usage，说明是 stream 末尾数据,qwen
            if not chunk.choices:
                if set_usage:
                    yield None, chunk.model_dump_json()  # '[DONE]' usage
                if assistant_content:
                    break
                else:
                    continue

            delta = chunk.choices[0].delta
            if delta.content:  # 以两个换行符 \n\n 结束当前传输的数据块
                assistant_content.append(delta.content)
                yield delta.content, chunk.model_dump_json()  # 获取字节流数据

            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                reasoning_content += delta.reasoning_content
                yield delta.reasoning_content, chunk.model_dump_json()

        if not assistant_content and not reasoning_content:
            raise ValueError("OpenAI API returned an empty stream response")
        if completion_chunk:
            completion = completion_chunk.model_dump()
            completion.update({
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": ''.join(assistant_content),
                                    "reasoning_content": reasoning_content},
                        "finish_reason": "stop",  # "length"
                        # "metadata": {"reasoning": reasoning}  # .split("</think>")[-1]
                    }
                ],
            })
            yield None, json.dumps(completion)

    except Exception as e:
        logger.error(f"OpenAI error:{e}, {payload}")
        error_message = f"OpenAI error occurred: {e}"
        fake_response = {
            "id": f"chatcmpl-{int(time.time())}", "object": "chat.completion.chunk",
            "created": int(time.time()), "model": payload.get("model", "unknown"),
            "choices": [{"index": 0, "delta": {"content": error_message}, "finish_reason": "stop"}],
        }
        yield error_message, json.dumps(fake_response)

    finally:
        if stream_id:
            await Active_Stream_Events.delete(stream_id)


async def ai_try(client: AsyncOpenAI, payload: dict, e: Exception) -> Tuple[str, Dict]:
    """
        封装 BadRequestError 的错误信息解析和修正处理逻辑。
        # 尝试解析结构化 JSON 错误信息
        # This model only supports streaming responses. Please set stream=True.
        # Rate limit reached for TPM
        # Model disabled
        # Unsupported model
        # Invalid URL
        # does not exist or you do not have access to it
        # Authenticaton failed, please make sure that a valid ModelScope token is supplied.
        # You exceeded your current quota, please check your plan and billing details.
        # parameter.enable_thinking must be set to false for non-streaming calls
        # openai.BadRequestError: Error code: 400 - {'error': {'code': 'invalid_parameter_error', 'param': None, 'message': 'This model only support stream mode, please enable the stream parameter to access the model. ', 'type': 'invalid_request_error'}, 'id': 'chatcmpl-4c7a91c5-c35c-9e94-8ff8-6e1b09a8a151', 'request_id': '4c7a91c5-c35c-9e94-8ff8-6e1b09a8a151'}
        # openai.BadRequestError: Error code: 400 - {'error': {'code': 'invalid_parameter_error', 'param': None, 'message': '<400> InternalError.Algo.InvalidParameter: Range of input length should be [1, 3072]', 'type': 'invalid_request_error'},
        # openai.BadRequestError: Error code: 400 - {'error': {'message': 'Invalid max_tokens value, the valid range of max_tokens is [1, 8192]', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_request_error'}}

       Args:
           e: 捕获的 BadRequestError 异常对象
           client:
           payload:模型参数

       Returns:
           fallback_callable 的执行结果（内容或完整响应）

       Raises:
           原始异常或 ValueError（当输入长度过大）
       """
    try:
        if hasattr(e, "response") and hasattr(e.response, "json"):
            error_json = e.response.json()
        elif hasattr(e, "response") and hasattr(e.response, "text"):
            error_json = json.loads(e.response.text)
        else:
            error_json = {}

        error_message = error_json.get("error", {}).get("message", "") or str(e)
    except Exception:
        error_message = str(e)

    logger.error(f"[BadRequestError] 捕获错误消息：{error_message}")

    stream_required_phrases = [
        "only support stream mode",
        "only supports streaming",
        "non-streaming calls",  # +/no_think
        "stream=true",
        "模型不支持sync调用"
    ]
    length_required_phrases = [
        "range of input length",
        "input length should be",
        "max input characters",
        "invalid max_tokens value",
        "the max_tokens must be less than"
    ]
    not_exists_phrases = ['model not exists', 'invalid model id']

    error_msg = error_message.lower()

    if any(p in error_msg for p in stream_required_phrases):
        payload["extra_body"] = {"enable_thinking": True}
        try:
            last_value = None
            async for chunk in stream_chat_completion(client, payload):
                last_value = chunk
            if last_value is None:
                raise RuntimeError("[stream_chat_completion] No data received from stream.")
            _, completion = last_value
            completion_data = json.loads(completion)
            content = completion_data.get('choices', [{}])[0].get('message', {}).get('content')
            await BaseReBot.async_save(model=payload.get('model'), messages=payload.get('messages', []),
                                       assistant_content=content, model_response=completion_data, dbpool=DB_Client,
                                       user='local', agent='chat', robot_id='stream_chat')
        except Exception as e:
            raise RuntimeError(f"[stream_chat_completion] Failed to complete streaming call: {e}") from e
        return content, completion_data
    elif any(p in error_msg for p in length_required_phrases):
        raise ValueError(f"[InputTokenError] 输入超过模型限制或无效：{error_message}")
        # payload['message']=cut_chat_history(payload['message'], max_size=33000)
    elif any(p in error_msg for p in not_exists_phrases):
        raise ValueError(f"[InputModelError] 模型无效<{payload.get('model')}>：{error_message}")
    else:
        # 内部 raise 会继续被下一个 except 捕获
        raise e


async def ai_batch(inputs: List[list[dict] | tuple[str, str]], task_id: str, model: str = 'qwen-long',
                   search_field: str = 'model', **kwargs):
    model_info, name = find_ai_model(model, -1, search_field)
    endpoint_map = {  # 选择Embedding文本向量模型进行调用时，url的值需填写"/v1/embeddings",其他模型填写/v1/chat/completions
        "model": "/v1/chat/completions",
        "embedding": "/v1/embeddings",
        "generation": "/v1/completions"
    }
    endpoint = endpoint_map.get(search_field, "/v1/chat/completions")
    input_filename = f'{task_id}_input.jsonl'
    file_path = Path(Config.DATA_FOLDER) / input_filename
    async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
        for i, msg in enumerate(inputs):
            messages = create_analyze_messages(msg[0], msg[1]) if isinstance(msg, tuple) else msg
            body = {"model": name, "messages": messages, **kwargs}
            request = {"custom_id": i, "method": "POST", "url": endpoint, "body": body}
            await f.write(json.dumps(request, separators=(',', ':'), ensure_ascii=False) + "\n")

    client = AI_Client.get(model_info['name'], None)
    # Step 1: 上传包含请求信息的JSONL文件,得到输入文件ID,如果您需要输入OSS文件,可将下行替换为：input_file_id = "实际的OSS文件URL或资源标识符"
    file_object = await client.files.create(file=file_path, purpose="batch")
    # Step 2: 基于输入文件ID,创建Batch任务
    batch = await client.batches.create(input_file_id=file_object.id, endpoint=endpoint, completion_window="24h")
    return {
        "input_file_id": file_object.id,
        "batch_id": batch.id,
        "input_file": input_filename,
        "client_name": model_info['name']
    }


async def ai_batch_result(batch_id: str, task_id: str, client_name: str,
                          interval: int = 10,  # 每次查询间隔秒数
                          timeout: int = 3600,  # 最长等待时间（秒），默认 1 小时
                          oss_expires: int = 86400
                          ):
    """
    轮询批处理任务状态，等待完成后下载结果文件。
    - batch_id: 批处理任务ID
    - client_name: AI_Client 的 key
    - task_id: 用来命名输出文件，避免混淆
    - interval: 轮询间隔秒数，<=0 不轮询
    - timeout: 超时时间，防止永远等待，<=0 永远等待
    """
    client = AI_Client.get(client_name, None)
    if not client:
        raise ValueError(f"Client '{client_name}' not found")

    async def check_batch_status(_future=None):
        """
        轮询批处理状态，仅返回状态指示
        """
        batch = await client.batches.retrieve(batch_id=batch_id)
        status = batch.status

        if status in ["completed", "failed", "expired", "cancelled"]:
            # 结束状态 → 返回结果对象
            print(f"[Batch状态] {batch_id}: {status}")
            return batch

        if interval <= 0:
            print(batch.model_dump())
            _future.set_result(batch)
            return batch

        return False  # 继续轮询

    polling_func = async_polling_check(interval=interval, timeout=timeout)(check_batch_status)
    future, handle = await polling_func()
    try:
        batch = await future  # 等待轮询完成
    except TimeoutError:
        return {"status": "timeout"}
    except Exception as e:
        logger.exception(f"[Batch异常] {batch_id} 异步轮询出错: {e}")
        return {"status": "error", "error": str(e)}
    finally:
        handle['cancelled'] = True

    # ✅ 任务完成，下载结果文件
    status = batch.status
    result = {"status": status, "request_counts": batch.request_counts.model_dump()}

    if status == "failed":
        result["errors"] = str(batch.errors)
        return result

    output_filename = f'{task_id}_{batch_id}_result.jsonl'
    error_filename = f'{task_id}_{batch_id}_error.jsonl'
    output_file_path = Path(Config.DATA_FOLDER) / output_filename
    error_file_path = Path(Config.DATA_FOLDER) / error_filename
    if batch.error_file_id and not error_file_path.exists():
        content = await client.files.content(batch.error_file_id)
        logger.error(f"[Batch error] {batch_id} 前1000字符:\n{content.text[:1000]}...\n")
        content.write_to_file(error_file_path)
        result["error_file"] = error_filename
        result["error_file_id"] = batch.error_file_id
        result["batch_id"] = batch.id or batch_id

    if batch.output_file_id and not output_file_path.exists():
        content = await client.files.content(batch.output_file_id)
        print(f"[Batch result] {batch_id} 前1000字符:\n{content.text[:1000]}...\n")
        content.write_to_file(output_file_path)
        result["result_file"] = output_filename
        result["output_file_id"] = batch.output_file_id
        result["batch_id"] = batch.id or batch_id
        if oss_expires != 0:
            object_name = f"upload/{output_filename}"
            result["output_file_url"], _ = await asyncio.to_thread(
                upload_file_to_oss_from_file,
                AliyunBucket,
                output_file_path, object_name, oss_expires)
            # os.remove(output_file_path)

    return result


Embedding_Cache = LRUCache(1000, redis=get_redis(), prefix="embedding", expire_sec=Config.REDIS_CACHE_SEC)


# https://www.openaidoc.com.cn/docs/guides/embeddings
async def ai_embeddings(inputs: Union[str, List[str], Tuple[str], List[Dict[str, str]]],
                        model_name: str = Config.DEFAULT_MODEL_EMBEDDING,
                        model_id: int = 0, get_embedding: bool = True, normalize: bool = False,
                        **kwargs) -> Union[List[List[float]], Dict]:
    """
    text = text.replace("\n", " ")
    检查缓存，如果没有就计算嵌入
    从远程服务获取嵌入，支持批量处理和缓存和多模型处理。
    :param inputs: 输入文本或文本列表
    :param model_name: 模型名称
    :param model_id: 模型序号
    :param get_embedding: 返回响应数据类型
    :param normalize: 归一化
    :return: 嵌入列表
    （1）文本数量不超过 16。
    （2）每个文本长度不超过 512 个 token，超出自动截断，token 统计信息，token 数 = 汉字数+单词数*1.3 （仅为估算逻辑，以实际返回为准)。
    （3） 批量最多 16 个，超过 16 后默认截断。
    """
    if not model_name:
        return []
    if not inputs:
        return []

    redis = get_redis()
    cache_key, embedding = await Embedding_Cache.get_cache(args=[model_name, model_id, get_embedding, normalize],
                                                           redis=redis, default=[], inputs=inputs)
    if embedding:
        # print(f"Embedding already cached for key: {cache_key}")
        if isinstance(embedding, list) and all(isinstance(vec, list) for vec in embedding):
            return embedding

    try:
        model_info, name = find_ai_model(model_name, model_id, 'embedding')
    except Exception as e:
        print(e)
        return []

    batch_size = 10  # DASHSCOPE_MAX_BATCH_SIZE = 16,25
    has_error = False
    payload = dict(
        model=name,
        encoding_format="float",
        # "dimensions":# 指定向量维度（text-embedding-v3及 text-embedding-v4）
    )
    payload.update(kwargs)

    client = AI_Client.get(model_info['name'], None)
    if client:  # openai.Embedding.create
        if isinstance(inputs, (list, tuple)) and len(inputs) > batch_size:
            create_embeddings = run_togather(max_concurrent=Config.MAX_CONCURRENT, batch_size=batch_size,
                                             input_key="input")(client.embeddings.create)
            results = await create_embeddings(inputs=inputs, **payload)
            if not get_embedding:
                results_data = results[0]
                all_data = []
                for i, result in enumerate(results):
                    for idx, item in enumerate(result.data):
                        item.index = len(all_data)
                        all_data.append(item)

                    if i > 0:
                        results_data.usage.prompt_tokens += result.usage.prompt_tokens
                        results_data.usage.total_tokens += result.usage.total_tokens

                results_data.data = all_data
                return results_data.model_dump()

            embeddings = [None] * len(inputs)
            input_idx = 0
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"Error encountered: {result}")
                    has_error = True
                    continue
                for idx, item in enumerate(result.data):
                    embeddings[input_idx] = item.embedding
                    input_idx += 1
        else:
            # await asyncio.to_thread(client.embeddings.create
            results = await client.embeddings.create(input=inputs, **payload)

            if not get_embedding:
                return results.model_dump()

            embeddings = [item.embedding for item in results.data]

        if normalize and not any(embedding is None for embedding in embeddings):
            embeddings = normalize_embeddings(embeddings, to_list=True)

        if not has_error:  # and len(embeddings) > batch_size
            await Embedding_Cache.set_cache(cache_key, embeddings, redis=redis)

        return embeddings

    # dashscope.TextEmbedding.call
    url = model_info['embedding_url'] if model_info.get('embedding_url') else model_info['base_url'] + '/embeddings'
    api_key = model_info['api_key']
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    payload["input"]: list | str = inputs
    embeddings = []
    cx = get_httpx_client()
    try:
        if isinstance(inputs, (list, tuple)) and len(inputs) > batch_size:
            for i in range(0, len(inputs), batch_size):
                batch = inputs[i:i + batch_size]
                payload["input"] = batch  # {"texts":batch}
                response = await cx.post(url, headers=headers, json=payload, timeout=Config.LLM_TIMEOUT_SEC)
                response.raise_for_status()
                data = response.json().get('data')  # "output"
                if data and len(data) == len(batch):
                    embeddings += [emb.get('embedding') for emb in data]
                else:
                    print(f"Error: Batch {i // batch_size + 1} response size mismatch.")
                    has_error = True
        else:
            response = await cx.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json().get('data')
            embeddings = [emb.get('embedding') for emb in data]

    except (httpx.HTTPStatusError, httpx.RequestError) as exc:
        print(exc)

    if normalize:
        embeddings = normalize_embeddings(embeddings, to_list=True)

    if not has_error:
        await Embedding_Cache.set_cache(cache_key, embeddings, redis=redis)

    return embeddings


@async_error_logger(1)
async def ai_reranker(query: str, documents: List[str], top_n: int, model_name="BAAI/bge-reranker-v2-m3", model_id=0,
                      **kwargs):
    if not model_name:
        return []
    model_info, name = find_ai_model(model_name, model_id, 'reranker')

    url = model_info['reranker_url']
    api_key = model_info.get('api_key')
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    payload = {
        "query": query,
        "model": name,
        "documents": documents,
        "top_n": top_n,
        "return_documents": False,
        # 'max_chunks_per_doc': 1024,  # 最大块数
        # 'overlap_tokens': 80,  # 重叠数量
    }
    payload.update(kwargs)
    async with httpx.AsyncClient(timeout=Config.HTTP_TIMEOUT_SEC) as cx:
        response = await cx.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            results = response.json().get('results')
            matches = [(match["document"].get("text", match["document"])
                        if match.get("document") else documents[match["index"]],
                        match["relevance_score"], match["index"])
                       for match in results]
            return matches
        else:
            print(response.text)
    return []


async def similarity_score_embeddings(query, tokens: List[str], filter_idx: List[int] = None,
                                      tokens_vector: List[float] = None,
                                      embeddings_calls: Callable[[...], Any] = ai_embeddings, **kwargs):
    """
    计算查询与一组标记之间的相似度分数。
    参数:
        query (str): 查询字符串。
        tokens (List[str]): 要比较的标记列表。
        filter_idx (List[int], optional): 要过滤出的标记索引。默认为 `None`，表示不进行过滤。
        tokens_vector (np.ndarray, optional): 预计算的标记向量数组。默认为 `None`。
        embeddings_calls (Callable, optional): 用于生成嵌入的异步函数，默认为 `None`。
        **kwargs: 传递给 `embeddings_calls` 的额外参数。

    返回:
        np.ndarray: 相似度得分数组（与 `tokens` 长度一致）。
    """
    if filter_idx is None:
        filter_idx = list(range(len(tokens)))
    else:
        filter_idx = [i for i in filter_idx if i < len(tokens)]
    similarity = np.full(len(tokens), np.nan)
    if not query:
        return similarity

    tokens_idx = [(i, token) for i, token in enumerate(tokens) if token]
    query_vector = await embeddings_calls(query, **kwargs)
    if tokens_vector is None:
        # list(np.array(idx_tokens)[:, 1])
        tokens_vector = await embeddings_calls([token for _, token in tokens_idx], **kwargs)

    matching_indices = np.isin([j[0] for j in tokens_idx], filter_idx)
    filter_embeddings = np.array(tokens_vector)[matching_indices]

    if len(filter_embeddings):
        # cosine_similarity_np(np.array(query_vector).reshape(1, -1), filter_embeddings).T
        sim_2d = np.array(query_vector).reshape(1, -1) @ filter_embeddings.T
        # matching_indices = np.isin(filter_idx, [j[0] for j in tokens_idx])
        # similarity[matching_indices] = sim_2d.reshape(-1)
        for idx, score in zip(filter_idx, sim_2d.flatten()):
            similarity[idx] = score

    return similarity


async def get_similar_embeddings(querys: Union[str, List[str]], tokens: List[str], topn: int = 10,
                                 embeddings_calls: Callable[[...], Any] = ai_embeddings,
                                 **kwargs) -> List[Tuple[str, List[Tuple[str, float, int]]]]:
    """
    使用嵌入计算查询与标记之间的相似度。
    :param querys: 查询词列表
    :param tokens: 比较词列表
    :param embeddings_calls: 嵌入生成函数
    :param topn: 返回相似结果的数量
    :param kwargs: 其他参数
    返回：
        List[Tuple[str, List[Tuple[str, float,int]]]]: 查询词与相似标记及其分数的映射。
    """
    if isinstance(querys, str):
        querys = [querys]

    query_vector, tokens_vector = await asyncio.gather(
        embeddings_calls(querys, **kwargs),
        embeddings_calls(tokens, **kwargs))

    if not query_vector or not tokens_vector:
        return []

    sim_matrix = np.array(query_vector) @ np.array(tokens_vector).T
    results = []
    for i, w in enumerate(querys):
        sim_scores = sim_matrix[i]
        top_indices = np.argsort(sim_scores)[::-1][:topn] if topn > 0 else np.arange(sim_scores.shape[0])
        top_scores = sim_scores[top_indices]
        top_match = [tokens[j] for j in top_indices]
        results.append((w, list(zip(top_match, top_scores, top_indices))))

    return results  # [(q,[(match,score,index),])]


def get_similar_words(querys: List[str], data: dict, exclude: List[str] = None, topn: int = 10,
                      cutoff: float = 0.0) -> List[Tuple[str, List[Tuple[str, float]]]]:
    '''
   计算查询词与数据中的词之间的相似度，并返回每个查询词的最相似词及其相似度。

    :param querys: 查询词列表
    :param data: 包含词汇和对应向量的字典，格式为：
                 {
                     "name": ["word1", "word2", ...],
                     "vectors": [[0.1, 0.2, ...], [0.3, 0.4, ...], ...]
                 }
    :param exclude: 要排除的词列表，这些词不会出现在结果中
    :param topn: 返回最相似的前 n 个词
    :param cutoff: 只有相似度大于该值的词才会被返回
    :return: 每个查询词的相似词和相似度的元组列表
    返回：
        List[Tuple[str, List[Tuple[str, float]]]]: 查询词与相似标记及其分数的映射。
    '''
    index_to_key = data['name']
    vectors = np.array(data['vectors'])

    # 获取索引
    query_mask = np.array([w in index_to_key for w in querys])
    exclude_mask = np.array([w in querys + (exclude or []) for w in index_to_key])
    # np.delete(vectors, exclude_indices, axis=0)

    # 计算余弦相似度
    sim_matrix = cosine_similarity_np(vectors[query_mask], vectors[~exclude_mask].T)

    results = []
    for i, w in enumerate(querys):
        if not query_mask[i]:
            continue
        sim_scores = sim_matrix[i]
        if topn > 0:
            top_indices = np.argsort(sim_scores)[::-1][:topn]  # 获取前 topn 个索引
        else:
            top_indices = np.arange(sim_scores.shape[0])  # 获取全部索引，不排序
        top_scores = sim_scores[top_indices]

        valid_indices = top_scores > cutoff  # 保留大于 cutoff 的相似度
        top_words = [index_to_key[j] for j in np.where(~exclude_mask)[0][top_indices[valid_indices]]]
        top_scores = top_scores[valid_indices]
        results.append((w, list(zip(top_words, top_scores))))

    return results  # [(w, list(sim[w].sort_values(ascending=False)[:topn].items())) for w in querys]


async def find_closest_matches_embeddings(querys: List[str], tokens: List[str],
                                          embeddings_calls: Callable[[...], Any] = ai_embeddings,
                                          **kwargs) -> Dict[str, Tuple[str, float]]:
    """
    使用嵌入计算查询与标记之间的最近匹配，找到每个查询的最佳匹配标记,topk=1。

    :param querys: 查询字符串列表。
    :param tokens: 标记列表，用于匹配查询。
    :param embeddings_calls: 嵌入生成函数，接收查询和标记返回嵌入向量（默认为 ai_embeddings）。
    :param kwargs: 其他参数，传递给 embeddings_calls 函数。
    :return: 返回一个字典，其中每个查询的最佳匹配标记及其相似度分数被映射到查询上。
    返回：
        Dict[str, Tuple[str, float]]: 查询与最近匹配标记的映射字典。
    """
    matches = {x: (x, 1.0) for x in querys if x in tokens}
    unmatched_queries = list(set(querys) - matches.keys())
    # 所有查询都已匹配，直接返回完全匹配项
    if not unmatched_queries:
        return matches
    query_vector, tokens_vector = await asyncio.gather(
        embeddings_calls(unmatched_queries, **kwargs),
        embeddings_calls(tokens, **kwargs))
    sim_matrix = np.array(query_vector) @ np.array(tokens_vector).T

    closest_matches = tokens[sim_matrix.argmax(axis=1)]  # idxmax
    closest_scores = sim_matrix.max(axis=1)
    matches.update(zip(unmatched_queries, zip(closest_matches, closest_scores)))
    return matches


async def process_llm_stream(llm_responses_stream, token_size=20, model_name="gpt-3.5-turbo"):
    """
    处理大模型返回的文本流，并按标点符号分割交给 TTS 朗读。
    :param llm_responses_stream: 大模型返回的文本流
    :param token_size: 标点不足时，允许的最小缓冲区长度
    :param  model_name: tokenizer model
    """
    from utils.utils import find_last_punctuation, get_string_no_punctuation_or_emoji
    response_message = []
    text_index = 0
    processed_chars = 0
    tokenizer = get_tokenizer(model_name)
    async for content in llm_responses_stream:
        response_message.append(content)
        if Active_Stream_Events.should_abort(stream_id='LLM_Controller'):  # 实时检查是否终止
            break

        # 获取当前未处理的文本
        full_text = "".join(response_message)
        current_text = full_text[processed_chars:]

        # 查找最后一个有效标点
        last_punct_pos = find_last_punctuation(current_text)
        if last_punct_pos != -1 or lang_token_size(current_text, tokenizer) > token_size:
            split_pos = last_punct_pos if last_punct_pos != -1 else token_size  # 选取最合适的切割点
            segment_text_raw = current_text[:split_pos + 1]
            segment_text = get_string_no_punctuation_or_emoji(segment_text_raw)  # 处理无效字符
            if segment_text:
                text_index += 1
                yield segment_text, text_index
                processed_chars += len(segment_text_raw)  # 更新已处理字符位置

    # 处理剩余未分割的文本
    remaining_text = "".join(response_message)[processed_chars:]
    if remaining_text:
        segment_text = get_string_no_punctuation_or_emoji(remaining_text)
        if segment_text:
            text_index += 1
            yield segment_text, text_index

    yield response_message, -1  # finish_task


async def start_llm_stream(new_llm_stream):
    """复位终止信号，并重新启动大模型流"""
    abort_event = Active_Stream_Events.get_abort_event(stream_id='LLM_Controller')
    if not abort_event:
        abort_event = await Active_Stream_Events.set_abort_event(stream_id='LLM_Controller')
    abort_event.clear()  # 重新启动前复位
    async for text, idx in process_llm_stream(new_llm_stream):
        if idx > 0:
            print(f"🔊 朗读: {text}")


async def call_ollama(prompt: str | list[str] = None, messages: list[dict] = None, model_name="mistral",
                      host='localhost', time_out: float = 100, stream=True, tools: list[dict] = None,
                      embed: bool = False, **kwargs):
    """
    https://github.com/ollama/ollama/blob/main/docs/api.md
    返回两种模式的结果：async generator（stream）或 JSON dict（非 stream）
    """
    if not prompt and not messages:
        raise ValueError("必须提供 prompt 或 messages 之一")
    # api/tags:列出本地模型,/api/ps:列出当前加载到内存中的模型

    if embed:
        url = f"http://{host}:11434/api/embed"
        payload = {"model": model_name, "input": prompt}
    elif messages is not None:
        url = f'http://{host}:11434/api/chat'
        payload = {
            "model": model_name,
            "messages": messages,
            # 如果 messages 数组为空，则模型将被加载到内存中, [{"role": "user","content": "Hello!","images": ["..."]}]
            "stream": stream
        }
        if tools:
            payload['tools'] = tools
    else:
        if isinstance(prompt, list):
            prompt = "\n".join(prompt)

        url = f"http://{host}:11434/api/generate"
        payload = {"model": model_name, "prompt": prompt, "stream": stream
                   # "suffix": "    return result",模型响应后的文本
                   # "images"：要包含在消息中的图像列表（对于多模态模型，例如llava)
                   # "keep_alive": 控制模型在请求后加载到内存中的时间（默认值：5m)
                   # "keep_alive": 0，如果提供了空提示，将从内存中卸载模型
                   # "format": "json",
                   # "options": {
                   #     "temperature": 0,"seed": 123，"top_k": 20,"top_p": 0.9, "stop": ["\n", "user:"],
                   #  "num_batch": 2,"num_gpu": 1,"main_gpu": 0,"use_mmap": true,"num_thread": 8
                   # },
                   }

    payload.update(**kwargs)

    if stream:
        return post_aiohttp_stream(url, payload, time_out)
        # content=''
        # async for chunk in post_aiohttp_stream(url,payload):
        #     content += chunk.get("response", "")
        # return content
    timeout = aiohttp.ClientTimeout(total=time_out, sock_read=time_out, sock_connect=10.0)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                raise Exception(f"Ollama Error: {response.status}: {await response.text()}")
            return await response.json()


async def openai_ollama_chat_completions(messages: list, model_name="qwen3:14b", host='localhost', stream=False,
                                         time_out=300, **kwargs):
    """
      模拟 OpenAI 的返回结构
      - 如果 stream=True，模拟流式返回
      - 如果 stream=False，返回完整 JSON 响应
      ['model', 'created_at', 'response', 'done', 'done_reason', 'context', 'total_duration', 'load_duration', 'prompt_eval_count', 'prompt_eval_duration', 'eval_count', 'eval_duration']
    """

    prompt = build_prompt(messages, use_role=True) + "\nAssistant:"
    response = await call_ollama(prompt, messages=messages, model_name=model_name, host=host, time_out=time_out,
                                 stream=stream, **kwargs)
    tokenizer = get_tokenizer(Config.DEFAULT_MODEL_ENCODING)
    if stream:
        async def stream_data():
            async for chunk in response:
                content = chunk.get("response", chunk.get('message', {}).get('content', ''))
                data = {"id": "chatcmpl-xyz", "object": "chat.completion.chunk",
                        "created": int(time.time()), "model": chunk.get('model', model_name),
                        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]},
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"  # (f"data: {data}\n\n").encode("utf-8")

        return stream_data()

    # print(json.dumps(response, indent=4))  # 打印响应数据
    content = response.get("response", response.get('message', {}).get('content', ''))
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),  # unix_timestamp(response["created_at"])
        "model": response.get('model', model_name),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": response.get('done_reason', "stop"),  # length/content_filter
            }
        ],
        "usage": {
            "prompt_tokens": lang_token_size(prompt, tokenizer),
            "completion_tokens": lang_token_size(content, tokenizer),  # .split()
            "total_tokens": lang_token_size(prompt, tokenizer) + lang_token_size(content, tokenizer),
        },
    }


__all__ = ['ai_analyze', 'ai_batch', 'ai_batch_result', 'ai_batch_run', 'ai_client_completions', 'ai_embeddings',
           'ai_files_messages', 'ai_generate_metadata', 'ai_reranker', 'ai_try', 'call_ollama', 'ai_speech_analyze',
           'dashscope_file_upload', 'find_closest_matches_embeddings',
           'get_similar_embeddings', 'get_similar_words',
           'openai_ollama_chat_completions', 'similarity_score_embeddings', 'stream_chat_completion',
           'Active_Stream_Events']

if __name__ == "__main__":
    from utils import get_module_functions

    funcs = get_module_functions('agents.ai_generates')
    print([i[0] for i in funcs])


    async def test():
        result = await call_ollama('你好', model_name="qwen3:14b", host='10.168.1.10', stream=False)

        print(result)
        print(result.keys())
        result = await call_ollama('你好啊', model_name="qwen3:14b", host='10.168.1.10', stream=True)
        # print(result)
        async for chunk in result:
            content = chunk.get("response")
            print(content, end="", flush=True)

        result = await call_ollama('你好啊', messages=[{"role": "user", "content": "Hello!"}], model_name="qwen3:14b",
                                   host='10.168.1.10', stream=True)
        print(result)
        async for chunk in result:
            content = chunk.get("message", [])
            print(content["content"], end="", flush=True)

        result = await openai_ollama_chat_completions(messages=[{"role": "user", "content": "你好啊"}],
                                                      model_name="qwen3:14b",
                                                      host='10.168.1.10', stream=False)
        print(result)

    # asyncio.run(test())
