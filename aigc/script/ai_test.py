from openai import AsyncOpenAI
import asyncio, time, json
from typing import Callable, Any
from functools import partial, wraps

Local_Base_Url = None  # 'http://47.110.156.41:7000/v1'
DeepSeek_API_Key = '***'
ai_client = AsyncOpenAI(base_url=Local_Base_Url or 'https://api.deepseek.com', timeout=300, api_key=DeepSeek_API_Key)


def run_togather(max_concurrent: int = 100, batch_size: int = -1, input_key: str = None):
    semaphore = asyncio.Semaphore(max_concurrent) if max_concurrent > 0 else None

    def decorator(func: Callable[..., Any]):
        @wraps(func)
        async def wrapper(*args, inputs: list = None, **kwargs):
            _args = list(args)
            if inputs is None and _args:
                inputs = _args.pop(0)  # 允许 positional 方式传 list
            if not isinstance(inputs, list):
                inputs = [inputs]

            async def run_semaphore(x):
                if semaphore:
                    async with semaphore:
                        if input_key:
                            return await func(*_args, **{input_key: x}, **kwargs)
                        return await func(x, *_args, **kwargs)
                if input_key:
                    return await func(*_args, **{input_key: x}, **kwargs)
                return await func(x, *_args, **kwargs)

            if batch_size <= 0:
                # 所有 input 独立请求
                tasks = [run_semaphore(item) for item in inputs]
            else:
                tasks = [run_semaphore(inputs[i:i + batch_size]) for i in range(0, len(inputs), batch_size)]

            return await asyncio.gather(*tasks, return_exceptions=True)  # 确保任务取消时不会引发异常

        return wrapper

    return decorator


async def ai_test(user_request: str, model='deepseek-reasoner', max_tokens=4096, temperature: float = 0.2, **kwargs):
    messages = [{'role': 'user', 'content': user_request}]
    time_st = time.time()
    payload = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
    )
    if Local_Base_Url:
        payload['user'] = 'infr'
    payload.update(kwargs)
    completion = await ai_client.chat.completions.create(**payload)
    time_end = time.time()
    return {
        "response": completion.choices[0].message.model_dump(),  # completion.model_dump(),
        'usage': completion.usage.model_dump() if completion.usage else None,
        'created': completion.created,
        "elapsed": round(time_end - time_st, 6),
    }


async def test(test_count=100, max_concurrent=100, model="deepseek-chat"):
    input_test = ['请返回"测试通过"'] * test_count
    process_func = run_togather(max_concurrent=max_concurrent, input_key="user_request")(ai_test)
    start_all = time.time()
    results = await process_func(inputs=input_test, model=model)
    end_all = time.time()

    success = [r for r in results if isinstance(r, dict)]
    fail = [r for r in results if not isinstance(r, dict)]
    avg_time = sum(r["elapsed"] for r in success) / len(success) if success else 0
    summary = {
        "total_requests": test_count,
        "success_count": len(success),
        "fail_count": len(fail),
        "avg_elapsed": avg_time,
        "total_elapsed": end_all - start_all
    }

    # 保存
    with open("data/test_ai_data_3.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)

    return summary


if __name__ == '__main__':
    summary = asyncio.run(test(test_count=100, max_concurrent=500, model="deepseek-reasoner"))
    print("测试完成：", summary)
