import aiohttp, aiofiles, asyncio
from pathlib import Path
import json


async def backup_to_webdev(file_path: str | Path, api_url: str, api_key: str = None,
                           username: str = None, password: str = None, metadata: dict = None,
                           timeout: float = 60.0, max_retries: int = 3, retry_delay: float = 2.0):
    """
    异步将本地文件备份到 WebDev 服务的 API，支持 API Key 或 Basic Auth 鉴权
    """
    # 转换路径对象并验证文件存在
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"备份文件不存在: {file_path}")

    # 准备请求头
    headers = {
        "User-Agent": "WebDevBackup/1.0",
        "X-Backup-Source": "backup-client-script"
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    auth = aiohttp.BasicAuth(login=username, password=password) if username and password else None

    # 重试机制
    for attempt in range(max_retries + 1):
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout), headers=headers,
                                             auth=auth) as session:

                async with aiofiles.open(file_path, "rb") as f:
                    file_content = await f.read()

                form = aiohttp.FormData()
                if metadata:
                    form.add_field("metadata", json.dumps(metadata), content_type="application/json")

                form.add_field(name="backup_file", value=file_content,
                               filename=file_path.name, content_type="application/octet-stream")

                async with session.post(url=api_url, data=form) as response:
                    # 检查HTTP状态
                    if response.status != 200:
                        error_text = await response.text()
                        print(f"API返回错误状态: {response.status} - {error_text}")
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                            message=f"API错误: {error_text[:200]}"
                        )

                    # 解析JSON响应
                    try:
                        result = await response.json()
                    except json.JSONDecodeError:
                        error_text = await response.text()
                        raise ValueError(f"无效的JSON响应: {error_text[:200]}")

                    # 验证业务逻辑成功
                    if not result.get("success"):
                        error_msg = result.get("error", "未知API错误")
                        raise ValueError(f"API业务错误: {error_msg}")

                    # 获取备份ID
                    backup_id = result.get("backup_id")
                    if not backup_id:
                        raise ValueError("响应中缺少backup_id字段")

                    print(f"备份成功！文件ID: {backup_id}, 大小: {result.get('size')}字节")
                    return backup_id

        except (aiohttp.ClientConnectionError, aiohttp.ServerTimeoutError) as e:
            if attempt < max_retries:
                print(f"网络错误 ({str(e)}), 尝试 {attempt + 1}/{max_retries}...")
                await asyncio.sleep(retry_delay * (attempt + 1))
            else:
                raise ConnectionError(f"网络连接失败: {str(e)}") from e

        except aiohttp.ClientError as e:
            raise ConnectionError(f"HTTP客户端错误: {str(e)}") from e

    return None
