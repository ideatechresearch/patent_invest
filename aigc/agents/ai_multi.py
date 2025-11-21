import httpx, requests
from http import HTTPStatus
from PIL import Image
import dashscope
# import qianfan
from dashscope.audio.tts import ResultCallback
from dashscope.audio.asr import Recognition, Transcription, RecognitionCallback
import time, base64, json, uuid, io
from urllib.parse import urlencode, urlparse, unquote

from service import get_httpx_client, async_polling_check
from secure import get_xfyun_signature, get_tencent_signature, get_xfyun_authorization, get_ark_signature, \
    get_baidu_access_token, get_aliyun_access_token, generate_hmac_signature
from config import Config


# from lagent import tool_api

def xunfei_ppt_theme(industry, style="简约", color="蓝色", appid: str = Config.XF_AppID,
                     api_secret: str = Config.XF_Secret_Key):
    url = "https://zwapi.xfyun.cn/api/ppt/v2/template/list"
    timestamp = int(time.time())
    signature = get_xfyun_signature(appid, api_secret, timestamp)
    headers = {
        "appId": appid,
        "timestamp": str(timestamp),
        "signature": signature,
        "Content-Type": "application/json; charset=utf-8"
    }
    # body ={
    #     "query": text,
    #     "templateId": templateId  # 模板ID举例，具体使用 /template/list 查询
    # }
    body = {
        "payType": "not_free",
        "style": style,  # 支持按照类型查询PPT 模板,风格类型： "简约","卡通","商务","创意","国风","清新","扁平","插画","节日"
        "color": color,  # 支持按照颜色查询PPT 模板,颜色类型： "蓝色","绿色","红色","紫色","黑色","灰色","黄色","粉色","橙色"
        "industry": industry,
        # 支持按照颜色查询PPT 模板,行业类型： "科技互联网","教育培训","政务","学院","电子商务","金融战略","法律","医疗健康","文旅体育","艺术广告","人力资源","游戏娱乐"
        "pageNum": 2,
        "pageSize": 10
    }

    response = requests.request("GET", url=url, headers=headers, params=body).text
    return response


# https://www.xfyun.cn/doc/spark/PPTv2.html
async def xunfei_ppt_create(text: str, templateid: str = "20240718489569D", appid: str = Config.XF_AppID,
                            api_secret: str = Config.XF_Secret_Key, max_retries=30):
    from requests_toolbelt.multipart.encoder import MultipartEncoder

    url = 'https://zwapi.xfyun.cn/api/ppt/v2/create'
    timestamp = int(time.time())
    signature = get_xfyun_signature(appid, api_secret, timestamp)
    form_data = MultipartEncoder(
        fields={
            # "file": (path, open(path, 'rb'), 'text/plain'),  # 如果需要上传文件，可以将文件路径通过path 传入
            # "fileUrl":"",   #文件地址（file、fileUrl、query必填其一）
            # "fileName":"",   # 文件名(带文件名后缀；如果传file或者fileUrl，fileName必填)
            "query": text,
            "templateId": templateid,  # 模板的ID,从PPT主题列表查询中获取
            "author": "XXXX",  # PPT作者名：用户自行选择是否设置作者名
            "isCardNote": str(True),  # 是否生成PPT演讲备注, True or False
            "search": str(True),  # 是否联网搜索,True or False
            "isFigure": str(True),  # 是否自动配图, True or False
            "aiImage": "normal"  # ai配图类型： normal、advanced （isFigure为true的话生效）；
            # normal-普通配图，20%正文配图；advanced-高级配图，50%正文配图
        }
    )

    print(form_data)
    headers = {
        "appId": appid,
        "timestamp": str(timestamp),
        "signature": signature,
        "Content-Type": form_data.content_type
    }

    response = requests.request(method="POST", url=url, data=form_data, headers=headers).text
    resp = json.loads(response)
    if resp.get('code') != 0:
        print('创建PPT任务失败,生成PPT返回结果：', response)
        return None

    task_id = resp['data']['sid']
    ppt_url = ''
    retries = 0
    # 轮询任务进度
    await asyncio.sleep(5)

    cx = get_httpx_client(time_out=Config.HTTP_TIMEOUT_SEC)
    while task_id is not None and retries < max_retries:
        task_url = f"https://zwapi.xfyun.cn/api/ppt/v2/progress?sid={task_id}"
        response = await cx.get(url=task_url, headers=headers)
        response.raise_for_status()
        resp = response.json()
        task_status = resp['data']['pptStatus']
        aiImageStatus = resp['data']['aiImageStatus']
        cardNoteStatus = resp['data']['cardNoteStatus']

        if ('done' == task_status and 'done' == aiImageStatus and 'done' == cardNoteStatus):
            ppt_url = resp['data']['pptUrl']
            break

        await asyncio.sleep(3)
        retries += 1

    return ppt_url


# https://www.xfyun.cn/doc/spark/ImageGeneration.html#%E9%89%B4%E6%9D%83%E8%AF%B4%E6%98%8E
async def xunfei_picture(text: str, data_folder=None):
    headers, url = get_xfyun_authorization(api_key=Config.XF_API_Key, api_secret=Config.XF_Secret_Key,
                                           host="spark-api.cn-huabei-1.xf-yun.com", path="/v2.1/tti", method='POST')
    url = 'http://spark-api.cn-huabei-1.xf-yun.com/v2.1/tti' + "?" + urlencode(headers)
    # 构造请求数据
    request_body = {
        "header": {
            "app_id": Config.XF_AppID,  # 你在平台申请的appid
            # 'uid'
            # "res_id": "your_res_id"  # 可选：自定义术语资源id
        },
        "parameter": {
            "chat": {
                "domain": "general",
                "temperature": 0.5,
                # "max_tokens": 4096,
                "width": 640,  # 默认大小 512*512
                "height": 480
            }
        },
        "payload": {
            "message": {
                "text": [
                    {
                        "role": "user",
                        "content": text
                    }
                ]
            }
        }
    }
    # 异步发送请求
    cx = get_httpx_client(time_out=Config.HTTP_TIMEOUT_SEC)
    response = await cx.post(url, json=request_body, headers={'content-type': "application/json"})
    if response.status_code != 200:
        return None, {"error": f"HTTP Error: {response.status_code}"}

    data = response.json()  # json.loads(response.text)
    code = data['header']['code']
    if code != 0:
        return None, {"error": f'请求错误: {code}, {data}'}

    text = data["payload"]["choices"]["text"]
    image_base = text[0]["content"]  # base64图片结果,base64_string_data
    image_id = data['header']['sid']
    # 解码 Base64 图像数据
    file_data = base64.b64decode(image_base)
    if data_folder:
        # 将解码后的数据转换为图片
        file_path = f"{data_folder}/{image_id}.jpg"  # .png
        img = Image.open(io.BytesIO(file_data))
        # r, g, b, a = img.split()
        # img = img.convert('RGB') 转换为 RGB 格式
        img.save(file_path)

        # buffer = io.BytesIO()
        # img.save(buffer, format="JPEG")  # 保存为 JPEG 格式
        # buffer.seek(0)
        # async with aiofiles.open(file_path, 'wb') as file:
        #     await file.write(buffer.read())
        return file_path, {"urls": '', 'id': image_id}

    return file_data, {"urls": '', 'id': image_id}


# https://www.volcengine.com/docs/6791/1361423
async def ark_visual_picture(image_data, image_urls: list[str], prompt: str = None, logo_info=None, style_name='3D风',
                             return_url=False, data_folder=None):
    style_mapping = {
        "3D风": ("img2img_disney_3d_style", ""),
        "写实风": ("img2img_real_mix_style", ""),
        "天使风": ("img2img_pastel_boys_style", ""),
        "动漫风": ("img2img_cartoon_style", ""),
        "日漫风": ("img2img_makoto_style", ""),
        "公主风": ("img2img_rev_animated_style", ""),
        "梦幻风": ("img2img_blueline_style", ""),
        "水墨风": ("img2img_water_ink_style", ""),
        "新莫奈花园": ("i2i_ai_create_monet", ""),
        "水彩风": ("img2img_water_paint_style", ""),
        "莫奈花园": ("img2img_comic_style", "img2img_comic_style_monet"),
        "精致美漫": ("img2img_comic_style", "img2img_comic_style_marvel"),
        "赛博机械": ("img2img_comic_style", "img2img_comic_style_future"),
        "精致韩漫": ("img2img_exquisite_style", ""),
        "国风-水墨": ("img2img_pretty_style", "img2img_pretty_style_ink"),
        "浪漫光影": ("img2img_pretty_style", "img2img_pretty_style_light"),
        "陶瓷娃娃": ("img2img_ceramics_style", ""),
        "中国红": ("img2img_chinese_style", ""),
        "丑萌粘土": ("img2img_clay_style", "img2img_clay_style_3d"),
        "可爱玩偶": ("img2img_clay_style", "img2img_clay_style_bubble"),
        "3D-游戏_Z时代": ("img2img_3d_style", "img2img_3d_style_era"),
        "动画电影": ("img2img_3d_style", "img2img_3d_style_movie"),
        "玩偶": ("img2img_3d_style", "img2img_3d_style_doll"),
        # "文生图-2.0": ("high_aes_general_v20", ''),
        "文生图-2.0Pro": ("high_aes_general_v20_L", ''),
        "文生图-2.1": ("high_aes_general_v21_L", ''),
        "角色特征保持": ("high_aes_ip_v20", ''),
        "人像融合": ('face_swap3_6', ''),  # 换脸图在前（最多三张），模板图在后（最多一张）
    }
    # inpainting涂抹消除,inpainting涂抹编辑,outpainting智能扩图
    request_body = {'req_key': style_mapping.get(style_name)[0],
                    'sub_req_key': style_mapping.get(style_name)[1],
                    'return_url': return_url  # 链接有效期为24小时
                    }
    if 'general' in request_body['req_key'] or prompt:
        request_body["prompt"] = prompt
        request_body["use_sr"] = True  # AIGC超分
        request_body["scale"] = 3.6  # 影响文本描述的程度
        request_body["seed"] = -1  # -1为不随机种子
        # request_body["use_pre_llm"] = True  #use_rephraser, prompt扩写, 对输入prompt进行扩写优化,辅助生成图片的场景下传True
    if image_urls and all(image_urls):
        request_body["image_urls"] = image_urls
    if image_data:  # 目标图片需小于 5 MB,小于4096*4096,支持JPG、JPEG、PNG格式,仅支持一张图,优先生效
        request_body["binary_data_base64"] = [base64.b64encode(image_data).decode("utf-8")]  # 输入图片base64数组
    if logo_info:
        request_body["logo_info"] = logo_info
        # {
        #     "add_logo": True,
        #     "position": 0,
        #     "language": 0,
        #     "opacity": 0.3,
        #     "logo_text_content": "这里是明水印内容"
        # }
    # 'CVSync2AsyncSubmitTask',JPCartoon
    headers, url = get_ark_signature(action='CVProcess', service='cv', host='visual.volcengineapi.com',
                                     region="cn-north-1", version="2022-08-31", http_method="POST", body=request_body,
                                     access_key_id=Config.VOLC_AK_ID_admin,
                                     secret_access_key=Config.VOLC_Secret_Key_admin,
                                     timenow=None)

    # print(headers,request_body)
    cx = get_httpx_client(time_out=Config.HTTP_TIMEOUT_SEC)
    response = await cx.post(url, json=request_body, headers=headers)
    if response.status_code != 200:
        return None, {"error": f"HTTP Error: {response.status_code},\n{response.text}"}
    response.raise_for_status()
    response_data = response.json()
    # print(response_data.keys())
    # {'code': 10000, 'data': {'1905703073': 1905703073, 'algorithm_base_resp': {'status_code': 0, 'status_message': 'Success'}, 'animeoutlineV4_16_strength_clip': 0.2, 'animeoutlineV4_16_strength_model': 0.2, 'apply_id_layer': '0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15', 'binary_data_base64': [], 'clip_skip': -2, 'cn_mode': 2, 'comfyui_cost': 4, 'controlnet_weight': 1, 'ddim_steps': 20, 'i2t_tag_text': '', 'id_weight': 0, 'image_urls': ['https://p26-aiop-sign.byteimg.com/tos-cn-i-vuqhorh59i/20241225114813D0211FF38607A612BD65-0~tplv-vuqhorh59i-image.image?rk3s=7f9e702d&x-expires=1735184899&x-signature=A1Jay2TWwSsmtwRjDGzK71gzXpg%3D'], 'long_resolution': 832, 'lora_map': {'animeoutlineV4_16': {'strength_clip': 0.2, 'strength_model': 0.2}, 'null': {'strength_clip': 0.7000000000000001, 'strength_model': 0.7000000000000001}}, 'null_strength_clip': 0.7000000000000001, 'null_strength_model': 0.7000000000000001, 'prompt': '(masterpiece), (((best quality))),light tone, sunny day, shinne,tyndall effect light， landscape in the movie of Suzume no Tojimari, daytime, meteor, aurora,', 'return_url': True, 'scale': 5, 'seed': -1, 'strength': 0.58, 'sub_prompts': ['(masterpiece), (((best quality))),light tone, sunny day, shinne,tyndall effect light， landscape in the movie of Suzume no Tojimari, daytime, meteor, aurora,'], 'sub_req_key': ''}, 'message': 'Success', 'request_id': '20241225114813D0211FF38607A612BD65', 'status': 10000, 'time_elapsed': '6.527672506s'}
    if response_data["status"] == 10000:
        image_base = response_data["data"].get("binary_data_base64", [])
        image_urls = response_data["data"].get("image_urls", [''])
        request_id = response_data["request_id"]
        if len(image_base) == 1:
            image_decode = base64.b64decode(image_base[0])
            if data_folder:
                # 将解码后的数据转换为图片
                file_path = f"{data_folder}/{request_id}.jpg"
                img = Image.open(io.BytesIO(image_decode))
                img.save(file_path)
                return file_path, {"urls": image_urls, 'id': request_id}
            return image_decode, {"urls": image_urls, 'id': request_id}
        return None, {"urls": image_urls, 'id': request_id}
    return None, response_data


async def ark_drawing_picture(image_data, image_urls: list[str], whitening: float = 1.0, dermabrasion: float = 1.2,
                              logo_info=None, style_name='3d人偶', return_url=False):
    style_mapping = {
        # 头像风格（单人、男女均支持)
        "美漫风格": "img2img_photoverse_american_comics",
        "商务证件照": "img2img_photoverse_executive_ID_photo",
        "3d人偶": "img2img_photoverse_3d_weird",
        "赛博朋克": "img2img_photoverse_cyberpunk",
        # 胸像写真风格(单人、只支持女生)
        "古堡": "img2img_xiezhen_gubao",
        "芭比牛仔": "img2img_xiezhen_babi_niuzai",
        "浴袍风格": "img2img_xiezhen_bathrobe",
        "蝴蝶机械": "img2img_xiezhen_butterfly_machine",
        "职场证件照": "img2img_xiezhen_zhichangzhengjianzhao",
        "圣诞": "img2img_xiezhen_christmas",
        "美式甜点师": "img2img_xiezhen_dessert",
        "old_money": "img2img_xiezhen_old_money",
        "最美校园": "img2img_xiezhen_school"
    }

    request_body = {'req_key': style_mapping.get(style_name),
                    'return_url': return_url,  # 链接有效期为24小时
                    "beautify_info": {"whitening": whitening,  # 自定义美白参数，float类型，数值越大，效果越明显，未做参数范围校验，建议[0, 2]
                                      "dermabrasion": dermabrasion  # 自定义磨皮参数，float类型, 数值越大，效果越明显，未做参数范围校验，建议[0, 2]
                                      }
                    }
    if image_urls and all(image_urls):
        request_body["image_urls"] = image_urls
    if image_data:  # 输入图片base64数组,仅支持一张图,优先生效
        request_body["binary_data_base64"] = [base64.b64encode(image_data).decode("utf-8")]
    if logo_info:
        request_body["logo_info"] = logo_info

    headers, url = get_ark_signature(action='HighAesSmartDrawing', service='cv', host='visual.volcengineapi.com',
                                     region="cn-north-1", version="2022-08-31", http_method="POST", body=request_body,
                                     access_key_id=Config.VOLC_AK_ID_admin,
                                     secret_access_key=Config.VOLC_Secret_Key_admin,
                                     timenow=None)

    # print(headers,request_body)
    cx = get_httpx_client(time_out=Config.HTTP_TIMEOUT_SEC)
    response = await cx.post(url, json=request_body, headers=headers)
    if response.status_code != 200:
        return None, {"error": f"HTTP Error: {response.status_code},\n{response.text}"}
    response.raise_for_status()
    response_data = response.json()
    if response_data["status"] == 10000:
        image_base = response_data["data"].get("binary_data_base64", [])
        image_urls = response_data["data"].get("image_urls", [''])
        if len(image_base) == 1:
            image_decode = base64.b64decode(image_base[0])
            return image_decode, {"urls": image_urls, 'id': response_data["request_id"]}
        return None, {"urls": image_urls, 'id': response_data["request_id"]}
    return None, response_data


# https://help.aliyun.com/zh/viapi/developer-reference/api-overview?spm=a2c4g.11186623.help-menu-142958.d_4_3_1.13e65733U2m63s
# https://help.aliyun.com/zh/viapi/developer-reference/api-version?spm=a2c4g.11186623.help-menu-142958.d_4_3_0.290f6593LRs5Lt&scm=20140722.H_464194._.OR_help-T_cn~zh-V_1
async def ali_cartoon_picture(image_url, style_name='复古漫画'):
    style_mapping = {
        "复古漫画": '0',
        "3D童话": '1',
        "二次元": '2',
        "小清新": '3',
        "未来科技": '4',
        "国画古风": '5',
        "将军百战": '6',
        "炫彩卡通": '7',
        "清雅国风": '8'
    }
    # 图片大小不超过10MB。支持的图片类型：JPEG、PNG、JPG、BMP、WEBP。
    request_body = {'Index': style_mapping.get(style_name, '0'),
                    'ImageUrl': image_url, }
    # 视觉智能开放平台各服务支持的区域为华东2（上海）
    parameters, url = get_aliyun_access_token(service="imageenhan", region="cn-shanghai",
                                              action='GenerateCartoonizedImage', http_method="POST",
                                              body=request_body, version='2019-09-30')

    print(request_body)
    cx = get_httpx_client(time_out=Config.HTTP_TIMEOUT_SEC)
    try:
        response = await cx.post(url, json=request_body)  # , headers=headers
        print(response.text)
        if response.status_code != 200:
            return {"error": f"HTTP Error: {response.status_code},\n{response.text}"}

        response.raise_for_status()
        response_data = response.json()
        request_body = {'JobId': response_data["RequestId"]}

        parameters, url = get_aliyun_access_token(service="imageenhan", region="cn-shanghai",
                                                  action='GetAsyncJobResult', http_method="POST",
                                                  body=request_body, version='2019-09-30')
        response = await cx.post(url, json=request_body)  # , headers=headers
        # response_data["Data"].get("ResultUrl")
        print(response.text)
        # 图片链接非法，请检查图片链接是否可访问 ,图片链接地域不对，请参考https://help.aliyun.com/document_detail/155645.html - imageUrl is invalid region oss url
        if response.status_code != 200:
            return {"error": f"HTTP Error: {response.status_code},\n{response.text}"}
        response_data = response.json()
        if response_data["Data"]["Status"] == "PROCESS_SUCCESS":
            image_url = response_data["Data"]["Result"].get("ImageUrls")
            return {"urls": image_url, 'id': response_data["RequestId"]}

        return {"error": "PROCESS"}
    except httpx.ConnectError as e:
        return {'error': f"Connection error: {e},Request URL: {url},Request body: {request_body}"}


# https://cloud.tencent.com/document/product/1668/88066
# https://cloud.tencent.com/document/product/1668/107799
async def tencent_drawing_picture(image_data, image_url: str = '', prompt: str = '', negative_prompt: str = '',
                                  style_name='日系动漫', return_url=False):
    # 单边分辨率小于5000且大于50，转成 Base64 字符串后小于 8MB，格式支持 jpg、jpeg、png、bmp、tiff、webp。
    style_mapping = {
        "水彩画": '104',
        "卡通插画": '107',
        "3D 卡通": '116',
        "日系动漫": '201',
        "唯美古风": '203',
        "2.5D 动画": '210',
        "木雕": '120',
        "黏土": '121',
        "清新日漫": '123',
        "小人书插画": '124',
        "国风工笔": '125',
        "玉石": '126',
        "瓷器": '127',
        "毛毡（亚洲版）": '135',
        "毛毡（欧美版）": '128',
        "美式复古": '129',
        "蒸汽朋克": '130',
        "赛博朋克": '131',
        "素描": '132',
        "莫奈花园": '133',
        "厚涂手绘": '134',
        "复古繁花": "flower",
        "芭比": "babi",
        "白领精英": "commerce",
        "婚纱日记": "wedding",
        "醉梦红尘": "gufeng",
        "暴富": "coin",
        "夏日水镜": "water",
        "复古港漫": "retro",
        "游乐场": "amusement",
        "宇航员": "astronaut",
        "休闲时刻": "cartoon",
        "回到童年": "star",
        "多巴胺": "dopamine",
        "心动初夏": "comic",
        "夏日沙滩": "beach"
    }

    style_type = style_mapping.get(style_name, '201')
    if style_type.isdigit():
        action = 'ImageToImage'  # 图像风格化
        payload = {'Strength': 0.6,  # 生成自由度(0, 1]
                   'EnhanceImage': 1,  # 画质增强开关
                   'RestoreFace': 1,  # 细节优化的面部数量上限，支持0 ~ 6，默认为0。
                   'RspImgType': 'url' if return_url else 'base64',
                   'Styles': [style_type],
                   'LogoAdd': 0
                   # 'ResultConfig': {"Resolution": "768:768"},  # origin
                   }
        if prompt:
            payload["Prompt"] = prompt
        if negative_prompt:
            payload["NegativePrompt"] = negative_prompt
    else:
        action = 'GenerateAvatar'  # 百变头像
        payload = {'RspImgType': 'url' if return_url else 'base64',
                   'Style': style_type,
                   'Type': 'human',  # pet,图像类型
                   'Filter': 1,  # 人像图的质量检测开关，默认开启，仅在人像模式下生效。
                   'LogoAdd': 0
                   }
    if image_data:
        payload["InputImage"] = base64.b64encode(image_data).decode("utf-8")
    if image_url:
        payload["InputUrl"] = image_url

    url = "https://aiart.tencentcloudapi.com"
    headers = get_tencent_signature(service="aiart", host="aiart.tencentcloudapi.com", body=payload,
                                    action=action, timestamp=int(time.time()), region="ap-shanghai",
                                    version='2022-12-29')

    cx = get_httpx_client(time_out=Config.HTTP_TIMEOUT_SEC)
    response = await cx.post(url, headers=headers, json=payload)
    response.raise_for_status()
    if response.status_code != 200:
        return None, {'error': f'{response.status_code},Request failed,Response content: {response.text}'}

    response_data = response.json()["Response"]
    if return_url:
        return None, {"urls": [response_data["ResultImage"]], 'id': response_data["RequestId"]}

    image_decode = base64.b64decode(response_data["ResultImage"])
    return image_decode, {"urls": [''], 'id': response_data["RequestId"]}


# https://cloud.tencent.com/document/product/1729/108738
async def tencent_generate_image(prompt: str = '', negative_prompt: str = '', style_name='不限定风格',
                                 return_url=True):
    style_mapping = {
        "默认": "000",
        "不限定风格": "000",
        "水墨画": "101",
        "概念艺术": "102",
        "油画1": "103",
        "油画2（梵高）": "118",
        "水彩画": "104",
        "像素画": "105",
        "厚涂风格": "106",
        "插图": "107",
        "剪纸风格": "108",
        "印象派1（莫奈）": "109",
        "印象派2": "119",
        "2.5D": "110",
        "古典肖像画": "111",
        "黑白素描画": "112",
        "赛博朋克": "113",
        "科幻风格": "114",
        "暗黑风格": "115",
        "3D": "116",
        "蒸汽波": "117",
        "日系动漫": "201",
        "怪兽风格": "202",
        "唯美古风": "203",
        "复古动漫": "204",
        "游戏卡通手绘": "301",
        "通用写实风格": "401"
    }
    payload = {'Style': style_mapping.get(style_name, '000'),
               'Prompt': prompt,
               'RspImgType': 'url' if return_url else 'base64',
               'LogoAdd': 0,
               "Resolution": "1024:1024",  # origin
               }

    if negative_prompt:
        payload["NegativePrompt"] = negative_prompt

    url = "https://hunyuan.tencentcloudapi.com"
    headers = get_tencent_signature(service="hunyuan", host="hunyuan.tencentcloudapi.com", body=payload,
                                    action='TextToImageLite', timestamp=int(time.time()), region="ap-guangzhou",
                                    version='2023-09-01')

    cx = get_httpx_client(time_out=Config.HTTP_TIMEOUT_SEC)
    response = await cx.post(url, headers=headers, json=payload)
    response.raise_for_status()
    if response.status_code != 200:
        return None, {'error': f'{response.status_code},Request failed,Response content: {response.text}'}

    response_data = response.json()["Response"]
    if return_url:
        return None, {"urls": [response_data["ResultImage"]], 'id': response_data["RequestId"]}

    image_decode = base64.b64decode(response_data["ResultImage"])
    return image_decode, {"urls": [''], 'id': response_data["RequestId"]}


# https://ai.baidu.com/ai-doc/OCR/Ek3h7y961,  https://aip.baidubce.com/rest/2.0/solution/v1/iocr/recognise"
# https://console.bce.baidu.com/ai/#/ai/ocr/overview/index
async def baidu_ocr_recognise(image_data, image_url, ocr_type='accurate_basic'):
    '''
    general:通用文字识别(含位置)
    accurate:通用文字识别(高进度含位置)
    accurate_basic:通用文字识别（高进度）
    general_basic:通用文字识别
    doc_analysis_office:办公文档识别
    idcard:身份证识别
    table:表格文字识别
    numbers:数字识别
    qrcode:二维码识别
    account_opening:开户许可证识别
    handwriting:手写文字识别
    webimage:
    '''
    headers = {
        'Content-Type': "application/x-www-form-urlencoded",
        'Accept': 'application/json',
        # 'charset': "utf-8",
    }
    url = f"https://aip.baidubce.com/rest/2.0/ocr/v1/{ocr_type}"
    access_token = get_baidu_access_token(Config.BAIDU_ocr_API_Key, Config.BAIDU_ocr_Secret_Key)
    params = {
        "access_token": access_token,
        "language_type": 'CHN_ENG',
    }
    if image_url:
        params["url"] = image_url
    if image_data:
        params["image"] = base64.b64encode(image_data).decode("utf-8")
    # 将图像数据编码为base64
    # image_b64 = base64.b64encode(image_data).decode().replace("\r", "")
    # if template_sign:
    #     params["templateSign"] = template_sign
    # if classifier_id:
    #     params["classifierId"] = classifier_id
    # # 请求模板的bodys
    # recognise_bodys = "access_token=" + access_token + "&templateSign=" + template_sign + "&image=" + quote(image_b64.encode("utf8"))
    # # 请求分类器的bodys
    # classifier_bodys = "access_token=" + access_token + "&classifierId=" + classifier_id + "&image=" + quote(image_b64.encode("utf8"))
    # request_body = "&".join(f"{key}={value}" for key, value in params.items())
    cx = get_httpx_client(time_out=Config.HTTP_TIMEOUT_SEC)
    response = await cx.post(url, headers=headers, data=params)
    response.raise_for_status()
    data = response.json()
    for key in ['result', 'results', 'error_msg']:
        if key in data:
            return data[key]
    return data


# https://cloud.tencent.com/document/product/866/36210
async def tencent_ocr_recognise(image_data, image_url, ocr_type='GeneralBasicOCR'):
    '''
    GeneralBasicOCR:通用印刷体识别,TextDetections
    RecognizeTableDDSNOCR: 表格识别,TableDetections
    RecognizeGeneralTextImageWarn:证件有效性检测告警
    GeneralAccurateOCR:通用印刷体识别（高精度版）
    VatInvoiceOCR:增值税发票识别
    VatInvoiceVerifyNew:增值税发票核验
    ImageEnhancement:文本图像增强,包括切边增强、图像矫正、阴影去除、摩尔纹去除等；
    QrcodeOCR:条形码和二维码的识别
    SmartStructuralOCRV2:智能结构化识别,智能提取各类证照、票据、表单、合同等结构化场景的key:value字段信息
    '''
    url = 'https://ocr.tencentcloudapi.com'
    host = url.split("//")[-1]
    payload = {
        # 'Action': ocr_type,
        # 'Version': '2018-11-19'
        # 'Region': 'ap-shanghai',
        # 'ImageBase64': '',
        # 'ImageUrl': image_url,
    }
    if image_url:
        payload['ImageUrl'] = image_url
    else:
        if isinstance(image_data, bytes):
            payload['ImageBase64'] = base64.b64encode(image_data).decode("utf-8")
        else:
            payload['ImageBase64'] = base64.b64encode(image_data)

    headers = get_tencent_signature('ocr', host, body=payload, action=ocr_type,
                                    secret_id=Config.TENCENT_SecretId, secret_key=Config.TENCENT_Secret_Key,
                                    version='2018-11-19')

    # payload = convert_keys_to_pascal_case(params)
    cx = get_httpx_client(time_out=Config.HTTP_TIMEOUT_SEC)

    response = await cx.post(url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    return data["Response"]  # "TextDetections"


# https://help.aliyun.com/zh/ocr/developer-reference/api-ocr-api-2021-07-07-dir/?spm=a2c4g.11186623.help-menu-252763.d_2_2_4.3aba47bauq0U2j
async def ali_ocr_recognise(image_data, image_url, ocr_type='Advanced'):
    # accurate,general_basic,webimage
    # RecognizeAllText
    url = 'https://ocr-api.cn-hangzhou.aliyuncs.com'
    token, _ = get_aliyun_access_token(service="ocr-api", region="cn-hangzhou", access_key_id=Config.ALIYUN_AK_ID,
                                       access_key_secret=Config.ALIYUN_Secret_Key, version='2021-07-07')
    headers = {
        'Content-Type': "application/x-www-form-urlencoded",
        'charset': "utf-8"
    }
    # 将图像数据编码为base64
    # image_b64 = base64.b64encode(image_data).decode().replace("\r", "")
    params = {
        "access_token": token,
        "language_type": 'CHN_ENG',

        'Type': ocr_type,  # Advanced,HandWriting,General,Table,GeneralStructure
        'PageNo': 1,
        'OutputRow': False,
        'OutputParagraph': False,
        'OutputKVExcel': False,
        'OutputTableHtml': False,
    }
    if image_data:
        params["body"] = base64.b64encode(image_data)  # quote(image_b64.encode("utf8"))
    if url:
        params["Url"] = image_url

    try:
        # if template_sign:
        #     params["templateSign"] = template_sign
        # if classifier_id:
        #     params["classifierId"] = classifier_id
        # # 请求模板的bodys
        # recognise_bodys = "access_token=" + access_token + "&templateSign=" + template_sign + "&image=" + quote(image_b64.encode("utf8"))
        # # 请求分类器的bodys
        # classifier_bodys = "access_token=" + access_token + "&classifierId=" + classifier_id + "&image=" + quote(image_b64.encode("utf8"))
        # request_body = "&".join(f"{key}={value}" for key, value in params.items())
        cx = get_httpx_client(time_out=Config.HTTP_TIMEOUT_SEC)
        response = await cx.post(url, headers=headers, json=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(e)
        # data=params ➔ 发送成表单 (Content-Type: application/x-www-form-urlencoded)
        response = requests.post(url, data=params, headers=headers, timeout=Config.HTTP_TIMEOUT_SEC)
        response.raise_for_status()
        return response.json()


# https://bailian.console.aliyun.com/?spm=5176.28197581.0.0.2e2d29a4n0Mukq#/model-market/detail/wanx-v1?tabKey=sdk
def dashscope_image_call(prompt: str, negative_prompt: str = '', image_url: str = '', style_name="默认",
                         model_name="wanx-v1", data_folder=None):
    from pathlib import PurePosixPath
    style_mapping = {
        "默认": "<auto>",
        "摄影": "<photography>",
        "人像写真": "<portrait>",
        "3D卡通": "<3d cartoon>",
        "动画": "<anime>",
        "油画": "<oil painting>",
        "水彩": "<watercolor>",
        "素描": "<sketch>",
        "中国画": "<chinese painting>",
        "扁平插画": "<flat illustration>"
    }
    style = style_mapping.get(style_name, "<auto>")
    rsp = dashscope.ImageSynthesis.call(model=model_name,  # "stable-diffusion-3.5-large"
                                        api_key=Config.Bailian_Service_Key,
                                        prompt=prompt, negative_prompt=negative_prompt, ref_img=image_url,
                                        n=1, size='1024*1024', style=style)

    # ref_strength：控制输出图像与参考图（垫图）的相似度。取值范围为[0.0, 1.0]。取值越大，代表生成的图像与参考图越相似。
    # ref_mode：基于参考图（垫图）生成图像的方式。取值有：repaint代表参考内容，为默认值；refonly代表参考风格。
    if rsp.status_code == HTTPStatus.OK:
        print(rsp)
        # 保存图片到当前文件夹
        image_urls = [result.url for result in rsp.output.results]
        if data_folder:
            image_path = []
            for result in rsp.output.results:
                file_name = PurePosixPath(unquote(urlparse(result.url).path)).parts[-1]
                file_path = f'{data_folder}/%s' % file_name
                with open(file_path, 'wb+') as f:
                    f.write(requests.get(result.url).content)
                image_path.append(file_path)

            return image_path, {"urls": image_urls, 'id': rsp.request_id}
        return requests.get(image_urls[0]).content, {"urls": image_urls, 'id': rsp.request_id}
    return None, {"error": 'Failed, status_code: %s, code: %s, message: %s' % (rsp.status_code, rsp.code, rsp.message)}


# https://help.aliyun.com/zh/model-studio/user-guide/cosplay-anime-character-generation?spm=0.0.0.i1
# https://help.aliyun.com/zh/model-studio/developer-reference/portrait-style-redraw-api-reference?spm=a2c4g.11186623.help-menu-2400256.d_3_3_2_1.3e2f56e5BtF0ok
async def wanx_image_generation(image_urls, style_name="复古漫画",
                                api_key=Config.DashScope_Service_Key, interval: int = 3, timeout: int = 60):
    # JPEG，PNG，JPG，BMP，WEB,不超过10M,不小于256*256，不超过5760*3240, 长宽比不超过2:1
    style_mapping = {
        "参考上传图像风格": -1,
        "复古漫画": 0,
        "3D童话": 1,
        "二次元": 2,
        "小清新": 3,
        "未来科技": 4,
        "国画古风": 5,
        "将军百战": 6,
        "炫彩卡通": 7,
        "清雅国风": 8,
        "喜迎新年": 9
    }
    if style_name == 'Cosplay动漫人物':
        model_name = "wanx-style-cosplay-v1"
        input_params = {
            "model_index": 1,
            "face_image_url": image_urls[0],
            "template_image_url": image_urls[1],
        }
    elif len(image_urls) > 1:
        model_name = "wanx-style-repaint-v1"
        input_params = {
            "style_index": -1,
            "image_url": image_urls[0],
            'style_ref_url': image_urls[1]
        }
    else:  # '人像风格重绘'
        model_name = "wanx-style-repaint-v1"
        input_params = {
            "style_index": style_mapping.get(style_name, 0),
            "image_url": image_urls[0],
        }

    url = 'https://dashscope.aliyuncs.com/api/v1/services/aigc/image-generation/generation'
    headers = {
        'Content-Type': 'application/json',
        "Authorization": f'Bearer {api_key}',
        'X-DashScope-Async': 'enable'  # 使用异步方式提交作业
    }
    task_headers = {"Authorization": f'Bearer {api_key}'}
    body = {
        "model": model_name,
        "input": input_params,
        # "parameters": {
        #     "style": "<auto>",
        #     "size": "1024*1024",
        #     "n": 1
        # }
    }
    cx = get_httpx_client(time_out=Config.HTTP_TIMEOUT_SEC)
    response = await cx.post(url, headers=headers, json=body)
    response.raise_for_status()
    data = response.json()
    task_id = data["output"]["task_id"]

    polling_func = async_polling_check(interval=interval, timeout=timeout)(query_aliyun_task_once)
    future, handle = await polling_func(cx, task_id, task_headers)
    try:
        result = await future  # 等待轮询完成
    except TimeoutError:
        return {"status": "timeout", "id": task_id}
    except Exception as e:
        return {"status": "error", "error": str(e), "id": task_id}
    finally:
        handle['cancelled'] = True

    return result


async def query_aliyun_task_once(httpx_client, task_id: str, task_headers: dict, _future=None, _handle=None):
    """
    单次查询 Aliyun 任务状态
    返回：
        - dict -> 任务完成结果
        - False -> 任务未完成，继续轮询
        - dict(error=...) -> 任务失败或异常
    """
    task_url = f'https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}'
    response = await httpx_client.get(task_url, headers=task_headers)
    resp = response.json()
    task_status = resp["output"]["task_status"]
    # "task_status":"PENDING""RUNNING","SUCCEEDED"->"results", "FAILED"->"message"
    if task_status == "SUCCEEDED":
        urls = [item['url'] for item in resp['output'].get('results', []) if 'url' in item]
        result = {"urls": urls or [resp['output'].get('result_url')], 'id': task_id, 'status': task_status.lower()}
        if urls:
            image_response = await httpx_client.get(urls[0])
            return image_response.content, result
        return None, result

    if task_status == "FAILED":
        error = resp['output'].get('message', 'Task failed')
        print(error)
        return None, {"urls": [], 'id': task_id, 'status': task_status.lower(), "error": error}

    # 其他状态，如 PENDING / RUNNING
    return False


# https://nls-portal.console.aliyun.com/overview
async def ali_speech_to_text(audio_data, format='pcm', rate=16000):
    """阿里云语音转文字"""
    params = {
        "appkey": Config.ALIYUN_nls_AppId,
        "format": format,  # 也可以传入其他格式，如 wav, mp3
        "sample_rate": rate,  # 音频采样率
        "version": "4.0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "SignatureMethod": "HMAC-SHA1",
        "SignatureVersion": "1.0",
        "SignatureNonce": str(uuid.uuid4())
    }
    signature = generate_hmac_signature(Config.ALIYUN_Secret_Key, "POST", params)
    params["signature"] = signature
    token, _ = get_aliyun_access_token(service="nls-meta", region="cn-shanghai", action='CreateToken',
                                       http_method="GET",
                                       access_key_id=Config.ALIYUN_AK_ID, access_key_secret=Config.ALIYUN_Secret_Key)
    if not token:
        print("No permission!")

    headers = {
        "Authorization": f"Bearer {Config.ALIYUN_AK_ID}",
        # "Content-Type": "audio/pcm",
        "Content-Type": "application/octet-stream",
        "X-NLS-Token": token,
    }

    # host = 'nls-gateway-cn-shanghai.aliyuncs.com'
    # conn = http.client.HTTPSConnection(host)
    # http://nls-meta.cn-shanghai.aliyuncs.com/
    # "wss://nls-gateway-cn-shanghai.aliyuncs.com/ws/v1"
    url = "https://nls-gateway.cn-shanghai.aliyuncs.com/stream/v1/asr"
    cx = get_httpx_client(time_out=Config.HTTP_TIMEOUT_SEC)
    response = await cx.post(url, headers=headers, params=params, data=audio_data.getvalue())

    result = response.json()
    if result.get("status") == 20000000:  # "SUCCESS":
        return {"text": result.get("result")}

    return {"error": result.get('message')}


# https://cloud.tencent.com/document/api/1093/37823
async def tencent_speech_to_text(audio_data, audio_url: str = None, interval: int = 3, timeout: int = 300) -> dict:
    '''
    本接口支持音频 URL 、本地音频文件两种请求方式,音频 URL 时长不能大于5小时，文件大小不超过1GB；本地音频文件不能大于5MB
    '''
    url = 'https://asr.tencentcloudapi.com'
    host = url.split("//")[-1]
    payload = {
        "ChannelNum": 1,
        "EngineModelType": "16k_zh",  # 8k_zh_large：中文电话场景专用大模型引擎【大模型版】
        "ResTextFormat": 0,  # 基础识别结果0-5
        "SourceType": 1,  # 音频数据来源
        # "CallbackUrl":#回调 URL
    }
    if audio_url:
        payload['Url'] = audio_url
        payload['SourceType'] = 0
    else:
        if isinstance(audio_data, bytes):
            pass
        elif isinstance(audio_data, io.BytesIO):
            audio_data = audio_data.getvalue()  # await audio_data.read()  # 读取二进制数据
        payload['Data'] = base64.b64encode(audio_data).decode("utf-8")

    headers = get_tencent_signature('asr', host, body=payload, action='CreateRecTask',
                                    secret_id=Config.TENCENT_SecretId, secret_key=Config.TENCENT_Secret_Key,
                                    version='2019-06-14')
    # payload = convert_keys_to_pascal_case(params)
    cx = get_httpx_client(time_out=Config.HTTP_TIMEOUT_SEC)
    response = await cx.post(url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    error = data["Response"].get("Error")
    if error:
        print(data)
        return {"error": error}

    response_data = data["Response"].get("Data")  # task_id,TaskId有效期为24小时
    if not response_data:
        return {"text": response_data, "error": 'no task_id'}

    body = response_data.copy()
    headers = get_tencent_signature('asr', host, body=body, action='DescribeTaskStatus',
                                    secret_id=Config.TENCENT_SecretId, secret_key=Config.TENCENT_Secret_Key,
                                    version='2019-06-14')
    task_id = response_data['TaskId']

    async def check_task_status(_future=None, _handle=None):
        response = await cx.post(url, headers=headers, json=body)
        response.raise_for_status()
        data = response.json()
        error = data["Response"].get("Error")
        if error:
            return {"error": error, "task_id": task_id}

        response_data = data["Response"].get("Data")
        status = response_data['Status']

        if status == 2:  # 完成
            return {"text": response_data['Result'], "task_id": task_id}  # 最终文本结果字段

        if status == 3:  # 失败
            return {"error": response_data['ErrorMsg'], "task_id": task_id}

        if interval <= 0:
            return response_data
        # 其他状态，任务未完成
        return False

    check_func = async_polling_check(interval=interval, timeout=timeout)(check_task_status)
    future, handle = await check_func()

    try:
        result = await future
    except TimeoutError:
        result = {"error": "timeout", "task_id": task_id}
    except Exception as e:
        result = {"error": str(e), "task_id": task_id}
    finally:
        handle["cancelled"] = True

    return result or response_data


# https://github.com/TencentCloud/tencentcloud-sdk-python/blob/master/tencentcloud/asr/v20190614/asr_client.py
def tencent_sdk_speech_to_text(audio_data, audio_url: str = None, poll_interval=3, timeout: int = 300):
    '''
    本接口支持音频 URL 、本地音频文件两种请求方式
    • 返回时效：异步回调，非实时返回。最长3小时返回识别结果，**大多数情况下，1小时的音频1-3分钟即可完成识别**。请注意：上述返回时长不含音频下载时延，且30分钟内发送超过1000小时录音或2万条任务的情况除外
    • 音频格式：wav、mp3、m4a、flv、mp4、wma、3gp、amr、aac、ogg-opus、flac
    • 音频限制：音频 URL 时长不能大于5小时，文件大小不超过1GB；本地音频文件不能大于5MB
    • 识别结果有效时间：识别结果在服务端保存24小时
    '''
    payload = {
        "ChannelNum": 1,
        "EngineModelType": "16k_zh",  # 8k_zh_large：中文电话场景专用大模型引擎【大模型版】
        "ResTextFormat": 0,  # 基础识别结果0-5
        "SourceType": 1,  # 音频数据来源
        # "CallbackUrl":#回调 URL
    }
    if audio_url:
        payload['Url'] = audio_url
        payload['SourceType'] = 0
    else:
        if isinstance(audio_data, bytes):
            pass
        elif isinstance(audio_data, io.BytesIO):
            audio_data = audio_data.getvalue()  # await audio_data.read()  # 读取二进制数据
        payload['Data'] = base64.b64encode(audio_data).decode("utf-8")

    from tencentcloud.common import credential
    from tencentcloud.asr.v20190614 import models
    from tencentcloud.asr.v20190614.asr_client import AsrClient
    secret_id = Config.TENCENT_SecretId
    secret_key = Config.TENCENT_Secret_Key
    cred = credential.Credential(secret_id, secret_key)
    client = AsrClient(cred, "ap-shanghai")

    # 1) 创建任务
    req = models.CreateRecTaskRequest()
    req.from_json_string(json.dumps(payload))
    resp = client.CreateRecTask(req)
    task_id = resp.Data.TaskId

    # 2) 轮询任务结果
    status_req = models.DescribeTaskStatusRequest()
    status_req.TaskId = task_id
    start = time.time()
    while True:
        result = client.DescribeTaskStatus(status_req)

        status = result.Data.Status  # 状态码
        # 状态说明：
        # 0 = 初始化
        # 1 = 识别中
        # 2 = 识别完成
        # 3 = 识别失败
        if status == 2:  # 完成
            return {
                "TaskId": task_id,
                "Text": result.Data.Result,  # 最终文本结果字段
                "Message": "Success"
            }

        if status == 3:  # 失败
            return {
                "TaskId": task_id,
                "Error": result.Data.ErrorMsg,
                "Message": "Failed"
            }

        if time.time() - start > timeout:
            return {
                "TaskId": task_id,
                "Message": "Timeout"
            }

        time.sleep(poll_interval)  # 默认接口请求频率限制：50次/秒


# https://console.bce.baidu.com/ai/#/ai/speech/overview/index
async def baidu_speech_to_text(audio_data, format='pcm', rate: int = 16000, dev_pid=1536):  #: io.BytesIO
    """
    dev_pid	语言	模型	是否有标点	备注
    1537	普通话(纯中文识别)	语音近场识别模型	有标点	支持自定义词库，适合日常通用场景
    1737	英语	英语模型	无标点	-
    1637	粤语	粤语模型	有标点	-
    1837	四川话	四川话模型	有标点	-
    1536	普通话(支持简单英文)	搜索模型	无标点	支持自定义词库，更适合搜索类语句
    1936	适用于粤语	远场模型	有标点	适用于远场拾音设备
    audio/pcm pcm（不压缩）、wav（不压缩，pcm编码）、amr（压缩格式）、m4a（压缩格式）
    """
    url = "https://vop.baidu.com/server_api"  # 'https://vop.baidu.com/pro_api'
    access_token = get_baidu_access_token(Config.BAIDU_speech_API_Key, Config.BAIDU_speech_Secret_Key)
    # Config.BAIDU_speech_AppId
    url = f"{url}?dev_pid={dev_pid}&cuid={Config.DEVICE_ID}&token={access_token}"
    headers = {'Content-Type': f'audio/{format}; rate={rate}'}

    cx = get_httpx_client(time_out=Config.HTTP_TIMEOUT_SEC)
    response = await cx.post(url, headers=headers, data=audio_data.getvalue())
    result = response.json()
    if result.get("err_no") == 0:
        return {"text": result.get("result")[0]}

    return {"error": result.get('err_msg')}


# Paraformer语音识别API基于通义实验室新一代非自回归端到端模型，提供基于实时音频流的语音识别以及对输入的各类音视频文件进行语音识别的能力。可被应用于：
# 对语音识别结果返回的即时性有严格要求的实时场景，如实时会议记录、实时直播字幕、电话客服等。
# 对音视频文件中语音内容的识别，从而进行内容理解分析、字幕生成等。
# 对电话客服呼叫中心录音进行识别，从而进行客服质检等
async def dashscope_speech_to_text(audio_path, format='wav', language: list[str] = ['zh', 'en'], rate=16000):
    recognition = Recognition(model='paraformer-realtime-v2', format=format, sample_rate=rate,
                              language_hints=language, callback=RecognitionCallback())  # None
    result = await asyncio.to_thread(recognition.call, audio_path)  # recognition.call(audio_path)
    if result.status_code == 200:
        texts = [sentence.get('text', '') for sentence in result.get_sentence()]
        return {"text": texts[0]}

    return {"error": result.message}


# SenseVoice语音识别大模型专注于高精度多语言语音识别、情感辨识和音频事件检测，支持超过50种语言的识别，整体效果优于Whisper模型，中文与粤语识别准确率相对提升在50%以上。
# SenseVoice语音识别提供的文件转写API，能够对常见的音频或音视频文件进行语音识别，并将结果返回给调用者。
# SenseVoice语音识别返回较为丰富的结果供调用者选择使用，包括全文级文字、句子级文字、词、时间戳、语音情绪和音频事件等。模型默认进行标点符号预测和逆文本正则化。
async def dashscope_speech_to_text_url(file_urls: list[str], model='paraformer-v1',
                                       language: list[str] = ['zh', 'en']) -> list:
    task_response = Transcription.async_call(
        model=model,  # paraformer-8k-v1, paraformer-mtl-v1
        file_urls=file_urls, language_hints=language)

    transcribe_response = Transcription.wait(task=task_response.output.task_id)
    transcription_texts = []
    for r in transcribe_response.output["results"]:
        if r["subtask_status"] == "SUCCEEDED":
            cx = get_httpx_client(time_out=Config.HTTP_TIMEOUT_SEC)
            response = await cx.get(r["transcription_url"])
            if response.status_code == 200:
                transcription_data = response.json()
                if len(transcription_data["transcripts"]) > 0:
                    transcription_texts.append({"file_url": transcription_data["file_url"],
                                                "transcripts": transcription_data["transcripts"][0]['text']
                                                })
                    # transcription_data["transcripts"][0]["sentences"][0]["text"]
                else:
                    print(f"No transcription text found in the response.Transcription Result: {response.text}")
            else:
                print(f"Failed to fetch transcription. Status code: {response.status_code}")
        else:
            print(f"Subtask status: {r['subtask_status']},id: {task_response.output.task_id}")
            transcription_texts.append(r['subtask_status'].get('results', [])[0])

    if len(file_urls) != len(transcription_texts):
        print(json.dumps(transcribe_response.output, indent=4, ensure_ascii=False))

    return transcription_texts


# 非流式合成
async def dashscope_text_to_speech(sentences, model="cosyvoice-v1", voice="longxiaochun"):
    synthesizer = dashscope.audio.tts_v2.SpeechSynthesizer(model=model, voice=voice)
    audio_data = synthesizer.call(sentences)  # sample_rate=48000
    return audio_data, synthesizer.get_last_request_id()

    # SpeechSynthesizer.call(model='sambert-zhichu-v1',
    #                        text='今天天气怎么样',
    #                        sample_rate=48000,
    #                        format='pcm',
    #                        callback=callback)
    # if result.get_audio_data() is not None:


if __name__ == "__main__":
    # https://nls-portal-service.aliyun.com/ptts?p=eyJleHBpcmF0aW9uIjoiMjAyNC0wOS0wNlQwOToxNjoyNy42MDRaIiwiY29uZGl0aW9ucyI6W1sic3RhcnRzLXdpdGgiLCIka2V5IiwidHRwLzEzODE0NTkxNjIwMDc4MjIiXV19&s=k4sDIZ4lCmUiQ%2BV%2FcTEnFteey54%3D&e=1725614187&d=ttp%2F1381459162007822&a=LTAIiIg37IN8xeMa&h=https%3A%2F%2Ftuatara-cn-shanghai.oss-cn-shanghai.aliyuncs.com&u=qnKV1N8muiAIFiL22JTrgdYExxHS%2BPSxccg9VPiL0Nc%3D
    fileLink = "https://gw.alipayobjects.com/os/bmw-prod/0574ee2e-f494-45a5-820f-63aee583045a.wav"
    import asyncio

    Config.load('../config.yaml')

    dashscope.api_key = Config.DashScope_Service_Key
    file_urls = ['https://dashscope.oss-cn-beijing.aliyuncs.com/samples/audio/paraformer/hello_world_female.wav',
                 'https://dashscope.oss-cn-beijing.aliyuncs.com/samples/audio/paraformer/hello_world_male.wav'
                 ]
    # r = requests.get(
    #     'https://dashscope.oss-cn-beijing.aliyuncs.com/samples/audio/paraformer/hello_world_female2.wav'
    # )
    # with open('asr_example.wav', 'wb') as f:
    #     f.write(r.content)

    # asyncio.run(dashscope_speech_to_text_url(file_urls))

    task_ids = "8d47c5d9-06bb-47aa-8986-1df91b6c8dd2"
    audio_file = 'data/nls-sample-16k.wav'


    # with open(audio_file, 'rb') as f:
    #     audio_data = io.BytesIO(f.read())

    # print(dashscope_speech_to_text('data/nls-sample-16k.wav'))

    # fetch()调用不会阻塞，将立即返回所查询任务的状态和结果
    # transcribe_response = dashscope.audio.asr.Transcription.fetch(task=task_id)
    # print(json.dumps(transcribe_response.output, indent=4, ensure_ascii=False))
    # for r in transcribe_response.output["results"]:
    #     if r["subtask_status"] == "SUCCEEDED":
    #         url = r["transcription_url"]
    #         response = requests.get(url)
    #         if response.status_code == 200:
    #             transcription_data = response.text  # 可以使用 response.json() 来处理 JSON 响应
    #             print(f"Transcription Result: {transcription_data}")
    #             data = response.json()
    #             print(data["transcripts"][0]['text'])
    #         else:
    #             print(f"Failed to fetch transcription. Status code: {response.status_code}")
    #     else:
    #         print(f"Subtask status: {r['subtask_status']}")

    async def test():
        # audio_file = 'data/nls-sample-16k.wav'
        # with open(audio_file, 'rb') as f:
        #     audio_data = io.BytesIO(f.read())
        # result = await  baidu_speech_to_text(audio_data, 'wav')  # ali_speech_to_text(audio_data,'wav')
        url = 'https://idea-aigc-images.oss-cn-hangzhou.aliyuncs.com/upload%2F11389284145.mp3?OSSAccessKeyId=LTAI5tGtSJt2pLhof7xj2qjM&Expires=1762846727&Signature=Wyu4tmTwKI3HkhIj4aZxPO%2Fvc%2FA%3D'
        result = await tencent_speech_to_text(None, url)
        print(result)


    asyncio.run(test())


    def test_speech():
        url = 'https://idea-aigc-images.oss-cn-hangzhou.aliyuncs.com/upload%2F11389284145.mp3?OSSAccessKeyId=LTAI5tGtSJt2pLhof7xj2qjM&Expires=1762846727&Signature=Wyu4tmTwKI3HkhIj4aZxPO%2Fvc%2FA%3D'
        result = tencent_sdk_speech_to_text(None, url)
        print(result)
        # {'TaskId': 13539536533, 'Text': '[0:1.980,0:3.200]  嗯。\n[0:5.800,0:30.080]  哎哎，好，可以了，听得到吗？听得到听得到吗？听得到，今天是2025年9月11号，现在是来做那个云南创投会网络科技有限公司经开分公司的上门核实，我是昆明东海营业部小学，很容一。\n[1:32.560,1:35.680]  这里是他们的临时办公室。\n[1:41.240,1:53.160]  啊，我们现在是跟这个云南汕头这彩网络科技有限公司，金山分公司的王冠林王总，还有我们的财务人员一起做一个上门的核实。\n[1:53.380,1:55.980]  先看一下身份证。\n[1:59.800,2:8.700]  有效期是正常的，然后这个是昆明市公安局西山分局的一个签发机关。\n[2:10.100,2:13.320]  看一下他们的这个营业执照。\n[2:13.800,2:15.800]  你看。\n[2:15.800,2:17.160]  好。\n[2:23.620,2:46.180]  王总，我们现在先跟您做一个上面的核实。首先的话，因为你们这个分公司，这一次是打算在我们曲靖市商业银行昆民分行开一个基本账户是吗？对，然后公司呢，是呃，我看是8月份成立的，然后基于你的这个这个公司，它主要的一个经营一块是什么呢？\n[2:46.180,3:46.500]  嗯，其实我们分公司当时打算成立，就是想成立在这个汽配城这边。呃，因为我们有一些上下游的关系都在这一边，所以到时候我们分公司等这个组织健全了以后的话，开始运作，主要的是运作汽配这一块的生意，所以做这一个打算，所以设立也设立在这一边，就主要这个分公司是想用来做这个汽配的。对对对，现在的话呢，又因为有一些变化，我看现在是要全部又要换来现在现的新的这个位置，就是采用北队这边啊呃是基于什么考虑呢？呃因为那边的呃呃总公司的那边的场地有一些这个场地，还有这个租租租用方面的一些呃变动吧，所以就打算总公司也搬到这边来。那么既然都在一起了，我们就打算把它整合在一起了啊那这一块场地请问一下你。\n[3:46.500,4:14.520]  是租的还是买房租租的租的是吗？啊，然后就是指外围的这一，我现在我们在的这个位置的这一块嘛，我们是租了他们的一部分，就是一部分连他们的那个呃呃这个维修的那一块，我看是还在的是吧？对对对，还是未来他们也还会在对，未来也在我们只租了这一个大的院子里面的一部分，就租了院子的一半这个样子啊啊啊。\n[4:14.940,5:6.440]  那这个呃这个分公司，他现在因为8月份才呃注册的，他有开展业务吗？还没有就是是呃业务是都还没有开展，也就是说他还没有一个收益，没有没有啊，那后续你开了这个红的话，呃你们是未来是有没有什么业务储备？嗯有啊有啊，就是我们现在整个第一次的整个业务的话，我们就要分出来了，分出来了，拿几块就给分公司来做啊。嗯，不是现在还在一个整合的过程，对对对，因为我们相当于做的也是我们原来总公司的这些业务储备，他不用他自己去拓展新的业务，到底有以后可能可能总公司这边有新业务拓展的话，有可能会交交一部分给到分公司这边去拓展。嗯，是这样的，那这边的人员配置有配置起来吗？\n[5:6.440,6:6.900]  嗯，还没有还没有是吧？有就目目前来说，就是暂时主要管理的就是你们两个人是吗？嗯，对对对，可以这样说啊，那这个我看到有提供一个这个这个这个这个水的发水电费的，水电费的发票是一个是水费的，这个是电费的。嗯，然后就是这个称就是这家公司的，他们签租赁合同的这家公司的瑞思的田汽车控股就是这个位置的嘛，对对，就是这个地址啊。然后这个营业执照的这个原件现在没有拿到，是原件，因为都放在老公司那边，要没说要拿的话，今天就没拿。然后就是你们的这个呃开户的话，是经办人过去办理是吗？对对对，是的是我们这个呃总公司。\n[6:6.900,6:26.780]  公司的一个经理罗总刘丽宇啊，然后这个分公司现在就相当于他也没有什么资产了，这一辈子是没有什么。\n[6:40.320,7:40.800]  好，给您宣读一下我们的风险提示，自2024年12月1日起，银行汴金社区的市级级以上公安机关，这了非法买卖出租借银行账户支付账户的3个以上或三次以上，或者为上诉卡账户账号提供实名求证。八助三张以上或三次以上的单位个人和相关组织者，假冒他人身份或者虚构代理关系开立银行账户、支付账户等等，将纳入电信网络诈骗严重失信主体名单，共享至全国信用信息共享平台，并通过信用中国网站对严重失信主体信息进行公示。惩戒对象信息纳入金融信用信息基础数据库，并限制惩戒对象名下银行账户的非固定业务，暂停了其新泰币支付账户，实施电信网络诈骗及其关联犯罪被追究刑事责任的，惩戒期限为3年。惩戒对象在惩戒期限内被多次惩戒的，惩戒期限累计执行。\n[7:40.800,8:1.980]  的信息现不能超里面。以上内容是账户相关法律责任和惩戒措施，请您依法依规司核使用清楚了。杨老师这边还有什么问题吗？我这边能下去看一下他实际的那个经营场所吗？可以的。\n[8:2.000,8:9.180]  我们再下去补录一下，嗯，就法人老师这边还有问题吗？没有了。\n[8:9.720,8:12.240]  好的谢谢王总啊嗯。\n[8:14.340,8:15.460]  嗯。\n[8:17.060,8:19.740]  从这个5万元。\n[8:19.740,8:21.060]  好。\n[8:29.340,8:30.580]  关于。\n[8:43.880,8:48.020]  那个老师们可以采访一下员工吗？\n[8:51.980,8:55.120]  这边是还在装修着。\n[8:55.700,9:1.800]  这个就是他们还在装修的那边，只是临时的一个办公场所。\n[9:1.800,9:6.480]  那个老师，你们可以采访一下他们的员工吗？\n[9:6.480,9:19.860]  老师有点听不清楚，因为这边在打地砖，是说什么是那个这边可以采访一下他们的员工吗？员工是吗？好的，我们拿过去给他们员工稍等一下。\n[9:20.760,9:26.000]  那个是的。\n[9:32.960,9:34.620]  接着。\n[9:34.620,9:53.820]  老师好像没在呀，他们因为这个蓝色的那个就是穿蓝色衣服的那个，不是他们的员工吗？这个是隔壁的这个维修的，他们这里有一个维修。\n[9:54.900,9:59.420]  就是说他们其实不是维修的这个对吧。\n[9:59.420,10:2.140]  对，他们是卖汽配的。\n[10:4.020,10:14.260]  就这一边，相当于他们刚刚有说到他们租的话，他们只是租了一部分，然后他们这个这个是别克吧，还是哪一家的？\n[10:14.280,10:17.020]  这是他们之前。\n[10:17.020,10:27.000]  嗯，之前这个的名字可以看一下吗？就是那个挂牌那里是不是这个公司的名字哦？我看一下啊，这边是不是的？\n[10:34.140,10:40.360]  这边是另一个门进来的，这里是他们的后门，应该不是一家公司的。\n[10:40.360,10:48.220]  那看一下这个，因为我们他们刚刚也有也有跟我们说的，这边不是他们的。\n[10:56.300,10:59.380]  哎呀，这边这边都没有。\n[10:59.380,11:1.080]  没有门头。\n[11:1.080,11:10.740]  好，那转回去看一下，刚才那个来的时候，就是来的时候那个口，然后有他们挂牌的地方，我看有我看是有一个那个金黄色的挂牌。\n[11:10.740,11:18.400]  哦，那个不是那个只是个保险公司的挂牌，说的是这里是那个保险的点。\n[11:18.460,11:20.820]  嗯，好，我知道了。\n[11:26.240,11:32.480]  于老师，听得到吗？是不是卡住了？我在装修，所有的牌子都没有见到。\n[11:33.200,11:36.600]  就门口的那个牌子也是拿掉的。\n[11:36.740,11:40.540]  最早有一个那个一汽丰田的那个。\n[11:40.700,11:48.340]  那个问一下别克那个他说的那个货架是他们的这两个。\n[11:51.580,11:57.680]  老师，你们是这个维修站的这个员工是吗？\n[12:0.140,12:9.900]  问货架是王总的这个的吗？这个货架这些是王总他们的吧？对哦。\n[12:9.900,12:20.460]  这个货架，是那个王总他们的，这个工作人员说了，是这个这一层上面还有吗？上面还有要上去看吗？\n[12:20.680,12:26.120]  嗯，看一下吧，看一下好。\n[12:33.020,12:37.660]  啊，这边也有哦，这边上面也有是吧。\n[12:42.740,12:46.620]  嗯，他们搭了一个这个货架在这边。\n[12:47.300,12:52.600]  老师，能问一下他们的那个就是这个工作人员他们公司的名称吗？\n[12:52.960,12:56.580]  就是他们自己搜名称啊。\n[12:56.580,13:13.060]  老师，你们这边是这个呃维修站的是叫什么维修站呀？呃，我们这是别克、别克、凯迪拉克还有丰田哦。他们是那个品4S店品牌的这个维修站是吗？哦，那有没有全称就是公司的全称。\n[13:13.280,13:21.380]  有没有公司的全称？有有的叫什么？大概呃。\n[13:36.180,13:48.160]  稍等啊，他开一下钉钉，嗯，好的，第3个哦，3个。\n[13:48.200,13:52.180]  老师，能近距离看一下吗？\n[13:53.720,13:56.060]  好，可以。\n[13:56.060,14:8.620]  啊，可以了，没有什么问题了。好好，那老师再见。嗯，因为这个楼梯有点陡，哎哟，好，可以结束了，老师啊。\n', 'Message': 'Success'}

    # test_speech()
