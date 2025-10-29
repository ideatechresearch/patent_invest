import httpx, requests, asyncio
from http import HTTPStatus
from PIL import Image
import dashscope
# import qianfan
from dashscope.audio.tts import ResultCallback
from dashscope.audio.asr import Recognition, Transcription, RecognitionCallback
import time, base64, json, uuid
from urllib.parse import urlencode, urlparse, unquote

from service import get_httpx_client
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
                                api_key=Config.DashScope_Service_Key, max_retries=20):
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
    task_status = data["output"]["task_status"]
    retries = 0
    # 轮询任务进度
    await asyncio.sleep(3)
    while task_id is not None and retries < max_retries:
        task_url = f'https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}'
        task_response = await cx.get(task_url, headers=task_headers)
        resp = task_response.json()
        task_status = resp["output"]["task_status"]
        # "task_status":"PENDING""RUNNING","SUCCEEDED"->"results", "FAILED"->"message"
        if task_status == 'SUCCEEDED':
            urls = [item['url'] for item in resp['output'].get('results', []) if 'url' in item]
            result = {"urls": urls or [resp['output'].get('result_url')], 'id': task_id}
            if urls:
                image_response = await cx.get(urls[0])
                return image_response.content, result

            return None, result

        if task_status == "FAILED":
            print(resp['output']['message'])
            break

        await asyncio.sleep(3)
        retries += 1

    return None, {"urls": [], 'id': task_id, 'status': task_status,
                  'error': "Task did not succeed within the maximum retry limit."}


# https://nls-portal.console.aliyun.com/overview
async def ali_speech_to_text(audio_data, format='pcm'):
    """阿里云语音转文字"""
    params = {
        "appkey": Config.ALIYUN_nls_AppId,
        "format": format,  # 也可以传入其他格式，如 wav, mp3
        "sample_rate": 16000,  # 音频采样率
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


# 1536: 适用于普通话输入法模型（支持简单的英文）。
# 1537: 适用于普通话输入法模型（纯中文）。
# 1737: 适用于英文。
# 1936: 适用于粤语。
# audio/pcm pcm（不压缩）、wav（不压缩，pcm编码）、amr（压缩格式）、m4a（压缩格式）
# https://console.bce.baidu.com/ai/#/ai/speech/overview/index
async def baidu_speech_to_text(audio_data, format='pcm', dev_pid=1536):  #: io.BytesIO
    url = "https://vop.baidu.com/server_api"  # 'https://vop.baidu.com/pro_api'
    access_token = get_baidu_access_token(Config.BAIDU_speech_API_Key, Config.BAIDU_speech_Secret_Key)
    # Config.BAIDU_speech_AppId
    url = f"{url}?dev_pid={dev_pid}&cuid={Config.DEVICE_ID}&token={access_token}"
    headers = {'Content-Type': f'audio/{format}; rate=16000'}

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
async def dashscope_speech_to_text(audio_path, format='wav', language: list[str] = ['zh', 'en']):
    recognition = Recognition(model='paraformer-realtime-v2', format=format, sample_rate=16000,
                              language_hints=language, callback=RecognitionCallback())  # None
    result = await asyncio.to_thread(recognition.call, audio_path)  # recognition.call(audio_path)
    if result.status_code == 200:
        texts = [sentence.get('text', '') for sentence in result.get_sentence()]
        return {"text": texts[0]}

    return {"error": result.message}


# SenseVoice语音识别大模型专注于高精度多语言语音识别、情感辨识和音频事件检测，支持超过50种语言的识别，整体效果优于Whisper模型，中文与粤语识别准确率相对提升在50%以上。
# SenseVoice语音识别提供的文件转写API，能够对常见的音频或音视频文件进行语音识别，并将结果返回给调用者。
# SenseVoice语音识别返回较为丰富的结果供调用者选择使用，包括全文级文字、句子级文字、词、时间戳、语音情绪和音频事件等。模型默认进行标点符号预测和逆文本正则化。
async def dashscope_speech_to_text_url(file_urls, model='paraformer-v1', language: list[str] = ['zh', 'en']):
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
                                                })  # transcription_data["transcripts"][0]["sentences"][0]["text"]
                else:
                    print(f"No transcription text found in the response.Transcription Result: {response.text}")
            else:
                print(f"Failed to fetch transcription. Status code: {response.status_code}")
        else:
            print(f"Subtask status: {r['subtask_status']}")

    if len(file_urls) != len(transcription_texts):
        print(json.dumps(transcribe_response.output, indent=4, ensure_ascii=False))

    return transcription_texts, task_response.output.task_id


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
    import io

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
        audio_file = 'data/nls-sample-16k.wav'
        with open(audio_file, 'rb') as f:
            audio_data = io.BytesIO(f.read())

        result = await  baidu_speech_to_text(audio_data, 'wav')  # ali_speech_to_text(audio_data,'wav')
