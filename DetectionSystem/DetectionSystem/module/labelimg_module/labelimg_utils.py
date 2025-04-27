import json
import os
from PIL import Image
import pandas as pd
import requests
import gradio as gr



BBOX_COLOR = ['红色','橙色','黄色','绿色','青色','蓝色','紫色','深粉色','浅红色','浅橙色','酸橙色','青柠色','钢青色','淡钢青色','淡紫色','热粉色','棕色']
IMAGE_API = "192.168.50.189"
IMAGE_PORT = 6656
num = 0  # 计数
BBOX_MAX_NUM = 6



def gr_value(value=None):
    return {"value": value, "__type__": "update"}

def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}


def get_one_images(brand="资生堂", idx=0, bbox_num=-1, topk=4):

    """
    只获取jpg格式的图片
    """
    res = {
    "data": {
        "bbox_candidates": [
            [
                {
                    "label": "欧莱雅新多重防护隔离液 水感倍护",
                    "url": "a.jpg"
                },
                {
                    "label": "欧莱雅新多重防护隔离露 水感轻肌",
                    "url": "a.jpg"
                },
                {
                    "label": "欧莱雅新多重防护隔离液 水感倍护",
                    "url": "a.jpg"
                },
                {
                    "label": "欧莱雅新多重防护隔离液 水感倍护",
                    "url": "a.jpg"
                }
            ],
            [
                {
                    "label": "欧莱雅多重防护隔离露 素颜亮采",
                    "url": "a.jpg"
                },
                {
                    "label": "欧莱雅多重防护隔离露 素颜亮采",
                    "url": "a.jpg"
                },
                {
                    "label": "欧莱雅小光圈喷雾",
                    "url": "a.jpg"
                },
                {
                    "label": "欧莱雅小光圈喷雾",
                    "url": "a.jpg"
                }
            ],
            [
                {
                    "label": "欧莱雅新多重防护隔离露 外御内护",
                    "url": "a.jpg"
                },
                {
                    "label": "欧莱雅新多重防护隔离露 外御内护",
                    "url": "a.jpg"
                },
                {
                    "label": "欧莱雅防脱精华液（小黑喷）",
                    "url": "a.jpg"
                },
                {
                    "label": "欧莱雅清润葡萄籽水嫩洁面乳",
                    "url": "a.jpg"
                }
            ],
            [
                {
                    "label": "欧莱雅多重防护隔离露 素颜亮采",
                    "url": "a.jpg"
                },
                {
                    "label": "欧莱雅多重防护隔离露 素颜亮采",
                    "url": "a.jpg"
                },
                {
                    "label": "欧莱雅小光圈喷雾",
                    "url": "a.jpg"
                },
                {
                    "label": "欧莱雅小光圈喷雾",
                    "url": "a.jpg"
                }
            ]
        ],
        "bbox_num": 4,
        "bboxes": [
            [
                243.12725830078125,
                236.93186950683594,
                384.913330078125,
                798.8110961914062
            ],
            [
                617.3644409179688,
                537.275146484375,
                689.0697021484375,
                793.4158935546875
            ],
            [
                392.7101745605469,
                274.6469421386719,
                526.1124267578125,
                779.217041015625
            ],
            [
                546.1234130859375,
                518.9520874023438,
                618.5255737304688,
                779.0012817382812
            ]
        ],
        "brand": "欧莱雅",
        "classes": [
            "欧莱雅新多重防护隔离液 水感倍护",
            "欧莱雅多重防护隔离露 素颜亮采",
            "欧莱雅新多重防护隔离露 外御内护",
            "欧莱雅多重防护隔离露 素颜亮采"
        ],
        "idx": 0,
        "pic": "a.jpg",
        "pic_height": 800,
        "pic_path": "a.jpg",
        "pic_width": 800,
        "product": "欧莱雅新多重防护隔离露 外御内护",
        "product_img": "a.jpg",
        "scores": [
            0.8991537094116211,
            0.891562283039093,
            0.8848311305046082,
            0.8565376996994019
        ],
        "title": "欧莱雅防晒小金管面部身体防晒霜隔离霜保湿防紫外线防晒乳SPF50+",
        "total": 919,
        "url": "a.jpg"
    },
    "msg": "success",
    "status": 0
}
    return res


def do_query(query_txt):
    """
    根据query_txt进行查询
    Args:
        query_txt (): 产品名称或别名,"资生堂新男士焕能紧致眼霜"
    Returns:
    """
    picture_url = 'a.jpg'
    official_name = '欧莱雅新多重防护隔离露 外御内护'
    return official_name, picture_url

def get_first_image():
    """
    获取第一张图片，占位显示用
    Returns:
    """
    img_path = "a.jpg"
    assert os.path.exists(img_path), f"图片不存在: {img_path}"
    img_array = show_img(img_path)
    return img_array

def show_img(path):
    return Image.open(path)





def save_bbox_api(data):
    """
    保存bbox信息到mysql
    Args:
        data ():
    Returns:
    """
    url = f"http://{IMAGE_API}:{IMAGE_PORT}/api/save_bbox"
    # 提交form格式数据
    headers = {'content-type': 'application/json'}
    post_data = {"data": data, "person": True}
    # 提交form格式数据
    try:
        r = requests.post(url, data=json.dumps(post_data), headers=headers)
    except Exception as e:
        print(e)
        print(f"注意：请检查服务器是否开启，或者检查网络是否正常：{url}")
    res = r.json()
    msg = res["msg"]
    assert res["status"] == 0, f"有报错信息，请检查,{msg}"
    return res
    
def just_get_next_image(bbox_num,brand_name):
    global num
    images_info = get_one_images(brand_name, num, bbox_num)
    images_data = images_info["data"]
    image_idx = images_data["idx"]
    total_num = images_data["total"]
    num = image_idx + 1
    if num > total_num:
        raise gr.Error(f"当前品牌的数据已经是最后一张图片了，当前的序号是: {image_idx}")
    img_path = images_data["pic_path"]
    title = images_data["title"]
    product = images_data["product"]  # 产品名称,以前标注的产品名称
    product_img = images_data["product_img"]  #我们的目标图片
    url = images_data["url"]  #店铺的链接
    assert os.path.exists(img_path), f"图片不存在: {img_path}"
    img = show_img(img_path)
    bboxes = images_data["bboxes"]
    classes = images_data["classes"]
    scores = images_data["scores"]
    pic_height = images_data["pic_height"]
    pic_width = images_data["pic_width"]
    bbox_candidates = images_data["bbox_candidates"]  # 每个bbox的多个候选可能的商品
    #下一张的图片信息
    bbox_controls, json_res, table_display = bboxes_format(bboxes, scores, classes, pic_height,pic_width,bbox_candidates)
    # 需要对bbox_controls展开返回，因为gradio的接收列表的参数
    target_name = product
    target_img = product_img
    return target_name,target_img,url, title, img, img_path,json_res, *bbox_controls, *table_display


def get_save_next_image(bbox_num,brand_name,picture_info, pred_img, img_path, *args):
    """
    保存当前的图片信息到json文件中，然后获取下一张图片的基本信息，并返回
    bbox_num:限制返回的图片包含的bbox数量
    picture_info:是当前的商品信息，即标题
    pred_img:是当前的预测图片, ndarry格式
    img_path:是当前的图片路径
    kwargs:是所有的参数，包括bbox_controls
    根据num获取下一个图片
    """
    if not brand_name:
        raise gr.Error(f"没有给定品牌的名称，请给定一个品牌名称")
    print(f"获取下一张图片,当前的序号是: {num}")
    # 上一张图片的信息
    if img_path:
        print(f"保存上一张图片的信息")
        bboxes = []
        # bbox, product_info, x, y, w, h,一共是6个参数一组，所以要除以6一定是整数
        bbox_controls_length = len(args)
        assert bbox_controls_length % 6 == 0, "bbox_controls的长度必须是6的倍数"
        for i in range(0, bbox_controls_length, 6):
            # 一个bbox的信息
            bbox = args[i:i + 6]
            bbox_enabled = bbox[0]
            if bbox_enabled:
                label = bbox[1]
                x = bbox[2]
                y = bbox[3]
                w = bbox[4]
                h = bbox[5]
                bbox = {
                    "label": label,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                }
                bboxes.append(bbox)
        # 遍历所有的bbox_controls，如果第一个参数Bbox是True，那么就保留
        height, width = pred_img.shape[:2]
        data = {
            "image_path": img_path,
            "url": picture_info,
            "bboxes": bboxes,
            "width": width,
            "height": height,
        }
        # 如果图片存在，就保存当前的图片信息到json文件中，如果不存在就不保存,json文件和图片是一一对应的
        save_bbox_api(data)
    return just_get_next_image(bbox_num,brand_name)



def bboxes_format(bboxes, scores, classes,height, width, bbox_candidates):
    """
    bboxes格式制作
    bbox_candidates: BBOX_MAX_NUM个bbox的候选标签列表
    """
    # 当预测的图片没有那么多bbox的时候，给个默认值, bbox, prodoct_info, x, y, w, h
    default_bbox = [False, "无", 0.4, 0.4, 0.2, 0.2]
    default_candidates = [] # 默认显示候选的标签
    num_boxes = len(bboxes)
    table_display = []
    bbox_controls = []
    json_res_list = []  # 存储到json_res一份, 只存储真实存在的bbox内容，
    # 确保每个bbox都有被操作，即使没有值
    if len(bboxes) > BBOX_MAX_NUM:
        print(f"bbox数量超过{BBOX_MAX_NUM}个，只取前{BBOX_MAX_NUM}个")
    for i in range(BBOX_MAX_NUM):
        if i < num_boxes:
            candidates = bbox_candidates[i]
            candidates = pd.DataFrame(candidates)
            table_display.append(candidates)
            product = classes[i] #第i个bbox的类别名称
            score = scores[i] #第i个bbox的分数
            # 存在的bbox，取出来,bbox默认格式是[x1, y1, x2, y2], 转换成[x, y, w, h], 并且需要归一化，分别除以图片的宽和高
            bbox = bboxes[i]
            x = bbox[0]/width   #6.38/1080
            y = bbox[1]/height  #274.3/894
            w = (bbox[2] - bbox[0])/width   #315.18-6.38  /1080 == 0.291
            h = (bbox[3] - bbox[1])/height  #816.3-274.3 / 849 == 0.659
            bbox_controls.extend([True, product, x, y, w, h])
            json_res_list.append({
                "label": product,
                "score": round(score, 2),
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "width": width,
                "height": height,
            })
        else:
            # 没有那么多的bbox，给个默认值
            bbox_controls.extend(default_bbox)
            table_display.append(default_candidates)
    #嵌套的格式，返回，bbox, prodoct_info, x, y, w, h
    # 导出成json的个是
    json_res = json.dumps(json_res_list)
    return [gr_value(v) for v in bbox_controls] ,json_res, table_display




def get_last_image(bbox_num,brand_name):
    """
    根据num获取上一个图片,和上一张图片的标注信息
    """
    global num
    print(f"获取上一张图片,当前的序号是: {num}")
    num -= 1
    if num < 0:
        raise gr.Error(f"当前品牌的数据已经是第一张图片了，当前的序号是: {num}")
    images_info = get_one_images(brand_name, num, bbox_num)
    images_data = images_info["data"]
    img_path = images_data["pic_path"]
    title = images_data["title"]
    product = images_data["product"]  # 产品名称,以前标注的产品名称
    product_img = images_data["product_img"]  #我们的目标图片
    url = images_data["url"]  # 店铺的链接
    img = show_img(img_path)
    bboxes = images_data["bboxes"]
    classes = images_data["classes"]
    scores = images_data["scores"]
    pic_height = images_data["pic_height"]
    pic_width = images_data["pic_width"]
    bbox_candidates = images_data["bbox_candidates"]  # 每个bbox的多个候选可能的商品
    # 每个都变成Dataframe格式
    bbox_controls, json_res, table_display = bboxes_format(bboxes, scores, classes, pic_height,pic_width,bbox_candidates)
    # 需要对bbox_controls展开返回，因为gradio的接收列表的参数
    # json_res, *bbox_controls是来自已有的缓存的json文件, 如果没有缓存文件，那么就是默认数据
    target_name = product
    target_img = product_img
    return target_name,target_img,url, title, img, img_path,json_res, *bbox_controls, *table_display

def get_image_size(img_path):
    """
    获取图片的大小
    """
    img = Image.open(img_path)
    width, height = img.size
    return width, height