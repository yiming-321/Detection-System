from ultralytics import YOLO 
import math
import cv2 
import cvzone
import numpy as np
import tempfile
import os
import deep_sort.deep_sort.deep_sort as ds

import gradio as gr
import os
import time
import pandas as pd
import requests
import base64
import json
import mimetypes
from PIL import Image
from enum import Enum
from pathlib import Path

mimetypes.init()
mimetypes.add_type("application/javascript", ".js")

BBOX_COLOR = ['红色','橙色','黄色','绿色','青色','蓝色','紫色','深粉色','浅红色','浅橙色','酸橙色','青柠色','钢青色','淡钢青色','淡紫色','热粉色','棕色']

def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}


class ScriptLoader:
    path_map = {
        "js": os.path.abspath(os.path.join(os.path.dirname(__file__), "javascript")),  # 指定加载js脚本的路径
        "py": os.path.abspath(os.path.join(os.path.dirname(__file__), "python"))  # 指定加载python脚本的路径
    }

    # 父类，支持加载javascript脚本和python脚本
    def __init__(self, script_type):
        self.script_type = script_type  # 'js'
        self.path = ScriptLoader.path_map[script_type]  # '/opt/lang/javascript'
        self.loaded_scripts = []  # eg: 最后被加载的具体的js的内容存放在这里

    @staticmethod
    def get_scripts(path: str, file_type: str) -> list[tuple[str, str]]:
        """Returns list of tuples
        每个元祖包含完整的文件路径和文件名
        """
        scripts = []
        dir_list = [os.path.join(path, f) for f in os.listdir(path)]  # 获取所有的脚本文件,eg: ['/opt/lang/javascript/t1.js']
        files_list = [f for f in dir_list if os.path.isfile(f)]  # 不要目录，只要文件,eg: ['/opt/lang/disney/javascript/bboxHint.js']
        for s in files_list:
            # Dont forget the "." for file extension
            if os.path.splitext(s)[1] == f".{file_type}":  # 只要js文件的后缀类型
                scripts.append((s, os.path.basename(s)))  # [('/opt/lang/javascript/t1.js', 't1.js')]
        return scripts


class JavaScriptLoader(ScriptLoader):
    def __init__(self):
        # 初始化父类，这个是指定加载js脚本
        super().__init__("js")
        # 复制一下原来的模板
        self.original_template = gr.routes.templates.TemplateResponse
        # Prep the js files
        self.load_js()
        # 把修改后的模板赋值给你的 gradio，方便调用
        gr.routes.templates.TemplateResponse = self.template_response

    def load_js(self):
        js_scripts = ScriptLoader.get_scripts(self.path,
                                              self.script_type)  # 获取所有的js脚本,eg:[('/opt/lang/javascript/t1.js', 't1.js')]
        for file_path, file_name in js_scripts:
            # file_name: t1.js, file_type: '/opt/lang/javascript/t1.js'
            with open(file_path, 'r', encoding="utf-8") as file:  # 读取js文件
                self.loaded_scripts.append(f"\n<!--{file_name}-->\n<script>\n{file.read()}\n</script>")

    def template_response(self, *args, **kwargs):
        """
        一旦gradio调用你的方法，你就调用原来的，你修改它包含你的脚本，然后你返回修改后的版本
        header里面包含你的脚本，返回给gradio
        """
        response = self.original_template(*args, **kwargs)
        response.body = response.body.replace(
            '</head>'.encode('utf-8'), f"{''.join(self.loaded_scripts)}\n</head>".encode("utf-8")
        )
        response.init_headers()
        return response


class BlendMode(Enum):  # i.e. LayerType
    # 区分前景色和背景色
    FOREGROUND = 'Foreground'
    BACKGROUND = 'Background'

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.value == other
        elif isinstance(other, BlendMode):
            return self.value == other.value
        else:
            raise TypeError(f'unsupported type: {type(other)}')


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

def gr_value(value=None):
    return {"value": value, "__type__": "update"}

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

IMAGE_API = "192.168.50.189"
IMAGE_PORT = 6656
num = 0  # 计数

BBOX_MAX_NUM = 6
is_t2i = 'false'
js_loader = JavaScriptLoader()
theme = gr.themes.Soft(
    primary_hue="sky",
)
#YoloV8官方模型，从左往右由小到大，第一次使用会自动下载
model_list = ["yolov8n.pt","yolov8s.pt","yolov8m.pt","yolov8l.pt","yolov8x.pt"]


#YoloV8官方模型标签数据，本次项目只使用了'person'
classNames=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors',
    'teddy bear', 'hair drier', 'toothbrush']



#视频处理
def processImage(inputPath,model):
    tracker = ds.DeepSort('deep_sort/deep_sort/deep/checkpoint/ckpt.t7')
    model=YOLO(model)#加载YOLO模型文件

    img = inputPath#从inputPath读入图片
    size =img.shape   #获取图片的大小
    # output_img = cv2.imread()#初始化图片写入
    # outputPath=tempfile.mkdtemp()#创建输出视频的临时文件夹的路径

    #输出格式为XVID格式的avi文件
    #如果需要使用h264编码或者需要保存为其他格式，可能需要下载openh264-1.8.0
    #下载地址：https://github.com/cisco/openh264/releases/tag/v1.8.0
    #下载完成后将dll文件放在当前文件夹内
    
    results=model(img,stream=False)
    detections=np.empty((0, 4))
    confarray = []
    for r in results:
        boxes=r.boxes
        for box in boxes:
            x1,y1,x2,y2=box.xywh[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)#将tensor类型转变为整型
            conf=math.ceil(box.conf[0]*100)/100#对conf取2位小数
            cls=int(box.cls[0])#获取物体类别标签
            #只检测和跟踪行人
            if cls==0:
                currentArray=np.array([x1,y1,x2,y2])
                confarray.append(conf)
                detections=np.vstack((detections,currentArray))#按行堆叠数据

    #行人跟踪
    resultsTracker=tracker.update(detections, confarray, img)
    for result in resultsTracker:
        x1,y1,x2,y2,Id=result
        x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)#将浮点数转变为整型
        cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
        cvzone.putTextRect(img,f'{int(Id)}',(max(-10,x1),max(40,y1)),scale=1.3,thickness=2)

    return img
    # output_viedo.write(img)#将处理后的图像写入视频
    # output_viedo.release()#释放
    # cap.release()#释放
    # print(video_save_path)
    # return video_save_path, video_save_path #Gradio的视频控件实际读取的是文件路径

#视频处理
def processVideo(inputPath,model):

    tracker = ds.DeepSort('deep_sort/deep_sort/deep/checkpoint/ckpt.t7') #加载deepsort权重文件
    model=YOLO(model)#加载YOLO模型文件

    cap = cv2.VideoCapture(inputPath)#从inputPath读入视频
    fps = cap.get(cv2.CAP_PROP_FPS) #获取视频的帧率
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))#获取视频的大小
    output_viedo = cv2.VideoWriter()#初始化视频写入
    outputPath=tempfile.mkdtemp()#创建输出视频的临时文件夹的路径

    #输出格式为XVID格式的avi文件
    #如果需要使用h264编码或者需要保存为其他格式，可能需要下载openh264-1.8.0
    #下载地址：https://github.com/cisco/openh264/releases/tag/v1.8.0
    #下载完成后将dll文件放在当前文件夹内
    output_type = "mp4"
    if output_type == "avi":
        fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
        video_save_path = os.path.join(outputPath,"output.avi")#创建输出视频路径
    if output_type == "mp4": #浏览器只支持播放h264编码的mp4视频文件
        fourcc = cv2.VideoWriter_fourcc('M','P','4','V')
        video_save_path = os.path.join(outputPath,"output.mp4")

    output_viedo.open(video_save_path , fourcc, fps, size, True)
    #对每一帧图片进行读取和处理
    while True:
        ret, img = cap.read()

        results=model(img,stream=True)
        detections=np.empty((0, 4))
        confarray = []
        if not(ret):
            break
        #读取推理的数据
        for r in results:
            boxes=r.boxes
            for box in boxes:
                x1,y1,x2,y2=box.xywh[0]
                x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)#将tensor类型转变为整型
                conf=math.ceil(box.conf[0]*100)/100#对conf取2位小数
                cls=int(box.cls[0])#获取物体类别标签
                #只检测和跟踪行人
                if cls==0:
                    currentArray=np.array([x1,y1,x2,y2])
                    confarray.append(conf)
                    detections=np.vstack((detections,currentArray))#按行堆叠数据

        #行人跟踪
        resultsTracker=tracker.update(detections, confarray, img)
        for result in resultsTracker:
            x1,y1,x2,y2,Id=result
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)#将浮点数转变为整型
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            cvzone.putTextRect(img,f'{int(Id)}',(max(-10,x1),max(40,y1)),scale=1.3,thickness=2)
        output_viedo.write(img)#将处理后的图像写入视频
    output_viedo.release()#释放
    cap.release()#释放
    print(video_save_path)
    return video_save_path, video_save_path #Gradio的视频控件实际读取的是文件路径




import torch

from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops


class DetectionPredictor(BasePredictor):

    def postprocess(self, preds, img, orig_imgs):
        """Postprocesses predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            if not isinstance(orig_imgs, torch.Tensor):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            path = self.batch[0]
            img_path = path[i] if isinstance(path, list) else path
            results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results


def predict(cfg=DEFAULT_CFG, use_python=False):
    """Runs YOLO model inference on input image(s)."""
    model = '/home/rl/wyh/yolov8-deepsort-tracking-main/best2.pt'
    source = '/home/rl/wyh/yolov8-deepsort-tracking-main/images/pitted_surface_180.jpg'

    args = dict(model=model, source=source)
    if use_python:
        from ultralytics import YOLO
        YOLO(model)(**args)
    else:
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()
        img = predictor.plotted_img
        return img



def processImage2(inputPath):
    from PIL import Image

    im = Image.fromarray(inputPath)
    im.save("filename.jpg")




    model = '/home/rl/wyh/yolov8-deepsort-tracking-main/best.pt'
    source = '/home/rl/wyh/yolov8-deepsort-tracking-main/filename.jpg'
    args = dict(model=model, source=source)
    predictor = DetectionPredictor(overrides=args)
    predictor.predict_cli()
    img = predictor.plotted_img
    return img


def processImage3(inputPath):
    from PIL import Image
    import cv2 as cv
    im = Image.fromarray(inputPath)
    im.save("filename.jpg")
    
    img = cv.imread('/home/rl/wyh/yolov8-deepsort-tracking-main/visial/allvisual/rolled-in_scale_238.jpg')
    return img





if __name__ == '__main__':
    with gr.Blocks() as demo:
        with gr.Tab("用户"):
        
            with gr.Tab("视频检测"):
                gr.Markdown("""
                    # YoloV8
                    YoloV8实时视频检测
                    """)
                with gr.Row():
                    with gr.Column():
                        input_video = gr.Video(label="输入视频")
                        model = gr.Dropdown(model_list, value="yolov8n.pt", label="Model")
                    with gr.Column():
                        output_video = gr.Video()
                        output_video_path = gr.Textbox(label="输出地址")
                button = gr.Button("Process")

                button.click(processVideo, inputs=[input_video,model], outputs=[output_video,output_video_path])
            with gr.Tab("图像检测"):
                gr.Markdown("""
                    # YoloV8
                    YoloV8图像检测
                    """)
                # with gr.Row():
                #     model = gr.Dropdown(model_list, value="yolov8n.pt", label="Model")
                #     confidence = gr.Slider(0, 1, 0.01, step=1, label="Confidence")

                # with gr.Row():
                #     img_input = gr.Image(label="Input img")
                #     img_output = gr.AnnotatedImage().style(
                #         color_map={"banana": "#a89a00", "carrot": "#ffae00"}
                #     )

                # section_btn = gr.Button("Identify Sections")
                # selected_section = gr.Textbox(label="Selected Section")
                # section_btn.click(processImage, inputs=[img_input, model], outputs=[img_output])
                with gr.Row():
                    with gr.Column():
                        img_input = gr.Image(label="输入图片")
                    with gr.Column():
                        # img_output = gr.AnnotatedImage().style(
                        # color_map={"banana": "#a89a00", "carrot": "#ffae00"}
                        img_output = gr.Image()

                    
                section_btn = gr.Button("目标检测")
                section_btn.click(processImage3, inputs=[img_input], outputs=[img_output])


        with gr.Tab("管理员"):

            with gr.Tab("模型训练"):
                gr.Markdown("""
                # Yolo-SD缺陷检测系统
                模型训练
                """)
                with gr.Row():
                    traindata = gr.Dropdown(choices=["nuedet","neudet_2","neudet_5","neudet-YOLO-SD2","neudet-YOLO-SD_2"],placeholder="数据集选择" ,label="Dataset")
                    trainmodel = gr.Dropdown(choices=["YOLOV5s","YOLOV5m","YOLOV5l","YOLOV8s","YOLOV8m","YOLOV8l",],placeholder="预训练.pt文件", label="pretrained weight")
                with gr.Row():
                    
                    with gr.Column():
                        gr.Markdown("""
                        设置训练参数
                        """)
                        epochs = gr.Slider(1, 100, step=1, label="epochs")
                        batch_size = gr.Slider(1, 100, step=1, label="batch_size")
                        img_size = gr.Slider(1, 100, step=1, label="img_size")
                        lr = gr.Slider(1, 100, step=1, label="lr")
                        trainyaml= gr.Dropdown(choices=["yolov8s.yaml","YOLOV8m.yaml","yolov8m_2.yaml","yolov8m_5.yaml"],placeholder="模型框架yaml文件", label="model parameters")
                        starttrain= gr.Button("开始训练")
                    with gr.Column():
                        # gr.Markdown("""
                        # 训练日志
                        # """)
                        trainlog = gr.Textbox(label='训练日志',lines=20,max_lines=20)


            with gr.Tab("图像标记"):
            
                with gr.Column(scale=3):
                    bbox_controls = []  # control set for each bbox
                    table_display = []
                    picture_info = gr.Textbox(label='图片来源', value="", elem_id='picture-index')
                    title = gr.Textbox(label='图片标题', value="", elem_id='picture-title')
                    img_path = gr.Textbox(label='图片路径', visible=False, elem_id='picture-index')
                    # 显示图片,elem_id被js查找,.style(height=400)
                    pred_img = gr.Image(value=get_first_image(), label="预测图片", elem_id="MD-bbox-ref-i2i")
                    for i in range(BBOX_MAX_NUM):
                        # Only when displaying & png generate info we use index i+1, in other cases we use i
                        with gr.Accordion(BBOX_COLOR[i], open=False, elem_id=f'MD-accordion-i2i-{i}'):
                            with gr.Row(variant='compact'):
                                bbox = gr.Checkbox(label=f'启用', value=False,elem_id=f'MD-i2i-{i}-enable',info="注意，当标签是无时无法对bbox移动操作")
                                bbox.change(fn=None, inputs=bbox, outputs=bbox, _js=f'e => onBoxEnableClick({is_t2i}, {i}, e)')
                                prodoct_info = gr.Text(label='标签', value="无", elem_id=f'MD-i2i-{i}-product')

                            with gr.Row(variant='compact',visible=True):
                                x = gr.Slider(label='x', value=0.4, minimum=0.0, maximum=1.0, step=0.01,
                                            elem_id=f'MD-i2i-{i}-x')
                                y = gr.Slider(label='y', value=0.4, minimum=0.0, maximum=1.0, step=0.01,
                                            elem_id=f'MD-i2i-{i}-y')

                            with gr.Row(variant='compact',visible=True):
                                w = gr.Slider(label='w', value=0.2, minimum=0.0, maximum=1.0, step=0.01,
                                            elem_id=f'MD-i2i-{i}-w')
                                h = gr.Slider(label='h', value=0.2, minimum=0.0, maximum=1.0, step=0.01,
                                            elem_id=f'MD-i2i-{i}-h')
                                # 更改产品，也需要刷新页面
                                prodoct_info.change(fn=None, inputs=[x, prodoct_info], outputs=x,
                                        _js=f'(v,p) => onBoxChange({is_t2i}, {i}, "x", v,p)')
                                # 点的位置会被修改，修改bbox是，这些都会被修改
                                x.change(fn=None, inputs=[x,prodoct_info], outputs=x, _js=f'(v,p) => onBoxChange({is_t2i}, {i}, "x", v,p)')
                                y.change(fn=None, inputs=[y,prodoct_info], outputs=y, _js=f'(v,p )=> onBoxChange({is_t2i}, {i}, "y", v,p)')
                                w.change(fn=None, inputs=[w,prodoct_info], outputs=w, _js=f'(v,p) => onBoxChange({is_t2i}, {i}, "w", v,p)')
                                h.change(fn=None, inputs=[h,prodoct_info], outputs=h, _js=f'(v,p) => onBoxChange({is_t2i}, {i}, "h", v,p)')


                            def on_select(evt: gr.SelectData):
                                image_url = evt.value
                                print(f"显示图片为{image_url}")
                                return image_url
                            with gr.Row(variant='compact',visible=True):
                                table = gr.Dataframe(elem_id=f'MD-i2i-{i}-table')
                                image = gr.Image(label="显示点击的图片", elem_id=f"MD-reference-{i}-image").style(height=200)
                                table.select(on_select, None, image)
                        control = [bbox, prodoct_info, x, y, w, h]
                        table_display.append(table)
                        # 方便对每个bbox进行操作, 这里不能用append，因为append会报错：AttributeError: 'list' object has no attribute '_id'
                        bbox_controls.extend(control)
                with gr.Column(scale=1):
                    with gr.Row(scale=0.5):
                        brand_name = gr.Textbox(label='品牌名称', value="", elem_id='brand-index',placeholder="eg:欧莱雅")
                        bbox_num = gr.Dropdown(list(range(-1, 7)), default=-1, label="bbox数量", info="限制返回的图片含有的商品的数量，默认不限制", elem_id="bbox-num")
                    # 显示一个目标图片
                    target_img = gr.Image(label="目标图片", elem_id="target-img")
                    target_name = gr.Textbox(label='目标名称', value="目标图片", elem_id='target-index')
                    # 如果不正确，需要输入一个正确的标签
                    query_txt = gr.Text(lines=1, placeholder='可以查询一个图片标签', label='查询名称')
                    # 查询按钮，快捷键q
                    query_btn = gr.Button(value='查询', variant='tool', elem_id="query_btn")
                    # 存储结果，传给js使用
                    json_res = gr.JSON(label="bbox info", visible=False, elem_id="json_res")
                    with gr.Row():
                        # 上一张，下一张按钮, 快捷键w,s,j
                        prev_btn = gr.Button(value='上一张', variant='tool', elem_id="prev_btn")
                        next_btn = gr.Button(value='下一张', variant='tool', elem_id="next_btn")
                        skip_btn = gr.Button(value='跳过', variant='tool', elem_id="skip_btn")
                next_btn.click(get_save_next_image, [bbox_num,brand_name, picture_info, pred_img,img_path,*bbox_controls], [target_name,target_img,picture_info, title,pred_img, img_path,json_res,*bbox_controls,*table_display]).then(fn=None,inputs=json_res,outputs=None,_js="initialThenUpdateBoxes")
                skip_btn.click(just_get_next_image, [bbox_num,brand_name], [target_name,target_img,picture_info, title,pred_img, img_path,json_res,*bbox_controls,*table_display]).then(fn=None,inputs=json_res,outputs=None,_js="initialThenUpdateBoxes")
                # 获取上一张的图片信息和bbox信息，然后重绘bbox,
                prev_btn.click(get_last_image, [bbox_num,brand_name], [target_name,target_img,picture_info, title, pred_img, img_path,json_res,*bbox_controls,*table_display]).then(fn=None,inputs=json_res,outputs=None,_js="initialThenUpdateBoxes")
                query_btn.click(do_query,[query_txt],[target_name, target_img])
                pass
        
        


    demo.launch(server_port=6006)