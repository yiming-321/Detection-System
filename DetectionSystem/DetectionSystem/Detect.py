from ultralytics import YOLO
import cv2
import numpy as np
import os
import deep_sort.deep_sort.deep_sort as ds
import gradio as gr
import mimetypes
from pathlib import Path
from front_utils import JavaScriptLoader, BlendMode
from module.labelimg_module.labelimg_utils import *

mimetypes.init()
mimetypes.add_type("application/javascript", ".js")
is_t2i = 'false'
js_loader = JavaScriptLoader()
theme = gr.themes.Soft(primary_hue="sky")
model_list = ["best.pt"]
cnt = 0  # 保存图片的标号


def get_detectable_classes(model_file):  # 获取给定模型文件可以检测的类别
    models = YOLO(model_file)
    class_names = list(models.names.values())  # 直接获取类别名称列表
    del models  # 删除模型实例释放资源
    return class_names


def start_processing(input_path, output_path, detects, progress=gr.Progress(track_tqdm=True)):
    global cnt
    cnt += 1
    if not detects:  # 默认检测第一种缺陷
        detects = 0
    model = YOLO(model_list[0])
    tracker = ds.DeepSort("deep_sort/deep_sort/deep/checkpoint/ckpt.t7")
    return detect_and_track(input_path, output_path, int(detects), model, tracker)


def start_processing_All(input_path, output_path):
    global cnt
    cnt += 1
    model = YOLO(model_list[0])
    tracker = ds.DeepSort("deep_sort/deep_sort/deep/checkpoint/ckpt.t7")
    return detect_and_track_All(input_path, output_path, model, tracker)


def extract_detections(results, detect_class):
    """
    从模型结果中提取和处理检测信息
    - results: YoloV8模型预测结果, 包含检测到的物体的位置、类别和置信度等信息
    - detect_class: 需要提取的目标类别的索引
    """
    # 初始化一个空的二维numpy数组，用于存放检测到的目标的位置信息
    detections = np.empty((0, 4))
    conf_array = []  # 初始化一个空列表，用于存放检测到的目标的置信度
    for res in results:  # 遍历检测结果
        for box in res.boxes:
            # 如果检测到的目标类别与指定的目标类别相匹配，提取目标的位置信息和置信度
            if box.cls[0].int() == detect_class:
                x1, y1, x2, y2 = box.xywh[0].int().tolist()  # 提取目标的位置信息，并从tensor转换为整数列表。
                conf = round(box.conf[0].item(), 2)  # 提取目标的置信度，从tensor中取出浮点数结果，并四舍五入到小数点后两位。
                detections = np.vstack((detections, np.array([x1, y1, x2, y2])))
                conf_array.append(conf)
    return detections, conf_array


def extract_detections_All(results):
    global detect_classes

    detections = np.empty((0, 4))
    conf_array = []  # 初始化一个空列表，用于存放检测到的目标的置信度
    detectClass = []
    # nums = [x for x in range(len(detectClass))]
    # 遍历检测结果
    for res in results:
        for box in res.boxes:
            if 0 <= box.cls[0].int() < 6:  # 一共六种缺陷种类
                x1, y1, x2, y2 = box.xywh[0].int().tolist()  # 提取目标的位置信息，并从tensor转换为整数列表。
                conf = round(box.conf[0].item(), 2)  # 提取目标的置信度，从tensor中取出浮点数结果，并四舍五入到小数点后两位。
                detections = np.vstack((detections, np.array([x1, y1, x2, y2])))
                conf_array.append(conf)
                detectClass.append(detect_classes[box.cls[0].int()])
    return detections, conf_array, detectClass


def putTextWithBackground(  # ui界面
        img,
        text,
        origin,
        font=cv2.FONT_HERSHEY_SIMPLEX,
        font_scale=1,
        text_color=(255, 255, 255),
        bg_color=(0, 0, 0),
        thickness=1,
):
    """绘制带有背景的文本

    :param img: 输入图像
    :param text: 要绘制的文本
    :param origin: 文本的左上角坐标
    :param font: 字体类型
    :param font_scale: 字体大小
    :param text_color: 文本的颜色
    :param bg_color: 背景的颜色
    :param thickness: 文本的线条厚度
    """
    # 计算文本的尺寸
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

    # 绘制背景矩形
    bottom_left = origin
    top_right = (origin[0] + text_width, origin[1] - text_height - 5)
    cv2.rectangle(img, bottom_left, top_right, bg_color, -1)

    # 在矩形上绘制文本
    text_origin = (origin[0], origin[1] - 5)
    cv2.putText(
        img,
        text,
        text_origin,
        font,
        font_scale,
        text_color,
        thickness,
        lineType=cv2.LINE_AA,
    )


def detect_and_track(input_path: str, output_path: str, detect_class: int, model, tracker):
    global cnt
    """
    - input_path: 输入文件的路径。
    - output_path: 处理后保存的路径。
    - detect_class: 需要检测目标类别的索引。
    - model: 用于目标检测的模型。
    - tracker: 用于目标跟踪的模型。
    """
    cap = cv2.imread(input_path)  # 读取图片文件
    cap = cv2.cvtColor(cap, cv2.COLOR_BGR2RGB)
    if len(cap) == 0:  # 检查视频文件是否成功打开
        print(f"Error opening video file {input_path}")
        exit(IOError)  # 若没有成功打开，则抛出异常并退出程序
    results = model(cap)
    detections, conf_array = extract_detections(results, detect_class)
    resultsTracker = tracker.update(detections, conf_array, cap)
    for x1, y1, x2, y2, Id in resultsTracker:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # 将位置信息转换为整数
        cv2.rectangle(cap, (x1, y1), (x2, y2), (255, 0, 255), 3)
        putTextWithBackground(cap, str(int(Id)), (max(-10, x1), max(40, y1)), font_scale=1,
                              text_color=(255, 255, 255), bg_color=(255, 0, 255))
    image_path = Path(output_path) / ("output" + str(cnt) + ".jpg")  # 设置输出图片的保存路径
    cv2.imwrite(image_path.as_posix(), cap)
    return cap, image_path.as_posix()


def detect_and_track_All(input_path: str, output_path: str, model, tracker):
    global cnt
    """
    - input_path: 输入文件的路径。
    - output_path: 处理后保存的路径。
    - detect_class: 需要检测目标类别的索引。
    - model: 用于目标检测的模型。
    - tracker: 用于目标跟踪的模型。
    """
    cap = cv2.imread(input_path)  # 读取图片文件
    cap = cv2.cvtColor(cap, cv2.COLOR_BGR2RGB)
    if len(cap) == 0:  # 检查视频文件是否成功打开
        print(f"Error opening video file {input_path}")
        exit(IOError)  # 若没有成功打开，则抛出异常并退出程序
    results = model(cap)
    detections, conf_array, detectClass = extract_detections_All(results)
    resultsTracker = tracker.update(detections, conf_array, cap)
    i = 0
    for x1, y1, x2, y2, Id in resultsTracker:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # 将位置信息转换为整数
        cv2.rectangle(cap, (x1, y1), (x2, y2), (255, 0, 255), 3)
        putTextWithBackground(cap, detectClass[i], (max(-10, x1), max(40, y1)), font_scale=1,
                              text_color=(255, 255, 255), bg_color=(255, 0, 255))
        i += 1
    image_path = Path(output_path) / ("output" + str(cnt) + ".jpg")  # 设置输出图片的保存路径
    cv2.imwrite(image_path.as_posix(), cap)
    return cap, image_path.as_posix()


if __name__ == '__main__':
    detect_classes = get_detectable_classes(model_list[0])  # 要求传入一个模型名
    Output_file = "Outputfile"
    with gr.Blocks() as demo:
        with gr.Tab("All"):  # 检测全部缺陷
            gr.Markdown("""
                    # Yolo-SD缺陷检测系统
                    检测全部缺陷
                    """)
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(label="Input Image", type="filepath")
                    with gr.Row():
                        start_button = gr.Button("Start")
                with gr.Column():
                    output_image = gr.Image(label="Output Image")
                    output_dir = gr.Textbox(label="Output dir",  # 结果的保存文件夹
                                            value=Output_file)
                    output_image_path = gr.Textbox(label="Output Path")  # 结果的保存地址
            start_button.click(start_processing_All,
                               inputs=[input_image, output_dir],
                               outputs=[output_image, output_image_path])

        with gr.Tab("Specification"):  # 检测特定缺陷
            gr.Markdown("""
                    # Yolo-SD缺陷检测系统
                    检测特定缺陷
                    """)
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(label="Input Image", type="filepath")
                    detect_class = gr.Dropdown(detect_classes, label="Class",
                                               type='index')  # 下拉菜单控件，用于选择要检测的目标类别

                    with gr.Row():
                        start_button = gr.Button("Start")
                with gr.Column():
                    output_image = gr.Image(label="Output Image")
                    output_dir = gr.Textbox(label="Output dir",  # 结果的保存文件夹
                                            value=Output_file)
                    output_image_path = gr.Textbox(label="Output Path")  # 结果的保存地址
            start_button.click(start_processing,
                               inputs=[input_image, output_dir, detect_class],
                               outputs=[output_image, output_image_path])

    demo.launch()
