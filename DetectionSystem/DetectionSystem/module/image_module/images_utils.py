import math
import numpy as np
from ultralytics import YOLO
import cv2
import tempfile
import deep_sort.deep_sort.deep_sort as ds
import cvzone


# 图片处理
def processImage(inputPath, model):
    tracker = ds.DeepSort('deep_sort/deep_sort/deep/checkpoint/ckpt.t7')  # 加载deepsort权重文件
    model = YOLO(model)  # 加载YOLO模型文件

    img = cv2.imread(inputPath)  # 从inputPath读入图片
    size = img.shape()  # 获取图片的大小
    output_img = cv2.imread()  # 初始化图片写入
    outputPath = tempfile.mkdtemp()  # 创建输出视频的临时文件夹的路径
    output_viedo = cv2.VideoWriter()  # 初始化视频写入
    cap = cv2.VideoCapture(inputPath)  # 从inputPath读入视频

    # 输出格式为XVID格式的avi文件
    # 如果需要使用h264编码或者需要保存为其他格式，可能需要下载openh264-1.8.0
    # 下载地址：https://github.com/cisco/openh264/releases/tag/v1.8.0
    # 下载完成后将dll文件放在当前文件夹内

    results = model(img, stream=True)
    detections = np.empty((0, 4))
    confarray = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xywh[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # 将tensor类型转变为整型
            conf = math.ceil(box.conf[0] * 100) / 100  # 对conf取2位小数
            cls = int(box.cls[0])  # 获取物体类别标签
            # 只检测和跟踪行人
            if cls == 0:
                currentArray = np.array([x1, y1, x2, y2])
                confarray.append(conf)
                detections = np.vstack((detections, currentArray))  # 按行堆叠数据

    # 行人跟踪
    resultsTracker = tracker.update(detections, confarray, img)
    for result in resultsTracker:
        x1, y1, x2, y2, Id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # 将浮点数转变为整型
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cvzone.putTextRect(img, f'{int(Id)}', (max(-10, x1), max(40, y1)), scale=1.3, thickness=2)
    output_viedo.write(img)  # 将处理后的图像写入视频
    output_viedo.release()  # 释放
    cap.release()  # 释放
    print(video_save_path)
    return video_save_path, video_save_path  # Gradio的视频控件实际读取的是文件路径
