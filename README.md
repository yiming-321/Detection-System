# Discerning Insight — 工业视觉质量检测平台
#### “Discerning Insight” 是一个基于计算机视觉和深度学习的工业质量检测平台，旨在实现高效、准确的产品缺陷识别与分类。<br>
### 项目结构
```plaintext
Detection-System/
├── Detect.py                # 主检测脚本
├── DetectionSystem/         # 核心检测模块
│   ├── __init__.py
│   ├── detector.py          # 检测器实现
│   └── utils.py             # 工具函数
├── yolov8n.pt              # 训练好的 YOLOv8 模型权重
├── requirements.txt         # 项目依赖
└── README.md                # 项目说明
```
### 功能简介
- 目标检测：利用 YOLOv8 模型进行实时目标检测。
- 缺陷分类：对检测到的目标进行缺陷类型分类。
- 视频输入支持：支持从视频文件或摄像头实时读取输入进行检测。
- 检测结果可视化：在图像或视频中标注检测结果，提供直观反馈。
### 安装与配置
#### 1. 克隆仓库：
   ```bash
   git clone https://github.com/yiming-321/Detection-System.git
   cd Detection-System
   ```
#### 2. 创建虚拟环境并安装依赖：
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows 用户使用 venv\Scripts\activate
   pip install -r requirements.txt
   ```
#### 3. 下载预训练模型：
   请确保将预训练的 YOLOv8 模型权重文件 yolov8n.pt 放置在项目根目录。
使用方法
```bash
python Detect.py --source <输入路径> --weights yolov8n.pt --img-size 640
```

- `--source`：输入源路径，可以是图片、视频文件或摄像头设备编号。
- `--weights`：YOLOv8 模型权重文件路径。
- `--img-size`：输入图像的尺寸，默认为 640。

例如，检测本地视频文件：

```bash
python Detect.py --source ./data/sample_video.mp4 --weights yolov8n.pt --img-size 640
```

或使用摄像头进行实时检测：

```bash
python Detect.py --source 0 --weights yolov8n.pt --img-size 640
```
#### 结果展示
检测结果将显示在窗口中，并保存为 runs/detect/exp 目录下的图像或视频文件，每个检测框将标注类别和置信度。<br>
<hr>

#### 贡献指南
欢迎提出问题、建议或贡献代码：<br>
- 提交 Issue：报告问题或提出功能请求。<br>
- 提交 Pull Request：贡献代码或文档改进。<br>
#### 许可协议
本项目采用 MIT 许可证，详情请参阅 LICENSE 文件。
