import gradio as gr

model_list = ["yolov8n.pt","yolov8s.pt","yolov8m.pt","yolov8l.pt","yolov8x.pt"]


    
gr.Markdown("""
    # YoloV8
    YoloV8实时视频检测
    """)
with gr.Row():
    with gr.Column():
        input_video = gr.Video(label="Input video")
        model = gr.Dropdown(model_list, value="yolov8n.pt", label="Model")
    with gr.Column():
        output_video = gr.Video()
        output_video_path = gr.Textbox(label="Output path")
button = gr.Button("Process")