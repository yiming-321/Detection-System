import gradio as gr

model_list = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]


class image_tap(object):
    gr.Markdown("""
        # YoloV8
        YoloV8图像检测
        """)
    # with gr.Row():
    #     with gr.Column():
    #         input_img = gr.Image(label="Input image or folder")
    #         model = gr.Dropdown(model_list, value="yolov8n.pt", label="Model1")
    #     with gr.Column():
    #         output_img = gr.Image()
    #         output_img_path = gr.Textbox(label="Output path1")
    # button1 = gr.Button("Process img")
    with gr.Row():
        model = gr.Dropdown(model_list, value="yolov8n.pt", label="Model")
        confidence = gr.Slider(0, 1, 0.01, step=1, label="Confidence")

    with gr.Row():
        img_input = gr.Image()
        img_output = gr.AnnotatedImage().style(
            color_map={"banana": "#a89a00", "carrot": "#ffae00"}
        )

    section_btn = gr.Button("Identify Sections")
    selected_section = gr.Textbox(label="Selected Section")
