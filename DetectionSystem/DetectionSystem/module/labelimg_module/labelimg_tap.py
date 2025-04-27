import gradio as gr

from module.labelimg_module.labelimg_utils import get_first_image, BBOX_MAX_NUM, BBOX_COLOR, get_save_next_image, just_get_next_image, get_last_image, do_query

is_t2i = 'false'


class labelimg_tap(object):
    with gr.Blocks() as demo2:
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
