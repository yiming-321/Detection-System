import os
import gradio as gr
from enum import Enum


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
        files_list = [f for f in dir_list if
                      os.path.isfile(f)]  # 不要目录，只要文件,eg: ['/opt/lang/disney/javascript/bboxHint.js']
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
