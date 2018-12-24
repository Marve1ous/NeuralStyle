from fun import run
from flask import Flask, request, jsonify, Response, render_template, redirect, url_for, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import base64

app = Flask(__name__)
CORS(app)

path = ['content', 'style']

global use
use = 0


@app.route('/upload', methods=['POST', 'GET'])
def upload():
    n = get_num()
    if n == 0 and request.method == 'POST':
        if request.form['content'] == 'no':
            return "No Content Image!"
        elif request.form['style'] == 'no':
            return "No Style Image!"
        elif request.form['style'] == 'yes':
            f2 = request.files['file2']
            style_path = os.path.join('static', path[1], secure_filename(f2.filename))
            f2.save(style_path)
        else:
            f2 = request.form['style']
            if os.path.exists(f2):
                style_path = f2
            else:
                return "Error: 该风格图不存在"
        f1 = request.files['file1']
        content_path = os.path.join('static', path[0], secure_filename(f1.filename))
        f1.save(content_path)
        set_num(1)
        return render_template("get.html", f1=content_path, f2=style_path)
    return "请求错误或迁移程序正在运行中，请稍后再试！"


@app.route('/get', methods=['GET', 'POST'])
def get():
    f1 = request.args.get('f1')  # file1
    f2 = request.args.get('f2')  # file2
    s = f2.split('/')[2].split('.')[0] + '_' + f1.split('/')[2]
    name = os.path.join('static', 'out', s)
    content_path = f1
    style_path = f2
    print(content_path, style_path)
    run(content_path=content_path, style_path=style_path, path=name, num_iterations=100)
    # 取消base64返回格式, 返回图片路径
    set_num(0)
    return name


@app.route('/', methods=['GET', 'POST'])
def index():
    num = get_num()
    print(num)
    return render_template("index.html")


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')


@app.route('/android', methods=['POST', 'GET'])
def android():
    n = get_num()
    if n == 0 and request.method == 'POST':
        if request.form['content'] == 'no':
            return "No Content Image!"
        elif request.form['style'] == 'no':
            return "No Style Image!"
        elif request.form['style'] == 'yes':
            f2 = request.files['file2']
            style_path = os.path.join('static', path[1], secure_filename(f2.filename))
            f2.save(style_path)
        else:
            f2 = request.form['style']
            if os.path.exists(f2):
                style_path = f2
            else:
                return "Error: 该风格图不存在"
        f1 = request.files['file1']
        content_path = os.path.join('static', path[0], secure_filename(f1.filename))
        f1.save(content_path)
        set_num(1)
        s = style_path.split('/')[2].split('.')[0] + '_' + content_path.split('/')[2]
        name = os.path.join('static', 'out', s)
        print(content_path, style_path)
        run(content_path=content_path, style_path=style_path, path=name, num_iterations=100)
        set_num(0)
        return name
    return "请求错误或迁移程序正在运行中，请稍后再试！"


def get_num():
    return use


def set_num(n):
    global use
    use = n


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
