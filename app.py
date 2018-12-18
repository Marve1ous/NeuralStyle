from fun import run
from flask import Flask, request, jsonify, Response, render_template, redirect, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import base64

app = Flask(__name__)
CORS(app)

path = ['content', 'style']


@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        f1 = request.files['file1']  # file1
        f2 = request.files['file2']  # file2
        content_path = os.path.join(path[0], secure_filename(f1.filename))
        style_path = os.path.join(path[1], secure_filename(f2.filename))
        f1.save(content_path)
        f2.save(style_path)
        return render_template("get.html", f1=content_path, f2=style_path)
    return "error"


@app.route('/get', methods=['GET', 'POST'])
def get():
    f1 = request.args.get('f1')  # file1
    f2 = request.args.get('f2')  # file2
    print(f1, f2)
    s = f1.split('/')[1].split('.')[0] + '_' + f2.split('/')[1]
    name = os.path.join('out', s)
    content_path = f1
    style_path = f2
    run(content_path=content_path, style_path=style_path, path=name, num_iterations=100)
    img = open(name, 'rb').read()
    base = base64.b64encode(img)
    return base


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template("index.html")


if __name__ == '__main__':
    app.run(host='0.0.0.0')
