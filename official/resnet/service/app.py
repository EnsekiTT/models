import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg'])
app = Flask(__name__)
app.secret_key = 'some_secret'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    message = "Hello World"

    return render_template('index.html', message=message)

@app.route('/post', methods=['GET', 'POST'])
def post():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(url_for('index'))
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(url_for('index'))
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            current_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(current_path)
            img = Image.open(current_path)
            img = img.resize((32,32))
            img.save(current_path)
            flash('upload: ' + filename)
            return redirect(url_for('index'))
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.debug = True # デバッグモード有効化
    app.run()
