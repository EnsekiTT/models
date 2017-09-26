from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/')
def index():
    title = "Resnet"
    message = "Hello World!"
    # index.html をレンダリングする
    return render_template('index.html',
                           message=message, title=title)

@app.route('/post', methods=['GET', 'POST'])
def post():
    if request.method == 'POST':
        image = request.form['image']
        print(image)
        plt.imshow(image)
        plt.show()
        # index.html をレンダリングする
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.debug = True # デバッグモード有効化
    app.run()
