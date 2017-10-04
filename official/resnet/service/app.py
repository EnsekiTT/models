import os
import sys
import argparse

from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename

import numpy as np
import tensorflow as tf

from PIL import Image
import matplotlib.pyplot as plt
import resnet_model

HEIGHT = 32
WIDTH = 32
DEPTH = 3
NUM_CLASSES = 10

parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--model_dir', type=str, default='/tmp/cifar10_model',
                    help='The directory where the model will be stored.')

parser.add_argument('--resnet_size', type=int, default=32,
                    help='The size of the ResNet model to use.')

FLAGS = parser.parse_args()


UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg'])
app = Flask(__name__)
app.secret_key = 'some_secret'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CIFAR_10_CLASSES = ["airplane", "automobile", "bird", 
                    "cat", "deer", "dog", "frog", 
                    "horse", "ship", "truck"]


def allowed_file(filename):
  return '.' in filename and \
         filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cifar10_model_fn(features, labels, mode):
  """Model function for CIFAR-10."""
  network = resnet_model.cifar10_resnet_v2_generator( FLAGS.resnet_size, NUM_CLASSES)

  inputs = tf.reshape(features, [-1, HEIGHT, WIDTH, DEPTH])
  logits = network(inputs, mode == tf.estimator.ModeKeys.TRAIN)

  predictions = {
      'classes': tf.argmax(logits, axis=1),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

def get_input_fn(img):
  def input_fn():
    data = tf.cast(np.asarray([np.asarray(img)]), tf.float32)
    dataset = tf.contrib.data.Dataset()
    dataset = dataset.from_tensors(data)
    iterator = dataset.make_one_shot_iterator()
    image = iterator.get_next()
    return image
  return input_fn

@app.route('/')
def index():
  message = "Put your image"

  return render_template('index.html', message=message)

@app.route('/post', methods=['POST'])
def post():
  if request.method == 'POST':
    if 'file' not in request.files:
      return jsonify({"message": 'No file part'})
    file = request.files['file']
    if file.filename == '':
      return jsonify({"message": 'No selected file'})
    if file and allowed_file(file.filename):
      print(file.filename)
      filename = secure_filename(file.filename)
      current_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
      file.save(current_path)
      img = Image.open(current_path)
      img = img.resize((32,32)).convert('RGB')
      img.save(current_path)
      input_fn = get_input_fn(img)
      cifar_classifier = tf.estimator.Estimator(
        model_fn=cifar10_model_fn, model_dir=FLAGS.model_dir)
      ans = cifar_classifier.predict(input_fn)
      ans_index = list(ans)[0]
      print(ans_index)
      return jsonify({
        "message": "ok",
        "filename": file.filename,
        "answer": CIFAR_10_CLASSES[ans_index['classes']],
        "probabilities": [float(i) for i in ans_index['probabilities']]})
  return jsonify({"message": "ng"})   

if __name__ == '__main__':
  app.debug = True # デバッグモード有効化
  app.run()
