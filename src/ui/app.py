# import tensorflow as tf

from flask import Flask, request, Response, jsonify, render_template, send_file
from PIL import Image
from flask.templating import render_template
from flask_cors import CORS
from werkzeug.datastructures import MultiDict
from dummy import fake_colorize
import cv2
import base64


import io

import numpy as np
# from util import colorize


app = Flask(__name__)
CORS(app)




@app.route('/input_image', methods=['POST', 'GET'])
def colorize_image():
    data = request.files
    print(data)
    print(request.files['file'])
 
    file = data['file']
    img = Image.open(file.stream)
    output = io.BytesIO()
    img.save(output, format='PNG')
    output.seek(0)
    output=fake_colorize(output)
    return send_file(output, mimetype='image/png')



if __name__ == "__main__":
 app.run(host='127.0.0.1', port=5000, debug=True)