# import tensorflow as tf

from flask import Flask, request, Response, jsonify, render_template, send_file
from PIL import Image, ImageColor
from flask.templating import render_template
from flask_cors import CORS
from werkzeug.datastructures import MultiDict, FileMultiDict
# from dummy import do_nothing
import cv2
import base64
import io
import numpy as np
import sys
import json

# from util import colorize
sys.path.append('../')
from dummy import do_nothing
from inception import get_suggested_colors



app = Flask(__name__)
CORS(app)



@app.route('/input_image', methods=['POST', 'GET'])
def colorize_image():
    data = request.files
    # print(data)
    print(request.files['file'])
    print(request.form["x"])
    x = request.form["x"]
    y = request.form["y"]
     # color is in hex format by default
    color = request.form["color"]
    #TODO: change to rgb if backend expects rgb
    rgb_color = ImageColor.getcolor(color, "RGB")
 
    file = data['file']
    img = Image.open(file.stream)
    output = io.BytesIO()
    img.save(output, format='PNG')
    output.seek(0)
    output=do_nothing(output, x, y, rgb_color)
    return send_file(output, mimetype='image/*')


@app.route('/suggested_colors', methods=['POST', 'GET'])
def send_suggested_colors():
    data = request.files
    file = data['file']
    pil_img = Image.open(file)
    img_array = np.array(pil_img)
    colors=get_suggested_colors(img_array)
    colors_list = [color.tolist() for color in colors]
    tuple_list = [tuple(x[0][0]) for x in colors_list]
    colors_json= json.dumps({'data': tuple_list})

    return Response(colors_json, mimetype='application/json')



if __name__ == "__main__":
 app.run(host='127.0.0.1', port=5000, debug=True)