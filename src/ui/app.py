# import tensorflow as tf

from flask import Flask, request, Response, jsonify, render_template, send_file
from PIL import Image
from flask.templating import render_template
from flask_cors import CORS
from werkzeug.datastructures import MultiDict, FileMultiDict
from dummy import do_nothing
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
    # print(data)
    print(request.files['file'])
    print(request.form["x"])
    x = request.form["x"]
    y = request.form["y"]
     # color is in hex format
    #TODO: change to rgb if backend expects rgb

    color = request.form["color"]
 
    file = data['file']
    img = Image.open(file.stream)
    output = io.BytesIO()
    img.save(output, format='PNG')
    output.seek(0)
    output=do_nothing(output)
    return send_file(output, mimetype='image/*')



if __name__ == "__main__":
 app.run(host='127.0.0.1', port=5000, debug=True)