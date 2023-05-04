import tensorflow as tf

from flask import Flask, request, Response, jsonify, render_template, send_file
from PIL import Image, ImageColor
from flask.templating import render_template
from flask_cors import CORS, cross_origin
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

model = tf.keras.models.load_model('../check/d.h5')

app = Flask(__name__)
CORS(app)



@app.route('/input_image', methods=['POST', 'GET'])
@cross_origin()
def colorize_image():
    data = request.files
    coords = request.form.get("coords")

    # this is a list of dictionaries
    coords = json.loads(coords)

    # convert to list of tuples (can keep dictionaries too)
    coord_tuples=[]

    for coord in coords:
        x = coord['x']
        y = coord['y']
        if (x,y) not in coord_tuples:
            coord_tuples.append((x,y))
    print(coord_tuples)

    # color is in hex format by default
    color = request.form["color"]
    #TODO: change to rgb if backend expects rgb
    rgb_color = ImageColor.getcolor(color, "RGB")

    file = data['file']
    image = Image.open(file.stream) 

    print('before', image)
    image = do_nothing(image, coord_tuples, rgb_color, model)
    image = Image.fromarray(image)
    print('after', image)


    output = io.BytesIO()
    image.save(output, format='PNG')
    output.seek(0)
    return send_file(output, mimetype='image/*')

#    predicted=do_nothing(image, coord_tuples, rgb_color, model)
#    predicted = Image.fromarray(predicted)
#    output = io.BytesIO()
#    output.seek(0)
#    predicted.save(output, format='PNG')
#    return send_file(output, mimetype='image/*')


@app.route('/suggested_colors', methods=['POST', 'GET'])
@cross_origin()
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
