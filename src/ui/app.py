import tensorflow as tf

from flask import Flask, request, Response, jsonify, render_template, send_file
from PIL import Image, ImageColor
from flask.templating import render_template
from flask_cors import CORS, cross_origin
from werkzeug.datastructures import MultiDict, FileMultiDict
import io
import numpy as np
import sys
import json

# from util import colorize
sys.path.append('../')
from dummy import do_nothing, true_do_nothing
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
    hint_tuples=[]

    for coord in coords:
        x = coord['x']
        y = coord['y']
        col = coord['color']
        # convert col to rgb
        col = ImageColor.getcolor(col, "RGB")
        if (x,y,col) not in hint_tuples:
            hint_tuples.append((x,y,col))
    print(hint_tuples)

 
    file = data['file']
    image = Image.open(file.stream) 

    # @Ethan, uncomment this once your do_nothing works with the new array
    # image = do_nothing(image, coord_tuples, rgb_color, model)
    # image = Image.fromarray(image)


    output = io.BytesIO()
    image.save(output, format='PNG')
    output.seek(0)
    output=true_do_nothing(output, hint_tuples)
    return send_file(output, mimetype='image/*')

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
