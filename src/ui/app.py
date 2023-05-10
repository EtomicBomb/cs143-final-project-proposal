import tensorflow as tf

from flask import Flask, request, Response, jsonify, send_file
from PIL import Image, ImageColor
from flask_cors import CORS, cross_origin
import io
import numpy as np
import sys
import json

sys.path.append('../')
sys.path.append('./inception')
from dummy import do_nothing
from inception import get_suggested_colors, get_colorized_inception

model = tf.keras.models.load_model('../check/new.h5')

app = Flask(__name__)
CORS(app)

@app.route('/input_image', methods=['POST', 'GET'])
@cross_origin()
def colorize_image():
    data = request.files
    try:
        coords = request.form.get("coords")

        # this is a list of dictionaries
        coords = json.loads(coords)
        print('received ', len(coords), 'hints')

        points = []
        colors = []
        for coord in coords:
            x = coord['x']
            y = coord['y']
            col = coord['color']
            col = ImageColor.getcolor(col, "RGB")
            points.append((y, x))
            colors.append(col)
    except:
       points=[]
       colors=[]

    file = data['file']
    image = Image.open(file.stream)

    image = do_nothing(image, points, colors, model)
    print('finished prediction')
    image = Image.fromarray(image)

    output = io.BytesIO()
    image.save(output, format='PNG')
    output.seek(0)
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


@app.route('/inception_image', methods=['POST', 'GET'])
@cross_origin()
def colorize_image_inception():
    data = request.files
 
    file = data['file']
    pil_img = Image.open(file)
    img_array = np.array(pil_img)
    image=get_colorized_inception(img_array)
    
    image = np.array(image, dtype=np.uint8)
    image = Image.fromarray(image)

    output = io.BytesIO()
    image.save(output, format='PNG')
    output.seek(0)
    return send_file(output, mimetype='image/*')



if __name__ == "__main__":
 app.run(host='127.0.0.1', port=5000, debug=True)
