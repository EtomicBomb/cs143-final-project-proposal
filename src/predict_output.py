import tensorflow as tf
from util import *
from params import *

def predict_output(image, points, colors, model):

    image = tf.cast(image, tf.float32)
    image = image / 255.0
    before_height, before_width, _ = image.shape
    image = tf.image.resize(image, (image_height, image_width))
    image = tf.image.rgb_to_yuv(image)
    image = image[:,:,:1]

    points = tf.cast(points, tf.float32)
    scale_points = [[image_height/before_height, image_width/before_width]]
    points = points * scale_points

    colors = tf.cast(colors, tf.dtypes.float32)
    colors = tf.image.rgb_to_yuv(colors / 255.0)
    colors = colors[:,1:]
    predicted = colorize(image, points, colors, model)
    predicted = tf.image.yuv_to_rgb(predicted)
    predicted = tf.image.resize(predicted, (before_height, before_width))
    predicted = predicted * 255.0
    predicted = tf.clip_by_value(predicted, 0, 255)
    predicted = tf.cast(predicted, tf.uint8)
    predicted = predicted.numpy()
    return predicted
