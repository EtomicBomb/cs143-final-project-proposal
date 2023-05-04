import tensorflow as tf
from util import *
from params import *
import numpy as np

def do_nothing(image, points, colors, model):
#    return tf.constant(image).numpy()

    points = tf.cast(points, tf.int32)
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    before_height, before_width, _ = image.shape
    image = tf.image.resize(image, (image_height, image_width))
    image = tf.image.rgb_to_yuv(image)

    image = image[:,:,:1]
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


def true_do_nothing(img, hints):
    for hint in hints:
        print("x=", hint[0], "y=", hint[1], "color=", hint[2])
    return img