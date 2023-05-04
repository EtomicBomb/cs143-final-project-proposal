import tensorflow as tf
from util import *
import numpy as np

def do_nothing(image, points, color, model):
#    return tf.constant(image).numpy()
    ##### REMOVE SECTION WHEN COLORS IS ARRAY #####
    num_points = len(points)
    colors = tf.reshape(color, (-1, 3))
    colors = tf.repeat(colors, num_points, axis=0)
    ##### END SECTION #####

    points = tf.cast(points, tf.int32)
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    image = tf.image.resize(image, (224, 224))
    image = tf.image.rgb_to_yuv(image)

    image = image[:,:,:1]
    colors = tf.cast(colors, tf.dtypes.float32)
    colors = tf.image.rgb_to_yuv(colors / 255.0)
    colors = colors[:,1:]
    predicted = colorize(image, points, colors, model)
    predicted = tf.image.yuv_to_rgb(predicted)
    predicted = predicted * 255.0
    predicted = tf.clip_by_value(predicted, 0, 255)
    predicted = tf.cast(predicted, tf.uint8)
    predicted = predicted.numpy()
    print('finished prediction')

    return predicted
