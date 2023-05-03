from params import *
import tensorflow as tf
from hints import *

def colorize(grey, points, colors, model):
    '''
    grey (height, width, 1) y from yuv [0, 1]
    points (n, 2) int32
    colors (n, 2) uv from yuv [0, 1]
    model () from tensorflow

    returns - (height, width, 3) yuv [0,1]
    '''

    shape = tf.shape(grey)
    height, width = shape[0], shape[1]

    hint_mask, hint_color = create_hints(height, width, points, colors, hint_threshold, hint_sample_variance)
    
    predicted, = model.predict((
        tf.expand_dims(grey, axis=0), 
        tf.expand_dims(hint_mask, axis=0), 
        tf.expand_dims(hint_color, axis=0),
    ))
    predicted = tf.concat((grey, predicted), axis=-1)

    return predicted

if __name__ == '__main__':
    from skimage.io import imread
    from skimage.io import imshow
    from skimage.transform import resize
    from skimage import img_as_float32
    import numpy as np
    import matplotlib.pyplot as plt

    # input can come from numpy, memory, anywhere
    image = imread('flower.jpg')
    image = img_as_float32(image)
    image = resize(image, (image_height, image_width, 3), anti_aliasing=True)

    points = [
#        [20,20],
#        [50,50],
#        [223,210],
#        [223,223],
    ]

    colors = [
#        [255, 0, 0],
#        [0, 255, 0],
#        [0, 0, 255],
#        [255, 255, 0],
    ]

    model = tf.keras.models.load_model('check/d.h5')

    # convert before giving to colorize

    image = tf.image.rgb_to_yuv(image)

#    # feed whole color and predict (for testing)
#    predicted, = model.predict((
#        tf.expand_dims(image[:,:,:1], axis=0), 
#        tf.expand_dims(tf.fill(image[:,:,:1].shape, True), axis=0), 
#        tf.expand_dims(image[:,:,1:], axis=0),
#    ))
#    predicted = tf.concat((image[:,:,:1], predicted), axis=-1)
#    predicted = tf.image.yuv_to_rgb(predicted)

    # predict using generated points
    image = image[:,:,:1]
    points = tf.cast(points, tf.dtypes.float32)
    colors = tf.cast(colors, tf.dtypes.float32)
    colors = tf.image.rgb_to_yuv(colors / 255.0)
    colors = colors[:,1:]
    predicted = colorize(image, points, colors, model)
    predicted = tf.image.yuv_to_rgb(predicted)

    imgplot = plt.imshow(predicted.numpy())
    plt.show()
