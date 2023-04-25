from params import *
import tensorflow as tf
from hints import *

def colorize(grey, points, colors, model):
    '''
    grey (height, width, 1) y from yuv [0, 1]
    points (n, 2)
    colors (n, 2) rgb [0, 1]
    model () from tensorflow

    returns - (height, width, 3) rgb [0,1]
    '''

    shape = tf.shape(grey)
    height, width = shape[0], shape[1]

    hint_mask, hint_color = create_hints(height, width, points, colors) 
    
    predicted, = model.predict((
        tf.expand_dims(grey, axis=0), 
        tf.expand_dims(hint_mask, axis=0), 
        tf.expand_dims(hint_color, axis=0),
    ))
    predicted = tf.clip_by_value(predicted, -0.5, 0.5)
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

    positions = np.array([
        [10,20],
        [20,50],
        [120,120],
    ], dtype=np.int32)

    colors = np.array([
        [0, 23, 50],
        [500, 23, 50],
        [323, 23, 50],
    ], dtype=np.float32)

    model = tf.keras.models.load_model('check/01-0.03.h5')

    # convert before giving to colorize

    image = tf.image.rgb_to_yuv(image)
    image = image[:,:,:1]

    colors = tf.image.rgb_to_yuv(colors)
    colors = colors[:,1:]

    predicted = colorize(image, positions, colors, model)

    predicted = tf.image.yuv_to_rgb(predicted)
    predicted = predicted.numpy()

    imgplot = plt.imshow(predicted)
    plt.show()
