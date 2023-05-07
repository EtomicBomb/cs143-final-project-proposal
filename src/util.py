from params import *
import tensorflow as tf
from hints import *

def colorize(grey, points, colors, model):
    '''
    grey (height, width, 1) y from yuv [0, 1]
    points (n, 2) float32
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
    from skimage.io import imshow, imsave
    from skimage.transform import resize
    from skimage import img_as_float32
    import numpy as np
    import matplotlib.pyplot as plt

    # input can come from numpy, memory, anywhere
    image = imread('flower.jpg')
#    image = imread('raw/40410686_272bc66faf_m.jpg')
#    image = imread('raw/130684941_d1abfa3be6_m.jpg')
#    image = imread('raw/2767658405_1e2043f44c_n.jpg')
#    image = imread('oxford/test/image_00658.jpg') # does a pretty poor job at this one
    image = imread('saves/b.jpg') 
#    image = imread('oxford/test/image_04027.jpg')
#    image = imread('oxford/test/image_08089.jpg')
#    image = imread('oxford/test/image_08033.jpg')
#    image = imread('oxford/test/image_07937.jpg')  # nails this one with no hints!
#    image = imread('oxford/test/image_08186.jpg') # you can hint the color of this flower! [70,70]
    image = img_as_float32(image)

    points = [
#        [20,20],
#        [50,50],
#        [65,80],
#        [223,210],
#        [50,112],
#        [175,175],
        [30, 220],
        [445, 53],
        [301, 123],
    ]

    colors = [
#        [255, 0, 0],
#        [0, 255, 0],
#        [0, 0, 255],
#        [255, 255, 255],
#        [255, 128, 0],
#        [179, 72, 118]
#        [255, 255, 0],
        [112, 219, 239],
        [59, 116, 84],
        [59, 116, 84],
    ]

    model = tf.keras.models.load_model('check/g.h5')

    # convert before giving to colorize

    image = tf.image.rgb_to_yuv(image)

    before_image = image
    before_height, before_width, _ = image.shape
    image = tf.image.resize(image, (image_height, image_width), antialias=True)

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
    points = tf.reshape(points, (-1, 2))
    points = tf.cast(points, tf.dtypes.float32)
    colors = tf.reshape(colors, (-1, 3))
    colors = tf.cast(colors, tf.dtypes.float32)
    colors = tf.image.rgb_to_yuv(colors / 255.0)
    colors = colors[:,1:]
    predicted = colorize(image, points, colors, model)
    predicted = tf.image.resize(predicted, (before_height, before_width))
    predicted = tf.concat((before_image[:,:,:1], predicted[:,:,1:]), axis=-1)
    predicted = tf.image.yuv_to_rgb(predicted)

    imsave('saves/save.png', predicted)

    s = make_sample(image_height, image_width, points, 5)
    s = tf.reduce_sum(s, axis=-1)



    imgplot = plt.imshow(s.numpy())
    plt.show()
    imgplot = plt.imshow(predicted.numpy())
    plt.show()
