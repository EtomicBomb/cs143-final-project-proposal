from params import *
import tensorflow as tf

@tf.function
def plot_point(accum, elem):
    hint_mask, hint_colors = accum
    position, color = elem

    shape = tf.shape(hint_mask)
    height, width = shape[0], shape[1]

    row, col = position[0], position[1]

    drow = tf.reshape(tf.range(height, dtype=tf.dtypes.float32) - row, (-1, 1, 1))
    dcol = tf.reshape(tf.range(width, dtype=tf.dtypes.float32) - col, (1, -1, 1))
    distances = drow**2 + dcol**2
    mask = distances <= hint_radius**2

    hint_mask |= mask
    a = tf.where(mask, color, 0)
    hint_colors += tf.where(mask, color, 0) # plus not right
    return hint_mask, hint_colors

@tf.function
def plot_points(height, width, positions, colors): 
    '''
    positions (n, 2) - list of row column values to plot the points
    colors (n, 2) - uv [0, 1]

    returns - mask (height, width, 1), colors (height, width, 2)
    '''
    hint_mask = tf.fill((height, width, 1), False)
    hint_color = tf.zeros((height, width, 2))
    hint_mask, hint_color = tf.foldl(plot_point, (positions, colors), initializer=(hint_mask, hint_color))
    return hint_mask, hint_color    

def colorize(grey, positions, colors, model):
    '''
    grey (height, width, 1) y(uv) [0, 1]
    positions (n, 2)
    colors (n, 3) rgb [0, 1]
    model ()

    returns - (height, width, 3) rgb [0,1]
    '''

    shape = tf.shape(grey)
    height, width = shape[0], shape[1]

    hint_mask, hint_color = plot_points(height, width, positions, colors) 
    
    predicted, = model.predict((
        tf.expand_dims(grey, axis=0), 
        tf.expand_dims(hint_mask, axis=0), 
        tf.expand_dims(hint_color, axis=0),
    ))
    predicted = tf.clip_by_value(predicted, -0.5, 0.5)
    print(np.min(grey.numpy()), np.max(grey.numpy()))
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
    image = imread('/home/ethan/Pictures/lisa.jpeg')
    image = img_as_float32(image)
    image = resize(image, (image_height, image_width, 3), anti_aliasing=True)

    positions = np.array([
        [10,20],
        [20,50],
        [120,120],
    ], dtype=np.float32)

    colors = np.array([
        [0, 23, 50],
        [500, 23, 50],
        [323, 23, 50],
    ], dtype=np.float32)

    model = tf.keras.models.load_model('check/01-0.01.h5')

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

    imshow(predicted)
