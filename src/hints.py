import tensorflow as tf

@tf.function
def create_hints_flat(hint_mask, channel):
    '''
    hint_mask (height, width, num_points)
    channel - (num_points,) - single channel [0, 1]

    returns 
        - hint_color (height, width, 2)
    '''
    channel = tf.reshape(channel, (1, 1, -1))
    channel = tf.where(hint_mask, channel, 0)
    channel = tf.reduce_sum(channel, axis=-1) # shouldn't be reduce sum
    return channel

@tf.function
def create_hints(height, width, points, colors):
    '''
    height - int
    width - int
    points - (num_points, 2) - int list of row column values to plot the points
    colors - (num_points, 2) - uv [0, 1]

    returns 
        - hint_mask (height, width, 1)
        - hint_color (height, width, 2)
    '''
    rows, cols = points[:,0], points[:,1]
    rows = tf.reshape(rows, (1, 1, -1))
    cols = tf.reshape(cols, (1, 1, -1))

    drow = tf.reshape(tf.range(width), (1, -1, 1))
    dcol = tf.reshape(tf.range(height), (-1, 1, 1))
    # this is confusing me, but drow has values ranging from (0, width), and so does cols
    hint_mask = (drow == cols) & (dcol == rows)

    u, v = colors[:,0], colors[:,1]
    hint_u = create_hints_flat(hint_mask, u)
    hint_v = create_hints_flat(hint_mask, v)
    hint_color = tf.stack((hint_u, hint_v), axis=-1)
    hint_mask = tf.reduce_any(hint_mask, axis=-1, keepdims=True)
    return hint_mask, hint_color

@tf.function
def sample_flat(image, points, sample_var): 
    '''
    image - (height, width, 1) float32
    points - (num_points, 2) int32 - MUST BE WITHIN BOUNDS!
    sample_var - float32 - variance of sample points

    returns (num_points,)
    '''
    shape = tf.shape(image)
    height, width = shape[0], shape[1]

    points = tf.cast(points, tf.dtypes.float32)
    rows, cols = points[:,0], points[:,1]
    rows = tf.reshape(rows, (1, 1, -1))
    cols = tf.reshape(cols, (1, 1, -1))

    drow = tf.reshape(tf.range(width, dtype=tf.dtypes.float32), (1, -1, 1))
    grow = tf.exp((-0.5/sample_var) * tf.square(drow-cols))

    dcol = tf.reshape(tf.range(height, dtype=tf.dtypes.float32), (-1, 1, 1))
    gcol = tf.exp((-0.5/sample_var) * tf.square(dcol-rows))

    sample = grow * gcol

    sampled = tf.reduce_sum(image * sample, axis=(0,1)) / tf.reduce_sum(sample, axis=(0,1))
    return sampled




@tf.function
def sample(image, points, sample_var): 
    '''
    image - (height, width, 2) float32
    points - (num_points, 2) int32 - MUST BE WITHIN BOUNDS!
    sample_var - float32 - variance of sample points

    returns (num_points, 2)
    '''
    u, v = image[:,:,:1], image[:,:,1:]
    u_sampled = sample_flat(u, points, sample_var)
    v_sampled = sample_flat(v, points, sample_var)
    return tf.stack((u_sampled, v_sampled), axis=-1)

if __name__ == '__main__':
    # show sample map
    import matplotlib.pyplot as plt
    import numpy as np
    height, width = 224, 224
    sample_var = 4

    row = tf.reshape(30.0, (1, 1, -1))
    col = tf.reshape(80.0, (1, 1, -1))

    drow = tf.reshape(tf.range(width, dtype=tf.dtypes.float32), (1, -1, 1))
    grow = tf.exp((-0.5/sample_var) * tf.square(drow-col))

    dcol = tf.reshape(tf.range(height, dtype=tf.dtypes.float32), (-1, 1, 1))
    gcol = tf.exp((-0.5/sample_var) * tf.square(dcol-row))

    sample = grow * gcol

    plt.imshow(sample)
    plt.show()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    image_height = 400 
    image_width = 300 
    num_points = 200
    hint_sample_variance = 9

    image = 'flower.jpg'
    image = tf.io.read_file(image)
    image = tf.io.decode_image(image, expand_animations=False, channels=3, dtype=tf.dtypes.float32)
    image =  tf.image.resize(image, (image_height, image_width))
    image = tf.image.rgb_to_yuv(image)
    grey = image[:,:,:1]
    color = image[:,:,1:]

    rng = tf.random.Generator.from_seed(0)
    points_rows = rng.uniform((num_points, 1), minval=0, maxval=image_height, dtype=tf.dtypes.int32)
    points_cols = rng.uniform((num_points, 1), minval=0, maxval=image_width, dtype=tf.dtypes.int32)
    points = tf.concat((points_rows, points_cols), axis=-1)

    colors = sample(color, points, hint_sample_variance)

    hint_mask, color_hint = create_hints(image_height, image_width, points, colors)
    reconstruction = tf.concat((grey, color_hint), axis=-1)
    reconstruction = tf.image.yuv_to_rgb(reconstruction)

    plt.imshow(tf.cast(hint_mask, tf.dtypes.float32))
    plt.show()
    plt.imshow(reconstruction)
    plt.show()
