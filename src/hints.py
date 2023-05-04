import tensorflow as tf

@tf.function
def prune_hint_mask(hint_mask):
    '''
    Makes sure there is at most one True value in every channel
    '''
    overlap_count = tf.cast(hint_mask, tf.int32)
    overlap_count = tf.math.cumsum(overlap_count, axis=-1)
    return hint_mask & (overlap_count < 2)

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
    channel = tf.reduce_sum(channel, axis=-1)
    return channel

@tf.function
def create_hints(height, width, points, colors, hint_threshold, sample_var):
    '''
    height - int
    width - int
    points - (num_points, 2) - int list of row column values to plot the points
    colors - (num_points, 2) - uv [0, 1]

    returns 
        - hint_mask (height, width, 1)
        - hint_color (height, width, 2)
    '''
#    rows, cols = points[:,0], points[:,1]
#    rows = tf.reshape(rows, (1, 1, -1))
#    cols = tf.reshape(cols, (1, 1, -1))
#
#    drow = tf.reshape(tf.range(width), (1, -1, 1))
#    dcol = tf.reshape(tf.range(height), (-1, 1, 1))
    # this is confusing me, but drow has values ranging from (0, width), and so does cols
#    hint_mask = (drow == cols) & (dcol == rows)
#    hint_mask = (drow-cols)**2 + (dcol-rows)**2 <= hint_radius**2

    hint_mask = make_sample(height, width, points, sample_var)
    hint_mask = hint_mask > hint_threshold
    hint_mask = prune_hint_mask(hint_mask)

    u, v = colors[:,0], colors[:,1]
    hint_u = create_hints_flat(hint_mask, u)
    hint_v = create_hints_flat(hint_mask, v)
    hint_color = tf.stack((hint_u, hint_v), axis=-1)
    hint_mask = tf.reduce_any(hint_mask, axis=-1, keepdims=True)
    return hint_mask, hint_color

@tf.function
def make_sample(height, width, points, sample_var):
    rows, cols = points[:,0], points[:,1]
    rows = tf.reshape(rows, (1, 1, -1))
    cols = tf.reshape(cols, (1, 1, -1))

    drow = tf.reshape(tf.range(width, dtype=tf.dtypes.float32), (1, -1, 1))
    grow = tf.exp((-0.5/sample_var) * tf.square(drow-cols))

    dcol = tf.reshape(tf.range(height, dtype=tf.dtypes.float32), (-1, 1, 1))
    gcol = tf.exp((-0.5/sample_var) * tf.square(dcol-rows))

    return grow * gcol

@tf.function
def sample(image, points, sample_var): 
    '''
    image - (height, width, 2) float32
    points - (num_points, 2) float32 - MUST BE WITHIN BOUNDS!
    sample_var - float32 - variance of sample points

    returns (num_points, 2)
    '''
    shape = tf.shape(image)
    height, width = shape[0], shape[1]
    u, v = image[:,:,:1], image[:,:,1:]
    sample = make_sample(height, width, points, sample_var)
    u_sampled = tf.reduce_sum(u * sample, axis=(0,1)) / tf.reduce_sum(sample, axis=(0,1))
    v_sampled = tf.reduce_sum(v * sample, axis=(0,1)) / tf.reduce_sum(sample, axis=(0,1))
    return tf.stack((u_sampled, v_sampled), axis=-1)

if __name__ == '__main__':
    # show sample map
    import matplotlib.pyplot as plt
    import numpy as np
    height, width = 224, 224
    sample_var = 4
    num_points = 50

    points = tf.cast([
#        [20,20],
#        [50,50],
        [65,80],
#        [223,210],
#        [50,112],
#        [175,175],
    ], dtype=tf.float32)


#    points = tf.random.normal((num_points, 2), mean=(height/2,width/2), stddev=(height/4,width/4))
#    points = tf.clip_by_value(points, 0, (height, width))
    s = make_sample(height, width, points, sample_var)

    s = tf.reduce_sum(s, axis=-1)


    plt.imshow(s)
    plt.show()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    height = 400 
    width = 300 
    num_points = 200
    hint_sample_variance = 4
    hint_threshold = 0.1

    image = 'flower.jpg'
    image = tf.io.read_file(image)
    image = tf.io.decode_image(image, expand_animations=False, channels=3, dtype=tf.dtypes.float32)
    image =  tf.image.resize(image, (height, width))
    image = tf.image.rgb_to_yuv(image)
    grey = image[:,:,:1]
    color = image[:,:,1:]

#    rng = tf.random.Generator.from_seed(0)
#    points_rows = rng.uniform((num_points, 1), minval=0, maxval=image_height, dtype=tf.dtypes.int32)
#    points_cols = rng.uniform((num_points, 1), minval=0, maxval=image_width, dtype=tf.dtypes.int32)
#    points = tf.concat((points_rows, points_cols), axis=-1)

    points = tf.random.normal((num_points, 2), mean=(height/2,width/2), stddev=(height/4,width/4))
    points = tf.clip_by_value(points, 0, (height, width))

    colors = sample(color, points, hint_sample_variance)

    hint_mask, color_hint = create_hints(height, width, points, colors, hint_threshold, hint_sample_variance)
    reconstruction = tf.concat((grey, color_hint), axis=-1)
    reconstruction = tf.image.yuv_to_rgb(reconstruction)

    plt.imshow(tf.cast(hint_mask, tf.dtypes.float32))
    plt.show()
    plt.imshow(reconstruction)
    plt.show()
