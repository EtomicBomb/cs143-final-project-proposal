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
    channel = tf.reduce_max(channel, axis=-1)
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

    drow = tf.reshape(tf.range(width, dtype=tf.dtypes.int32), (1, -1, 1))
    dcol = tf.reshape(tf.range(height, dtype=tf.dtypes.int32), (-1, 1, 1))
    hint_mask = (drow == rows) & (dcol == cols)

    u, v = colors[:,0], colors[:,1]
    hint_u = create_hints_flat(hint_mask, u)
    hint_v = create_hints_flat(hint_mask, v)
    hint_color = tf.stack((hint_u, hint_v), axis=-1)
    hint_mask = tf.reduce_any(hint_mask, axis=-1)
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
    grow = tf.exp(-(0.5/sample_var) * tf.square(drow-rows))

    dcol = tf.reshape(tf.range(height, dtype=tf.dtypes.float32), (-1, 1, 1))
    gcol = tf.exp(-(0.5/sample_var) * tf.square(dcol-cols))

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
    import matplotlib.pyplot as plt

    points = tf.constant([
        [1, 2],
        [30, 40],
        [50, 60],
        [0, 4],
    ], dtype=tf.dtypes.int32)

    colors = tf.constant([
        [0.3, 0.9],
        [0.5, 0.5],
        [.2, 0.2],
        [1., 1.],
    ], dtype=tf.dtypes.float32)

    sample_var = 4
    width = 100
    height = 100

    hint_mask, color_hint = create_hints(height, width, points, colors)
    color_hint = tf.concat((color_hint, tf.fill((height, width, 1), 0.5)), axis=-1)
    plt.imshow(color_hint)
    plt.show()
    plt.imshow(tf.cast(hint_mask, tf.dtypes.float32))
    plt.show()

    u = tf.fill((height, width, 1), 0.5)
    v = tf.fill((height, width, 1), 0.0)
    image = tf.concat((u, v), axis=-1)
    print(sample(image, points, sample_var))
