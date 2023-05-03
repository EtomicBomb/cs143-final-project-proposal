import tensorflow as tf
import tensorflow_probability as tfp
from params import *
from util import *
import glob

@tf.function
def path_to_training_example(image):
    image = tf.io.read_file(image)
    image = tf.io.decode_image(image, expand_animations=False, channels=3, dtype=tf.float32)
    image = tf.image.resize(image, (image_height, image_width))
    image = tf.image.rgb_to_yuv(image)
    grey = image[:,:,:1]
    color = image[:,:,1:]

    shape = tf.shape(image)
    height, width = shape[0], shape[1]
    
    should_expose = tf.random.uniform(shape=(), minval=0, maxval=1.0, dtype=tf.float32)
    if should_expose < expose_everything_frac:
        hint_mask = tf.fill(grey.shape, True)
        hint_color = color
    else:
        num_points = tfp.distributions.Geometric(probs=hint_points_prob).sample()
        num_points = tf.cast(num_points, tf.int64)
        points = tf.random.normal((num_points, 2), mean=(height/2,width/2), stddev=(height/4,width/4))
        points = tf.clip_by_value(points, 0, (height, width))
#        points_rows = tf.random.uniform((num_points,1), minval=0, maxval=height, dtype=tf.int32)
#        points_cols = tf.random.uniform((num_points,1), minval=0, maxval=width, dtype=tf.int32)
#        points = tf.concat((points_rows, points_cols), axis=-1)
#        points = tf.cast(points, dtype=tf.int32)

        colors = sample(color, points, hint_sample_variance)
        hint_mask, hint_color = create_hints(height, width, points, colors, hint_threshold, hint_sample_variance)

    return ((grey, hint_mask, hint_color), color)

def create_dataset(source):
    data = tf.data.Dataset.list_files(source)
    data = data.repeat(file_include_times)
    data_count = data.cardinality().numpy()
    print(data_count)
    data = data.shuffle(data_count)
    data = data.map(path_to_training_example)
    return data

if __name__ == '__main__':
    print('saving dataset to disk')
    create_dataset(train_source).save(train_dest)
    create_dataset(test_source).save(test_dest)
