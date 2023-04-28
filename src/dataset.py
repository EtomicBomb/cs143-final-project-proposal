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
        points_rows = tf.random.uniform((num_points,1), minval=0, maxval=height, dtype=tf.int32)
        points_cols = tf.random.uniform((num_points,1), minval=0, maxval=width, dtype=tf.int32)
        points = tf.concat((points_rows, points_cols), axis=-1)

        colors = sample(color, points, hint_sample_variance)
        hint_mask, hint_color = create_hints(height, width, points, colors, hint_radius)

    return ((grey, hint_mask, hint_color), color)

@tf.function
def create_dataset(data):
    data = data.repeat(file_include_times)
    data = data.map(path_to_training_example)
    data = data.shuffle(shuffle_buffer_size)
    return data

data = tf.data.Dataset.list_files(dataset_path)
data = data.shuffle(shuffle_buffer_size)
validation_count = int(validation_frac * data.cardinality().numpy())
create_dataset(data.skip(validation_count)).save(training_path)
create_dataset(data.take(validation_count)).save(validation_path)



