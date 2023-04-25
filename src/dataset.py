import tensorflow as tf
import tensorflow_probability as tfp
from params import *
from util import *
import glob

rng = tf.random.Generator.from_seed(0)

@tf.function
def sample_geometric(rng, prob):
    geometric =  tfp.distributions.Geometric(probs=prob)
    num_points_seed = rng.uniform_full_int((2,), dtype=tf.dtypes.int32)
    num_points = geometric.sample(sample_shape=(), seed=num_points_seed)
    num_points = tf.cast(num_points, tf.dtypes.int64)
    return num_points

@tf.function
def path_to_training_example(image):
    image = tf.io.read_file(image)
    image = tf.io.decode_image(image, expand_animations=False, channels=3, dtype=tf.dtypes.float32)
    image = tf.image.resize(image, (image_height, image_width))
    image = tf.image.rgb_to_yuv(image)
    grey = image[:,:,:1]
    color = image[:,:,1:]

    shape = tf.shape(image)
    height, width = shape[0], shape[1]

    num_points = sample_geometric(rng, hint_points_prob)
    points_rows = rng.uniform((num_points, 1), minval=0, maxval=height, dtype=tf.dtypes.int32)
    points_cols = rng.uniform((num_points, 1), minval=0, maxval=width, dtype=tf.dtypes.int32)
    points = tf.concat((points_rows, points_cols), axis=-1)

    colors = sample(color, points, hint_sample_variance)
    hint_mask, hint_color = create_hints(height, width, points, colors)
    return ((grey, hint_mask, hint_color), color)

shuffle_seed = rng.uniform_full_int((), dtype=tf.dtypes.int64)

data = tf.data.Dataset.list_files(dataset_path)
data = data.repeat(file_include_times)
data = data.map(path_to_training_example)
data = data.shuffle(shuffle_buffer_size, seed=shuffle_seed)
data = data.batch(batch_size)

validation_count = int(validation_frac * data.cardinality().numpy())
training_data = data.skip(validation_count)
training_data.save(training_path)
validation_data = data.take(validation_count)
validation_data.save(validation_path)

