train_source = 'oxford/train/*'
test_source = 'oxford/test/*'
test_dest = 'data/test'
train_dest = 'data/train'

model_number = 2
load_weights_from = 'check/d.h5'
learning_rate = 0.000300
#learning_rate = 0.000030
#learning_rate = 0.000010
#learning_rate = 0.000003
leaky_relu_slope = 0.1
batch_size = 16
epochs_count = 20
shuffle_buffer_size = 10000
file_include_times = 2
hint_points_prob = 1.0 / 10.0
hint_sample_variance = 4
hint_threshold = 0.1
expose_everything_frac = 0.05
image_height = 224
image_width = 224
