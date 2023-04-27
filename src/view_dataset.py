import tensorflow as tf
from params import *
import matplotlib.pyplot as plt

dataset = tf.data.Dataset.load(training_path)

for elem in dataset.take(100).unbatch():
    ((grey, hint_mask, hint_color), color) = elem
    
#    height, width, _ = grey.shape
#    assert width == 224 and height == 224
#
#    height, width, _ = hint_mask.shape
#    assert width == 224 and height == 224
#
#    height, width, _ = color.shape
#    assert width == 224 and height == 224

    input_image = tf.concat((grey, color), axis=-1)
    input_image = tf.image.yuv_to_rgb(input_image)

#    just_hints = tf.concat((tf.fill(grey.shape, 0.3), hint_color), axis=-1)
    just_hints = tf.concat((grey, hint_color), axis=-1)
    just_hints = tf.image.yuv_to_rgb(just_hints)

    print(tf.reduce_mean(hint_color))

    fig, (ax1, ax2, ax3) = plt.subplots(3)

    import numpy as np
    
    print(np.min(hint_color.numpy()))
    print(np.max(hint_color.numpy()))
    print(np.min(color.numpy()))
    print(np.max(color.numpy()))
    ax1.imshow(input_image)
    ax2.imshow(tf.cast(hint_mask, tf.dtypes.float32), vmin=0.0, vmax=1.0)
    ax3.imshow(just_hints)
    plt.show()

    print(elem)
