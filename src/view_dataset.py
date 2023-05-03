import tensorflow as tf
import numpy as np
from params import *
import matplotlib.pyplot as plt

dataset = tf.data.Dataset.load(train_dest)

for elem in dataset.take(100):
    ((grey, hint_mask, hint_color), color) = elem
    
    input_image = tf.concat((grey, color), axis=-1)
    input_image = tf.image.yuv_to_rgb(input_image)

    just_hints = tf.concat((grey, hint_color), axis=-1)
    just_hints = tf.image.yuv_to_rgb(just_hints)

    fig, (ax1, ax2, ax3) = plt.subplots(3)

    print(np.min(hint_color.numpy()))
    print(np.max(hint_color.numpy()))
    print(np.min(color.numpy()))
    print(np.max(color.numpy()))
    print(elem)
    ax1.imshow(input_image)
    ax2.imshow(tf.cast(hint_mask, tf.dtypes.float32), vmin=0.0, vmax=1.0)
    ax3.imshow(just_hints)
    plt.show()

