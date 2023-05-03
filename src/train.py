import os
import sys
import argparse
import re
import os
import random
from datetime import datetime
import tensorflow as tf
import numpy as np
import tensorflow as tf

from params import *
from util import *

def create_model_dummy():
    grey_in = tf.keras.Input(shape=(None, None, 1))
    hint_mask_in = tf.keras.Input(shape=(None, None, 1))
    hint_color_in = tf.keras.Input(shape=(None, None, 2))

    x = tf.concat((grey_in, hint_mask_in, hint_color_in), axis=-1)

    x = tf.keras.layers.Conv2D(2, kernel_size=3, strides=1, padding='same')(x)

    x = tf.math.sigmoid(x) - 0.5 # output in [-0.5, 0.5]
    x = tf.keras.Model(inputs=(grey_in, hint_mask_in, hint_color_in), outputs=x)
    return x

def cool_blue(x, num_kernels):
    x = tf.keras.layers.Conv2D(num_kernels, kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.GroupNormalization(groups=32)(x)
    x = tf.keras.layers.LeakyReLU(alpha=leaky_relu_slope)(x)
    x = tf.keras.layers.Conv2D(num_kernels, kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.GroupNormalization(groups=32)(x)
    x = tf.keras.layers.LeakyReLU(alpha=leaky_relu_slope)(x)
    return x

def cool_orange(x, num_kernels):
    x = tf.keras.layers.Conv2D(num_kernels, kernel_size=4, strides=2, padding='same')(x)
    x = tf.keras.layers.GroupNormalization(groups=32)(x)
    return x

def cool_green(x, num_kernels):
    x = tf.keras.layers.Conv2DTranspose(num_kernels, kernel_size=4, strides=2, padding='same')(x)
    x = tf.keras.layers.GroupNormalization(groups=32)(x)
    x = tf.keras.layers.LeakyReLU(alpha=leaky_relu_slope)(x)
    return x 

def create_model_cool():
    # https://www.mdpi.com/2673-2688/1/4/29/htm
    # https://www.mdpi.com/ai/ai-01-00029/article_deploy/html/images/ai-01-00029-g002.png

    grey_in = tf.keras.Input(shape=(None, None, 1))
    hint_mask_in = tf.keras.Input(shape=(None, None, 1))
    hint_color_in = tf.keras.Input(shape=(None, None, 2))
    valid = tf.ones_like(grey_in)
    x = tf.concat((grey_in, hint_mask_in, hint_color_in, valid), axis=-1)

    x = cool_blue(x, 64)
    x1 = x
    x = cool_orange(x, 64)
    x = cool_blue(x, 128)
    x2 = x
    x = cool_orange(x, 128)
    x = cool_blue(x, 256)
    x3 = x
    x = cool_orange(x, 256)
    x = cool_blue(x, 512)
    x4 = x
    x = cool_orange(x, 512)
    x = cool_blue(x, 512)
    x = cool_green(x, 512)
#    x = tf.concat((x, x4), axis=-1)
    x = cool_blue(x, 512)
    x = cool_green(x, 256)
    x = tf.concat((x, x3), axis=-1)
    x = cool_blue(x, 256)
    x = cool_green(x, 128)
#    x = tf.concat((x, x2), axis=-1)
    x = cool_blue(x, 128)
    x = cool_green(x, 64)
#    x = tf.concat((x, x1), axis=-1)
    x = cool_blue(x, 64)
    # navy
    x = tf.keras.layers.Conv2D(2, kernel_size=1, strides=1, padding='same')(x)
    x = tf.math.sigmoid(x) - 0.5 # output in [-0.5, 0.5]

    x = tf.keras.Model(inputs=(grey_in, hint_mask_in, hint_color_in), outputs=x)
    return x

if model_number == 0:
    model = tf.keras.models.load_model(load_weights_from)
elif model_number == 1:
    model = create_model_dummy()
elif model_number == 2:
    model = create_model_cool()
else:
    raise ValueError('bad model number ' + str(model_number))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    run_eagerly=True,
    loss=tf.keras.losses.Huber())

model.summary()

training_data = tf.data.Dataset.load(train_dest).batch(batch_size)
validation_data = tf.data.Dataset.load(test_dest).batch(batch_size)

model.fit(
    x=training_data,
    validation_data=validation_data,
    epochs=epochs_count,
    batch_size=None,           
    callbacks=[
        tf.keras.callbacks.TensorBoard(
            log_dir='log/',
            update_freq='batch',
            profile_batch=0),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='check/{epoch:02d}-{val_loss:.6f}.h5',
            monitor='val_loss',
            verbose=1,
            save_best_only=True),
    ],
)
